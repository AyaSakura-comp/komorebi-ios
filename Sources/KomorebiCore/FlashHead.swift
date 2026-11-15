import Accelerate
import Foundation
import KomorebiANEDirect

/// FlashHead-style LM head (Embedl, arXiv 2603.14591) over a zh / en / ja "small dictionary":
///  1. offline: keep only tokens whose text is ASCII, contains CJK / kana / bopomofo, CJK punctuation, emoji, or is a
///     control / byte token; cluster their tied embedding rows with balanced spherical k-means;
///  2. per token: centroid scores -> top `probes` clusters -> int8 SDOT scores of their members ->
///     exact Q6_K logits for the best `rescore` candidates -> argmax.
/// Only greedy argmax is needed by Komorebi, so no full-vocabulary distribution is ever built.
public final class FlashHeadLMHead: @unchecked Sendable {
    public struct Config: Sendable {
        public var clusters: Int = 4096
        public var probes: Int = 64
        public var rescore: Int = 8
        public var iterations: Int = 10
        public var sampleSize: Int = 65536
        /// Subtract the dictionary's mean embedding before clustering (h·μ is the same for every token).
        public var center: Bool = true
        /// Unit-normalise rows (spherical k-means) instead of Euclidean k-means on the rows.
        public var spherical: Bool = false
        /// Probe score = h·c + radiusWeight * |h| * r_c (r_c = RMS radius of the cluster), an inner-product bound.
        public var radiusWeight: Float = 0
        /// Query-aware clustering: keys = dimScale ⊙ (e − μ), queries = h ⊘ dimScale (q·k = h·(e − μ)).
        /// Typically the RMS of the final hidden state per dimension.
        public var dimScale: [Float]? = nil
        public init() {}
    }

    public let dim: Int
    public let clusters: Int
    public var probes: Int
    public var rescore: Int
    public let tokenIDs: [Int32]          // small-dictionary tokens in cluster order
    public let clusterStart: [Int32]      // clusters + 1
    private let centroids: [Float]        // clusters x dim
    private let radius: [Float]           // RMS radius per cluster
    public var radiusWeight: Float
    private let queryScale: [Float]?      // 1 / dimScale
    private let rowsI8: [Int8]            // tokens x dim, cluster order
    private let rowScale: [Float]
    private let gguf: GGUFFile
    private let embInfo: GGUFTensorInfo
    private let embRowBytes: Int
    private let maxClusterSize: Int

    public var dictionarySize: Int { tokenIDs.count }

    private init(gguf: GGUFFile, dim: Int, clusters: Int, probes: Int, rescore: Int, tokenIDs: [Int32], clusterStart: [Int32],
                 centroids: [Float], radius: [Float], radiusWeight: Float, queryScale: [Float]?, rowsI8: [Int8], rowScale: [Float]) throws {
        self.gguf = gguf
        self.queryScale = queryScale
        self.radius = radius
        self.radiusWeight = radiusWeight
        self.dim = dim
        self.clusters = clusters
        self.probes = probes
        self.rescore = rescore
        self.tokenIDs = tokenIDs
        self.clusterStart = clusterStart
        self.centroids = centroids
        self.rowsI8 = rowsI8
        self.rowScale = rowScale
        self.embInfo = try gguf.tensor("token_embd.weight")
        self.embRowBytes = embInfo.type.rowByteSize(elements: dim)
        var m: Int32 = 1
        for c in 0..<clusters { m = max(m, clusterStart[c + 1] - clusterStart[c]) }
        self.maxClusterSize = Int(m)
    }

    // MARK: - Small dictionary

    /// zh-Hant / zh-Hans / en / ja tokens + control and byte tokens; other scripts and `<unusedN>` are dropped.
    public static func smallDictionary(vocab: [String], tokenTypes: [Int]) -> [Int32] {
        func cjk(_ v: UInt32) -> Bool {
            (0x3000...0x30FF).contains(v) || (0x3100...0x312F).contains(v) || (0x31A0...0x31FF).contains(v)
                || (0x3200...0x9FFF).contains(v) || (0xF900...0xFAFF).contains(v) || (0xFE30...0xFE4F).contains(v)
                || (0xFF00...0xFFEF).contains(v) || (0x20000...0x2FA1F).contains(v)
        }
        func punct(_ v: UInt32) -> Bool {
            (0x2000...0x206F).contains(v) || [0x00B7, 0x00D7, 0x00B0, 0x2581].contains(v) || (0x2190...0x21FF).contains(v)
                || (0x2460...0x24FF).contains(v) || (0x25A0...0x27BF).contains(v)
        }
        func emoji(_ v: UInt32) -> Bool { (0x1F300...0x1FAFF).contains(v) || v == 0xFE0F || v == 0x200D }
        var keep: [Int32] = []
        keep.reserveCapacity(180_000)
        for (i, tok) in vocab.enumerated() {
            let type = i < tokenTypes.count ? tokenTypes[i] : 1
            if type != 1 {
                keep.append(Int32(i)); continue
            }
            if tok.hasPrefix("<unused") { continue }
            let scalars = tok.unicodeScalars.map { $0.value }
            guard !scalars.isEmpty else { continue }
            let basic = scalars.allSatisfy { $0 < 128 || $0 == 0x2581 }
            let cjkOK = scalars.contains(where: cjk) && scalars.allSatisfy { $0 < 128 || $0 == 0x2581 || cjk($0) || punct($0) }
            let symOK = scalars.allSatisfy { $0 < 128 || $0 == 0x2581 || punct($0) || emoji($0) }
            if basic || cjkOK || symOK { keep.append(Int32(i)) }
        }
        return keep
    }

    // MARK: - Build (balanced spherical k-means)

    public static func build(gguf: GGUFFile, tokenizer: Gemma4Tokenizer, config: Config = Config(), progress: ((String) -> Void)? = nil) throws -> FlashHeadLMHead {
        let info = try gguf.tensor("token_embd.weight")
        let dim = info.dimensions[0]
        let rowBytes = info.type.rowByteSize(elements: dim)
        let base = gguf.pointer(for: info)
        let keep = smallDictionary(vocab: tokenizer.vocab, tokenTypes: tokenizer.tokenTypes)
        let V = keep.count
        let C = config.clusters
        progress?("FlashHead: dictionary \(V) tokens, \(C) clusters")

        func dequant(_ tok: Int32, _ dst: UnsafeMutablePointer<Float>) {
            let src = base.advanced(by: Int(tok) * rowBytes)
            switch info.type {
            case .q6_K: GGUFQuantization.dequantizeQ6KRow(src: src, dst: dst, elements: dim)
            case .q4_0: GGUFQuantization.dequantizeQ4_0Row(src: src, dst: dst, elements: dim)
            case .f16:
                let h = src.assumingMemoryBound(to: UInt16.self)
                for j in 0..<dim { dst[j] = Float(Float16(bitPattern: h[j])) }
            default: memcpy(dst, src, dim * 4)
            }
        }
        func normalize(_ p: UnsafeMutablePointer<Float>) {
            var ss: Float = 0
            vDSP_svesq(p, 1, &ss, vDSP_Length(dim))
            var inv = ss > 0 ? 1 / sqrtf(ss) : 0
            vDSP_vsmul(p, 1, &inv, p, 1, vDSP_Length(dim))
        }
        // Dictionary mean embedding (for centring).
        var mean = [Float](repeating: 0, count: dim)
        if config.center {
            let parts = 16
            var partial = [[Float]](repeating: [Float](repeating: 0, count: dim), count: parts)
            DispatchQueue.concurrentPerform(iterations: parts) { pi in
                let row = UnsafeMutablePointer<Float>.allocate(capacity: dim); defer { row.deallocate() }
                var acc = [Float](repeating: 0, count: dim)
                for t in Swift.stride(from: pi, to: V, by: parts) {
                    dequant(keep[t], row)
                    vDSP_vadd(acc, 1, row, 1, &acc, 1, vDSP_Length(dim))
                }
                partial[pi] = acc
            }
            for acc in partial { vDSP_vadd(mean, 1, acc, 1, &mean, 1, vDSP_Length(dim)) }
            var inv = 1 / Float(V)
            vDSP_vsmul(mean, 1, &inv, &mean, 1, vDSP_Length(dim))
        }
        let meanCopy = mean
        let dimScale = config.dimScale
        func prep(_ tok: Int32, _ dst: UnsafeMutablePointer<Float>) {
            dequant(tok, dst)
            if config.center { vDSP_vsub(meanCopy, 1, dst, 1, dst, 1, vDSP_Length(dim)) }   // dst - mean
            if let dimScale { vDSP_vmul(dst, 1, dimScale, 1, dst, 1, vDSP_Length(dim)) }
            if config.spherical { normalize(dst) }
        }
        // score(x, c) = x·c (spherical) or x·c - |c|²/2 (Euclidean nearest centroid)
        var cBias = [Float](repeating: 0, count: C)
        func updateBias(_ cent: [Float]) {
            guard !config.spherical else { return }
            cent.withUnsafeBufferPointer { cb in
                for c in 0..<C {
                    var ss: Float = 0
                    vDSP_svesq(cb.baseAddress!.advanced(by: c * dim), 1, &ss, vDSP_Length(dim))
                    cBias[c] = -0.5 * ss
                }
            }
        }

        // 1. k-means on a strided sample of unit-norm rows.
        let S = min(config.sampleSize, V)
        let stride = max(1, V / S)
        let sample = UnsafeMutablePointer<Float>.allocate(capacity: S * dim); defer { sample.deallocate() }
        DispatchQueue.concurrentPerform(iterations: S) { i in
            prep(keep[min(V - 1, i * stride)], sample.advanced(by: i * dim))
        }
        var cent = [Float](repeating: 0, count: C * dim)
        cent.withUnsafeMutableBufferPointer { cb in
            for c in 0..<C { memcpy(cb.baseAddress!.advanced(by: c * dim), sample.advanced(by: ((c * S) / C) * dim), dim * 4) }
        }
        let batch = 4096
        let scores = UnsafeMutablePointer<Float>.allocate(capacity: batch * C); defer { scores.deallocate() }
        var assign = [Int32](repeating: 0, count: S)
        updateBias(cent)
        for it in 0..<config.iterations {
            // assignment
            var b0 = 0
            while b0 < S {
                let nb = min(batch, S - b0)
                cent.withUnsafeBufferPointer { cb in
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(nb), Int32(C), Int32(dim), 1,
                                sample.advanced(by: b0 * dim), Int32(dim), cb.baseAddress!, Int32(dim), 0, scores, Int32(C))
                }
                assign.withUnsafeMutableBufferPointer { ab in
                    cBias.withUnsafeBufferPointer { bb in
                        DispatchQueue.concurrentPerform(iterations: nb) { r in
                            let row = scores.advanced(by: r * C)
                            vDSP_vadd(row, 1, bb.baseAddress!, 1, row, 1, vDSP_Length(C))
                            var mx: Float = 0
                            var idx: vDSP_Length = 0
                            vDSP_maxvi(row, 1, &mx, &idx, vDSP_Length(C))
                            ab[b0 + r] = Int32(idx)
                        }
                    }
                }
                b0 += nb
            }
            // update
            var sums = [Float](repeating: 0, count: C * dim)
            var counts = [Int](repeating: 0, count: C)
            sums.withUnsafeMutableBufferPointer { sb in
                for i in 0..<S {
                    let c = Int(assign[i])
                    counts[c] += 1
                    vDSP_vadd(sb.baseAddress!.advanced(by: c * dim), 1, sample.advanced(by: i * dim), 1, sb.baseAddress!.advanced(by: c * dim), 1, vDSP_Length(dim))
                }
            }
            var empty = 0
            cent.withUnsafeMutableBufferPointer { cb in
                sums.withUnsafeBufferPointer { sb in
                    for c in 0..<C {
                        let dst = cb.baseAddress!.advanced(by: c * dim)
                        if counts[c] == 0 {
                            empty += 1
                            memcpy(dst, sample.advanced(by: ((c * 7919) % S) * dim), dim * 4)   // reseed
                        } else {
                            memcpy(dst, sb.baseAddress!.advanced(by: c * dim), dim * 4)
                            if config.spherical {
                                normalize(dst)
                            } else {
                                var inv = 1 / Float(counts[c])
                                vDSP_vsmul(dst, 1, &inv, dst, 1, vDSP_Length(dim))
                            }
                        }
                    }
                }
            }
            updateBias(cent)
            progress?("FlashHead: k-means iteration \(it + 1)/\(config.iterations) (empty \(empty))")
        }

        // 2. Balanced assignment of every dictionary token (cap = 2x average, best of top-8 centroids).
        let cap = max(8, (V * 2 + C - 1) / C)
        let topN = 8
        var prefs = [Int32](repeating: 0, count: V * topN)
        var best = [Float](repeating: 0, count: V)
        let rowsBuf = UnsafeMutablePointer<Float>.allocate(capacity: batch * dim); defer { rowsBuf.deallocate() }
        var b0 = 0
        while b0 < V {
            let nb = min(batch, V - b0)
            DispatchQueue.concurrentPerform(iterations: nb) { r in
                prep(keep[b0 + r], rowsBuf.advanced(by: r * dim))
            }
            cent.withUnsafeBufferPointer { cb in
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(nb), Int32(C), Int32(dim), 1,
                            rowsBuf, Int32(dim), cb.baseAddress!, Int32(dim), 0, scores, Int32(C))
            }
            prefs.withUnsafeMutableBufferPointer { pb in
                best.withUnsafeMutableBufferPointer { bb in
                    DispatchQueue.concurrentPerform(iterations: nb) { r in
                        let row = scores.advanced(by: r * C)
                        var top = [(Float, Int32)](repeating: (-.infinity, 0), count: topN)
                        for c in 0..<C {
                            let v = row[c] + cBias[c]
                            if v > top[topN - 1].0 {
                                var k = topN - 1
                                while k > 0 && top[k - 1].0 < v { top[k] = top[k - 1]; k -= 1 }
                                top[k] = (v, Int32(c))
                            }
                        }
                        for k in 0..<topN { pb[(b0 + r) * topN + k] = top[k].1 }
                        bb[b0 + r] = top[0].0
                    }
                }
            }
            b0 += nb
        }
        var fill = [Int](repeating: 0, count: C)
        var cluster = [Int32](repeating: -1, count: V)
        let order = (0..<V).sorted { best[$0] > best[$1] }     // confident tokens pick first
        for t in order {
            var placed = false
            for k in 0..<topN {
                let c = Int(prefs[t * topN + k])
                if fill[c] < cap { fill[c] += 1; cluster[t] = Int32(c); placed = true; break }
            }
            if !placed {
                let c = fill.enumerated().min(by: { $0.element < $1.element })!.offset
                fill[c] += 1; cluster[t] = Int32(c)
            }
        }
        progress?("FlashHead: assigned \(V) tokens (cap \(cap), max fill \(fill.max() ?? 0))")

        // Cluster RMS radius over prepared (centred / normalised) vectors.
        var radSum = [Double](repeating: 0, count: C)
        do {
            let lock = NSLock()
            DispatchQueue.concurrentPerform(iterations: 32) { pi in
                let row = UnsafeMutablePointer<Float>.allocate(capacity: dim); defer { row.deallocate() }
                var local = [Double](repeating: 0, count: C)
                cent.withUnsafeBufferPointer { cb in
                    for t in Swift.stride(from: pi, to: V, by: 32) {
                        prep(keep[t], row)
                        let c = Int(cluster[t])
                        vDSP_vsub(cb.baseAddress!.advanced(by: c * dim), 1, row, 1, row, 1, vDSP_Length(dim))
                        var ss: Float = 0
                        vDSP_svesq(row, 1, &ss, vDSP_Length(dim))
                        local[c] += Double(ss)
                    }
                }
                lock.lock(); for c in 0..<C { radSum[c] += local[c] }; lock.unlock()
            }
        }
        let rad: [Float] = (0..<C).map { fill[$0] > 0 ? Float(sqrt(radSum[$0] / Double(fill[$0]))) : 0 }

        // 3. Pack members in cluster order with int8 rows.
        var start = [Int32](repeating: 0, count: C + 1)
        for c in 0..<C { start[c + 1] = start[c] + Int32(fill[c]) }
        var cursor = start
        var ids = [Int32](repeating: 0, count: V)
        for t in 0..<V {
            let c = Int(cluster[t])
            ids[Int(cursor[c])] = keep[t]
            cursor[c] += 1
        }
        var i8 = [Int8](repeating: 0, count: V * dim)
        var scl = [Float](repeating: 0, count: V)
        i8.withUnsafeMutableBufferPointer { ib in
            scl.withUnsafeMutableBufferPointer { sb in
                DispatchQueue.concurrentPerform(iterations: (V + 255) / 256) { blk in
                    let row = UnsafeMutablePointer<Float>.allocate(capacity: dim); defer { row.deallocate() }
                    for r in (blk * 256)..<min(V, (blk + 1) * 256) {
                        dequant(ids[r], row)
                        var s: Float = 0
                        kk_quantize_i8(row, Int32(dim), ib.baseAddress!.advanced(by: r * dim), &s)
                        sb[r] = s
                    }
                }
            }
        }
        return try FlashHeadLMHead(gguf: gguf, dim: dim, clusters: C, probes: config.probes, rescore: config.rescore,
                                   tokenIDs: ids, clusterStart: start, centroids: cent, radius: rad, radiusWeight: config.radiusWeight,
                                   queryScale: config.dimScale.map { $0.map { 1 / max($0, 1e-12) } }, rowsI8: i8, rowScale: scl)
    }

    // MARK: - Persistence

    private static let magic: UInt32 = 0x4B46_4844   // "KFHD"

    public func save(to url: URL) throws {
        var d = Data()
        func put<T>(_ v: T) { withUnsafeBytes(of: v) { d.append(contentsOf: $0) } }
        func putArray<T>(_ a: [T]) { a.withUnsafeBytes { d.append(contentsOf: $0) } }
        put(Self.magic); put(UInt32(3)); put(Int32(dim)); put(Int32(clusters)); put(Int32(tokenIDs.count)); put(radiusWeight)
        put(Int32(queryScale == nil ? 0 : 1)); if let queryScale { putArray(queryScale) }
        putArray(tokenIDs); putArray(clusterStart); putArray(centroids); putArray(radius); putArray(rowScale); putArray(rowsI8)
        try d.write(to: url)
    }

    public static func load(from url: URL, gguf: GGUFFile, probes: Int = 64, rescore: Int = 8) throws -> FlashHeadLMHead {
        let d = try Data(contentsOf: url, options: .alwaysMapped)
        var off = 0
        func get<T>(_: T.Type) -> T {
            let v = d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: off, as: T.self) }
            off += MemoryLayout<T>.size
            return v
        }
        func getArray<T>(_: T.Type, _ n: Int) -> [T] {
            let a = [T](unsafeUninitializedCapacity: n) { buf, count in
                d.withUnsafeBytes { raw in
                    _ = memcpy(buf.baseAddress!, raw.baseAddress!.advanced(by: off), n * MemoryLayout<T>.stride)
                }
                count = n
            }
            off += n * MemoryLayout<T>.stride
            return a
        }
        guard get(UInt32.self) == magic, get(UInt32.self) == 3 else { throw GemmaEngineError.inferenceFailed("bad FlashHead file") }
        let dim = Int(get(Int32.self)), C = Int(get(Int32.self)), V = Int(get(Int32.self))
        let rw = get(Float.self)
        let qs: [Float]? = get(Int32.self) == 1 ? getArray(Float.self, dim) : nil
        let ids = getArray(Int32.self, V)
        let start = getArray(Int32.self, C + 1)
        let cent = getArray(Float.self, C * dim)
        let rad = getArray(Float.self, C)
        let scl = getArray(Float.self, V)
        let i8 = getArray(Int8.self, V * dim)
        return try FlashHeadLMHead(gguf: gguf, dim: dim, clusters: C, probes: probes, rescore: rescore,
                                   tokenIDs: ids, clusterStart: start, centroids: cent, radius: rad, radiusWeight: rw, queryScale: qs, rowsI8: i8, rowScale: scl)
    }

    // MARK: - Query

    /// Argmax token over the small dictionary for one final-norm hidden row `h` (fp32, dim).
    public func argmax(_ h: UnsafePointer<Float>) -> Int32 { argmaxBatch(h, count: 1)[0] }

    /// Argmax for `count` rows (row-major, dim each): centroid scores for all rows in one GEMM, then per-row retrieval.
    public func argmaxBatch(_ hs: UnsafePointer<Float>, count: Int) -> [Int32] {
        let C = clusters, D = dim
        var cs = [Float](repeating: 0, count: count * C)
        var qbuf = [Float]()
        var qptr = hs
        if let queryScale {
            qbuf = [Float](repeating: 0, count: count * D)
            for r in 0..<count { vDSP_vmul(hs.advanced(by: r * D), 1, queryScale, 1, &qbuf[r * D], 1, vDSP_Length(D)) }
            qptr = qbuf.withUnsafeBufferPointer { $0.baseAddress! }
        }
        withExtendedLifetime(qbuf) {
            centroids.withUnsafeBufferPointer { cb in
                if count == 1 {
                    cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(C), Int32(D), 1, cb.baseAddress!, Int32(D), qptr, 1, 0, &cs, 1)
                } else {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(count), Int32(C), Int32(D), 1,
                                qptr, Int32(D), cb.baseAddress!, Int32(D), 0, &cs, Int32(C))
                }
            }
        }
        if radiusWeight != 0 {
            for r in 0..<count {
                var ss: Float = 0
                vDSP_svesq(hs.advanced(by: r * D), 1, &ss, vDSP_Length(D))
                var w = radiusWeight * sqrtf(ss)
                cs.withUnsafeMutableBufferPointer { cb in
                    radius.withUnsafeBufferPointer { rb in
                        vDSP_vsma(rb.baseAddress!, 1, &w, cb.baseAddress!.advanced(by: r * C), 1, cb.baseAddress!.advanced(by: r * C), 1, vDSP_Length(C))
                    }
                }
            }
        }
        return cs.withUnsafeBufferPointer { csb in
            (0..<count).map { r in retrieve(hs.advanced(by: r * D), csb.baseAddress!.advanced(by: r * C)) }
        }
    }

    /// Diagnostics: is `token` in the dictionary, which cluster rank (by centroid score) holds it, its exact logit rank
    /// inside that cluster, and the best exact logit among the probed candidates vs the token's logit.
    public func diagnose(_ h: UnsafePointer<Float>, token: Int32) -> (inDictionary: Bool, clusterRank: Int, clusterSize: Int) {
        guard let idx = tokenIDs.firstIndex(of: token) else { return (false, -1, 0) }
        var c = 0
        while c < clusters && Int(clusterStart[c + 1]) <= idx { c += 1 }
        var cs = [Float](repeating: 0, count: clusters)
        var q = Array(UnsafeBufferPointer(start: h, count: dim))
        if let queryScale { vDSP_vmul(q, 1, queryScale, 1, &q, 1, vDSP_Length(dim)) }
        centroids.withUnsafeBufferPointer { cb in
            cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(clusters), Int32(dim), 1, cb.baseAddress!, Int32(dim), q, 1, 0, &cs, 1)
        }
        if radiusWeight != 0 {
            var ss: Float = 0
            vDSP_svesq(h, 1, &ss, vDSP_Length(dim))
            for i in 0..<clusters { cs[i] += radiusWeight * sqrtf(ss) * radius[i] }
        }
        let mine = cs[c]
        let rank = cs.filter { $0 > mine }.count
        return (true, rank, Int(clusterStart[c + 1] - clusterStart[c]))
    }

    private func retrieve(_ h: UnsafePointer<Float>, _ cs: UnsafePointer<Float>) -> Int32 {
        let C = clusters, D = dim
        // top-P clusters
        let P = min(probes, C)
        var top = [(Float, Int32)](repeating: (-.infinity, 0), count: P)
        for c in 0..<C {
            let v = cs[c]
            if v > top[P - 1].0 {
                var k = P - 1
                while k > 0 && top[k - 1].0 < v { top[k] = top[k - 1]; k -= 1 }
                top[k] = (v, Int32(c))
            }
        }
        // int8 candidate scores
        var q = [Int8](repeating: 0, count: D)
        var hs: Float = 0
        kk_quantize_i8(h, Int32(D), &q, &hs)
        let R = max(1, min(rescore, 64))
        var keep = [(Float, Int32)](repeating: (-.infinity, -1), count: R)
        var tmp = [Int32](repeating: 0, count: maxClusterSize)
        rowsI8.withUnsafeBufferPointer { rb in
            rowScale.withUnsafeBufferPointer { sb in
                for (_, c) in top {
                    let a = Int(clusterStart[Int(c)]), b = Int(clusterStart[Int(c) + 1])
                    guard b > a else { continue }
                    kk_dot_i8_rows(rb.baseAddress!.advanced(by: a * D), Int32(D), Int32(b - a), q, &tmp)
                    for r in 0..<(b - a) {
                        let v = Float(tmp[r]) * sb[a + r]
                        if v > keep[R - 1].0 {
                            var k = R - 1
                            while k > 0 && keep[k - 1].0 < v { keep[k] = keep[k - 1]; k -= 1 }
                            keep[k] = (v, Int32(a + r))
                        }
                    }
                }
            }
        }
        // exact rescoring from the GGUF rows
        let base = gguf.pointer(for: embInfo)
        let row = UnsafeMutablePointer<Float>.allocate(capacity: D); defer { row.deallocate() }
        var bestTok: Int32 = tokenIDs.first ?? 0
        var bestVal: Float = -.infinity
        for (_, idx) in keep where idx >= 0 {
            let tok = tokenIDs[Int(idx)]
            let src = base.advanced(by: Int(tok) * embRowBytes)
            if embInfo.type == .q6_K {
                GGUFQuantization.dequantizeQ6KRow(src: src, dst: row, elements: D)
            } else {
                GGUFQuantization.dequantizeQ4_0Row(src: src, dst: row, elements: D)
            }
            var v: Float = 0
            vDSP_dotpr(row, 1, h, 1, &v, vDSP_Length(D))
            if v > bestVal { bestVal = v; bestTok = tok }
        }
        return bestTok
    }
}
