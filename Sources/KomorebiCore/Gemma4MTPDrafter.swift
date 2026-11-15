import Accelerate
import Darwin
import Foundation
import KomorebiANEDirect

/// Gemma 4 E2B MTP assistant drafter (`google/gemma-4-E2B-it-assistant`, `gemma4_assistant`), CPU implementation.
///
/// Mirrors vLLM `gemma4_mtp.py` + `Gemma4Speculator`:
///  - input = concat(target_embed(token) * sqrt(1536), backbone_hidden) -> pre_projection (3072 -> 256)
///  - 4 decoder layers with Q-only attention reading the target's KV cache (sliding -> target layer 13,
///    full -> target layer 14), query position fixed at the last verified position for every draft step
///  - final norm -> centroid-masked tied LM head (2048 centroids, top-32 -> 4096 candidates) -> argmax
///  - post_projection (256 -> 1536) gives the backbone hidden for the next draft step.
/// Weights stay bf16 in the mmapped safetensors file.
public final class Gemma4MTPDrafter: @unchecked Sendable {
    public struct KVSource {
        public let k: UnsafePointer<Float>
        public let v: UnsafePointer<Float>
        public let headDim: Int
        public let sliding: Bool
    }

    let hidden: Int, backbone: Int, nHeads: Int, inter: Int, vocab: Int, numCentroids: Int, topK: Int, perCentroid: Int
    let window: Int
    let eps: Float = 1e-6
    let layerSliding: [Bool]
    let headDims: [Int]

    private let mapBase: UnsafeMutableRawPointer
    private let mapLength: Int
    private var tensors: [String: (offset: Int, shape: [Int], dtype: String)] = [:]
    private var dataStart = 0

    private struct Layer {
        let inNorm: [Float], qNorm: [Float], postAttnNorm: [Float], preFFNorm: [Float], postFFNorm: [Float]
        let q: UnsafePointer<UInt16>, o: UnsafePointer<UInt16>, gate: UnsafePointer<UInt16>, up: UnsafePointer<UInt16>, down: UnsafePointer<UInt16>
        let scalar: Float
    }
    private var layers: [Layer] = []
    private var finalNorm: [Float] = []
    private var preProj: UnsafePointer<UInt16>!
    private var postProj: UnsafePointer<UInt16>!
    private var centroids: UnsafePointer<UInt16>!
    private var embed: UnsafePointer<UInt16>!
    private var ordering: [Int32] = []
    /// Diagnostics: when > 0, the first draft step of each `propose` keeps its `debugTopK` best candidates and the
    /// whole candidate set (to tell a near miss from a token the centroid head never offered).
    public var debugTopK = 0
    public private(set) var lastTopCandidates: [Int32] = []
    public private(set) var lastCandidateSet: Set<Int32> = []
    private var firstStep = false

    public init(directory: URL) throws {
        let cfgData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard let cfg = try JSONSerialization.jsonObject(with: cfgData) as? [String: Any],
              let text = cfg["text_config"] as? [String: Any] else {
            throw GemmaEngineError.inferenceFailed("drafter config.json unreadable")
        }
        hidden = text["hidden_size"] as? Int ?? 256
        backbone = cfg["backbone_hidden_size"] as? Int ?? 1536
        nHeads = text["num_attention_heads"] as? Int ?? 4
        inter = text["intermediate_size"] as? Int ?? 2048
        vocab = text["vocab_size"] as? Int ?? 262144
        numCentroids = cfg["num_centroids"] as? Int ?? 2048
        topK = cfg["centroid_intermediate_top_k"] as? Int ?? 32
        perCentroid = vocab / numCentroids
        window = text["sliding_window"] as? Int ?? 512
        let types = text["layer_types"] as? [String] ?? ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
        layerSliding = types.map { $0 == "sliding_attention" }
        let hd = text["head_dim"] as? Int ?? 256, ghd = text["global_head_dim"] as? Int ?? 512
        headDims = layerSliding.map { $0 ? hd : ghd }

        let path = directory.appendingPathComponent("model.safetensors").path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw GemmaEngineError.inferenceFailed("cannot open drafter weights") }
        defer { close(fd) }
        var st = stat()
        fstat(fd, &st)
        mapLength = Int(st.st_size)
        guard let m = mmap(nil, mapLength, PROT_READ, MAP_PRIVATE, fd, 0), m != MAP_FAILED else {
            throw GemmaEngineError.inferenceFailed("cannot mmap drafter weights")
        }
        mapBase = m
        let headerLen = Int(mapBase.loadUnaligned(as: UInt64.self))
        let header = Data(bytes: mapBase.advanced(by: 8), count: headerLen)
        guard let json = try JSONSerialization.jsonObject(with: header) as? [String: Any] else {
            throw GemmaEngineError.inferenceFailed("bad safetensors header")
        }
        dataStart = 8 + headerLen
        for (name, v) in json where name != "__metadata__" {
            guard let d = v as? [String: Any], let offs = d["data_offsets"] as? [Int], let shape = d["shape"] as? [Int],
                  let dtype = d["dtype"] as? String else { continue }
            tensors[name] = (dataStart + offs[0], shape, dtype)
        }

        func bf16(_ name: String) throws -> UnsafePointer<UInt16> {
            guard let t = tensors[name], t.dtype == "BF16" else { throw GemmaEngineError.inferenceFailed("drafter tensor \(name) missing") }
            return UnsafePointer(mapBase.advanced(by: t.offset).assumingMemoryBound(to: UInt16.self))
        }
        func f32(_ name: String) throws -> [Float] {
            guard let t = tensors[name] else { throw GemmaEngineError.inferenceFailed("drafter tensor \(name) missing") }
            let n = t.shape.reduce(1, *)
            let p = mapBase.advanced(by: t.offset).assumingMemoryBound(to: UInt16.self)
            return (0..<n).map { Float(bitPattern: UInt32(p[$0]) << 16) }
        }
        for i in 0..<layerSliding.count {
            let pfx = "model.layers.\(i)."
            layers.append(Layer(
                inNorm: try f32(pfx + "input_layernorm.weight"),
                qNorm: try f32(pfx + "self_attn.q_norm.weight"),
                postAttnNorm: try f32(pfx + "post_attention_layernorm.weight"),
                preFFNorm: try f32(pfx + "pre_feedforward_layernorm.weight"),
                postFFNorm: try f32(pfx + "post_feedforward_layernorm.weight"),
                q: try bf16(pfx + "self_attn.q_proj.weight"),
                o: try bf16(pfx + "self_attn.o_proj.weight"),
                gate: try bf16(pfx + "mlp.gate_proj.weight"),
                up: try bf16(pfx + "mlp.up_proj.weight"),
                down: try bf16(pfx + "mlp.down_proj.weight"),
                scalar: try f32(pfx + "layer_scalar").first ?? 1
            ))
        }
        finalNorm = try f32("model.norm.weight")
        preProj = try bf16("pre_projection.weight")
        postProj = try bf16("post_projection.weight")
        centroids = try bf16("masked_embedding.centroids.weight")
        embed = try bf16("model.embed_tokens.weight")
        guard let ord = tensors["masked_embedding.token_ordering"], ord.dtype == "I64" else {
            throw GemmaEngineError.inferenceFailed("drafter token_ordering missing")
        }
        let op = mapBase.advanced(by: ord.offset).assumingMemoryBound(to: Int64.self)
        ordering = (0..<vocab).map { Int32(op[$0]) }
    }

    deinit { munmap(mapBase, mapLength) }

    // MARK: - math helpers

    /// y = W x with W bf16 [rows, cols]; parallel over row blocks for big matrices.
    private func gemv(_ W: UnsafePointer<UInt16>, rows: Int, cols: Int, _ x: UnsafePointer<Float>, _ y: UnsafeMutablePointer<Float>) {
        if rows * cols < 262_144 {
            kk_gemv_bf16(W, Int32(rows), Int32(cols), x, y)
            return
        }
        let blocks = min(8, max(1, rows / 64))
        let per = (rows + blocks - 1) / blocks
        DispatchQueue.concurrentPerform(iterations: blocks) { b in
            let r0 = b * per, r1 = min(rows, r0 + per)
            if r1 > r0 { kk_gemv_bf16(W.advanced(by: r0 * cols), Int32(r1 - r0), Int32(cols), x, y.advanced(by: r0)) }
        }
    }

    private func rmsNorm(_ x: UnsafeMutablePointer<Float>, _ n: Int, _ w: [Float]?, into y: UnsafeMutablePointer<Float>) {
        var ss: Float = 0
        vDSP_svesq(x, 1, &ss, vDSP_Length(n))
        var inv = 1 / sqrtf(ss / Float(n) + eps)
        vDSP_vsmul(x, 1, &inv, y, 1, vDSP_Length(n))
        if let w { vDSP_vmul(y, 1, w, 1, y, 1, vDSP_Length(n)) }
    }

    private static func geluTanh(_ x: Float) -> Float {
        let inner = min(15, max(-15, 0.7978845608 * (x + 0.044715 * x * x * x)))
        return 0.5 * x * (1 + tanhf(inner))
    }

    // MARK: - drafting

    /// Proposes `count` tokens after `token` (which sits at position `position + 1`), starting from the target's
    /// final-norm hidden `targetHidden` at `position`. `targetEmbed(token, dst)` writes the target's scaled embedding.
    public func propose(token: Int32, targetHidden: UnsafePointer<Float>, position: Int, count: Int,
                        kv: (Bool) -> KVSource,
                        rope: (_ sliding: Bool, _ position: Int) -> (cos: UnsafePointer<Float16>, sin: UnsafePointer<Float16>),
                        targetEmbed: (Int32, UnsafeMutablePointer<Float>) -> Void) -> [Int32] {
        var out: [Int32] = []
        let x = UnsafeMutablePointer<Float>.allocate(capacity: 2 * backbone); defer { x.deallocate() }
        let back = UnsafeMutablePointer<Float>.allocate(capacity: backbone); defer { back.deallocate() }
        back.update(from: targetHidden, count: backbone)
        var tok = token
        firstStep = true
        for _ in 0..<count {
            targetEmbed(tok, x)
            x.advanced(by: backbone).update(from: back, count: backbone)
            let next = step(x: x, position: position, kv: kv, rope: rope, backboneOut: back)
            out.append(next)
            tok = next
        }
        return out
    }

    private func step(x: UnsafeMutablePointer<Float>, position P: Int, kv: (Bool) -> KVSource,
                      rope: (Bool, Int) -> (cos: UnsafePointer<Float16>, sin: UnsafePointer<Float16>),
                      backboneOut: UnsafeMutablePointer<Float>) -> Int32 {
        let H = hidden
        let maxQ = nHeads * (headDims.max() ?? 512)
        let h = UnsafeMutablePointer<Float>.allocate(capacity: H); defer { h.deallocate() }
        let a = UnsafeMutablePointer<Float>.allocate(capacity: max(H, inter)); defer { a.deallocate() }
        let t = UnsafeMutablePointer<Float>.allocate(capacity: max(H, inter)); defer { t.deallocate() }
        let q = UnsafeMutablePointer<Float>.allocate(capacity: maxQ); defer { q.deallocate() }
        let att = UnsafeMutablePointer<Float>.allocate(capacity: maxQ); defer { att.deallocate() }
        gemv(preProj, rows: H, cols: 2 * backbone, x, h)

        for (li, L) in layers.enumerated() {
            let hd = headDims[li], sliding = layerSliding[li]
            // attention (Q only)
            rmsNorm(h, H, L.inNorm, into: a)
            gemv(L.q, rows: nHeads * hd, cols: H, a, q)
            let (cosT, sinT) = rope(sliding, P)
            let half = hd / 2
            for hh in 0..<nHeads {
                let qh = q.advanced(by: hh * hd)
                rmsNorm(qh, hd, L.qNorm, into: qh)
                for j in 0..<half {
                    let c = Float(cosT[j]), s = Float(sinT[j])
                    let x1 = qh[j], x2 = qh[j + half]
                    qh[j] = x1 * c - x2 * s
                    qh[j + half] = x2 * c + x1 * s
                }
            }
            let src = kv(sliding)
            let lo = sliding ? max(0, P - window + 1) : 0
            let T = P - lo + 1
            let sc = UnsafeMutablePointer<Float>.allocate(capacity: nHeads * T); defer { sc.deallocate() }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(nHeads), Int32(T), Int32(hd), 1,
                        q, Int32(hd), src.k.advanced(by: lo * hd), Int32(hd), 0, sc, Int32(T))
            for hh in 0..<nHeads {
                let row = sc.advanced(by: hh * T)
                var mx: Float = 0
                vDSP_maxv(row, 1, &mx, vDSP_Length(T))
                var neg = -mx
                vDSP_vsadd(row, 1, &neg, row, 1, vDSP_Length(T))
                var n32 = Int32(T)
                vvexpf(row, row, &n32)
                var sum: Float = 0
                vDSP_sve(row, 1, &sum, vDSP_Length(T))
                var inv = 1 / sum
                vDSP_vsmul(row, 1, &inv, row, 1, vDSP_Length(T))
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(nHeads), Int32(hd), Int32(T), 1,
                        sc, Int32(T), src.v.advanced(by: lo * hd), Int32(hd), 0, att, Int32(hd))
            gemv(L.o, rows: H, cols: nHeads * hd, att, t)
            rmsNorm(t, H, L.postAttnNorm, into: t)
            vDSP_vadd(h, 1, t, 1, h, 1, vDSP_Length(H))
            // MLP
            rmsNorm(h, H, L.preFFNorm, into: a)
            let g = UnsafeMutablePointer<Float>.allocate(capacity: inter); defer { g.deallocate() }
            let u = UnsafeMutablePointer<Float>.allocate(capacity: inter); defer { u.deallocate() }
            gemv(L.gate, rows: inter, cols: H, a, g)
            gemv(L.up, rows: inter, cols: H, a, u)
            for i in 0..<inter { g[i] = Self.geluTanh(g[i]) * u[i] }
            gemv(L.down, rows: H, cols: inter, g, t)
            rmsNorm(t, H, L.postFFNorm, into: t)
            vDSP_vadd(h, 1, t, 1, h, 1, vDSP_Length(H))
            var s = L.scalar
            vDSP_vsmul(h, 1, &s, h, 1, vDSP_Length(H))
        }
        let dh = UnsafeMutablePointer<Float>.allocate(capacity: H); defer { dh.deallocate() }
        rmsNorm(h, H, finalNorm, into: dh)

        // centroid-masked tied head
        let cs = UnsafeMutablePointer<Float>.allocate(capacity: numCentroids); defer { cs.deallocate() }
        kk_gemv_bf16(centroids, Int32(numCentroids), Int32(H), dh, cs)
        var top = [(Float, Int)](repeating: (-.infinity, 0), count: topK)
        for c in 0..<numCentroids {
            let v = cs[c]
            if v > top[topK - 1].0 {
                var k = topK - 1
                while k > 0 && top[k - 1].0 < v { top[k] = top[k - 1]; k -= 1 }
                top[k] = (v, c)
            }
        }
        var cand = [Int32](repeating: 0, count: topK * perCentroid)
        for (i, (_, c)) in top.enumerated() {
            for j in 0..<perCentroid { cand[i * perCentroid + j] = ordering[c * perCentroid + j] }
        }
        var logits = [Float](repeating: 0, count: cand.count)
        kk_dot_bf16_rows(embed, Int32(H), cand, Int32(cand.count), dh, &logits)
        var best: Float = 0
        var bi: vDSP_Length = 0
        vDSP_maxvi(logits, 1, &best, &bi, vDSP_Length(logits.count))
        if debugTopK > 0 && firstStep {
            firstStep = false
            let order = logits.indices.sorted { logits[$0] > logits[$1] }.prefix(debugTopK)
            lastTopCandidates = order.map { cand[$0] }
            lastCandidateSet = Set(cand)
        }
        gemv(postProj, rows: backbone, cols: H, dh, backboneOut)
        return cand[Int(bi)]
    }
}
