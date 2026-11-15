import Accelerate
import Foundation
import KomorebiANEDirect

// Shared by the direct-ANE engines (Gemma 4, Qwen3-ASR): MIL text writer, weight blob with LUT palettes,
// k-means palettisation.

// MARK: - MIL text writer

final class MILWriter {
    static let header = "program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3500.14.1\"}, {\"coremlc-version\", \"3500.32.1\"}})]\n{\n"
    var text = ""
    private var n = 0
    let S: Int
    init(S: Int) { self.S = S }

    func raw(_ s: String) { text += s }
    func line(_ s: String) { text += "        " + s + "\n" }
    private func v(_ p: String) -> String { n += 1; return "\(p)\(n)" }
    private func ty(_ d: [Int]) -> String { "tensor<fp16, [\(d.map(String.init).joined(separator: ", "))]>" }
    static func f16(_ x: Float) -> String { String(format: "%a", Double(Float(Float16(x)))) }

    func scalar(_ x: Float) -> String {
        let nm = v("c"); line("fp16 \(nm) = const()[name = string(\"\(nm)\"), val = fp16(\(MILWriter.f16(x)))];"); return nm
    }
    private func ints(_ xs: [Int]) -> String {
        let nm = v("i")
        line("tensor<int32, [\(xs.count)]> \(nm) = const()[name = string(\"\(nm)\"), val = tensor<int32, [\(xs.count)]>([\(xs.map(String.init).joined(separator: ", "))])];")
        return nm
    }
    private func int1(_ x: Int) -> String { let nm = v("a"); line("int32 \(nm) = const()[name = string(\"\(nm)\"), val = int32(\(x))];"); return nm }
    private func bool(_ b: Bool) -> String { let nm = v("b"); line("bool \(nm) = const()[name = string(\"\(nm)\"), val = bool(\(b))];"); return nm }
    private func str(_ s: String) -> String { let nm = v("s"); line("string \(nm) = const()[name = string(\"\(nm)\"), val = string(\"\(s)\")];"); return nm }

    @discardableResult
    func op(_ d: [Int], _ call: String, name: String? = nil) -> String {
        let nm = name ?? v("t"); line("\(ty(d)) \(nm) = \(call)[name = string(\"\(nm)\")];"); return nm
    }
    func linear(_ x: String, _ w: String, _ d: [Int], name: String? = nil) -> String { op(d, "linear(weight = \(w), x = \(x))", name: name) }
    func mul(_ a: String, _ b: String, _ d: [Int], name: String? = nil) -> String { op(d, "mul(x = \(a), y = \(b))", name: name) }
    func add(_ a: String, _ b: String, _ d: [Int], name: String? = nil) -> String { op(d, "add(x = \(a), y = \(b))", name: name) }
    func gelu(_ x: String, _ d: [Int]) -> String { let m = str("TANH_APPROXIMATION"); return op(d, "gelu(mode = \(m), x = \(x))") }
    func reshape(_ x: String, _ d: [Int], name: String? = nil) -> String { let s = ints(d); return op(d, "reshape(shape = \(s), x = \(x))", name: name) }
    func linear(_ x: String, _ w: String, bias b: String, _ d: [Int], name: String? = nil) -> String {
        op(d, "linear(bias = \(b), weight = \(w), x = \(x))", name: name)
    }
    /// Exact (erf) GELU, as PyTorch's `nn.GELU()` / MLX's `nn.gelu`.
    func geluExact(_ x: String, _ d: [Int], name: String? = nil) -> String {
        let m = str(MILWriter.exactGELUMode); return op(d, "gelu(mode = \(m), x = \(x))", name: name)
    }
    /// Experiments: which MIL GELU mode stands in for the exact GELU.
    nonisolated(unsafe) static var exactGELUMode = ProcessInfo.processInfo.environment["KOMOREBI_GELU_MODE"] ?? "EXACT"
    func silu(_ x: String, _ d: [Int]) -> String { op(d, "silu(x = \(x))") }
    /// LayerNorm over the last axis with weight and bias.
    func layerNorm(_ x: String, dims: [Int], gamma: String, beta: String, eps: Float, name: String? = nil) -> String {
        let axes = ints([-1]), e = scalar(eps)
        return op(dims, "layer_norm(axes = \(axes), beta = \(beta), epsilon = \(e), gamma = \(gamma), x = \(x))", name: name)
    }
    func transpose(_ x: String, perm: [Int], _ d: [Int], name: String? = nil) -> String {
        let p = ints(perm); return op(d, "transpose(perm = \(p), x = \(x))", name: name)
    }
    func matmul(_ x: String, _ y: String, transposeX: Bool = false, transposeY: Bool = false, _ d: [Int]) -> String {
        let tx = bool(transposeX), ty = bool(transposeY)
        return op(d, "matmul(transpose_x = \(tx), transpose_y = \(ty), x = \(x), y = \(y))")
    }
    func softmax(_ x: String, axis: Int, _ d: [Int]) -> String { let a = int1(axis); return op(d, "softmax(axis = \(a), x = \(x))") }
    /// 2-D convolution (NCHW), groups 1, dilation 1, explicit padding [top, bottom, left, right].
    func conv(_ x: String, weight: String, bias: String, strides: [Int], pad: [Int], _ d: [Int]) -> String {
        let st = ints(strides), pd = ints(pad), dl = ints([1, 1]), g = int1(1), pt = str("custom")
        return op(d, "conv(bias = \(bias), dilations = \(dl), groups = \(g), pad = \(pd), pad_type = \(pt), strides = \(st), weight = \(weight), x = \(x))")
    }
    func slice(_ x: String, begin: [Int], end: [Int], _ d: [Int]) -> String {
        let b = ints(begin), e = ints(end)
        return op(d, "slice_by_index(begin = \(b), end = \(e), x = \(x))")
    }
    func concatLast(_ xs: [String], _ d: [Int], name: String? = nil) -> String {
        let ax = int1(-1), il = bool(false)
        return op(d, "concat(axis = \(ax), interleave = \(il), values = (\(xs.joined(separator: ", "))))", name: name)
    }
    func splitHalves(_ x: String, half: Int, _ d: [Int]) -> (String, String) {
        let ax = int1(-1), ss = ints([half, half])
        let a = v("p"), b = v("p")
        line("\(ty(d)) \(a), \(ty(d)) \(b) = split(axis = \(ax), split_sizes = \(ss), x = \(x))[name = string(\"\(a)\")];")
        return (a, b)
    }
    /// RMSNorm over the last axis via layer_norm([x, -x]) (mean 0, var = mean(x^2)), robust to fp16 overflow.
    func rmsNorm(_ x: String, dims: [Int], weight: String?, eps: Float, name: String? = nil) -> String {
        let C = dims.last!
        var d2 = dims; d2[d2.count - 1] = 2 * C
        let neg = mul(x, scalar(-1), dims)
        let cat = concatLast([x, neg], d2)
        let axes = ints([-1]), e = scalar(eps)
        let ln = op(d2, "layer_norm(axes = \(axes), epsilon = \(e), x = \(cat))")
        let (a, _) = splitHalves(ln, half: C, dims)
        if let weight { return mul(a, weight, dims, name: name) }
        if let name { return mul(a, scalar(1), dims, name: name) }
        return a
    }
    /// NEOX rotate-half RoPE with per-row cos/sin inputs of shape [1, S, 1, hd] (halves duplicated).
    func rope(_ x: String, dims: [Int], cos: String, sin: String) -> String {
        let hd = dims.last!
        var hdims = dims; hdims[hdims.count - 1] = hd / 2
        let (x1, x2) = splitHalves(x, half: hd / 2, hdims)
        let nx2 = mul(x2, scalar(-1), hdims)
        let rot = concatLast([nx2, x1], dims)
        return add(mul(x, cos, dims), mul(rot, sin, dims), dims)
    }
}

// MARK: - Weights -> MIL blob (grouped-channel LUT palettes)

final class WeightBank {
    let bits: Int
    let group: Int
    let dryRun: Bool
    let blob = ANEBlobBuilder()
    private var dryOffset: UInt64 = 64
    private(set) var declarations: [String] = []
    private(set) var rows: [String: Int] = [:]

    /// Lloyd iterations and start of the k-means palettes (the Gemma engine keeps 16 / quantiles, so its cached
    /// palettes stay valid; Qwen3-ASR uses 300 / cube-root, about half the error at 6-8 bits).
    var iterations = 16
    var companded = false

    /// `dryRun`: only lay out offsets (same layout as `ANEBlobBuilder`) so MIL can be regenerated without quantising.
    init(bits: Int, group: Int, dryRun: Bool = false) { self.bits = bits; self.group = group; self.dryRun = dryRun }

    private func reserve(_ length: Int) -> UInt64 {
        let off = dryOffset
        dryOffset += 64 + UInt64((length + 63) & ~63)
        return off
    }

    private static func blobRef(_ off: UInt64) -> String { "BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(\(off)))" }

    func vec(_ name: String, _ gguf: GGUFFile, _ tensor: String) throws {
        let vals = try gguf.readFloat32Tensor(tensor)
        let h = vals.map { Float16($0).bitPattern }
        let off = dryRun ? reserve(h.count * 2) : h.withUnsafeBytes { blob.append($0.baseAddress!, length: $0.count, dtype: 1) }
        declarations.append("tensor<fp16, [\(vals.count)]> \(name) = const()[name = string(\"\(name)\"), val = tensor<fp16, [\(vals.count)]>(\(WeightBank.blobRef(off)))];")
    }

    func lut(_ name: String, _ gguf: GGUFFile, _ tensor: String, rowStart: Int = 0, rowCount: Int? = nil, rowList: [Int32]? = nil) throws {
        let info = try gguf.tensor(tensor)
        let K = info.dimensions[0]
        let N = rowList?.count ?? rowCount ?? (info.dimensions.count > 1 ? info.dimensions[1] : 1)
        let base = gguf.pointer(for: info)
        let rowBytes = info.type.rowByteSize(elements: K)
        let type = info.type
        guard N % group == 0 else { throw GemmaEngineError.inferenceFailed("\(tensor): rows \(N) not divisible by LUT group \(group)") }
        if dryRun {
            let E = 1 << bits
            let offI = reserve(N * K * bits / 8)
            let offL = reserve((N / group) * E * 2)
            declarations.append("tensor<fp16, [\(N), \(K)]> \(name) = constexpr_lut_to_dense(indices = tensor<uint\(bits), [\(N), \(K)]>(\(WeightBank.blobRef(offI))), lut = tensor<fp16, [\(N / group), 1, \(E), 1]>(\(WeightBank.blobRef(offL))))[name = string(\"\(name)\")];")
            rows[name] = N
            return
        }
        let q = try LUTQuantizer.quantize(rows: N, cols: K, bits: bits, group: group, iterations: iterations, companded: companded) { r0, cnt, dst in
            for r in 0..<cnt {
                let srcRow = rowList.map { Int($0[r0 + r]) } ?? (rowStart + r0 + r)
                let src = base.advanced(by: srcRow * rowBytes)
                let d = dst.advanced(by: r * K)
                switch type {
                case .q4_0: GGUFQuantization.dequantizeQ4_0Row(src: src, dst: d, elements: K)
                case .q6_K: GGUFQuantization.dequantizeQ6KRow(src: src, dst: d, elements: K)
                case .q8_0: GGUFQuantization.dequantizeQ8_0Row(src: src, dst: d, elements: K)
                case .f16:
                    let h = src.assumingMemoryBound(to: UInt16.self)
                    for j in 0..<K { d[j] = Float(Float16(bitPattern: h[j])) }
                case .f32:
                    memcpy(d, src, K * 4)
                default:
                    break
                }
            }
        } validate: {
            [.q4_0, .q6_K, .q8_0, .f16, .f32].contains(type)
        }
        let code: UInt32 = bits == 4 ? 11 : bits == 8 ? 3 : bits == 6 ? 13 : bits == 3 ? 12 : bits == 2 ? 10 : 9
        let offI = q.indices.withUnsafeBytes { blob.append($0.baseAddress!, length: $0.count, dtype: code) }
        let offL = q.lut.withUnsafeBytes { blob.append($0.baseAddress!, length: $0.count, dtype: 1) }
        let E = 1 << bits
        declarations.append("tensor<fp16, [\(N), \(K)]> \(name) = constexpr_lut_to_dense(indices = tensor<uint\(bits), [\(N), \(K)]>(\(WeightBank.blobRef(offI))), lut = tensor<fp16, [\(N / group), 1, \(E), 1]>(\(WeightBank.blobRef(offL))))[name = string(\"\(name)\")];")
        rows[name] = N
    }

    /// A dense fp16 constant of any shape (row-major `values`), e.g. convolution kernels or position tables.
    func dense(_ name: String, shape: [Int], values: () throws -> [Float]) rethrows {
        let count = shape.reduce(1, *)
        let ty = "tensor<fp16, [\(shape.map(String.init).joined(separator: ", "))]>"
        let off: UInt64
        if dryRun {
            off = reserve(count * 2)
        } else {
            let vals = try values()
            precondition(vals.count == count, "\(name): \(vals.count) values for shape \(shape)")
            let h = vals.map { Float16($0).bitPattern }
            off = h.withUnsafeBytes { blob.append($0.baseAddress!, length: $0.count, dtype: 1) }
        }
        declarations.append("\(ty) \(name) = const()[name = string(\"\(name)\"), val = \(ty)(\(WeightBank.blobRef(off)))];")
    }

    /// Every element of a GGUF tensor (any supported type) as fp32, row-major.
    static func floats(_ gguf: GGUFFile, _ tensor: String) throws -> [Float] {
        let info = try gguf.tensor(tensor)
        let K = info.dimensions[0]
        let N = info.elementCount / K
        var out = [Float](repeating: 0, count: N * K)
        try out.withUnsafeMutableBufferPointer { dst in
            for r in 0..<N { try gguf.dequantizeRow(tensor: info, row: r, into: dst.baseAddress!.advanced(by: r * K)) }
        }
        return out
    }

    func finish() -> Data { blob.finish() }
}

/// 1-D k-means palettisation per group of `group` consecutive rows (one LUT per group), packed LSB-first.
enum LUTQuantizer {
    struct Result { var indices: [UInt8]; var lut: [UInt16] }

    /// `iterations`: Lloyd iterations from the quantile start (stops early once the centroids settle). The Gemma
    /// engine keeps 16 so its cached palettes stay what they were.
    /// `companded`: start from centroids spread like p(x)^(1/3) (the high-resolution optimum) instead of at the
    /// quantiles, which leaves too few levels for the tails at 6-8 bits (2-5x the optimal error).
    static func quantize(rows N: Int, cols K: Int, bits: Int, group G: Int, iterations: Int = 16, companded: Bool = false,
                         fetch: (Int, Int, UnsafeMutablePointer<Float>) -> Void,
                         validate: () -> Bool) throws -> Result {
        guard validate() else { throw GemmaEngineError.inferenceFailed("unsupported tensor type for LUT conversion") }
        let E = 1 << bits
        let groups = N / G
        let n = G * K
        let groupBytes = n * bits / 8
        var indices = [UInt8](repeating: 0, count: groups * groupBytes)
        var lut = [UInt16](repeating: 0, count: groups * E)
        indices.withUnsafeMutableBufferPointer { ib in
            lut.withUnsafeMutableBufferPointer { lb in
                let ibase = ib.baseAddress!, lbase = lb.baseAddress!
                DispatchQueue.concurrentPerform(iterations: groups) { g in
                    let vals = UnsafeMutablePointer<Float>.allocate(capacity: n)
                    let sorted = UnsafeMutablePointer<Float>.allocate(capacity: n)
                    let pre = UnsafeMutablePointer<Double>.allocate(capacity: n + 1)
                    defer { vals.deallocate(); sorted.deallocate(); pre.deallocate() }
                    fetch(g * G, G, vals)
                    sorted.update(from: vals, count: n)
                    vDSP_vsort(sorted, vDSP_Length(n), 1)
                    pre[0] = 0
                    for i in 0..<n { pre[i + 1] = pre[i] + Double(sorted[i]) }
                    var uniq: [Float] = []
                    uniq.reserveCapacity(1024)
                    for i in 0..<n where i == 0 || sorted[i] != sorted[i - 1] { uniq.append(sorted[i]) }

                    var cent = [Float](repeating: 0, count: E)
                    if uniq.count <= E {
                        for e in 0..<E { cent[e] = uniq[min(e, uniq.count - 1)] }
                    } else {
                        var dup = false
                        if companded {
                            // Histogram density -> cumulative p^(1/3) -> one centroid per equal share.
                            let bins = 1024
                            let lo = sorted[0], hi = sorted[n - 1]
                            let width = (hi - lo) / Float(bins)
                            var hist = [Double](repeating: 0, count: bins)
                            for i in 0..<n { hist[min(bins - 1, Int((sorted[i] - lo) / width))] += 1 }
                            var cum = [Double](repeating: 0, count: bins + 1)
                            for b in 0..<bins { cum[b + 1] = cum[b] + pow(hist[b], 1.0 / 3.0) }
                            var b = 0
                            for e in 0..<E {
                                let target = (Double(e) + 0.5) / Double(E) * cum[bins]
                                while b < bins - 1 && cum[b + 1] < target { b += 1 }
                                let frac = cum[b + 1] > cum[b] ? (target - cum[b]) / (cum[b + 1] - cum[b]) : 0.5
                                cent[e] = lo + (Float(b) + Float(frac)) * width
                                if e > 0 && cent[e] <= cent[e - 1] { dup = true }
                            }
                        } else {
                            for e in 0..<E {
                                cent[e] = sorted[min(n - 1, (2 * e + 1) * n / (2 * E))]
                                if e > 0 && cent[e] <= cent[e - 1] { dup = true }
                            }
                        }
                        if dup { for e in 0..<E { cent[e] = uniq[min(uniq.count - 1, (2 * e + 1) * uniq.count / (2 * E))] } }
                        func lowerBound(_ x: Float) -> Int {
                            var lo = 0, hi = n
                            while lo < hi { let mid = (lo + hi) >> 1; if sorted[mid] < x { lo = mid + 1 } else { hi = mid } }
                            return lo
                        }
                        var bnd = [Float](repeating: 0, count: E)
                        for _ in 0..<iterations {
                            for e in 0..<(E - 1) { bnd[e] = 0.5 * (cent[e] + cent[e + 1]) }
                            var lo = 0
                            var moved = false
                            for e in 0..<E {
                                let hi = e == E - 1 ? n : lowerBound(bnd[e])
                                if hi > lo {
                                    let c = Float((pre[hi] - pre[lo]) / Double(hi - lo))
                                    if c != cent[e] { moved = true }
                                    cent[e] = c
                                }
                                lo = max(lo, hi)
                            }
                            if !moved { break }
                        }
                    }
                    var bnd = [Float](repeating: .infinity, count: E)
                    for e in 0..<(E - 1) { bnd[e] = 0.5 * (cent[e] + cent[e + 1]) }
                    for e in 0..<E { lbase[g * E + e] = Float16(cent[e]).bitPattern }
                    let dst = ibase.advanced(by: g * groupBytes)
                    var bit = 0
                    for i in 0..<n {
                        let x = vals[i]
                        var lo = 0, hi = E - 1          // index = number of boundaries <= x
                        while lo < hi { let mid = (lo + hi) >> 1; if bnd[mid] <= x { lo = mid + 1 } else { hi = mid } }
                        var q = lo
                        for _ in 0..<bits {
                            if q & 1 != 0 { dst[bit >> 3] |= UInt8(1 << (bit & 7)) }
                            q >>= 1
                            bit += 1
                        }
                    }
                }
            }
        }
        return Result(indices: indices, lut: lut)
    }
}
