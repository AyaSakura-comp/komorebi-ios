import Accelerate
import Foundation

/// Background-safe multi-threaded CPU Q4_0 / Q6_K fallback sharing the exact same `mmap`'d GGUF file.
/// Runs cleanly inside iOS background audio sessions (where Metal GPU submission is prohibited) and on iOS Simulator
/// if no ANE CoreML bundle is installed.
public final class Gemma4CPUEngine: @unchecked Sendable {
    public let gguf: GGUFFile
    public let config: Gemma4Config
    public let maxContext: Int

    private let tokenEmbdInfo: GGUFTensorInfo
    private let perLayerTokEmbdInfo: GGUFTensorInfo
    private let outputNorm: [Float]
    private let perLayerProjNorm: [Float]
    private let perLayerModelProjPtr: UnsafePointer<UInt16>
    private let ropeFreqs: [Float]

    private struct CPULayer {
        let attnNorm: [Float]
        let wQ: GGUFTensorInfo
        let attnQNorm: [Float]
        let wK: GGUFTensorInfo?
        let wV: GGUFTensorInfo?
        let attnKNorm: [Float]?
        let wO: GGUFTensorInfo
        let postAttnNorm: [Float]
        let ffnNorm: [Float]
        let ffnGate: GGUFTensorInfo
        let ffnUp: GGUFTensorInfo
        let ffnDown: GGUFTensorInfo
        let postFFWNorm: [Float]
        let inpGate: GGUFTensorInfo
        let proj: GGUFTensorInfo
        let postNorm: [Float]
        let outScale: Float
    }
    private let layers: [CPULayer]

    private var kCaches: [[Float]] = []
    private var vCaches: [[Float]] = []

    private let embdScale: Float
    private let perLayerEmbdScale: Float
    private let invSqrt2: Float = 1.0 / sqrtf(2.0)
    private let projNormEps: Float

    private var cachedPrefixTokens: [Int32] = []
    private var cachedPrefixK: [[Float]] = []
    private var cachedPrefixV: [[Float]] = []

    public init(gguf: GGUFFile, maxContext: Int = 1024) throws {
        self.gguf = gguf
        let cfg = Gemma4Config.from(gguf: gguf.metadata)
        self.config = cfg
        self.maxContext = min(maxContext, 2048)

        self.embdScale = sqrtf(Float(cfg.hiddenSize))
        self.perLayerEmbdScale = sqrtf(Float(cfg.perLayerInputDim)) / sqrtf(2.0)
        self.projNormEps = cfg.rmsNormEps * Float(cfg.hiddenSize)

        self.tokenEmbdInfo = try gguf.tensor("token_embd.weight")
        self.perLayerTokEmbdInfo = try gguf.tensor("per_layer_token_embd.weight")
        self.outputNorm = try gguf.readFloat32Tensor("output_norm.weight")
        self.perLayerProjNorm = try gguf.readFloat32Tensor("per_layer_proj_norm.weight")
        self.ropeFreqs = try gguf.readFloat32Tensor("rope_freqs.weight")
        let projInfo = try gguf.tensor("per_layer_model_proj.weight")
        self.perLayerModelProjPtr = gguf.pointer(for: projInfo).assumingMemoryBound(to: UInt16.self)

        var built: [CPULayer] = []
        built.reserveCapacity(cfg.numLayers)
        for il in 0..<cfg.numLayers {
            let hasKV = cfg.hasOwnKV(layer: il)
            let outScale = (try? gguf.readFloat32Tensor("blk.\(il).layer_output_scale.weight").first) ?? 1.0
            built.append(CPULayer(
                attnNorm: try gguf.readFloat32Tensor("blk.\(il).attn_norm.weight"),
                wQ: try gguf.tensor("blk.\(il).attn_q.weight"),
                attnQNorm: try gguf.readFloat32Tensor("blk.\(il).attn_q_norm.weight"),
                wK: hasKV ? try gguf.tensor("blk.\(il).attn_k.weight") : nil,
                wV: hasKV ? try gguf.tensor("blk.\(il).attn_v.weight") : nil,
                attnKNorm: hasKV ? try gguf.readFloat32Tensor("blk.\(il).attn_k_norm.weight") : nil,
                wO: try gguf.tensor("blk.\(il).attn_output.weight"),
                postAttnNorm: try gguf.readFloat32Tensor("blk.\(il).post_attention_norm.weight"),
                ffnNorm: try gguf.readFloat32Tensor("blk.\(il).ffn_norm.weight"),
                ffnGate: try gguf.tensor("blk.\(il).ffn_gate.weight"),
                ffnUp: try gguf.tensor("blk.\(il).ffn_up.weight"),
                ffnDown: try gguf.tensor("blk.\(il).ffn_down.weight"),
                postFFWNorm: try gguf.readFloat32Tensor("blk.\(il).post_ffw_norm.weight"),
                inpGate: try gguf.tensor("blk.\(il).inp_gate.weight"),
                proj: try gguf.tensor("blk.\(il).proj.weight"),
                postNorm: try gguf.readFloat32Tensor("blk.\(il).post_norm.weight"),
                outScale: outScale
            ))
        }
        self.layers = built

        for il in 0..<cfg.numOwnKVLayers {
            let slots = cfg.isSlidingLayer[il] ? cfg.slidingWindow : self.maxContext
            let hd = cfg.headDim(layer: il)
            kCaches.append([Float](repeating: 0, count: slots * hd))
            vCaches.append([Float](repeating: 0, count: slots * hd))
        }
    }

    public func clearPrefixCache() {
        cachedPrefixTokens = []
        cachedPrefixK = []
        cachedPrefixV = []
    }

    public func generate(
        inputIDs: [Int32],
        maxNewTokens: Int = 256,
        stopTokenIDs: Set<Int32>,
        staticPrefixLength: Int = 0,
        shouldCancel: (() -> Bool)? = nil,
        onToken: ((Int32) -> Void)? = nil
    ) throws -> (tokenIDs: [Int32], stats: GenerationStats) {
        guard !inputIDs.isEmpty else {
            return ([], GenerationStats(backend: "CPU Q4_0", prefixCachedTokens: 0, prefillTokens: 0, prefillSeconds: 0, decodeTokens: 0, decodeSeconds: 0))
        }

        var pos = 0
        if staticPrefixLength > 0,
           staticPrefixLength < inputIDs.count,
           cachedPrefixTokens.count == staticPrefixLength,
           Array(inputIDs.prefix(staticPrefixLength)) == cachedPrefixTokens {
            kCaches = cachedPrefixK
            vCaches = cachedPrefixV
            pos = staticPrefixLength
        }
        let cachedCount = pos

        let t0 = CFAbsoluteTimeGetCurrent()
        var nextTok: Int32 = 0
        while pos < inputIDs.count {
            if shouldCancel?() == true { break }
            let wantLogits = (pos == inputIDs.count - 1)
            nextTok = step(tokenID: inputIDs[pos], pos: pos, computeLogits: wantLogits)
            pos += 1
            if staticPrefixLength > 0 && pos == staticPrefixLength && cachedPrefixTokens != Array(inputIDs.prefix(staticPrefixLength)) {
                cachedPrefixTokens = Array(inputIDs.prefix(staticPrefixLength))
                cachedPrefixK = kCaches
                cachedPrefixV = vCaches
            }
        }
        let t1 = CFAbsoluteTimeGetCurrent()

        var out: [Int32] = []
        let t2 = CFAbsoluteTimeGetCurrent()
        if maxNewTokens > 0 && !stopTokenIDs.contains(nextTok) {
            out.append(nextTok)
            onToken?(nextTok)
            while out.count < maxNewTokens && pos < maxContext - 1 {
                if shouldCancel?() == true { break }
                let tok = step(tokenID: out.last!, pos: pos, computeLogits: true)
                pos += 1
                if stopTokenIDs.contains(tok) { break }
                out.append(tok)
                onToken?(tok)
            }
        }
        let t3 = CFAbsoluteTimeGetCurrent()

        let stats = GenerationStats(
            backend: "CPU Q4_0 (Accelerate)",
            prefixCachedTokens: cachedCount,
            prefillTokens: inputIDs.count - cachedCount,
            prefillSeconds: max(t1 - t0, 1e-6),
            decodeTokens: out.count,
            decodeSeconds: max(t3 - t2, 1e-6)
        )
        return (out, stats)
    }

    private func step(tokenID: Int32, pos: Int, computeLogits: Bool) -> Int32 {
        let H = config.hiddenSize
        let plDim = config.perLayerInputDim
        let nL = config.numLayers
        let tid = max(0, min(config.vocabSize - 1, Int(tokenID)))

        var hidden = [Float](repeating: 0, count: H)
        hidden.withUnsafeMutableBufferPointer { dst in
            gguf.dequantizeQ6KRow(tensor: tokenEmbdInfo, rowIndex: tid, into: dst.baseAddress!, count: H)
        }
        vDSP.multiply(embdScale, hidden, result: &hidden)

        var pleRaw = [Float](repeating: 0, count: nL * plDim)
        pleRaw.withUnsafeMutableBufferPointer { dst in
            gguf.dequantizeQ6KRow(tensor: perLayerTokEmbdInfo, rowIndex: tid, into: dst.baseAddress!, count: nL * plDim)
        }
        vDSP.multiply(perLayerEmbdScale, pleRaw, result: &pleRaw)

        // PLE model projection F16 -> [nL * plDim]
        var pleProj = [Float](repeating: 0, count: nL * plDim)
        let projPtr = perLayerModelProjPtr
        hidden.withUnsafeBufferPointer { hBuf in
            pleProj.withUnsafeMutableBufferPointer { outBuf in
                DispatchQueue.concurrentPerform(iterations: 35) { il in
                    let r0 = il * plDim
                    let hPtr = hBuf.baseAddress!
                    for r in r0..<(r0 + plDim) {
                        let wRow = projPtr.advanced(by: r * H)
                        var sum: Float = 0
                        for k in 0..<H {
                            sum += Float(Float16(bitPattern: wRow[k])) * hPtr[k]
                        }
                        outBuf[r] = sum
                    }
                }
            }
        }
        for il in 0..<nL {
            let base = il * plDim
            rmsNormInPlace(&pleProj, offset: base, count: plDim, weight: perLayerProjNorm, weightScale: invSqrt2, eps: projNormEps)
            for j in 0..<plDim {
                pleProj[base + j] += pleRaw[base + j]
            }
        }

        let maxQ = config.numAttentionHeads * max(config.slidingHeadDim, config.fullHeadDim)
        var normBuf = [Float](repeating: 0, count: H)
        var qBuf = [Float](repeating: 0, count: maxQ)
        var kBuf = [Float](repeating: 0, count: max(config.slidingHeadDim, config.fullHeadDim))
        var vBuf = [Float](repeating: 0, count: max(config.slidingHeadDim, config.fullHeadDim))
        var attnBuf = [Float](repeating: 0, count: maxQ)

        for il in 0..<nL {
            let l = layers[il]
            let isSWA = config.isSlidingLayer[il]
            let hd = config.headDim(layer: il)
            let qDim = config.numAttentionHeads * hd
            let interDim = config.intermediateSize(layer: il)
            let kvLayer = config.kvSourceLayer[il]
            let win = config.slidingWindow

            rmsNormCopy(hidden, weight: l.attnNorm, into: &normBuf, count: H, eps: config.rmsNormEps)

            matvecQ4_0(tensor: l.wQ, x: normBuf, into: &qBuf, K: H, N: qDim)
            for h in 0..<config.numAttentionHeads {
                rmsNormInPlace(&qBuf, offset: h * hd, count: hd, weight: l.attnQNorm, weightScale: 1.0, eps: config.rmsNormEps)
                applyRoPE(&qBuf, offset: h * hd, headDim: hd, pos: pos, base: isSWA ? config.ropeFreqBaseSWA : config.ropeFreqBaseFull, useFreqs: !isSWA)
            }

            if let wK = l.wK, let wV = l.wV, let kn = l.attnKNorm {
                matvecQ4_0(tensor: wK, x: normBuf, into: &kBuf, K: H, N: hd)
                matvecQ4_0(tensor: wV, x: normBuf, into: &vBuf, K: H, N: hd)
                rmsNormInPlace(&kBuf, offset: 0, count: hd, weight: kn, weightScale: 1.0, eps: config.rmsNormEps)
                rmsNormInPlace(&vBuf, offset: 0, count: hd, weight: nil, weightScale: 1.0, eps: config.rmsNormEps)
                applyRoPE(&kBuf, offset: 0, headDim: hd, pos: pos, base: isSWA ? config.ropeFreqBaseSWA : config.ropeFreqBaseFull, useFreqs: !isSWA)

                let slot = isSWA ? (pos % win) : pos
                kCaches[kvLayer].withUnsafeMutableBufferPointer { dst in
                    kBuf.withUnsafeBufferPointer { src in
                        _ = memcpy(dst.baseAddress!.advanced(by: slot * hd), src.baseAddress!, hd * 4)
                    }
                }
                vCaches[kvLayer].withUnsafeMutableBufferPointer { dst in
                    vBuf.withUnsafeBufferPointer { src in
                        _ = memcpy(dst.baseAddress!.advanced(by: slot * hd), src.baseAddress!, hd * 4)
                    }
                }
            }

            // MQA attention
            let startPos = isSWA ? max(0, pos - win + 1) : 0
            kCaches[kvLayer].withUnsafeBufferPointer { kc in
                vCaches[kvLayer].withUnsafeBufferPointer { vc in
                    let kcPtr = kc.baseAddress!
                    let vcPtr = vc.baseAddress!
                    for h in 0..<config.numAttentionHeads {
                        let qOff = h * hd
                        var mMax: Float = -1e30
                        var lSum: Float = 0
                        var acc = [Float](repeating: 0, count: hd)
                        for kp in startPos...pos {
                            let slot = isSWA ? (kp % win) : kp
                            var dotVal: Float = 0
                            qBuf.withUnsafeBufferPointer { qb in
                                vDSP_dotpr(qb.baseAddress!.advanced(by: qOff), 1, kcPtr.advanced(by: slot * hd), 1, &dotVal, vDSP_Length(hd))
                            }
                            let newMax = max(mMax, dotVal)
                            let alpha = expf(mMax - newMax)
                            let p = expf(dotVal - newMax)
                            lSum = lSum * alpha + p
                            let vPtr = vcPtr.advanced(by: slot * hd)
                            for d in 0..<hd {
                                acc[d] = acc[d] * alpha + p * vPtr[d]
                            }
                            mMax = newMax
                        }
                        let invL = lSum > 0 ? 1.0 / lSum : 0
                        for d in 0..<hd {
                            attnBuf[qOff + d] = acc[d] * invL
                        }
                    }
                }
            }

            var projAttn = [Float](repeating: 0, count: H)
            matvecQ4_0(tensor: l.wO, x: attnBuf, into: &projAttn, K: qDim, N: H)
            rmsNormInPlace(&projAttn, offset: 0, count: H, weight: l.postAttnNorm, weightScale: 1.0, eps: config.rmsNormEps)
            for i in 0..<H { hidden[i] += projAttn[i] }

            // FFN
            rmsNormCopy(hidden, weight: l.ffnNorm, into: &normBuf, count: H, eps: config.rmsNormEps)
            var gate = [Float](repeating: 0, count: interDim)
            var up = [Float](repeating: 0, count: interDim)
            matvecQ4_0(tensor: l.ffnGate, x: normBuf, into: &gate, K: H, N: interDim)
            matvecQ4_0(tensor: l.ffnUp, x: normBuf, into: &up, K: H, N: interDim)
            for i in 0..<interDim {
                let x = gate[i]
                let inner = min(15.0, max(-15.0, 0.7978845608 * (x + 0.044715 * x * x * x)))
                gate[i] = 0.5 * x * (1.0 + tanhf(inner)) * up[i]
            }
            var ffnDown = [Float](repeating: 0, count: H)
            matvecQ4_0(tensor: l.ffnDown, x: gate, into: &ffnDown, K: interDim, N: H)
            rmsNormInPlace(&ffnDown, offset: 0, count: H, weight: l.postFFWNorm, weightScale: 1.0, eps: config.rmsNormEps)
            for i in 0..<H { hidden[i] += ffnDown[i] }

            // PLE
            var peGate = [Float](repeating: 0, count: plDim)
            matvecQ4_0(tensor: l.inpGate, x: hidden, into: &peGate, K: H, N: plDim)
            let plOff = il * plDim
            for i in 0..<plDim {
                let x = peGate[i]
                let inner = min(15.0, max(-15.0, 0.7978845608 * (x + 0.044715 * x * x * x)))
                peGate[i] = 0.5 * x * (1.0 + tanhf(inner)) * pleProj[plOff + i]
            }
            var peProj = [Float](repeating: 0, count: H)
            matvecQ4_0(tensor: l.proj, x: peGate, into: &peProj, K: plDim, N: H)
            rmsNormInPlace(&peProj, offset: 0, count: H, weight: l.postNorm, weightScale: 1.0, eps: config.rmsNormEps)
            let s = l.outScale
            for i in 0..<H { hidden[i] = (hidden[i] + peProj[i]) * s }
        }

        guard computeLogits else { return 0 }

        rmsNormInPlace(&hidden, offset: 0, count: H, weight: outputNorm, weightScale: 1.0, eps: config.rmsNormEps)

        // Tied Q6_K LM Head parallel argmax
        let V = config.vocabSize
        let numChunks = max(1, ProcessInfo.processInfo.activeProcessorCount)
        let chunkSize = (V + numChunks - 1) / numChunks
        var bestIDs = [Int32](repeating: 0, count: numChunks)
        var bestVals = [Float](repeating: -1e30, count: numChunks)
        let rowBytes = tokenEmbdInfo.type.rowByteSize(elements: H)
        let tokBase = gguf.pointer(for: tokenEmbdInfo)

        hidden.withUnsafeBufferPointer { hBuf in
            let xPtr = hBuf.baseAddress!
            bestIDs.withUnsafeMutableBufferPointer { idsBuf in
                bestVals.withUnsafeMutableBufferPointer { valsBuf in
                    DispatchQueue.concurrentPerform(iterations: numChunks) { cid in
                        let start = cid * chunkSize
                        let end = min(V, start + chunkSize)
                        var rowTemp = [Float](repeating: 0, count: H)
                        var localBestVal: Float = -1e30
                        var localBestID: Int32 = 0
                        let softcap = config.finalLogitSoftcapping

                        for row in start..<end {
                            rowTemp.withUnsafeMutableBufferPointer { rt in
                                GGUFQuantization.dequantizeQ6KRow(src: tokBase.advanced(by: row * rowBytes), dst: rt.baseAddress!, elements: H)
                                var d: Float = 0
                                vDSP_dotpr(rt.baseAddress!, 1, xPtr, 1, &d, vDSP_Length(H))
                                if softcap > 0 {
                                    d = softcap * tanhf(min(15.0, max(-15.0, d / softcap)))
                                }
                                if d > localBestVal {
                                    localBestVal = d
                                    localBestID = Int32(row)
                                }
                            }
                        }
                        idsBuf[cid] = localBestID
                        valsBuf[cid] = localBestVal
                    }
                }
            }
        }

        var globalBestVal: Float = -1e30
        var globalBestID: Int32 = 0
        for c in 0..<numChunks where bestVals[c] > globalBestVal {
            globalBestVal = bestVals[c]
            globalBestID = bestIDs[c]
        }
        return globalBestID
    }

    // MARK: - Math Helpers

    private func rmsNormCopy(_ src: [Float], weight: [Float], into dst: inout [Float], count: Int, eps: Float) {
        var ss: Float = 0
        src.withUnsafeBufferPointer { sb in
            vDSP_svesq(sb.baseAddress!, 1, &ss, vDSP_Length(count))
        }
        let inv = 1.0 / sqrtf(ss / Float(count) + eps)
        for i in 0..<count {
            dst[i] = src[i] * inv * weight[i]
        }
    }

    private func rmsNormInPlace(_ vec: inout [Float], offset: Int, count: Int, weight: [Float]?, weightScale: Float, eps: Float) {
        var ss: Float = 0
        vec.withUnsafeBufferPointer { vb in
            vDSP_svesq(vb.baseAddress!.advanced(by: offset), 1, &ss, vDSP_Length(count))
        }
        let inv = (1.0 / sqrtf(ss / Float(count) + eps)) * weightScale
        if let w = weight {
            for i in 0..<count { vec[offset + i] *= inv * w[i] }
        } else {
            for i in 0..<count { vec[offset + i] *= inv }
        }
    }

    private func applyRoPE(_ vec: inout [Float], offset: Int, headDim: Int, pos: Int, base: Float, useFreqs: Bool) {
        let half = headDim / 2
        for j in 0..<half {
            let ff = useFreqs ? ropeFreqs[j] : 1.0
            let angle = Float(pos) * powf(base, -2.0 * Float(j) / Float(headDim)) / ff
            let c = cosf(angle)
            let s = sinf(angle)
            let x0 = vec[offset + j]
            let x1 = vec[offset + j + half]
            vec[offset + j] = x0 * c - x1 * s
            vec[offset + j + half] = x0 * s + x1 * c
        }
    }

    private func matvecQ4_0(tensor: GGUFTensorInfo, x: [Float], into y: inout [Float], K: Int, N: Int) {
        let base = gguf.pointer(for: tensor)
        let blocksPerRow = K / 32
        let rowBytes = blocksPerRow * 18
        let rowsPerGroup = 16
        let numGroups = (N + rowsPerGroup - 1) / rowsPerGroup

        x.withUnsafeBufferPointer { xBuf in
            let xPtr = xBuf.baseAddress!
            y.withUnsafeMutableBufferPointer { yBuf in
                let yPtr = yBuf.baseAddress!
                DispatchQueue.concurrentPerform(iterations: numGroups) { g in
                    let rStart = g * rowsPerGroup
                    let rEnd = min(N, rStart + rowsPerGroup)
                    for row in rStart..<rEnd {
                        let rPtr = base.advanced(by: row * rowBytes)
                        var sum: Float = 0
                        for b in 0..<blocksPerRow {
                            let blk = rPtr.advanced(by: b * 18)
                            let d = Float(Float16(bitPattern: blk.loadUnaligned(as: UInt16.self)))
                            let qs = blk.advanced(by: 2).assumingMemoryBound(to: UInt8.self)
                            let xBlk = xPtr.advanced(by: b * 32)
                            var bSum: Float = 0
                            for j in 0..<16 {
                                let q = qs[j]
                                let w0 = Float(Int32(q & 0x0F) - 8)
                                let w1 = Float(Int32(q >> 4) - 8)
                                bSum += w0 * xBlk[j] + w1 * xBlk[j + 16]
                            }
                            sum += bSum * d
                        }
                        yPtr[row] = sum
                    }
                }
            }
        }
    }
}
