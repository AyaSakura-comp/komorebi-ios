import Foundation
import Metal
import Darwin

public enum GemmaEngineError: LocalizedError {
    case metalUnavailable
    case shaderLibraryNotFound
    case shaderFunctionMissing(String)
    case bufferAllocationFailed(String)
    case inferenceFailed(String)

    public var errorDescription: String? {
        switch self {
        case .metalUnavailable: return "Metal device is unavailable on this device."
        case .shaderLibraryNotFound: return "Could not load Gemma4 Metal shader library."
        case .shaderFunctionMissing(let n): return "Missing Metal compute function: \(n)"
        case .bufferAllocationFailed(let n): return "Metal buffer allocation failed: \(n)"
        case .inferenceFailed(let n): return "Inference error: \(n)"
        }
    }
}

public struct GenerationStats: Sendable {
    public let backend: String
    public let prefixCachedTokens: Int
    public let prefillTokens: Int
    public let prefillSeconds: Double
    public let decodeTokens: Int
    public let decodeSeconds: Double

    public var prefillTPS: Double {
        prefillSeconds > 0 ? Double(prefillTokens) / prefillSeconds : 0
    }

    public var decodeTPS: Double {
        decodeSeconds > 0 ? Double(decodeTokens) / decodeSeconds : 0
    }
}

public struct GenerationResult: Sendable {
    public let text: String
    public let tokenIDs: [Int32]
    public let stats: GenerationStats
}

/// Pure Metal compute engine for Gemma 4 E2B (Q4_0 weights + Q6_K tied LM head + F16 PLE proj).
/// Includes:
///  - Zero-fault `mmap` Q6_K `per_layer_token_embd` row lookup (~7KB per token vs 1.93GB table)
///  - Multi-row SIMD-group Q4_0 matvec (`M = 1`) and batched Q4_0 GEMM (`M > 1` prefill)
///  - KV-shared pruning (`layers 15..34` run only on last token during prefill)
///  - System-prompt prefix KV snapshot cache for instant repeat dictations
public final class Gemma4MetalEngine: @unchecked Sendable {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let gguf: GGUFFile
    public let config: Gemma4Config
    public let maxContext: Int

    // Compute pipelines
    private let psoQ4Matvec: MTLComputePipelineState
    private let psoQ4Matmul: MTLComputePipelineState
    private let psoQ4MMSimd: MTLComputePipelineState
    private let psoF16MMSimd: MTLComputePipelineState
    private let psoQ6Matvec: MTLComputePipelineState
    private let psoF16Matmul: MTLComputePipelineState
    private let psoRMSNorm: MTLComputePipelineState
    private let psoPLECombine: MTLComputePipelineState
    private let psoRoPE: MTLComputePipelineState
    private let psoWriteKV: MTLComputePipelineState
    private let psoAttention: MTLComputePipelineState
    private let psoGeGLU: MTLComputePipelineState
    private let psoResAddScale: MTLComputePipelineState
    private let psoArgmax1: MTLComputePipelineState
    private let psoArgmax2: MTLComputePipelineState

    // Weight backing storage
    private struct TensorRef {
        let buffer: MTLBuffer
        let offset: Int
        let rows: Int
        let cols: Int
    }

    private var mmapZeroCopyBuffer: MTLBuffer?
    private var mmapBaseOffsetInFile: Int = 0
    private var ownedBuffers: [MTLBuffer] = []

    private let tokenEmbdInfo: GGUFTensorInfo
    private let perLayerTokEmbdInfo: GGUFTensorInfo

    private let outputNorm: TensorRef
    private let perLayerModelProj: TensorRef
    private let perLayerProjNorm: TensorRef
    private let ropeFreqs: TensorRef
    private let lmHeadRef: TensorRef

    private struct LayerWeights {
        let attnNorm: TensorRef
        let wQ: TensorRef
        let attnQNorm: TensorRef
        let wK: TensorRef?
        let wV: TensorRef?
        let attnKNorm: TensorRef?
        let wO: TensorRef
        let postAttnNorm: TensorRef
        let ffnNorm: TensorRef
        let ffnGate: TensorRef
        let ffnUp: TensorRef
        let ffnDown: TensorRef
        let postFFWNorm: TensorRef
        let inpGate: TensorRef
        let proj: TensorRef
        let postNorm: TensorRef
        let outScale: Float
    }
    private let layerWeights: [LayerWeights]

    // Activation scratch buffers (sized for prefill batches up to `maxBatch`)
    private let maxBatch = 64
    private let scratchHiddenA: MTLBuffer
    private let scratchHiddenB: MTLBuffer
    private let scratchNorm: MTLBuffer
    private let scratchQ: MTLBuffer
    private let scratchK: MTLBuffer
    private let scratchV: MTLBuffer
    private let scratchAttnOut: MTLBuffer
    private let scratchFFNGate: MTLBuffer
    private let scratchFFNUp: MTLBuffer
    private let scratchFFNDown: MTLBuffer
    private let scratchPLEProj: MTLBuffer
    private let scratchPLERaw: MTLBuffer
    private let scratchPLECombined: MTLBuffer
    private let scratchPEGate: MTLBuffer
    private let scratchPEProj: MTLBuffer
    private let scratchLogits: MTLBuffer
    private let scratchArgmaxVals: MTLBuffer
    private let scratchArgmaxIDs: MTLBuffer
    private let scratchBestToken: MTLBuffer

    // FP16 KV caches for own-KV layers (`0..<numOwnKVLayers`)
    private var kCaches: [MTLBuffer] = []
    private var vCaches: [MTLBuffer] = []

    // Prefix KV cache snapshot to skip re-prefilling identical system instructions across utterances
    private struct PrefixSnapshot {
        let tokens: [Int32]
        let kSnapshots: [Data]
        let vSnapshots: [Data]
    }
    private var cachedPrefix: PrefixSnapshot?

    // Command recording state (per forwardChunk). In profiling mode every dispatch is committed separately
    // so its GPU time (gpuEndTime - gpuStartTime) can be attributed to a kernel label.
    private var curCmd: MTLCommandBuffer?
    private var curEnc: MTLComputeCommandEncoder?
    private var enc: MTLComputeCommandEncoder { curEnc! }

    /// Opt-in per-kernel GPU profiling (slow: one command buffer per dispatch). Also enabled by env `KOMOREBI_METAL_PROFILE=1`.
    public var profilingEnabled = ProcessInfo.processInfo.environment["KOMOREBI_METAL_PROFILE"] == "1"
    public private(set) var profileGPUSeconds: [String: Double] = [:]
    public private(set) var profileCounts: [String: Int] = [:]
    public private(set) var profileCPUEmbedSeconds: Double = 0

    public func resetProfile() {
        profileGPUSeconds = [:]
        profileCounts = [:]
        profileCPUEmbedSeconds = 0
    }

    public func profileReport() -> String {
        let total = profileGPUSeconds.values.reduce(0, +)
        var lines = [String(format: "GPU total %.1f ms | CPU embed %.1f ms", total * 1000, profileCPUEmbedSeconds * 1000)]
        for (k, v) in profileGPUSeconds.sorted(by: { $0.value > $1.value }) {
            lines.append(String(format: "  %-14@ %8.2f ms  %5.1f%%  (%d dispatches)", k as NSString, v * 1000, total > 0 ? v / total * 100 : 0, profileCounts[k] ?? 0))
        }
        return lines.joined(separator: "\n")
    }

    @inline(__always)
    private func prof(_ label: String) {
        guard profilingEnabled, let e = curEnc, let c = curCmd else { return }
        e.endEncoding()
        c.commit()
        c.waitUntilCompleted()
        profileGPUSeconds[label, default: 0] += c.gpuEndTime - c.gpuStartTime
        profileCounts[label, default: 0] += 1
        curCmd = commandQueue.makeCommandBuffer()
        curEnc = curCmd?.makeComputeCommandEncoder()
    }

    private let argmaxStagesGroups = 256
    private let embdScale: Float
    private let perLayerEmbdScale: Float
    private let invSqrt2: Float = 1.0 / sqrtf(2.0)
    private let projNormEps: Float

    public init(gguf: GGUFFile, maxContext: Int = 1024, device: MTLDevice? = nil) throws {
        guard let dev = device ?? MTLCreateSystemDefaultDevice() else {
            throw GemmaEngineError.metalUnavailable
        }
        guard let cq = dev.makeCommandQueue() else {
            throw GemmaEngineError.metalUnavailable
        }
        self.device = dev
        self.commandQueue = cq
        self.gguf = gguf
        let cfg = Gemma4Config.from(gguf: gguf.metadata)
        self.config = cfg
        self.maxContext = min(maxContext, 2048)

        self.embdScale = sqrtf(Float(cfg.hiddenSize))
        self.perLayerEmbdScale = sqrtf(Float(cfg.perLayerInputDim)) / sqrtf(2.0)
        self.projNormEps = cfg.rmsNormEps * Float(cfg.hiddenSize)

        let lib = try Self.loadMetalLibrary(device: dev)
        func makePSO(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = lib.makeFunction(name: name) else {
                throw GemmaEngineError.shaderFunctionMissing(name)
            }
            return try dev.makeComputePipelineState(function: fn)
        }

        self.psoQ4Matvec = try makePSO("q4_0_matvec_f32")
        self.psoQ4Matmul = try makePSO("q4_0_matmul_f32")
        self.psoQ4MMSimd = try makePSO("q4_0_mm_simd_f32")
        self.psoF16MMSimd = try makePSO("f16_mm_simd_f32")
        self.psoQ6Matvec = try makePSO("q6_k_matvec_f32")
        self.psoF16Matmul = try makePSO("f16_matmul_f32")
        self.psoRMSNorm = try makePSO("rmsnorm_f32")
        self.psoPLECombine = try makePSO("ple_project_combine_f32")
        self.psoRoPE = try makePSO("rope_neox_f32")
        self.psoWriteKV = try makePSO("write_kv_cache_f16")
        self.psoAttention = try makePSO("mqa_attention_f32")
        self.psoGeGLU = try makePSO("geglu_tanh_f32")
        self.psoResAddScale = try makePSO("residual_add_scale_f32")
        self.psoArgmax1 = try makePSO("softcap_argmax_stage1_f32")
        self.psoArgmax2 = try makePSO("softcap_argmax_stage2_f32")

        self.tokenEmbdInfo = try gguf.tensor("token_embd.weight")
        self.perLayerTokEmbdInfo = try gguf.tensor("per_layer_token_embd.weight")

        // Try zero-copy mapping starting at page-aligned floor of `token_embd.weight` (~1.39 GB contiguous).
        let pageSize = Int(getpagesize())
        let tokenFileStart = gguf.tensorDataOffset + Int(tokenEmbdInfo.offset)
        let alignedStart = (tokenFileStart / pageSize) * pageSize
        let mapLength = ((gguf.fileSize - alignedStart + pageSize - 1) / pageSize) * pageSize
        if mapLength <= dev.maxBufferLength,
           alignedStart + (gguf.fileSize - alignedStart) <= gguf.fileSize {
            let ptr = UnsafeMutableRawPointer(mutating: gguf.basePointer.advanced(by: alignedStart))
            if let zc = dev.makeBuffer(
                bytesNoCopy: ptr,
                length: gguf.fileSize - alignedStart,
                options: .storageModeShared,
                deallocator: nil
            ) {
                self.mmapZeroCopyBuffer = zc
                self.mmapBaseOffsetInFile = alignedStart
            }
        }

        var localOwned: [MTLBuffer] = []
        let mmapRef = self.mmapZeroCopyBuffer
        let mmapFileBase = self.mmapBaseOffsetInFile

        func bindTensor(_ name: String) throws -> TensorRef {
            let info = try gguf.tensor(name)
            let cols = info.dimensions[0]
            let rows = info.dimensions.count > 1 ? info.dimensions[1] : 1
            let fileOffset = gguf.tensorDataOffset + Int(info.offset)
            let bytes = info.byteSize

            if let zc = mmapRef, fileOffset >= mmapFileBase, fileOffset + bytes <= gguf.fileSize {
                return TensorRef(buffer: zc, offset: fileOffset - mmapFileBase, rows: rows, cols: cols)
            }
            let srcPtr = gguf.pointer(for: info)
            guard let buf = dev.makeBuffer(bytes: srcPtr, length: max(bytes, 16), options: .storageModeShared) else {
                throw GemmaEngineError.bufferAllocationFailed(name)
            }
            localOwned.append(buf)
            return TensorRef(buffer: buf, offset: 0, rows: rows, cols: cols)
        }

        self.outputNorm = try bindTensor("output_norm.weight")
        self.perLayerModelProj = try bindTensor("per_layer_model_proj.weight")
        self.perLayerProjNorm = try bindTensor("per_layer_proj_norm.weight")
        self.ropeFreqs = try bindTensor("rope_freqs.weight")
        self.lmHeadRef = try bindTensor("token_embd.weight")

        var layers: [LayerWeights] = []
        layers.reserveCapacity(cfg.numLayers)
        for il in 0..<cfg.numLayers {
            let outScaleArr = try? gguf.readFloat32Tensor("blk.\(il).layer_output_scale.weight")
            let outScale = outScaleArr?.first ?? 1.0
            let hasKV = cfg.hasOwnKV(layer: il)

            layers.append(LayerWeights(
                attnNorm: try bindTensor("blk.\(il).attn_norm.weight"),
                wQ: try bindTensor("blk.\(il).attn_q.weight"),
                attnQNorm: try bindTensor("blk.\(il).attn_q_norm.weight"),
                wK: hasKV ? try bindTensor("blk.\(il).attn_k.weight") : nil,
                wV: hasKV ? try bindTensor("blk.\(il).attn_v.weight") : nil,
                attnKNorm: hasKV ? try bindTensor("blk.\(il).attn_k_norm.weight") : nil,
                wO: try bindTensor("blk.\(il).attn_output.weight"),
                postAttnNorm: try bindTensor("blk.\(il).post_attention_norm.weight"),
                ffnNorm: try bindTensor("blk.\(il).ffn_norm.weight"),
                ffnGate: try bindTensor("blk.\(il).ffn_gate.weight"),
                ffnUp: try bindTensor("blk.\(il).ffn_up.weight"),
                ffnDown: try bindTensor("blk.\(il).ffn_down.weight"),
                postFFWNorm: try bindTensor("blk.\(il).post_ffw_norm.weight"),
                inpGate: try bindTensor("blk.\(il).inp_gate.weight"),
                proj: try bindTensor("blk.\(il).proj.weight"),
                postNorm: try bindTensor("blk.\(il).post_norm.weight"),
                outScale: outScale
            ))
        }
        self.layerWeights = layers
        self.ownedBuffers = localOwned

        func allocShared(_ bytes: Int, _ label: String) throws -> MTLBuffer {
            guard let buf = dev.makeBuffer(length: max(bytes, 64), options: .storageModeShared) else {
                throw GemmaEngineError.bufferAllocationFailed(label)
            }
            return buf
        }

        let f32 = MemoryLayout<Float>.stride
        let H = cfg.hiddenSize
        let maxQDim = cfg.numAttentionHeads * max(cfg.slidingHeadDim, cfg.fullHeadDim)
        let maxKVDim = max(cfg.slidingHeadDim, cfg.fullHeadDim)
        let maxFFN = max(cfg.intermediateSizeBase, cfg.sharedLayerIntermediateSize)
        let plTotal = cfg.numLayers * cfg.perLayerInputDim

        self.scratchHiddenA = try allocShared(maxBatch * H * f32, "hiddenA")
        self.scratchHiddenB = try allocShared(maxBatch * H * f32, "hiddenB")
        self.scratchNorm = try allocShared(maxBatch * H * f32, "norm")
        self.scratchQ = try allocShared(maxBatch * maxQDim * f32, "Q")
        self.scratchK = try allocShared(maxBatch * maxKVDim * f32, "K")
        self.scratchV = try allocShared(maxBatch * maxKVDim * f32, "V")
        self.scratchAttnOut = try allocShared(maxBatch * maxQDim * f32, "attnOut")
        self.scratchFFNGate = try allocShared(maxBatch * maxFFN * f32, "ffnGate")
        self.scratchFFNUp = try allocShared(maxBatch * maxFFN * f32, "ffnUp")
        self.scratchFFNDown = try allocShared(maxBatch * H * f32, "ffnDown")
        self.scratchPLEProj = try allocShared(maxBatch * plTotal * f32, "pleProj")
        self.scratchPLERaw = try allocShared(maxBatch * plTotal * f32, "pleRaw")
        self.scratchPLECombined = try allocShared(maxBatch * plTotal * f32, "pleCombined")
        self.scratchPEGate = try allocShared(maxBatch * cfg.perLayerInputDim * f32, "peGate")
        self.scratchPEProj = try allocShared(maxBatch * H * f32, "peProj")
        self.scratchLogits = try allocShared(cfg.vocabSize * f32, "logits")
        self.scratchArgmaxVals = try allocShared(argmaxStagesGroups * f32, "argmaxVals")
        self.scratchArgmaxIDs = try allocShared(argmaxStagesGroups * MemoryLayout<Int32>.stride, "argmaxIDs")
        self.scratchBestToken = try allocShared(MemoryLayout<Int32>.stride, "bestToken")

        var ks: [MTLBuffer] = []
        var vs: [MTLBuffer] = []
        for il in 0..<cfg.numOwnKVLayers {
            let slots = cfg.isSlidingLayer[il] ? cfg.slidingWindow : self.maxContext
            let hd = cfg.headDim(layer: il)
            let bytes = slots * hd * MemoryLayout<UInt16>.stride
            ks.append(try allocShared(bytes, "kCache_\(il)"))
            vs.append(try allocShared(bytes, "vCache_\(il)"))
        }
        self.kCaches = ks
        self.vCaches = vs
    }

    public var isZeroCopyWeightsActive: Bool {
        mmapZeroCopyBuffer != nil
    }

    public func clearPrefixCache() {
        cachedPrefix = nil
    }

    /// Generates tokens greedily with real Metal prefill/decode instrumentation and automatic LCP prefix KV reuse.
    public func generate(
        inputIDs: [Int32],
        maxNewTokens: Int = 256,
        stopTokenIDs: Set<Int32>,
        staticPrefixLength: Int = 0,
        shouldCancel: (() -> Bool)? = nil,
        onToken: ((Int32) -> Void)? = nil
    ) throws -> (tokenIDs: [Int32], stats: GenerationStats) {
        guard !inputIDs.isEmpty else {
            return ([], GenerationStats(backend: "Metal Q4_0", prefixCachedTokens: 0, prefillTokens: 0, prefillSeconds: 0, decodeTokens: 0, decodeSeconds: 0))
        }
        guard inputIDs.count < maxContext else {
            throw GemmaEngineError.inferenceFailed("Prompt length \(inputIDs.count) exceeds maxContext \(maxContext)")
        }

        // Check if static system prefix KV snapshot matches the beginning of `inputIDs`.
        var startPos = 0
        if staticPrefixLength > 0,
           staticPrefixLength < inputIDs.count,
           let snap = cachedPrefix,
           snap.tokens.count == staticPrefixLength,
           Array(inputIDs.prefix(staticPrefixLength)) == snap.tokens {
            restoreKVSnapshot(snap)
            startPos = staticPrefixLength
        }

        let tPrefillStart = CFAbsoluteTimeGetCurrent()
        var pos = startPos
        var nextToken: Int32 = 0

        // If `staticPrefixLength > 0` and cache missed, prefill `0..<staticPrefixLength` and snapshot KV.
        if startPos == 0 && staticPrefixLength > 8 && staticPrefixLength < inputIDs.count {
            let prefixTokens = Array(inputIDs[0..<staticPrefixLength])
            _ = try prefillSequence(tokens: prefixTokens, startPos: 0, computeLogitsOnLast: false)
            cachedPrefix = captureKVSnapshot(tokens: prefixTokens)
            pos = staticPrefixLength
        }

        let remainingPrompt = Array(inputIDs[pos...])
        if !remainingPrompt.isEmpty {
            nextToken = try prefillSequence(tokens: remainingPrompt, startPos: pos, computeLogitsOnLast: true)
            pos += remainingPrompt.count
        }
        let tPrefillEnd = CFAbsoluteTimeGetCurrent()

        var generated: [Int32] = []
        generated.reserveCapacity(min(maxNewTokens, 256))

        let tDecodeStart = CFAbsoluteTimeGetCurrent()
        if maxNewTokens > 0 && !stopTokenIDs.contains(nextToken) {
            generated.append(nextToken)
            onToken?(nextToken)

            while generated.count < maxNewTokens && pos < maxContext - 1 {
                if shouldCancel?() == true { break }
                let tok = try decodeStep(token: generated.last!, pos: pos)
                pos += 1
                if stopTokenIDs.contains(tok) { break }
                generated.append(tok)
                onToken?(tok)
            }
        }
        let tDecodeEnd = CFAbsoluteTimeGetCurrent()

        let prefillCount = inputIDs.count - startPos
        let stats = GenerationStats(
            backend: isZeroCopyWeightsActive ? "Metal Q4_0 (mmap zero-copy)" : "Metal Q4_0",
            prefixCachedTokens: startPos,
            prefillTokens: prefillCount,
            prefillSeconds: tPrefillEnd - tPrefillStart,
            decodeTokens: generated.count,
            decodeSeconds: max(tDecodeEnd - tDecodeStart, 1e-6)
        )
        return (generated, stats)
    }

    // MARK: - Prefill & Decode Forward Passes

    private func prefillSequence(tokens: [Int32], startPos: Int, computeLogitsOnLast: Bool) throws -> Int32 {
        var idx = 0
        var lastPred: Int32 = 0
        while idx < tokens.count {
            let chunkLen = min(maxBatch, tokens.count - idx)
            let isLastChunk = (idx + chunkLen == tokens.count)
            let slice = Array(tokens[idx..<(idx + chunkLen)])
            lastPred = try forwardChunk(
                tokens: slice,
                posOffset: startPos + idx,
                computeLogitsOnLast: computeLogitsOnLast && isLastChunk
            )
            idx += chunkLen
        }
        return lastPred
    }

    private func decodeStep(token: Int32, pos: Int) throws -> Int32 {
        try forwardChunk(tokens: [token], posOffset: pos, computeLogitsOnLast: true)
    }

    private func forwardChunk(tokens: [Int32], posOffset: Int, computeLogitsOnLast: Bool) throws -> Int32 {
        let M = tokens.count
        let H = config.hiddenSize
        let plDim = config.perLayerInputDim
        let nL = config.numLayers
        let plRowTotal = nL * plDim

        // 1. CPU zero-fault mmap row lookup for `token_embd` & `per_layer_token_embd` directly into shared MTLBuffers.
        let tEmbed0 = CFAbsoluteTimeGetCurrent()
        let hidPtr = scratchHiddenA.contents().bindMemory(to: Float.self, capacity: M * H)
        let pleRawPtr = scratchPLERaw.contents().bindMemory(to: Float.self, capacity: M * plRowTotal)

        for m in 0..<M {
            let tid = max(0, min(config.vocabSize - 1, Int(tokens[m])))
            let hDst = hidPtr.advanced(by: m * H)
            gguf.dequantizeQ6KRow(tensor: tokenEmbdInfo, rowIndex: tid, into: hDst, count: H)
            for j in 0..<H { hDst[j] *= embdScale }

            let pDst = pleRawPtr.advanced(by: m * plRowTotal)
            gguf.dequantizeQ6KRow(tensor: perLayerTokEmbdInfo, rowIndex: tid, into: pDst, count: plRowTotal)
            for j in 0..<plRowTotal { pDst[j] *= perLayerEmbdScale }
        }

        if profilingEnabled { profileCPUEmbedSeconds += CFAbsoluteTimeGetCurrent() - tEmbed0 }

        curCmd = commandQueue.makeCommandBuffer()
        curEnc = curCmd?.makeComputeCommandEncoder()
        guard curEnc != nil else {
            throw GemmaEngineError.inferenceFailed("Unable to create Metal command encoder")
        }

        // 2. Per-layer projection & PLE combine -> `scratchPLECombined` shaped [nLayers, M, 256]
        encodeF16Matmul(enc, w: perLayerModelProj, x: scratchHiddenA, y: scratchPLEProj, M: M)
        encodePLECombine(enc, proj: scratchPLEProj, rawScaled: scratchPLERaw, out: scratchPLECombined, M: M)

        // 3. Transformer decoder layers `0 ..< numLayers`
        var activeHidden = scratchHiddenA
        var altHidden = scratchHiddenB

        // KV-sharing pruning: layers `numOwnKVLayers..<nL` never write KV, so their output only matters for
        // the row whose logits we need. Rows without logits stop after writing K/V in the last own-KV layer;
        // when logits are needed, only row `M - 1` continues from that layer's Q projection onward.
        let nOwn = config.numOwnKVLayers
        let pruneShared = nOwn < nL && (!computeLogitsOnLast || M > 1)
        var activeM = M
        var hiddenByteOffset = 0
        var normByteOffset = 0
        var pleTokenIndex = 0
        var activePosOffset = posOffset

        layerLoop: for il in 0..<nL {

            let lw = layerWeights[il]
            let isSWA = config.isSlidingLayer[il]
            let hd = config.headDim(layer: il)
            let qDim = config.numAttentionHeads * hd
            let interDim = config.intermediateSize(layer: il)
            let kvLayer = config.kvSourceLayer[il]
            let winSize: UInt32 = isSWA ? UInt32(config.slidingWindow) : 0
            let freqBase: Float = isSWA ? config.ropeFreqBaseSWA : config.ropeFreqBaseFull

            // 3a. Attn RMSNorm
            encodeRMSNorm(enc, x: activeHidden, xOff: hiddenByteOffset, w: lw.attnNorm, y: scratchNorm, yOff: 0, rows: activeM, dim: H, eps: config.rmsNormEps, useWeight: true)

            // 3b. K, V projections + norms + RoPE + KV Cache write (only layers 0..<nOwn)
            if let wK = lw.wK, let wV = lw.wV, let kNorm = lw.attnKNorm {
                encodeQ4(enc, w: wK, x: scratchNorm, y: scratchK, M: activeM, label: "k")
                encodeQ4(enc, w: wV, x: scratchNorm, y: scratchV, M: activeM, label: "v")
                encodeRMSNorm(enc, x: scratchK, xOff: 0, w: kNorm, y: scratchK, yOff: 0, rows: activeM * config.numKeyValueHeads, dim: hd, eps: config.rmsNormEps, useWeight: true)
                encodeRMSNorm(enc, x: scratchV, xOff: 0, w: nil, y: scratchV, yOff: 0, rows: activeM * config.numKeyValueHeads, dim: hd, eps: config.rmsNormEps, useWeight: false)
                encodeRoPE(enc, qk: scratchK, hd: hd, nHeads: config.numKeyValueHeads, M: activeM, posOff: activePosOffset, freqBase: freqBase, useFreqs: !isSWA)
                encodeWriteKV(enc, k: scratchK, v: scratchV, kCache: kCaches[kvLayer], vCache: vCaches[kvLayer], hd: hd, posOff: activePosOffset, winSize: winSize, M: activeM)
            }

            // Last own-KV layer: everything after the KV write only feeds KV-shared layers.
            if pruneShared && il == nOwn - 1 {
                if !computeLogitsOnLast { break layerLoop }
                if activeM > 1 {
                    let lastRow = (activeM - 1) * H * MemoryLayout<Float>.stride
                    hiddenByteOffset = lastRow
                    normByteOffset = lastRow
                    pleTokenIndex = M - 1
                    activePosOffset = posOffset + (M - 1)
                    activeM = 1
                }
            }

            // 3c. Q projection + Q norm + RoPE
            encodeQ4(enc, w: lw.wQ, x: scratchNorm, xOff: normByteOffset, y: scratchQ, M: activeM, label: "q")
            normByteOffset = 0
            encodeRMSNorm(enc, x: scratchQ, xOff: 0, w: lw.attnQNorm, y: scratchQ, yOff: 0, rows: activeM * config.numAttentionHeads, dim: hd, eps: config.rmsNormEps, useWeight: true)
            encodeRoPE(enc, qk: scratchQ, hd: hd, nHeads: config.numAttentionHeads, M: activeM, posOff: activePosOffset, freqBase: freqBase, useFreqs: !isSWA)

            // 3d. MQA Attention (`scale = 1.0`, sliding ring or full causal)
            encodeAttention(enc, q: scratchQ, kCache: kCaches[kvLayer], vCache: vCaches[kvLayer], out: scratchAttnOut, hd: hd, posOff: activePosOffset, winSize: winSize, M: activeM)

            // 3e. Output proj + post_attention_norm + residual
            encodeQ4(enc, w: lw.wO, x: scratchAttnOut, y: scratchNorm, M: activeM, K: qDim, N: H, label: "o")
            encodeRMSNorm(enc, x: scratchNorm, xOff: 0, w: lw.postAttnNorm, y: scratchNorm, yOff: 0, rows: activeM, dim: H, eps: config.rmsNormEps, useWeight: true)
            encodeResAddScale(enc, a: activeHidden, aOff: hiddenByteOffset, b: scratchNorm, bOff: 0, out: altHidden, outOff: 0, scale: 1.0, count: activeM * H)
            // After writing `altHidden` at offset 0, next ops see offset 0!
            hiddenByteOffset = 0
            swap(&activeHidden, &altHidden)

            // 3f. FFN block: rms_norm -> gate/up -> GeGLU -> down -> post_ffw_norm + residual
            encodeRMSNorm(enc, x: activeHidden, xOff: 0, w: lw.ffnNorm, y: scratchNorm, yOff: 0, rows: activeM, dim: H, eps: config.rmsNormEps, useWeight: true)
            encodeQ4(enc, w: lw.ffnGate, x: scratchNorm, y: scratchFFNGate, M: activeM, K: H, N: interDim, label: "gate")
            encodeQ4(enc, w: lw.ffnUp, x: scratchNorm, y: scratchFFNUp, M: activeM, K: H, N: interDim, label: "up")
            encodeGeGLU(enc, gate: scratchFFNGate, gateOff: 0, up: scratchFFNUp, upOff: 0, count: activeM * interDim)
            encodeQ4(enc, w: lw.ffnDown, x: scratchFFNGate, y: scratchFFNDown, M: activeM, K: interDim, N: H, label: "down")
            encodeRMSNorm(enc, x: scratchFFNDown, xOff: 0, w: lw.postFFWNorm, y: scratchNorm, yOff: 0, rows: activeM, dim: H, eps: config.rmsNormEps, useWeight: true)
            encodeResAddScale(enc, a: activeHidden, aOff: 0, b: scratchNorm, bOff: 0, out: altHidden, outOff: 0, scale: 1.0, count: activeM * H)
            swap(&activeHidden, &altHidden)

            // 3g. Per-Layer Embedding (PLE) residual + `layer_output_scale`
            // `scratchPLECombined` is `[nLayers, M, 256]`. Slice `(il * M + pleTokenIndex) * 256`.
            let pleByteOff = (il * M + pleTokenIndex) * plDim * MemoryLayout<Float>.stride
            encodeQ4(enc, w: lw.inpGate, x: activeHidden, y: scratchPEGate, M: activeM, K: H, N: plDim, label: "ple_gate")
            encodeGeGLU(enc, gate: scratchPEGate, gateOff: 0, up: scratchPLECombined, upOff: pleByteOff, count: activeM * plDim)
            encodeQ4(enc, w: lw.proj, x: scratchPEGate, y: scratchPEProj, M: activeM, K: plDim, N: H, label: "ple_proj")
            encodeRMSNorm(enc, x: scratchPEProj, xOff: 0, w: lw.postNorm, y: scratchPEProj, yOff: 0, rows: activeM, dim: H, eps: config.rmsNormEps, useWeight: true)
            encodeResAddScale(enc, a: activeHidden, aOff: 0, b: scratchPEProj, bOff: 0, out: altHidden, outOff: 0, scale: lw.outScale, count: activeM * H)
            swap(&activeHidden, &altHidden)
        }

        if computeLogitsOnLast {
            let finalRowOff = (activeM - 1) * H * MemoryLayout<Float>.stride
            encodeRMSNorm(enc, x: activeHidden, xOff: finalRowOff, w: outputNorm, y: scratchNorm, yOff: 0, rows: 1, dim: H, eps: config.rmsNormEps, useWeight: true)
            encodeQ6Matvec(enc, w: lmHeadRef, x: scratchNorm, y: scratchLogits)
            encodeArgmax(enc)
        }

        guard let cmd = curCmd, let finalEnc = curEnc else {
            throw GemmaEngineError.inferenceFailed("Metal command buffer missing")
        }
        finalEnc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        curEnc = nil
        curCmd = nil

        if let err = cmd.error {
            throw GemmaEngineError.inferenceFailed("Metal command error: \(err.localizedDescription)")
        }

        if computeLogitsOnLast {
            return scratchBestToken.contents().bindMemory(to: Int32.self, capacity: 1)[0]
        }
        return 0
    }

    // MARK: - Kernel Encoders

    private func encodeQ4(
        _ enc: MTLComputeCommandEncoder,
        w: TensorRef,
        x: MTLBuffer,
        xOff: Int = 0,
        y: MTLBuffer,
        M: Int,
        K: Int? = nil,
        N: Int? = nil,
        label: String = "q4"
    ) {
        defer { prof((M == 1 ? "mv." : "mm.") + label) }
        var k = UInt32(K ?? w.cols)
        var n = UInt32(N ?? w.rows)
        if M == 1 {
            enc.setComputePipelineState(psoQ4Matvec)
            enc.setBuffer(w.buffer, offset: w.offset, index: 0)
            enc.setBuffer(x, offset: xOff, index: 1)
            enc.setBuffer(y, offset: 0, index: 2)
            enc.setBytes(&k, length: 4, index: 3)
            enc.setBytes(&n, length: 4, index: 4)
            let groups = (Int(n) + 15) / 16
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else {
            // Prefill: simdgroup-matrix GEMM (32 tokens x 64 rows per threadgroup).
            var m = UInt32(M)
            enc.setComputePipelineState(psoQ4MMSimd)
            enc.setBuffer(w.buffer, offset: w.offset, index: 0)
            enc.setBuffer(x, offset: xOff, index: 1)
            enc.setBuffer(y, offset: 0, index: 2)
            enc.setBytes(&k, length: 4, index: 3)
            enc.setBytes(&n, length: 4, index: 4)
            enc.setBytes(&m, length: 4, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: (Int(n) + 63) / 64, height: (M + 31) / 32, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
    }

    private func encodeF16Matmul(_ enc: MTLComputeCommandEncoder, w: TensorRef, x: MTLBuffer, y: MTLBuffer, M: Int) {
        defer { prof("f16_ple_model_proj") }
        var k = UInt32(w.cols)
        var n = UInt32(w.rows)
        if M > 1 {
            var m = UInt32(M)
            enc.setComputePipelineState(psoF16MMSimd)
            enc.setBuffer(w.buffer, offset: w.offset, index: 0)
            enc.setBuffer(x, offset: 0, index: 1)
            enc.setBuffer(y, offset: 0, index: 2)
            enc.setBytes(&k, length: 4, index: 3)
            enc.setBytes(&n, length: 4, index: 4)
            enc.setBytes(&m, length: 4, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: (w.rows + 63) / 64, height: (M + 31) / 32, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            return
        }
        enc.setComputePipelineState(psoF16Matmul)
        enc.setBuffer(w.buffer, offset: w.offset, index: 0)
        enc.setBuffer(x, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        enc.setBytes(&k, length: 4, index: 3)
        enc.setBytes(&n, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: (w.rows + 3) / 4, height: M, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    private func encodeQ6Matvec(_ enc: MTLComputeCommandEncoder, w: TensorRef, x: MTLBuffer, y: MTLBuffer) {
        defer { prof("lm_head_q6") }
        var k = UInt32(w.cols)
        var n = UInt32(w.rows)
        enc.setComputePipelineState(psoQ6Matvec)
        enc.setBuffer(w.buffer, offset: w.offset, index: 0)
        enc.setBuffer(x, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        enc.setBytes(&k, length: 4, index: 3)
        enc.setBytes(&n, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: (w.rows + 15) / 16, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    private func encodeRMSNorm(
        _ enc: MTLComputeCommandEncoder,
        x: MTLBuffer,
        xOff: Int,
        w: TensorRef?,
        y: MTLBuffer,
        yOff: Int,
        rows: Int,
        dim: Int,
        eps: Float,
        useWeight: Bool
    ) {
        defer { prof("rmsnorm") }
        var d = UInt32(dim)
        var ws: UInt32 = 0
        var epsVar = eps
        var uw: UInt32 = useWeight ? 1 : 0
        enc.setComputePipelineState(psoRMSNorm)
        enc.setBuffer(x, offset: xOff, index: 0)
        enc.setBuffer(w?.buffer ?? outputNorm.buffer, offset: w?.offset ?? outputNorm.offset, index: 1)
        enc.setBuffer(y, offset: yOff, index: 2)
        enc.setBytes(&d, length: 4, index: 3)
        enc.setBytes(&ws, length: 4, index: 4)
        enc.setBytes(&epsVar, length: 4, index: 5)
        enc.setBytes(&uw, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    private func encodePLECombine(
        _ enc: MTLComputeCommandEncoder,
        proj: MTLBuffer,
        rawScaled: MTLBuffer,
        out: MTLBuffer,
        M: Int
    ) {
        defer { prof("ple_combine") }
        var mVar = UInt32(M)
        var lVar = UInt32(config.numLayers)
        var epsVar = projNormEps
        var invS2 = invSqrt2
        enc.setComputePipelineState(psoPLECombine)
        enc.setBuffer(proj, offset: 0, index: 0)
        enc.setBuffer(perLayerProjNorm.buffer, offset: perLayerProjNorm.offset, index: 1)
        enc.setBuffer(rawScaled, offset: 0, index: 2)
        enc.setBuffer(out, offset: 0, index: 3)
        enc.setBytes(&mVar, length: 4, index: 4)
        enc.setBytes(&lVar, length: 4, index: 5)
        enc.setBytes(&epsVar, length: 4, index: 6)
        enc.setBytes(&invS2, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: config.numLayers, height: M, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    private func encodeRoPE(
        _ enc: MTLComputeCommandEncoder,
        qk: MTLBuffer,
        hd: Int,
        nHeads: Int,
        M: Int,
        posOff: Int,
        freqBase: Float,
        useFreqs: Bool
    ) {
        defer { prof("rope") }
        var hdVar = UInt32(hd)
        var nhVar = UInt32(nHeads)
        var poVar = UInt32(posOff)
        var fbVar = freqBase
        var ufVar: UInt32 = useFreqs ? 1 : 0
        enc.setComputePipelineState(psoRoPE)
        enc.setBuffer(qk, offset: 0, index: 0)
        enc.setBuffer(ropeFreqs.buffer, offset: ropeFreqs.offset, index: 1)
        enc.setBytes(&hdVar, length: 4, index: 2)
        enc.setBytes(&nhVar, length: 4, index: 3)
        enc.setBytes(&poVar, length: 4, index: 4)
        enc.setBytes(&fbVar, length: 4, index: 5)
        enc.setBytes(&ufVar, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: hd / 2, height: nHeads, depth: M),
                            threadsPerThreadgroup: MTLSize(width: min(hd / 2, 64), height: 1, depth: 1))
    }

    private func encodeWriteKV(
        _ enc: MTLComputeCommandEncoder,
        k: MTLBuffer,
        v: MTLBuffer,
        kCache: MTLBuffer,
        vCache: MTLBuffer,
        hd: Int,
        posOff: Int,
        winSize: UInt32,
        M: Int
    ) {
        defer { prof("write_kv") }
        var hdVar = UInt32(hd)
        var poVar = UInt32(posOff)
        var wsVar = winSize
        enc.setComputePipelineState(psoWriteKV)
        enc.setBuffer(k, offset: 0, index: 0)
        enc.setBuffer(v, offset: 0, index: 1)
        enc.setBuffer(kCache, offset: 0, index: 2)
        enc.setBuffer(vCache, offset: 0, index: 3)
        enc.setBytes(&hdVar, length: 4, index: 4)
        enc.setBytes(&poVar, length: 4, index: 5)
        enc.setBytes(&wsVar, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: hd, height: M, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(hd, 128), height: 1, depth: 1))
    }

    private func encodeAttention(
        _ enc: MTLComputeCommandEncoder,
        q: MTLBuffer,
        kCache: MTLBuffer,
        vCache: MTLBuffer,
        out: MTLBuffer,
        hd: Int,
        posOff: Int,
        winSize: UInt32,
        M: Int
    ) {
        defer { prof("attention") }
        var hdVar = UInt32(hd)
        var nhVar = UInt32(config.numAttentionHeads)
        var poVar = UInt32(posOff)
        var wsVar = winSize
        enc.setComputePipelineState(psoAttention)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(kCache, offset: 0, index: 1)
        enc.setBuffer(vCache, offset: 0, index: 2)
        enc.setBuffer(out, offset: 0, index: 3)
        enc.setBytes(&hdVar, length: 4, index: 4)
        enc.setBytes(&nhVar, length: 4, index: 5)
        enc.setBytes(&poVar, length: 4, index: 6)
        enc.setBytes(&wsVar, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: config.numAttentionHeads, height: M, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    private func encodeGeGLU(
        _ enc: MTLComputeCommandEncoder,
        gate: MTLBuffer,
        gateOff: Int,
        up: MTLBuffer,
        upOff: Int,
        count: Int
    ) {
        defer { prof("geglu") }
        var cnt = UInt32(count)
        enc.setComputePipelineState(psoGeGLU)
        enc.setBuffer(gate, offset: gateOff, index: 0)
        enc.setBuffer(up, offset: upOff, index: 1)
        enc.setBytes(&cnt, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private func encodeResAddScale(
        _ enc: MTLComputeCommandEncoder,
        a: MTLBuffer,
        aOff: Int,
        b: MTLBuffer,
        bOff: Int,
        out: MTLBuffer,
        outOff: Int,
        scale: Float,
        count: Int
    ) {
        defer { prof("residual") }
        var sVar = scale
        var cVar = UInt32(count)
        enc.setComputePipelineState(psoResAddScale)
        enc.setBuffer(a, offset: aOff, index: 0)
        enc.setBuffer(b, offset: bOff, index: 1)
        enc.setBuffer(out, offset: outOff, index: 2)
        enc.setBytes(&sVar, length: 4, index: 3)
        enc.setBytes(&cVar, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private func encodeArgmax(_ enc: MTLComputeCommandEncoder) {
        defer { prof("argmax") }
        var vs = UInt32(config.vocabSize)
        var sc = config.finalLogitSoftcapping
        var nParts = UInt32(argmaxStagesGroups)

        enc.setComputePipelineState(psoArgmax1)
        enc.setBuffer(scratchLogits, offset: 0, index: 0)
        enc.setBuffer(scratchArgmaxVals, offset: 0, index: 1)
        enc.setBuffer(scratchArgmaxIDs, offset: 0, index: 2)
        enc.setBytes(&vs, length: 4, index: 3)
        enc.setBytes(&sc, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: argmaxStagesGroups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        enc.setComputePipelineState(psoArgmax2)
        enc.setBuffer(scratchArgmaxVals, offset: 0, index: 0)
        enc.setBuffer(scratchArgmaxIDs, offset: 0, index: 1)
        enc.setBuffer(scratchBestToken, offset: 0, index: 2)
        enc.setBytes(&nParts, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    // MARK: - Prefix KV Cache Snapshot

    private func captureKVSnapshot(tokens: [Int32]) -> PrefixSnapshot {
        let len = tokens.count
        var ks: [Data] = []
        var vs: [Data] = []
        ks.reserveCapacity(config.numOwnKVLayers)
        vs.reserveCapacity(config.numOwnKVLayers)
        for il in 0..<config.numOwnKVLayers {
            let hd = config.headDim(layer: il)
            let validSlots = config.isSlidingLayer[il] ? min(len, config.slidingWindow) : len
            let byteCount = validSlots * hd * MemoryLayout<UInt16>.stride
            ks.append(Data(bytes: kCaches[il].contents(), count: byteCount))
            vs.append(Data(bytes: vCaches[il].contents(), count: byteCount))
        }
        return PrefixSnapshot(tokens: tokens, kSnapshots: ks, vSnapshots: vs)
    }

    private func restoreKVSnapshot(_ snap: PrefixSnapshot) {
        for il in 0..<config.numOwnKVLayers {
            let kData = snap.kSnapshots[il]
            kData.withUnsafeBytes { raw in
                if let base = raw.baseAddress {
                    _ = memcpy(kCaches[il].contents(), base, kData.count)
                }
            }
            let vData = snap.vSnapshots[il]
            vData.withUnsafeBytes { raw in
                if let base = raw.baseAddress {
                    _ = memcpy(vCaches[il].contents(), base, vData.count)
                }
            }
        }
    }

    // MARK: - Metal Shader Loading (SwiftPM resource + Xcode default library + runtime source fallback)

    private static func loadMetalLibrary(device: MTLDevice) throws -> MTLLibrary {
        if let lib = try? device.makeDefaultLibrary(bundle: Bundle.module) {
            return lib
        }
        if let mainLib = device.makeDefaultLibrary(),
           mainLib.functionNames.contains("q4_0_matvec_f32") {
            return mainLib
        }
        if let url = Bundle.module.url(forResource: "Gemma4Shaders", withExtension: "metal"),
           let src = try? String(contentsOf: url, encoding: .utf8) {
            return try device.makeLibrary(source: src, options: nil)
        }
        if let url = Bundle.main.url(forResource: "Gemma4Shaders", withExtension: "metal"),
           let src = try? String(contentsOf: url, encoding: .utf8) {
            return try device.makeLibrary(source: src, options: nil)
        }
        throw GemmaEngineError.shaderLibraryNotFound
    }
}
