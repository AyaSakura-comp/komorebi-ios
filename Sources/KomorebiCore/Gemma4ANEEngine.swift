import Accelerate
import CoreML
import Foundation

/// Stateful CoreML ANE Engine for `mlboydaisuke/gemma-4-E2B-stateful-coreml` (3-chunk Linear + MLState).
/// Runs natively on Apple Neural Engine (`.cpuAndNeuralEngine`), which iOS permits EVEN while the
/// Host App is running in the background under `UIBackgroundModes: audio` for the Keyboard Extension!
@available(iOS 18.0, macOS 15.0, *)
public final class Gemma4ANEEngine: @unchecked Sendable {
    public struct BundleConfig: Codable {
        public let hidden_size: Int
        public let num_hidden_layers: Int
        public let context_length: Int
        public let sliding_window: Int
        public let vocab_size: Int
        public let per_layer_dim: Int
        public let embed_scale: Double
        public let per_layer_embed_scale: Double
    }

    public let modelDirectory: URL
    public let bundleConfig: BundleConfig

    private let chunk1: MLModel
    private let chunk2: MLModel
    private let chunk3: MLModel
    private let chunk1Prefill: MLModel?
    private let chunk2Prefill: MLModel?
    private let chunk3Prefill: MLModel?
    private let prefillBatchSize: Int

    private let embedTokens: MappedQ8EmbeddingLookup
    private let embedPerLayer: MappedQ8EmbeddingLookup
    private let cosSlidingTable: Data
    private let sinSlidingTable: Data
    private let cosFullTable: Data
    private let sinFullTable: Data

    private let maskFull: MLMultiArray
    private let maskSliding: MLMultiArray
    private let posArray: MLMultiArray
    private let ringArray: MLMultiArray

    private var batchHidden: MLMultiArray?
    private var batchPLRaw: MLMultiArray?
    private var batchMaskFull: MLMultiArray?
    private var batchMaskSliding: MLMultiArray?
    private var batchCosS: MLMultiArray?
    private var batchSinS: MLMultiArray?
    private var batchCosF: MLMultiArray?
    private var batchSinF: MLMultiArray?

    private var persistedTokens: [Int32] = []
    private var persistedMaxPos: Int = 0
    private var persistedState1: MLState?
    private var persistedState2: MLState?

    public init(directory: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine) throws {
        self.modelDirectory = directory
        let cfgData = try Data(contentsOf: directory.appendingPathComponent("model_config.json"))
        let mc = try JSONDecoder().decode(BundleConfig.self, from: cfgData)
        self.bundleConfig = mc

        let mlCfg = MLModelConfiguration()
        mlCfg.computeUnits = computeUnits

        func openChunk(_ name: String) throws -> MLModel {
            let compiled = directory.appendingPathComponent("\(name).mlmodelc")
            if FileManager.default.fileExists(atPath: compiled.path) {
                return try MLModel(contentsOf: compiled, configuration: mlCfg)
            }
            let pkg = directory.appendingPathComponent("\(name).mlpackage")
            let compiledURL = try MLModel.compileModel(at: pkg)
            return try MLModel(contentsOf: compiledURL, configuration: mlCfg)
        }

        self.chunk1 = try openChunk("chunk_1")
        self.chunk2 = try openChunk("chunk_2")
        self.chunk3 = try openChunk("chunk_3")

        var foundT = 1
        var p1: MLModel?
        var p2: MLModel?
        var p3: MLModel?
        for candT in [16, 8, 4] {
            let pcfg = MLModelConfiguration()
            pcfg.computeUnits = computeUnits
            pcfg.functionName = "prefill_b\(candT)"
            let c1u = directory.appendingPathComponent("chunk_1.mlmodelc")
            let c2u = directory.appendingPathComponent("chunk_2.mlmodelc")
            let c3u = directory.appendingPathComponent("chunk_3.mlmodelc")
            if let m1 = try? MLModel(contentsOf: c1u, configuration: pcfg),
               let m2 = try? MLModel(contentsOf: c2u, configuration: pcfg),
               let m3 = try? MLModel(contentsOf: c3u, configuration: pcfg) {
                p1 = m1; p2 = m2; p3 = m3; foundT = candT
                break
            }
        }
        self.chunk1Prefill = p1
        self.chunk2Prefill = p2
        self.chunk3Prefill = p3
        self.prefillBatchSize = foundT

        self.embedTokens = try MappedQ8EmbeddingLookup(
            dataURL: directory.appendingPathComponent("embed_tokens_q8.bin"),
            scalesURL: directory.appendingPathComponent("embed_tokens_scales.bin"),
            dim: mc.hidden_size,
            scale: Float(mc.embed_scale)
        )
        self.embedPerLayer = try MappedQ8EmbeddingLookup(
            dataURL: directory.appendingPathComponent("embed_tokens_per_layer_q8.bin"),
            scalesURL: directory.appendingPathComponent("embed_tokens_per_layer_scales.bin"),
            dim: mc.num_hidden_layers * mc.per_layer_dim,
            scale: Float(mc.per_layer_embed_scale)
        )

        self.cosSlidingTable = try Data(contentsOf: directory.appendingPathComponent("cos_sliding.npy"), options: .mappedIfSafe)
        self.sinSlidingTable = try Data(contentsOf: directory.appendingPathComponent("sin_sliding.npy"), options: .mappedIfSafe)
        self.cosFullTable = try Data(contentsOf: directory.appendingPathComponent("cos_full.npy"), options: .mappedIfSafe)
        self.sinFullTable = try Data(contentsOf: directory.appendingPathComponent("sin_full.npy"), options: .mappedIfSafe)

        self.maskFull = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: mc.context_length)], dataType: .float16)
        self.maskSliding = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: mc.sliding_window)], dataType: .float16)
        self.posArray = try MLMultiArray(shape: [1], dataType: .int32)
        self.ringArray = try MLMultiArray(shape: [1], dataType: .int32)
    }

    public func clearPrefixCache() {
        persistedTokens = []
        persistedMaxPos = 0
        persistedState1 = nil
        persistedState2 = nil
    }

    public func generate(
        inputIDs: [Int32],
        maxNewTokens: Int = 256,
        stopTokenIDs: Set<Int32>,
        shouldCancel: (() -> Bool)? = nil,
        onToken: ((Int32) -> Void)? = nil
    ) async throws -> (tokenIDs: [Int32], stats: GenerationStats) {
        guard !inputIDs.isEmpty, inputIDs.count < bundleConfig.context_length else {
            throw GemmaEngineError.inferenceFailed("ANE prompt length \(inputIDs.count) out of range")
        }

        let W = bundleConfig.sliding_window
        var resumeAt = 0
        if let _ = persistedState1, let _ = persistedState2, !persistedTokens.isEmpty, persistedMaxPos < W {
            let cap = min(persistedTokens.count, inputIDs.count)
            var l = 0
            while l < cap && persistedTokens[l] == inputIDs[l] { l += 1 }
            if l >= 16 && l < inputIDs.count {
                resumeAt = l
            }
        }

        let s1: MLState
        let s2: MLState
        if resumeAt > 0, let ps1 = persistedState1, let ps2 = persistedState2 {
            s1 = ps1
            s2 = ps2
            persistedTokens = []
            persistedMaxPos = 0
        } else {
            s1 = chunk1.makeState()
            s2 = chunk2.makeState()
            persistedState1 = s1
            persistedState2 = s2
            persistedTokens = []
            persistedMaxPos = 0
        }

        let opts = MLPredictionOptions()
        var pos = resumeAt
        var i = resumeAt
        var prefillPred: Int32 = 0

        let t0 = CFAbsoluteTimeGetCurrent()
        if chunk1Prefill != nil && prefillBatchSize > 1 {
            let T = prefillBatchSize
            while i + T <= inputIDs.count {
                if shouldCancel?() == true { break }
                if (pos % W) + T > W { break }
                prefillPred = try await prefillStep(inputIDs: inputIDs, start: i, pos: pos, T: T, s1: s1, s2: s2, opts: opts)
                pos += T
                i += T
            }
        }
        while i < inputIDs.count {
            if shouldCancel?() == true { break }
            prefillPred = try await decodeStep(token: inputIDs[i], pos: pos, s1: s1, s2: s2, opts: opts)
            pos += 1
            i += 1
        }
        let t1 = CFAbsoluteTimeGetCurrent()

        var decoded: [Int32] = []
        let t2 = CFAbsoluteTimeGetCurrent()
        if maxNewTokens > 0 && inputIDs.count > resumeAt && !stopTokenIDs.contains(prefillPred) {
            decoded.append(prefillPred)
            onToken?(prefillPred)
            while decoded.count < maxNewTokens && pos < bundleConfig.context_length {
                if shouldCancel?() == true { break }
                let next = try await decodeStep(token: decoded.last!, pos: pos, s1: s1, s2: s2, opts: opts)
                pos += 1
                if stopTokenIDs.contains(next) { break }
                decoded.append(next)
                onToken?(next)
            }
        }
        let t3 = CFAbsoluteTimeGetCurrent()

        if !decoded.isEmpty {
            var nextPersisted = inputIDs
            nextPersisted.append(contentsOf: decoded.dropLast())
            persistedTokens = nextPersisted
            persistedMaxPos = pos
        }

        let stats = GenerationStats(
            backend: "CoreML ANE (NPU)",
            prefixCachedTokens: resumeAt,
            prefillTokens: inputIDs.count - resumeAt,
            prefillSeconds: max(t1 - t0, 1e-6),
            decodeTokens: decoded.count,
            decodeSeconds: max(t3 - t2, 1e-6)
        )
        return (decoded, stats)
    }

    private func decodeStep(token: Int32, pos: Int, s1: MLState, s2: MLState, opts: MLPredictionOptions) async throws -> Int32 {
        let H = bundleConfig.hidden_size
        let PL = bundleConfig.num_hidden_layers * bundleConfig.per_layer_dim
        let ctx = bundleConfig.context_length
        let W = bundleConfig.sliding_window
        let tid = max(0, Int(token))

        let hidden = try embedTokens.lookup(tid, shape: [1, 1, NSNumber(value: H)])
        let plRaw = try embedPerLayer.lookup(tid, shape: [1, 1, NSNumber(value: PL)])

        let neg = Float16(-65504).bitPattern
        let mf = maskFull.dataPointer.bindMemory(to: UInt16.self, capacity: ctx)
        let fullValid = min(max(pos, 0), ctx - 1)
        for i in 0..<ctx { mf[i] = i <= fullValid ? 0 : neg }

        let ms = maskSliding.dataPointer.bindMemory(to: UInt16.self, capacity: W)
        let slidValid = min(pos + 1, W)
        for i in 0..<W { ms[i] = i < slidValid ? 0 : neg }

        posArray.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = Int32(pos)
        ringArray.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = Int32(pos % W)

        let cosS = try lookupRoPE(table: cosSlidingTable, pos: pos, dim: 256)
        let sinS = try lookupRoPE(table: sinSlidingTable, pos: pos, dim: 256)
        let cosF = try lookupRoPE(table: cosFullTable, pos: pos, dim: 512)
        let sinF = try lookupRoPE(table: sinFullTable, pos: pos, dim: 512)

        let p1 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: hidden),
            "causal_mask_full": MLFeatureValue(multiArray: maskFull),
            "causal_mask_sliding": MLFeatureValue(multiArray: maskSliding),
            "per_layer_raw": MLFeatureValue(multiArray: plRaw),
            "cos_s": MLFeatureValue(multiArray: cosS),
            "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF),
            "sin_f": MLFeatureValue(multiArray: sinF),
            "current_pos": MLFeatureValue(multiArray: posArray),
            "ring_pos": MLFeatureValue(multiArray: ringArray),
        ])
        let out1 = try await chunk1.prediction(from: p1, using: s1, options: opts)
        guard let h1 = out1.featureValue(for: "hidden_states_out"),
              let plc = out1.featureValue(for: "per_layer_combined_out") else {
            throw GemmaEngineError.inferenceFailed("ANE chunk_1 missing outputs")
        }

        let p2 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": h1,
            "causal_mask_full": MLFeatureValue(multiArray: maskFull),
            "causal_mask_sliding": MLFeatureValue(multiArray: maskSliding),
            "per_layer_combined": plc,
            "cos_s": MLFeatureValue(multiArray: cosS),
            "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF),
            "sin_f": MLFeatureValue(multiArray: sinF),
            "current_pos": MLFeatureValue(multiArray: posArray),
            "ring_pos": MLFeatureValue(multiArray: ringArray),
        ])
        let out2 = try await chunk2.prediction(from: p2, using: s2, options: opts)
        guard let h2 = out2.featureValue(for: "hidden_states_out"),
              let kv13k = out2.featureValue(for: "kv13_k"),
              let kv13v = out2.featureValue(for: "kv13_v"),
              let kv14k = out2.featureValue(for: "kv14_k"),
              let kv14v = out2.featureValue(for: "kv14_v") else {
            throw GemmaEngineError.inferenceFailed("ANE chunk_2 missing outputs")
        }

        let p3 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": h2,
            "causal_mask_full": MLFeatureValue(multiArray: maskFull),
            "causal_mask_sliding": MLFeatureValue(multiArray: maskSliding),
            "per_layer_combined": plc,
            "cos_s": MLFeatureValue(multiArray: cosS),
            "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF),
            "sin_f": MLFeatureValue(multiArray: sinF),
            "kv13_k": kv13k, "kv13_v": kv13v,
            "kv14_k": kv14k, "kv14_v": kv14v,
        ])
        let out3 = try await chunk3.prediction(from: p3, options: opts)
        guard let tokArr = out3.featureValue(for: "token_id")?.multiArrayValue else {
            throw GemmaEngineError.inferenceFailed("ANE chunk_3 missing token_id")
        }
        return tokArr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]
    }

    private func prefillStep(inputIDs: [Int32], start: Int, pos: Int, T: Int, s1: MLState, s2: MLState, opts: MLPredictionOptions) async throws -> Int32 {
        guard let c1 = chunk1Prefill, let c2 = chunk2Prefill, let c3 = chunk3Prefill else {
            throw GemmaEngineError.inferenceFailed("ANE prefill chunks unavailable")
        }
        let H = bundleConfig.hidden_size
        let PL = bundleConfig.num_hidden_layers * bundleConfig.per_layer_dim
        let ctx = bundleConfig.context_length
        let W = bundleConfig.sliding_window

        if batchHidden == nil || batchHidden!.shape[1].intValue != T {
            batchHidden = try MLMultiArray(shape: [1, NSNumber(value: T), NSNumber(value: H)], dataType: .float16)
            batchPLRaw = try MLMultiArray(shape: [1, NSNumber(value: T), NSNumber(value: PL)], dataType: .float16)
            batchMaskFull = try MLMultiArray(shape: [1, 1, NSNumber(value: T), NSNumber(value: ctx)], dataType: .float16)
            batchMaskSliding = try MLMultiArray(shape: [1, 1, NSNumber(value: T), NSNumber(value: W)], dataType: .float16)
            batchCosS = try MLMultiArray(shape: [1, 1, NSNumber(value: T), 256], dataType: .float16)
            batchSinS = try MLMultiArray(shape: [1, 1, NSNumber(value: T), 256], dataType: .float16)
            batchCosF = try MLMultiArray(shape: [1, 1, NSNumber(value: T), 512], dataType: .float16)
            batchSinF = try MLMultiArray(shape: [1, 1, NSNumber(value: T), 512], dataType: .float16)
        }

        let hPtr = batchHidden!.dataPointer.bindMemory(to: UInt16.self, capacity: T * H)
        let plPtr = batchPLRaw!.dataPointer.bindMemory(to: UInt16.self, capacity: T * PL)
        for t in 0..<T {
            let tid = max(0, Int(inputIDs[start + t]))
            let r = try embedTokens.lookup(tid, shape: [1, 1, NSNumber(value: H)])
            _ = memcpy(hPtr.advanced(by: t * H), r.dataPointer, H * 2)
            let pr = try embedPerLayer.lookup(tid, shape: [1, 1, NSNumber(value: PL)])
            _ = memcpy(plPtr.advanced(by: t * PL), pr.dataPointer, PL * 2)
        }

        let neg = Float16(-65504).bitPattern
        let mf = batchMaskFull!.dataPointer.bindMemory(to: UInt16.self, capacity: T * ctx)
        let ms = batchMaskSliding!.dataPointer.bindMemory(to: UInt16.self, capacity: T * W)
        for t in 0..<T {
            let p = pos + t
            let full = min(max(p, 0), ctx - 1)
            for i in 0..<ctx { mf[t * ctx + i] = i <= full ? 0 : neg }
            let slid = min(p + 1, W)
            for i in 0..<W { ms[t * W + i] = i < slid ? 0 : neg }
        }

        fillBatchRoPE(table: cosSlidingTable, dst: batchCosS!, pos: pos, T: T, dim: 256)
        fillBatchRoPE(table: sinSlidingTable, dst: batchSinS!, pos: pos, T: T, dim: 256)
        fillBatchRoPE(table: cosFullTable, dst: batchCosF!, pos: pos, T: T, dim: 512)
        fillBatchRoPE(table: sinFullTable, dst: batchSinF!, pos: pos, T: T, dim: 512)
        posArray.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = Int32(pos)
        ringArray.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = Int32(pos % W)

        let p1 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: batchHidden!),
            "causal_mask_full": MLFeatureValue(multiArray: batchMaskFull!),
            "causal_mask_sliding": MLFeatureValue(multiArray: batchMaskSliding!),
            "per_layer_raw": MLFeatureValue(multiArray: batchPLRaw!),
            "cos_s": MLFeatureValue(multiArray: batchCosS!),
            "sin_s": MLFeatureValue(multiArray: batchSinS!),
            "cos_f": MLFeatureValue(multiArray: batchCosF!),
            "sin_f": MLFeatureValue(multiArray: batchSinF!),
            "current_pos": MLFeatureValue(multiArray: posArray),
            "ring_pos": MLFeatureValue(multiArray: ringArray),
        ])
        let out1 = try await c1.prediction(from: p1, using: s1, options: opts)
        guard let h1 = out1.featureValue(for: "hidden_states_out"),
              let plc = out1.featureValue(for: "per_layer_combined_out") else {
            throw GemmaEngineError.inferenceFailed("ANE prefill chunk_1 missing outputs")
        }

        let p2 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": h1,
            "causal_mask_full": MLFeatureValue(multiArray: batchMaskFull!),
            "causal_mask_sliding": MLFeatureValue(multiArray: batchMaskSliding!),
            "per_layer_combined": plc,
            "cos_s": MLFeatureValue(multiArray: batchCosS!),
            "sin_s": MLFeatureValue(multiArray: batchSinS!),
            "cos_f": MLFeatureValue(multiArray: batchCosF!),
            "sin_f": MLFeatureValue(multiArray: batchSinF!),
            "current_pos": MLFeatureValue(multiArray: posArray),
            "ring_pos": MLFeatureValue(multiArray: ringArray),
        ])
        let out2 = try await c2.prediction(from: p2, using: s2, options: opts)
        guard let h2 = out2.featureValue(for: "hidden_states_out"),
              let kv13k = out2.featureValue(for: "kv13_k"),
              let kv13v = out2.featureValue(for: "kv13_v"),
              let kv14k = out2.featureValue(for: "kv14_k"),
              let kv14v = out2.featureValue(for: "kv14_v") else {
            throw GemmaEngineError.inferenceFailed("ANE prefill chunk_2 missing outputs")
        }

        let p3 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": h2,
            "causal_mask_full": MLFeatureValue(multiArray: batchMaskFull!),
            "causal_mask_sliding": MLFeatureValue(multiArray: batchMaskSliding!),
            "per_layer_combined": plc,
            "cos_s": MLFeatureValue(multiArray: batchCosS!),
            "sin_s": MLFeatureValue(multiArray: batchSinS!),
            "cos_f": MLFeatureValue(multiArray: batchCosF!),
            "sin_f": MLFeatureValue(multiArray: batchSinF!),
            "kv13_k": kv13k, "kv13_v": kv13v,
            "kv14_k": kv14k, "kv14_v": kv14v,
        ])
        let out3 = try await c3.prediction(from: p3, options: opts)
        guard let tokArr = out3.featureValue(for: "token_id")?.multiArrayValue else {
            throw GemmaEngineError.inferenceFailed("ANE prefill chunk_3 missing token_id")
        }
        return tokArr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]
    }

    private func npyHeaderSize(_ data: Data) -> Int {
        data.withUnsafeBytes { raw in
            guard raw.count >= 10 else { return 128 }
            let b = raw.bindMemory(to: UInt8.self)
            return 10 + (Int(b[8]) | (Int(b[9]) << 8))
        }
    }

    private func lookupRoPE(table: Data, pos: Int, dim: Int) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: dim)], dataType: .float16)
        let hdr = npyHeaderSize(table)
        let rowBytes = dim * 2
        let off = hdr + pos * rowBytes
        if off + rowBytes <= table.count {
            table.withUnsafeBytes { raw in
                _ = memcpy(arr.dataPointer, raw.baseAddress!.advanced(by: off), rowBytes)
            }
        }
        return arr
    }

    private func fillBatchRoPE(table: Data, dst: MLMultiArray, pos: Int, T: Int, dim: Int) {
        let hdr = npyHeaderSize(table)
        let rowBytes = dim * 2
        let p = dst.dataPointer.bindMemory(to: UInt16.self, capacity: T * dim)
        table.withUnsafeBytes { raw in
            for t in 0..<T {
                let off = hdr + (pos + t) * rowBytes
                if off + rowBytes <= table.count {
                    _ = memcpy(p.advanced(by: t * dim), raw.baseAddress!.advanced(by: off), rowBytes)
                } else {
                    _ = memset(p.advanced(by: t * dim), 0, rowBytes)
                }
            }
        }
    }
}

private final class MappedQ8EmbeddingLookup {
    private let data: Data
    private let scales: Data
    private let dim: Int
    private let scale: Float
    private var f32Scratch: [Float]

    init(dataURL: URL, scalesURL: URL, dim: Int, scale: Float) throws {
        self.data = try Data(contentsOf: dataURL, options: .mappedIfSafe)
        self.scales = try Data(contentsOf: scalesURL, options: .mappedIfSafe)
        self.dim = dim
        self.scale = scale
        self.f32Scratch = [Float](repeating: 0, count: dim)
    }

    func lookup(_ tokenID: Int, shape: [NSNumber]) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: shape, dataType: .float16)
        let dstPtr = result.dataPointer.bindMemory(to: UInt16.self, capacity: dim)
        data.withUnsafeBytes { rawPtr in
            let int8Ptr = rawPtr.baseAddress!.assumingMemoryBound(to: Int8.self).advanced(by: tokenID * dim)
            scales.withUnsafeBytes { scaleRaw in
                let scaleBits = scaleRaw.baseAddress!.assumingMemoryBound(to: UInt16.self)[tokenID]
                let rowScale = Float(Float16(bitPattern: scaleBits)) / 127.0 * scale
                vDSP.convertElements(of: UnsafeBufferPointer(start: int8Ptr, count: dim), to: &f32Scratch)
                vDSP.multiply(rowScale, f32Scratch, result: &f32Scratch)
                f32Scratch.withUnsafeBufferPointer { src in
                    var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src.baseAddress!), height: 1, width: UInt(dim), rowBytes: dim * 4)
                    var dstBuf = vImage_Buffer(data: UnsafeMutableRawPointer(dstPtr), height: 1, width: UInt(dim), rowBytes: dim * 2)
                    vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0)
                }
            }
        }
        return result
    }
}
