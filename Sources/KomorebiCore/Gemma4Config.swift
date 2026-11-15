import Foundation

/// Hyperparameters and per-layer geometry for Gemma 4 E2B (and compatible dense Gemma 4 variants).
public struct Gemma4Config: Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let numKVSharedLayers: Int
    public let numOwnKVLayers: Int
    public let perLayerInputDim: Int

    public let slidingHeadDim: Int
    public let fullHeadDim: Int
    public let slidingWindow: Int

    public let intermediateSizeBase: Int
    public let sharedLayerIntermediateSize: Int

    public let rmsNormEps: Float
    public let ropeFreqBaseSWA: Float
    public let ropeFreqBaseFull: Float
    public let finalLogitSoftcapping: Float

    /// `true` = sliding_attention, `false` = full_attention. Length == `numLayers`.
    public let isSlidingLayer: [Bool]
    /// For layer `il`, returns `il` if `il < numOwnKVLayers`, else the donor KV layer index (e.g., 13 or 14).
    public let kvSourceLayer: [Int]
    /// Maps layer index `il` (`0..<numLayers`) to cache index `0..<numOwnKVLayers`.
    public let ownKVCacheIndex: [Int]

    public let bosTokenID: Int32
    public let eosTokenID: Int32
    public let turnEndTokenID: Int32

    public static func from(gguf: GGUFMetadata) -> Gemma4Config {
        let arch = gguf.string("general.architecture") ?? "gemma4"
        let vocabSize = gguf.arrayCount("tokenizer.ggml.tokens") ?? 262_144
        let hiddenSize = gguf.int("\(arch).embedding_length") ?? 1536
        let numLayers = gguf.int("\(arch).block_count") ?? 35
        let numHeads = gguf.int("\(arch).attention.head_count") ?? 8
        let numKVHeads = gguf.int("\(arch).attention.head_count_kv") ?? 1
        let numKVShared = gguf.int("\(arch).attention.shared_kv_layers") ?? 20
        let numOwnKV = max(2, numLayers - numKVShared)
        let perLayerDim = gguf.int("\(arch).embedding_length_per_layer_input") ?? 256

        let fullHeadDim = gguf.int("\(arch).attention.key_length") ?? 512
        let swaHeadDim = gguf.int("\(arch).attention.key_length_swa") ?? 256
        let slidingWindow = gguf.int("\(arch).attention.sliding_window") ?? 512

        let ffnLengths = gguf.intArray("\(arch).feed_forward_length")
        let baseFF = ffnLengths?.first ?? (gguf.int("\(arch).feed_forward_length") ?? 6144)
        let lastFF = ffnLengths?.last ?? (numKVShared > 0 ? baseFF * 2 : baseFF)

        let eps = gguf.float("\(arch).attention.layer_norm_rms_eps") ?? 1e-6
        let freqBaseSWA = gguf.float("\(arch).rope.freq_base_swa") ?? 10_000.0
        let freqBaseFull = gguf.float("\(arch).rope.freq_base") ?? 1_000_000.0
        let softcap = gguf.float("\(arch).final_logit_softcapping") ?? 30.0

        let defaultPattern = (0..<numLayers).map { ($0 % 5) != 4 }
        let isSWA = gguf.boolArray("\(arch).attention.sliding_window_pattern") ?? defaultPattern

        var kvSource = [Int](repeating: 0, count: numLayers)
        for il in 0..<numLayers {
            if il < numOwnKV {
                kvSource[il] = il
            } else {
                kvSource[il] = numOwnKV - (isSWA[il] ? 2 : 1)
            }
        }

        let bos = Int32(gguf.int("tokenizer.ggml.bos_token_id") ?? 2)
        let eos = Int32(gguf.int("tokenizer.ggml.eos_token_id") ?? 1)

        return Gemma4Config(
            vocabSize: vocabSize,
            hiddenSize: hiddenSize,
            numLayers: numLayers,
            numAttentionHeads: numHeads,
            numKeyValueHeads: numKVHeads,
            numKVSharedLayers: numKVShared,
            numOwnKVLayers: numOwnKV,
            perLayerInputDim: perLayerDim,
            slidingHeadDim: swaHeadDim,
            fullHeadDim: fullHeadDim,
            slidingWindow: slidingWindow,
            intermediateSizeBase: baseFF,
            sharedLayerIntermediateSize: lastFF,
            rmsNormEps: eps,
            ropeFreqBaseSWA: freqBaseSWA,
            ropeFreqBaseFull: freqBaseFull,
            finalLogitSoftcapping: softcap,
            isSlidingLayer: isSWA,
            kvSourceLayer: kvSource,
            ownKVCacheIndex: kvSource,
            bosTokenID: bos,
            eosTokenID: eos,
            turnEndTokenID: 106
        )
    }

    @inline(__always)
    public func headDim(layer: Int) -> Int {
        isSlidingLayer[layer] ? slidingHeadDim : fullHeadDim
    }

    @inline(__always)
    public func intermediateSize(layer: Int) -> Int {
        layer < numOwnKVLayers ? intermediateSizeBase : sharedLayerIntermediateSize
    }

    @inline(__always)
    public func hasOwnKV(layer: Int) -> Bool {
        layer < numOwnKVLayers
    }
}
