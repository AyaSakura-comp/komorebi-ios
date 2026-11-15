import Accelerate
import Foundation
import KomorebiANEDirect

// Qwen3-ASR-1.7B on the Apple Neural Engine (same private in-memory MIL compiler as the Gemma 4 engine).
//
//  - CPU: Whisper log-mel features (`Qwen3ASRMel`), token embeddings (Q8_0 rows), decoder attention with the KV cache
//    (GQA, cblas), exact rescoring of the LM head's best candidates, byte-level BPE decoding.
//  - ANE: the audio encoder (conv front end + 24 transformer layers + projection, one 8-second window per run) and the
//    Qwen3 1.7B decoder as boundary programs (tail of layer l-1 fused with the head of layer l, like the Gemma engine),
//    plus a full-vocabulary LM head.
// The weights come from `Tools/qwen3-asr/convert_gguf.py` (Q8_0 GGUF) and are re-quantised into LUT palettes once;
// the palettes are cached next to the other ANE caches.

/// Hyper-parameters and prompt pieces of the converted Qwen3-ASR GGUF.
public struct Qwen3ASRConfig: Sendable {
    public let numLayers: Int
    public let hidden: Int
    public let ffn: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let vocabSize: Int
    public let rmsEps: Float
    public let ropeBase: Float
    public let audioLayers: Int
    public let audioHidden: Int
    public let audioFFN: Int
    public let audioHeads: Int
    public let melBins: Int
    public let downsampleHidden: Int
    public let nWindow: Int
    public let nWindowInfer: Int
    public let audioOutput: Int
    public let audioLNEps: Float
    /// Whisper mel filter bank, `[nFFT / 2 + 1][melBins]`.
    public let melFilters: [Float]
    public let nFFT: Int
    public let hop: Int
    /// Prompt ids before the audio pads (`… <|audio_start|>`) and after them (`<|audio_end|> … assistant\n`).
    public let promptPrefix: [Int32]
    public let promptSuffix: [Int32]
    /// `language Chinese<asr_text>`: forces the language instead of letting the model name it.
    public let languageChinese: [Int32]
    public let audioTokenID: Int32
    public let asrTextTokenID: Int32
    public let eosTokenIDs: Set<Int32>

    /// Mel frames per encoder chunk (the conv front end runs on 1-second chunks).
    public var chunkFrames: Int { nWindow * 2 }
    /// Encoder tokens of a full chunk (100 frames -> 13).
    public var tokensPerChunk: Int { Self.convOut(Self.convOut(Self.convOut(chunkFrames))) }
    /// Chunks per attention window (8 -> 104 tokens attend to each other).
    public var chunksPerWindow: Int { nWindowInfer / chunkFrames }
    public var windowTokens: Int { tokensPerChunk * chunksPerWindow }
    public var freqAfterConv: Int { Self.convOut(Self.convOut(Self.convOut(melBins))) }

    /// Length after one 3x3 / stride 2 / pad 1 convolution.
    static func convOut(_ n: Int) -> Int { n <= 0 ? 0 : (n - 1) / 2 + 1 }

    /// Audio tokens for `frames` mel frames (`_get_feat_extract_output_lengths`).
    public func audioTokenCount(frames: Int) -> Int {
        let rest = frames % chunkFrames
        return Self.convOut(Self.convOut(Self.convOut(rest))) + (frames / chunkFrames) * tokensPerChunk
    }

    static func from(_ m: GGUFMetadata) throws -> Qwen3ASRConfig {
        func int(_ k: String) throws -> Int {
            guard let v = m.int("qwen3asr.\(k)") else { throw GemmaEngineError.inferenceFailed("Qwen3-ASR GGUF: missing \(k)") }
            return v
        }
        func ids(_ k: String) throws -> [Int32] {
            guard let v = m.intArray("qwen3asr.\(k)") else { throw GemmaEngineError.inferenceFailed("Qwen3-ASR GGUF: missing \(k)") }
            return v.map { Int32($0) }
        }
        guard let mel = m.floatArray("qwen3asr.audio.mel_filters") else {
            throw GemmaEngineError.inferenceFailed("Qwen3-ASR GGUF: missing the mel filter bank")
        }
        return Qwen3ASRConfig(
            numLayers: try int("block_count"), hidden: try int("embedding_length"), ffn: try int("feed_forward_length"),
            numHeads: try int("attention.head_count"), numKVHeads: try int("attention.head_count_kv"),
            headDim: try int("attention.key_length"), vocabSize: try int("vocab_size"),
            rmsEps: m.float("qwen3asr.attention.layer_norm_rms_epsilon") ?? 1e-6,
            ropeBase: m.float("qwen3asr.rope.freq_base") ?? 1_000_000,
            audioLayers: try int("audio.block_count"), audioHidden: try int("audio.embedding_length"),
            audioFFN: try int("audio.feed_forward_length"), audioHeads: try int("audio.head_count"),
            melBins: try int("audio.num_mel_bins"), downsampleHidden: try int("audio.downsample_hidden_size"),
            nWindow: try int("audio.n_window"), nWindowInfer: try int("audio.n_window_infer"),
            audioOutput: try int("audio.output_dim"), audioLNEps: m.float("qwen3asr.audio.layer_norm_epsilon") ?? 1e-5,
            melFilters: mel, nFFT: try int("audio.n_fft"), hop: try int("audio.hop_length"),
            promptPrefix: try ids("prompt.prefix"), promptSuffix: try ids("prompt.suffix"),
            languageChinese: try ids("prompt.language_chinese"),
            audioTokenID: Int32(try int("audio_token_id")), asrTextTokenID: Int32(try int("asr_text_token_id")),
            eosTokenIDs: Set(try ids("eos_token_ids")))
    }
}

// MARK: - Log-mel features

/// Whisper log-mel spectrogram exactly as `WhisperFeatureExtractor` computes it for Qwen3-ASR: reflect-padded STFT
/// (n_fft 400, hop 160, periodic Hann), power spectrum, Slaney mel filters, log10, clamp to max - 8, (x + 4) / 4.
public enum Qwen3ASRMel {
    private static let lock = NSLock()
    private static var dftCache: (n: Int, cos: [Float], sin: [Float], window: [Float])?

    /// Features as `[frames][melBins]` (frame-major) and the frame count (`samples / hop`).
    public static func compute(_ samples: [Float], config: Qwen3ASRConfig) -> (features: [Float], frames: Int) {
        let nFFT = config.nFFT, hop = config.hop, nMel = config.melBins
        let bins = nFFT / 2 + 1
        let pad = nFFT / 2
        let n = samples.count
        let frames = n / hop
        guard frames > 0, n > pad else { return ([], 0) }
        let (cosM, sinM, window) = dft(nFFT)

        // Reflect padding ("reflect": the edge sample is not repeated).
        var padded = [Float](repeating: 0, count: n + 2 * pad)
        for i in 0..<pad { padded[i] = samples[pad - i] }
        for i in 0..<n { padded[pad + i] = samples[i] }
        for k in 0..<pad { padded[pad + n + k] = samples[max(0, n - 2 - k)] }

        // Windowed frames [frames][nFFT] -> real / imaginary parts [frames][bins] with two GEMMs (n_fft = 400 is not
        // a size vDSP's FFT supports).
        var framed = [Float](repeating: 0, count: frames * nFFT)
        padded.withUnsafeBufferPointer { src in
            framed.withUnsafeMutableBufferPointer { dst in
                for f in 0..<frames {
                    vDSP_vmul(src.baseAddress! + f * hop, 1, window, 1, dst.baseAddress! + f * nFFT, 1, vDSP_Length(nFFT))
                }
            }
        }
        var re = [Float](repeating: 0, count: frames * bins)
        var im = [Float](repeating: 0, count: frames * bins)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(frames), Int32(bins), Int32(nFFT), 1,
                    framed, Int32(nFFT), cosM, Int32(nFFT), 0, &re, Int32(bins))
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(frames), Int32(bins), Int32(nFFT), 1,
                    framed, Int32(nFFT), sinM, Int32(nFFT), 0, &im, Int32(bins))
        var power = [Float](repeating: 0, count: frames * bins)
        vDSP_vmma(re, 1, re, 1, im, 1, im, 1, &power, 1, vDSP_Length(frames * bins))   // re*re + im*im
        var mel = [Float](repeating: 0, count: frames * nMel)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(frames), Int32(nMel), Int32(bins), 1,
                    power, Int32(bins), config.melFilters, Int32(nMel), 0, &mel, Int32(nMel))
        mel.withUnsafeMutableBufferPointer { buf in
            let p = buf.baseAddress!
            let len = vDSP_Length(buf.count)
            var floor: Float = 1e-10
            vDSP_vthr(p, 1, &floor, p, 1, len)
            var count = Int32(buf.count)
            vvlog10f(p, p, &count)
            var mx: Float = -.infinity
            vDSP_maxv(p, 1, &mx, len)
            var lo = mx - 8
            vDSP_vthr(p, 1, &lo, p, 1, len)
            var quarter: Float = 0.25, one: Float = 1
            vDSP_vsmsa(p, 1, &quarter, &one, p, 1, len)   // (x + 4) / 4
        }
        return (mel, frames)
    }

    private static func dft(_ n: Int) -> ([Float], [Float], [Float]) {
        lock.lock(); defer { lock.unlock() }
        if let c = dftCache, c.n == n { return (c.cos, c.sin, c.window) }
        let bins = n / 2 + 1
        var c = [Float](repeating: 0, count: bins * n), s = [Float](repeating: 0, count: bins * n)
        for k in 0..<bins {
            for i in 0..<n {
                let a = 2 * Double.pi * Double((k * i) % n) / Double(n)
                c[k * n + i] = Float(cos(a))
                s[k * n + i] = Float(sin(a))
            }
        }
        let w = (0..<n).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(n))) }   // periodic Hann
        dftCache = (n, c, s, w)
        return (c, s, w)
    }
}

// MARK: - Byte-level BPE decoding

/// Turns Qwen token ids back into text (GPT-2 byte-level BPE; control tokens are skipped).
public final class Qwen3TokenDecoder: @unchecked Sendable {
    private let pieces: [[UInt8]?]   // nil = control token (skipped)

    init(metadata m: GGUFMetadata) throws {
        guard let tokens = m.stringArray("tokenizer.ggml.tokens") else { throw GGUFError.missingTensor("tokenizer.ggml.tokens") }
        let types = m.intArray("tokenizer.ggml.token_type") ?? [Int](repeating: 1, count: tokens.count)
        var byteOf = [UInt32: UInt8]()
        var bs = Array(33...126) + Array(161...172) + Array(174...255)
        var cs = bs
        var extra = 0
        for b in 0..<256 where !bs.contains(b) { bs.append(b); cs.append(256 + extra); extra += 1 }
        for (b, c) in zip(bs, cs) { byteOf[UInt32(c)] = UInt8(b) }
        pieces = tokens.enumerated().map { i, tok in
            switch types[i] {
            case 3, 5: return nil                                   // control / unused
            case 4: return Array(tok.utf8)                          // user-defined (e.g. <asr_text>): literal text
            default: return tok.unicodeScalars.compactMap { byteOf[$0.value] }
            }
        }
    }

    public func decode(_ ids: [Int32]) -> String {
        var bytes: [UInt8] = []
        for id in ids where id >= 0 && Int(id) < pieces.count {
            if let p = pieces[Int(id)] { bytes.append(contentsOf: p) }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// The transcript part of the model's answer (`language X<asr_text>text`); empty when it found no speech.
    public static func transcript(_ answer: String) -> (language: String?, text: String) {
        guard answer.hasPrefix("language "), let r = answer.range(of: "<asr_text>") else {
            return (nil, answer.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        let lang = String(answer[answer.index(answer.startIndex, offsetBy: 9)..<r.lowerBound]).trimmingCharacters(in: .whitespaces)
        let text = String(answer[r.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        return (lang, lang.lowercased() == "none" ? "" : text)
    }
}

// MARK: - Engine

public struct Qwen3ASRResult: Sendable {
    public var text: String
    public var language: String?
    public var tokenIDs: [Int32]
    public var audioTokens: Int
    public var melSeconds: Double
    public var encoderSeconds: Double
    public var prefillSeconds: Double
    public var decodeSeconds: Double
    public var totalSeconds: Double { melSeconds + encoderSeconds + prefillSeconds + decodeSeconds }

    public var summary: String {
        String(format: "%d audio tokens, mel %.0f ms, encoder %.0f ms, prefill %.0f ms, decode %d tok %.0f ms (%.1f tok/s)",
               audioTokens, melSeconds * 1000, encoderSeconds * 1000, prefillSeconds * 1000, tokenIDs.count,
               decodeSeconds * 1000, decodeSeconds > 0 ? Double(tokenIDs.count) / decodeSeconds : 0)
    }
}

public final class Qwen3ASRANEEngine: @unchecked Sendable {

    public struct Options: Sendable {
        public var decoderLUTBits: Int
        public var encoderLUTBits: Int
        public var headLUTBits: Int
        public var lutGroup: Int
        public var prefillChunk: Int
        /// Encoder layers per ANE program.
        public var encoderLayersPerProgram: Int
        public var cacheDirectory: URL?
        public init(decoderLUTBits: Int = 6, encoderLUTBits: Int = 8, headLUTBits: Int = 6, lutGroup: Int = 8,
                    prefillChunk: Int = 64, encoderLayersPerProgram: Int = 6, cacheDirectory: URL? = nil) {
            self.decoderLUTBits = decoderLUTBits
            self.encoderLUTBits = encoderLUTBits
            self.headLUTBits = headLUTBits
            self.lutGroup = lutGroup
            self.prefillChunk = prefillChunk
            self.encoderLayersPerProgram = encoderLayersPerProgram
            self.cacheDirectory = cacheDirectory
        }
    }

    public let gguf: GGUFFile
    public let config: Qwen3ASRConfig
    public let options: Options
    public let tokens: Qwen3TokenDecoder
    public private(set) var buildSeconds: Double = 0
    public private(set) var weightBytes: Int = 0
    /// Programs that aned already had compiled (the rest were compiled now).
    public private(set) var cachedPrograms = 0
    public private(set) var programCount = 0
    /// Exact-rescoring width for the ANE LM head (candidates re-scored against the Q8_0 rows on the CPU).
    public var headRescore = 16

    private var front: ANEProgram!
    private var blocks: [ANEProgram] = []
    private var programs: [ANEProgram] = []
    private var lmHead: ANEProgram!
    private let P: Int
    private let embInfo: GGUFTensorInfo

    // CPU state
    private var kCache: [UnsafeMutablePointer<Float>] = []
    private var vCache: [UnsafeMutablePointer<Float>] = []
    private var capacity = 0
    private var cosT: UnsafeMutablePointer<Float16>?
    private var sinT: UnsafeMutablePointer<Float16>?
    private var qScratch: UnsafeMutablePointer<Float>
    private var oScratch: UnsafeMutablePointer<Float>
    private var headF32: UnsafeMutablePointer<Float>

    public init(gguf: GGUFFile, options: Options = Options(), progress: ((String) -> Void)? = nil) throws {
        guard ane_rt_available() else {
            throw GemmaEngineError.inferenceFailed("AppleNeuralEngine private framework unavailable")
        }
        let cfg = try Qwen3ASRConfig.from(gguf.metadata)
        self.gguf = gguf
        self.config = cfg
        self.options = options
        self.tokens = try Qwen3TokenDecoder(metadata: gguf.metadata)
        self.P = options.prefillChunk
        self.embInfo = try gguf.tensor("token_embd.weight")
        self.qScratch = .allocate(capacity: options.prefillChunk * cfg.numHeads * cfg.headDim)
        self.oScratch = .allocate(capacity: options.prefillChunk * cfg.numHeads * cfg.headDim)
        self.headF32 = .allocate(capacity: cfg.vocabSize)
        try build(progress: progress)
    }

    deinit {
        for p in kCache { p.deallocate() }
        for p in vCache { p.deallocate() }
        cosT?.deallocate(); sinT?.deallocate()
        qScratch.deallocate(); oScratch.deallocate(); headF32.deallocate()
    }

    public var backendName: String {
        "ANE direct (decoder LUT\(options.decoderLUTBits), encoder LUT\(options.encoderLUTBits), head LUT\(options.headLUTBits))"
    }

    // MARK: Build

    private struct Pending {
        let tag: String
        let mil: String
        let blob: Data
        let procs: [[String: [NSNumber]]]
        let inputs: [String]
        let outputs: [String]
    }

    private func build(progress: ((String) -> Void)?) throws {
        let t0 = CFAbsoluteTimeGetCurrent()
        let cfg = config
        let key = "qwen3asr-d\(options.decoderLUTBits)e\(options.encoderLUTBits)h\(options.headLUTBits)g\(options.lutGroup)p\(P)-\(gguf.fileSize)-v2"
        let cacheDir = options.cacheDirectory?.appendingPathComponent(key, isDirectory: true)
        if let cacheDir { try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true) }

        func prepare(_ tag: String, procs: [[String: [NSNumber]]], inputs: [String], outputs: [String],
                     build: (_ dryRun: Bool) throws -> (String, Data?)) throws -> Pending {
            // The palettised weights are cached; the MIL text is regenerated from a sizes-only dry run.
            let blobURL = cacheDir?.appendingPathComponent("\(tag).bin")
            if let blobURL, let b = try? Data(contentsOf: blobURL, options: .alwaysMapped) {
                return Pending(tag: tag, mil: try build(true).0, blob: b, procs: procs, inputs: inputs, outputs: outputs)
            }
            progress?("quantising \(tag)")
            let built = try build(false)
            var blob = built.1 ?? Data()
            if let blobURL, (try? blob.write(to: blobURL)) != nil, let mapped = try? Data(contentsOf: blobURL, options: .alwaysMapped) {
                blob = mapped
            }
            return Pending(tag: tag, mil: built.0, blob: blob, procs: procs, inputs: inputs, outputs: outputs)
        }
        func bytes(_ shape: [Int]) -> NSNumber { NSNumber(value: shape.reduce(2, *)) }

        var pending: [Pending] = []
        // Audio encoder: front end, then groups of layers (the last one also runs ln_post + projection).
        let C = cfg.chunksPerWindow, W = cfg.windowTokens, D = cfg.audioHidden
        pending.append(try prepare("enc-front", procs: [["in": [bytes([C, 1, cfg.chunkFrames, cfg.melBins])], "out": [bytes([1, W, D])]]],
                                   inputs: ["a_mel"], outputs: ["o0_x"]) { dry in try buildFront(dryRun: dry) })
        let per = max(1, options.encoderLayersPerProgram)
        let groups = stride(from: 0, to: cfg.audioLayers, by: per).map { ($0, min(cfg.audioLayers, $0 + per)) }
        for (gi, g) in groups.enumerated() {
            let last = gi == groups.count - 1
            let outC = last ? cfg.audioOutput : D
            pending.append(try prepare("enc-\(g.0)-\(g.1)", procs: [["in": [bytes([1, W, D]), bytes([1, 1, 1, Self.maskWidth(W)])],
                                                                       "out": [bytes([1, W, outC])]]],
                                       inputs: ["a_x", "b_mask"], outputs: ["o0_x"]) { dry in
                try buildEncoderBlock(layers: g.0..<g.1, post: last, dryRun: dry)
            })
        }
        // Decoder boundary programs and the LM head.
        for b in 0...cfg.numLayers {
            let io = boundaryIO(b)
            let procs: [[String: [NSNumber]]] = [1, P].map { S in
                ["in": io.inputs.map { NSNumber(value: S * $0.1 * 2) }, "out": io.outputs.map { NSNumber(value: S * $0.1 * 2) }]
            }
            pending.append(try prepare("b\(b)", procs: procs, inputs: io.inputs.map { $0.0 }, outputs: io.outputs.map { $0.0 }) { dry in
                try buildBoundary(b, dryRun: dry)
            })
        }
        pending.append(try prepare("lmhead", procs: [["in": [bytes([1, 1, cfg.hidden])], "out": [bytes([1, 1, cfg.vocabSize])]]],
                                   inputs: ["a_x"], outputs: ["o0_logits"]) { dry in try buildLMHead(dryRun: dry) })

        var loaded = [Result<ANEProgram, Error>?](repeating: nil, count: pending.count)
        let lock = NSLock()
        var finished = 0
        DispatchQueue.concurrentPerform(iterations: pending.count) { i in
            let item = pending[i]
            let r = Result { try ANEProgram.compileMIL(item.mil, weights: item.blob, procedures: item.procs.map { $0 as [AnyHashable: Any] }) }
            lock.lock()
            loaded[i] = r
            finished += 1
            let n = finished
            lock.unlock()
            progress?("ANE program \(n)/\(pending.count)")
        }
        for (i, item) in pending.enumerated() {
            var result = loaded[i]!
            // A compile can fail transiently when many run at once (aned reported "cannot remove <model hash>" once on
            // a Mac): retry it alone.
            var attempt = 0
            while case .failure = result, attempt < 2 {
                attempt += 1
                Thread.sleep(forTimeInterval: 0.5)
                result = Result { try ANEProgram.compileMIL(item.mil, weights: item.blob, procedures: item.procs.map { $0 as [AnyHashable: Any] }) }
            }
            let prog: ANEProgram
            switch result {
            case .success(let p): prog = p
            case .failure(let e): throw GemmaEngineError.inferenceFailed("ANE compile \(item.tag) failed: \(e.localizedDescription)")
            }
            for p in 0..<item.procs.count {
                let inN = prog.inputNames(Int32(p)), outN = prog.outputNames(Int32(p)).map { $0.replacingOccurrences(of: "@output", with: "") }
                guard inN == item.inputs, outN == item.outputs else {
                    throw GemmaEngineError.inferenceFailed("ANE program \(item.tag) I/O order mismatch: \(inN) / \(outN)")
                }
            }
            weightBytes += item.blob.count
            if prog.loadedFromCache { cachedPrograms += 1 }
            if item.tag == "enc-front" { front = prog }
            else if item.tag.hasPrefix("enc-") { blocks.append(prog) }
            else if item.tag == "lmhead" { lmHead = prog }
            else { programs.append(prog) }
        }
        programCount = pending.count
        buildSeconds = CFAbsoluteTimeGetCurrent() - t0
    }

    private func makeBank(_ bits: Int, dryRun: Bool) -> WeightBank {
        let bank = WeightBank(bits: bits, group: options.lutGroup, dryRun: dryRun)
        bank.iterations = 300
        bank.companded = true
        return bank
    }

    /// A linear weight as LUT palettes, or dense fp16 when the bank's bits are 16 (precision experiments).
    private func weight(_ bank: WeightBank, _ name: String, _ tensor: String) throws {
        if bank.bits >= 16 {
            let info = try gguf.tensor(tensor)
            let g = gguf
            try bank.dense(name, shape: [info.elementCount / info.dimensions[0], info.dimensions[0]]) { try WeightBank.floats(g, tensor) }
        } else {
            try bank.lut(name, gguf, tensor)
        }
    }

    /// The encoder mask input is padded to a multiple of 32 fp16 (64-byte rows); the program slices it.
    private static func maskWidth(_ w: Int) -> Int { (w + 31) / 32 * 32 }

    private func buildFront(dryRun: Bool) throws -> (String, Data?) {
        let cfg = config
        let C = cfg.chunksPerWindow, F = cfg.chunkFrames, M = cfg.melBins, Dh = cfg.downsampleHidden
        let T1 = Qwen3ASRConfig.convOut(F), M1 = Qwen3ASRConfig.convOut(M)
        let T2 = Qwen3ASRConfig.convOut(T1), M2 = Qwen3ASRConfig.convOut(M1)
        let T3 = Qwen3ASRConfig.convOut(T2), M3 = Qwen3ASRConfig.convOut(M2)
        let D = cfg.audioHidden
        let bank = makeBank(options.encoderLUTBits, dryRun: dryRun)
        let g = gguf
        // The input is [chunk, 1, time, mel] (mel last, so rows are 64-byte aligned); the checkpoint's kernels are
        // [out, in, mel, time], so their last two axes are swapped.
        for (i, cin) in [(1, 1), (2, Dh), (3, Dh)] {
            try bank.dense("w\(i)", shape: [Dh, cin, 3, 3]) {
                let w = try WeightBank.floats(g, "a.conv\(i).weight")
                var t = [Float](repeating: 0, count: w.count)
                for o in 0..<(Dh * cin) { for a in 0..<3 { for b in 0..<3 { t[o * 9 + a * 3 + b] = w[o * 9 + b * 3 + a] } } }
                return t
            }
            try bank.vec("c\(i)", gguf, "a.conv\(i).bias")
        }
        try weight(bank, "wout", "a.conv_out.weight")
        try bank.dense("pos", shape: [T3, D]) { Self.sinusoids(length: T3, channels: D) }

        var mil = MILWriter.header
        let w = MILWriter(S: 1)
        w.raw("    func main<ios18>(tensor<fp16, [\(C), 1, \(F), \(M)]> a_mel) {\n")
        for d in bank.declarations { w.line(d) }
        let pad = [1, 1, 1, 1], st = [2, 2]
        let c1 = w.geluExact(w.conv("a_mel", weight: "w1", bias: "c1", strides: st, pad: pad, [C, Dh, T1, M1]), [C, Dh, T1, M1])
        let c2 = w.geluExact(w.conv(c1, weight: "w2", bias: "c2", strides: st, pad: pad, [C, Dh, T2, M2]), [C, Dh, T2, M2])
        let c3 = w.geluExact(w.conv(c2, weight: "w3", bias: "c3", strides: st, pad: pad, [C, Dh, T3, M3]), [C, Dh, T3, M3])
        let tr = w.transpose(c3, perm: [0, 2, 1, 3], [C, T3, Dh, M3])          // [chunk, time, channel, mel]
        let flat = w.reshape(tr, [C, T3, Dh * M3])                             // feature = channel * mel + f
        let lin = w.linear(flat, "wout", [C, T3, D])
        let x = w.add(lin, "pos", [C, T3, D])
        _ = w.reshape(x, [1, C * T3, D], name: "o0_x")
        w.raw("    } -> (o0_x);\n")
        mil += w.text + "}\n"
        return (mil, dryRun ? nil : bank.finish())
    }

    private func buildEncoderBlock(layers: Range<Int>, post: Bool, dryRun: Bool) throws -> (String, Data?) {
        let cfg = config
        let W = cfg.windowTokens, D = cfg.audioHidden, nH = cfg.audioHeads, hd = D / nH, F = cfg.audioFFN
        let MW = Self.maskWidth(W)
        let eps = cfg.audioLNEps
        let bank = makeBank(options.encoderLUTBits, dryRun: dryRun)
        for l in layers {
            let p = "a.blk.\(l)."
            try bank.vec("g1_\(l)", gguf, p + "attn_norm.weight"); try bank.vec("e1_\(l)", gguf, p + "attn_norm.bias")
            for (n, t) in [("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"), ("o", "attn_out"), ("f1", "ffn_up"), ("f2", "ffn_down")] {
                try weight(bank, "w\(n)_\(l)", p + t + ".weight")
                try bank.vec("b\(n)_\(l)", gguf, p + t + ".bias")
            }
            try bank.vec("g2_\(l)", gguf, p + "ffn_norm.weight"); try bank.vec("e2_\(l)", gguf, p + "ffn_norm.bias")
        }
        if post {
            try bank.vec("gp", gguf, "a.ln_post.weight"); try bank.vec("ep", gguf, "a.ln_post.bias")
            try weight(bank, "wp1", "a.proj1.weight"); try bank.vec("bp1", gguf, "a.proj1.bias")
            try weight(bank, "wp2", "a.proj2.weight"); try bank.vec("bp2", gguf, "a.proj2.bias")
        }
        var mil = MILWriter.header
        let w = MILWriter(S: W)
        w.raw("    func main<ios18>(tensor<fp16, [1, \(W), \(D)]> a_x, tensor<fp16, [1, 1, 1, \(MW)]> b_mask) {\n")
        for d in bank.declarations { w.line(d) }
        let mask = w.slice("b_mask", begin: [0, 0, 0, 0], end: [1, 1, 1, W], [1, 1, 1, W])
        let scale = w.scalar(1 / sqrtf(Float(hd)))
        var x = "a_x"
        for l in layers {
            let xn = w.layerNorm(x, dims: [1, W, D], gamma: "g1_\(l)", beta: "e1_\(l)", eps: eps)
            func heads(_ t: String) -> String { w.transpose(w.reshape(t, [1, W, nH, hd]), perm: [0, 2, 1, 3], [1, nH, W, hd]) }
            let q = heads(w.mul(w.linear(xn, "wq_\(l)", bias: "bq_\(l)", [1, W, D]), scale, [1, W, D]))
            let k = heads(w.linear(xn, "wk_\(l)", bias: "bk_\(l)", [1, W, D]))
            let v = heads(w.linear(xn, "wv_\(l)", bias: "bv_\(l)", [1, W, D]))
            let s = w.add(w.matmul(q, k, transposeY: true, [1, nH, W, W]), mask, [1, nH, W, W])
            let pr = w.softmax(s, axis: -1, [1, nH, W, W])
            let o = w.reshape(w.transpose(w.matmul(pr, v, [1, nH, W, hd]), perm: [0, 2, 1, 3], [1, W, nH, hd]), [1, W, D])
            x = w.add(x, w.linear(o, "wo_\(l)", bias: "bo_\(l)", [1, W, D]), [1, W, D])
            let xn2 = w.layerNorm(x, dims: [1, W, D], gamma: "g2_\(l)", beta: "e2_\(l)", eps: eps)
            let h = w.geluExact(w.linear(xn2, "wf1_\(l)", bias: "bf1_\(l)", [1, W, F]), [1, W, F])
            x = w.add(x, w.linear(h, "wf2_\(l)", bias: "bf2_\(l)", [1, W, D]), [1, W, D], name: (!post && l == layers.upperBound - 1) ? "o0_x" : nil)
        }
        if post {
            let xn = w.layerNorm(x, dims: [1, W, D], gamma: "gp", beta: "ep", eps: eps)
            let p1 = w.geluExact(w.linear(xn, "wp1", bias: "bp1", [1, W, D]), [1, W, D])
            _ = w.linear(p1, "wp2", bias: "bp2", [1, W, cfg.audioOutput], name: "o0_x")
        }
        w.raw("    } -> (o0_x);\n")
        mil += w.text + "}\n"
        return (mil, dryRun ? nil : bank.finish())
    }

    /// (symbol, channels) per input/output of decoder boundary program `b`.
    private func boundaryIO(_ b: Int) -> (inputs: [(String, Int)], outputs: [(String, Int)]) {
        let cfg = config
        let H = cfg.hidden, hd = cfg.headDim, nL = cfg.numLayers
        var ins: [(String, Int)] = [("a_h", H)]
        var outs: [(String, Int)] = []
        if b >= 1 { ins.append(("b_attn", cfg.numHeads * hd)) }
        if b < nL {
            ins.append(("d_cos", hd))
            ins.append(("e_sin", hd))
            if b >= 1 { outs.append(("o0_h", H)) }
            outs.append(("o1_q", cfg.numHeads * hd))
            outs.append(("o2_k", cfg.numKVHeads * hd))
            outs.append(("o3_v", cfg.numKVHeads * hd))
        } else {
            outs.append(("o0_x", H))
        }
        return (ins, outs)
    }

    private func buildBoundary(_ b: Int, dryRun: Bool) throws -> (String, Data?) {
        let cfg = config
        let H = cfg.hidden, nH = cfg.numHeads, nKV = cfg.numKVHeads, hd = cfg.headDim, nL = cfg.numLayers
        let eps = cfg.rmsEps
        let bank = makeBank(options.decoderLUTBits, dryRun: dryRun)
        if b >= 1 {
            let l = b - 1
            try weight(bank, "wo", "blk.\(l).attn_output.weight")
            try bank.vec("n_ffn", gguf, "blk.\(l).ffn_norm.weight")
            try weight(bank, "wg", "blk.\(l).ffn_gate.weight")
            try weight(bank, "wu", "blk.\(l).ffn_up.weight")
            try weight(bank, "wd", "blk.\(l).ffn_down.weight")
        }
        if b < nL {
            try bank.vec("n_at", gguf, "blk.\(b).attn_norm.weight")
            try weight(bank, "wq", "blk.\(b).attn_q.weight")
            try weight(bank, "wk", "blk.\(b).attn_k.weight")
            try weight(bank, "wv", "blk.\(b).attn_v.weight")
            try bank.vec("n_q", gguf, "blk.\(b).attn_q_norm.weight")
            try bank.vec("n_k", gguf, "blk.\(b).attn_k_norm.weight")
        } else {
            try bank.vec("n_out", gguf, "output_norm.weight")
        }
        let io = boundaryIO(b)
        var mil = MILWriter.header
        for (fname, S) in [("p0_decode", 1), ("p1_prefill", P)] {
            let w = MILWriter(S: S)
            let sig = io.inputs.map { name, c -> String in
                (name == "d_cos" || name == "e_sin") ? "tensor<fp16, [1, \(S), 1, \(c)]> \(name)" : "tensor<fp16, [1, \(S), \(c)]> \(name)"
            }.joined(separator: ", ")
            w.raw("    func \(fname)<ios18>(\(sig)) {\n")
            for d in bank.declarations { w.line(d) }
            var hIn = "a_h"
            if b >= 1 {
                let o = w.linear("b_attn", "wo", [1, S, H])
                let h1 = w.add("a_h", o, [1, S, H])
                let x = w.rmsNorm(h1, dims: [1, S, H], weight: "n_ffn", eps: eps)
                let g = w.linear(x, "wg", [1, S, cfg.ffn])
                let u = w.linear(x, "wu", [1, S, cfg.ffn])
                let a = w.mul(w.silu(g, [1, S, cfg.ffn]), u, [1, S, cfg.ffn])
                let dn = w.linear(a, "wd", [1, S, H])
                hIn = w.add(h1, dn, [1, S, H], name: b < nL ? "o0_h" : nil)
            }
            var outs: [String] = []
            if b < nL {
                if b >= 1 { outs.append("o0_h") }
                let x = w.rmsNorm(hIn, dims: [1, S, H], weight: "n_at", eps: eps)
                let q = w.reshape(w.linear(x, "wq", [1, S, nH * hd]), [1, S, nH, hd])
                let qr = w.rope(w.rmsNorm(q, dims: [1, S, nH, hd], weight: "n_q", eps: eps), dims: [1, S, nH, hd], cos: "d_cos", sin: "e_sin")
                _ = w.reshape(qr, [1, S, nH * hd], name: "o1_q")
                let k = w.reshape(w.linear(x, "wk", [1, S, nKV * hd]), [1, S, nKV, hd])
                let kr = w.rope(w.rmsNorm(k, dims: [1, S, nKV, hd], weight: "n_k", eps: eps), dims: [1, S, nKV, hd], cos: "d_cos", sin: "e_sin")
                _ = w.reshape(kr, [1, S, nKV * hd], name: "o2_k")
                _ = w.linear(x, "wv", [1, S, nKV * hd], name: "o3_v")
                outs += ["o1_q", "o2_k", "o3_v"]
            } else {
                _ = w.rmsNorm(hIn, dims: [1, S, H], weight: "n_out", eps: eps, name: "o0_x")
                outs.append("o0_x")
            }
            w.raw("    } -> (\(outs.joined(separator: ", ")));\n")
            mil += w.text
        }
        mil += "}\n"
        return (mil, dryRun ? nil : bank.finish())
    }

    private func buildLMHead(dryRun: Bool) throws -> (String, Data?) {
        let cfg = config
        let H = cfg.hidden, Vocab = cfg.vocabSize
        let chunk = 16384
        let bank = makeBank(options.headLUTBits, dryRun: dryRun)
        let nChunks = (Vocab + chunk - 1) / chunk
        for c in 0..<nChunks {
            let lo = c * chunk, hi = min(Vocab, lo + chunk)
            try bank.lut("e\(c)", gguf, "token_embd.weight", rowStart: lo, rowCount: hi - lo)
        }
        var mil = MILWriter.header
        let w = MILWriter(S: 1)
        w.raw("    func main<ios18>(tensor<fp16, [1, 1, \(H)]> a_x) {\n")
        for d in bank.declarations { w.line(d) }
        let parts = (0..<nChunks).map { w.linear("a_x", "e\($0)", [1, 1, bank.rows["e\($0)"]!]) }
        _ = w.concatLast(parts, [1, 1, Vocab], name: "o0_logits")
        w.raw("    } -> (o0_logits);\n")
        mil += w.text + "}\n"
        return (mil, dryRun ? nil : bank.finish())
    }

    static func sinusoids(length: Int, channels: Int, maxTimescale: Double = 10000) -> [Float] {
        let half = channels / 2
        let inc = log(maxTimescale) / Double(half - 1)
        var out = [Float](repeating: 0, count: length * channels)
        for p in 0..<length {
            for j in 0..<half {
                let t = Double(p) * exp(-inc * Double(j))
                out[p * channels + j] = Float(sin(t))
                out[p * channels + half + j] = Float(cos(t))
            }
        }
        return out
    }

    // MARK: Encoder

    /// Audio embeddings `[tokens][audioOutput]` (fp16) for log-mel `features` (`[frames][melBins]`).
    public func encode(features: [Float], frames: Int) throws -> [Float16] {
        let cfg = config
        let F = cfg.chunkFrames, M = cfg.melBins, C = cfg.chunksPerWindow, W = cfg.windowTokens, Dout = cfg.audioOutput
        let total = cfg.audioTokenCount(frames: frames)
        var out = [Float16](repeating: 0, count: total * Dout)
        let chunks = (frames + F - 1) / F
        var produced = 0
        var c0 = 0
        while c0 < chunks {
            let nC = min(C, chunks - c0)
            // Front end input: nC chunks of up to F frames, zero padded.
            let mel = front.inputPtr(0, procedure: 0).assumingMemoryBound(to: Float16.self)
            memset(mel, 0, C * F * M * 2)
            var valid = 0
            for c in 0..<nC {
                let f0 = (c0 + c) * F
                let nF = min(F, frames - f0)
                features.withUnsafeBufferPointer { src in
                    for f in 0..<nF {
                        let s = src.baseAddress!.advanced(by: (f0 + f) * M)
                        let d = mel.advanced(by: (c * F + f) * M)
                        for j in 0..<M { d[j] = Float16(s[j]) }
                    }
                }
                valid += cfg.audioTokenCount(frames: nF)
            }
            try front.eval(0)
            // Chain the encoder groups; padded tokens (after `valid`) are masked out of every attention.
            var src = front.outputPtr(0, procedure: 0)
            for (gi, prog) in blocks.enumerated() {
                memcpy(prog.inputPtr(0, procedure: 0), src, W * cfg.audioHidden * 2)
                let m = prog.inputPtr(1, procedure: 0).assumingMemoryBound(to: Float16.self)
                for j in 0..<Self.maskWidth(W) { m[j] = j < valid ? 0 : Float16(-30000) }
                try prog.eval(0)
                src = prog.outputPtr(0, procedure: 0)
                _ = gi
            }
            let n = min(valid, total - produced)
            out.withUnsafeMutableBufferPointer { dst in
                _ = memcpy(dst.baseAddress!.advanced(by: produced * Dout), src, n * Dout * 2)
            }
            produced += n
            c0 += nC
        }
        return out
    }

    /// Runs encoder program `index` on `input` (`[valid][audioHidden]`, padded to the window) and returns its output rows.
    public func encoderBlock(_ index: Int, input: [Float], valid: Int) throws -> [Float] {
        let cfg = config
        let W = cfg.windowTokens, D = cfg.audioHidden
        let prog = blocks[index]
        let x = prog.inputPtr(0, procedure: 0).assumingMemoryBound(to: Float16.self)
        memset(x, 0, W * D * 2)
        for i in 0..<min(input.count, W * D) { x[i] = Float16(input[i]) }
        let m = prog.inputPtr(1, procedure: 0).assumingMemoryBound(to: Float16.self)
        for j in 0..<Self.maskWidth(W) { m[j] = j < valid ? 0 : Float16(-30000) }
        try prog.eval(0)
        let outC = index == blocks.count - 1 ? cfg.audioOutput : D
        let o = prog.outputPtr(0, procedure: 0).assumingMemoryBound(to: Float16.self)
        return (0..<(valid * outC)).map { Float(o[$0]) }
    }

    public var encoderProgramCount: Int { blocks.count }

    /// Runs only the conv front end on the first window (tests).
    public func frontEnd(features: [Float], frames: Int) throws -> [Float16] {
        let cfg = config
        let F = cfg.chunkFrames, M = cfg.melBins, C = cfg.chunksPerWindow, W = cfg.windowTokens, D = cfg.audioHidden
        let mel = front.inputPtr(0, procedure: 0).assumingMemoryBound(to: Float16.self)
        memset(mel, 0, C * F * M * 2)
        for f in 0..<min(frames, C * F) { for j in 0..<M { mel[f * M + j] = Float16(features[f * M + j]) } }
        try front.eval(0)
        let o = front.outputPtr(0, procedure: 0).assumingMemoryBound(to: Float16.self)
        return Array(UnsafeBufferPointer(start: o, count: W * D))
    }

    // MARK: Transcription

    /// Transcribes 16 kHz mono `samples`. `language` "Chinese" forces Chinese output (nil lets the model decide).
    public func transcribe(_ samples: [Float], language: String? = nil, maxNewTokens: Int? = nil,
                           shouldCancel: (() -> Bool)? = nil) throws -> Qwen3ASRResult {
        let cfg = config
        let t0 = CFAbsoluteTimeGetCurrent()
        let (features, frames) = Qwen3ASRMel.compute(samples, config: cfg)
        let t1 = CFAbsoluteTimeGetCurrent()
        guard frames > 0 else {
            return Qwen3ASRResult(text: "", language: nil, tokenIDs: [], audioTokens: 0, melSeconds: t1 - t0, encoderSeconds: 0, prefillSeconds: 0, decodeSeconds: 0)
        }
        let audio = try encode(features: features, frames: frames)
        let nAudio = audio.count / cfg.audioOutput
        let t2 = CFAbsoluteTimeGetCurrent()

        var prompt = cfg.promptPrefix + [Int32](repeating: cfg.audioTokenID, count: nAudio) + cfg.promptSuffix
        if language?.lowercased() == "chinese" { prompt += cfg.languageChinese }
        let audioStart = cfg.promptPrefix.count
        // 13 audio tokens a second; speech stays well under that many text tokens a second.
        let maxNew = maxNewTokens ?? (nAudio + 24)
        try ensureCapacity(prompt.count + maxNew + 1)

        // Prefill: text rows from the embedding table, audio rows from the encoder.
        var next: Int32 = 0
        var idx = 0
        while idx < prompt.count {
            let n = min(P, prompt.count - idx)
            let base = idx
            next = try forward(rows: n, p0: idx, needToken: idx + n == prompt.count) { r, dst in
                let pos = base + r
                if pos >= audioStart && pos < audioStart + nAudio {
                    audio.withUnsafeBufferPointer { a in
                        _ = memcpy(dst, a.baseAddress!.advanced(by: (pos - audioStart) * cfg.audioOutput), cfg.hidden * 2)
                    }
                } else {
                    self.embed(prompt[pos], into: dst)
                }
            } ?? 0
            idx += n
        }
        let t3 = CFAbsoluteTimeGetCurrent()

        var out: [Int32] = []
        var pos = prompt.count
        while !cfg.eosTokenIDs.contains(next) && out.count < maxNew && pos < capacity - 1 {
            if shouldCancel?() == true { break }
            out.append(next)
            let tok = next
            next = try forward(rows: 1, p0: pos, needToken: true) { _, dst in self.embed(tok, into: dst) } ?? 0
            pos += 1
        }
        let t4 = CFAbsoluteTimeGetCurrent()
        if capacity > 640 { releaseKV() }   // a very long sentence: give the memory back
        var answer = tokens.decode(out)
        if language?.lowercased() == "chinese" { answer = "language Chinese<asr_text>" + answer }
        let parsed = Qwen3TokenDecoder.transcript(answer)
        return Qwen3ASRResult(text: parsed.text, language: parsed.language, tokenIDs: out, audioTokens: nAudio,
                              melSeconds: t1 - t0, encoderSeconds: t2 - t1, prefillSeconds: t3 - t2, decodeSeconds: t4 - t3)
    }

    /// Runs every program once so the first sentence does not pay for the first evaluations.
    public func warmUp() throws {
        _ = try transcribe([Float](repeating: 0, count: 16000), maxNewTokens: 2)
    }

    private func embed(_ token: Int32, into dst: UnsafeMutablePointer<Float16>) {
        let H = config.hidden
        let row = UnsafeMutablePointer<Float>.allocate(capacity: H); defer { row.deallocate() }
        try? gguf.dequantizeRow(tensor: embInfo, row: Int(max(0, min(Int32(config.vocabSize - 1), token))), into: row)
        for j in 0..<H { dst[j] = Float16(row[j]) }
    }

    private func releaseKV() {
        for p in kCache { p.deallocate() }
        for p in vCache { p.deallocate() }
        kCache = []; vCache = []
        cosT?.deallocate(); sinT?.deallocate()
        cosT = nil; sinT = nil
        capacity = 0
    }

    private func ensureCapacity(_ needed: Int) throws {
        guard needed > capacity else { return }
        let cfg = config
        let cap = (needed + 63) / 64 * 64
        let kvD = cfg.numKVHeads * cfg.headDim
        for p in kCache { p.deallocate() }
        for p in vCache { p.deallocate() }
        kCache = (0..<cfg.numLayers).map { _ in UnsafeMutablePointer<Float>.allocate(capacity: cap * kvD) }
        vCache = (0..<cfg.numLayers).map { _ in UnsafeMutablePointer<Float>.allocate(capacity: cap * kvD) }
        cosT?.deallocate(); sinT?.deallocate()
        let hd = cfg.headDim, half = hd / 2
        let c = UnsafeMutablePointer<Float16>.allocate(capacity: cap * hd)
        let s = UnsafeMutablePointer<Float16>.allocate(capacity: cap * hd)
        for p in 0..<cap {
            for j in 0..<half {
                let ang = Double(p) * pow(Double(cfg.ropeBase), -2.0 * Double(j) / Double(hd))
                let cv = Float16(Float(cos(ang))), sv = Float16(Float(sin(ang)))
                c[p * hd + j] = cv; c[p * hd + j + half] = cv
                s[p * hd + j] = sv; s[p * hd + j + half] = sv
            }
        }
        cosT = c; sinT = s
        capacity = cap
    }

    /// Runs `n` (<= P) rows at positions `p0 ..< p0 + n` through the decoder; `fill(r, dst)` writes row r's input
    /// embedding. Returns the greedy next token of the last row when `needToken`.
    private func forward(rows n: Int, p0: Int, needToken: Bool,
                         fill: (Int, UnsafeMutablePointer<Float16>) -> Void) throws -> Int32? {
        let cfg = config
        let H = cfg.hidden, nL = cfg.numLayers, hd = cfg.headDim
        let proc: Int32 = n == 1 ? 0 : 1
        let S = n == 1 ? 1 : P
        let pro = programs[0]
        let hIn = pro.inputPtr(0, procedure: proc).assumingMemoryBound(to: Float16.self)
        memset(hIn, 0, S * H * 2)
        for r in 0..<n { fill(r, hIn.advanced(by: r * H)) }
        fillRoPE(pro, proc: proc, base: 1, p0: p0, n: n, S: S)
        try pro.eval(proc)
        storeKV(pro, proc: proc, layer: 0, kIndex: 1, p0: p0, n: n)

        var prev = pro
        var prevQ = 0, prevH = -1
        for b in 1...nL {
            let prog = programs[b]
            let q = prev.outputPtr(Int32(prevQ), procedure: proc).assumingMemoryBound(to: Float16.self)
            attention(layer: b - 1, q: q, rows: n, p0: p0, out: prog.inputPtr(1, procedure: proc).assumingMemoryBound(to: Float16.self), S: S)
            let hSrc = prevH < 0 ? prev.inputPtr(0, procedure: proc) : prev.outputPtr(Int32(prevH), procedure: proc)
            memcpy(prog.inputPtr(0, procedure: proc), hSrc, S * H * 2)
            if b < nL { fillRoPE(prog, proc: proc, base: 2, p0: p0, n: n, S: S) }
            try prog.eval(proc)
            if b < nL {
                storeKV(prog, proc: proc, layer: b, kIndex: 2, p0: p0, n: n)
                prev = prog
                prevQ = 1
                prevH = 0
            }
        }
        _ = hd
        guard needToken else { return nil }
        // LM head on the last row, then exact rescoring of its best candidates against the Q8_0 embedding rows.
        let x = programs[nL].outputPtr(0, procedure: proc).advanced(by: (n - 1) * H * 2)
        memcpy(lmHead.inputPtr(0, procedure: 0), x, H * 2)
        try lmHead.eval(0)
        let logits = lmHead.outputPtr(0, procedure: 0).assumingMemoryBound(to: Float16.self)
        let V = cfg.vocabSize
        Self.f16ToF32(logits, headF32, V)
        let R = max(1, headRescore)
        var top = [(Float, Int)](repeating: (-.infinity, 0), count: R)
        var floorV: Float = -.infinity
        for j in 0..<V where headF32[j] > floorV {
            var k = R - 1
            while k > 0 && top[k - 1].0 < headF32[j] { top[k] = top[k - 1]; k -= 1 }
            top[k] = (headF32[j], j)
            floorV = top[R - 1].0
        }
        let h = UnsafeMutablePointer<Float>.allocate(capacity: H); defer { h.deallocate() }
        let row = UnsafeMutablePointer<Float>.allocate(capacity: H); defer { row.deallocate() }
        Self.f16ToF32(x.assumingMemoryBound(to: Float16.self), h, H)
        var best = Int32(top[0].1)
        var bestV: Float = -.infinity
        for (_, j) in top {
            try gguf.dequantizeRow(tensor: embInfo, row: j, into: row)
            var v: Float = 0
            vDSP_dotpr(row, 1, h, 1, &v, vDSP_Length(H))
            if v > bestV { bestV = v; best = Int32(j) }
        }
        return best
    }

    private func fillRoPE(_ prog: ANEProgram, proc: Int32, base: Int, p0: Int, n: Int, S: Int) {
        let hd = config.headDim
        let c = prog.inputPtr(Int32(base), procedure: proc).assumingMemoryBound(to: Float16.self)
        let s = prog.inputPtr(Int32(base + 1), procedure: proc).assumingMemoryBound(to: Float16.self)
        memcpy(c, cosT!.advanced(by: p0 * hd), n * hd * 2)
        memcpy(s, sinT!.advanced(by: p0 * hd), n * hd * 2)
    }

    private func storeKV(_ prog: ANEProgram, proc: Int32, layer: Int, kIndex: Int, p0: Int, n: Int) {
        let kvD = config.numKVHeads * config.headDim
        let k = prog.outputPtr(Int32(kIndex), procedure: proc).assumingMemoryBound(to: Float16.self)
        let v = prog.outputPtr(Int32(kIndex + 1), procedure: proc).assumingMemoryBound(to: Float16.self)
        Self.f16ToF32(k, kCache[layer].advanced(by: p0 * kvD), n * kvD)
        Self.f16ToF32(v, vCache[layer].advanced(by: p0 * kvD), n * kvD)
    }

    /// Grouped-query attention for `rows` query rows at positions `p0 ..< p0 + rows` (causal), fp32 via cblas.
    private func attention(layer l: Int, q: UnsafePointer<Float16>, rows n: Int, p0: Int, out: UnsafeMutablePointer<Float16>, S: Int) {
        let cfg = config
        let nH = cfg.numHeads, nKV = cfg.numKVHeads, hd = cfg.headDim
        let G = nH / nKV
        let kvD = nKV * hd
        let Tk = p0 + n
        let scale = 1 / sqrtf(Float(hd))
        let qf = qScratch, of = oScratch
        Self.f16ToF32(q, qf, n * nH * hd)
        let K = kCache[l], Vc = vCache[l]
        DispatchQueue.concurrentPerform(iterations: nKV) { g in
            let M = n * G
            let qg = UnsafeMutablePointer<Float>.allocate(capacity: M * hd)
            let sc = UnsafeMutablePointer<Float>.allocate(capacity: M * Tk)
            let og = UnsafeMutablePointer<Float>.allocate(capacity: M * hd)
            defer { qg.deallocate(); sc.deallocate(); og.deallocate() }
            for r in 0..<n { memcpy(qg.advanced(by: r * G * hd), qf.advanced(by: (r * nH + g * G) * hd), G * hd * 4) }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(M), Int32(Tk), Int32(hd), scale,
                        qg, Int32(hd), K.advanced(by: g * hd), Int32(kvD), 0, sc, Int32(Tk))
            for i in 0..<M {
                let qpos = p0 + i / G
                let row = sc.advanced(by: i * Tk)
                let cnt = vDSP_Length(qpos + 1)
                var mx: Float = 0
                vDSP_maxv(row, 1, &mx, cnt)
                var neg = -mx
                vDSP_vsadd(row, 1, &neg, row, 1, cnt)
                var n32 = Int32(cnt)
                vvexpf(row, row, &n32)
                var sum: Float = 0
                vDSP_sve(row, 1, &sum, cnt)
                var inv = 1 / sum
                vDSP_vsmul(row, 1, &inv, row, 1, cnt)
                if qpos + 1 < Tk { vDSP_vclr(row.advanced(by: qpos + 1), 1, vDSP_Length(Tk - qpos - 1)) }
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(M), Int32(hd), Int32(Tk), 1,
                        sc, Int32(Tk), Vc.advanced(by: g * hd), Int32(kvD), 0, og, Int32(hd))
            for r in 0..<n { memcpy(of.advanced(by: (r * nH + g * G) * hd), og.advanced(by: r * G * hd), G * hd * 4) }
        }
        Self.f32ToF16(of, out, n * nH * hd)
        if n < S { memset(out.advanced(by: n * nH * hd), 0, (S - n) * nH * hd * 2) }
    }

    static func f16ToF32(_ src: UnsafePointer<Float16>, _ dst: UnsafeMutablePointer<Float>, _ count: Int) {
        var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
        var d = vImage_Buffer(data: UnsafeMutableRawPointer(dst), height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
        vImageConvert_Planar16FtoPlanarF(&s, &d, 0)
    }

    static func f32ToF16(_ src: UnsafePointer<Float>, _ dst: UnsafeMutablePointer<Float16>, _ count: Int) {
        var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
        var d = vImage_Buffer(data: UnsafeMutableRawPointer(dst), height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
        vImageConvert_PlanarFtoPlanar16F(&s, &d, 0)
    }
}
