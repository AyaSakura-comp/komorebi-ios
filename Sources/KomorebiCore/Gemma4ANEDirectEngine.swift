import Accelerate
import Foundation
import KomorebiANEDirect

/// Gemma 4 E2B on the Apple Neural Engine through the private in-memory MIL compiler (no CoreML runtime).
///
/// Layout:
///  - ANE program `b = 0` (prologue): layer 0 attn-norm + Q/K/V (+ q/k/v norms, RoPE) and the PLE model projection.
///  - ANE program `b = 1 ..< nL`: tail of layer `b-1` (O proj, post-attn norm, GeGLU FFN, PLE gate) fused with the
///    head of layer `b` (attn-norm, Q/K/V, norms, RoPE).
///  - ANE program `b = nL`: tail of the last layer + final norm.  Plus an LM-head program (tied Q6_K embeddings).
///  - CPU: token / per-layer embeddings, PLE combine, KV cache + MQA attention (cblas), argmax.
/// Every program carries two MIL functions that share one weight blob: `p0_decode` (S = 1) and `p1_prefill` (S = chunk).
/// Weights are re-quantised from the GGUF into grouped-channel LUT palettes (`constexpr_lut_to_dense`), the format the
/// ANE decompresses at run time (bandwidth scales with LUT bits).
public final class Gemma4ANEDirectEngine: @unchecked Sendable {

    public struct Options: Sendable {
        public var lutBits: Int
        public var lutGroup: Int
        public var prefillChunk: Int
        /// Rows of the speculative-decoding verify procedure (= draft tokens + 1).
        public var verifyChunk: Int
        public var maxContext: Int
        public var cacheDirectory: URL?
        /// Also build the ANE LM-head program when FlashHead is used (for audits / comparisons).
        public var keepANELMHead: Bool = false
        /// Capacity of the built-in KV session (nil: `maxContext`). `maxContext` bounds every session.
        public var defaultSessionCapacity: Int?
        public init(lutBits: Int = 4, lutGroup: Int = 8, prefillChunk: Int = 64, verifyChunk: Int = 4, maxContext: Int = 1024, cacheDirectory: URL? = nil) {
            self.lutBits = lutBits
            self.lutGroup = lutGroup
            self.prefillChunk = prefillChunk
            self.verifyChunk = verifyChunk
            self.maxContext = maxContext
            self.cacheDirectory = cacheDirectory
        }
    }

    /// Wall-clock breakdown of the most recent `generate` call (seconds).
    public struct Profile: Sendable {
        public var aneSeconds: [Double] = []
        public var aneCalls: [Int] = []
        public var lmHeadSeconds: Double = 0
        public var attentionSeconds: Double = 0
        public var embedPLESeconds: Double = 0
        public var glueSeconds: Double = 0
        public var argmaxSeconds: Double = 0
        public var drafterSeconds: Double = 0
        /// Diagnostics (drafter.debugTopK > 0): where the target's token ranked among the first draft's candidates.
        public var rankTop1 = 0, rankTop5 = 0, inCandidates = 0, rankChecks = 0
        public var verifySteps: Int = 0
        public var draftedTokens: Int = 0
        public var acceptedTokens: Int = 0

        public func minus(_ o: Profile) -> Profile {
            var r = self
            for i in 0..<min(r.aneSeconds.count, o.aneSeconds.count) { r.aneSeconds[i] -= o.aneSeconds[i]; r.aneCalls[i] -= o.aneCalls[i] }
            r.lmHeadSeconds -= o.lmHeadSeconds; r.attentionSeconds -= o.attentionSeconds; r.embedPLESeconds -= o.embedPLESeconds
            r.glueSeconds -= o.glueSeconds; r.argmaxSeconds -= o.argmaxSeconds
            r.drafterSeconds -= o.drafterSeconds; r.verifySteps -= o.verifySteps
            r.draftedTokens -= o.draftedTokens; r.acceptedTokens -= o.acceptedTokens
            return r
        }

        public func report() -> String {
            let ane = aneSeconds.reduce(0, +)
            let total = ane + lmHeadSeconds + attentionSeconds + embedPLESeconds + glueSeconds + argmaxSeconds + drafterSeconds
            func pct(_ x: Double) -> String { total > 0 ? String(format: "%5.1f%%", x / total * 100) : "  -  " }
            var lines = [String(format: "total %.1f ms", total * 1000)]
            lines.append(String(format: "  ANE layer programs  %8.2f ms %@  (%d evals)", ane * 1000, pct(ane), aneCalls.reduce(0, +)))
            lines.append(String(format: "  LM head             %8.2f ms %@", lmHeadSeconds * 1000, pct(lmHeadSeconds)))
            if verifySteps > 0 {
                lines.append(String(format: "  MTP drafter (CPU)   %8.2f ms %@  (%d verify steps, %d/%d drafts accepted = %.0f%%, %.2f tok/step)",
                                    drafterSeconds * 1000, pct(drafterSeconds), verifySteps, acceptedTokens, draftedTokens,
                                    draftedTokens > 0 ? Double(acceptedTokens) / Double(draftedTokens) * 100 : 0,
                                    Double(acceptedTokens + verifySteps) / Double(verifySteps)))
            }
            lines.append(String(format: "  CPU attention+KV    %8.2f ms %@", attentionSeconds * 1000, pct(attentionSeconds)))
            lines.append(String(format: "  CPU embed+PLE       %8.2f ms %@", embedPLESeconds * 1000, pct(embedPLESeconds)))
            lines.append(String(format: "  CPU glue/copies     %8.2f ms %@", glueSeconds * 1000, pct(glueSeconds)))
            lines.append(String(format: "  CPU argmax/glue LM  %8.2f ms %@", argmaxSeconds * 1000, pct(argmaxSeconds)))
            return lines.joined(separator: "\n")
        }
    }

    public let gguf: GGUFFile
    public let config: Gemma4Config
    public let options: Options
    public private(set) var lastProfile = Profile()
    public private(set) var buildSeconds: Double = 0
    public private(set) var compileSeconds: Double = 0
    /// Seconds spent regenerating MIL text from the cached weight layout at load.
    public private(set) var milSeconds: Double = 0
    public private(set) var weightBytes: Int = 0

    private var programs: [ANEProgram] = []
    private var lmHead: ANEProgram?
    private let P: Int
    private let V: Int
    private let maxCtx: Int
    /// FlashHead LM head (small dictionary + clustered retrieval). When nil the ANE LM-head program is used.
    public let flashHead: FlashHeadLMHead?
    /// MTP assistant drafter; when set, decoding is speculative (draft `draftTokens`, verify on the ANE).
    public var drafter: Gemma4MTPDrafter?
    /// Maximum draft length per verify step (bounded by `verifyChunk - 1`).
    public var draftTokens: Int = 3
    /// Adapt the draft length to recent acceptance (all accepted -> +1, none accepted -> -1).
    public var adaptiveDraft = true
    /// Audit: decide every token with the exact Q6_K full-vocabulary head on the CPU and count how often
    /// FlashHead / the ANE LUT LM head pick something else.
    public var auditExactLMHead = false
    public private(set) var auditTotal = 0
    public private(set) var auditFlashMismatch = 0
    public private(set) var auditANEHeadMismatch = 0
    public func resetAudit() { auditTotal = 0; auditFlashMismatch = 0; auditANEHeadMismatch = 0 }
    /// Audit hook: (final-norm hidden, exact token) for every audited decision.
    public var auditTap: ((UnsafePointer<Float>, Int32) -> Void)?
    private var lastProc: Int32 = 0
    private var lastChunkRows = 1
    /// Small-dictionary LM head rows (token IDs, padded to a multiple of 8); nil = full vocabulary.
    public let headRows: [Int32]?
    /// Exact-rescoring width for the ANE LM head (candidates re-scored against the Q6_K rows on the CPU).
    public var headRescore = 16
    private var headF32: UnsafeMutablePointer<Float>?
    private enum LogitRows { case none, last, all }

    private let tokenEmbdInfo: GGUFTensorInfo
    private let perLayerTokEmbdInfo: GGUFTensorInfo
    private let perLayerProjNorm: [Float]
    private let embdScale: Float
    private let perLayerEmbdScale: Float
    private let projNormEps: Float

    // CPU state
    /// The active KV session's buffers (see `withSession`).
    private var kCache: [UnsafeMutablePointer<Float>] = []
    private var vCache: [UnsafeMutablePointer<Float>] = []
    /// KV session used when no other one is active (dictation prompts).
    public private(set) var defaultSession: KVSession!
    private var session: KVSession!
    private var cosS: UnsafeMutablePointer<Float16>, sinS: UnsafeMutablePointer<Float16>
    private var cosF: UnsafeMutablePointer<Float16>, sinF: UnsafeMutablePointer<Float16>
    private var logitsF32: UnsafeMutablePointer<Float>?
    // Persistent attention scratch (scores sized [P * nHeads, maxCtx]) so long contexts don't page-fault every call.
    private var qScratch: UnsafeMutablePointer<Float>
    private var oScratch: UnsafeMutablePointer<Float>
    private var scoreScratch: UnsafeMutablePointer<Float>
    private var pleScratch: UnsafeMutablePointer<Float16>
    /// Profile of the prefill part / decode part of the most recent `generate`.
    public private(set) var prefillProfile = Profile()
    public private(set) var decodeProfile = Profile()

    /// `dictionary`: token IDs of the small-dictionary LM head (ANE logits over these rows, then exact CPU rescoring of the
    /// best candidates). nil = full-vocabulary ANE LM head.
    public init(gguf: GGUFFile, options: Options = Options(), flashHead: FlashHeadLMHead? = nil, dictionary: [Int32]? = nil,
                progress: ((String) -> Void)? = nil) throws {
        guard ane_rt_available() else {
            throw GemmaEngineError.inferenceFailed("AppleNeuralEngine private framework unavailable")
        }
        self.gguf = gguf
        let cfg = Gemma4Config.from(gguf: gguf.metadata)
        self.config = cfg
        self.options = options
        self.P = options.prefillChunk
        self.V = max(2, min(options.verifyChunk, options.prefillChunk))
        self.flashHead = flashHead
        if let dictionary {
            var rows = dictionary
            while rows.count % 8 != 0 { rows.append(rows.last ?? 0) }   // LUT group size
            self.headRows = rows
        } else {
            self.headRows = nil
        }
        let arch = gguf.metadata.string("general.architecture") ?? "gemma4"
        self.maxCtx = min(options.maxContext, gguf.metadata.int("\(arch).context_length") ?? 32768)
        self.embdScale = sqrtf(Float(cfg.hiddenSize))
        self.perLayerEmbdScale = sqrtf(Float(cfg.perLayerInputDim)) / sqrtf(2.0)
        self.projNormEps = cfg.rmsNormEps * Float(cfg.hiddenSize)
        self.tokenEmbdInfo = try gguf.tensor("token_embd.weight")
        self.perLayerTokEmbdInfo = try gguf.tensor("per_layer_token_embd.weight")
        self.perLayerProjNorm = try gguf.readFloat32Tensor("per_layer_proj_norm.weight")

        let rowsMax = options.prefillChunk * cfg.numAttentionHeads
        self.qScratch = .allocate(capacity: rowsMax * cfg.fullHeadDim)
        self.oScratch = .allocate(capacity: rowsMax * cfg.fullHeadDim)
        self.scoreScratch = .allocate(capacity: rowsMax * self.maxCtx)
        self.pleScratch = .allocate(capacity: cfg.numLayers * options.prefillChunk * cfg.perLayerInputDim)
        self.cosS = .allocate(capacity: self.maxCtx * cfg.slidingHeadDim)
        self.sinS = .allocate(capacity: self.maxCtx * cfg.slidingHeadDim)
        self.cosF = .allocate(capacity: self.maxCtx * cfg.fullHeadDim)
        self.sinF = .allocate(capacity: self.maxCtx * cfg.fullHeadDim)
        let main = KVSession(capacity: min(options.defaultSessionCapacity ?? self.maxCtx, self.maxCtx),
                             headDims: (0..<cfg.numOwnKVLayers).map { cfg.headDim(layer: $0) })
        self.defaultSession = main
        self.session = main
        self.kCache = main.k
        self.vCache = main.v
        try buildRoPETables()

        let t0 = CFAbsoluteTimeGetCurrent()
        var compileTotal = 0.0
        let cacheDir = options.cacheDirectory.map {
            $0.appendingPathComponent("gemma4-ane-l\(options.lutBits)g\(options.lutGroup)p\(options.prefillChunk)-\(gguf.fileSize)-v1", isDirectory: true)
        }
        if let cacheDir { try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true) }

        var milSeconds = 0.0
        /// One program to load: MIL text, weight blob (memory-mapped from the cache), procedure I/O sizes.
        struct PendingProgram {
            let tag: String
            let mil: String
            let blob: Data
            let procs: [[String: [NSNumber]]]
            let io: (inputs: [(String, Int)], outputs: [(String, Int)])?
        }
        func prepare(_ tag: String, procs: [[String: [NSNumber]]], io: (inputs: [(String, Int)], outputs: [(String, Int)])?,
                     build: (_ dryRun: Bool) throws -> (String, Data?)) throws -> PendingProgram {
            // The weight blob (k-means LUTs) is cached; the MIL text is regenerated from a sizes-only dry run,
            // so procedures can change without re-quantising.
            var mil: String
            var blob: Data
            let blobURL = cacheDir?.appendingPathComponent("\(tag).bin")
            if let blobURL, let b = try? Data(contentsOf: blobURL, options: .alwaysMapped) {
                let tm = CFAbsoluteTimeGetCurrent()
                mil = try build(true).0
                milSeconds += CFAbsoluteTimeGetCurrent() - tm
                blob = b
            } else {
                let built = try build(false)
                mil = built.0
                blob = built.1 ?? Data()
                if let blobURL, (try? blob.write(to: blobURL)) != nil,
                   let mapped = try? Data(contentsOf: blobURL, options: .alwaysMapped) {
                    blob = mapped  // keep only the mapped copy while all programs load
                }
            }
            if let cacheDir { try? mil.write(to: cacheDir.appendingPathComponent("\(tag).mil"), atomically: true, encoding: .utf8) }
            return PendingProgram(tag: tag, mil: mil, blob: blob, procs: procs, io: io)
        }

        let nL = cfg.numLayers
        var pending: [PendingProgram] = []
        for b in 0...nL {
            let io = boundaryIO(b)
            let procs: [[String: [NSNumber]]] = [1, V, P].map { S in
                ["in": io.inputs.map { NSNumber(value: S * $0.1 * 2) }, "out": io.outputs.map { NSNumber(value: S * $0.1 * 2) }]
            }
            pending.append(try prepare("b\(b)", procs: procs, io: io) { dry in try buildBoundaryProgram(b, dryRun: dry) })
        }
        let withLMHead = flashHead == nil || options.keepANELMHead
        if withLMHead {
            let nRows = headRows?.count ?? cfg.vocabSize
            let tag = headRows == nil ? "lmhead" : "lmhead-dict\(nRows)"
            let procs: [[String: [NSNumber]]] = [1, V].map { S in
                ["in": [NSNumber(value: S * cfg.hiddenSize * 2)], "out": [NSNumber(value: S * nRows * 2)]]
            }
            pending.append(try prepare(tag, procs: procs, io: nil) { dry in try buildLMHeadProgram(dryRun: dry) })
        }

        // Load every program in parallel: describing a program hashes its weights (SHA-256 over ~1.1 GB in all), which
        // took 2.3 s on one A16 core; the loads themselves are independent.
        var loaded = [Result<ANEProgram, Error>?](repeating: nil, count: pending.count)
        let lock = NSLock()
        var finished = 0
        DispatchQueue.concurrentPerform(iterations: pending.count) { i in
            let item = pending[i]
            let result = Result { try ANEProgram.compileMIL(item.mil, weights: item.blob, procedures: item.procs.map { $0 as [AnyHashable: Any] }) }
            lock.lock()
            loaded[i] = result
            finished += 1
            let n = finished
            lock.unlock()
            progress?("ANE program \(n)/\(pending.count)")
        }
        for (i, item) in pending.enumerated() {
            let prog: ANEProgram
            switch loaded[i]! {
            case .success(let p): prog = p
            case .failure(let error): throw GemmaEngineError.inferenceFailed("ANE compile \(item.tag) failed: \(error.localizedDescription)")
            }
            compileTotal += prog.compileSeconds + prog.loadSeconds
            weightBytes += item.blob.count
            if let io = item.io {
                for p in 0..<3 {
                    let inN = prog.inputNames(Int32(p)), outN = prog.outputNames(Int32(p))
                    let expIn = io.inputs.map { $0.0 }, expOut = io.outputs.map { $0.0 }
                    guard inN == expIn, outN.map({ $0.replacingOccurrences(of: "@output", with: "") }) == expOut else {
                        throw GemmaEngineError.inferenceFailed("ANE program \(item.tag) I/O order mismatch: \(inN) / \(outN)")
                    }
                }
                programs.append(prog)
            } else {
                lmHead = prog
            }
        }
        // Loaded on the ANE: drop the weight blobs (1.1 GB of mapped address space; the app's address space is limited).
        for p in programs { p.releaseWeights() }
        lmHead?.releaseWeights()
        self.compileSeconds = compileTotal
        self.milSeconds = milSeconds
        self.buildSeconds = CFAbsoluteTimeGetCurrent() - t0
    }

    deinit {
        logitsF32?.deallocate()
        headF32?.deallocate()
        qScratch.deallocate(); oScratch.deallocate(); scoreScratch.deallocate(); pleScratch.deallocate()
        cosS.deallocate(); sinS.deallocate(); cosF.deallocate(); sinF.deallocate()
    }

    public func clearPrefixCache() { session.reset() }

    /// A new, empty KV session (capacity at most `maxContext`) for a separate conversation.
    public func makeSession(capacity: Int) -> KVSession {
        KVSession(capacity: min(capacity, maxCtx), headDims: (0..<config.numOwnKVLayers).map { config.headDim(layer: $0) })
    }

    /// Runs `body` with `kv` as the active KV session (prompts reuse and extend its cache), then switches back.
    public func withSession<T>(_ kv: KVSession, _ body: () throws -> T) rethrows -> T {
        let previous = session!
        activate(kv)
        defer { activate(previous) }
        return try body()
    }

    private func activate(_ kv: KVSession) {
        session = kv
        kCache = kv.k
        vCache = kv.v
    }

    /// Identifies the model a saved KV session belongs to.
    public var kvFingerprint: String {
        "gemma4-e2b-\(gguf.fileSize)-l\(options.lutBits)g\(options.lutGroup)"
    }

    /// Context size of the active session.
    public var activeCapacity: Int { session.capacity }

    /// Runs every program once in both procedures (first eval after load is slow) so the first request is warm.
    public func warmUp() throws {
        let toks = [Int32](repeating: config.bosTokenID, count: min(P, 8))
        _ = try generate(inputIDs: toks, maxNewTokens: 3, stopTokenIDs: [], staticPrefixLength: 0)
        // Also run the verify procedure (speculative decoding) once in every program and the LM head.
        _ = try forwardChunk(tokens: [Int32](repeating: config.bosTokenID, count: V), p0: toks.count, logits: .all)
        session.reset()
    }

    /// Measures program/procedure switching patterns (ms per eval).
    public func benchmarkSwitching() throws -> String {
        func time(_ label: String, _ body: () throws -> Int) rethrows -> String {
            let t0 = CFAbsoluteTimeGetCurrent()
            let evals = try body()
            return String(format: "%-48@ %7.3f ms/eval", label as NSString, (CFAbsoluteTimeGetCurrent() - t0) * 1000 / Double(evals))
        }
        let nOwn = config.numOwnKVLayers
        var out: [String] = []
        out.append(try time("chain b0..b14, prefill proc, x5") { for _ in 0..<5 { for b in 0..<nOwn { try programs[b].eval(2) } }; return 5 * nOwn })
        out.append(try time("chain b0..b14, decode proc, x5") { for _ in 0..<5 { for b in 0..<nOwn { try programs[b].eval(0) } }; return 5 * nOwn })
        out.append(try time("chain b0..b14, verify proc, x5") { for _ in 0..<5 { for b in 0..<nOwn { try programs[b].eval(1) } }; return 5 * nOwn })
        return out.joined(separator: "\n")
    }

    /// Times each ANE program in isolation (inputs as left by the last run). Returns a per-program table.
    public func benchmarkPrograms(iterations: Int = 20) throws -> String {
        var lines = ["prog   layer-tail -> layer-head        decode S=1   verify S=\(V)  prefill S=\(P)"]
        var totals = [0.0, 0.0, 0.0]
        for (b, prog) in programs.enumerated() {
            var t = [0.0, 0.0, 0.0]
            for p in 0..<3 {
                try prog.eval(Int32(p))
                let t0 = CFAbsoluteTimeGetCurrent()
                for _ in 0..<iterations { try prog.eval(Int32(p)) }
                t[p] = (CFAbsoluteTimeGetCurrent() - t0) / Double(iterations)
            }
            for i in 0..<3 { totals[i] += t[i] }
            let tail = b == 0 ? "embed/PLE" : "L\(b - 1) O+FFN+PLE"
            let head = b == config.numLayers ? "final norm" : "L\(b) QKV\(config.hasOwnKV(layer: b) ? "" : "(Q only)")"
            lines.append(String(format: "b%-3d  %-16@ -> %-14@ %8.3f ms %8.3f ms %8.3f ms", b, tail as NSString, head as NSString, t[0] * 1000, t[1] * 1000, t[2] * 1000))
        }
        lines.append(String(format: "sum of layer programs                    %8.3f ms %8.3f ms %8.3f ms", totals[0] * 1000, totals[1] * 1000, totals[2] * 1000))
        if let lmHead {
            try lmHead.eval(0)
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iterations { try lmHead.eval(0) }
            let label = "ANE LM head (\(headRows?.count ?? config.vocabSize) x \(config.hiddenSize))"
            lines.append(String(format: "%-40@ %8.3f ms", label as NSString, (CFAbsoluteTimeGetCurrent() - t0) / Double(iterations) * 1000))
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - I/O layout

    /// (symbol, channels) per input/output of boundary program `b`.
    private func boundaryIO(_ b: Int) -> (inputs: [(String, Int)], outputs: [(String, Int)]) {
        let cfg = config
        let H = cfg.hiddenSize, nH = cfg.numAttentionHeads, nL = cfg.numLayers
        var ins: [(String, Int)] = [("a_h", H)]
        var outs: [(String, Int)] = []
        if b >= 1 {
            ins.append(("b_attn", nH * cfg.headDim(layer: b - 1)))
            ins.append(("c_ple", cfg.perLayerInputDim))
        }
        if b < nL {
            let hd = cfg.headDim(layer: b)
            ins.append(("d_cos", hd))
            ins.append(("e_sin", hd))
            if b >= 1 { outs.append(("o0_h", H)) }
            outs.append(("o1_q", nH * hd))
            if cfg.hasOwnKV(layer: b) {
                outs.append(("o2_k", hd))
                outs.append(("o3_v", hd))
            }
            if b == 0 { outs.append(("o4_pl", nL * cfg.perLayerInputDim)) }
        } else {
            outs.append(("o0_x", H))
        }
        return (ins, outs)
    }

    // MARK: - Program construction

    private func buildBoundaryProgram(_ b: Int, dryRun: Bool) throws -> (String, Data?) {
        let cfg = config
        let H = cfg.hiddenSize, nH = cfg.numAttentionHeads, nL = cfg.numLayers, plD = cfg.perLayerInputDim
        let eps = cfg.rmsNormEps
        let bank = WeightBank(bits: options.lutBits, group: options.lutGroup, dryRun: dryRun)

        // Weights (S-independent declarations)
        var tailOutScale: Float = 1
        if b >= 1 {
            let l = b - 1
            try bank.lut("wo", gguf, "blk.\(l).attn_output.weight")
            try bank.vec("n_pa", gguf, "blk.\(l).post_attention_norm.weight")
            try bank.vec("n_ffn", gguf, "blk.\(l).ffn_norm.weight")
            try bank.lut("wg", gguf, "blk.\(l).ffn_gate.weight")
            try bank.lut("wu", gguf, "blk.\(l).ffn_up.weight")
            try bank.lut("wd", gguf, "blk.\(l).ffn_down.weight")
            try bank.vec("n_pf", gguf, "blk.\(l).post_ffw_norm.weight")
            try bank.lut("wig", gguf, "blk.\(l).inp_gate.weight")
            try bank.lut("wpr", gguf, "blk.\(l).proj.weight")
            try bank.vec("n_po", gguf, "blk.\(l).post_norm.weight")
            tailOutScale = (try? gguf.readFloat32Tensor("blk.\(l).layer_output_scale.weight").first) ?? 1
        }
        if b < nL {
            try bank.vec("n_at", gguf, "blk.\(b).attn_norm.weight")
            try bank.lut("wq", gguf, "blk.\(b).attn_q.weight")
            try bank.vec("n_q", gguf, "blk.\(b).attn_q_norm.weight")
            if cfg.hasOwnKV(layer: b) {
                try bank.lut("wk", gguf, "blk.\(b).attn_k.weight")
                try bank.lut("wv", gguf, "blk.\(b).attn_v.weight")
                try bank.vec("n_k", gguf, "blk.\(b).attn_k_norm.weight")
            }
            if b == 0 { try bank.lut("wpl", gguf, "per_layer_model_proj.weight") }
        } else {
            try bank.vec("n_out", gguf, "output_norm.weight")
        }

        let io = boundaryIO(b)
        var mil = MILWriter.header
        let functions = [("p0_decode", 1), ("p1_verify", V), ("p2_prefill", P)]
        for (fname, S) in functions {
            let w = MILWriter(S: S)
            let sig = io.inputs.map { name, c -> String in
                (name == "d_cos" || name == "e_sin") ? "tensor<fp16, [1, \(S), 1, \(c)]> \(name)" : "tensor<fp16, [1, \(S), \(c)]> \(name)"
            }.joined(separator: ", ")
            w.raw("    func \(fname)<ios18>(\(sig)) {\n")
            for d in bank.declarations { w.line(d) }

            var hIn = "a_h"
            if b >= 1 {
                let o = w.linear("b_attn", "wo", [1, S, H])
                let on = w.rmsNorm(o, dims: [1, S, H], weight: "n_pa", eps: eps)
                let h1 = w.add("a_h", on, [1, S, H])
                let x = w.rmsNorm(h1, dims: [1, S, H], weight: "n_ffn", eps: eps)
                let F = bank.rows["wg"]!
                let g = w.linear(x, "wg", [1, S, F])
                let u = w.linear(x, "wu", [1, S, F])
                let a = w.mul(w.gelu(g, [1, S, F]), u, [1, S, F])
                let dn = w.linear(a, "wd", [1, S, H])
                let dnn = w.rmsNorm(dn, dims: [1, S, H], weight: "n_pf", eps: eps)
                let h2 = w.add(h1, dnn, [1, S, H])
                let pg = w.linear(h2, "wig", [1, S, plD])
                let a2 = w.mul(w.gelu(pg, [1, S, plD]), "c_ple", [1, S, plD])
                let pp = w.linear(a2, "wpr", [1, S, H])
                let ppn = w.rmsNorm(pp, dims: [1, S, H], weight: "n_po", eps: eps)
                let h3s = w.add(h2, ppn, [1, S, H])
                let outName = b < nL ? "o0_h" : "h3"
                hIn = w.mul(h3s, w.scalar(tailOutScale), [1, S, H], name: outName)
            }
            var outs: [String] = []
            if b < nL {
                if b >= 1 { outs.append("o0_h") }
                let hd = cfg.headDim(layer: b)
                let x = w.rmsNorm(hIn, dims: [1, S, H], weight: "n_at", eps: eps)
                let q = w.linear(x, "wq", [1, S, nH * hd])
                let q4 = w.reshape(q, [1, S, nH, hd])
                let qn = w.rmsNorm(q4, dims: [1, S, nH, hd], weight: "n_q", eps: eps)
                let qr = w.rope(qn, dims: [1, S, nH, hd], cos: "d_cos", sin: "e_sin")
                _ = w.reshape(qr, [1, S, nH * hd], name: "o1_q")
                outs.append("o1_q")
                if cfg.hasOwnKV(layer: b) {
                    let k = w.linear(x, "wk", [1, S, hd])
                    let k4 = w.reshape(k, [1, S, 1, hd])
                    let kn = w.rmsNorm(k4, dims: [1, S, 1, hd], weight: "n_k", eps: eps)
                    let kr = w.rope(kn, dims: [1, S, 1, hd], cos: "d_cos", sin: "e_sin")
                    _ = w.reshape(kr, [1, S, hd], name: "o2_k")
                    let v = w.linear(x, "wv", [1, S, hd])
                    _ = w.rmsNorm(v, dims: [1, S, hd], weight: nil, eps: eps, name: "o3_v")
                    outs.append(contentsOf: ["o2_k", "o3_v"])
                }
                if b == 0 {
                    _ = w.linear("a_h", "wpl", [1, S, nL * plD], name: "o4_pl")
                    outs.append("o4_pl")
                }
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

    private func buildLMHeadProgram(dryRun: Bool) throws -> (String, Data?) {
        let cfg = config
        let H = cfg.hiddenSize
        let rows = headRows ?? (0..<cfg.vocabSize).map { Int32($0) }
        let total = rows.count
        let chunk = 16384
        let bank = WeightBank(bits: options.lutBits, group: options.lutGroup, dryRun: dryRun)
        let nChunks = (total + chunk - 1) / chunk
        for c in 0..<nChunks {
            let lo = c * chunk, hi = min(total, lo + chunk)
            if headRows == nil {
                try bank.lut("e\(c)", gguf, "token_embd.weight", rowStart: lo, rowCount: hi - lo)
            } else {
                try bank.lut("e\(c)", gguf, "token_embd.weight", rowList: Array(rows[lo..<hi]))
            }
        }
        var mil = MILWriter.header
        for (fname, S) in [("p0_decode", 1), ("p1_verify", V)] {
            let w = MILWriter(S: S)
            w.raw("    func \(fname)<ios18>(tensor<fp16, [1, \(S), \(H)]> a_x) {\n")
            for d in bank.declarations { w.line(d) }
            var parts: [String] = []
            for c in 0..<nChunks { parts.append(w.linear("a_x", "e\(c)", [1, S, bank.rows["e\(c)"]!])) }
            _ = w.concatLast(parts, [1, S, total], name: "o0_logits")
            w.raw("    } -> (o0_logits);\n")
            mil += w.text
        }
        mil += "}\n"
        return (mil, dryRun ? nil : bank.finish())
    }

    private func buildRoPETables() throws {
        let cfg = config
        let freqs = try gguf.readFloat32Tensor("rope_freqs.weight")
        func table(hd: Int, base: Float, useFreqs: Bool, c: UnsafeMutablePointer<Float16>, s: UnsafeMutablePointer<Float16>) {
            let half = hd / 2
            let invFreq = (0..<half).map { j in powf(base, -2.0 * Float(j) / Float(hd)) / (useFreqs ? freqs[j] : 1.0) }
            for pos in 0..<maxCtx {
                for j in 0..<half {
                    let ang = Float(pos) * invFreq[j]
                    let cv = Float16(cosf(ang)), sv = Float16(sinf(ang))
                    c[pos * hd + j] = cv; c[pos * hd + j + half] = cv
                    s[pos * hd + j] = sv; s[pos * hd + j + half] = sv
                }
            }
        }
        table(hd: cfg.slidingHeadDim, base: cfg.ropeFreqBaseSWA, useFreqs: false, c: cosS, s: sinS)
        table(hd: cfg.fullHeadDim, base: cfg.ropeFreqBaseFull, useFreqs: true, c: cosF, s: sinF)
    }

    // MARK: - Generation

    public func generate(
        inputIDs: [Int32],
        maxNewTokens: Int = 256,
        stopTokenIDs: Set<Int32>,
        staticPrefixLength: Int = 0,
        shouldCancel: (() -> Bool)? = nil,
        onToken: ((Int32) -> Void)? = nil
    ) throws -> (tokenIDs: [Int32], stats: GenerationStats) {
        guard !inputIDs.isEmpty else {
            return ([], GenerationStats(backend: backendName, prefixCachedTokens: 0, prefillTokens: 0, prefillSeconds: 0, decodeTokens: 0, decodeSeconds: 0))
        }
        guard inputIDs.count < session.capacity else {
            throw GemmaEngineError.inferenceFailed("Prompt length \(inputIDs.count) exceeds the context (\(session.capacity))")
        }
        lastProfile = Profile(aneSeconds: [Double](repeating: 0, count: programs.count), aneCalls: [Int](repeating: 0, count: programs.count))

        let t0 = CFAbsoluteTimeGetCurrent()
        // Keys/values already in the session for the longest common prefix are reused; at least the last prompt token
        // is run again for its logits. (`staticPrefixLength` is kept for callers; the prefix match covers it.)
        let cached = session.tokens
        var start = 0
        let limit = min(cached.count, inputIDs.count - 1)
        while start < limit && cached[start] == inputIDs[start] { start += 1 }
        let prefixHit = start > 0
        session.tokens = Array(inputIDs[0..<start])
        var next = try prefill(Array(inputIDs[start...]), p0: start, needLogits: true)
        session.tokens = inputIDs
        let t1 = CFAbsoluteTimeGetCurrent()
        prefillProfile = lastProfile

        var out: [Int32] = []
        var pos = inputIDs.count
        if maxNewTokens > 0 && !stopTokenIDs.contains(next) {
            out.append(next)
            onToken?(next)
            if let drafter, maxNewTokens > 1 {
                try speculativeDecode(drafter: drafter, firstToken: next, lastPos: inputIDs.count - 1, maxNewTokens: maxNewTokens,
                                      stopTokenIDs: stopTokenIDs, shouldCancel: shouldCancel, onToken: onToken, out: &out)
            } else {
                while out.count < maxNewTokens && pos < session.capacity - 1 {
                    if shouldCancel?() == true { break }
                    let fed = next
                    next = try forwardChunk(tokens: [next], p0: pos, logits: .last).last ?? 0
                    session.tokens.append(fed)
                    pos += 1
                    if stopTokenIDs.contains(next) { break }
                    out.append(next)
                    onToken?(next)
                }
            }
        }
        let t2 = CFAbsoluteTimeGetCurrent()
        decodeProfile = lastProfile.minus(prefillProfile)
        let stats = GenerationStats(
            backend: backendName,
            prefixCachedTokens: prefixHit ? start : 0,
            prefillTokens: inputIDs.count - (prefixHit ? start : 0),
            prefillSeconds: max(t1 - t0, 1e-6),
            decodeTokens: out.count,
            decodeSeconds: max(t2 - t1, 1e-6)
        )
        return (out, stats)
    }

    public var backendName: String {
        "ANE direct (LUT\(options.lutBits)\(flashHead != nil ? ", FlashHead" : "")\(drafter != nil ? ", MTP×\(draftTokens)" : ""))"
    }

    /// Greedy speculative decoding: the MTP assistant drafts on the CPU, one ANE pass verifies `drafts + 1` rows.
    private func speculativeDecode(drafter: Gemma4MTPDrafter, firstToken: Int32, lastPos: Int, maxNewTokens: Int,
                                   stopTokenIDs: Set<Int32>, shouldCancel: (() -> Bool)?, onToken: ((Int32) -> Void)?,
                                   out: inout [Int32]) throws {
        let cfg = config
        let H = cfg.hiddenSize
        let nOwn = cfg.numOwnKVLayers
        let slideSrc = (0..<nOwn).last { cfg.isSlidingLayer[$0] } ?? 0
        let fullSrc = (0..<nOwn).last { !cfg.isSlidingLayer[$0] } ?? (nOwn - 1)
        let hBuf = UnsafeMutablePointer<Float>.allocate(capacity: H); defer { hBuf.deallocate() }
        hiddenRow(lastChunkRows - 1, into: hBuf)
        var P = lastPos          // KV is valid through position P
        var pending = firstToken // emitted, sits at position P + 1, not yet run through the target
        let gammaMax = max(0, min(draftTokens, V - 1))
        var gamma = gammaMax
        let emb = tokenEmbdInfo, scale = embdScale
        while out.count < maxNewTokens {
            if shouldCancel?() == true { break }
            let room = session.capacity - 2 - (P + 1)
            if room < 0 { break }
            let k = min(gamma, maxNewTokens - out.count, room)
            let td = CFAbsoluteTimeGetCurrent()
            let drafts: [Int32] = k == 0 ? [] : drafter.propose(
                token: pending, targetHidden: hBuf, position: P, count: k,
                kv: { sliding in
                    let l = sliding ? slideSrc : fullSrc
                    return .init(k: UnsafePointer(self.kCache[l]), v: UnsafePointer(self.vCache[l]), headDim: cfg.headDim(layer: l), sliding: sliding)
                },
                rope: { sliding, pos in
                    sliding ? (UnsafePointer(self.cosS.advanced(by: pos * cfg.slidingHeadDim)), UnsafePointer(self.sinS.advanced(by: pos * cfg.slidingHeadDim)))
                            : (UnsafePointer(self.cosF.advanced(by: pos * cfg.fullHeadDim)), UnsafePointer(self.sinF.advanced(by: pos * cfg.fullHeadDim)))
                },
                targetEmbed: { tok, dst in
                    self.gguf.dequantizeQ6KRow(tensor: emb, rowIndex: Int(tok), into: dst, count: H)
                    var sc = scale
                    vDSP_vsmul(dst, 1, &sc, dst, 1, vDSP_Length(H))
                })
            lastProfile.drafterSeconds += CFAbsoluteTimeGetCurrent() - td
            let g = try forwardChunk(tokens: [pending] + drafts, p0: P + 1, logits: .all)
            var acc = 0
            while acc < drafts.count && drafts[acc] == g[acc] { acc += 1 }
            if drafter.debugTopK > 0, !drafts.isEmpty {
                let truth = g[0]
                lastProfile.rankChecks += 1
                if drafter.lastTopCandidates.first == truth { lastProfile.rankTop1 += 1 }
                if drafter.lastTopCandidates.prefix(5).contains(truth) { lastProfile.rankTop5 += 1 }
                if drafter.lastCandidateSet.contains(truth) { lastProfile.inCandidates += 1 }
            }
            // Keys/values of the tokens that were right ([pending] + accepted drafts) are now in the cache.
            session.tokens.append(pending)
            session.tokens.append(contentsOf: drafts.prefix(acc))
            lastProfile.verifySteps += 1
            lastProfile.draftedTokens += drafts.count
            lastProfile.acceptedTokens += acc
            if adaptiveDraft && !drafts.isEmpty {
                if acc == drafts.count { gamma = min(gammaMax, gamma + 1) } else if acc == 0 { gamma = max(1, gamma - 1) }
            }
            var done = false
            for i in 0...acc {
                if stopTokenIDs.contains(g[i]) { done = true; break }
                out.append(g[i])
                onToken?(g[i])
                if out.count >= maxNewTokens { done = true; break }
            }
            if done { break }
            hiddenRow(acc, into: hBuf)
            pending = g[acc]
            P += 1 + acc
        }
    }

    /// Exact greedy token over the full Q6_K tied head (CPU, parallel) — reference for audits.
    private func exactArgmax(_ h: UnsafePointer<Float>) -> Int32 {
        let cfg = config
        let H = cfg.hiddenSize, Vocab = cfg.vocabSize
        let rowBytes = tokenEmbdInfo.type.rowByteSize(elements: H)
        let base = gguf.pointer(for: tokenEmbdInfo)
        let parts = 32
        var bestV = [Float](repeating: -.infinity, count: parts)
        var bestI = [Int32](repeating: 0, count: parts)
        bestV.withUnsafeMutableBufferPointer { bv in
            bestI.withUnsafeMutableBufferPointer { bi in
                DispatchQueue.concurrentPerform(iterations: parts) { p in
                    let row = UnsafeMutablePointer<Float>.allocate(capacity: H); defer { row.deallocate() }
                    let a = p * Vocab / parts, b = (p + 1) * Vocab / parts
                    for t in a..<b {
                        GGUFQuantization.dequantizeQ6KRow(src: base.advanced(by: t * rowBytes), dst: row, elements: H)
                        var v: Float = 0
                        vDSP_dotpr(row, 1, h, 1, &v, vDSP_Length(H))
                        if v > bv[p] { bv[p] = v; bi[p] = Int32(t) }
                    }
                }
            }
        }
        var best = 0
        for p in 1..<parts where bestV[p] > bestV[best] { best = p }
        return bestI[best]
    }

    /// Final-norm hidden row `r` of the last `forwardChunk`, as fp32.
    private func hiddenRow(_ r: Int, into dst: UnsafeMutablePointer<Float>) {
        let H = config.hiddenSize
        let x = programs[config.numLayers].outputPtr(0, procedure: lastProc).assumingMemoryBound(to: Float16.self)
        Self.f16ToF32(x.advanced(by: r * H), dst, H)
    }

    /// Greedy token for each requested row of the last `forwardChunk`.
    private func argmaxRows(_ rows: [Int], proc: Int32) throws -> [Int32] {
        let cfg = config
        let H = cfg.hiddenSize
        let tL = CFAbsoluteTimeGetCurrent()
        defer { lastProfile.lmHeadSeconds += CFAbsoluteTimeGetCurrent() - tL }
        if auditExactLMHead {
            let h = UnsafeMutablePointer<Float>.allocate(capacity: H); defer { h.deallocate() }
            var out: [Int32] = []
            for r in rows {
                hiddenRow(r, into: h)
                let exact = exactArgmax(h)
                auditTap?(h, exact)
                auditTotal += 1
                if let fh = flashHead, fh.argmax(h) != exact { auditFlashMismatch += 1 }
                if lmHead != nil, try aneHeadArgmaxRows([r], proc: proc).first != exact { auditANEHeadMismatch += 1 }
                out.append(exact)
            }
            return out
        }
        if let fh = flashHead {
            let hs = UnsafeMutablePointer<Float>.allocate(capacity: rows.count * H); defer { hs.deallocate() }
            for (i, r) in rows.enumerated() { hiddenRow(r, into: hs.advanced(by: i * H)) }
            return fh.argmaxBatch(hs, count: rows.count)
        }
        return try aneHeadArgmaxRows(rows, proc: proc)
    }

    /// ANE LM head for up to `V` rows in one eval, then exact Q6_K rescoring of the best `headRescore` candidates per row.
    private func aneHeadArgmaxRows(_ rows: [Int], proc: Int32) throws -> [Int32] {
        let cfg = config
        let H = cfg.hiddenSize
        guard let lmHead else { throw GemmaEngineError.inferenceFailed("no LM head") }
        let nRows = headRows?.count ?? cfg.vocabSize
        var out: [Int32] = []
        var idx = 0
        if headF32 == nil { headF32 = .allocate(capacity: V * nRows) }
        let xs = programs[cfg.numLayers].outputPtr(0, procedure: proc)
        let h = UnsafeMutablePointer<Float>.allocate(capacity: H); defer { h.deallocate() }
        let rowF = UnsafeMutablePointer<Float>.allocate(capacity: H); defer { rowF.deallocate() }
        let embBase = gguf.pointer(for: tokenEmbdInfo)
        let embRowBytes = tokenEmbdInfo.type.rowByteSize(elements: H)
        while idx < rows.count {
            let n = min(V, rows.count - idx)
            let hp: Int32 = n == 1 ? 0 : 1
            let inp = lmHead.inputPtr(0, procedure: hp)
            for i in 0..<n { memcpy(inp.advanced(by: i * H * 2), xs.advanced(by: rows[idx + i] * H * 2), H * 2) }
            try lmHead.eval(hp)
            let logits = lmHead.outputPtr(0, procedure: hp).assumingMemoryBound(to: Float16.self)
            Self.f16ToF32(logits, headF32!, n * nRows)
            for i in 0..<n {
                let lg = headF32!.advanced(by: i * nRows)
                let R = max(1, headRescore)
                var top = [(Float, Int)](repeating: (-.infinity, 0), count: R)
                var floor: Float = -.infinity
                for j in 0..<nRows where lg[j] > floor {
                    var k = R - 1
                    while k > 0 && top[k - 1].0 < lg[j] { top[k] = top[k - 1]; k -= 1 }
                    top[k] = (lg[j], j)
                    floor = top[R - 1].0
                }
                hiddenRow(rows[idx + i], into: h)
                var best: Int32 = 0
                var bestV: Float = -.infinity
                for (_, j) in top {
                    let tok = headRows.map { $0[j] } ?? Int32(j)
                    GGUFQuantization.dequantizeQ6KRow(src: embBase.advanced(by: Int(tok) * embRowBytes), dst: rowF, elements: H)
                    var v: Float = 0
                    vDSP_dotpr(rowF, 1, h, 1, &v, vDSP_Length(H))
                    if v > bestV { bestV = v; best = tok }
                }
                out.append(best)
            }
            idx += n
        }
        return out
    }

    private func prefill(_ tokens: [Int32], p0: Int, needLogits: Bool) throws -> Int32 {
        var idx = 0
        var last: Int32 = 0
        while idx < tokens.count {
            let n = min(P, tokens.count - idx)
            let isLast = idx + n == tokens.count
            let r = try forwardChunk(tokens: Array(tokens[idx..<(idx + n)]), p0: p0 + idx, logits: needLogits && isLast ? .last : .none)
            if let t = r.last { last = t }
            idx += n
        }
        return last
    }

    /// Runs `tokens` (<= P) at positions `p0 ..< p0 + n`. Writes KV for own-KV layers; returns argmax if `needLogits`.
    private func forwardChunk(tokens: [Int32], p0: Int, logits: LogitRows) throws -> [Int32] {
        let cfg = config
        let n = tokens.count
        let proc: Int32 = n == 1 ? 0 : (n <= V ? 1 : 2)
        let S = n == 1 ? 1 : (n <= V ? V : P)
        lastProc = proc
        lastChunkRows = n
        let H = cfg.hiddenSize, nL = cfg.numLayers, plD = cfg.perLayerInputDim, nOwn = cfg.numOwnKVLayers
        var tA = CFAbsoluteTimeGetCurrent()
        func lap(_ add: (inout Profile, Double) -> Void) {
            let t = CFAbsoluteTimeGetCurrent(); add(&lastProfile, t - tA); tA = t
        }

        // 1. Embeddings -> prologue input, RoPE rows for layer 0.
        let pro = programs[0]
        let hIn = pro.inputPtr(0, procedure: proc).assumingMemoryBound(to: Float16.self)
        memset(hIn, 0, S * H * 2)
        let gg = gguf, embInfo = tokenEmbdInfo, plInfo = perLayerTokEmbdInfo, eScale = embdScale, V = cfg.vocabSize
        DispatchQueue.concurrentPerform(iterations: n) { r in
            let rowF = UnsafeMutablePointer<Float>.allocate(capacity: H); defer { rowF.deallocate() }
            let tid = max(0, min(V - 1, Int(tokens[r])))
            gg.dequantizeQ6KRow(tensor: embInfo, rowIndex: tid, into: rowF, count: H)
            for j in 0..<H { hIn[r * H + j] = Float16(rowF[j] * eScale) }
        }
        fillRoPE(pro, proc: proc, layer: 0, inputBase: 1, p0: p0, n: n, S: S)
        lap { $0.embedPLESeconds += $1 }

        try evalProgram(0, proc: proc)
        tA = CFAbsoluteTimeGetCurrent()

        // 2. PLE combine on CPU -> ple[l][r][256] fp16
        let pl = pro.outputPtr(cfg.hasOwnKV(layer: 0) ? 3 : 1, procedure: proc).assumingMemoryBound(to: Float16.self)
        let ple = pleScratch
        ple.update(repeating: 0, count: nL * S * plD)
        let invS2: Float = 1 / sqrtf(2)
        let pnEps = projNormEps, pnW = perLayerProjNorm, plScale = perLayerEmbdScale
        DispatchQueue.concurrentPerform(iterations: n) { r in
            let rowF = UnsafeMutablePointer<Float>.allocate(capacity: nL * plD); defer { rowF.deallocate() }
            let tid = max(0, min(V - 1, Int(tokens[r])))
            gg.dequantizeQ6KRow(tensor: plInfo, rowIndex: tid, into: rowF, count: nL * plD)
            for l in 0..<nL {
                var ss: Float = 0
                let src = pl.advanced(by: r * nL * plD + l * plD)
                for j in 0..<plD { let v = Float(src[j]); ss += v * v }
                let inv = invS2 / sqrtf(ss / Float(plD) + pnEps)
                let dst = ple.advanced(by: (l * S + r) * plD)
                for j in 0..<plD {
                    dst[j] = Float16(Float(src[j]) * inv * pnW[j] + rowF[l * plD + j] * plScale)
                }
            }
        }
        lap { $0.embedPLESeconds += $1 }

        // 3. Layer chain. KV of layer 0 from the prologue.
        storeKV(prog: pro, proc: proc, layer: 0, kIndex: 1, p0: p0, n: n)
        let lastB = logits == .none ? nOwn - 1 : nL
        lap { $0.glueSeconds += $1 }

        var prevProg = pro
        var prevQIndex = 0
        var prevHIndex = -1   // prologue has no hidden output; use its input
        for b in 1...max(1, lastB) where b <= lastB {
            let l = b - 1
            let prog = programs[b]
            // attention for layer l (q from previous program)
            let qPtr = prevProg.outputPtr(Int32(prevQIndex), procedure: proc).assumingMemoryBound(to: Float16.self)
            let attnIn = prog.inputPtr(1, procedure: proc).assumingMemoryBound(to: Float16.self)
            attention(layer: l, q: qPtr, rows: n, p0: p0, out: attnIn, S: S)
            lap { $0.attentionSeconds += $1 }
            // hidden + ple + rope
            let hSrc: UnsafeMutableRawPointer = prevHIndex < 0 ? prevProg.inputPtr(0, procedure: proc) : prevProg.outputPtr(Int32(prevHIndex), procedure: proc)
            memcpy(prog.inputPtr(0, procedure: proc), hSrc, S * H * 2)
            memcpy(prog.inputPtr(2, procedure: proc), ple.advanced(by: l * S * plD), S * plD * 2)
            if b < nL { fillRoPE(prog, proc: proc, layer: b, inputBase: 3, p0: p0, n: n, S: S) }
            lap { $0.glueSeconds += $1 }

            try evalProgram(b, proc: proc)
            tA = CFAbsoluteTimeGetCurrent()

            if b < nL {
                if cfg.hasOwnKV(layer: b) { storeKV(prog: prog, proc: proc, layer: b, kIndex: 2, p0: p0, n: n) }
                prevProg = prog
                prevQIndex = 1
                prevHIndex = 0
            }
            lap { $0.glueSeconds += $1 }
        }
        guard logits != .none else { return [] }

        // 4. LM head (FlashHead over the small dictionary, or the ANE LM-head program).
        return try argmaxRows(logits == .all ? Array(0..<n) : [n - 1], proc: proc)
    }

    private func evalProgram(_ b: Int, proc: Int32) throws {
        let t = CFAbsoluteTimeGetCurrent()
        do { try programs[b].eval(proc) } catch {
            throw GemmaEngineError.inferenceFailed("ANE eval b\(b): \(error.localizedDescription)")
        }
        lastProfile.aneSeconds[b] += CFAbsoluteTimeGetCurrent() - t
        lastProfile.aneCalls[b] += 1
    }

    private func fillRoPE(_ prog: ANEProgram, proc: Int32, layer: Int, inputBase: Int, p0: Int, n: Int, S: Int) {
        let hd = config.headDim(layer: layer)
        let swa = config.isSlidingLayer[layer]
        let c = prog.inputPtr(Int32(inputBase), procedure: proc).assumingMemoryBound(to: Float16.self)
        let s = prog.inputPtr(Int32(inputBase + 1), procedure: proc).assumingMemoryBound(to: Float16.self)
        memcpy(c, (swa ? cosS : cosF).advanced(by: p0 * hd), n * hd * 2)
        memcpy(s, (swa ? sinS : sinF).advanced(by: p0 * hd), n * hd * 2)
    }

    private func storeKV(prog: ANEProgram, proc: Int32, layer: Int, kIndex: Int, p0: Int, n: Int) {
        let hd = config.headDim(layer: layer)
        let k = prog.outputPtr(Int32(kIndex), procedure: proc).assumingMemoryBound(to: Float16.self)
        let v = prog.outputPtr(Int32(kIndex + 1), procedure: proc).assumingMemoryBound(to: Float16.self)
        Self.f16ToF32(k, kCache[layer].advanced(by: p0 * hd), n * hd)
        Self.f16ToF32(v, vCache[layer].advanced(by: p0 * hd), n * hd)
    }

    /// MQA attention (scale 1.0) for `rows` query rows at positions `p0 ..< p0 + rows`, fp32 via cblas (AMX).
    /// Sliding layers only touch keys in the union of their windows; softmax is vectorised and row-parallel.
    private func attention(layer il: Int, q: UnsafePointer<Float16>, rows: Int, p0: Int, out: UnsafeMutablePointer<Float16>, S: Int) {
        let cfg = config
        let hd = cfg.headDim(layer: il), nH = cfg.numAttentionHeads
        let kv = cfg.kvSourceLayer[il]
        let swa = cfg.isSlidingLayer[il]
        let W = cfg.slidingWindow
        let kStart = swa ? max(0, p0 - W + 1) : 0
        let Tk = p0 + rows - kStart
        let M = rows * nH
        let qf = qScratch, sc = scoreScratch, of = oScratch
        Self.f16ToF32(q, qf, M * hd)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(M), Int32(Tk), Int32(hd), 1,
                    qf, Int32(hd), kCache[kv].advanced(by: kStart * hd), Int32(hd), 0, sc, Int32(Tk))
        let rowsPerTask = max(1, 4096 / max(Tk, 1))
        let tasks = (M + rowsPerTask - 1) / rowsPerTask
        DispatchQueue.concurrentPerform(iterations: tasks) { t in
            for i in (t * rowsPerTask)..<min(M, (t + 1) * rowsPerTask) {
                let qpos = p0 + i / nH
                let lo = (swa ? max(0, qpos - W + 1) : 0) - kStart
                let hi = qpos - kStart
                let row = sc.advanced(by: i * Tk)
                let valid = row.advanced(by: lo)
                let cnt = vDSP_Length(hi - lo + 1)
                var mx: Float = 0
                vDSP_maxv(valid, 1, &mx, cnt)
                var neg = -mx
                vDSP_vsadd(valid, 1, &neg, valid, 1, cnt)
                var n32 = Int32(cnt)
                vvexpf(valid, valid, &n32)
                var sum: Float = 0
                vDSP_sve(valid, 1, &sum, cnt)
                var inv = 1 / sum
                vDSP_vsmul(valid, 1, &inv, valid, 1, cnt)
                if lo > 0 { vDSP_vclr(row, 1, vDSP_Length(lo)) }
                if hi + 1 < Tk { vDSP_vclr(row.advanced(by: hi + 1), 1, vDSP_Length(Tk - hi - 1)) }
            }
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(M), Int32(hd), Int32(Tk), 1,
                    sc, Int32(Tk), vCache[kv].advanced(by: kStart * hd), Int32(hd), 0, of, Int32(hd))
        Self.f32ToF16(of, out, M * hd)
        if rows < S { memset(out.advanced(by: M * hd), 0, (S - rows) * nH * hd * 2) }
    }

    private static func f16ToF32(_ src: UnsafePointer<Float16>, _ dst: UnsafeMutablePointer<Float>, _ count: Int) {
        var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
        var d = vImage_Buffer(data: UnsafeMutableRawPointer(dst), height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
        vImageConvert_Planar16FtoPlanarF(&s, &d, 0)
    }

    private static func f32ToF16(_ src: UnsafePointer<Float>, _ dst: UnsafeMutablePointer<Float16>, _ count: Int) {
        var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
        var d = vImage_Buffer(data: UnsafeMutableRawPointer(dst), height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
        vImageConvert_PlanarFtoPlanar16F(&s, &d, 0)
    }
}
