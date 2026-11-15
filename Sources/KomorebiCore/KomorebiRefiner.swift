import Foundation
import KomorebiIPC
import KomorebiANEDirect

/// High-level Komorebi text refiner on top of the direct-ANE Gemma 4 E2B engine (`Gemma4ANEDirectEngine`):
/// sentence clean-up, draft merging and lasso rewrites with the Komorebi prompts.
public actor KomorebiRefiner {
    public private(set) var gguf: GGUFFile?
    public private(set) var tokenizer: Gemma4Tokenizer?
    /// Direct-ANE engine (private MIL compiler, small-dictionary LM head, optional MTP drafter).
    public private(set) var aneDirect: Gemma4ANEDirectEngine?

    private var cancelCounter: UInt64 = 0
    /// How the last `draftMerge` ended (for the app's diagnostics log).
    public private(set) var lastMergeNote = ""

    /// Live previews are cancelled from any thread (the actor is busy generating), so the flag lives outside it.
    private final class PreviewGate: @unchecked Sendable {
        private let lock = NSLock()
        private var value: UInt64 = 0
        var current: UInt64 { lock.lock(); defer { lock.unlock() }; return value }
        func bump() { lock.lock(); value &+= 1; lock.unlock() }
    }
    private nonisolated let previewGate = PreviewGate()

    public init() {}

    /// Maps the GGUF (weights stay on disk until the ANE engine re-quantises them) and loads the tokenizer.
    public func loadGGUF(url: URL) throws {
        let file = try GGUFFile(url: url, loadTokenizerArrays: true)
        self.tokenizer = try Gemma4Tokenizer(metadata: file.metadata)
        self.gguf = file
    }

    /// Builds (first run: re-quantises weights to ANE LUT palettes and caches them) and loads the direct-ANE engine,
    /// attaches the MTP drafter when `drafterDirectory` holds one and, with `warmUpNow`, warms every program up.
    /// Returns a timing breakdown for the log.
    @discardableResult
    public func loadANEDirect(cacheDirectory: URL, drafterDirectory: URL?, maxContext: Int = 1024, warmUpNow: Bool = true,
                              progress: (@Sendable (String) -> Void)? = nil) throws -> String {
        guard let gguf, let tokenizer else { throw GemmaEngineError.inferenceFailed("load the GGUF first") }
        try? FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        let t0 = CFAbsoluteTimeGetCurrent()
        let staged0 = ane_rt_staged_bytes(), stageS0 = ane_rt_stage_seconds(), desc0 = ane_rt_describe_seconds()
        let dict = FlashHeadLMHead.smallDictionary(vocab: tokenizer.vocab, tokenTypes: tokenizer.tokenTypes)
        let t1 = CFAbsoluteTimeGetCurrent()
        let engine = try Gemma4ANEDirectEngine(
            gguf: gguf,
            options: .init(lutBits: 4, lutGroup: 8, prefillChunk: 64, verifyChunk: 4, maxContext: maxContext, cacheDirectory: cacheDirectory),
            dictionary: dict,
            progress: progress
        )
        let t2 = CFAbsoluteTimeGetCurrent()
        if let drafterDirectory {
            engine.drafter = try? Gemma4MTPDrafter(directory: drafterDirectory)
            engine.draftTokens = 3
        }
        let t3 = CFAbsoluteTimeGetCurrent()
        var warm = 0.0
        if warmUpNow {
            progress?("ANE 暖機中…")
            try engine.warmUp()
            warm = CFAbsoluteTimeGetCurrent() - t3
        }
        self.aneDirect = engine
        return String(format: "dictionary %.2f s, engine %.2f s (MIL %.2f s, describe/hash %.2f s, ANE load %.2f s, staged %.0f MB in %.2f s), drafter %.2f s%@",
                      t1 - t0, t2 - t1, engine.milSeconds, ane_rt_describe_seconds() - desc0, engine.compileSeconds,
                      Double(ane_rt_staged_bytes() - staged0) / 1e6, ane_rt_stage_seconds() - stageS0, t3 - t2,
                      warmUpNow ? String(format: ", warm-up %.2f s", warm) : "")
    }

    /// First evaluation of every ANE program after loading is slow; this runs it once so dictation is fast.
    /// Requests that arrive meanwhile wait for it (actor). Returns the seconds it took.
    public func warmUpANE() throws -> Double {
        guard let engine = aneDirect else { return 0 }
        let t = CFAbsoluteTimeGetCurrent()
        try engine.warmUp()
        return CFAbsoluteTimeGetCurrent() - t
    }

    public var aneDirectDescription: String? { aneDirect?.backendName }

    /// True once the ANE engine is ready to generate.
    public var isLoaded: Bool {
        tokenizer != nil && aneDirect != nil
    }

    /// Stops a running live preview at its next token (a finished sentence takes priority).
    public nonisolated func cancelPreview() {
        previewGate.bump()
    }

    /// Live preview while the sentence is still being spoken: the per-sentence clean-up of the partial transcript.
    /// Returns nil when it was cancelled or the output is unusable.
    public func previewSentence(raw: String, context: String) async throws -> GenerationResult? {
        let gate = previewGate
        let token = gate.current
        let turns = KomorebiPrompts.refineTurns(raw: raw, context: context)
        let result = try await generateChat(turns: turns, maxNewTokens: max(96, raw.count * 3 + 32),
                                            shouldCancel: { gate.current != token })
        guard gate.current == token else { return nil }
        let cleaned = KomorebiPrompts.cleanModelOutput(result.text)
        guard KomorebiPrompts.acceptRefined(raw: raw, refined: cleaned) else { return nil }
        return GenerationResult(text: cleaned, tokenIDs: result.tokenIDs, stats: result.stats)
    }

    public func cancelPending() {
        cancelCounter &+= 1
    }

    /// Cleans one finalized ASR utterance (Sentence mode) or first utterance of Draft mode.
    public func refineSentence(raw: String, context: String = "") async throws -> GenerationResult {
        let turns = KomorebiPrompts.refineTurns(raw: raw, context: context)
        let result = try await generateChat(turns: turns, maxNewTokens: max(96, raw.count * 3 + 32))
        let cleaned = KomorebiPrompts.cleanModelOutput(result.text)
        let accepted = KomorebiPrompts.acceptRefined(raw: raw, refined: cleaned) ? cleaned : raw
        return GenerationResult(text: accepted, tokenIDs: result.tokenIDs, stats: result.stats)
    }

    /// Integrates a newly finalized ASR utterance into an existing live draft,
    /// mirroring `GemmaRefiner.draft_merge` + `retry_unchanged` in `realtime_service.py`.
    public func draftMerge(draft: String, raw: String) async throws -> GenerationResult {
        let trimmedDraft = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedDraft.isEmpty {
            lastMergeNote = "first sentence (refine)"
            return try await refineSentence(raw: raw, context: "")
        }

        let turns = KomorebiPrompts.draftTurns(draft: trimmedDraft, raw: raw)
        let heard = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let maxToks = min(2048, trimmedDraft.count + 2 * raw.count + 128)
        let first = try await generateChat(turns: turns, maxNewTokens: maxToks)
        let cleaned = KomorebiPrompts.cleanModelOutput(first.text)

        // The old draft came back (Komorebi checks for an identical draft; E2B often also touches a comma, so
        // punctuation and spaces are ignored here): what was just said was dropped. Ask once more, leaning on it.
        let unchanged = cleaned.isEmpty || Self.words(cleaned) == Self.words(trimmedDraft)
        if unchanged && !heard.isEmpty {
            let retryTurns = KomorebiPrompts.draftRetryTurns(draft: trimmedDraft, raw: raw)
            let retried = try await generateChat(turns: retryTurns, maxNewTokens: maxToks)
            let retryClean = KomorebiPrompts.cleanModelOutput(retried.text)
            if KomorebiPrompts.acceptDraftRetry(draft: trimmedDraft, raw: raw, updated: retryClean) {
                lastMergeNote = "unchanged; retry accepted"
                return GenerationResult(text: retryClean, tokenIDs: retried.tokenIDs, stats: retried.stats)
            }
            // Still nothing usable: keep the sentence as heard rather than lose it.
            lastMergeNote = "unchanged; retry rejected (\(retryClean.prefix(40))); appended as heard"
            return GenerationResult(text: KomorebiPrompts.appendDraft(trimmedDraft, heard), tokenIDs: retried.tokenIDs, stats: retried.stats)
        }

        if KomorebiPrompts.acceptDraft(draft: trimmedDraft, raw: raw, updated: cleaned) {
            lastMergeNote = "rewritten"
            return GenerationResult(text: cleaned, tokenIDs: first.tokenIDs, stats: first.stats)
        }
        // Rejected rewrite: keep the sentence, appended as heard (the reference service does the same).
        lastMergeNote = "rewrite rejected (\(cleaned.prefix(40))); appended as heard"
        return GenerationResult(text: KomorebiPrompts.appendDraft(trimmedDraft, heard), tokenIDs: first.tokenIDs, stats: first.stats)
    }

    /// Text without punctuation and spaces, to tell a real edit from a re-punctuated copy.
    static func words(_ s: String) -> String {
        String(s.unicodeScalars.filter { !CharacterSet.punctuationCharacters.contains($0) && !CharacterSet.whitespacesAndNewlines.contains($0) && !CharacterSet.symbols.contains($0) }.map(Character.init))
    }

    /// Lasso selection rewrite (`selection_edit`): replaces/deletes/rewrites `selected` based on spoken instruction.
    public func selectionEdit(before: String, selected: String, after: String, raw: String) async throws -> GenerationResult {
        let turns = KomorebiPrompts.selectionTurns(before: before, selected: selected, after: after, raw: raw)
        let res = try await generateChat(turns: turns, maxNewTokens: max(128, selected.count * 3 + raw.count * 2 + 48))
        // Komorebi: when the rewrite is unusable, the sentence replaces the selection as heard.
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if let parsed = KomorebiPrompts.parseSelectionJSON(res.text),
           KomorebiPrompts.acceptSelection(selected: selected, raw: raw, replacement: parsed) {
            text = parsed.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return GenerationResult(text: text, tokenIDs: res.tokenIDs, stats: res.stats)
    }

    public func generateChat(turns: [ChatTurn], maxNewTokens: Int = 256,
                             shouldCancel: (@Sendable () -> Bool)? = nil) async throws -> GenerationResult {
        guard let tokenizer else {
            throw GemmaEngineError.inferenceFailed("Tokenizer not loaded")
        }
        guard let engine = aneDirect else { throw GemmaEngineError.engineNotLoaded }

        let fullPrompt = tokenizer.formatChatPrompt(turns: turns, addGenerationPrompt: true)
        let inputIDs = tokenizer.encode(fullPrompt, addBOS: true)

        // Static prefix (`<bos><|turn>system\n...<turn|>\n`) whose KV cache is reused between dictations.
        var prefixLen = 0
        if let first = turns.first, first.role == .system {
            let sysHeader = "<|turn>system\n\(first.content.trimmingCharacters(in: .whitespacesAndNewlines))<turn|>\n"
            prefixLen = tokenizer.encode(sysHeader, addBOS: true).count
        }

        let stopIDs: Set<Int32> = [tokenizer.eosTokenID, tokenizer.turnEndTokenID]
        let (outIDs, stats) = try engine.generate(inputIDs: inputIDs, maxNewTokens: maxNewTokens,
                                                  stopTokenIDs: stopIDs, staticPrefixLength: prefixLen,
                                                  shouldCancel: shouldCancel)
        let decoded = tokenizer.decode(outIDs, skipSpecialTokens: true)
        return GenerationResult(text: decoded, tokenIDs: outIDs, stats: stats)
    }
}
