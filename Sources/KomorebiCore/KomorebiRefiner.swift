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
            return try await refineSentence(raw: raw, context: "")
        }

        let turns = KomorebiPrompts.draftTurns(draft: trimmedDraft, raw: raw)
        let maxToks = max(128, trimmedDraft.count + raw.count * 3 + 48)
        let first = try await generateChat(turns: turns, maxNewTokens: maxToks)
        let cleaned = KomorebiPrompts.cleanModelOutput(first.text)

        // Retry with worked few-shot correction examples if model returned unchanged old draft (E2B parity).
        if cleaned == trimmedDraft || cleaned.isEmpty {
            let retryTurns = KomorebiPrompts.draftRetryTurns(draft: trimmedDraft, raw: raw)
            let retried = try await generateChat(turns: retryTurns, maxNewTokens: maxToks)
            let retryClean = KomorebiPrompts.cleanModelOutput(retried.text)
            if !retryClean.isEmpty && KomorebiPrompts.acceptDraftRetry(draft: trimmedDraft, raw: raw, updated: retryClean) {
                return GenerationResult(text: retryClean, tokenIDs: retried.tokenIDs, stats: retried.stats)
            }
        }

        if KomorebiPrompts.acceptDraft(draft: trimmedDraft, raw: raw, updated: cleaned) && !cleaned.isEmpty {
            return GenerationResult(text: cleaned, tokenIDs: first.tokenIDs, stats: first.stats)
        }
        // Rejected rewrite: keep the sentence, appended as heard (the reference service does the same).
        return GenerationResult(text: KomorebiPrompts.appendDraft(trimmedDraft, raw.trimmingCharacters(in: .whitespacesAndNewlines)),
                                tokenIDs: first.tokenIDs, stats: first.stats)
    }

    /// Lasso selection rewrite (`selection_edit`): replaces/deletes/rewrites `selected` based on spoken instruction.
    public func selectionEdit(before: String, selected: String, after: String, raw: String) async throws -> GenerationResult {
        let turns = KomorebiPrompts.selectionTurns(before: before, selected: selected, after: after, raw: raw)
        let res = try await generateChat(turns: turns, maxNewTokens: max(128, selected.count * 3 + raw.count * 2 + 48))
        let text = KomorebiPrompts.parseSelectionJSON(res.text) ?? selected
        return GenerationResult(text: text, tokenIDs: res.tokenIDs, stats: res.stats)
    }

    public func generateChat(turns: [ChatTurn], maxNewTokens: Int = 256) async throws -> GenerationResult {
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
                                                  stopTokenIDs: stopIDs, staticPrefixLength: prefixLen)
        let decoded = tokenizer.decode(outIDs, skipSpecialTokens: true)
        return GenerationResult(text: decoded, tokenIDs: outIDs, stats: stats)
    }
}
