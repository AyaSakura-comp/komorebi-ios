import Foundation
import KomorebiIPC

/// High-level Komorebi text refiner wrapping `Gemma4MetalEngine` (foreground primary),
/// `Gemma4ANEEngine` (background-safe NPU if installed), and `Gemma4CPUEngine` (background/Simulator fallback).
public actor KomorebiRefiner {
    public private(set) var gguf: GGUFFile?
    public private(set) var tokenizer: Gemma4Tokenizer?
    public private(set) var metalEngine: Gemma4MetalEngine?
    public private(set) var cpuEngine: Gemma4CPUEngine?
    /// Direct-ANE engine (private MIL compiler, small-dictionary LM head, optional MTP drafter). Default backend.
    public private(set) var aneDirect: Gemma4ANEDirectEngine?
    private var aneEngineBox: AnyObject?
    private var metalMaxContext = 1024

    private var cancelCounter: UInt64 = 0

    public init() {}

    public func loadGGUF(url: URL, preferMetal: Bool = true, maxContext: Int = 1024) throws {
        let file = try GGUFFile(url: url, loadTokenizerArrays: true)
        let tok = try Gemma4Tokenizer(metadata: file.metadata)
        self.gguf = file
        self.tokenizer = tok

        self.metalMaxContext = maxContext
        if preferMetal {
            self.metalEngine = try? Gemma4MetalEngine(gguf: file, maxContext: maxContext)
        }
        self.cpuEngine = try Gemma4CPUEngine(gguf: file, maxContext: maxContext)
    }

    /// Builds (first run: re-quantises weights to ANE LUT palettes and caches them) and loads the direct-ANE engine,
    /// attaches the MTP drafter when `drafterDirectory` holds one, and warms every program up.
    public func loadANEDirect(cacheDirectory: URL, drafterDirectory: URL?, maxContext: Int = 1024,
                              progress: (@Sendable (String) -> Void)? = nil) throws {
        guard let gguf, let tokenizer else { throw GemmaEngineError.inferenceFailed("load the GGUF first") }
        try? FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        let dict = FlashHeadLMHead.smallDictionary(vocab: tokenizer.vocab, tokenTypes: tokenizer.tokenTypes)
        let engine = try Gemma4ANEDirectEngine(
            gguf: gguf,
            options: .init(lutBits: 4, lutGroup: 8, prefillChunk: 64, verifyChunk: 4, maxContext: maxContext, cacheDirectory: cacheDirectory),
            dictionary: dict,
            progress: progress
        )
        if let drafterDirectory {
            engine.drafter = try? Gemma4MTPDrafter(directory: drafterDirectory)
            engine.draftTokens = 3
        }
        progress?("ANE 暖機中…")
        try engine.warmUp()
        self.aneDirect = engine
    }

    /// Metal is optional (it heats the phone); it is created only when the user picks it.
    public func ensureMetalLoaded() {
        guard metalEngine == nil, let gguf else { return }
        metalEngine = try? Gemma4MetalEngine(gguf: gguf, maxContext: metalMaxContext)
    }

    public var aneDirectDescription: String? { aneDirect?.backendName }

    public func loadTokenizerFromGGUF(url: URL) throws {
        let file = try GGUFFile(url: url, loadTokenizerArrays: true)
        self.gguf = file
        self.tokenizer = try Gemma4Tokenizer(metadata: file.metadata)
    }

    @available(iOS 18.0, macOS 15.0, *)
    public func loadANECoreML(directory: URL) throws {
        self.aneEngineBox = try Gemma4ANEEngine(directory: directory)
    }

    public var isLoaded: Bool {
        tokenizer != nil && (metalEngine != nil || cpuEngine != nil || aneEngineBox != nil || aneDirect != nil)
    }

    public var isANEAvailable: Bool {
        aneDirect != nil || aneEngineBox != nil
    }

    public func cancelPending() {
        cancelCounter &+= 1
    }

    /// Cleans one finalized ASR utterance (Sentence mode) or first utterance of Draft mode.
    public func refineSentence(
        raw: String,
        context: String = "",
        inBackground: Bool = false,
        preference: InferenceBackendPreference = .auto
    ) async throws -> GenerationResult {
        let turns = KomorebiPrompts.refineTurns(raw: raw, context: context)
        let result = try await generateChat(
            turns: turns,
            maxNewTokens: max(96, raw.count * 3 + 32),
            inBackground: inBackground,
            preference: preference
        )
        let cleaned = KomorebiPrompts.cleanModelOutput(result.text)
        let accepted = KomorebiPrompts.acceptRefined(raw: raw, refined: cleaned) ? cleaned : raw
        return GenerationResult(text: accepted, tokenIDs: result.tokenIDs, stats: result.stats)
    }

    /// Integrates a newly finalized ASR utterance into an existing live draft,
    /// mirroring `GemmaRefiner.draft_merge` + `retry_unchanged` in `realtime_service.py`.
    public func draftMerge(
        draft: String,
        raw: String,
        inBackground: Bool = false,
        preference: InferenceBackendPreference = .auto
    ) async throws -> GenerationResult {
        let trimmedDraft = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedDraft.isEmpty {
            return try await refineSentence(raw: raw, context: "", inBackground: inBackground, preference: preference)
        }

        let turns = KomorebiPrompts.draftTurns(draft: trimmedDraft, raw: raw)
        let maxToks = max(128, trimmedDraft.count + raw.count * 3 + 48)
        let first = try await generateChat(turns: turns, maxNewTokens: maxToks, inBackground: inBackground, preference: preference)
        let cleaned = KomorebiPrompts.cleanModelOutput(first.text)

        // Retry with worked few-shot correction examples if model returned unchanged old draft (E2B parity).
        if cleaned == trimmedDraft || cleaned.isEmpty {
            let retryTurns = KomorebiPrompts.draftRetryTurns(draft: trimmedDraft, raw: raw)
            let retried = try await generateChat(turns: retryTurns, maxNewTokens: maxToks, inBackground: inBackground, preference: preference)
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
    public func selectionEdit(
        before: String,
        selected: String,
        after: String,
        raw: String,
        inBackground: Bool = false,
        preference: InferenceBackendPreference = .auto
    ) async throws -> GenerationResult {
        let turns = KomorebiPrompts.selectionTurns(before: before, selected: selected, after: after, raw: raw)
        let res = try await generateChat(
            turns: turns,
            maxNewTokens: max(128, selected.count * 3 + raw.count * 2 + 48),
            inBackground: inBackground,
            preference: preference
        )
        let text = KomorebiPrompts.parseSelectionJSON(res.text) ?? selected
        return GenerationResult(text: text, tokenIDs: res.tokenIDs, stats: res.stats)
    }

    public func generateChat(
        turns: [ChatTurn],
        maxNewTokens: Int = 256,
        inBackground: Bool = false,
        preference: InferenceBackendPreference = .auto
    ) async throws -> GenerationResult {
        guard let tokenizer else {
            throw GemmaEngineError.inferenceFailed("Tokenizer not loaded")
        }

        let fullPrompt = tokenizer.formatChatPrompt(turns: turns, addGenerationPrompt: true)
        let inputIDs = tokenizer.encode(fullPrompt, addBOS: true)

        // Compute static prefix length (`<bos><|turn>system\n...<turn|>\n<|turn>user\n`) for KV cache snapshot reuse.
        var prefixLen = 0
        if let first = turns.first, first.role == .system {
            let sysHeader = "<|turn>system\n\(first.content.trimmingCharacters(in: .whitespacesAndNewlines))<turn|>\n"
            prefixLen = tokenizer.encode(sysHeader, addBOS: true).count
        }

        let stopIDs: Set<Int32> = [tokenizer.eosTokenID, tokenizer.turnEndTokenID]

        let resolved = resolveBackend(inBackground: inBackground, preference: preference)
        let (outIDs, stats): ([Int32], GenerationStats)

        switch resolved {
        case .ane:
            if let direct = aneDirect {
                (outIDs, stats) = try direct.generate(inputIDs: inputIDs, maxNewTokens: maxNewTokens, stopTokenIDs: stopIDs, staticPrefixLength: prefixLen)
            } else if #available(iOS 18.0, macOS 15.0, *), let ane = aneEngineBox as? Gemma4ANEEngine {
                (outIDs, stats) = try await ane.generate(inputIDs: inputIDs, maxNewTokens: maxNewTokens, stopTokenIDs: stopIDs)
            } else if let cpu = cpuEngine {
                (outIDs, stats) = try cpu.generate(inputIDs: inputIDs, maxNewTokens: maxNewTokens, stopTokenIDs: stopIDs, staticPrefixLength: prefixLen)
            } else {
                throw GemmaEngineError.inferenceFailed("No ANE or CPU engine loaded")
            }

        case .metal:
            ensureMetalLoaded()
            guard let metal = metalEngine else {
                if let cpu = cpuEngine {
                    (outIDs, stats) = try cpu.generate(inputIDs: inputIDs, maxNewTokens: maxNewTokens, stopTokenIDs: stopIDs, staticPrefixLength: prefixLen)
                } else {
                    throw GemmaEngineError.metalUnavailable
                }
                break
            }
            (outIDs, stats) = try metal.generate(
                inputIDs: inputIDs,
                maxNewTokens: maxNewTokens,
                stopTokenIDs: stopIDs,
                staticPrefixLength: prefixLen
            )

        case .cpu, .auto:
            guard let cpu = cpuEngine else {
                throw GemmaEngineError.inferenceFailed("CPU engine not loaded")
            }
            (outIDs, stats) = try cpu.generate(
                inputIDs: inputIDs,
                maxNewTokens: maxNewTokens,
                stopTokenIDs: stopIDs,
                staticPrefixLength: prefixLen
            )
        }

        let decoded = tokenizer.decode(outIDs, skipSpecialTokens: true)
        return GenerationResult(text: decoded, tokenIDs: outIDs, stats: stats)
    }

    private func resolveBackend(inBackground: Bool, preference: InferenceBackendPreference) -> InferenceBackendPreference {
        // Critical iOS safety guard: never submit Metal command buffers while Host App is backgrounded
        // (`kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted`).
        let hasANE = aneDirect != nil || aneEngineBox != nil
        if inBackground {
            if preference == .cpu { return .cpu }
            return hasANE ? .ane : .cpu
        }
        switch preference {
        case .auto, .ane:
            // ANE first: the GPU runs hot on the phone; Metal only when no ANE engine could be built.
            if hasANE { return .ane }
            ensureMetalLoaded()
            return metalEngine != nil ? .metal : .cpu
        case .metal:
            ensureMetalLoaded()
            return metalEngine != nil ? .metal : (hasANE ? .ane : .cpu)
        case .cpu:
            return .cpu
        }
    }
}
