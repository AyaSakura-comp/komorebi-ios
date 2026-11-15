import XCTest
@testable import KomorebiCore
@testable import KomorebiIPC

final class KomorebiCoreTests: XCTestCase {
    private let ggufPath = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("models/gemma-4-e2b/gemma-4-E2B_q4_0-it.gguf")

    func testPromptWrappersAndSanityFilters() {
        XCTAssertTrue(KomorebiPrompts.acceptRefined(raw: "呃我們明天三點開會", refined: "我們明天三點開會。"))
        XCTAssertFalse(KomorebiPrompts.acceptRefined(raw: "明天開會", refined: "身為 AI 語言模型，我很樂意幫您整理。"))
        XCTAssertEqual(KomorebiPrompts.parseSelectionJSON("```json\n{\"text\": \"太棒了吧！\"}\n```"), "太棒了吧！")
        XCTAssertEqual(KomorebiPrompts.parseSelectionJSON("{\"text\": \"\"}"), "")
        XCTAssertFalse(KomorebiPrompts.acceptDraftRetry(draft: "明天早上十點要在會議室開會報告季度成果", raw: "把報告刪掉", updated: "太短了"))
    }

    func testGGUFLoadAndTokenizerRoundTrip() throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: ggufPath.path), "Gemma 4 E2B GGUF not found at ~/models")

        let gguf = try GGUFFile(url: ggufPath, loadTokenizerArrays: true)
        let cfg = Gemma4Config.from(gguf: gguf.metadata)

        XCTAssertEqual(cfg.hiddenSize, 1536)
        XCTAssertEqual(cfg.numLayers, 35)
        XCTAssertEqual(cfg.numOwnKVLayers, 15)
        XCTAssertEqual(cfg.numKVSharedLayers, 20)
        XCTAssertEqual(cfg.slidingHeadDim, 256)
        XCTAssertEqual(cfg.fullHeadDim, 512)
        XCTAssertEqual(cfg.vocabSize, 262_144)

        let tokenizer = try Gemma4Tokenizer(metadata: gguf.metadata)
        let sample = "明天下午三點開會，喔不對，改成四點！Hello 123\n換行測試"
        let encoded = tokenizer.encode(sample, addBOS: false)
        XCTAssertFalse(encoded.isEmpty)
        let decoded = tokenizer.decode(encoded, skipSpecialTokens: true)
        XCTAssertEqual(decoded, sample)

        let chatIDs = tokenizer.encodeChat(turns: KomorebiPrompts.refineTurns(raw: "那個我今天想去買咖啡"), addGenerationPrompt: true)
        XCTAssertEqual(chatIDs.first, 2) // <bos>
        XCTAssertTrue(chatIDs.contains(105)) // <|turn>
        XCTAssertTrue(chatIDs.contains(106)) // <turn|>
    }

    func testMetalEngineEndToEndAndBenchmark() async throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: ggufPath.path), "Gemma 4 E2B GGUF not found at ~/models")

        let refiner = KomorebiRefiner()
        try await refiner.loadGGUF(url: ggufPath, preferMetal: true, maxContext: 1024)
        let loaded = await refiner.isLoaded
        XCTAssertTrue(loaded)

        // 1. Cold prefill + decode (sentence with hesitation & self-correction)
        let res1 = try await refiner.refineSentence(
            raw: "呃那個我們明天下午三點開會，喔不對不對，改成四點半好不好",
            preference: .metal
        )
        print("CLEANUP 1: '\(res1.text)' | backend=\(res1.stats.backend) prefill=\(res1.stats.prefillTokens) tok @ \(String(format: "%.1f", res1.stats.prefillTPS)) tok/s | decode=\(res1.stats.decodeTokens) tok @ \(String(format: "%.1f", res1.stats.decodeTPS)) tok/s")
        XCTAssertTrue(res1.text.contains("四點半") || res1.text.contains("四點"), "Expected self-correction to 四點半, got: \(res1.text)")
        XCTAssertFalse(res1.text.contains("三點"))

        // 2. Warm prefix KV cache reuse + homophone / punctuation cleanup
        let res2 = try await refiner.refineSentence(
            raw: "我等一下要去買一杯冰拿鐵跟一塊起司蛋糕",
            preference: .metal
        )
        print("CLEANUP 2 (prefix cached \(res2.stats.prefixCachedTokens) tok): '\(res2.text)' | prefill=\(res2.stats.prefillTokens) tok @ \(String(format: "%.1f", res2.stats.prefillTPS)) tok/s | decode=\(res2.stats.decodeTokens) tok @ \(String(format: "%.1f", res2.stats.decodeTPS)) tok/s")
        XCTAssertGreaterThan(res2.stats.prefixCachedTokens, 20)
        XCTAssertTrue(res2.text.contains("拿鐵"))

        // 3. Draft correction merge (Komorebi canonical worked example)
        let res3 = try await refiner.draftMerge(
            draft: "明天要帶雨傘跟護照。",
            raw: "護照不用帶了",
            preference: .metal
        )
        print("DRAFT MERGE: '\(res3.text)' | prefill=\(String(format: "%.1f", res3.stats.prefillTPS)) tok/s | decode=\(String(format: "%.1f", res3.stats.decodeTPS)) tok/s")
        XCTAssertTrue(res3.text.contains("雨傘"))
        XCTAssertFalse(res3.text.contains("護照"))

        // 4. Selection JSON edit
        let res4 = try await refiner.selectionEdit(
            before: "今天天氣真好，",
            selected: "我們等一下去看電影",
            after: "，好嗎？",
            raw: "換句話說，我們改去吃日本料理",
            preference: .metal
        )
        print("SELECTION EDIT: '\(res4.text)'")
        XCTAssertTrue(res4.text.contains("日本料理"))

        // 5. Verify Metal GPU matches CPU Float32 reference tokens identically
        let cpuCheck = try await refiner.refineSentence(
            raw: "呃那個我們明天下午三點開會，喔不對不對，改成四點半好不好",
            preference: .cpu
        )
        print("CPU CHECK: '\(cpuCheck.text)' | backend=\(cpuCheck.stats.backend) prefill=\(String(format: "%.1f", cpuCheck.stats.prefillTPS)) tok/s | decode=\(String(format: "%.1f", cpuCheck.stats.decodeTPS)) tok/s")
        XCTAssertEqual(cpuCheck.text, res1.text)
    }

    /// Prefill micro-benchmark + per-kernel GPU profile on the real Komorebi refine prompt.
    func testMetalPrefillProfile() throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: ggufPath.path), "Gemma 4 E2B GGUF not found at ~/models")
        let gguf = try GGUFFile(url: ggufPath, loadTokenizerArrays: true)
        let tok = try Gemma4Tokenizer(metadata: gguf.metadata)
        let engine = try Gemma4MetalEngine(gguf: gguf, maxContext: 1024)
        let turns = KomorebiPrompts.refineTurns(raw: "呃那個我們明天下午三點開會，喔不對不對，改成四點半好不好")
        let ids = tok.encode(tok.formatChatPrompt(turns: turns, addGenerationPrompt: true), addBOS: true)
        let sys = "<|turn>system\n\(turns[0].content.trimmingCharacters(in: .whitespacesAndNewlines))<turn|>\n"
        let prefixLen = tok.encode(sys, addBOS: true).count
        let stops: Set<Int32> = [tok.eosTokenID, tok.turnEndTokenID]

        func bestPrefillTPS(staticPrefix: Int, runs: Int = 5) throws -> (Double, Int32) {
            var best = 0.0
            var first: Int32 = -1
            for _ in 0..<runs {
                engine.clearPrefixCache()
                let r = try engine.generate(inputIDs: ids, maxNewTokens: 1, stopTokenIDs: stops, staticPrefixLength: staticPrefix)
                best = max(best, r.stats.prefillTPS)
                first = r.tokenIDs.first ?? -1
            }
            return (best, first)
        }

        _ = try bestPrefillTPS(staticPrefix: 0, runs: 2) // warm-up (pipeline compile, page-in)
        let (oneShot, t1) = try bestPrefillTPS(staticPrefix: 0)
        let (withPrefix, t2) = try bestPrefillTPS(staticPrefix: prefixLen)
        print(String(format: "PREFILL BENCH: %d tok | one-shot %.1f tok/s | refiner-path (prefix %d + snapshot) %.1f tok/s | first token %d/%d", ids.count, oneShot, prefixLen, withPrefix, t1, t2))
        XCTAssertEqual(t1, t2)

        engine.profilingEnabled = true
        engine.resetProfile()
        engine.clearPrefixCache()
        _ = try engine.generate(inputIDs: ids, maxNewTokens: 1, stopTokenIDs: stops, staticPrefixLength: prefixLen)
        print("PREFILL PROFILE (\(ids.count) tok, refiner path):\n" + engine.profileReport())
        engine.profilingEnabled = false
    }

    func testANECoreMLEngineAndBenchmark() async throws {
        guard #available(macOS 15.0, iOS 18.0, *) else {
            throw XCTSkip("CoreML MLState requires macOS 15+ / iOS 18+")
        }
        let coremlDir = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("models/gemma-4-e2b-coreml")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: coremlDir.appendingPathComponent("chunk_1.mlmodelc").path), "ANE CoreML bundle not found at ~/models/gemma-4-e2b-coreml")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: ggufPath.path), "GGUF tokenizer not found")

        let refiner = KomorebiRefiner()
        try await refiner.loadTokenizerFromGGUF(url: ggufPath)
        try await refiner.loadANECoreML(directory: coremlDir)

        let res1 = try await refiner.refineSentence(
            raw: "呃那個我們明天下午三點開會，喔不對不對，改成四點半好不好",
            preference: .ane
        )
        print("ANE CLEANUP 1: '\(res1.text)' | backend=\(res1.stats.backend) prefill=\(res1.stats.prefillTokens) tok @ \(String(format: "%.1f", res1.stats.prefillTPS)) tok/s | decode=\(res1.stats.decodeTokens) tok @ \(String(format: "%.1f", res1.stats.decodeTPS)) tok/s")
        XCTAssertFalse(res1.text.isEmpty)

        let res2 = try await refiner.refineSentence(
            raw: "我等一下要去買一杯冰拿鐵跟一塊起司蛋糕",
            preference: .ane
        )
        print("ANE CLEANUP 2 (prefix cached \(res2.stats.prefixCachedTokens) tok): '\(res2.text)' | prefill=\(res2.stats.prefillTokens) tok @ \(String(format: "%.1f", res2.stats.prefillTPS)) tok/s | decode=\(res2.stats.decodeTokens) tok @ \(String(format: "%.1f", res2.stats.decodeTPS)) tok/s")
        XCTAssertGreaterThan(res2.stats.prefixCachedTokens, 20)
    }
}
