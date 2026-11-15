import XCTest
@testable import KomorebiCore
@testable import KomorebiIPC
import KomorebiANEDirect

final class KomorebiCoreTests: XCTestCase {
    private let ggufPath = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("models/gemma-4-e2b/gemma-4-E2B_q4_0-it.gguf")

    func testPromptWrappersAndSanityFilters() {
        XCTAssertTrue(KomorebiPrompts.acceptRefined(raw: "呃我們明天三點開會", refined: "我們明天三點開會。"))
        XCTAssertFalse(KomorebiPrompts.acceptRefined(raw: "明天開會", refined: String(repeating: "這是一篇回答而不是整理後的句子。", count: 3)))
        XCTAssertFalse(KomorebiPrompts.acceptDraftRetry(draft: "明天早上十點開會。", raw: "嗯", updated: "明天早上十點開會。"))
        XCTAssertTrue(KomorebiPrompts.acceptSelection(selected: "香蕉、", raw: "刪掉", replacement: ""))
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

    func testAppendDraftSpacing() {
        XCTAssertEqual(KomorebiPrompts.appendDraft("明天開會。", "改成四點。"), "明天開會。改成四點。")
        XCTAssertEqual(KomorebiPrompts.appendDraft("Hello", "world"), "Hello world")
        XCTAssertEqual(KomorebiPrompts.appendDraft("版本2", "3"), "版本2 3")
        XCTAssertEqual(KomorebiPrompts.appendDraft("", "第一句"), "第一句")
    }

    /// End to end through `KomorebiRefiner` on the Neural Engine (macOS on Apple silicon, or a device).
    func testRefinerOnANE() async throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: ggufPath.path), "Gemma 4 E2B GGUF not found at ~/models")
        try XCTSkipUnless(ane_rt_available(), "AppleNeuralEngine private framework unavailable")
        let drafter = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("models/gemma-4-e2b-assistant")
        let cache = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("komorebi-ane")

        let refiner = KomorebiRefiner()
        try await refiner.loadGGUF(url: ggufPath)
        try await refiner.loadANEDirect(
            cacheDirectory: cache,
            drafterDirectory: FileManager.default.fileExists(atPath: drafter.appendingPathComponent("model.safetensors").path) ? drafter : nil
        )

        let res1 = try await refiner.refineSentence(raw: "呃那個我們明天下午三點開會，喔不對不對，改成四點半好不好")
        print("ANE REFINE: '\(res1.text)' | \(res1.stats.backend) prefill \(String(format: "%.0f", res1.stats.prefillTPS)) tok/s | decode \(String(format: "%.1f", res1.stats.decodeTPS)) tok/s")
        XCTAssertFalse(res1.text.isEmpty)
        XCTAssertTrue(res1.stats.backend.contains("ANE"))

        let merged = try await refiner.draftMerge(draft: res1.text, raw: "然後記得帶筆電")
        print("ANE DRAFT MERGE: '\(merged.text)' | decode \(String(format: "%.1f", merged.stats.decodeTPS)) tok/s")
        XCTAssertTrue(merged.text.count > res1.text.count)

        let deleted = try await refiner.selectionEdit(before: "我買了蘋果、", selected: "香蕉、", after: "和葡萄。", raw: "刪掉")
        let replaced = try await refiner.selectionEdit(before: "下午去", selected: "郵局", after: "寄包裹。", raw: "改成便利商店")
        print("ANE SELECTION: 刪掉 -> '\(deleted.text)' | 改成便利商店 -> '\(replaced.text)'")
        XCTAssertEqual(deleted.text, "")
        XCTAssertTrue(replaced.text.contains("便利商店"))
    }
}
