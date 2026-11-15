import XCTest
@testable import KomorebiCore
import KomorebiANEDirect

/// `MTP_DIAG=1`: first-draft accuracy of the MTP drafter against LUT4 and LUT8 targets, and where the target's
/// token ranked when the draft missed (top-5 / in the centroid candidate set at all).
final class MTPDiagTests: XCTestCase {
    func testDrafterAccuracyByTargetPrecision() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MTP_DIAG"] == "1", "set MTP_DIAG=1")
        let home = URL(fileURLWithPath: NSHomeDirectory())
        let gguf = try GGUFFile(url: home.appendingPathComponent("models/gemma-4-e2b/gemma-4-E2B_q4_0-it.gguf"), loadTokenizerArrays: true)
        let tok = try Gemma4Tokenizer(metadata: gguf.metadata)
        let dict = FlashHeadLMHead.smallDictionary(vocab: tok.vocab, tokenTypes: tok.tokenTypes)
        let cache = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("komorebi-ane")
        let stops: Set<Int32> = [tok.eosTokenID, tok.turnEndTokenID]
        let longDraft = "明天早上九點在三樓會議室開季度檢討會，請大家準時出席。會議前要先把上季的業績報表整理好，寄給所有部門主管。"
        let prompts: [(String, [ChatTurn])] = [
            ("chat 小葉", [ChatTurn(role: .system, content: "你是「小葉」，個性傲嬌但一定會幫忙。用台灣繁體中文口語回答，一百字內。"),
                         ChatTurn(role: .user, content: "我今天好累，推薦一個放鬆的方法。")]),
            ("zh story", [ChatTurn(role: .user, content: "用繁體中文寫一段大約一百字的台北旅遊介紹。")]),
            ("english", [ChatTurn(role: .user, content: "Explain in three sentences why the sky is blue.")]),
            ("draft rewrite", KomorebiPrompts.draftTurns(draft: longDraft, raw: "然後下午的討論改成三點")),
        ]
        for bits in [4, 8] {
            let engine = try Gemma4ANEDirectEngine(gguf: gguf, options: .init(lutBits: bits, lutGroup: 8, prefillChunk: 64, verifyChunk: 4,
                                                                             maxContext: 1024, cacheDirectory: cache), dictionary: dict)
            let drafter = try Gemma4MTPDrafter(directory: home.appendingPathComponent("models/gemma-4-e2b-assistant"))
            drafter.debugTopK = 5
            engine.drafter = drafter
            engine.draftTokens = 1
            engine.adaptiveDraft = false
            try engine.warmUp()
            for (name, turns) in prompts {
                engine.clearPrefixCache()
                let ids = tok.encode(tok.formatChatPrompt(turns: turns, addGenerationPrompt: true), addBOS: true)
                let r = try engine.generate(inputIDs: ids, maxNewTokens: 128, stopTokenIDs: stops)
                let p = engine.decodeProfile
                let n = Double(max(1, p.rankChecks))
                print(String(format: "MTPDIAG LUT%d %-13@ first draft right %4.1f%% | in drafter top-5 %4.1f%% | in its 4096 candidates %4.1f%% | %d steps, %.1f tok/s",
                             bits, name as NSString, Double(p.rankTop1) / n * 100, Double(p.rankTop5) / n * 100,
                             Double(p.inCandidates) / n * 100, p.rankChecks, r.stats.decodeTPS))
            }
        }
    }
}
