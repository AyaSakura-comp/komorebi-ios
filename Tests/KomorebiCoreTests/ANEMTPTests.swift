import XCTest
@testable import KomorebiCore
import KomorebiANEDirect

/// Small-dictionary LM head (ANE logits + exact rescoring) and MTP speculative decoding on the direct-ANE engine.
final class ANEMTPTests: XCTestCase {
    private let ggufPath = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("models/gemma-4-e2b/gemma-4-E2B_q4_0-it.gguf")
    private let drafterDir = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("models/gemma-4-e2b-assistant")

    func testSmallDictionaryHeadAndMTP() throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: ggufPath.path), "GGUF not found")
        let env = ProcessInfo.processInfo.environment
        let gguf = try GGUFFile(url: ggufPath, loadTokenizerArrays: true)
        let tok = try Gemma4Tokenizer(metadata: gguf.metadata)
        let cache = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("komorebi-ane")
        let dict = FlashHeadLMHead.smallDictionary(vocab: tok.vocab, tokenTypes: tok.tokenTypes)
        let opts = Gemma4ANEDirectEngine.Options(lutBits: Int(env["ANE_BITS"] ?? "") ?? 4, lutGroup: 8, prefillChunk: 64, verifyChunk: 4, maxContext: 2048, cacheDirectory: cache)
        let engine = try Gemma4ANEDirectEngine(gguf: gguf, options: opts, dictionary: dict, progress: { if $0.contains("LM") { print("  build: \($0)") } })
        print(String(format: "ENGINE: dictionary %d tokens, build %.1f s (ANE compile+load %.1f s)", dict.count, engine.buildSeconds, engine.compileSeconds))
        try engine.warmUp()

        let longDraft = "明天早上九點在三樓會議室開季度檢討會，請大家準時出席。會議前要先把上季的業績報表整理好，寄給所有部門主管。會後中午一起吃便當，下午兩點再討論新產品的上市時程。"
        let prompts: [(String, [ChatTurn])] = [
            ("refine", KomorebiPrompts.refineTurns(raw: "呃那個我們明天下午三點開會，喔不對不對，改成四點半好不好")),
            ("draft", KomorebiPrompts.draftTurns(draft: longDraft, raw: "然後下午的討論改成三點")),
            ("zh-story", [ChatTurn(role: .user, content: "用繁體中文寫一段大約一百字的台北旅遊介紹。")]),
            ("english", [ChatTurn(role: .user, content: "Explain in three sentences why the sky is blue.")]),
            ("japanese", [ChatTurn(role: .user, content: "日本語で、東京のおすすめ観光地を三つ紹介してください。")]),
        ]
        let stops: Set<Int32> = [tok.eosTokenID, tok.turnEndTokenID]
        let maxNew = Int(env["MTP_MAX_NEW"] ?? "") ?? 128
        func ids(_ turns: [ChatTurn]) -> [Int32] { tok.encode(tok.formatChatPrompt(turns: turns, addGenerationPrompt: true), addBOS: true) }

        // 1. Audit the small-dictionary ANE head (+ exact rescoring) against the exact full Q6_K head.
        engine.auditExactLMHead = true
        engine.resetAudit()
        for (_, turns) in prompts {
            engine.clearPrefixCache()
            _ = try engine.generate(inputIDs: ids(turns), maxNewTokens: 64, stopTokenIDs: stops)
        }
        print(String(format: "LM HEAD AUDIT: small-dictionary ANE head + exact rescoring disagrees with the exact full head in %d / %d decisions (%.2f%%)",
                     engine.auditANEHeadMismatch, engine.auditTotal, Double(engine.auditANEHeadMismatch) / Double(max(1, engine.auditTotal)) * 100))
        engine.auditExactLMHead = false

        // 2. Plain decoding.
        var base: [String: (text: String, tps: Double, n: Int)] = [:]
        for (name, turns) in prompts {
            engine.clearPrefixCache()
            let r = try engine.generate(inputIDs: ids(turns), maxNewTokens: maxNew, stopTokenIDs: stops)
            let text = tok.decode(r.tokenIDs, skipSpecialTokens: true)
            base[name] = (text, r.stats.decodeTPS, r.stats.decodeTokens)
            print(String(format: "PLAIN   %-9@ %3d tok @ %6.1f tok/s | %@", name as NSString, r.stats.decodeTokens, r.stats.decodeTPS,
                         String(text.prefix(70)).replacingOccurrences(of: "\n", with: " ") as NSString))
        }
        print("PROFILE plain decode (last prompt):\n" + engine.decodeProfile.report())

        // 3. MTP drafter.
        try XCTSkipUnless(FileManager.default.fileExists(atPath: drafterDir.appendingPathComponent("model.safetensors").path), "drafter not found")
        engine.drafter = try Gemma4MTPDrafter(directory: drafterDir)
        for gamma in [1, 2, 3] {
            engine.draftTokens = gamma
            var sumTok = 0, sumSec = 0.0, baseSec = 0.0, acc = 0, drafted = 0, steps = 0
            for (name, turns) in prompts {
                engine.clearPrefixCache()
                let r = try engine.generate(inputIDs: ids(turns), maxNewTokens: maxNew, stopTokenIDs: stops)
                let text = tok.decode(r.tokenIDs, skipSpecialTokens: true)
                let p = engine.decodeProfile
                sumTok += r.stats.decodeTokens; sumSec += r.stats.decodeSeconds
                baseSec += Double(base[name]!.n) / base[name]!.tps
                acc += p.acceptedTokens; drafted += p.draftedTokens; steps += p.verifySteps
                print(String(format: "MTP γ=%d %-9@ %3d tok @ %6.1f tok/s (x%.2f)  accept %3.0f%%  %.2f tok/step  same-text:%@", gamma, name as NSString,
                             r.stats.decodeTokens, r.stats.decodeTPS, r.stats.decodeTPS / base[name]!.tps,
                             p.draftedTokens > 0 ? Double(p.acceptedTokens) / Double(p.draftedTokens) * 100 : 0,
                             Double(p.acceptedTokens + p.verifySteps) / Double(max(1, p.verifySteps)),
                             (text == base[name]!.text ? "yes" : "NO") as NSString))
                if text != base[name]!.text { print("   plain: \(base[name]!.text.prefix(90))\n   mtp  : \(text.prefix(90))") }
            }
            print(String(format: "MTP γ=%d overall: %.1f tok/s vs plain %.1f tok/s (x%.2f), accept %.0f%%, %.2f tok/step", gamma,
                         Double(sumTok) / sumSec, Double(sumTok) / baseSec, baseSec / sumSec,
                         Double(acc) / Double(max(1, drafted)) * 100, Double(acc + steps) / Double(max(1, steps))))
            print("PROFILE MTP γ=\(gamma) decode (last prompt):\n" + engine.decodeProfile.report())
        }
    }
}
