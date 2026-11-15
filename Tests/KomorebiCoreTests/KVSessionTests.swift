import XCTest
@testable import KomorebiCore
import KomorebiANEDirect

/// Multi-turn chat on a separate KV session: only new tokens are prefilled, outputs equal a fresh run, and a saved
/// session restores the same cache.
final class KVSessionTests: XCTestCase {
    func testChatSessionReuseAndPersistence() throws {
        let ggufURL = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("models/gemma-4-e2b/gemma-4-E2B_q4_0-it.gguf")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: ggufURL.path), "GGUF not found")
        try XCTSkipUnless(ane_rt_available(), "ANE unavailable")
        let gguf = try GGUFFile(url: ggufURL, loadTokenizerArrays: true)
        let tok = try Gemma4Tokenizer(metadata: gguf.metadata)
        let cache = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("komorebi-ane")
        let dict = FlashHeadLMHead.smallDictionary(vocab: tok.vocab, tokenTypes: tok.tokenTypes)
        let engine = try Gemma4ANEDirectEngine(gguf: gguf, options: .init(lutBits: 4, lutGroup: 8, prefillChunk: 64, verifyChunk: 4,
                                                                         maxContext: 4096, cacheDirectory: cache), dictionary: dict)
        try engine.warmUp()
        let stops: Set<Int32> = [tok.eosTokenID, tok.turnEndTokenID]
        let chat = engine.makeSession(capacity: 4096)

        let first = tok.encodeChat(turns: [ChatTurn(role: .system, content: "你是簡潔的助理，用台灣繁體中文回答。"),
                                          ChatTurn(role: .user, content: "推薦一本適合初學者的程式設計書。")])
        let r1 = try engine.withSession(chat) { try engine.generate(inputIDs: first, maxNewTokens: 60, stopTokenIDs: stops) }
        let after1 = chat.tokens
        // Every answer token that was run through the model is cached (one cut off by maxNewTokens was not run).
        XCTAssertEqual(Array((first + r1.tokenIDs).prefix(after1.count)), after1)
        XCTAssertGreaterThanOrEqual(after1.count, first.count + r1.tokenIDs.count - 1)
        print("TURN1 prefill \(r1.stats.prefillTokens) cached \(r1.stats.prefixCachedTokens): \(tok.decode(r1.tokenIDs))")

        // Turn 2 continues from the exact cached tokens: only the new turn is prefilled.
        let suffix = tok.encode("<turn|>\n<|turn>user\n那第二本呢？<turn|>\n<|turn>model\n", addBOS: false)
        let second = after1 + suffix
        let r2 = try engine.withSession(chat) { try engine.generate(inputIDs: second, maxNewTokens: 60, stopTokenIDs: stops) }
        print("TURN2 prefill \(r2.stats.prefillTokens) cached \(r2.stats.prefixCachedTokens) @ \(String(format: "%.0f", r2.stats.prefillTPS)) tok/s: \(tok.decode(r2.tokenIDs))")
        XCTAssertEqual(r2.stats.prefixCachedTokens, after1.count)
        XCTAssertEqual(r2.stats.prefillTokens, suffix.count)

        // Same prompt from scratch on the default session gives the same answer.
        engine.clearPrefixCache()
        let fresh = try engine.generate(inputIDs: second, maxNewTokens: 60, stopTokenIDs: stops)
        // Cached positions came from other ANE procedures (decode S=1/verify S=4 vs prefill S=64), whose fp16 sums
        // differ in the last bits, so a near-tie can flip later; the start must agree.
        XCTAssertEqual(Array(fresh.tokenIDs.prefix(8)), Array(r2.tokenIDs.prefix(8)), "reused cache ~ fresh prefill")

        // Save, restore into a new session, continue: same as continuing in the original session.
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("kv-test.kkv")
        let t0 = CFAbsoluteTimeGetCurrent()
        try chat.save(to: url, fingerprint: engine.kvFingerprint)
        let restored = engine.makeSession(capacity: 4096)
        XCTAssertTrue(restored.load(from: url, fingerprint: engine.kvFingerprint))
        let size = (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
        print(String(format: "SAVE+LOAD %d tokens, %.1f MB, %.2f s", restored.tokens.count, Double(size) / 1e6, CFAbsoluteTimeGetCurrent() - t0))
        XCTAssertEqual(restored.tokens, chat.tokens)
        XCTAssertFalse(engine.makeSession(capacity: 4096).load(from: url, fingerprint: "other-model"))
        let third = chat.tokens + tok.encode("<turn|>\n<|turn>user\n謝謝<turn|>\n<|turn>model\n", addBOS: false)
        let restoredCount = restored.tokens.count
        let a = try engine.withSession(chat) { try engine.generate(inputIDs: third, maxNewTokens: 30, stopTokenIDs: stops) }
        let b = try engine.withSession(restored) { try engine.generate(inputIDs: third, maxNewTokens: 30, stopTokenIDs: stops) }
        XCTAssertEqual(a.tokenIDs, b.tokenIDs, "restored cache answers like the original")
        XCTAssertEqual(b.stats.prefixCachedTokens, restoredCount)
        XCTAssertEqual(b.stats.prefillTokens, third.count - restoredCount)
        try? FileManager.default.removeItem(at: url)
    }
}
