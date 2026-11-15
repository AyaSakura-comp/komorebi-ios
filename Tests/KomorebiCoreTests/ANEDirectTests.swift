import XCTest
@testable import KomorebiCore
import KomorebiANEDirect

/// Direct-ANE (private MIL compiler) Gemma 4 E2B engine: correctness vs. the Metal engine, speed, profile.
/// Env: ANE_BITS (LUT bits, default 4), ANE_GROUP (rows per LUT, default 8), ANE_CHUNK (prefill S, default 64).
final class ANEDirectTests: XCTestCase {
    private let ggufPath = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("models/gemma-4-e2b/gemma-4-E2B_q4_0-it.gguf")

    func testANEDirectEngine() throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: ggufPath.path), "GGUF not found")
        let env = ProcessInfo.processInfo.environment
        let bits = Int(env["ANE_BITS"] ?? "") ?? 4
        let group = Int(env["ANE_GROUP"] ?? "") ?? 8
        let chunk = Int(env["ANE_CHUNK"] ?? "") ?? 64

        let gguf = try GGUFFile(url: ggufPath, loadTokenizerArrays: true)
        let tok = try Gemma4Tokenizer(metadata: gguf.metadata)
        let cache = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("komorebi-ane")
        let engine = try Gemma4ANEDirectEngine(
            gguf: gguf,
            options: .init(lutBits: bits, lutGroup: group, prefillChunk: chunk, maxContext: 1024, cacheDirectory: cache),
            progress: { msg in if msg.hasSuffix("/35") || msg.contains("LM") { print("  build: \(msg)") } }
        )
        print(String(format: "ANE DIRECT BUILD: %.1f s total (ANE compile+load %.1f s), weights %.0f MB, %d compiles",
                     engine.buildSeconds, engine.compileSeconds, Double(engine.weightBytes) / 1e6, ane_rt_compile_count()))

        let stops: Set<Int32> = [tok.eosTokenID, tok.turnEndTokenID]
        func run(_ raw: String) throws -> GenerationResult {
            let turns = KomorebiPrompts.refineTurns(raw: raw)
            let ids = tok.encode(tok.formatChatPrompt(turns: turns, addGenerationPrompt: true), addBOS: true)
            let sys = "<|turn>system\n\(turns[0].content.trimmingCharacters(in: .whitespacesAndNewlines))<turn|>\n"
            let prefix = tok.encode(sys, addBOS: true).count
            let r = try engine.generate(inputIDs: ids, maxNewTokens: 64, stopTokenIDs: stops, staticPrefixLength: prefix)
            return GenerationResult(text: tok.decode(r.tokenIDs, skipSpecialTokens: true), tokenIDs: r.tokenIDs, stats: r.stats)
        }

        // Cold (prefix + rest), then warm prefix reuse, then a second sentence.
        engine.clearPrefixCache()
        let r1 = try run("呃那個我們明天下午三點開會，喔不對不對，改成四點半好不好")
        print("ANE CLEANUP 1: '\(r1.text)' | prefill=\(r1.stats.prefillTokens) tok @ \(String(format: "%.1f", r1.stats.prefillTPS)) tok/s | decode=\(r1.stats.decodeTokens) tok @ \(String(format: "%.1f", r1.stats.decodeTPS)) tok/s")
        print("PROFILE (cold request):\n" + engine.lastProfile.report())
        let r2 = try run("我等一下要去買一杯冰拿鐵跟一塊起司蛋糕")
        print("ANE CLEANUP 2 (prefix cached \(r2.stats.prefixCachedTokens)): '\(r2.text)' | prefill=\(r2.stats.prefillTokens) tok @ \(String(format: "%.1f", r2.stats.prefillTPS)) tok/s | decode=\(r2.stats.decodeTokens) tok @ \(String(format: "%.1f", r2.stats.decodeTPS)) tok/s")
        print("PROFILE (warm request):\n" + engine.lastProfile.report())

        // Steady-state decode profile: long generation from a fixed prompt.
        engine.clearPrefixCache()
        let ids = tok.encode(tok.formatChatPrompt(turns: [ChatTurn(role: .user, content: "用繁體中文寫一段大約一百字的台北旅遊介紹。")], addGenerationPrompt: true), addBOS: true)
        let long = try engine.generate(inputIDs: ids, maxNewTokens: 128, stopTokenIDs: [], staticPrefixLength: 0)
        print(String(format: "ANE DECODE BENCH: %d tok @ %.1f tok/s (prefill %d tok @ %.1f tok/s)", long.stats.decodeTokens, long.stats.decodeTPS, long.stats.prefillTokens, long.stats.prefillTPS))
        print("PROFILE (decode-heavy):\n" + engine.lastProfile.report())
        print("TEXT: " + tok.decode(long.tokenIDs, skipSpecialTokens: true).prefix(120))

        // Prefill-only profile (warm programs): full 186-token prompt, no prefix cache, 1 output token.
        do {
            let turns = KomorebiPrompts.refineTurns(raw: "呃那個我們明天下午三點開會，喔不對不對，改成四點半好不好")
            let pids = tok.encode(tok.formatChatPrompt(turns: turns, addGenerationPrompt: true), addBOS: true)
            for i in 0..<3 {
                let r = try engine.generate(inputIDs: pids, maxNewTokens: 1, stopTokenIDs: [], staticPrefixLength: 0)
                if i == 2 {
                    print(String(format: "ANE PREFILL-ONLY: %d tok in %.1f ms = %.1f tok/s", r.stats.prefillTokens, r.stats.prefillSeconds * 1000, r.stats.prefillTPS))
                    print("PROFILE (prefill only):\n" + engine.lastProfile.report())
                    let per = zip(engine.lastProfile.aneSeconds, engine.lastProfile.aneCalls).map { $1 > 0 ? $0 / Double($1) * 1000 : 0 }
                    print("per-program ms/eval: " + per.map { String(format: "%.2f", $0) }.joined(separator: " "))
                }
            }
        }
        print("ANE SWITCH BENCH:\n" + (try engine.benchmarkSwitching()))
        print("ANE PROGRAM BENCH:\n" + (try engine.benchmarkPrograms()))
        XCTAssertTrue(r1.text.contains("四點"), "unexpected: \(r1.text)")
        XCTAssertTrue(r2.text.contains("拿鐵"), "unexpected: \(r2.text)")
    }

    /// Long-context benchmark: ANE_LONG_TOKENS-token prompt (default 20000) + ANE_DECODE_TOKENS forced decode (default 256).
    func testANEDirectLongContext() throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: ggufPath.path), "GGUF not found")
        let env = ProcessInfo.processInfo.environment
        let bits = Int(env["ANE_BITS"] ?? "") ?? 4
        let target = Int(env["ANE_LONG_TOKENS"] ?? "") ?? 20000
        let decodeN = Int(env["ANE_DECODE_TOKENS"] ?? "") ?? 256

        let gguf = try GGUFFile(url: ggufPath, loadTokenizerArrays: true)
        let tok = try Gemma4Tokenizer(metadata: gguf.metadata)
        let cache = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("komorebi-ane")
        let engine = try Gemma4ANEDirectEngine(
            gguf: gguf,
            options: .init(lutBits: bits, lutGroup: 8, prefillChunk: 64, maxContext: target + decodeN + 128, cacheDirectory: cache)
        )
        print(String(format: "LONG CTX ENGINE: LUT%d, maxContext %d, build %.1f s", bits, target + decodeN + 128, engine.buildSeconds))

        // Prompt: this repository's Swift sources, then a summary request.
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let head = tok.encode("<|turn>user\n以下是一個 iOS 專案的 Swift 原始碼：\n\n", addBOS: true)
        let tail = tok.encode("\n\n請用繁體中文，用三點條列摘要這個專案在做什麼。<turn|>\n<|turn>model\n", addBOS: false)
        let need = target - head.count - tail.count
        var body: [Int32] = []
        let tTok = CFAbsoluteTimeGetCurrent()
        var files: [URL] = []
        for dir in ["Sources", "App", "KeyboardExtension"] {
            if let e = FileManager.default.enumerator(at: repo.appendingPathComponent(dir), includingPropertiesForKeys: nil) {
                for case let u as URL in e where u.pathExtension == "swift" { files.append(u) }
            }
        }
        files.sort { $0.path < $1.path }
        var fileCount = 0
        outer: while body.count < need {
            for f in files {
                guard let text = try? String(contentsOf: f, encoding: .utf8) else { continue }
                body += tok.encode("// File: \(f.path.replacingOccurrences(of: repo.path + "/", with: ""))\n\(text)\n\n", addBOS: false)
                fileCount += 1
                if body.count >= need { break outer }
            }
        }
        let ids = head + Array(body.prefix(need)) + tail
        print(String(format: "LONG CTX PROMPT: %d tokens from %d source files (tokenized in %.1f s)", ids.count, fileCount, CFAbsoluteTimeGetCurrent() - tTok))
        XCTAssertEqual(ids.count, target)

        try engine.warmUp()
        let r = try engine.generate(inputIDs: ids, maxNewTokens: decodeN, stopTokenIDs: [], staticPrefixLength: 0)
        print(String(format: "LONG CTX PREFILL: %d tok in %.2f s = %.1f tok/s", r.stats.prefillTokens, r.stats.prefillSeconds, r.stats.prefillTPS))
        print(String(format: "LONG CTX DECODE: %d tok in %.2f s = %.2f tok/s (context %d -> %d)", r.stats.decodeTokens, r.stats.decodeSeconds, r.stats.decodeTPS, ids.count, ids.count + r.stats.decodeTokens))
        print("PROFILE prefill:\n" + engine.prefillProfile.report())
        print("PROFILE decode:\n" + engine.decodeProfile.report())
        let stops: Set<Int32> = [tok.eosTokenID, tok.turnEndTokenID]
        let answer = Array(r.tokenIDs.prefix { !stops.contains($0) })
        print("LONG CTX ANSWER (\(answer.count) tok before stop):\n" + tok.decode(answer, skipSpecialTokens: true))
        XCTAssertEqual(r.stats.decodeTokens, decodeN)
    }
}
