import XCTest
@testable import KomorebiCore

final class FlashHeadDiagTests: XCTestCase {
    func testFlashHeadVariants() throws {
        let ggufPath = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("models/gemma-4-e2b/gemma-4-E2B_q4_0-it.gguf")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: ggufPath.path))
        try XCTSkipUnless(ProcessInfo.processInfo.environment["FH_DIAG"] == "1", "set FH_DIAG=1 to run the FlashHead clustering study")
        let gguf = try GGUFFile(url: ggufPath, loadTokenizerArrays: true)
        let tok = try Gemma4Tokenizer(metadata: gguf.metadata)
        let cache = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("komorebi-ane")
        // Collect (hidden, exact token) samples over zh / en / ja generations with the exact head.
        let engine = try Gemma4ANEDirectEngine(gguf: gguf, options: .init(lutBits: 4, maxContext: 1024, cacheDirectory: cache))
        var samples: [([Float], Int32)] = []
        engine.auditExactLMHead = true
        engine.auditTap = { h, t in samples.append((Array(UnsafeBufferPointer(start: h, count: 1536)), t)) }
        for p in ["用繁體中文寫一段大約一百字的台北旅遊介紹。", "Explain in three sentences why the sky is blue.", "日本語で、東京のおすすめ観光地を三つ紹介してください。"] {
            engine.clearPrefixCache()
            _ = try engine.generate(inputIDs: tok.encode(tok.formatChatPrompt(turns: [ChatTurn(role: .user, content: p)], addGenerationPrompt: true), addBOS: true), maxNewTokens: 48, stopTokenIDs: [])
        }
        print("samples: \(samples.count)")
        // Fit the per-dimension query RMS on even samples, evaluate on odd ones.
        let fit = samples.enumerated().filter { $0.offset % 2 == 0 }.map { $0.element }
        samples = samples.enumerated().filter { $0.offset % 2 == 1 }.map { $0.element }
        var rms = [Float](repeating: 0, count: 1536)
        for (h, _) in fit { for d in 0..<1536 { rms[d] += h[d] * h[d] } }
        rms = rms.map { sqrtf($0 / Float(fit.count)) }
        struct V { let name: String; let center: Bool; let spherical: Bool; let rw: Float; let clusters: Int; let scaled: Bool }
        let variants = [
            V(name: "spherical, centred", center: true, spherical: true, rw: 0, clusters: 4096, scaled: false),
            V(name: "QA spherical, centred", center: true, spherical: true, rw: 0, clusters: 4096, scaled: true),
            V(name: "QA euclid, centred", center: true, spherical: false, rw: 0, clusters: 4096, scaled: true),
            V(name: "QA spherical, centred 2048c", center: true, spherical: true, rw: 0, clusters: 2048, scaled: true),
            V(name: "QA spherical, centred 8192c", center: true, spherical: true, rw: 0, clusters: 8192, scaled: true),
        ]
        for v in variants {
            var cfg = FlashHeadLMHead.Config()
            cfg.center = v.center; cfg.spherical = v.spherical; cfg.radiusWeight = v.rw; cfg.clusters = v.clusters
            if v.scaled { cfg.dimScale = rms }
            let fh = try FlashHeadLMHead.build(gguf: gguf, tokenizer: tok, config: cfg)
            var ranks: [Int] = []
            var agree64 = 0, agree128 = 0, agree256 = 0
            for (h, t) in samples {
                h.withUnsafeBufferPointer { hb in
                    ranks.append(fh.diagnose(hb.baseAddress!, token: t).clusterRank)
                    fh.probes = 64; if fh.argmax(hb.baseAddress!) == t { agree64 += 1 }
                    fh.probes = 128; if fh.argmax(hb.baseAddress!) == t { agree128 += 1 }
                    fh.probes = 256; if fh.argmax(hb.baseAddress!) == t { agree256 += 1 }
                }
            }
            ranks.sort()
            let n = Double(samples.count)
            print(String(format: "%-30@ rank median %4d p90 %4d max %4d | top-1 agree: P64 %.1f%%  P128 %.1f%%  P256 %.1f%%", v.name as NSString,
                         ranks[ranks.count / 2], ranks[ranks.count * 9 / 10], ranks.last!, Double(agree64) / n * 100, Double(agree128) / n * 100, Double(agree256) / n * 100))
        }
    }
}
