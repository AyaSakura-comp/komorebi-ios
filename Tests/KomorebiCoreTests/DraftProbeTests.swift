import XCTest
@testable import KomorebiCore
import KomorebiANEDirect

/// `DRAFT_PROBE=1`: shows what Gemma returns for a draft merge, step by step (first answer, retry, acceptance).
final class DraftProbeTests: XCTestCase {
    func testDraftMergeProbe() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["DRAFT_PROBE"] == "1", "set DRAFT_PROBE=1")
        let gguf = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("models/gemma-4-e2b/gemma-4-E2B_q4_0-it.gguf")
        let refiner = KomorebiRefiner()
        try await refiner.loadGGUF(url: gguf)
        try await refiner.loadANEDirect(cacheDirectory: FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("komorebi-ane"),
                                        drafterDirectory: URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("models/gemma-4-e2b-assistant"))
        let cases: [(String, String)] = [
            ("這個段句看起來好像很好一些。還可以，這個做起來。就是比正常 Apple 的語音輸入法還好一點。這個比 Apple 的語音輸入法還好一點。", "刪除前面所有的段落"),
            ("這個段句看起來好像很好一些。就是比正常 Apple 的語音輸入法還好一點。", "這個比 Apple 的語音輸入法還好一點不對還不太好"),
        ]
        for (draft, raw) in cases {
            let first = try await refiner.generateChat(turns: KomorebiPrompts.draftTurns(draft: draft, raw: raw), maxNewTokens: 256)
            let firstClean = KomorebiPrompts.cleanModelOutput(first.text)
            print("PROBE draft=\(draft)\nPROBE raw=\(raw)\nPROBE first=\(firstClean) | acceptDraft=\(KomorebiPrompts.acceptDraft(draft: draft, raw: raw, updated: firstClean)) unchanged=\(firstClean == draft)")
            let retry = try await refiner.generateChat(turns: KomorebiPrompts.draftRetryTurns(draft: draft, raw: raw), maxNewTokens: 256)
            let retryClean = KomorebiPrompts.cleanModelOutput(retry.text)
            print("PROBE retry=\(retryClean) | acceptRetry=\(KomorebiPrompts.acceptDraftRetry(draft: draft, raw: raw, updated: retryClean))")
            let merged = try await refiner.draftMerge(draft: draft, raw: raw)
            print("PROBE final=\(merged.text) | decode \(String(format: "%.1f", merged.stats.decodeTPS)) tok/s, \(merged.stats.decodeTokens) tok\n")
        }
    }
}
