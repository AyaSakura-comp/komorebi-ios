import Foundation
import KomorebiCore
import KomorebiIPC
import XCTest
@testable import KomorebiVoice

final class KomorebiAppTests: XCTestCase {
    func testIPCRoundTripStateAndCommands() throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("komorebi-test-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let store = KomorebiIPCStore(baseURL: tmp)
        let state = KomorebiIPCState(
            sequence: 7,
            hostAppAlive: true,
            audioSessionActive: true,
            modelsReady: true,
            phase: .listening,
            micLevel: 0.42,
            partialText: "明天下午",
            commitID: 3,
            committedText: "明天下午三點開會。",
            isDraftReplacement: true,
            noiseSuppressionActive: true,
            micIsolationLabel: "語音隔離 (AI)",
            telemetry: KomorebiTelemetry(backendUsed: "ANE direct (LUT4, MTP×3)", prefillTokens: 150, prefillTPS: 83.9, decodeTokens: 12, decodeTPS: 49.0, asrLatencyMs: 65)
        )
        store.writeState(state)
        let decoded = store.readState()
        XCTAssertEqual(decoded, state)

        let cmd = KomorebiIPCCommand(id: 99, action: .toggleNoiseSuppression, useDraftMode: true)
        store.sendCommand(cmd)
        XCTAssertEqual(store.readCommand(), cmd)
    }

    func testPromptWrappersAndTokenizerConfig() {
        XCTAssertEqual(KomorebiPrompts.cleanModelOutput("  我們出發了！  \n"), "我們出發了！")
        XCTAssertEqual(KomorebiPrompts.parseSelectionJSON("{\"text\":\"改寫後\"}"), "改寫後")
        XCTAssertTrue(KomorebiPrompts.acceptDraft(draft: "原本的草稿", raw: "加上最後一句", updated: "原本的草稿，加上最後一句。"))
    }

    func testOpenCCS2TWPBundleConversion() throws {
        let converter = try SenseVoiceASR.makeTWConverter()
        let out = converter.convert("我们需要优化软件内存使用")
        XCTAssertEqual(out, "我們需要優化軟體記憶體使用")
    }

    @MainActor
    func testAudioPipelineInitialState() {
        let key = "komorebi_noise_suppression_enabled"
        UserDefaults.standard.removeObject(forKey: key)
        defer { UserDefaults.standard.removeObject(forKey: key) }

        let mm = ModelManager()
        let pipeline = AudioPipelineManager(modelManager: mm)
        XCTAssertFalse(pipeline.isListening)
        XCTAssertEqual(pipeline.mode, .draft)
        XCTAssertTrue(pipeline.noiseSuppressionEnabled)
        pipeline.noiseSuppressionEnabled = false
        XCTAssertFalse(UserDefaults.standard.bool(forKey: key))
        pipeline.undoLastCommit()
    }
}
