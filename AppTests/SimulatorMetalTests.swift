import Foundation
import KomorebiCore
import KomorebiIPC
import Metal
import XCTest
@testable import KomorebiVoice

final class SimulatorMetalTests: XCTestCase {
    func testMetalEngineOnSimulator() async throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device unavailable in this environment")
        }
        guard let ggufURL = await MainActor.run(body: { ModelManager().gemma4GGUFURL }) else {
            throw XCTSkip("GGUF not present in container")
        }

        let refiner = KomorebiRefiner()
        try await refiner.loadGGUF(url: ggufURL, preferMetal: true, maxContext: 1024)
        let hasMetal = await refiner.metalEngine != nil
        XCTAssertTrue(hasMetal, "KomorebiRefiner should load Gemma4MetalEngine on \(dev.name)")

        let sample = "呃那個我們明天下午三點開會，喔不對不對，改成四點半好不好"
        let res = try await refiner.refineSentence(raw: sample, preference: .metal)
        print("Metal Benchmark (\(dev.name)): backend=\(res.stats.backend) prefill=\(res.stats.prefillTokens) tok @ \(String(format: "%.1f", res.stats.prefillTPS)) t/s | decode=\(res.stats.decodeTokens) tok @ \(String(format: "%.1f", res.stats.decodeTPS)) t/s | text=\(res.text)")

        XCTAssertFalse(res.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        XCTAssertTrue(res.stats.backend.contains("Metal"))
    }
}
