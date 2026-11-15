import AVFoundation
import Foundation
import KomorebiCore
import KomorebiIPC
import XCTest
@testable import KomorebiVoice

final class CrossAppSpeechPipelineTests: XCTestCase {
    @MainActor
    func testSilentSpeechToTranscriptAndAppGroupIPC() async throws {
        let mm = ModelManager()
        try XCTSkipUnless(mm.coreModelsReady, "Core models not present in container")
        guard let sileroURL = mm.sileroURL,
              let svModel = mm.senseVoiceModelURL,
              let svTokens = mm.senseVoiceTokensURL,
              let ggufURL = mm.gemma4GGUFURL else {
            XCTFail("Missing model URLs despite coreModelsReady == true")
            return
        }

        // Synthesize genuine 16 kHz Traditional Mandarin speech audio completely silently (AVSpeechSynthesizer.write)
        let synth = AVSpeechSynthesizer()
        let utterance = AVSpeechUtterance(string: "我們明天下午四點半開會好不好")
        utterance.voice = AVSpeechSynthesisVoice(language: "zh-TW") ?? AVSpeechSynthesisVoice(language: "zh-CN")
        utterance.rate = 0.48

        guard let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false) else {
            XCTFail("Failed to allocate 16kHz format")
            return
        }

        var pcm16k: [Float] = []
        let doneExp = expectation(description: "Synthesize speech audio silently")
        var didComplete = false

        synth.write(utterance) { buffer in
            guard let pcmBuf = buffer as? AVAudioPCMBuffer else { return }
            if pcmBuf.frameLength == 0 {
                if !didComplete {
                    didComplete = true
                    doneExp.fulfill()
                }
                return
            }
            if pcmBuf.format.sampleRate == 16_000 && pcmBuf.format.channelCount == 1 && pcmBuf.format.commonFormat == .pcmFormatFloat32,
               let ch0 = pcmBuf.floatChannelData?[0] {
                pcm16k.append(contentsOf: UnsafeBufferPointer(start: ch0, count: Int(pcmBuf.frameLength)))
            } else if let conv = AVAudioConverter(from: pcmBuf.format, to: targetFormat) {
                let cap = AVAudioFrameCount(Double(pcmBuf.frameLength) * (16_000.0 / pcmBuf.format.sampleRate) + 32)
                if let outBuf = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: cap) {
                    var supplied = false
                    var err: NSError?
                    let status = conv.convert(to: outBuf, error: &err) { _, outStatus in
                        if supplied {
                            outStatus.pointee = .noDataNow
                            return nil
                        }
                        supplied = true
                        outStatus.pointee = .haveData
                        return pcmBuf
                    }
                    if status != .error, err == nil, let ch0 = outBuf.floatChannelData?[0] {
                        pcm16k.append(contentsOf: UnsafeBufferPointer(start: ch0, count: Int(outBuf.frameLength)))
                    }
                }
            }
        }

        await fulfillment(of: [doneExp], timeout: 15.0)
        // Add 0.6s trailing silence so SileroTurnsVAD detects end-of-turn naturally
        pcm16k.append(contentsOf: [Float](repeating: 0, count: 9_600))

        // Verify App Group shared container resolves cleanly
        let containerPath = KomorebiAppConfig.containerURL().path
        print("IPC_CONTAINER_PATH: \(containerPath)")
        XCTAssertTrue(
            FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: KomorebiAppConfig.appGroupID) != nil,
            "App Group container must be accessible when entitled"
        )

        let vad = SileroTurnsVAD(modelURL: sileroURL)
        let asr = try SenseVoiceASR(modelURL: svModel, tokensURL: svTokens)
        // The Simulator has no Neural Engine: check that the GGUF tokenizer loads, and send the transcript as heard.
        let refiner = KomorebiRefiner()
        try await refiner.loadGGUF(url: ggufURL)
        let tokenizerLoaded = await refiner.tokenizer != nil
        XCTAssertTrue(tokenizerLoaded)

        var speechSegments: [[Float]] = []
        for offset in stride(from: 0, to: pcm16k.count, by: SileroTurnsVAD.windowSamples) {
            let end = min(offset + SileroTurnsVAD.windowSamples, pcm16k.count)
            let chunk = Array(pcm16k[offset..<end])
            for ev in vad.push(samples: chunk) {
                if case .finalizedSegment(let seg) = ev {
                    speechSegments.append(seg)
                }
            }
        }
        for ev in vad.flush() {
            if case .finalizedSegment(let seg) = ev {
                speechSegments.append(seg)
            }
        }

        let speechPCM = speechSegments.flatMap { $0 }.isEmpty ? pcm16k : speechSegments.flatMap { $0 }
        let transcript = asr.transcribe(samples: speechPCM, isPartial: false)
        print("SILENT_SYNTH_ASR: segments=\(speechSegments.count) raw='\(transcript.text)' latencyMs=\(String(format: "%.1f", transcript.latencyMs))")
        XCTAssertFalse(transcript.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

        // On a device the ANE engine refines this; here the recognised text is published as is.
        let refinedText = transcript.text

        // Verify App Group IPC state round-trip
        let testState = KomorebiIPCState(
            sequence: 42,
            hostAppAlive: true,
            hostInForeground: false,
            audioSessionActive: true,
            modelsReady: true,
            phase: .idle,
            commitID: 7,
            committedText: refinedText,
            isDraftReplacement: false
        )
        KomorebiIPCStore.shared.writeState(testState)
        let loadedState = KomorebiIPCStore.shared.readState()
        XCTAssertEqual(loadedState.committedText, refinedText)
        XCTAssertEqual(loadedState.commitID, 7)
    }
}
