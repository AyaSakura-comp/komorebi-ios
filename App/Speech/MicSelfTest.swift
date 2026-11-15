import AVFoundation
import Combine
import Foundation
import KomorebiIPC
import UIKit

/// `--mic-selftest`: checks on a device that the keyboard can start listening without opening the app.
/// Starts listening in the foreground, stops (standby), waits until the app is in the background (switch to another
/// app), then sends the keyboard's start/stop/end commands over the App Group and checks that audio flows.
/// Output: stdout (`devicectl … --console`) and Documents/komorebi-diag.log.
@MainActor
enum MicSelfTest {
    static let isActive = ProcessInfo.processInfo.arguments.contains("--mic-selftest")

    static func run(pipeline: AudioPipelineManager) async {
        func log(_ s: String) {
            print("MIC_SELFTEST \(s)")
            fflush(stdout)
            pipeline.diag("selftest: \(s)")
        }
        func send(_ action: KomorebiIPCCommand.Action) {
            KomorebiIPCStore.shared.sendCommand(KomorebiIPCCommand(id: UInt64(Date().timeIntervalSince1970 * 1000), action: action, useDraftMode: true))
        }
        func sleep(_ s: Double) async { try? await Task.sleep(nanoseconds: UInt64(s * 1e9)) }

        log("start (standby \(pipeline.micStandbyMinutes) min)")
        await pipeline.startListening()
        let b0 = pipeline.processedAudioBuffers
        await sleep(3)
        log("foreground listening=\(pipeline.isListening) buffers/3s=\(pipeline.processedAudioBuffers - b0)")

        pipeline.stopListening()
        let b1 = pipeline.processedAudioBuffers
        await sleep(2)
        log("stopped: standby=\(pipeline.micStandby) micLive=\(pipeline.isMicLive) buffers/2s=\(pipeline.processedAudioBuffers - b1)")
        log("READY_FOR_BACKGROUND")

        let deadline = Date().addingTimeInterval(60)
        while UIApplication.shared.applicationState != .background && Date() < deadline { await sleep(0.2) }
        log("app in background=\(UIApplication.shared.applicationState == .background)")
        await sleep(2)
        log("standby in background: micLive=\(pipeline.isMicLive) state=\(KomorebiIPCStore.shared.readState().micLive)")

        send(.startListening)
        await sleep(1)
        let b2 = pipeline.processedAudioBuffers
        await sleep(3)
        log("background start via IPC: listening=\(pipeline.isListening) buffers/3s=\(pipeline.processedAudioBuffers - b2)")

        send(.stopListening)
        await sleep(1)
        log("background stop via IPC: listening=\(pipeline.isListening) standby=\(pipeline.micStandby)")

        send(.endMicSession)
        await sleep(1)
        log("end via IPC: micLive=\(pipeline.isMicLive)")
        log("DONE")
        if ProcessInfo.processInfo.arguments.contains("--exit-after") {
            await sleep(0.5)
            exit(0)
        }
    }
}

/// `--nav-probe [--open-app <bundle id>]`: prints which "go back" private API this iOS has and, with `--open-app`,
/// whether LSApplicationWorkspace may open another app (the keyboard's host app is reopened that way).
@MainActor
enum NavProbe {
    static let isActive = ProcessInfo.processInfo.arguments.contains("--nav-probe")

    static func run(pipeline: AudioPipelineManager) async {
        print("NAV_PROBE\n" + SystemNavigation.probe())
        let group = FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: KomorebiAppConfig.appGroupID)
        let base = KomorebiAppConfig.containerURL()
        print("NAV_PROBE app group container: \(group?.path ?? "nil") | IPC dir: \(base.path)")
        print("NAV_PROBE IPC dir files: \((try? FileManager.default.contentsOfDirectory(atPath: base.path))?.sorted().joined(separator: ", ") ?? "-")")
        let probe = (try? String(contentsOf: base.appendingPathComponent("keyboard-probe.log"), encoding: .utf8)) ?? "(none)"
        print("NAV_PROBE keyboard probe: \(probe)")
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "--open-app"), i + 1 < args.count {
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            let t0 = Date()
            let r = SystemNavigation.openApplication(bundleID: args[i + 1])
            print("NAV_PROBE open-app: \(r)")
            pipeline.diag("nav probe open-app: \(r)")
            var left: Double?
            while Date().timeIntervalSince(t0) < 4 {
                if UIApplication.shared.applicationState != .active { left = Date().timeIntervalSince(t0); break }
                try? await Task.sleep(nanoseconds: 50_000_000)
            }
            print("NAV_PROBE left the foreground: \(left.map { String(format: "after %.2f s", $0) } ?? "no (still active after 4 s)")")
        }
        print("NAV_PROBE_DONE")
        fflush(stdout)
        if args.contains("--exit-after") { exit(0) }
    }
}

/// `--preview-selftest`: synthesizes Mandarin speech, feeds it at real-time pace through VAD → ASR → Gemma, and logs
/// what the keyboard's preview line shows while "speaking" and when the draft arrives.
@MainActor
enum PreviewSelfTest {
    static let isActive = ProcessInfo.processInfo.arguments.contains("--preview-selftest")

    static func run(pipeline: AudioPipelineManager) async {
        func log(_ s: String) { print("PREVIEW_SELFTEST \(s)"); fflush(stdout) }
        await pipeline.loadAllEnginesIfNeeded()
        try? await Task.sleep(nanoseconds: 1_500_000_000)  // let the background warm-up finish
        let sentences = ["呃那個我們明天下午三點在三樓會議室開會，喔不對不對，改成四點半，記得帶筆電",
                         "然後會後一起去吃晚餐"]
        var samples: [Float] = []
        for text in sentences {
            samples += await synthesize(text)
            samples += [Float](repeating: 0, count: 16_000)  // 1 s pause between sentences
        }
        log(String(format: "speech %.1f s, live preview %@", Double(samples.count) / 16_000, pipeline.livePreviewEnabled ? "on" : "off"))

        pipeline.resetDraft()
        pipeline.beginSimulatedListening()
        var shown = ""
        let observer = pipeline.$preeditText.combineLatest(pipeline.$preeditTail).sink { main, tail in
            let text = tail.isEmpty ? main : "\(main)〔\(tail)〕"
            guard text != shown else { return }
            shown = text
            _ = main
        }
        let draftObserver = pipeline.$editorText.sink { text in
            _ = text
        }
        startTime = CFAbsoluteTimeGetCurrent()
        // What the keyboard sees: the text field (draft commits) and its preview line.
        var lastCommit = KomorebiIPCStore.shared.readState().commitID
        var lastLine = ""
        let ipcWatch = Task { @MainActor in
            while !Task.isCancelled {
                let st = KomorebiIPCStore.shared.readState()
                if st.commitID != lastCommit {
                    lastCommit = st.commitID
                    log(String(format: "t=%.1fs FIELD: %@", CFAbsoluteTimeGetCurrent() - startTime, st.committedText))
                }
                let line = st.partialText + (st.partialTail.isEmpty ? "" : "〔\(st.partialTail)〕")
                if line != lastLine {
                    lastLine = line
                    if !line.isEmpty { log(String(format: "t=%.1fs   line: %@", CFAbsoluteTimeGetCurrent() - startTime, line)) }
                }
                try? await Task.sleep(nanoseconds: 50_000_000)
            }
        }
        let chunk = 1024
        var offset = 0
        while offset < samples.count {
            let end = min(offset + chunk, samples.count)
            pipeline.feedSimulatedSamples(Array(samples[offset..<end]))
            offset = end
            try? await Task.sleep(nanoseconds: UInt64(Double(chunk) / 16_000 * 1e9))
        }
        log(String(format: "t=%.1fs speech fed", CFAbsoluteTimeGetCurrent() - startTime))
        let deadline = Date().addingTimeInterval(20)
        while pipeline.isMergingSentence && Date() < deadline { try? await Task.sleep(nanoseconds: 100_000_000) }
        try? await Task.sleep(nanoseconds: 300_000_000)
        log("final draft: \(pipeline.editorText)")
        ipcWatch.cancel()
        observer.cancel()
        draftObserver.cancel()
        log("DONE")
        if ProcessInfo.processInfo.arguments.contains("--exit-after") { exit(0) }
    }

    private static var startTime: CFAbsoluteTime = 0
    private static var synthesizer: AVSpeechSynthesizer?

    /// Mandarin (Taiwan) speech as 16 kHz mono Float32.
    private static func synthesize(_ text: String) async -> [Float] {
        let synth = AVSpeechSynthesizer()
        synthesizer = synth
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "zh-TW")
        let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!
        return await withCheckedContinuation { cont in
            var out: [Float] = []
            var finished = false
            synth.write(utterance) { buffer in
                guard let pcm = buffer as? AVAudioPCMBuffer else { return }
                if pcm.frameLength == 0 {
                    if !finished { finished = true; cont.resume(returning: out) }
                    return
                }
                guard let conv = AVAudioConverter(from: pcm.format, to: target),
                      let dst = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: AVAudioFrameCount(Double(pcm.frameLength) * 16_000 / pcm.format.sampleRate) + 64) else { return }
                var fed = false
                _ = conv.convert(to: dst, error: nil) { _, status in
                    if fed { status.pointee = .noDataNow; return nil }
                    fed = true
                    status.pointee = .haveData
                    return pcm
                }
                if let ch = dst.floatChannelData?[0] { out += Array(UnsafeBufferPointer(start: ch, count: Int(dst.frameLength))) }
            }
        }
    }
}
