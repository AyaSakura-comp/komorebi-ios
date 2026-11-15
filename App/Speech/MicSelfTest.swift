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
