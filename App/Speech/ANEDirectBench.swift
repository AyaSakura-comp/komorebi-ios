import AVFoundation
import Foundation
import UIKit
import KomorebiCore
import KomorebiIPC

/// On-device benchmark of the direct-ANE (private MIL compiler) Gemma 4 E2B engine.
/// Launch with argument `--ane-bench` (optionally env `ANE_BITS`, `ANE_GROUP`, `ANE_CHUNK`).
/// Output goes to stdout (visible with `devicectl device process launch --console`) and Documents/ane-bench.log.
enum ANEDirectBench {
    /// True when launched with `--ane-bench`. The app then skips its own engine loading at launch, so the bench is
    /// the only engine in the process (two engines = two 3 GB GGUF mappings and contended ANE/CPU time).
    static let isActive = ProcessInfo.processInfo.arguments.contains("--ane-bench")

    static func runIfRequested() {
        guard isActive else { return }
        DispatchQueue.main.async { UIApplication.shared.isIdleTimerDisabled = true }
        startSilentAudioKeepAlive()
        let t = Thread { run() }
        t.stackSize = 8 << 20
        t.qualityOfService = .userInitiated
        t.start()
    }

    private static var keepAliveEngine: AVAudioEngine?
    private static var keepAlivePlayer: AVAudioPlayerNode?

    /// Silent looping playback so the `audio` background mode keeps the benchmark running if the screen locks.
    private static func startSilentAudioKeepAlive() {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default, options: [.mixWithOthers])
            try session.setActive(true)
            let engine = AVAudioEngine()
            let player = AVAudioPlayerNode()
            let format = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!
            engine.attach(player)
            engine.connect(player, to: engine.mainMixerNode, format: format)
            let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 44100)!
            buf.frameLength = 44100   // zero-filled = silence
            try engine.start()
            player.scheduleBuffer(buf, at: nil, options: .loops)
            player.play()
            keepAliveEngine = engine
            keepAlivePlayer = player
        } catch {
            print("ANE_BENCH keep-alive audio failed: \(error)")
        }
    }

    private static let logURL: URL = {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("ane-bench.log")
    }()

    private static func log(_ s: String) {
        print(s)
        fflush(stdout)
        if let d = (s + "\n").data(using: .utf8) {
            if let h = try? FileHandle(forWritingTo: logURL) {
                h.seekToEndOfFile(); h.write(d); try? h.close()
            } else {
                try? d.write(to: logURL)
            }
        }
    }

    private static func residentMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1e6 : -1
    }

    private static func run() {
        try? FileManager.default.removeItem(at: logURL)
        let env = ProcessInfo.processInfo.environment
        let bits = Int(env["ANE_BITS"] ?? "") ?? 4
        let group = Int(env["ANE_GROUP"] ?? "") ?? 8
        let chunk = Int(env["ANE_CHUNK"] ?? "") ?? 64
        log("ANE_BENCH start: LUT\(bits) g\(group) prefill S=\(chunk), footprint \(String(format: "%.0f", residentMB())) MB")
        log("ANE_BENCH state: \(deviceState())")
        do {
            let fm = FileManager.default
            let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let candidates = [
                docs.appendingPathComponent("Models/gemma-4-e2b/gemma-4-E2B_q4_0-it.gguf"),
                KomorebiAppConfig.modelsDirectoryURL().appendingPathComponent("gemma-4-e2b/gemma-4-E2B_q4_0-it.gguf"),
            ]
            guard let ggufURL = candidates.first(where: { fm.fileExists(atPath: $0.path) }) else {
                log("ANE_BENCH error: GGUF not found"); return
            }
            let gguf = try GGUFFile(url: ggufURL, loadTokenizerArrays: true)
            let tok = try Gemma4Tokenizer(metadata: gguf.metadata)
            let cache = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("komorebi-ane")
            try? fm.createDirectory(at: cache, withIntermediateDirectories: true)
            let dict = FlashHeadLMHead.smallDictionary(vocab: tok.vocab, tokenTypes: tok.tokenTypes)
            let engine = try Gemma4ANEDirectEngine(
                gguf: gguf,
                options: .init(lutBits: bits, lutGroup: group, prefillChunk: chunk, verifyChunk: 4, maxContext: 1024, cacheDirectory: cache),
                dictionary: dict,
                progress: { msg in if msg.hasSuffix("0/35") || msg.contains("LM") { log("  build: \(msg) (footprint \(String(format: "%.0f", residentMB())) MB)") } }
            )
            log(String(format: "ANE_BENCH build: %.1f s (ANE compile+load %.1f s), weights %.0f MB, footprint %.0f MB",
                       engine.buildSeconds, engine.compileSeconds, Double(engine.weightBytes) / 1e6, residentMB()))
            let drafterCandidates = [docs.appendingPathComponent("Models/gemma-4-e2b-assistant"),
                                     KomorebiAppConfig.modelsDirectoryURL().appendingPathComponent("gemma-4-e2b-assistant")]
            let drafterDir = drafterCandidates.first { fm.fileExists(atPath: $0.appendingPathComponent("model.safetensors").path) }
            let drafter = drafterDir.flatMap { try? Gemma4MTPDrafter(directory: $0) }
            log("ANE_BENCH drafter: \(drafter != nil ? "loaded" : "missing")")
            try engine.warmUp()
            engine.drafter = drafter
            if drafter != nil { try engine.warmUp() }

            let stops: Set<Int32> = [tok.eosTokenID, tok.turnEndTokenID]
            let longDraft = "明天早上九點在三樓會議室開季度檢討會，請大家準時出席。會議前要先把上季的業績報表整理好，寄給所有部門主管。會後中午一起吃便當，下午兩點再討論新產品的上市時程。"
            let prompts: [(String, [ChatTurn])] = [
                ("refine", KomorebiPrompts.refineTurns(raw: "呃那個我們明天下午三點開會，喔不對不對，改成四點半好不好")),
                ("draft", KomorebiPrompts.draftTurns(draft: longDraft, raw: "然後下午的討論改成三點")),
                ("zh-story", [ChatTurn(role: .user, content: "用繁體中文寫一段大約一百字的台北旅遊介紹。")]),
                ("english", [ChatTurn(role: .user, content: "Explain in three sentences why the sky is blue.")]),
            ]
            func ids(_ t: [ChatTurn]) -> [Int32] { tok.encode(tok.formatChatPrompt(turns: t, addGenerationPrompt: true), addBOS: true) }
            let args = ProcessInfo.processInfo.arguments
            if let i = args.firstIndex(of: "--bg-loop"), i + 1 < args.count, let secs = Double(args[i + 1]) {
                try stateLoop(engine: engine, drafter: drafter, seconds: secs,
                              english: ids(prompts[3].1), draft: ids(prompts[1].1), stops: stops)
                finish()
                return
            }
            for useMTP in drafter == nil ? [false] : [false, true] {
                engine.drafter = useMTP ? drafter : nil
                for (name, turns) in prompts {
                    engine.clearPrefixCache()
                    let r = try engine.generate(inputIDs: ids(turns), maxNewTokens: 128, stopTokenIDs: stops)
                    let p = engine.decodeProfile
                    log(String(format: "ANE_BENCH %@ %-9@ prefill %3d tok @ %6.1f tok/s | decode %3d tok @ %5.1f tok/s%@ | %@",
                               useMTP ? "MTP  " : "plain", name as NSString, r.stats.prefillTokens, r.stats.prefillTPS,
                               r.stats.decodeTokens, r.stats.decodeTPS,
                               useMTP ? String(format: " (accept %.0f%%, %.2f tok/step)", p.draftedTokens > 0 ? Double(p.acceptedTokens) / Double(p.draftedTokens) * 100 : 0,
                                               Double(p.acceptedTokens + p.verifySteps) / Double(max(1, p.verifySteps))) : "",
                               String(tok.decode(r.tokenIDs, skipSpecialTokens: true).prefix(40)).replacingOccurrences(of: "\n", with: " ")))
                }
                log("PROFILE \(useMTP ? "MTP" : "plain") decode (last prompt):\n" + engine.decodeProfile.report())
            }
            // Prefill throughput on a ~186-token prompt.
            engine.drafter = nil
            var best = 0.0
            for _ in 0..<3 {
                engine.clearPrefixCache()
                let r = try engine.generate(inputIDs: ids(prompts[0].1), maxNewTokens: 1, stopTokenIDs: [], staticPrefixLength: 0)
                best = max(best, r.stats.prefillTPS)
            }
            log(String(format: "ANE_BENCH prefill-only (warm): best %.1f tok/s", best))
            log("ANE_BENCH per-program (isolated):\n" + (try engine.benchmarkPrograms(iterations: 10)))
            log(String(format: "ANE_BENCH footprint at end: %.0f MB", residentMB()))
            log("ANE_BENCH state: \(deviceState())")
        } catch {
            log("ANE_BENCH error: \(error.localizedDescription)")
        }
        finish()
    }

    private static func finish() {
        log("ANE_BENCH_DONE")
        if ProcessInfo.processInfo.arguments.contains("--exit-after") {
            Thread.sleep(forTimeInterval: 1)
            exit(0)
        }
    }

    /// `--bg-loop <seconds>`: repeats a short plain decode + an MTP draft rewrite and logs speed next to the
    /// app / thermal / power state, so foreground vs background (or throttled) ANE speed can be compared live.
    private static func stateLoop(engine: Gemma4ANEDirectEngine, drafter: Gemma4MTPDrafter?, seconds: Double,
                                  english: [Int32], draft: [Int32], stops: Set<Int32>) throws {
        let fmt = DateFormatter()
        fmt.dateFormat = "HH:mm:ss"
        let tEnd = CFAbsoluteTimeGetCurrent() + seconds
        var iter = 0
        while CFAbsoluteTimeGetCurrent() < tEnd {
            iter += 1
            let before = deviceState()
            engine.drafter = nil
            engine.clearPrefixCache()
            let a = try engine.generate(inputIDs: english, maxNewTokens: 32, stopTokenIDs: [])
            let p = engine.decodeProfile
            let msPerEval = p.aneSeconds.reduce(0, +) * 1000 / Double(max(1, p.aneCalls.reduce(0, +)))
            var mtp = 0.0
            if let drafter {
                engine.drafter = drafter
                engine.clearPrefixCache()
                mtp = try engine.generate(inputIDs: draft, maxNewTokens: 64, stopTokenIDs: stops).stats.decodeTPS
            }
            log(String(format: "ANE_LOOP %@ #%-3d plain %5.1f tok/s (%.2f ms/eval) | MTP draft %5.1f tok/s | %@",
                       fmt.string(from: Date()), iter, a.stats.decodeTPS, msPerEval, mtp, before))
        }
        engine.drafter = drafter
    }

    private static func appState() -> String {
        let read = { () -> UIApplication.State in UIApplication.shared.applicationState }
        let s = Thread.isMainThread ? read() : DispatchQueue.main.sync(execute: read)
        switch s {
        case .active: return "active"
        case .inactive: return "inactive"
        case .background: return "background"
        @unknown default: return "unknown"
        }
    }

    /// App state, thermal state, Low Power Mode and battery, e.g. "app=active thermal=nominal lowPower=0 battery=unplugged 80%".
    static func deviceState() -> String {
        let pi = ProcessInfo.processInfo
        let thermal: String
        switch pi.thermalState {
        case .nominal: thermal = "nominal"
        case .fair: thermal = "fair"
        case .serious: thermal = "serious"
        case .critical: thermal = "critical"
        @unknown default: thermal = "unknown"
        }
        let read = { () -> (String, Float) in
            UIDevice.current.isBatteryMonitoringEnabled = true
            let st: String
            switch UIDevice.current.batteryState {
            case .charging: st = "charging"
            case .full: st = "full"
            case .unplugged: st = "unplugged"
            default: st = "unknown"
            }
            return (st, UIDevice.current.batteryLevel)
        }
        let (battery, level) = Thread.isMainThread ? read() : DispatchQueue.main.sync(execute: read)
        return "app=\(appState()) thermal=\(thermal) lowPower=\(pi.isLowPowerModeEnabled ? 1 : 0) battery=\(battery) \(Int(level * 100))%"
    }
}
