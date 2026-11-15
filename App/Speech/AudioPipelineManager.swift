import AVFoundation
import Accelerate
import Combine
import Foundation
import KomorebiCore
import KomorebiIPC
import UIKit

/// Lets the audio tap drop buffers while the microphone session is only on standby (thread-safe).
private final class AudioGate: @unchecked Sendable {
    private let lock = NSLock()
    private var open = false
    private var passed: UInt64 = 0

    var isOpen: Bool {
        get { lock.lock(); defer { lock.unlock() }; return open }
        set { lock.lock(); open = newValue; lock.unlock() }
    }

    /// Counts and admits a buffer when open.
    func admit() -> Bool {
        lock.lock(); defer { lock.unlock() }
        if open { passed &+= 1 }
        return open
    }

    var passedBuffers: UInt64 { lock.lock(); defer { lock.unlock() }; return passed }
}

public enum DictationMode: String, CaseIterable, Codable, Sendable {
    case sentence = "sentence"
    case draft = "draft"

    public var label: String {
        switch self {
        case .sentence: return "逐句模式"
        case .draft: return "草稿整合"
        }
    }
}

/// Manages 16 kHz microphone capture (with background audio survival for Keyboard Extension),
/// CPU Silero VAD + SenseVoice-Small + OpenCC streaming partial/final transcription,
/// Gemma 4 E2B text refinement on the Apple Neural Engine (direct MIL engine), and bidirectional App Group IPC.
@MainActor
public final class AudioPipelineManager: ObservableObject {
    @Published public private(set) var phase: DictationPhase = .idle
    @Published public private(set) var isListening: Bool = false
    @Published public private(set) var isEngineLoaded: Bool = false
    @Published public private(set) var isLoadingModels: Bool = false
    @Published public private(set) var micLevel: Float = 0
    /// The microphone session is running but not listening: the keyboard can start listening again without
    /// opening the app, because iOS only lets an app *start* recording while it is in the foreground.
    @Published public private(set) var micStandby: Bool = false
    public private(set) var micStandbyUntil: TimeInterval = 0
    /// Minutes the microphone stays on (standby) after listening stops; 0 turns it off right away.
    @Published public var micStandbyMinutes: Int {
        didSet {
            UserDefaults.standard.set(micStandbyMinutes, forKey: Self.micStandbyDefaultsKey)
            if micStandby {
                if micStandbyMinutes == 0 { endMicSession() } else { enterStandby() }
            }
        }
    }
    public static let micStandbyChoices = [0, 1, 5, 10, 30]
    /// Clean up the sentence while it is still being spoken (shown in the keyboard's preview line), not only after
    /// the pause. The finished sentence is still merged into the draft as before.
    @Published public var livePreviewEnabled: Bool {
        didSet {
            UserDefaults.standard.set(livePreviewEnabled, forKey: Self.livePreviewDefaultsKey)
            if !livePreviewEnabled { endLivePreview() }
        }
    }

    // Editor states (also synced with Keyboard Extension over App Group IPC)
    @Published public var mode: DictationMode = .draft
    @Published public var noiseSuppressionEnabled: Bool {
        didSet {
            UserDefaults.standard.set(noiseSuppressionEnabled, forKey: Self.noiseSuppressionDefaultsKey)
            guard oldValue != noiseSuppressionEnabled else { return }
            if audioEngine != nil {
                Task { await restartAudioTapPreservingDraft() }
            } else {
                publishIPCState(committedText: nil, isDraftReplacement: false)
            }
        }
    }
    @Published public private(set) var hardwareVoiceProcessingActive: Bool = false
    @Published public private(set) var systemMicModeLabel: String = "標準"
    public private(set) var hostInForeground: Bool = true
    @Published public var editorText: String = ""
    @Published public var preeditText: String = ""
    /// Words heard after the last live clean-up (dimmed after `preeditText`).
    @Published public private(set) var preeditTail: String = ""
    @Published public private(set) var freshRange: NSRange?
    @Published public private(set) var statusBanner: String = "準備就緒"
    @Published public private(set) var telemetry: KomorebiTelemetry = KomorebiTelemetry()

    private static let noiseSuppressionDefaultsKey = "komorebi_noise_suppression_enabled"
    private static let micStandbyDefaultsKey = "komorebi_mic_standby_minutes"
    private static let livePreviewDefaultsKey = "komorebi_live_preview_enabled"

    public let modelManager: ModelManager
    public let refiner = KomorebiRefiner()

    private var vad: SileroTurnsVAD?
    private var asr: SenseVoiceASR?

    private var audioEngine: AVAudioEngine?
    private nonisolated let audioGate = AudioGate()
    private var standbyTask: Task<Void, Never>?
    private var heartbeatTask: Task<Void, Never>?
    private var engineLoadTask: Task<Void, Never>?
    private var audioObservers: [NSObjectProtocol] = []
    private var audioConverter: AVAudioConverter?
    private var targetFormat: AVAudioFormat?

    private let processingQueue = DispatchQueue(label: "dev.ayasakura.komorebi.audio-vad", qos: .userInitiated)
    private var partialTaskInFlight = false
    /// Finished sentences waiting for the draft merge, with the live preview shown for each when it finished.
    private var refineQueue: [(pcm: [Float], preview: (raw: String, refined: String)?, serial: Int)] = []
    /// Draft mode: the cleaned-up sentence being spoken, shown in the text field after the draft until its merge
    /// replaces it.
    private var provisional: (serial: Int, text: String)?
    /// Sentences whose provisional text the user deleted or sent away (field cleared, 定稿): their merge is dropped.
    private var droppedUtterances: Set<Int> = []
    /// Words not yet cleaned up, for the keyboard's (dimmed) preview line when the clean-up goes to the field.
    private var keyboardPreviewRaw = ""
    private var aneReady = false
    private var previewTask: Task<Void, Never>?
    private var lastPreviewRequest = ""
    private var lastPreviewStart: CFAbsoluteTime = 0
    private var latestPartialRaw = ""
    private var livePreview: (raw: String, refined: String)?
    private var utteranceSerial = 0
    private var refineTaskRunning = false

    private final class WeakPipelineRef: @unchecked Sendable {
        weak var value: AudioPipelineManager?
        init(_ value: AudioPipelineManager) { self.value = value }
    }

    private nonisolated static let darwinObserverLock = NSLock()
    private nonisolated(unsafe) static var activeDarwinObservers: [UInt: WeakPipelineRef] = [:]

    private var ipcSequence: UInt64 = 0
    private var ipcCommitID: UInt64 = 0
    /// Every published state repeats the latest commit, so a reader that misses the commit's own publish (another
    /// state was written right after it) still gets the same (commitID, text) pair, never an empty text.
    private var lastCommitText = ""
    private var lastCommitIsDraftReplacement = false
    private var lastHandledCommandID: UInt64 = 0
    private var pendingStartListeningOnForeground: Bool = false
    private var lastContextTail: String = ""
    private var historyStack: [String] = []

    // Draft sync with the keyboard: the host text field shows the draft, and the user can edit it there.
    /// Last keyboard draft edit folded into `editorText` (starts at the one already on disk, which is stale).
    private var lastDraftEditID: UInt64 = KomorebiIPCStore.shared.readDraftEdit()?.id ?? 0
    /// Bumped whenever the draft is replaced by an edit (keyboard, 定稿, reset) rather than by a merge.
    private var draftEditVersion: UInt64 = 0
    /// Recent draft commits and the sentence each folded in, to rebase an edit made before the keyboard saw them.
    private var recentDraftCommits: [(commitID: UInt64, raw: String)] = []
    /// Random per launch: lets the keyboard notice a restarted host (whose draft is empty) and detach its draft.
    private let hostSessionID = UInt64.random(in: 1...UInt64.max)

    public init(modelManager: ModelManager) {
        self.modelManager = modelManager
        if UserDefaults.standard.object(forKey: Self.noiseSuppressionDefaultsKey) == nil {
            UserDefaults.standard.set(true, forKey: Self.noiseSuppressionDefaultsKey)
        }
        self.noiseSuppressionEnabled = UserDefaults.standard.bool(forKey: Self.noiseSuppressionDefaultsKey)
        self.micStandbyMinutes = (UserDefaults.standard.object(forKey: Self.micStandbyDefaultsKey) as? Int) ?? 10
        self.livePreviewEnabled = (UserDefaults.standard.object(forKey: Self.livePreviewDefaultsKey) as? Bool) ?? true
        superInitRefresh()
    }

    deinit {
        let observerPtr = Unmanaged.passUnretained(self).toOpaque()
        let key = UInt(bitPattern: observerPtr)
        Self.darwinObserverLock.lock()
        Self.activeDarwinObservers.removeValue(forKey: key)
        Self.darwinObserverLock.unlock()
        let center = CFNotificationCenterGetDarwinNotifyCenter()
        CFNotificationCenterRemoveEveryObserver(center, observerPtr)
    }

    private func superInitRefresh() {
        refreshSystemMicMode()
        setupDarwinCommandObserver()
        observeAudioSession()
        publishIPCState(committedText: nil, isDraftReplacement: false)
    }

    /// A call, Siri or another app can take the microphone; a route change stops the engine. Either way the session
    /// must not look alive to the keyboard when it is not.
    private func observeAudioSession() {
        let nc = NotificationCenter.default
        audioObservers.append(nc.addObserver(forName: AVAudioSession.interruptionNotification, object: nil, queue: .main) { [weak self] note in
            guard let raw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
                  AVAudioSession.InterruptionType(rawValue: raw) == .began else { return }
            Task { @MainActor in self?.handleAudioInterruption() }
        })
        audioObservers.append(nc.addObserver(forName: .AVAudioEngineConfigurationChange, object: nil, queue: .main) { [weak self] note in
            let engine = note.object as AnyObject?
            Task { @MainActor in self?.handleEngineConfigurationChange(engine) }
        })
    }

    private func handleAudioInterruption() {
        guard audioEngine != nil else { return }
        let wasListening = isListening
        diag("audio session interrupted (listening=\(wasListening))")
        teardownMicSession()
        phase = refineTaskRunning ? .refining : .idle
        statusBanner = wasListening ? "麥克風被中斷（來電或其他 App），請再點一次語音輸入" : "麥克風待命被中斷"
        publishIPCState(committedText: nil, isDraftReplacement: false)
    }

    private func handleEngineConfigurationChange(_ object: AnyObject?) {
        guard let engine = audioEngine, object === engine, !engine.isRunning else { return }
        diag("audio engine configuration changed; restarting the microphone")
        Task { await restartAudioTapPreservingDraft() }
    }

    /// Appends a line to Documents/komorebi-diag.log (and the console) for checking the microphone session on a device.
    func diag(_ message: String) {
        let line = "\(ISO8601DateFormatter().string(from: Date())) [\(UIApplication.shared.applicationState == .background ? "bg" : "fg")] \(message)\n"
        NSLog("KOMOREBI_DIAG %@", message)
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("komorebi-diag.log")
        if let size = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int, size > 256_000 {
            try? FileManager.default.removeItem(at: url)
        }
        guard let data = line.data(using: .utf8) else { return }
        if let handle = try? FileHandle(forWritingTo: url) {
            handle.seekToEndOfFile()
            handle.write(data)
            try? handle.close()
        } else {
            try? data.write(to: url)
        }
    }

    public func refreshSystemMicMode() {
        switch AVCaptureDevice.activeMicrophoneMode {
        case .voiceIsolation:
            systemMicModeLabel = "語音隔離 (AI)"
        case .wideSpectrum:
            systemMicModeLabel = "寬廣頻譜"
        case .standard:
            systemMicModeLabel = "標準"
        @unknown default:
            systemMicModeLabel = "標準"
        }
    }

    public func openSystemMicrophoneModePicker() {
        AVCaptureDevice.showSystemUserInterface(.microphoneModes)
    }

    // MARK: - Model Loading

    /// Loads VAD, ASR and Gemma once; a caller that arrives while a load is running waits for it (the keyboard's
    /// first tap opens the app, whose launch is already loading, and must still end up listening).
    public func loadAllEnginesIfNeeded() async {
        if isEngineLoaded { return }
        if let running = engineLoadTask {
            await running.value
            return
        }
        let task = Task { @MainActor in await self.performEngineLoad() }
        engineLoadTask = task
        await task.value
        engineLoadTask = nil
    }

    private func performEngineLoad() async {
        modelManager.scanInstalledModels()
        guard modelManager.coreModelsReady,
              let sileroURL = modelManager.sileroURL,
              let svModel = modelManager.senseVoiceModelURL,
              let svTokens = modelManager.senseVoiceTokensURL,
              let ggufURL = modelManager.gemma4GGUFURL else {
            statusBanner = "請先下載模型，才能離線語音輸入。"
            publishIPCState(committedText: nil, isDraftReplacement: false)
            return
        }

        isLoadingModels = true
        statusBanner = "載入 Silero VAD、SenseVoice 與 Gemma 4 E2B…"
        publishIPCState(committedText: nil, isDraftReplacement: false)
        let loadStart = CFAbsoluteTimeGetCurrent()

        do {
            let vadInstance = SileroTurnsVAD(modelURL: sileroURL, numThreads: 1)
            let asrInstance = try SenseVoiceASR(modelURL: svModel, tokensURL: svTokens, numThreads: 2)
            self.vad = vadInstance
            self.asr = asrInstance
            let tSpeech = CFAbsoluteTimeGetCurrent()

            // Gemma 4 E2B runs only on the Apple Neural Engine (direct MIL engine). The first launch re-quantises the
            // weights for the ANE and compiles the programs (about a minute); later launches load them from the cache.
            try await refiner.loadGGUF(url: ggufURL)
            let tGGUF = CFAbsoluteTimeGetCurrent()
            diag(String(format: "startup: VAD+SenseVoice %.2f s, GGUF %.2f s", tSpeech - loadStart, tGGUF - tSpeech))
            var aneFailure: String?
            if isRunningInSimulator {
                aneFailure = "模擬器沒有 Neural Engine"
            } else {
                let cacheDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                    .appendingPathComponent("komorebi-ane", isDirectory: true)
                do {
                    // Warm-up runs after "ready" (below): listening and recognition do not need Gemma, and the first
                    // sentence's clean-up waits for the warm-up if it is still running.
                    let breakdown = try await refiner.loadANEDirect(
                        cacheDirectory: cacheDir, drafterDirectory: modelManager.gemma4DrafterDirectory, warmUpNow: false,
                        progress: { msg in Task { @MainActor in self.statusBanner = "準備 ANE：\(msg)" } })
                    diag("startup: ANE \(breakdown)")
                } catch {
                    aneFailure = error.localizedDescription
                }
            }

            // Listening only needs VAD + ASR; without the ANE engine sentences are inserted as recognised.
            isEngineLoaded = true
            aneReady = await refiner.aneDirect != nil
            if let desc = await refiner.aneDirectDescription {
                statusBanner = "就緒：\(desc)"
            } else {
                statusBanner = "就緒：僅語音辨識（Gemma 未載入：\(aneFailure ?? "未知原因")）"
            }
        } catch {
            phase = .error
            statusBanner = "模型載入失敗：\(error.localizedDescription)"
        }
        isLoadingModels = false
        let ready = CFAbsoluteTimeGetCurrent() - loadStart
        NSLog("KOMOREBI_STATUS engines loaded in %.1f s: %@", ready, statusBanner)
        diag(String(format: "startup: ready in %.2f s", ready))
        publishIPCState(committedText: nil, isDraftReplacement: false)
        if await refiner.aneDirect != nil {
            Task {
                let seconds = (try? await refiner.warmUpANE()) ?? -1
                diag(String(format: "startup: ANE warm-up %.2f s (after ready)", seconds))
                NSLog("KOMOREBI_STATUS ANE warm-up %.2f s", seconds)
            }
        }
    }

    private var isRunningInSimulator: Bool {
#if targetEnvironment(simulator)
        return true
#else
        return false
#endif
    }

    // MARK: - Listening Lifecycle (Works in Foreground + Background for Keyboard Extension)

    public func toggleListening() {
        if isListening {
            stopListening()
        } else {
            Task { await startListening() }
        }
    }

    public var isMicLive: Bool { audioEngine != nil }
    /// Audio buffers that reached VAD since launch (diagnostics / self-test).
    public var processedAudioBuffers: UInt64 { audioGate.passedBuffers }

    /// Starts listening. A new microphone session needs the app in the foreground; while one is running (listening
    /// or standby) this also works in the background, which is how the keyboard starts again without opening the app.
    /// `returnToCaller`: after the microphone starts, go back to the app that opened Komorebi (keyboard's first tap).
    public func startListening(returnToCaller: Bool = false, hostBundleID: String? = nil) async {
        if isListening { return }
        if audioEngine == nil {
            await waitUntilActive(timeout: 2)
            let granted = await requestMicPermission()
            guard granted else {
                phase = .error
                statusBanner = "請至系統設定開啟麥克風權限。"
                publishIPCState(committedText: nil, isDraftReplacement: false)
                return
            }
            do {
                try startAudioCaptureEngine(resetVAD: true)
                startHeartbeat()
                diag("microphone session started")
            } catch {
                diag("microphone session failed: \(error.localizedDescription)")
                phase = .error
                statusBanner = "啟動麥克風失敗：\(error.localizedDescription)"
                publishIPCState(committedText: nil, isDraftReplacement: false)
                return
            }
        }
        if returnToCaller {
            let result = await SystemNavigation.returnToPreviousApp(hostBundleID: hostBundleID)
            diag("return to the previous app: \(result)")
        }
        if !isEngineLoaded {
            publishIPCState(committedText: nil, isDraftReplacement: false)
            await loadAllEnginesIfNeeded()
        }
        guard isEngineLoaded, vad != nil, asr != nil else {
            enterStandby()
            publishIPCState(committedText: nil, isDraftReplacement: false)
            return
        }
        cancelStandby()
        vad?.reset()
        micStandby = false
        micStandbyUntil = 0
        isListening = true
        audioGate.isOpen = true
        phase = .listening
        let nsTag = hardwareVoiceProcessingActive ? "iOS 降噪開啟" : (noiseSuppressionEnabled ? "系統語音模式" : "原始收音")
        statusBanner = "聆聽中（\(nsTag)，可切至任意 App 使用鍵盤）"
        refreshSystemMicMode()
        diag("listening")
        publishIPCState(committedText: nil, isDraftReplacement: false)
    }

    private func waitUntilActive(timeout: Double) async {
        let deadline = Date().addingTimeInterval(timeout)
        while UIApplication.shared.applicationState != .active && Date() < deadline {
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
    }

    /// Keeps the microphone session alive (audio is dropped) for `micStandbyMinutes`.
    private func enterStandby() {
        guard audioEngine != nil else { return }
        guard micStandbyMinutes > 0 else {
            teardownMicSession()
            return
        }
        micStandby = true
        let seconds = Double(micStandbyMinutes) * 60
        micStandbyUntil = Date().timeIntervalSince1970 + seconds
        standbyTask?.cancel()
        standbyTask = Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            guard let self, !Task.isCancelled, self.micStandby, !self.isListening else { return }
            self.diag("standby ended")
            self.teardownMicSession()
            self.statusBanner = "麥克風待命結束，已關閉"
            self.publishIPCState(committedText: nil, isDraftReplacement: false)
        }
    }

    private func cancelStandby() {
        standbyTask?.cancel()
        standbyTask = nil
    }

    /// Stops the engine and releases the microphone; the next start needs the app in the foreground again.
    private func teardownMicSession() {
        audioGate.isOpen = false
        cancelStandby()
        heartbeatTask?.cancel()
        heartbeatTask = nil
        micStandby = false
        micStandbyUntil = 0
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        audioConverter = nil
        hardwareVoiceProcessingActive = false
        isListening = false
        micLevel = 0
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
    }

    /// While the microphone session runs, republish state every 2 s so the keyboard knows the host is reachable.
    private func startHeartbeat() {
        heartbeatTask?.cancel()
        heartbeatTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                guard let self, self.audioEngine != nil else { return }
                self.publishIPCState(committedText: nil, isDraftReplacement: false)
            }
        }
    }

    private func startAudioCaptureEngine(resetVAD: Bool) throws {
        let session = AVAudioSession.sharedInstance()
        let sessionMode: AVAudioSession.Mode = noiseSuppressionEnabled ? .voiceChat : .measurement
        try session.setCategory(
            .playAndRecord,
            mode: sessionMode,
            options: [.mixWithOthers, .defaultToSpeaker, .allowBluetoothHFP]
        )
        try session.setPreferredSampleRate(16_000)
        try session.setActive(true)

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode

        // Attach Apple VoiceProcessingIO (noise suppression, AEC, AGC) BEFORE querying outputFormat.
        var vpActive = false
        if noiseSuppressionEnabled {
            do {
                try inputNode.setVoiceProcessingEnabled(true)
                inputNode.isVoiceProcessingBypassed = false
                inputNode.isVoiceProcessingAGCEnabled = true
                inputNode.voiceProcessingOtherAudioDuckingConfiguration = AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
                    enableAdvancedDucking: false,
                    duckingLevel: .min
                )
                vpActive = inputNode.isVoiceProcessingEnabled
            } catch {
                // Simulator or unsupported external routes fall back cleanly without breaking capture.
                vpActive = false
            }
        }
        self.hardwareVoiceProcessingActive = vpActive

        let inputFormat = inputNode.outputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0 else {
            throw NSError(
                domain: "KomorebiAudio",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "找不到可用的麥克風輸入裝置。"]
            )
        }
        guard let desired = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16_000,
            channels: 1,
            interleaved: false
        ) else {
            throw NSError(domain: "KomorebiAudio", code: -2, userInfo: [NSLocalizedDescriptionKey: "無法建立 16 kHz PCM 格式。"])
        }

        let converter = AVAudioConverter(from: inputFormat, to: desired)
        self.audioConverter = converter
        self.targetFormat = desired

        if resetVAD {
            vad?.reset()
        }

        let bufferSize: AVAudioFrameCount = 1024
        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: inputFormat) { [weak self] pcmBuf, _ in
            self?.handleRawAudioBuffer(pcmBuf, inputFormat: inputFormat, desired: desired)
        }

        engine.prepare()
        try engine.start()
        self.audioEngine = engine
    }

    private func restartAudioTapPreservingDraft() async {
        guard audioEngine != nil else { return }
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        audioConverter = nil
        do {
            try startAudioCaptureEngine(resetVAD: false)
            let nsTag = hardwareVoiceProcessingActive ? "iOS 降噪開啟" : (noiseSuppressionEnabled ? "系統語音模式" : "原始收音")
            self.statusBanner = isListening ? "聆聽中（\(nsTag)）" : statusBanner
            refreshSystemMicMode()
            publishIPCState(committedText: nil, isDraftReplacement: false)
        } catch {
            diag("microphone restart failed: \(error.localizedDescription)")
            teardownMicSession()
            phase = .error
            statusBanner = "重啟麥克風失敗：\(error.localizedDescription)"
            publishIPCState(committedText: nil, isDraftReplacement: false)
        }
    }

    /// Stops listening. With `keepMicWarm` the microphone session stays on standby for `micStandbyMinutes`, so the
    /// keyboard can start again without opening the app; otherwise the microphone is released.
    public func stopListening(keepMicWarm: Bool = true) {
        let wasListening = isListening
        audioGate.isOpen = false
        isListening = false
        micLevel = 0
        refiner.cancelPreview()
        if wasListening, let vad {
            processingQueue.async { [weak self] in
                let flushEvents = vad.flush()
                self?.dispatchVADEvents(flushEvents)
            }
        }
        if keepMicWarm && micStandbyMinutes > 0 && audioEngine != nil {
            enterStandby()
            statusBanner = "已暫停；麥克風待命 \(micStandbyMinutes) 分鐘，鍵盤可直接再開始"
        } else {
            teardownMicSession()
            statusBanner = "已停止聆聽"
        }
        diag(micStandby ? "stopped; standby \(micStandbyMinutes) min" : "stopped; microphone off")
        phase = refineTaskRunning ? .refining : .idle
        publishIPCState(committedText: nil, isDraftReplacement: false)
    }

    /// Turns the microphone off now (from standby or while listening).
    public func endMicSession() {
        stopListening(keepMicWarm: false)
    }

    public func flushCurrentUtterance() {
        guard let vad else { return }
        processingQueue.async { [weak self] in
            let flushEvents = vad.flush()
            self?.dispatchVADEvents(flushEvents)
        }
    }

    public func resetDraft() {
        if let p = provisional {
            droppedUtterances.insert(p.serial)
            provisional = nil
        }
        historyStack.append(editorText)
        editorText = ""
        preeditText = ""
        freshRange = nil
        lastContextTail = ""
        draftEditVersion &+= 1
        publishIPCState(committedText: "", isDraftReplacement: true)
    }

    /// Folds the keyboard's latest edit of the draft (the draft as it now is in the host text field) into `editorText`.
    /// Sentences merged after the commit the keyboard's edit was based on are appended to it as heard.
    private func applyKeyboardDraftEdit() {
        guard let edit = KomorebiIPCStore.shared.readDraftEdit(), edit.id > lastDraftEditID else { return }
        lastDraftEditID = edit.id
        if let p = provisional {
            let committedIntact = edit.text.hasPrefix(editorText)
            if committedIntact && edit.text.count > editorText.count {
                // Only the sentence still being spoken was touched; its next clean-up or merge redraws it.
                publishIPCState(committedText: nil, isDraftReplacement: false)
                return
            }
            // That sentence was deleted, sent away with the field, or kept as typed text (定稿): drop its merge.
            droppedUtterances.insert(p.serial)
            provisional = nil
        }
        var text = edit.text
        let missed = recentDraftCommits.filter { $0.commitID > edit.baseCommitID }
        for commit in missed { text = KomorebiPrompts.appendDraft(text, commit.raw) }
        if text != editorText {
            historyStack.append(editorText)
            editorText = text
            freshRange = nil
        }
        if text.isEmpty { lastContextTail = "" }
        draftEditVersion &+= 1
        if missed.isEmpty {
            publishIPCState(committedText: nil, isDraftReplacement: false)  // acknowledge: the field already shows it
        } else {
            publishIPCState(committedText: text, isDraftReplacement: true)
        }
    }

    public func undoLastCommit() {
        if let prev = historyStack.popLast() {
            editorText = prev
            freshRange = nil
        }
    }

    public func applySelectionVoiceEdit(before: String, selected: String, after: String, spokenInstruction: String) async {
        guard !spokenInstruction.isEmpty else { return }
        phase = .refining
        statusBanner = "Gemma 4 正在改寫圈選文字…"
        do {
            let res = try await refiner.selectionEdit(
                before: before,
                selected: selected,
                after: after,
                raw: spokenInstruction
            )
            historyStack.append(editorText)
            editorText = before + res.text + after
            freshRange = NSRange(location: before.count, length: res.text.count)
            telemetry.backendUsed = res.stats.backend
            telemetry.prefillTokens = res.stats.prefillTokens
            telemetry.prefillTPS = res.stats.prefillTPS
            telemetry.decodeTokens = res.stats.decodeTokens
            telemetry.decodeTPS = res.stats.decodeTPS
            statusBanner = "已更新圈選區塊"
        } catch {
            statusBanner = "改寫失敗：\(error.localizedDescription)"
        }
        phase = isListening ? .listening : .idle
        publishIPCState(committedText: nil, isDraftReplacement: false)
    }

    // MARK: - Internal Audio -> VAD -> Partial / Final Pipeline

    private nonisolated func handleRawAudioBuffer(
        _ buffer: AVAudioPCMBuffer,
        inputFormat: AVAudioFormat,
        desired: AVAudioFormat
    ) {
        guard audioGate.admit() else { return }  // standby: the session stays alive, the audio is dropped
        let converted: [Float]
        if inputFormat.sampleRate == 16_000 && inputFormat.channelCount == 1 && inputFormat.commonFormat == .pcmFormatFloat32 {
            guard let ch0 = buffer.floatChannelData?[0] else { return }
            converted = Array(UnsafeBufferPointer(start: ch0, count: Int(buffer.frameLength)))
        } else {
            let ratio = 16_000.0 / inputFormat.sampleRate
            let outCap = AVAudioFrameCount(Double(buffer.frameLength) * ratio + 32)
            guard let conv = AVAudioConverter(from: inputFormat, to: desired),
                  let outBuf = AVAudioPCMBuffer(pcmFormat: desired, frameCapacity: outCap) else { return }
            var supplied = false
            var err: NSError?
            let status = conv.convert(to: outBuf, error: &err) { _, outStatus in
                if supplied {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                supplied = true
                outStatus.pointee = .haveData
                return buffer
            }
            guard status != .error, err == nil, let ch0 = outBuf.floatChannelData?[0] else { return }
            converted = Array(UnsafeBufferPointer(start: ch0, count: Int(outBuf.frameLength)))
        }

        // Compute RMS meter level
        var rms: Float = 0
        vDSP_rmsqv(converted, 1, &rms, vDSP_Length(converted.count))
        let scaledLevel = min(1.0, rms * 10.0)

        Task { @MainActor [weak self] in
            self?.micLevel = scaledLevel
        }

        processingQueue.async { [weak self] in
            guard let self else { return }
            let vadEvents: [SileroTurnsVAD.Event] = MainActor.assumeIsolatedOrSync { self.vad?.push(samples: converted) ?? [] }
            self.dispatchVADEvents(vadEvents)
        }
    }

    private nonisolated func dispatchVADEvents(_ events: [SileroTurnsVAD.Event]) {
        guard !events.isEmpty else { return }
        for ev in events {
            switch ev {
            case .speechStarted:
                Task { @MainActor in
                    self.phase = .transcribing
                    self.publishIPCState(committedText: nil, isDraftReplacement: false)
                }

            case .partialWindow(let pcm):
                Task { @MainActor in
                    guard !self.partialTaskInFlight, let asr = self.asr else { return }
                    self.partialTaskInFlight = true
                    let serial = self.utteranceSerial
                    Task.detached(priority: .userInitiated) {
                        let res = asr.transcribe(samples: pcm, isPartial: true)
                        await MainActor.run {
                            self.partialTaskInFlight = false
                            // A partial that started before the sentence finished belongs to the finished sentence.
                            if self.isListening && !res.text.isEmpty && self.utteranceSerial == serial {
                                self.showPartial(res.text)
                                self.telemetry.asrLatencyMs = res.latencyMs
                                self.publishIPCState(committedText: nil, isDraftReplacement: false)
                                self.maybeStartPreview()
                            }
                        }
                    }
                }

            case .finalizedSegment(let pcm):
                Task { @MainActor in
                    self.enqueueFinalizedSegment(pcm)
                }
            }
        }
    }

    // MARK: - Simulated speech (self-test)

    /// Starts "listening" without the microphone so a self-test can feed recorded/synthesized speech.
    func beginSimulatedListening() {
        isListening = true
        audioGate.isOpen = true
        phase = .listening
        vad?.reset()
        publishIPCState(committedText: nil, isDraftReplacement: false)
    }

    /// Feeds 16 kHz mono samples through the same VAD → ASR → Gemma path as the microphone.
    func feedSimulatedSamples(_ samples: [Float]) {
        processingQueue.async { [weak self] in
            guard let self else { return }
            let events: [SileroTurnsVAD.Event] = MainActor.assumeIsolatedOrSync { self.vad?.push(samples: samples) ?? [] }
            self.dispatchVADEvents(events)
        }
    }

    var isMergingSentence: Bool { refineTaskRunning || !refineQueue.isEmpty }

    // MARK: - Live preview (clean-up while speaking)

    /// Draft mode with live preview: each clean-up goes straight into the text field, after the draft.
    private var previewInField: Bool { mode == .draft && livePreviewEnabled && aneReady }

    /// The draft as the text field should show it: merged sentences plus the cleaned-up sentence being spoken.
    private func displayedDraft() -> String {
        guard let p = provisional else { return editorText }
        return KomorebiPrompts.appendDraft(editorText, p.text)
    }

    @discardableResult
    private func publishDraftDisplay() -> UInt64 {
        publishIPCState(committedText: displayedDraft(), isDraftReplacement: true)
    }

    private func clearProvisional(for serial: Int) {
        guard let p = provisional, p.serial == serial else { return }
        provisional = nil
        publishDraftDisplay()
    }

    /// Shows the sentence being spoken: once it has a live clean-up, that stays on screen and only the words heard
    /// since (by length; SenseVoice re-transcribes the whole sentence, so earlier words can change) follow it dimmed.
    private func showPartial(_ raw: String) {
        latestPartialRaw = raw
        (preeditText, preeditTail) = Self.previewDisplay(raw: raw, preview: livePreview)
        keyboardPreviewRaw = livePreview == nil ? raw : preeditTail
    }

    static func previewDisplay(raw: String, preview: (raw: String, refined: String)?) -> (main: String, tail: String) {
        guard let preview else { return (raw, "") }
        let closing: Set<Character> = ["。", "．", ".", "？", "?", "！", "!", "，", ","]
        func trimmed(_ s: String) -> String {
            var s = s
            while let last = s.last, closing.contains(last) || last.isWhitespace { s.removeLast() }
            return s
        }
        // Partial transcripts end with their own full stop; only real new words count.
        let newRaw = trimmed(raw), oldRaw = trimmed(preview.raw)
        let extra = newRaw.count - oldRaw.count
        guard extra > 0 else { return (preview.refined, "") }
        // The clean-up closes the unfinished sentence; drop that while more words follow.
        return (trimmed(preview.refined), String(newRaw.suffix(extra)))
    }

    /// While a sentence is being spoken, cleans up its partial transcript on the ANE (one preview at a time, at most
    /// every 0.5 s, never while a finished sentence is being merged).
    private func maybeStartPreview() {
        guard livePreviewEnabled, aneReady, isListening, !refineTaskRunning, previewTask == nil else { return }
        let raw = latestPartialRaw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard raw.count >= 4, raw != lastPreviewRequest,
              CFAbsoluteTimeGetCurrent() - lastPreviewStart >= 0.5 else { return }
        lastPreviewRequest = raw
        lastPreviewStart = CFAbsoluteTimeGetCurrent()
        let serial = utteranceSerial
        previewTask = Task { [weak self] in
            guard let self else { return }
            // No earlier text as context: for a very short sentence Gemma sometimes returned the context itself.
            let result = try? await self.refiner.previewSentence(raw: raw, context: "")
            self.previewTask = nil
            guard let result, self.utteranceSerial == serial, self.isListening,
                  !self.droppedUtterances.contains(serial) else { return }
            self.livePreview = (raw, result.text)
            self.showPartial(self.latestPartialRaw)
            if self.previewInField {
                self.provisional = (serial, result.text)
                self.publishDraftDisplay()
            } else {
                self.publishIPCState(committedText: nil, isDraftReplacement: false)
            }
            self.maybeStartPreview()  // the transcript may have grown meanwhile
        }
    }

    private func endLivePreview() {
        refiner.cancelPreview()
        utteranceSerial += 1
        livePreview = nil
        lastPreviewRequest = ""
        latestPartialRaw = ""
    }

    private func enqueueFinalizedSegment(_ pcm: [Float]) {
        let preview = livePreview
        let serial = utteranceSerial
        endLivePreview()
        refineQueue.append((pcm, preview, serial))
        if !refineTaskRunning {
            Task { await drainRefineQueue() }
        }
    }

    private func drainRefineQueue() async {
        refineTaskRunning = true
        defer { refineTaskRunning = false }

        while !refineQueue.isEmpty {
            let item = refineQueue.removeFirst()
            let pcm = item.pcm
            guard let asr else { continue }
            if droppedUtterances.remove(item.serial) != nil {
                clearProvisional(for: item.serial)
                continue
            }

            phase = .transcribing
            let transcript = await Task.detached(priority: .userInitiated) {
                asr.transcribe(samples: pcm, isPartial: false)
            }.value

            telemetry.asrLatencyMs = transcript.latencyMs
            let raw = transcript.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !raw.isEmpty else {
                preeditText = ""
                keyboardPreviewRaw = ""
                phase = isListening ? .listening : .idle
                if provisional?.serial == item.serial { clearProvisional(for: item.serial) } else {
                    publishIPCState(committedText: nil, isDraftReplacement: false)
                }
                continue
            }

            phase = .refining
            // While the sentence is merged, keep showing its cleaned-up preview.
            (preeditText, preeditTail) = Self.previewDisplay(raw: raw, preview: item.preview)
            keyboardPreviewRaw = item.preview == nil ? raw : preeditTail
            statusBanner = "Gemma 4 E2B 整理文字中…"
            publishIPCState(committedText: nil, isDraftReplacement: false)

            do {
                if mode == .draft {
                    applyKeyboardDraftEdit()
                    let prev = editorText
                    let version = draftEditVersion
                    let res = try await refiner.draftMerge(draft: prev, raw: raw)
                    applyKeyboardDraftEdit()
                    if droppedUtterances.remove(item.serial) != nil {
                        // The user deleted or sent the sentence while it was being merged.
                        clearProvisional(for: item.serial)
                        preeditText = ""
                        preeditTail = ""
                        keyboardPreviewRaw = ""
                        phase = isListening ? .listening : .idle
                        continue
                    }
                    var updated = res.text
                    if draftEditVersion != version || editorText != prev {
                        // The draft was edited while Gemma rewrote it (backspace in the field, 定稿, typing in the
                        // app): the edited text wins and this sentence is appended as heard, so nothing deleted
                        // comes back.
                        updated = KomorebiPrompts.appendDraft(editorText, raw)
                    }
                    let current = editorText
                    historyStack.append(current)
                    editorText = updated
                    if provisional?.serial == item.serial { provisional = nil }  // the merge replaces it in the field
                    preeditText = ""
                    preeditTail = ""
                    keyboardPreviewRaw = ""
                    markFreshDelta(previous: current, updated: updated)
                    updateTelemetry(res.stats)
                    let commitID = publishDraftDisplay()
                    recentDraftCommits.append((commitID, raw))
                    if recentDraftCommits.count > 16 { recentDraftCommits.removeFirst(recentDraftCommits.count - 16) }
                } else {
                    let res = try await refiner.refineSentence(raw: raw, context: lastContextTail)
                    let prev = editorText
                    historyStack.append(prev)
                    let appended = prev.isEmpty ? res.text : res.text
                    editorText = prev + appended
                    preeditText = ""
                    freshRange = NSRange(location: prev.count, length: appended.count)
                    lastContextTail = String(editorText.suffix(KomorebiPrompts.refineContextChars))
                    updateTelemetry(res.stats)
                    publishIPCState(committedText: appended, isDraftReplacement: false)
                }
                statusBanner = isListening ? "聆聽中" : "已完成整理"
            } catch {
                // Safe degraded fallback to SenseVoice + OpenCC raw text
                let prev = editorText
                historyStack.append(prev)
                preeditText = ""
                preeditTail = ""
                keyboardPreviewRaw = ""
                statusBanner = "整理降級為辨識原文：\(error.localizedDescription)"
                if mode == .draft {
                    editorText = KomorebiPrompts.appendDraft(prev, raw)
                    if provisional?.serial == item.serial { provisional = nil }
                    publishDraftDisplay()
                } else {
                    editorText = prev + raw
                    publishIPCState(committedText: raw, isDraftReplacement: false)
                }
            }

            phase = isListening ? .listening : .idle
        }
    }

    private func updateTelemetry(_ stats: GenerationStats) {
        telemetry.backendUsed = stats.backend
        telemetry.prefillTokens = stats.prefillTokens
        telemetry.prefillTPS = stats.prefillTPS
        telemetry.decodeTokens = stats.decodeTokens
        telemetry.decodeTPS = stats.decodeTPS
    }

    private func markFreshDelta(previous: String, updated: String) {
        let pChars = Array(previous)
        let uChars = Array(updated)
        var commonPrefix = 0
        while commonPrefix < pChars.count && commonPrefix < uChars.count && pChars[commonPrefix] == uChars[commonPrefix] {
            commonPrefix += 1
        }
        var pSuffix = pChars.count
        var uSuffix = uChars.count
        while pSuffix > commonPrefix && uSuffix > commonPrefix && pChars[pSuffix - 1] == uChars[uSuffix - 1] {
            pSuffix -= 1
            uSuffix -= 1
        }
        if uSuffix > commonPrefix {
            freshRange = NSRange(location: commonPrefix, length: uSuffix - commonPrefix)
        } else {
            freshRange = nil
        }
    }

    private func requestMicPermission() async -> Bool {
        if #available(iOS 17.0, *) {
            return await AVAudioApplication.requestRecordPermission()
        } else {
            return await withCheckedContinuation { cont in
                AVAudioSession.sharedInstance().requestRecordPermission { granted in
                    cont.resume(returning: granted)
                }
            }
        }
    }

    // MARK: - App Group + Darwin IPC

    public func setHostForeground(_ active: Bool) {
        hostInForeground = active
        publishIPCState(committedText: nil, isDraftReplacement: false)
        if active && pendingStartListeningOnForeground {
            pendingStartListeningOnForeground = false
            Task { await startListening() }
        }
    }

    @discardableResult
    public func publishIPCState(committedText: String?, isDraftReplacement: Bool) -> UInt64 {
        ipcSequence &+= 1
        if let committedText {
            ipcCommitID &+= 1
            lastCommitText = committedText
            lastCommitIsDraftReplacement = isDraftReplacement
        }
        let state = KomorebiIPCState(
            sequence: ipcSequence,
            updatedAt: Date().timeIntervalSince1970,
            hostAppAlive: true,
            hostInForeground: hostInForeground,
            audioSessionActive: isListening,
            modelsReady: modelManager.coreModelsReady,
            phase: phase,
            micLevel: micLevel,
            partialText: previewInField ? "" : preeditText,
            commitID: ipcCommitID,
            committedText: lastCommitText,
            isDraftReplacement: lastCommitIsDraftReplacement,
            noiseSuppressionActive: noiseSuppressionEnabled,
            micIsolationLabel: systemMicModeLabel,
            errorMessage: phase == .error ? statusBanner : nil,
            telemetry: telemetry,
            draftEditAck: lastDraftEditID,
            hostSessionID: hostSessionID,
            micLive: audioEngine != nil,
            micStandbyUntil: micStandby ? micStandbyUntil : 0,
            enginesLoading: isLoadingModels,
            partialTail: previewInField ? keyboardPreviewRaw : (preeditText.isEmpty ? "" : preeditTail)
        )
        KomorebiIPCStore.shared.writeState(state)
        return ipcCommitID
    }

    private func setupDarwinCommandObserver() {
        let center = CFNotificationCenterGetDarwinNotifyCenter()
        let observerPtr = Unmanaged.passUnretained(self).toOpaque()
        let key = UInt(bitPattern: observerPtr)
        Self.darwinObserverLock.lock()
        Self.activeDarwinObservers[key] = WeakPipelineRef(self)
        Self.darwinObserverLock.unlock()
        CFNotificationCenterAddObserver(
            center,
            observerPtr,
            { _, observer, _, _, _ in
                guard let observer else { return }
                let key = UInt(bitPattern: observer)
                AudioPipelineManager.darwinObserverLock.lock()
                let mgr = AudioPipelineManager.activeDarwinObservers[key]?.value
                AudioPipelineManager.darwinObserverLock.unlock()
                guard let mgr else { return }
                Task { @MainActor in
                    mgr.handleIncomingCommand()
                }
            },
            KomorebiAppConfig.commandNotification as CFString,
            nil,
            .deliverImmediately
        )
        CFNotificationCenterAddObserver(
            center,
            observerPtr,
            { _, observer, _, _, _ in
                guard let observer else { return }
                let key = UInt(bitPattern: observer)
                AudioPipelineManager.darwinObserverLock.lock()
                let mgr = AudioPipelineManager.activeDarwinObservers[key]?.value
                AudioPipelineManager.darwinObserverLock.unlock()
                guard let mgr else { return }
                Task { @MainActor in
                    mgr.applyKeyboardDraftEdit()
                }
            },
            KomorebiAppConfig.draftEditNotification as CFString,
            nil,
            .deliverImmediately
        )
    }

    private func handleIncomingCommand() {
        guard let cmd = KomorebiIPCStore.shared.readCommand(),
              cmd.id != lastHandledCommandID else { return }
        lastHandledCommandID = cmd.id
        self.mode = cmd.useDraftMode ? .draft : .sentence

        switch cmd.action {
        case .startListening:
            if audioEngine != nil || UIApplication.shared.applicationState != .background {
                diag("keyboard: start listening")
                Task { await startListening() }
            } else {
                pendingStartListeningOnForeground = true
            }
        case .stopListening:
            stopListening(keepMicWarm: true)
        case .endMicSession:
            endMicSession()
        case .flushUtterance:
            flushCurrentUtterance()
        case .resetDraft:
            resetDraft()
        case .toggleNoiseSuppression:
            noiseSuppressionEnabled.toggle()
        case .refineManualText:
            if let text = cmd.textPayload, !text.isEmpty {
                Task { await refineManualTextForKeyboard(text) }
            }
        case .heartbeat:
            publishIPCState(committedText: nil, isDraftReplacement: false)
        }
    }

    private func refineManualTextForKeyboard(_ raw: String) async {
        if !isEngineLoaded { await loadAllEnginesIfNeeded() }
        guard isEngineLoaded else { return }
        phase = .refining
        publishIPCState(committedText: nil, isDraftReplacement: false)
        if let res = try? await refiner.refineSentence(raw: raw) {
            updateTelemetry(res.stats)
            publishIPCState(committedText: res.text, isDraftReplacement: false)
        }
        phase = isListening ? .listening : .idle
    }
}

private extension MainActor {
    static func assumeIsolatedOrSync<T>(_ operation: @MainActor () -> T) -> T {
        if Thread.isMainThread {
            return MainActor.assumeIsolated { operation() }
        } else {
            return DispatchQueue.main.sync {
                MainActor.assumeIsolated { operation() }
            }
        }
    }
}
