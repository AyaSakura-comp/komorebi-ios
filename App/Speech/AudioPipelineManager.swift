import AVFoundation
import Accelerate
import Combine
import Foundation
import KomorebiCore
import KomorebiIPC
import UIKit

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
/// Gemma 4 E2B Metal / ANE / CPU text refinement, and bidirectional App Group IPC.
@MainActor
public final class AudioPipelineManager: ObservableObject {
    @Published public private(set) var phase: DictationPhase = .idle
    @Published public private(set) var isListening: Bool = false
    @Published public private(set) var isEngineLoaded: Bool = false
    @Published public private(set) var isLoadingModels: Bool = false
    @Published public private(set) var micLevel: Float = 0

    // Editor states (also synced with Keyboard Extension over App Group IPC)
    @Published public var mode: DictationMode = .draft
    @Published public var backendPreference: InferenceBackendPreference = .auto
    @Published public var noiseSuppressionEnabled: Bool {
        didSet {
            UserDefaults.standard.set(noiseSuppressionEnabled, forKey: Self.noiseSuppressionDefaultsKey)
            guard oldValue != noiseSuppressionEnabled else { return }
            if isListening {
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
    @Published public private(set) var freshRange: NSRange?
    @Published public private(set) var statusBanner: String = "準備就緒"
    @Published public private(set) var telemetry: KomorebiTelemetry = KomorebiTelemetry()

    private static let noiseSuppressionDefaultsKey = "komorebi_noise_suppression_enabled"

    public let modelManager: ModelManager
    public let refiner = KomorebiRefiner()

    private var vad: SileroTurnsVAD?
    private var asr: SenseVoiceASR?

    private var audioEngine: AVAudioEngine?
    private var audioConverter: AVAudioConverter?
    private var targetFormat: AVAudioFormat?

    private let processingQueue = DispatchQueue(label: "dev.ayasakura.komorebi.audio-vad", qos: .userInitiated)
    private var partialTaskInFlight = false
    private var refineQueue: [[Float]] = []
    private var refineTaskRunning = false

    private final class WeakPipelineRef: @unchecked Sendable {
        weak var value: AudioPipelineManager?
        init(_ value: AudioPipelineManager) { self.value = value }
    }

    private nonisolated static let darwinObserverLock = NSLock()
    private nonisolated(unsafe) static var activeDarwinObservers: [UInt: WeakPipelineRef] = [:]

    private var ipcSequence: UInt64 = 0
    private var ipcCommitID: UInt64 = 0
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
        publishIPCState(committedText: nil, isDraftReplacement: false)
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

    public func loadAllEnginesIfNeeded() async {
        guard !isLoadingModels else { return }
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
        let loadStart = CFAbsoluteTimeGetCurrent()

        do {
            let vadInstance = SileroTurnsVAD(modelURL: sileroURL, numThreads: 1)
            let asrInstance = try SenseVoiceASR(modelURL: svModel, tokensURL: svTokens, numThreads: 2)
            self.vad = vadInstance
            self.asr = asrInstance

            // Metal is not loaded by default: the GPU heats the phone. The direct-ANE engine is the default backend
            // in both foreground and background (first launch re-quantises weights for the ANE and caches them).
            try await refiner.loadGGUF(url: ggufURL, preferMetal: false, maxContext: 1024)
            var aneFailure: String?
            if !isRunningInSimulator {
                let cacheDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                    .appendingPathComponent("komorebi-ane", isDirectory: true)
                do {
                    try await refiner.loadANEDirect(cacheDirectory: cacheDir, drafterDirectory: modelManager.gemma4DrafterDirectory,
                                                    progress: { msg in Task { @MainActor in self.statusBanner = "準備 ANE：\(msg)" } })
                } catch {
                    aneFailure = error.localizedDescription
                }
            }
            if await refiner.aneDirect == nil {
                if #available(iOS 18.0, *), !isRunningInSimulator, let aneDir = modelManager.gemma4CoreMLDirectory {
                    try? await refiner.loadANECoreML(directory: aneDir)
                }
                if await !refiner.isANEAvailable { await refiner.ensureMetalLoaded() }
            }

            isEngineLoaded = await refiner.isLoaded
            let hasMetal = await refiner.metalEngine != nil
            let hasANE = await refiner.isANEAvailable
            if let desc = await refiner.aneDirectDescription {
                statusBanner = "就緒：\(desc)"
            } else if hasANE {
                statusBanner = "就緒：CoreML ANE" + (aneFailure.map { "（ANE 直跑失敗：\($0)）" } ?? "")
            } else if hasMetal {
                statusBanner = "就緒：Metal GPU / 背景 CPU" + (aneFailure.map { "（ANE 失敗：\($0)）" } ?? "")
            } else {
                statusBanner = "就緒：CPU Q4_0"
            }
        } catch {
            phase = .error
            statusBanner = "模型載入失敗：\(error.localizedDescription)"
        }
        isLoadingModels = false
        NSLog("KOMOREBI_STATUS engines loaded in %.1f s: %@", CFAbsoluteTimeGetCurrent() - loadStart, statusBanner)
        publishIPCState(committedText: nil, isDraftReplacement: false)
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

    public func startListening() async {
        if !isEngineLoaded {
            await loadAllEnginesIfNeeded()
        }
        guard isEngineLoaded, vad != nil, asr != nil else { return }
        if isListening { return }

        let granted = await requestMicPermission()
        guard granted else {
            phase = .error
            statusBanner = "請至系統設定開啟麥克風權限。"
            publishIPCState(committedText: nil, isDraftReplacement: false)
            return
        }

        do {
            try startAudioCaptureEngine(resetVAD: true)
            self.isListening = true
            self.phase = .listening
            let nsTag = hardwareVoiceProcessingActive ? "iOS 降噪開啟" : (noiseSuppressionEnabled ? "系統語音模式" : "原始收音")
            self.statusBanner = "聆聽中（\(nsTag)，可切至任意 App 使用鍵盤）"
            refreshSystemMicMode()
            publishIPCState(committedText: nil, isDraftReplacement: false)
        } catch {
            phase = .error
            statusBanner = "啟動麥克風失敗：\(error.localizedDescription)"
            publishIPCState(committedText: nil, isDraftReplacement: false)
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
        guard isListening else { return }
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        audioConverter = nil
        do {
            try startAudioCaptureEngine(resetVAD: false)
            let nsTag = hardwareVoiceProcessingActive ? "iOS 降噪開啟" : (noiseSuppressionEnabled ? "系統語音模式" : "原始收音")
            self.statusBanner = "聆聽中（\(nsTag)）"
            refreshSystemMicMode()
            publishIPCState(committedText: nil, isDraftReplacement: false)
        } catch {
            isListening = false
            hardwareVoiceProcessingActive = false
            phase = .error
            statusBanner = "切換降噪重啟麥克風失敗：\(error.localizedDescription)"
            publishIPCState(committedText: nil, isDraftReplacement: false)
        }
    }

    public func stopListening() {
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        audioConverter = nil
        isListening = false
        hardwareVoiceProcessingActive = false
        micLevel = 0

        if let vad {
            processingQueue.async { [weak self] in
                let flushEvents = vad.flush()
                self?.dispatchVADEvents(flushEvents)
            }
        }

        phase = refineTaskRunning ? .refining : .idle
        statusBanner = "已暫停聆聽"
        publishIPCState(committedText: nil, isDraftReplacement: false)
    }

    public func flushCurrentUtterance() {
        guard let vad else { return }
        processingQueue.async { [weak self] in
            let flushEvents = vad.flush()
            self?.dispatchVADEvents(flushEvents)
        }
    }

    public func resetDraft() {
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
        let isBG = UIApplication.shared.applicationState == .background
        do {
            let res = try await refiner.selectionEdit(
                before: before,
                selected: selected,
                after: after,
                raw: spokenInstruction,
                inBackground: isBG,
                preference: backendPreference
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
                    Task.detached(priority: .userInitiated) {
                        let res = asr.transcribe(samples: pcm, isPartial: true)
                        await MainActor.run {
                            self.partialTaskInFlight = false
                            if self.isListening && !res.text.isEmpty {
                                self.preeditText = res.text
                                self.telemetry.asrLatencyMs = res.latencyMs
                                self.publishIPCState(committedText: nil, isDraftReplacement: false)
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

    private func enqueueFinalizedSegment(_ pcm: [Float]) {
        refineQueue.append(pcm)
        if !refineTaskRunning {
            Task { await drainRefineQueue() }
        }
    }

    private func drainRefineQueue() async {
        refineTaskRunning = true
        defer { refineTaskRunning = false }

        while !refineQueue.isEmpty {
            let pcm = refineQueue.removeFirst()
            guard let asr else { continue }

            phase = .transcribing
            let transcript = await Task.detached(priority: .userInitiated) {
                asr.transcribe(samples: pcm, isPartial: false)
            }.value

            telemetry.asrLatencyMs = transcript.latencyMs
            let raw = transcript.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !raw.isEmpty else {
                preeditText = ""
                phase = isListening ? .listening : .idle
                publishIPCState(committedText: nil, isDraftReplacement: false)
                continue
            }

            phase = .refining
            preeditText = raw
            statusBanner = "Gemma 4 E2B 整理文字中…"
            publishIPCState(committedText: nil, isDraftReplacement: false)

            let isBG = UIApplication.shared.applicationState == .background
            do {
                if mode == .draft {
                    applyKeyboardDraftEdit()
                    let prev = editorText
                    let version = draftEditVersion
                    let res = try await refiner.draftMerge(
                        draft: prev,
                        raw: raw,
                        inBackground: isBG,
                        preference: backendPreference
                    )
                    applyKeyboardDraftEdit()
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
                    preeditText = ""
                    markFreshDelta(previous: current, updated: updated)
                    updateTelemetry(res.stats)
                    let commitID = publishIPCState(committedText: updated, isDraftReplacement: true)
                    recentDraftCommits.append((commitID, raw))
                    if recentDraftCommits.count > 16 { recentDraftCommits.removeFirst(recentDraftCommits.count - 16) }
                } else {
                    let res = try await refiner.refineSentence(
                        raw: raw,
                        context: lastContextTail,
                        inBackground: isBG,
                        preference: backendPreference
                    )
                    let prev = editorText
                    historyStack.append(prev)
                    let appended = prev.isEmpty ? res.text : res.text
                    editorText = prev + appended
                    preeditText = ""
                    freshRange = NSRange(location: prev.count, length: appended.count)
                    lastContextTail = String(editorText.suffix(120))
                    updateTelemetry(res.stats)
                    publishIPCState(committedText: appended, isDraftReplacement: false)
                }
                statusBanner = isListening ? "聆聽中" : "已完成整理"
            } catch {
                // Safe degraded fallback to SenseVoice + OpenCC raw text
                let prev = editorText
                historyStack.append(prev)
                editorText = prev + raw
                preeditText = ""
                statusBanner = "整理降級為辨識原文：\(error.localizedDescription)"
                publishIPCState(committedText: raw, isDraftReplacement: false)
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
        if committedText != nil {
            ipcCommitID &+= 1
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
            partialText: preeditText,
            commitID: ipcCommitID,
            committedText: committedText ?? "",
            isDraftReplacement: isDraftReplacement,
            noiseSuppressionActive: noiseSuppressionEnabled,
            micIsolationLabel: systemMicModeLabel,
            errorMessage: phase == .error ? statusBanner : nil,
            telemetry: telemetry,
            draftEditAck: lastDraftEditID,
            hostSessionID: hostSessionID
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
            if UIApplication.shared.applicationState == .background {
                pendingStartListeningOnForeground = true
            } else {
                Task { await startListening() }
            }
        case .stopListening:
            stopListening()
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
        let isBG = UIApplication.shared.applicationState == .background
        if let res = try? await refiner.refineSentence(raw: raw, inBackground: isBG, preference: backendPreference) {
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
