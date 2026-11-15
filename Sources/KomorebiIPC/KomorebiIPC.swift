import Foundation

/// Shared constants and lightweight IPC structs linking Host App (`KomorebiVoice`)
/// and Keyboard Extension (`KomorebiKeyboard`).
/// Kept free of heavy frameworks so the Keyboard Extension stays well under iOS Jetsam limits.
public enum KomorebiAppConfig {
    public static let appGroupID = "group.dev.ayasakura.komorebi"
    public static let urlScheme = "komorebi"
    public static let startListeningURL = URL(string: "komorebi://listen")!
    public static let openAppURL = URL(string: "komorebi://open")!

    public static let stateFileName = "komorebi_ipc_state.json"
    public static let commandFileName = "komorebi_ipc_command.json"

    public static let draftEditFileName = "komorebi_ipc_draft_edit.json"

    public static let stateChangedNotification = "dev.ayasakura.komorebi.ipc.state"
    public static let commandNotification = "dev.ayasakura.komorebi.ipc.command"
    public static let draftEditNotification = "dev.ayasakura.komorebi.ipc.draftedit"

    public static func containerURL() -> URL {
        if let url = FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: appGroupID) {
            return url
        }
        // Fallback for unsigned Simulator / macOS unit-test runs.
        return FileManager.default.temporaryDirectory.appendingPathComponent("komorebi-ipc", isDirectory: true)
    }

    public static func modelsDirectoryURL() -> URL {
        let fm = FileManager.default
        if let appGroup = fm.containerURL(forSecurityApplicationGroupIdentifier: appGroupID) {
            return appGroup.appendingPathComponent("Models", isDirectory: true)
        }
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? fm.temporaryDirectory
        return docs.appendingPathComponent("Models", isDirectory: true)
    }
}

public enum DictationPhase: String, Codable, Sendable {
    case idle
    case listening
    case transcribing
    case refining
    case error
}

public struct KomorebiTelemetry: Codable, Equatable, Sendable {
    public var backendUsed: String
    public var prefillTokens: Int
    public var prefillTPS: Double
    public var decodeTokens: Int
    public var decodeTPS: Double
    public var asrLatencyMs: Double

    public init(
        backendUsed: String = "—",
        prefillTokens: Int = 0,
        prefillTPS: Double = 0,
        decodeTokens: Int = 0,
        decodeTPS: Double = 0,
        asrLatencyMs: Double = 0
    ) {
        self.backendUsed = backendUsed
        self.prefillTokens = prefillTokens
        self.prefillTPS = prefillTPS
        self.decodeTokens = decodeTokens
        self.decodeTPS = decodeTPS
        self.asrLatencyMs = asrLatencyMs
    }
}

public struct KomorebiIPCState: Codable, Equatable, Sendable {
    public var sequence: UInt64
    public var updatedAt: TimeInterval
    public var hostAppAlive: Bool
    public var hostInForeground: Bool
    public var audioSessionActive: Bool
    public var modelsReady: Bool
    public var phase: DictationPhase
    public var micLevel: Float
    public var partialText: String
    /// Monotonically incremented whenever a refined utterance / commit is appended.
    public var commitID: UInt64
    public var committedText: String
    /// Optional replacement for existing inserted text in Draft mode.
    public var isDraftReplacement: Bool
    public var noiseSuppressionActive: Bool
    public var micIsolationLabel: String
    public var errorMessage: String?
    public var telemetry: KomorebiTelemetry
    /// Id of the last `KomorebiDraftEdit` the host has folded into its draft. The keyboard applies a draft commit only
    /// once its own latest edit is acknowledged, so a draft computed before the edit never brings deleted text back.
    public var draftEditAck: UInt64
    /// Random per host launch; a change means the host restarted with an empty draft.
    public var hostSessionID: UInt64
    /// The host's microphone session is running (listening, or on standby). While it runs, the keyboard can start
    /// listening through IPC without opening the app (iOS only lets an app start recording in the foreground).
    public var micLive: Bool
    /// When the standby microphone session ends if listening is not started again (0 = not on standby).
    public var micStandbyUntil: TimeInterval
    /// The host is still loading the speech and Gemma engines.
    public var enginesLoading: Bool
    /// Words recognised after the last live clean-up of the sentence being spoken (shown dimmed after `partialText`).
    public var partialTail: String

    public init(
        sequence: UInt64 = 0,
        updatedAt: TimeInterval = Date().timeIntervalSince1970,
        hostAppAlive: Bool = false,
        hostInForeground: Bool = false,
        audioSessionActive: Bool = false,
        modelsReady: Bool = false,
        phase: DictationPhase = .idle,
        micLevel: Float = 0,
        partialText: String = "",
        commitID: UInt64 = 0,
        committedText: String = "",
        isDraftReplacement: Bool = false,
        noiseSuppressionActive: Bool = true,
        micIsolationLabel: String = "標準",
        errorMessage: String? = nil,
        telemetry: KomorebiTelemetry = KomorebiTelemetry(),
        draftEditAck: UInt64 = 0,
        hostSessionID: UInt64 = 0,
        micLive: Bool = false,
        micStandbyUntil: TimeInterval = 0,
        enginesLoading: Bool = false,
        partialTail: String = ""
    ) {
        self.sequence = sequence
        self.updatedAt = updatedAt
        self.hostAppAlive = hostAppAlive
        self.hostInForeground = hostInForeground
        self.audioSessionActive = audioSessionActive
        self.modelsReady = modelsReady
        self.phase = phase
        self.micLevel = micLevel
        self.partialText = partialText
        self.commitID = commitID
        self.committedText = committedText
        self.isDraftReplacement = isDraftReplacement
        self.noiseSuppressionActive = noiseSuppressionActive
        self.micIsolationLabel = micIsolationLabel
        self.errorMessage = errorMessage
        self.telemetry = telemetry
        self.draftEditAck = draftEditAck
        self.hostSessionID = hostSessionID
        self.micLive = micLive
        self.micStandbyUntil = micStandbyUntil
        self.enginesLoading = enginesLoading
        self.partialTail = partialTail
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sequence = try c.decodeIfPresent(UInt64.self, forKey: .sequence) ?? 0
        updatedAt = try c.decodeIfPresent(TimeInterval.self, forKey: .updatedAt) ?? Date().timeIntervalSince1970
        hostAppAlive = try c.decodeIfPresent(Bool.self, forKey: .hostAppAlive) ?? false
        hostInForeground = try c.decodeIfPresent(Bool.self, forKey: .hostInForeground) ?? false
        audioSessionActive = try c.decodeIfPresent(Bool.self, forKey: .audioSessionActive) ?? false
        modelsReady = try c.decodeIfPresent(Bool.self, forKey: .modelsReady) ?? false
        phase = try c.decodeIfPresent(DictationPhase.self, forKey: .phase) ?? .idle
        micLevel = try c.decodeIfPresent(Float.self, forKey: .micLevel) ?? 0
        partialText = try c.decodeIfPresent(String.self, forKey: .partialText) ?? ""
        commitID = try c.decodeIfPresent(UInt64.self, forKey: .commitID) ?? 0
        committedText = try c.decodeIfPresent(String.self, forKey: .committedText) ?? ""
        isDraftReplacement = try c.decodeIfPresent(Bool.self, forKey: .isDraftReplacement) ?? false
        noiseSuppressionActive = try c.decodeIfPresent(Bool.self, forKey: .noiseSuppressionActive) ?? true
        micIsolationLabel = try c.decodeIfPresent(String.self, forKey: .micIsolationLabel) ?? "標準"
        errorMessage = try c.decodeIfPresent(String.self, forKey: .errorMessage)
        telemetry = try c.decodeIfPresent(KomorebiTelemetry.self, forKey: .telemetry) ?? KomorebiTelemetry()
        draftEditAck = try c.decodeIfPresent(UInt64.self, forKey: .draftEditAck) ?? 0
        hostSessionID = try c.decodeIfPresent(UInt64.self, forKey: .hostSessionID) ?? 0
        micLive = try c.decodeIfPresent(Bool.self, forKey: .micLive) ?? false
        micStandbyUntil = try c.decodeIfPresent(TimeInterval.self, forKey: .micStandbyUntil) ?? 0
        enginesLoading = try c.decodeIfPresent(Bool.self, forKey: .enginesLoading) ?? false
        partialTail = try c.decodeIfPresent(String.self, forKey: .partialTail) ?? ""
    }
}

/// The keyboard's edit of the draft that is shown in the host text field (backspace, the field cleared after a
/// message was sent, the cursor moved away, 定稿). `text` is the draft as it now is in the field; `baseCommitID` is
/// the last draft commit the keyboard had applied when it made the edit, so the host can re-append sentences it
/// merged after that point (the field's text wins and those sentences are appended as heard).
public struct KomorebiDraftEdit: Codable, Equatable, Sendable {
    public var id: UInt64
    public var baseCommitID: UInt64
    public var text: String
    public var timestamp: TimeInterval

    public init(id: UInt64, baseCommitID: UInt64, text: String, timestamp: TimeInterval = Date().timeIntervalSince1970) {
        self.id = id
        self.baseCommitID = baseCommitID
        self.text = text
        self.timestamp = timestamp
    }
}

public struct KomorebiIPCCommand: Codable, Equatable, Sendable {
    public enum Action: String, Codable, Sendable {
        case startListening
        case stopListening
        /// Turns the microphone off now instead of leaving it on standby.
        case endMicSession
        case flushUtterance
        case resetDraft
        case toggleNoiseSuppression
        case refineManualText
        case heartbeat
    }

    public var id: UInt64
    public var timestamp: TimeInterval
    public var action: Action
    public var textPayload: String?
    public var useDraftMode: Bool

    public init(
        id: UInt64,
        timestamp: TimeInterval = Date().timeIntervalSince1970,
        action: Action,
        textPayload: String? = nil,
        useDraftMode: Bool = false
    ) {
        self.id = id
        self.timestamp = timestamp
        self.action = action
        self.textPayload = textPayload
        self.useDraftMode = useDraftMode
    }
}

/// Thread-safe App Group file + Darwin notification store for Host App <-> Keyboard Extension.
public final class KomorebiIPCStore: @unchecked Sendable {
    public static let shared = KomorebiIPCStore()

    private let baseURL: URL
    private let stateURL: URL
    private let commandURL: URL
    private let draftEditURL: URL
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    public init(baseURL: URL = KomorebiAppConfig.containerURL()) {
        self.baseURL = baseURL
        self.stateURL = baseURL.appendingPathComponent(KomorebiAppConfig.stateFileName)
        self.commandURL = baseURL.appendingPathComponent(KomorebiAppConfig.commandFileName)
        self.draftEditURL = baseURL.appendingPathComponent(KomorebiAppConfig.draftEditFileName)
        try? FileManager.default.createDirectory(at: baseURL, withIntermediateDirectories: true)
    }

    public func writeState(_ state: KomorebiIPCState) {
        guard let data = try? encoder.encode(state) else { return }
        try? data.write(to: stateURL, options: [.atomic])
        postDarwinNotification(KomorebiAppConfig.stateChangedNotification)
    }

    public func readState() -> KomorebiIPCState {
        guard let data = try? Data(contentsOf: stateURL),
              let state = try? decoder.decode(KomorebiIPCState.self, from: data) else {
            return KomorebiIPCState()
        }
        return state
    }

    public func sendCommand(_ command: KomorebiIPCCommand) {
        guard let data = try? encoder.encode(command) else { return }
        try? data.write(to: commandURL, options: [.atomic])
        postDarwinNotification(KomorebiAppConfig.commandNotification)
    }

    public func readCommand() -> KomorebiIPCCommand? {
        guard let data = try? Data(contentsOf: commandURL),
              let cmd = try? decoder.decode(KomorebiIPCCommand.self, from: data) else {
            return nil
        }
        return cmd
    }

    /// Draft edits use their own file, so a quick backspace never overwrites (or is overwritten by) a command.
    public func writeDraftEdit(_ edit: KomorebiDraftEdit) {
        guard let data = try? encoder.encode(edit) else { return }
        try? data.write(to: draftEditURL, options: [.atomic])
        postDarwinNotification(KomorebiAppConfig.draftEditNotification)
    }

    public func readDraftEdit() -> KomorebiDraftEdit? {
        guard let data = try? Data(contentsOf: draftEditURL),
              let edit = try? decoder.decode(KomorebiDraftEdit.self, from: data) else {
            return nil
        }
        return edit
    }

    public func postDarwinNotification(_ name: String) {
        let center = CFNotificationCenterGetDarwinNotifyCenter()
        CFNotificationCenterPostNotification(center, CFNotificationName(name as CFString), nil, nil, true)
    }
}
