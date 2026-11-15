import CryptoKit
import Foundation
import KomorebiIPC

public enum ModelArtifactID: String, CaseIterable, Identifiable, Sendable {
    case sileroVAD = "silero_vad"
    case senseVoiceInt8 = "sensevoice_int8"
    case gemma4GGUFQ4 = "gemma4_q4_0"
    case gemma4MTPDrafter = "gemma4_mtp_drafter"
    case qwen3ASR = "qwen3_asr_1_7b"

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .sileroVAD: return "Silero VAD v6.2 (CPU)"
        case .senseVoiceInt8: return "SenseVoice-Small int8 + OpenCC s2twp (CPU)"
        case .gemma4GGUFQ4: return "Gemma 4 E2B Q4_0 QAT（首次啟動轉成 ANE 權重）"
        case .gemma4MTPDrafter: return "Gemma 4 E2B MTP drafter（推測解碼加速）"
        case .qwen3ASR: return "Qwen3-ASR-1.7B Q8_0（整句辨識，首次啟動轉成 ANE 權重）"
        }
    }

    public var sizeDescription: String {
        switch self {
        case .sileroVAD: return "2.3 MB"
        case .senseVoiceInt8: return "228 MB"
        case .gemma4GGUFQ4: return "3.12 GiB"
        case .gemma4MTPDrafter: return "150 MB（建議，ANE 解碼約快 1.4–2.5 倍）"
        case .qwen3ASR: return "2.03 GiB（建議，台灣口音中英夾雜錯字約 SenseVoice 的一半；用 Tools/qwen3-asr 轉檔）"
        }
    }

    public var isRequiredForCore: Bool {
        self != .gemma4MTPDrafter && self != .qwen3ASR
    }
}

@MainActor
public final class ModelManager: NSObject, ObservableObject, URLSessionDownloadDelegate {
    @Published public private(set) var installed: Set<ModelArtifactID> = []
    @Published public private(set) var activeDownload: ModelArtifactID?
    @Published public private(set) var downloadProgress: Double = 0
    @Published public private(set) var statusMessage: String = ""
    @Published public var lanServerBaseURL: String = UserDefaults.standard.string(forKey: "komorebi_lan_server") ?? ""

    public let rootDirectory: URL

    private var continuation: CheckedContinuation<URL, Error>?
    private lazy var session: URLSession = {
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 60
        cfg.timeoutIntervalForResource = 3600
        return URLSession(configuration: cfg, delegate: self, delegateQueue: nil)
    }()

    public override init() {
        self.rootDirectory = KomorebiAppConfig.modelsDirectoryURL()
        super.init()
        try? FileManager.default.createDirectory(at: rootDirectory, withIntermediateDirectories: true)
        scanInstalledModels()
    }

    public var sileroURL: URL? {
        locate([
            rootDirectory.appendingPathComponent("sherpa-models/silero_vad.onnx"),
            rootDirectory.appendingPathComponent("silero_vad.onnx"),
        ])
    }

    public var senseVoiceModelURL: URL? {
        locate([
            rootDirectory.appendingPathComponent("sherpa-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx"),
            rootDirectory.appendingPathComponent("sensevoice/model.int8.onnx"),
        ])
    }

    public var senseVoiceTokensURL: URL? {
        locate([
            rootDirectory.appendingPathComponent("sherpa-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt"),
            rootDirectory.appendingPathComponent("sensevoice/tokens.txt"),
        ])
    }

    /// Directory of `google/gemma-4-E2B-it-assistant` (config.json + model.safetensors).
    public var gemma4DrafterDirectory: URL? {
        guard let weights = locate([rootDirectory.appendingPathComponent("gemma-4-e2b-assistant/model.safetensors")]) else { return nil }
        let dir = weights.deletingLastPathComponent()
        return FileManager.default.fileExists(atPath: dir.appendingPathComponent("config.json").path) ? dir : nil
    }

    public var gemma4GGUFURL: URL? {
        locate([
            rootDirectory.appendingPathComponent("gemma-4-e2b/gemma-4-E2B_q4_0-it.gguf"),
            rootDirectory.appendingPathComponent("gemma-4-E2B_q4_0-it.gguf"),
        ])
    }

    /// Qwen3-ASR-1.7B converted by Tools/qwen3-asr/convert_gguf.py (finished sentences on the ANE).
    public var qwen3ASRGGUFURL: URL? {
        locate([rootDirectory.appendingPathComponent("qwen3-asr/qwen3-asr-1.7b-q8_0.gguf")])
    }

    public var coreModelsReady: Bool {
        sileroURL != nil && senseVoiceModelURL != nil && senseVoiceTokensURL != nil && gemma4GGUFURL != nil
    }

    public func scanInstalledModels() {
        var found = Set<ModelArtifactID>()
        if sileroURL != nil { found.insert(.sileroVAD) }
        if senseVoiceModelURL != nil && senseVoiceTokensURL != nil { found.insert(.senseVoiceInt8) }
        if gemma4GGUFURL != nil { found.insert(.gemma4GGUFQ4) }
        if gemma4DrafterDirectory != nil { found.insert(.gemma4MTPDrafter) }
        if qwen3ASRGGUFURL != nil { found.insert(.qwen3ASR) }
        installed = found
    }

    /// Searches App Group Models, App Documents, App Bundle resources, and Simulator/macOS `~/models`.
    private func locate(_ primaryCandidates: [URL]) -> URL? {
        let fm = FileManager.default
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first
        let homeModels = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("models", isDirectory: true)

        for candidate in primaryCandidates {
            if fm.fileExists(atPath: candidate.path) { return candidate }
            let rel = candidate.path.replacingOccurrences(of: rootDirectory.path + "/", with: "")
            if let docs {
                let dModels = docs.appendingPathComponent("Models").appendingPathComponent(rel)
                if fm.fileExists(atPath: dModels.path) { return dModels }
                let d = docs.appendingPathComponent(rel)
                if fm.fileExists(atPath: d.path) { return d }
            }
            if let bun = Bundle.main.resourceURL?.appendingPathComponent("Models/\(rel)"),
               fm.fileExists(atPath: bun.path) {
                return bun
            }
            let hm = homeModels.appendingPathComponent(rel)
            if fm.fileExists(atPath: hm.path) { return hm }
        }
        return nil
    }

    public func downloadMissingCoreModels() async {
        do {
            if sileroURL == nil {
                try await downloadSileroVAD()
            }
            if senseVoiceModelURL == nil || senseVoiceTokensURL == nil {
                try await downloadSenseVoiceInt8()
            }
            if gemma4GGUFURL == nil {
                try await downloadGemma4Q4_0()
            }
            if gemma4DrafterDirectory == nil {
                try? await downloadGemma4Drafter()
            }
            scanInstalledModels()
            statusMessage = "所有核心語音與 Gemma 4 E2B 模型皆已就緒。"
        } catch {
            statusMessage = "下載失敗：\(error.localizedDescription)"
        }
        activeDownload = nil
    }

    public func downloadArtifact(_ id: ModelArtifactID) async {
        do {
            switch id {
            case .sileroVAD: try await downloadSileroVAD()
            case .senseVoiceInt8: try await downloadSenseVoiceInt8()
            case .gemma4GGUFQ4: try await downloadGemma4Q4_0()
            case .gemma4MTPDrafter: try await downloadGemma4Drafter()
            case .qwen3ASR: try await downloadQwen3ASR()
            }
            scanInstalledModels()
            statusMessage = "\(id.title) 下載完成。"
        } catch {
            statusMessage = "\(id.title) 下載失敗：\(error.localizedDescription)"
        }
        activeDownload = nil
    }

    private func resolvedURL(relativePath: String, upstreamURL: String) -> URL {
        let base = lanServerBaseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        if !base.isEmpty {
            let trimmed = base.hasSuffix("/") ? String(base.dropLast()) : base
            if let url = URL(string: "\(trimmed)/\(relativePath)") {
                return url
            }
        }
        return URL(string: upstreamURL)!
    }

    private func downloadSileroVAD() async throws {
        activeDownload = .sileroVAD
        downloadProgress = 0
        statusMessage = "下載 Silero VAD v6.2…"
        let dest = rootDirectory.appendingPathComponent("sherpa-models/silero_vad.onnx")
        let src = resolvedURL(
            relativePath: "sherpa-models/silero_vad.onnx",
            upstreamURL: "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2/src/silero_vad/data/silero_vad.onnx"
        )
        try await fetchFile(from: src, to: dest)
    }

    private func downloadSenseVoiceInt8() async throws {
        activeDownload = .senseVoiceInt8
        downloadProgress = 0
        statusMessage = "下載 SenseVoice-Small (int8 ONNX)…"
        let dir = rootDirectory.appendingPathComponent("sherpa-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17", isDirectory: true)
        let modelDest = dir.appendingPathComponent("model.int8.onnx")
        let tokensDest = dir.appendingPathComponent("tokens.txt")

        // Pull directly from HuggingFace mirror / LAN server (individual int8 files = 228 MB instead of 1 GB tarball!)
        let tokSrc = resolvedURL(
            relativePath: "sherpa-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
            upstreamURL: "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/tokens.txt"
        )
        try await fetchFile(from: tokSrc, to: tokensDest)

        let modelSrc = resolvedURL(
            relativePath: "sherpa-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx",
            upstreamURL: "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/model.int8.onnx"
        )
        try await fetchFile(from: modelSrc, to: modelDest)
    }

    private func downloadGemma4Q4_0() async throws {
        activeDownload = .gemma4GGUFQ4
        downloadProgress = 0
        statusMessage = "下載 Q4_0 Gemma 4 E2B GGUF (3.12 GiB)…"
        let dest = rootDirectory.appendingPathComponent("gemma-4-e2b/gemma-4-E2B_q4_0-it.gguf")
        let src = resolvedURL(
            relativePath: "gemma-4-e2b/gemma-4-E2B_q4_0-it.gguf",
            upstreamURL: "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B_q4_0-it.gguf"
        )
        try await fetchFile(from: src, to: dest)
    }

    /// The converted GGUF is not hosted upstream: it comes from the LAN server (Tools/qwen3-asr, docs/DEPLOYMENT.md).
    private func downloadQwen3ASR() async throws {
        guard !lanServerBaseURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw NSError(domain: "Komorebi", code: 1, userInfo: [NSLocalizedDescriptionKey:
                "Qwen3-ASR 要先在 Mac 上用 Tools/qwen3-asr 轉檔，再設定區網伺服器下載或直接複製到 App（見 docs/DEPLOYMENT.md）"])
        }
        activeDownload = .qwen3ASR
        downloadProgress = 0
        statusMessage = "下載 Qwen3-ASR-1.7B Q8_0 GGUF (2.03 GiB)…"
        let rel = "qwen3-asr/qwen3-asr-1.7b-q8_0.gguf"
        try await fetchFile(from: resolvedURL(relativePath: rel, upstreamURL: "https://invalid.local/\(rel)"),
                            to: rootDirectory.appendingPathComponent(rel))
    }

    private func downloadGemma4Drafter() async throws {
        activeDownload = .gemma4MTPDrafter
        downloadProgress = 0
        statusMessage = "下載 Gemma 4 E2B MTP drafter…"
        let dir = rootDirectory.appendingPathComponent("gemma-4-e2b-assistant", isDirectory: true)
        for (name, progress) in [("config.json", false), ("model.safetensors", true)] {
            let src = resolvedURL(
                relativePath: "gemma-4-e2b-assistant/\(name)",
                upstreamURL: "https://huggingface.co/google/gemma-4-E2B-it-assistant/resolve/main/\(name)"
            )
            try await fetchFile(from: src, to: dir.appendingPathComponent(name), updateProgress: progress)
        }
    }

    private func fetchFile(from url: URL, to destination: URL, updateProgress: Bool = true) async throws {
        let fm = FileManager.default
        try fm.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)

        let tmpURL: URL
        if updateProgress {
            tmpURL = try await withCheckedThrowingContinuation { cont in
                self.continuation = cont
                let task = self.session.downloadTask(with: url)
                task.resume()
            }
        } else {
            let (tempFile, resp) = try await session.download(from: url)
            try validatedHTTPResponse(resp, url: url)
            tmpURL = tempFile
        }

        if fm.fileExists(atPath: destination.path) {
            try fm.removeItem(at: destination)
        }
        try fm.moveItem(at: tmpURL, to: destination)
    }

    private func validatedHTTPResponse(_ response: URLResponse?, url: URL) throws {
        if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            throw NSError(
                domain: "KomorebiModelDownload",
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: "HTTP \(http.statusCode): \(url.lastPathComponent)"]
            )
        }
    }

    // MARK: - URLSessionDownloadDelegate

    nonisolated public func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let p = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        Task { @MainActor in
            self.downloadProgress = p
        }
    }

    nonisolated public func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        if let http = downloadTask.response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            let err = NSError(
                domain: "KomorebiModelDownload",
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: "HTTP \(http.statusCode) 下載失敗"]
            )
            Task { @MainActor in
                self.continuation?.resume(throwing: err)
                self.continuation = nil
            }
            return
        }
        let staged = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        do {
            try FileManager.default.copyItem(at: location, to: staged)
            Task { @MainActor in
                self.continuation?.resume(returning: staged)
                self.continuation = nil
            }
        } catch {
            Task { @MainActor in
                self.continuation?.resume(throwing: error)
                self.continuation = nil
            }
        }
    }

    nonisolated public func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error {
            Task { @MainActor in
                self.continuation?.resume(throwing: error)
                self.continuation = nil
            }
        }
    }
}
