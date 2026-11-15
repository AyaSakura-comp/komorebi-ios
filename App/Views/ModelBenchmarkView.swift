import KomorebiCore
import KomorebiIPC
import SwiftUI

struct ModelBenchmarkView: View {
    @ObservedObject var modelManager: ModelManager
    @ObservedObject var pipeline: AudioPipelineManager
    @Environment(\.colorScheme) private var scheme

    @State private var benchRunning = false
    @State private var benchOutput: String = ""
    @State private var benchResults: [String] = []

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    audioNoiseSuppressionSection
                    engineSection
                    modelsSection
                    lanTransferSection
                    benchmarkSection
                }
                .padding(16)
            }
            .background(KomorebiTheme.paper(scheme).ignoresSafeArea())
            .navigationTitle("模型與推論引擎")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var audioNoiseSuppressionSection: some View {
        cardContainer(title: "iOS 內建降噪與語音隔離") {
            VStack(alignment: .leading, spacing: 10) {
                Toggle("Apple VoiceProcessingIO 降噪 (推薦)", isOn: $pipeline.noiseSuppressionEnabled)
                    .font(.footnote.weight(.medium))
                    .tint(KomorebiTheme.sage(scheme))

                Text("透過 AVAudioSession .voiceChat 搭配 AVAudioInputNode.setVoiceProcessingEnabled(true) 與 AGC，自動過濾環境雜訊與回聲，並設為最低混音衰減 (.min) 以免壓低音樂。")
                    .font(.caption)
                    .foregroundStyle(KomorebiTheme.inkMuted(scheme))

                Divider().overlay(KomorebiTheme.border(scheme))

                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("系統控制中心麥克風模式")
                            .font(.caption.weight(.medium))
                            .foregroundStyle(KomorebiTheme.ink(scheme))
                        Text("目前：\(pipeline.systemMicModeLabel)")
                            .font(.caption2)
                            .foregroundStyle(KomorebiTheme.sage(scheme))
                    }
                    Spacer()
                    Button("切換為語音隔離") {
                        pipeline.openSystemMicrophoneModePicker()
                    }
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(KomorebiTheme.subtle(scheme))
                    .foregroundStyle(KomorebiTheme.sage(scheme))
                    .clipShape(Capsule())
                }
            }
        }
    }

    private var engineSection: some View {
        cardContainer(title: "推論引擎路由 (Gemma 4 E2B)") {
            VStack(alignment: .leading, spacing: 10) {
                Picker("推論引擎", selection: $pipeline.backendPreference) {
                    ForEach(InferenceBackendPreference.allCases, id: \.self) { pref in
                        Text(pref.displayName).tag(pref)
                    }
                }
                .pickerStyle(.menu)

                Text("• 前景主 App：使用純 Metal Q4_0 Zero-Copy Kernel（GPU 加速）。\n• 背景（在其他 App 用語音鍵盤時）：iOS 禁止背景提交 MTLCommandBuffer，自動切換至 ANE CoreML (.cpuAndNeuralEngine) 或 Accelerate CPU。")
                    .font(.caption)
                    .foregroundStyle(KomorebiTheme.inkMuted(scheme))
            }
        }
    }

    private var modelsSection: some View {
        cardContainer(title: "離線模型資產 (App Group)") {
            VStack(spacing: 12) {
                ForEach(ModelArtifactID.allCases) { artifact in
                    let isReady = modelManager.installed.contains(artifact)
                    HStack(spacing: 10) {
                        Image(systemName: isReady ? "checkmark.circle.fill" : "circle.dashed")
                            .foregroundStyle(isReady ? KomorebiTheme.sage(scheme) : KomorebiTheme.inkMuted(scheme))
                        VStack(alignment: .leading, spacing: 2) {
                            Text(artifact.title)
                                .font(.footnote.weight(.medium))
                                .foregroundStyle(KomorebiTheme.ink(scheme))
                            Text(artifact.sizeDescription)
                                .font(.caption2)
                                .foregroundStyle(KomorebiTheme.inkMuted(scheme))
                        }
                        Spacer()
                        if !isReady {
                            Button("下載") {
                                Task { await modelManager.downloadArtifact(artifact) }
                            }
                            .font(.caption.weight(.semibold))
                            .padding(.horizontal, 10)
                            .padding(.vertical, 5)
                            .background(KomorebiTheme.subtle(scheme))
                            .clipShape(Capsule())
                            .disabled(modelManager.activeDownload != nil)
                        }
                    }
                }

                if modelManager.activeDownload != nil {
                    VStack(alignment: .leading, spacing: 4) {
                        ProgressView(value: modelManager.downloadProgress)
                            .tint(KomorebiTheme.sage(scheme))
                        Text(modelManager.statusMessage)
                            .font(.caption2)
                            .foregroundStyle(KomorebiTheme.inkMuted(scheme))
                    }
                } else if !modelManager.statusMessage.isEmpty {
                    Text(modelManager.statusMessage)
                        .font(.caption2)
                        .foregroundStyle(KomorebiTheme.inkMuted(scheme))
                        .frame(maxWidth: .infinity, alignment: .leading)
                }

                HStack(spacing: 10) {
                    Button {
                        Task {
                            await modelManager.downloadMissingCoreModels()
                            await pipeline.loadAllEnginesIfNeeded()
                        }
                    } label: {
                        Text(modelManager.coreModelsReady ? "重新檢查並載入" : "一鍵下載核心模型")
                            .font(.footnote.weight(.semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 9)
                            .background(KomorebiTheme.sage(scheme))
                            .foregroundStyle(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                    }
                    .disabled(modelManager.activeDownload != nil)
                }
            }
        }
    }

    private var lanTransferSection: some View {
        cardContainer(title: "區網 Mac 伴隨伺服器高速傳檔") {
            VStack(alignment: .leading, spacing: 8) {
                TextField("http://192.168.1.x:8765", text: $modelManager.lanServerBaseURL)
                    .font(.footnote.monospaced())
                    .textFieldStyle(.roundedBorder)
                    .autocapitalization(.none)
                    .disableAutocorrection(true)

                Text("在 Mac 執行 `bash scripts/serve-models.sh`，填入顯示的 URL 後按上方下載，免重抓 HuggingFace。")
                    .font(.caption2)
                    .foregroundStyle(KomorebiTheme.inkMuted(scheme))
            }
        }
    }

    private var benchmarkSection: some View {
        cardContainer(title: "實機 Metal / ANE / CPU Prefill & Decode 實測") {
            VStack(alignment: .leading, spacing: 10) {
                Button {
                    Task { await runOnDeviceBenchmarks() }
                } label: {
                    HStack {
                        if benchRunning {
                            ProgressView().tint(.white)
                        }
                        Text(benchRunning ? "實測推論中…" : "執行實機 Benchmark")
                    }
                    .font(.footnote.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 9)
                    .background(KomorebiTheme.persimmon(scheme))
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                }
                .disabled(benchRunning || !modelManager.coreModelsReady)

                ForEach(benchResults, id: \.self) { line in
                    Text(line)
                        .font(.caption.monospaced())
                        .foregroundStyle(KomorebiTheme.ink(scheme))
                }

                if !benchOutput.isEmpty {
                    Text("整理結果：\(benchOutput)")
                        .font(.caption)
                        .foregroundStyle(KomorebiTheme.sage(scheme))
                }
            }
        }
    }

    private func runOnDeviceBenchmarks() async {
        benchRunning = true
        benchResults = []
        benchOutput = ""
        defer { benchRunning = false }

        if !pipeline.isEngineLoaded {
            await pipeline.loadAllEnginesIfNeeded()
        }

        let sample = "呃那個我們明天下午三點開會，喔不對不對，改成四點半好不好"
#if targetEnvironment(simulator)
        benchResults.append("ane: 模擬器無實體 ANE NPU；以下實測 Metal GPU vs CPU Q4_0：")
        let prefs: [InferenceBackendPreference] = [.metal, .cpu]
#else
        let prefs: [InferenceBackendPreference] = [.metal, .ane, .cpu]
#endif
        for pref in prefs {
            do {
                let res = try await pipeline.refiner.refineSentence(raw: sample, preference: pref)
                let line = "\(res.stats.backend): prefill=\(res.stats.prefillTokens) tok @ \(String(format: "%.1f", res.stats.prefillTPS)) t/s | decode=\(res.stats.decodeTokens) tok @ \(String(format: "%.1f", res.stats.decodeTPS)) t/s"
                benchResults.append(line)
                if benchOutput.isEmpty {
                    benchOutput = res.text
                }
            } catch {
                benchResults.append("\(pref.rawValue): \(error.localizedDescription)")
            }
        }
    }

    @ViewBuilder
    private func cardContainer<Content: View>(title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(KomorebiTheme.ink(scheme))
            content()
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(KomorebiTheme.card(scheme))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(KomorebiTheme.border(scheme), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }
}
