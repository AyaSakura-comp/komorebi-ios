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

                Divider().overlay(KomorebiTheme.border(scheme))

                HStack {
                    Text("停止後麥克風待命")
                        .font(.footnote.weight(.medium))
                        .foregroundStyle(KomorebiTheme.ink(scheme))
                    Spacer()
                    Picker("停止後麥克風待命", selection: $pipeline.micStandbyMinutes) {
                        ForEach(AudioPipelineManager.micStandbyChoices, id: \.self) { minutes in
                            Text(minutes == 0 ? "不待命" : "\(minutes) 分鐘").tag(minutes)
                        }
                    }
                    .pickerStyle(.menu)
                    .tint(KomorebiTheme.sage(scheme))
                }

                Text("iOS 只允許 App 在前景開始錄音，所以鍵盤第一次點「語音輸入」會切到 Komorebi 開麥克風，並試著自動切回原本的 App。停止聆聽後麥克風會待命這段時間，期間在鍵盤再點「語音輸入」不會跳轉；待命時狀態列會顯示麥克風使用中，但聲音不會被處理。")
                    .font(.caption)
                    .foregroundStyle(KomorebiTheme.inkMuted(scheme))

                if pipeline.micStandby {
                    Button("現在關閉麥克風") {
                        pipeline.endMicSession()
                    }
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(KomorebiTheme.subtle(scheme))
                    .foregroundStyle(KomorebiTheme.persimmon(scheme))
                    .clipShape(Capsule())
                }
            }
        }
    }

    private var engineSection: some View {
        cardContainer(title: "推論引擎 (Gemma 4 E2B)") {
            VStack(alignment: .leading, spacing: 8) {
                Text(pipeline.statusBanner)
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(KomorebiTheme.ink(scheme))
                Toggle("邊說邊整理", isOn: $pipeline.livePreviewEnabled)
                    .font(.footnote.weight(.medium))
                    .tint(KomorebiTheme.sage(scheme))
                Text("說話時每隔約半秒就用 Gemma 先整理目前這一句，整理好就直接接在輸入框的草稿後面（鍵盤預覽列只以淡色顯示還沒整理的字）；停頓後整句合併的結果會取代它。說話時 ANE 會持續工作。")
                    .font(.caption)
                    .foregroundStyle(KomorebiTheme.inkMuted(scheme))
                Toggle("句子講完用 Qwen3-ASR 重新辨識", isOn: $pipeline.qwen3FinalEnabled)
                    .font(.footnote.weight(.medium))
                    .tint(KomorebiTheme.sage(scheme))
                Text(pipeline.finalASRStatus)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(KomorebiTheme.ink(scheme))
                Text("跟桌面版 Komorebi 一樣：說話時的即時字幕用 SenseVoice（快），停頓後整句交給 Qwen3-ASR-1.7B 在 ANE 上重新辨識，台灣口音中英夾雜的錯字約少一半，每句多約 1 秒。")
                    .font(.caption)
                    .foregroundStyle(KomorebiTheme.inkMuted(scheme))
                Text("Gemma 4 E2B 只跑在 Apple Neural Engine 上（自製 MIL 引擎：LUT4 權重、小字典 LM head、MTP 推測解碼），前景與背景都一樣快，不用 GPU、不燙。第一次啟動會把權重轉成 ANE 格式並編譯（約 1 分鐘），之後幾秒內載入。")
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
        cardContainer(title: "ANE Prefill & Decode 實測") {
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
        do {
            let res = try await pipeline.refiner.refineSentence(raw: sample)
            benchResults.append("\(res.stats.backend): prefill=\(res.stats.prefillTokens) tok @ \(String(format: "%.1f", res.stats.prefillTPS)) t/s | decode=\(res.stats.decodeTokens) tok @ \(String(format: "%.1f", res.stats.decodeTPS)) t/s")
            benchOutput = res.text
        } catch {
            benchResults.append("ANE: \(error.localizedDescription)")
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
