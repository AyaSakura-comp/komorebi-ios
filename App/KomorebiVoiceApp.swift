import SwiftUI
import BackgroundTasks
import KomorebiIPC

@main
struct KomorebiVoiceApp: App {
    @StateObject private var modelManager: ModelManager
    @StateObject private var pipeline: AudioPipelineManager
    @Environment(\.scenePhase) private var scenePhase

    init() {
        // Diagnostic: does this device allow GPU work in background (BGContinuedProcessingTask .gpu)?
        if #available(iOS 26.0, *) {
            NSLog("KOMOREBI_DIAG background_gpu_supported=%@", BGTaskScheduler.supportedResources.contains(.gpu) ? "YES" : "NO")
        }
        ANEDirectBench.runIfRequested()
        let mm = ModelManager()
        let pipe = AudioPipelineManager(modelManager: mm)
        _modelManager = StateObject(wrappedValue: mm)
        _pipeline = StateObject(wrappedValue: pipe)
    }

    var body: some Scene {
        WindowGroup {
            ContentView(modelManager: modelManager, pipeline: pipeline)
                .onOpenURL { url in
                    handleIncomingURL(url)
                }
                .onChange(of: scenePhase) { _, newPhase in
                    if newPhase == .active {
                        modelManager.scanInstalledModels()
                        pipeline.refreshSystemMicMode()
                    }
                    pipeline.setHostForeground(newPhase == .active)
                }
                .task {
                    modelManager.scanInstalledModels()
                    if modelManager.coreModelsReady && !ANEDirectBench.isActive {
                        await pipeline.loadAllEnginesIfNeeded()
                    }
                }
        }
    }

    private func handleIncomingURL(_ url: URL) {
        guard url.scheme == KomorebiAppConfig.urlScheme else { return }
        if url.host == "listen" {
            Task {
                await pipeline.startListening()
            }
        } else if url.host == "stop" {
            pipeline.stopListening()
        }
    }
}
