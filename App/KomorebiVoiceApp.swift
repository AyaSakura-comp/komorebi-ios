import SwiftUI
import KomorebiIPC

@main
struct KomorebiVoiceApp: App {
    @StateObject private var modelManager: ModelManager
    @StateObject private var pipeline: AudioPipelineManager
    @Environment(\.scenePhase) private var scenePhase

    init() {
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
                    if NavProbe.isActive {
                        await NavProbe.run(pipeline: pipeline)
                        return
                    }
                    modelManager.scanInstalledModels()
                    if modelManager.coreModelsReady && !ANEDirectBench.isActive {
                        if MicSelfTest.isActive {
                            await MicSelfTest.run(pipeline: pipeline)
                        } else {
                            await pipeline.loadAllEnginesIfNeeded()
                        }
                    }
                }
        }
    }

    private func handleIncomingURL(_ url: URL) {
        guard url.scheme == KomorebiAppConfig.urlScheme else { return }
        if url.host == "listen" {
            // Opened by the keyboard: start the microphone, then go straight back to the app the user was typing in.
            pipeline.diag("keyboard opened komorebi://listen")
            Task {
                await pipeline.startListening(returnToCaller: true)
            }
        } else if url.host == "stop" {
            pipeline.stopListening()
        }
    }
}
