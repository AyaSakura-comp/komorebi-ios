import SwiftUI

struct ContentView: View {
    @ObservedObject var modelManager: ModelManager
    @ObservedObject var pipeline: AudioPipelineManager
    @ObservedObject var chat: ChatModel
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        TabView {
            DeskEditorView(pipeline: pipeline, modelManager: modelManager)
                .tabItem {
                    Label("寫字桌", systemImage: "pencil.and.outline")
                }

            ChatView(chat: chat, pipeline: pipeline)
                .tabItem {
                    Label("聊天", systemImage: "bubble.left.and.bubble.right")
                }

            KeyboardGuideView(modelManager: modelManager, pipeline: pipeline)
                .tabItem {
                    Label("語音鍵盤", systemImage: "keyboard")
                }

            ModelBenchmarkView(modelManager: modelManager, pipeline: pipeline)
                .tabItem {
                    Label("模型與效能", systemImage: "cpu")
                }
        }
        .tint(KomorebiTheme.sage(scheme))
    }
}
