import SwiftUI
import UIKit

struct KeyboardGuideView: View {
    @ObservedObject var modelManager: ModelManager
    @ObservedObject var pipeline: AudioPipelineManager
    @Environment(\.colorScheme) private var scheme

    @State private var scratchInput: String = ""
    @FocusState private var sandboxFocused: Bool

    var body: some View {
        NavigationStack {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(spacing: 16) {
                        statusCard
                        stepsCard
                        sandboxCard
                            .id("sandboxCard")
                    }
                    .padding(16)
                }
                .scrollDismissesKeyboard(.interactively)
                .onChange(of: sandboxFocused) { _, focused in
                    if focused {
                        withAnimation(.easeOut(duration: 0.25)) {
                            proxy.scrollTo("sandboxCard", anchor: .bottom)
                        }
                    }
                }
            }
            .background(KomorebiTheme.paper(scheme).ignoresSafeArea())
            .navigationTitle("語音小鍵盤設定")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var statusCard: some View {
        HStack(spacing: 12) {
            Image(systemName: modelManager.coreModelsReady ? "keyboard.badge.ellipsis" : "exclamationmark.triangle.fill")
                .font(.title2)
                .foregroundStyle(modelManager.coreModelsReady ? KomorebiTheme.sage(scheme) : KomorebiTheme.persimmon(scheme))

            VStack(alignment: .leading, spacing: 2) {
                Text(modelManager.coreModelsReady ? "主程式離線引擎就緒" : "尚缺核心模型")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(KomorebiTheme.ink(scheme))
                Text("背景語音輸入由主 App 持有 Audio 背景權限並透過 App Group 即時回傳小鍵盤。")
                    .font(.caption)
                    .foregroundStyle(KomorebiTheme.inkMuted(scheme))
            }
            Spacer()
        }
        .padding(14)
        .background(KomorebiTheme.card(scheme))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(KomorebiTheme.border(scheme), lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private var stepsCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("啟用系統「Komorebi 語音鍵盤」三步驟")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(KomorebiTheme.ink(scheme))

            guideRow(num: "1", title: "新增系統鍵盤", desc: "前往 iOS 設定 → 一般 → 鍵盤 → 鍵盤 → 新增「Komorebi 語音鍵盤」。")
            guideRow(num: "2", title: "開啟「允許完整取用」", desc: "iOS 僅在開啟完整取用時允許鍵盤透過 App Group 與主 App 交換辨識結果，所有模型與音訊皆 100% 留在本機，不上傳任何資料。")
            guideRow(num: "3", title: "在任意 App 切換小鍵盤開講", desc: "按地球鍵切換到 Komorebi，點擊麥克風即可在 LINE、備忘錄、Safari 即時口述與 Gemma 4 潤飾。")

            Button {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            } label: {
                Label("前往 iOS 系統設定", systemImage: "gearshape.fill")
                    .font(.footnote.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(KomorebiTheme.sage(scheme))
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            }
        }
        .padding(14)
        .background(KomorebiTheme.card(scheme))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(KomorebiTheme.border(scheme), lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private var sandboxCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("小鍵盤實測區")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(KomorebiTheme.ink(scheme))
                Spacer()
                if sandboxFocused {
                    Button("收起鍵盤") {
                        sandboxFocused = false
                        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                    }
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(KomorebiTheme.sage(scheme))
                    .accessibilityIdentifier("sandboxDismissKeyboard")
                }
            }
            TextField("點此切換成 Komorebi 語音小鍵盤試用…", text: $scratchInput, axis: .vertical)
                .lineLimit(3...6)
                .textFieldStyle(.roundedBorder)
                .font(.body)
                .focused($sandboxFocused)
        }
        .padding(14)
        .background(KomorebiTheme.card(scheme))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(KomorebiTheme.border(scheme), lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private func guideRow(num: String, title: String, desc: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Text(num)
                .font(.caption.weight(.bold))
                .frame(width: 22, height: 22)
                .background(KomorebiTheme.subtle(scheme))
                .foregroundStyle(KomorebiTheme.sage(scheme))
                .clipShape(Circle())
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(KomorebiTheme.ink(scheme))
                Text(desc)
                    .font(.caption)
                    .foregroundStyle(KomorebiTheme.inkMuted(scheme))
            }
        }
    }
}
