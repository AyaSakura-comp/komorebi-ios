import SwiftUI

/// Voice chat with 小葉 (Gemma 4 E2B on the ANE): its own KV cache, context window management, read-aloud answers.
struct ChatView: View {
    @ObservedObject var chat: ChatModel
    @ObservedObject var pipeline: AudioPipelineManager
    @Environment(\.colorScheme) private var scheme
    @State private var editingPersona = false
    @FocusState private var inputFocused: Bool

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                contextBar
                messageList
                inputBar
            }
            .background(KomorebiTheme.paper(scheme).ignoresSafeArea())
            .navigationTitle("小葉")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) { menu }
            }
            .sheet(isPresented: $editingPersona) { personaEditor }
        }
        .onAppear { chat.isActive = true }
        .onDisappear { chat.leave() }
    }

    private var contextBar: some View {
        let used = chat.contextTokens, cap = ChatModel.contextCapacity
        return VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                ProgressView(value: Double(min(used, cap)), total: Double(cap))
                    .tint(used > cap * 8 / 10 ? KomorebiTheme.persimmon(scheme) : KomorebiTheme.sage(scheme))
                Text("上下文 \(used)/\(cap)")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(KomorebiTheme.inkMuted(scheme))
            }
            Text(chat.cachedTokens > 0 ? "KV 快取 \(chat.cachedTokens) tokens：下一句只需 prefill 新的部分" : "KV 快取：第一句會從頭 prefill")
                .font(.caption2)
                .foregroundStyle(KomorebiTheme.inkMuted(scheme))
            if !chat.memory.isEmpty {
                Text("較早的對話已整理成重點：\(chat.memory)")
                    .font(.caption2)
                    .foregroundStyle(KomorebiTheme.sage(scheme))
                    .lineLimit(2)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(KomorebiTheme.subtle(scheme).opacity(0.6))
    }

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    if chat.messages.isEmpty {
                        Text(chat.engineReady ? "跟小葉說點什麼吧。她嘴硬，但一定會幫你。" : "等 Gemma 載入完成就能聊天（「模型與效能」頁看狀態）。")
                            .font(.footnote)
                            .foregroundStyle(KomorebiTheme.inkMuted(scheme))
                            .frame(maxWidth: .infinity, alignment: .center)
                            .padding(.top, 40)
                    }
                    ForEach(chat.messages) { message in
                        bubble(message).id(message.id)
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
            }
            .scrollDismissesKeyboard(.interactively)
            .onChange(of: chat.messages) { _, _ in
                withAnimation(.easeOut(duration: 0.15)) { proxy.scrollTo("bottom", anchor: .bottom) }
            }
        }
    }

    private func bubble(_ m: ChatMessage) -> some View {
        let mine = m.role == .user
        return HStack {
            if mine { Spacer(minLength: 40) }
            VStack(alignment: mine ? .trailing : .leading, spacing: 3) {
                Text(m.text.isEmpty && m.streaming ? "…" : m.text)
                    .font(.body)
                    .foregroundStyle(mine ? Color.white : KomorebiTheme.ink(scheme))
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(mine ? KomorebiTheme.sage(scheme) : KomorebiTheme.card(scheme))
                    .clipShape(RoundedRectangle(cornerRadius: 14))
                    .overlay(
                        RoundedRectangle(cornerRadius: 14)
                            .stroke(mine ? Color.clear : KomorebiTheme.border(scheme), lineWidth: 1)
                    )
                    .textSelection(.enabled)
                if !m.meta.isEmpty {
                    Text(m.meta)
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(KomorebiTheme.inkMuted(scheme))
                }
            }
            if !mine { Spacer(minLength: 40) }
        }
    }

    private var inputBar: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(chat.listening && !chat.liveTranscript.isEmpty ? "聽到：\(chat.liveTranscript)" : chat.status)
                .font(.caption)
                .foregroundStyle(chat.listening ? KomorebiTheme.persimmon(scheme) : KomorebiTheme.inkMuted(scheme))
                .lineLimit(2)
            HStack(alignment: .bottom, spacing: 8) {
                Button {
                    inputFocused = false
                    chat.toggleMic()
                } label: {
                    Image(systemName: chat.listening ? "stop.fill" : "mic.fill")
                        .font(.body.weight(.semibold))
                        .frame(width: 40, height: 40)
                        .background(chat.listening ? KomorebiTheme.persimmon(scheme) : KomorebiTheme.subtle(scheme))
                        .foregroundStyle(chat.listening ? Color.white : KomorebiTheme.ink(scheme))
                        .clipShape(Circle())
                }
                .disabled(!chat.engineReady)

                TextField("跟小葉說話…", text: $chat.input, axis: .vertical)
                    .lineLimit(1...4)
                    .focused($inputFocused)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 9)
                    .background(KomorebiTheme.card(scheme))
                    .clipShape(RoundedRectangle(cornerRadius: 18))
                    .overlay(RoundedRectangle(cornerRadius: 18).stroke(KomorebiTheme.border(scheme), lineWidth: 1))
                    .onSubmit { chat.sendInput() }

                Button {
                    if chat.generating { chat.stopGenerating() } else { chat.sendInput() }
                } label: {
                    Image(systemName: chat.generating ? "stop.circle.fill" : "arrow.up.circle.fill")
                        .font(.system(size: 34))
                        .foregroundStyle(KomorebiTheme.sage(scheme))
                }
                .disabled(!chat.engineReady || (!chat.generating && chat.input.trimmingCharacters(in: .whitespaces).isEmpty))
            }
        }
        .padding(.horizontal, 12)
        .padding(.top, 8)
        .padding(.bottom, 10)
        .background(KomorebiTheme.paper(scheme))
    }

    private var menu: some View {
        Menu {
            Toggle("朗讀回答", isOn: $chat.speakAnswers)
            Toggle("對話模式（說完自動再聽）", isOn: $chat.conversationMode)
            Button("編輯角色…") { editingPersona = true }
            Divider()
            Button("清除對話", role: .destructive) { chat.clear() }
        } label: {
            Image(systemName: "ellipsis.circle")
        }
    }

    private var personaEditor: some View {
        NavigationStack {
            TextEditor(text: $chat.persona)
                .font(.footnote)
                .padding(10)
                .navigationTitle("角色設定（system prompt）")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) {
                        Button("預設") { chat.persona = ChatModel.defaultPersona }
                    }
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("完成") {
                            chat.personaChanged()
                            editingPersona = false
                        }
                    }
                }
        }
    }
}
