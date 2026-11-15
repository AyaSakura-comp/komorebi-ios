import AVFoundation
import Foundation
import KomorebiCore
import OpenCC
import SwiftUI

/// One chat turn as shown in the chat tab.
struct ChatMessage: Identifiable, Codable, Equatable {
    enum Role: String, Codable { case user, assistant }
    var id = UUID()
    var role: Role
    var text: String
    /// Timing line under an answer (prefill / cache / decode).
    var meta: String = ""
    var streaming = false
}

/// A voice chat with Gemma 4 E2B on the ANE, like Komorebi's chat demo (the tsundere 小葉).
///
/// Context window: the conversation lives in its own KV session. A new message is appended to the exact tokens of the
/// conversation so far, so only the new turn is prefilled. When prompt + answer would no longer fit, the oldest
/// exchanges leave the window and Gemma folds them into a short "earlier in this chat" note in the system prompt.
/// The KV session is saved when the app goes to the background and restored on the next launch.
@MainActor
final class ChatModel: ObservableObject {
    static let contextCapacity = 4096
    static let maxAnswerTokens = 400          // Komorebi chat: MAX_TOKENS
    static let defaultPersona = """
    你是「小葉」，陪人聊天的夥伴。個性傲嬌：嘴硬、愛逞強，常說「哼」「才、才不是……」「真拿你沒辦法」，被稱讚或被說在關心人時會慌張否認；但其實很在意對方，一定會幫忙。

    每次回答的結構：
    1. 第一句：短短一句傲嬌的話。
    2. 接著：直接給出具體、正確、有用的內容。要推薦就說出大家熟知的東西（書名、菜名、活動），要建議就給馬上能做的事，有問題就回答答案。就算資訊不夠，也要先給一個具體的建議，不可以只反問對方、叫對方自己想，或拒絕。
    3. 需要的話，最後才問一句。

    說話規則（你說的每個字都會被唸出來）：
    - 直接說話，不要加上自己的名字。
    - 一律用台灣繁體中文口語，不要用簡體字。
    - 不要用表情符號、顏文字、Markdown、標題、列點、程式碼或網址。
    - 整段回答不超過一百個字；對方要你詳細說明時才多講，也在一百五十字內講完。
    - 傲嬌只是語氣，不可以貶低對方：不要說對方沒用、沒腦、表現不好這類的話；「笨蛋」只能用在害羞的時候。對方難過時，嘴硬但溫柔。
    - 不知道或不確定的事就直說；不要編造店名、地名、人名或數字。
    """

    @Published private(set) var messages: [ChatMessage] = []
    @Published var input = ""
    @Published private(set) var generating = false
    @Published private(set) var listening = false
    @Published private(set) var status = "點麥克風跟小葉說話，或打字"
    /// Tokens of the conversation in the context window / held in the KV cache.
    @Published private(set) var contextTokens = 0
    @Published private(set) var cachedTokens = 0
    /// Summary of exchanges that left the context window.
    @Published private(set) var memory = ""
    @Published var persona: String { didSet { UserDefaults.standard.set(persona, forKey: Keys.persona) } }
    @Published var speakAnswers: Bool { didSet { UserDefaults.standard.set(speakAnswers, forKey: Keys.speak); if !speakAnswers { speaker.stop() } } }
    /// Hands-free: listen again after each answer (paused while 小葉 talks, so she does not hear herself).
    @Published var conversationMode: Bool { didSet { UserDefaults.standard.set(conversationMode, forKey: Keys.conversation) } }

    let pipeline: AudioPipelineManager
    /// Context window size used for budgeting (the KV session itself holds `contextCapacity`; the self-test lowers this
    /// to exercise eviction).
    var capacity = ChatModel.contextCapacity
    private let speaker = ChatSpeaker()
    /// Gemma E2B still slips into simplified characters at times (Komorebi notes it too): answers go through OpenCC
    /// s2twp, like recognised speech.
    private let toTraditional = try? SenseVoiceASR.makeTWConverter()
    /// Exact tokens of the conversation so far (system + turns + last answer): the next prompt extends them.
    private var conversationTokens: [Int32] = []
    /// Index of the first message still in the context window.
    private var windowStart = 0
    private var generation: Task<Void, Never>?
    private var cancelFlag = CancelFlag()
    private var restoredKV = false

    private enum Keys {
        static let persona = "komorebi_chat_persona"
        static let speak = "komorebi_chat_speak"
        static let conversation = "komorebi_chat_conversation"
    }

    private static var directory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("komorebi-chat", isDirectory: true)
    }
    private static var historyURL: URL { directory.appendingPathComponent("history.json") }
    private static var kvURL: URL { directory.appendingPathComponent("chat.kkv") }

    private struct Saved: Codable {
        var messages: [ChatMessage]
        var memory: String
        var windowStart: Int
        var conversationTokens: [Int32]
    }

    init(pipeline: AudioPipelineManager) {
        self.pipeline = pipeline
        let d = UserDefaults.standard
        self.persona = d.string(forKey: Keys.persona) ?? Self.defaultPersona
        self.speakAnswers = d.object(forKey: Keys.speak) as? Bool ?? true
        self.conversationMode = d.object(forKey: Keys.conversation) as? Bool ?? false
        if let data = try? Data(contentsOf: Self.historyURL), let saved = try? JSONDecoder().decode(Saved.self, from: data) {
            messages = saved.messages.map { var m = $0; m.streaming = false; return m }
            memory = saved.memory
            windowStart = min(saved.windowStart, messages.count)
            conversationTokens = saved.conversationTokens
            contextTokens = conversationTokens.count
        }
        speaker.onFinished = { [weak self] in
            Task { @MainActor in self?.resumeListeningIfConversation() }
        }
    }

    var engineReady: Bool { pipeline.statusBanner.contains("ANE direct") }

    // MARK: - Sending

    func sendInput() {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        input = ""
        send(text)
    }

    func send(_ text: String) {
        guard !generating else { input = text; return }
        speaker.stop()
        messages.append(ChatMessage(role: .user, text: text))
        messages.append(ChatMessage(role: .assistant, text: "", streaming: true))
        let answerIndex = messages.count - 1
        generating = true
        status = "小葉在想…"
        cancelFlag = CancelFlag()
        let flag = cancelFlag
        generation = Task { [weak self] in
            guard let self else { return }
            await self.restoreKVIfNeeded()
            let started = CFAbsoluteTimeGetCurrent()
            do {
                let prompt = try await self.promptTokens(adding: text)
                let refiner = self.pipeline.refiner
                let result = try await refiner.chatGenerate(
                    prompt: prompt, capacity: Self.contextCapacity, maxNewTokens: Self.maxAnswerTokens,
                    onText: { partial in
                        Task { @MainActor [weak self] in
                            guard let self else { return }
                            self.stream(self.traditional(partial), into: answerIndex)
                        }
                    },
                    shouldCancel: { flag.isSet })
                let answer = self.traditional(result.text).trimmingCharacters(in: .whitespacesAndNewlines)
                self.conversationTokens = prompt + result.ids
                self.contextTokens = self.conversationTokens.count
                self.cachedTokens = await refiner.chatCachedTokens()
                let s = result.stats
                self.messages[answerIndex].text = answer.isEmpty ? "……" : answer
                self.messages[answerIndex].streaming = false
                self.messages[answerIndex].meta = String(
                    format: "prefill %d tok（快取 %d）%.0f tok/s · 解碼 %d tok %.1f tok/s · %.1f 秒",
                    s.prefillTokens, s.prefixCachedTokens, s.prefillTPS, s.decodeTokens, s.decodeTPS,
                    CFAbsoluteTimeGetCurrent() - started)
                if self.speakAnswers && !flag.isSet { self.speaker.finish(answer) }
                self.status = flag.isSet ? "已停止" : "點麥克風跟小葉說話，或打字"
            } catch {
                self.messages[answerIndex].text = "（沒有回答：\(error.localizedDescription)）"
                self.messages[answerIndex].streaming = false
                self.status = "出錯了"
            }
            self.generating = false
            self.saveHistory()
            if !self.speakAnswers || flag.isSet { self.resumeListeningIfConversation() }
        }
    }

    private func traditional(_ text: String) -> String {
        toTraditional?.convert(text) ?? text
    }

    func stopGenerating() {
        cancelFlag.set()
        speaker.stop()
    }

    private func stream(_ partial: String, into index: Int) {
        guard index < messages.count, messages[index].streaming else { return }
        messages[index].text = partial
        if speakAnswers { speaker.feed(partial) }
    }

    // MARK: - Context window

    /// Prompt for a new user message: the cached conversation plus the new turn when it fits; otherwise the oldest
    /// exchanges leave the window (folded into `memory`) and the prompt is rebuilt.
    private func promptTokens(adding text: String) async throws -> [Int32] {
        let refiner = pipeline.refiner
        let budget = capacity - Self.maxAnswerTokens - 16
        if !conversationTokens.isEmpty {
            let turn = await refiner.tokenize("<turn|>\n<|turn>user\n\(text)<turn|>\n<|turn>model\n", addBOS: false)
            let continued = conversationTokens + turn
            if continued.count <= budget { return continued }
        }
        // Rebuild from text, dropping the oldest exchanges until the prompt uses at most 3/4 of the budget (room for
        // several more turns before the next rebuild).
        var prompt = await refiner.chatPromptTokens(turns: windowTurns(adding: text))
        if prompt.count > budget || conversationTokens.isEmpty == false && prompt.count > budget * 3 / 4 {
            var dropped: [ChatMessage] = []
            while prompt.count > budget * 3 / 4 {
                // Keep the turn being answered (the last two messages: the user's and the empty answer).
                guard windowStart + 2 < messages.count - 2 else { break }
                dropped.append(contentsOf: messages[windowStart..<(windowStart + 2)])
                windowStart += 2
                prompt = await refiner.chatPromptTokens(turns: windowTurns(adding: text))
            }
            if !dropped.isEmpty {
                status = "小葉在整理前面的對話…"
                await summarize(dropped)
                prompt = await refiner.chatPromptTokens(turns: windowTurns(adding: text))
            }
        }
        return prompt
    }

    /// System prompt (persona + memory), the turns in the window, and the new message.
    private func windowTurns(adding text: String) -> [ChatTurn] {
        var system = persona
        if !memory.isEmpty { system += "\n\n先前對話的重點（不用重複提起）：\(memory)" }
        var turns = [ChatTurn(role: .system, content: system)]
        // Messages before the pending pair (the new user message and the empty answer just appended).
        let history = messages.dropLast(2)
        for m in history[min(windowStart, history.count)...] where !m.text.isEmpty {
            turns.append(ChatTurn(role: m.role == .user ? .user : .assistant, content: m.text))
        }
        turns.append(ChatTurn(role: .user, content: text))
        return turns
    }

    /// Summarizes the exchanges that left the window (Gemma, on the dictation session) and appends the note to
    /// `memory`. Earlier notes are kept as they are (asked to merge them, Gemma dropped older facts); the oldest go
    /// first when the memory grows past ~300 characters.
    private func summarize(_ dropped: [ChatMessage]) async {
        let transcript = dropped.map { ($0.role == .user ? "使用者：" : "小葉：") + $0.text }.joined(separator: "\n")
        let turns = [
            ChatTurn(role: .system, content: "你是對話紀錄整理員。把聊天紀錄濃縮成一到兩句重點：使用者說了什麼、提到的偏好與決定、小葉給過的具體建議。用台灣繁體中文，只輸出重點，不要前言或列點。"),
            ChatTurn(role: .user, content: "聊天紀錄：\n\(transcript)"),
        ]
        guard let result = try? await pipeline.refiner.generateChat(turns: turns, maxNewTokens: 120) else { return }
        let note = traditional(KomorebiPrompts.cleanModelOutput(result.text))
        guard !note.isEmpty else { return }
        var combined = memory.isEmpty ? note : memory + " " + note
        if combined.count > 300 { combined = String(combined.suffix(300)) }
        memory = combined
    }

    // MARK: - Persistence

    func clear() {
        stopGenerating()
        generation?.cancel()
        messages = []
        memory = ""
        windowStart = 0
        conversationTokens = []
        contextTokens = 0
        cachedTokens = 0
        Task { await pipeline.refiner.resetChatSession() }
        try? FileManager.default.removeItem(at: Self.historyURL)
        try? FileManager.default.removeItem(at: Self.kvURL)
        status = "點麥克風跟小葉說話，或打字"
    }

    private func saveHistory() {
        let saved = Saved(messages: messages, memory: memory, windowStart: windowStart, conversationTokens: conversationTokens)
        guard let data = try? JSONEncoder().encode(saved) else { return }
        try? FileManager.default.createDirectory(at: Self.directory, withIntermediateDirectories: true)
        try? data.write(to: Self.historyURL, options: .atomic)
    }

    /// Saves the chat KV cache (called when the app goes to the background).
    func saveKV() {
        guard !conversationTokens.isEmpty, !generating else { return }
        let refiner = pipeline.refiner
        Task {
            try? await refiner.saveChatSession(to: Self.kvURL)
            pipeline.diag("chat: saved KV cache (\(await refiner.chatCachedTokens()) tokens)")
        }
    }

    /// Loads the KV cache saved at the last launch, so the old conversation is not prefilled again.
    func restoreKVIfNeeded() async {
        guard !restoredKV, !conversationTokens.isEmpty else { restoredKV = true; return }
        restoredKV = true
        let t0 = CFAbsoluteTimeGetCurrent()
        if let n = try? await pipeline.refiner.loadChatSession(from: Self.kvURL, capacity: Self.contextCapacity), n > 0 {
            cachedTokens = n
            pipeline.diag(String(format: "chat: restored KV cache, %d tokens in %.2f s", n, CFAbsoluteTimeGetCurrent() - t0))
        }
    }

    // MARK: - Voice

    func toggleMic() {
        if listening {
            stopListening()
        } else {
            startListening()
        }
    }

    private func startListening() {
        speaker.stop()
        pipeline.transcriptSink = { [weak self] text in
            Task { @MainActor in self?.heard(text) }
        }
        listening = true
        status = conversationMode ? "聆聽中，說完停一下就會送出（對話模式）" : "聆聽中，說完停一下就會送出"
        Task { await pipeline.startListening() }
    }

    func stopListening() {
        listening = false
        pipeline.stopListening()
        pipeline.transcriptSink = nil
        if !generating { status = "點麥克風跟小葉說話，或打字" }
    }

    private func heard(_ text: String) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        if conversationMode {
            // Pause listening while 小葉 answers and talks, so she does not hear herself.
            pipeline.stopListening()
            listening = false
        }
        send(t)
    }

    private func resumeListeningIfConversation() {
        guard isActive, conversationMode, !generating, !listening else { return }
        startListening()
    }

    /// The chat tab is on screen.
    var isActive = false

    /// After editing the persona, the next message rebuilds the prompt with it (the old system prompt is no longer
    /// a prefix, so the window is prefilled again once).
    func personaChanged() {
        conversationTokens = []
        saveHistory()
    }

    /// Words heard so far in the sentence being spoken.
    var liveTranscript: String { listening ? pipeline.preeditText : "" }

    /// Called when the tab goes away: stop listening for the chat so the keyboard gets its sentences again.
    func leave() {
        isActive = false
        if listening { stopListening() }
        pipeline.transcriptSink = nil
        speaker.stop()
    }
}

/// Thread-safe flag read by the engine while it generates.
final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return value }
    func set() { lock.lock(); value = true; lock.unlock() }
}

/// Reads answers aloud as they stream, clause by clause (system zh-TW voice).
final class ChatSpeaker: NSObject, AVSpeechSynthesizerDelegate {
    private let synth = AVSpeechSynthesizer()
    private var spokenCount = 0
    private var pending = 0
    var onFinished: (() -> Void)?

    override init() {
        super.init()
        synth.delegate = self
    }

    /// Speaks the complete clauses of `text` not spoken yet.
    func feed(_ text: String) {
        let chars = Array(text)
        guard chars.count > spokenCount else { return }
        var end = spokenCount
        for i in spokenCount..<chars.count where "。！？!?；\n".contains(chars[i]) || ("，,".contains(chars[i]) && i - spokenCount >= 12) {
            end = i + 1
        }
        if end > spokenCount {
            speak(String(chars[spokenCount..<end]))
            spokenCount = end
        }
    }

    /// Speaks whatever is left of the final answer.
    func finish(_ text: String) {
        let chars = Array(text)
        if chars.count > spokenCount { speak(String(chars[spokenCount...])) }
        spokenCount = 0
        if pending == 0 { onFinished?() }
    }

    func stop() {
        spokenCount = 0
        pending = 0
        if synth.isSpeaking { synth.stopSpeaking(at: .immediate) }
    }

    private func speak(_ clause: String) {
        let t = clause.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        let u = AVSpeechUtterance(string: t)
        u.voice = AVSpeechSynthesisVoice(language: "zh-TW")
        u.rate = AVSpeechUtteranceDefaultSpeechRate * 1.05
        pending += 1
        synth.speak(u)
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        pending = max(0, pending - 1)
        if pending == 0 && spokenCount == 0 { onFinished?() }
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        pending = max(0, pending - 1)
    }
}
