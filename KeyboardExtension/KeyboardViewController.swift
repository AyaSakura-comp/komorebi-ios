import Combine
import Foundation
import KomorebiIPC
import SwiftUI
import UIKit

@MainActor
final class KeyboardBridgeModel: ObservableObject {
    @Published var state: KomorebiIPCState = KomorebiIPCStore.shared.readState()
    @Published var useDraftMode: Bool = true
    @Published var hasFullAccess: Bool = false

    private var commandID: UInt64 = UInt64(Date().timeIntervalSince1970 * 1000)
    private var lastAppliedCommitID: UInt64 = 0
    /// The draft as this keyboard put it in the text field (it sits right before the cursor).
    private var lastInsertedDraft: String = ""
    private var draftEditID: UInt64 = UInt64(Date().timeIntervalSince1970 * 1000)
    /// Last draft edit sent to the host; draft commits wait until the host has acknowledged it.
    private var pendingDraftEditID: UInt64 = 0
    private var hostSessionID: UInt64 = 0
    private var mismatchTicks = 0
    /// No integrity checks until then: the proxy's context lags a moment behind our own insertions/deletions.
    private var quietUntil: TimeInterval = 0

    private weak var controller: KeyboardViewController?
    private var pollTimer: Timer?
    private var darwinRegistered = false

    init(controller: KeyboardViewController) {
        self.controller = controller
        let current = KomorebiIPCStore.shared.readState()
        self.state = current
        self.lastAppliedCommitID = current.commitID
        self.hostSessionID = current.hostSessionID
        self.hasFullAccess = controller.hasFullAccess
        registerDarwinObserver()
    }

    func startObserving() {
        hasFullAccess = controller?.hasFullAccess ?? false
        refreshStateFromStore()
        sendCommand(.heartbeat)

        pollTimer?.invalidate()
        pollTimer = Timer.scheduledTimer(withTimeInterval: 0.15, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.refreshStateFromStore()
            }
        }
    }

    func stopObserving() {
        pollTimer?.invalidate()
        pollTimer = nil
    }

    /// The host answered recently (it republishes every 2 s while its microphone session runs).
    var hostReachable: Bool {
        state.hostAppAlive && Date().timeIntervalSince1970 - state.updatedAt < 6
    }

    var isListening: Bool { state.audioSessionActive && hostReachable }

    /// The host's microphone session is running but not listening: 語音輸入 starts without opening the app.
    var micOnStandby: Bool { state.micLive && !state.audioSessionActive && hostReachable }

    func toggleListening() {
        hasFullAccess = controller?.hasFullAccess ?? false
        if !hasFullAccess { return }

        if isListening {
            sendCommand(.stopListening)
        } else if hostReachable && state.modelsReady && (state.micLive || state.hostInForeground) {
            // The microphone session is still running (standby): start over IPC, no app switch.
            sendCommand(.startListening)
        } else {
            // Keyboards cannot record, and iOS only lets the app start a microphone session in the foreground:
            // open `komorebi://listen` so the app starts the microphone (swipe back afterwards).
            controller?.openHostURL(KomorebiAppConfig.startListeningURL)
        }
    }

    func endMicSession() {
        sendCommand(.endMicSession)
    }

    func flushUtterance() {
        sendCommand(.flushUtterance)
    }

    /// 定稿: the text stays in the field as it is; what is said next starts a new draft.
    func finalizeDraft() {
        lastInsertedDraft = ""
        sendDraftEdit("")
    }

    func toggleNoiseSuppression() {
        sendCommand(.toggleNoiseSuppression)
    }

    func commitNewline() {
        controller?.textDocumentProxy.insertText("\n")
        lastInsertedDraft = ""
        sendDraftEdit("")
    }

    func insertSpace() {
        controller?.textDocumentProxy.insertText(" ")
    }

    /// Backspace edits the draft too: the host's copy is updated, so the next sentence never brings the text back.
    func deleteBackward() {
        guard let proxy = controller?.textDocumentProxy else { return }
        refreshStateFromStore()  // apply a draft update that is already waiting, then delete from that
        let before = proxy.documentContextBeforeInput ?? ""
        let draftAtCursor = !lastInsertedDraft.isEmpty && Self.fieldEndsWith(before: before, draft: lastInsertedDraft)
        proxy.deleteBackward()
        guard !lastInsertedDraft.isEmpty else { return }
        if draftAtCursor {
            lastInsertedDraft.removeLast()
        } else {
            lastInsertedDraft = Self.draftRemainder(before: String(before.dropLast()), draft: lastInsertedDraft)
        }
        sendDraftEdit(lastInsertedDraft)
    }

    func switchKeyboard() {
        controller?.advanceToNextInputMode()
    }

    func dismissKeyboard() {
        controller?.dismissKeyboard()
    }

    private func sendCommand(_ action: KomorebiIPCCommand.Action, payload: String? = nil) {
        commandID &+= 1
        let cmd = KomorebiIPCCommand(
            id: commandID,
            action: action,
            textPayload: payload,
            useDraftMode: useDraftMode
        )
        KomorebiIPCStore.shared.sendCommand(cmd)
    }

    private func sendDraftEdit(_ text: String) {
        draftEditID &+= 1
        pendingDraftEditID = draftEditID
        quietUntil = Date().timeIntervalSince1970 + 0.6
        mismatchTicks = 0
        KomorebiIPCStore.shared.writeDraftEdit(KomorebiDraftEdit(id: draftEditID, baseCommitID: lastAppliedCommitID, text: text))
    }

    fileprivate func refreshStateFromStore() {
        let next = KomorebiIPCStore.shared.readState()
        self.state = next

        if next.hostSessionID != hostSessionID {
            // The host restarted and its draft is empty: leave the text in the field alone and start a new draft.
            hostSessionID = next.hostSessionID
            lastInsertedDraft = ""
            pendingDraftEditID = 0
        }

        guard next.commitID != lastAppliedCommitID, next.draftEditAck >= pendingDraftEditID else {
            checkDraftIntegrity()
            return
        }
        if next.isDraftReplacement, !lastInsertedDraft.isEmpty, let proxy = controller?.textDocumentProxy,
           !Self.fieldEndsWith(before: proxy.documentContextBeforeInput ?? "", draft: lastInsertedDraft) {
            // The field changed since our last update (text deleted, message sent, cursor moved). Tell the host
            // first; it rebases this update on the field's text and publishes it again.
            resyncDraftFromField()
            return
        }
        lastAppliedCommitID = next.commitID
        applyCommittedText(next.committedText, isDraftReplacement: next.isDraftReplacement)
        quietUntil = Date().timeIntervalSince1970 + 0.6
    }

    /// Notices edits made to the draft in the field by other means (another keyboard, the app clearing the field
    /// after sending, the cursor moved away) and tells the host, so its draft matches what the user sees.
    private func checkDraftIntegrity() {
        guard !lastInsertedDraft.isEmpty, let proxy = controller?.textDocumentProxy,
              Date().timeIntervalSince1970 >= quietUntil else {
            mismatchTicks = 0
            return
        }
        if Self.fieldEndsWith(before: proxy.documentContextBeforeInput ?? "", draft: lastInsertedDraft) {
            mismatchTicks = 0
            return
        }
        mismatchTicks += 1
        if mismatchTicks >= 3 { resyncDraftFromField() }
    }

    private func resyncDraftFromField() {
        let before = controller?.textDocumentProxy.documentContextBeforeInput ?? ""
        lastInsertedDraft = Self.draftRemainder(before: before, draft: lastInsertedDraft)
        sendDraftEdit(lastInsertedDraft)
    }

    /// Whether the text before the cursor ends with `draft`. Some apps return only part of the text before the
    /// cursor; a long enough context that is itself the end of the draft also counts.
    nonisolated static func fieldEndsWith(before: String, draft: String) -> Bool {
        if draft.isEmpty || before.hasSuffix(draft) { return true }
        return before.count >= 8 && before.count < draft.count && draft.hasSuffix(before)
    }

    /// The longest beginning of `draft` that still ends the text before the cursor ("" if none): what is left of
    /// the draft after its end was deleted, or nothing when the field was cleared or the cursor moved away.
    nonisolated static func draftRemainder(before: String, draft: String) -> String {
        var chars = Array(draft)
        while !chars.isEmpty {
            chars.removeLast()
            let candidate = String(chars)
            if !candidate.isEmpty, fieldEndsWith(before: before, draft: candidate) { return candidate }
        }
        return ""
    }

    private func applyCommittedText(_ text: String, isDraftReplacement: Bool) {
        guard let proxy = controller?.textDocumentProxy else { return }

        if !isDraftReplacement {
            guard !text.isEmpty else { return }
            proxy.insertText(text)
            lastInsertedDraft = ""
            return
        }

        // Draft mode diff-based replacement inside any app's text field
        if lastInsertedDraft.isEmpty {
            if !text.isEmpty {
                proxy.insertText(text)
                lastInsertedDraft = text
            }
            return
        }

        let before = proxy.documentContextBeforeInput ?? ""
        if Self.fieldEndsWith(before: before, draft: lastInsertedDraft) {
            let oldChars = Array(lastInsertedDraft)
            let newChars = Array(text)
            var prefixCount = 0
            while prefixCount < oldChars.count && prefixCount < newChars.count && oldChars[prefixCount] == newChars[prefixCount] {
                prefixCount += 1
            }

            let deleteCount = oldChars.count - prefixCount
            for _ in 0..<deleteCount {
                proxy.deleteBackward()
            }
            let suffix = String(newChars.dropFirst(prefixCount))
            if !suffix.isEmpty {
                proxy.insertText(suffix)
            }
            lastInsertedDraft = text
        } else {
            // Cursor moved outside prior draft: append cleanly and track new draft tail
            if !text.isEmpty {
                proxy.insertText(text)
                lastInsertedDraft = text
            } else {
                lastInsertedDraft = ""
            }
        }
    }

    private func registerDarwinObserver() {
        guard !darwinRegistered else { return }
        darwinRegistered = true
        let center = CFNotificationCenterGetDarwinNotifyCenter()
        let ptr = Unmanaged.passUnretained(self).toOpaque()
        CFNotificationCenterAddObserver(
            center,
            ptr,
            { _, observer, _, _, _ in
                guard let observer else { return }
                let model = Unmanaged<KeyboardBridgeModel>.fromOpaque(observer).takeUnretainedValue()
                Task { @MainActor in
                    model.refreshStateFromStore()
                }
            },
            KomorebiAppConfig.stateChangedNotification as CFString,
            nil,
            .deliverImmediately
        )
    }

    deinit {
        pollTimer?.invalidate()
        let center = CFNotificationCenterGetDarwinNotifyCenter()
        CFNotificationCenterRemoveEveryObserver(center, Unmanaged.passUnretained(self).toOpaque())
    }
}

final class KeyboardViewController: UIInputViewController {
    private lazy var bridge = KeyboardBridgeModel(controller: self)
    private var hostingController: UIHostingController<KeyboardView>?

    override func viewDidLoad() {
        super.viewDidLoad()

        let rootView = KeyboardView(bridge: bridge, needsInputModeSwitchKey: needsInputModeSwitchKey)
        let host = UIHostingController(rootView: rootView)
        host.view.translatesAutoresizingMaskIntoConstraints = false
        host.view.backgroundColor = .clear

        addChild(host)
        view.addSubview(host.view)
        NSLayoutConstraint.activate([
            host.view.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            host.view.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            host.view.topAnchor.constraint(equalTo: view.topAnchor),
            host.view.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            host.view.heightAnchor.constraint(equalToConstant: 210),
        ])
        host.didMove(toParent: self)
        self.hostingController = host
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        bridge.startObserving()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        bridge.stopObserving()
    }

    /// Opens host app URL (`komorebi://listen`) via responder chain when Host App isn't already recording.
    @MainActor
    func openHostURL(_ url: URL) {
        let modernSel = sel_registerName("openURL:options:completionHandler:")
        let legacySel = sel_registerName("openURL:")
        let appClass: AnyClass? = NSClassFromString("UIApplication")
        var fallbackScene: UIScene?
        var responder: UIResponder? = self
        while let current = responder {
            if let appClass, current.isKind(of: appClass) {
                if current.responds(to: modernSel),
                   let imp = class_getMethodImplementation(type(of: current), modernSel) {
                    typealias OpenURLFn = @convention(c) (AnyObject, Selector, NSURL, NSDictionary, OpaquePointer?) -> Void
                    let fn = unsafeBitCast(imp, to: OpenURLFn.self)
                    fn(current, modernSel, url as NSURL, [:] as NSDictionary, nil)
                    return
                }
                if current.responds(to: legacySel) {
                    _ = current.perform(legacySel, with: url)
                    return
                }
            }
            if fallbackScene == nil, let scene = current as? UIScene {
                fallbackScene = scene
            }
            responder = current.next
        }
        fallbackScene?.open(url, options: nil, completionHandler: nil)
    }
}
