import KomorebiCore
import KomorebiIPC
import SwiftUI
import UIKit

struct DeskEditorView: View {
    @ObservedObject var pipeline: AudioPipelineManager
    @ObservedObject var modelManager: ModelManager
    @Environment(\.colorScheme) private var scheme

    @State private var selectedRange: NSRange = NSRange(location: 0, length: 0)
    @State private var selectionInstruction: String = ""
    @State private var copiedToast: Bool = false

    private var hasSelection: Bool {
        selectedRange.length > 0 && selectedRange.location + selectedRange.length <= (pipeline.editorText as NSString).length
    }

    var body: some View {
        NavigationStack {
            ZStack {
                KomorebiTheme.paper(scheme)
                    .ignoresSafeArea()

                VStack(spacing: 12) {
                    headerStrip

                    if !modelManager.coreModelsReady {
                        missingModelsBanner
                    }

                    paperCard

                    if hasSelection {
                        selectionRewriteBar
                    }

                    controlDock
                }
                .padding(.horizontal, 16)
                .padding(.top, 8)
                .padding(.bottom, 12)
            }
            .navigationBarHidden(true)
        }
    }

    // MARK: - Subviews

    private var headerStrip: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 10) {
                HStack(spacing: 6) {
                    Text("木漏れ日")
                        .font(.system(.title3, design: .serif, weight: .semibold))
                        .foregroundStyle(KomorebiTheme.ink(scheme))
                    Text("Komorebi")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(KomorebiTheme.inkMuted(scheme))
                }

                Spacer()

                Picker("模式", selection: $pipeline.mode) {
                    ForEach(DictationMode.allCases, id: \.self) { m in
                        Text(m.label).tag(m)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 156)
            }

            Text(pipeline.statusBanner)
                .font(.caption)
                .foregroundStyle(pipeline.phase == .error ? KomorebiTheme.persimmon(scheme) : KomorebiTheme.inkMuted(scheme))
                .lineLimit(1)
                .minimumScaleFactor(0.85)
        }
    }

    private var missingModelsBanner: some View {
        HStack(spacing: 10) {
            Image(systemName: "arrow.down.circle.fill")
                .foregroundStyle(KomorebiTheme.persimmon(scheme))
            VStack(alignment: .leading, spacing: 2) {
                Text("尚未安裝離線語音與 Gemma 4 模型")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(KomorebiTheme.ink(scheme))
                Text("請切換至「模型與效能」下載 Q4_0 與 SenseVoice，即可 100% 離線口述。")
                    .font(.caption)
                    .foregroundStyle(KomorebiTheme.inkMuted(scheme))
            }
            Spacer()
        }
        .padding(12)
        .background(KomorebiTheme.card(scheme))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(KomorebiTheme.border(scheme), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var paperCard: some View {
        VStack(alignment: .leading, spacing: 0) {
            ZStack(alignment: .topLeading) {
                SelectableDeskTextView(
                    text: $pipeline.editorText,
                    selectedRange: $selectedRange,
                    freshRange: pipeline.freshRange,
                    inkColor: UIColor(KomorebiTheme.ink(scheme)),
                    freshTintColor: UIColor(KomorebiTheme.freshTint(scheme))
                )

                if pipeline.editorText.isEmpty {
                    Text("點擊下方「口述」開始離線語音輸入，或直接在此輸入並選取文字讓 Gemma 4 改寫…")
                        .font(.body)
                        .foregroundStyle(KomorebiTheme.inkMuted(scheme).opacity(0.6))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 8)
                        .allowsHitTesting(false)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(.horizontal, 14)
            .padding(.top, 12)

            if !pipeline.preeditText.isEmpty {
                HStack(spacing: 8) {
                    Circle()
                        .fill(KomorebiTheme.persimmon(scheme))
                        .frame(width: 7, height: 7)
                    Text(pipeline.preeditText)
                        .font(.callout)
                        .foregroundStyle(KomorebiTheme.ink(scheme))
                        .lineLimit(2)
                    Spacer()
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(KomorebiTheme.preeditTint(scheme))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .padding(.horizontal, 12)
                .padding(.bottom, 8)
            }

            Divider()
                .overlay(KomorebiTheme.border(scheme))

            telemetryFooter
        }
        .background(KomorebiTheme.card(scheme))
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(KomorebiTheme.border(scheme), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .shadow(color: .black.opacity(scheme == .dark ? 0.25 : 0.04), radius: 10, y: 4)
    }

    private var telemetryFooter: some View {
        HStack(spacing: 8) {
            Label(pipeline.telemetry.backendUsed, systemImage: "cpu")
                .font(.caption2.monospaced())
                .foregroundStyle(KomorebiTheme.sage(scheme))

            Button {
                pipeline.noiseSuppressionEnabled.toggle()
            } label: {
                Label(
                    pipeline.noiseSuppressionEnabled ? "降噪 ON" : "原始收音",
                    systemImage: pipeline.noiseSuppressionEnabled ? "waveform.badge.mic" : "mic"
                )
                .font(.caption2.weight(.medium))
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(
                    pipeline.noiseSuppressionEnabled
                        ? KomorebiTheme.sage(scheme).opacity(0.15)
                        : KomorebiTheme.subtle(scheme)
                )
                .foregroundStyle(
                    pipeline.noiseSuppressionEnabled
                        ? KomorebiTheme.sage(scheme)
                        : KomorebiTheme.inkMuted(scheme)
                )
                .clipShape(Capsule())
            }

            if pipeline.telemetry.decodeTPS > 0 {
                Text("P \(String(format: "%.0f", pipeline.telemetry.prefillTPS)) · D \(String(format: "%.1f", pipeline.telemetry.decodeTPS)) t/s")
                    .font(.caption2.monospaced())
                    .foregroundStyle(KomorebiTheme.inkMuted(scheme))
            }

            Spacer()

            Text("\(pipeline.editorText.count) 字")
                .font(.caption2.monospaced())
                .foregroundStyle(KomorebiTheme.inkMuted(scheme))
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(KomorebiTheme.subtle(scheme).opacity(0.65))
    }

    private var selectionRewriteBar: some View {
        HStack(spacing: 8) {
            Image(systemName: "wand.and.stars")
                .foregroundStyle(KomorebiTheme.sage(scheme))
            TextField("對反白文字下達改寫指令（例：改得更委婉）", text: $selectionInstruction)
                .font(.footnote)
                .textFieldStyle(.plain)
                .submitLabel(.go)
                .onSubmit { triggerSelectionRewrite() }

            Button("改寫") {
                triggerSelectionRewrite()
            }
            .font(.footnote.weight(.semibold))
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(KomorebiTheme.sage(scheme))
            .foregroundStyle(.white)
            .clipShape(Capsule())
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(KomorebiTheme.card(scheme))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(KomorebiTheme.sage(scheme).opacity(0.5), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var controlDock: some View {
        HStack(spacing: 16) {
            Button {
                pipeline.undoLastCommit()
            } label: {
                VStack(spacing: 3) {
                    Image(systemName: "arrow.uturn.backward")
                        .font(.body)
                    Text("復原")
                        .font(.caption2)
                }
                .frame(width: 48, height: 48)
                .foregroundStyle(KomorebiTheme.ink(scheme))
                .background(KomorebiTheme.subtle(scheme))
                .clipShape(Circle())
            }

            Button {
                pipeline.resetDraft()
            } label: {
                VStack(spacing: 3) {
                    Image(systemName: "doc.badge.plus")
                        .font(.body)
                    Text("新草稿")
                        .font(.caption2)
                }
                .frame(width: 48, height: 48)
                .foregroundStyle(KomorebiTheme.ink(scheme))
                .background(KomorebiTheme.subtle(scheme))
                .clipShape(Circle())
            }

            Spacer()

            ZStack {
                if pipeline.isListening {
                    Circle()
                        .stroke(KomorebiTheme.persimmon(scheme).opacity(0.35), lineWidth: 3)
                        .frame(width: 72 + CGFloat(pipeline.micLevel) * 20, height: 72 + CGFloat(pipeline.micLevel) * 20)
                        .animation(.easeOut(duration: 0.08), value: pipeline.micLevel)
                }

                Button {
                    pipeline.toggleListening()
                } label: {
                    VStack(spacing: 2) {
                        Image(systemName: pipeline.isListening ? "stop.fill" : "mic.fill")
                            .font(.title2)
                        Text(pipeline.isListening ? "停止" : "口述")
                            .font(.caption2.weight(.semibold))
                    }
                    .frame(width: 68, height: 68)
                    .background(pipeline.isListening ? KomorebiTheme.persimmon(scheme) : KomorebiTheme.sage(scheme))
                    .foregroundStyle(.white)
                    .clipShape(Circle())
                    .shadow(color: (pipeline.isListening ? KomorebiTheme.persimmon(scheme) : KomorebiTheme.sage(scheme)).opacity(0.3), radius: 8, y: 4)
                }
                .disabled(!modelManager.coreModelsReady)
            }

            Spacer()

            Button {
                if pipeline.isListening {
                    pipeline.flushCurrentUtterance()
                } else {
                    UIPasteboard.general.string = pipeline.editorText
                    copiedToast = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.3) {
                        copiedToast = false
                    }
                }
            } label: {
                VStack(spacing: 3) {
                    Image(systemName: pipeline.isListening ? "checkmark.circle" : (copiedToast ? "checkmark" : "doc.on.doc"))
                        .font(.body)
                    Text(pipeline.isListening ? "整句" : (copiedToast ? "已複製" : "複製"))
                        .font(.caption2)
                }
                .frame(width: 48, height: 48)
                .foregroundStyle(KomorebiTheme.ink(scheme))
                .background(KomorebiTheme.subtle(scheme))
                .clipShape(Circle())
            }

            Menu {
                ForEach(InferenceBackendPreference.allCases, id: \.self) { pref in
                    Button {
                        pipeline.backendPreference = pref
                    } label: {
                        HStack {
                            Text(pref.displayName)
                            if pipeline.backendPreference == pref {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
            } label: {
                VStack(spacing: 3) {
                    Image(systemName: "cpu")
                        .font(.body)
                    Text(backendShortTag(pipeline.backendPreference))
                        .font(.caption2)
                }
                .frame(width: 48, height: 48)
                .foregroundStyle(KomorebiTheme.ink(scheme))
                .background(KomorebiTheme.subtle(scheme))
                .clipShape(Circle())
            }
        }
        .padding(.vertical, 4)
    }

    private func backendShortTag(_ pref: InferenceBackendPreference) -> String {
        switch pref {
        case .auto: return "自動"
        case .metal: return "Metal"
        case .ane: return "ANE"
        case .cpu: return "CPU"
        }
    }

    private func triggerSelectionRewrite() {
        let ns = pipeline.editorText as NSString
        guard hasSelection, !selectionInstruction.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        let before = ns.substring(to: selectedRange.location)
        let selected = ns.substring(with: selectedRange)
        let after = ns.substring(from: selectedRange.location + selectedRange.length)
        let instr = selectionInstruction
        selectionInstruction = ""
        Task {
            await pipeline.applySelectionVoiceEdit(before: before, selected: selected, after: after, spokenInstruction: instr)
        }
    }
}

private struct SelectableDeskTextView: UIViewRepresentable {
    @Binding var text: String
    @Binding var selectedRange: NSRange
    let freshRange: NSRange?
    let inkColor: UIColor
    let freshTintColor: UIColor

    func makeUIView(context: Context) -> UITextView {
        let tv = UITextView()
        tv.delegate = context.coordinator
        tv.backgroundColor = .clear
        tv.font = .preferredFont(forTextStyle: .body)
        tv.adjustsFontForContentSizeCategory = true
        tv.textColor = inkColor
        tv.isEditable = true
        tv.isSelectable = true
        tv.keyboardDismissMode = .interactive

        let bar = UIToolbar(frame: CGRect(x: 0, y: 0, width: 320, height: 44))
        bar.sizeToFit()
        let flex = UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: nil, action: nil)
        let done = UIBarButtonItem(
            title: "收起鍵盤",
            style: .done,
            target: context.coordinator,
            action: #selector(Coordinator.dismissKeyboard)
        )
        done.accessibilityIdentifier = "dismissKeyboardButton"
        bar.items = [flex, done]
        tv.inputAccessoryView = bar
        context.coordinator.textView = tv
        return tv
    }

    func updateUIView(_ tv: UITextView, context: Context) {
        context.coordinator.parent = self
        if tv.text != text {
            let saved = tv.selectedRange
            let attr = NSMutableAttributedString(
                string: text,
                attributes: [
                    .font: UIFont.preferredFont(forTextStyle: .body),
                    .foregroundColor: inkColor,
                ]
            )
            if let fr = freshRange, fr.location >= 0, fr.location + fr.length <= (text as NSString).length {
                attr.addAttribute(.backgroundColor, value: freshTintColor, range: fr)
            }
            tv.attributedText = attr
            let clampedLoc = min(saved.location, (text as NSString).length)
            tv.selectedRange = NSRange(location: clampedLoc, length: 0)
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    final class Coordinator: NSObject, UITextViewDelegate {
        var parent: SelectableDeskTextView
        weak var textView: UITextView?
        init(_ parent: SelectableDeskTextView) { self.parent = parent }

        @objc func dismissKeyboard() {
            textView?.resignFirstResponder()
        }

        func textViewDidChange(_ textView: UITextView) {
            if parent.text != textView.text {
                parent.text = textView.text
            }
        }

        func textViewDidChangeSelection(_ textView: UITextView) {
            if parent.selectedRange != textView.selectedRange {
                DispatchQueue.main.async {
                    self.parent.selectedRange = textView.selectedRange
                }
            }
        }
    }
}
