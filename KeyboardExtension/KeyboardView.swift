import KomorebiIPC
import SwiftUI

private enum KeyboardPalette {
    static func paper(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(red: 0.09, green: 0.106, blue: 0.094) : Color(red: 0.98, green: 0.976, blue: 0.965)
    }
    static func card(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(red: 0.122, green: 0.145, blue: 0.125) : .white
    }
    static func subtle(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(red: 0.149, green: 0.176, blue: 0.153) : Color(red: 0.945, green: 0.949, blue: 0.925)
    }
    static func ink(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(red: 0.902, green: 0.914, blue: 0.886) : Color(red: 0.188, green: 0.231, blue: 0.200)
    }
    static func muted(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(red: 0.592, green: 0.635, blue: 0.596) : Color(red: 0.412, green: 0.447, blue: 0.408)
    }
    static func border(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(red: 0.20, green: 0.231, blue: 0.204) : Color(red: 0.871, green: 0.875, blue: 0.839)
    }
    static func sage(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(red: 0.561, green: 0.682, blue: 0.549) : Color(red: 0.310, green: 0.396, blue: 0.314)
    }
    static func persimmon(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(red: 0.851, green: 0.541, blue: 0.424) : Color(red: 0.741, green: 0.447, blue: 0.329)
    }
}

struct KeyboardView: View {
    @ObservedObject var bridge: KeyboardBridgeModel
    let needsInputModeSwitchKey: Bool
    @Environment(\.colorScheme) private var scheme

    private var isListening: Bool {
        bridge.isListening
    }

    var body: some View {
        VStack(spacing: 8) {
            statusHeader
            preeditArea
            actionBar
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(KeyboardPalette.paper(scheme))
    }

    private var statusHeader: some View {
        HStack(spacing: 8) {
            HStack(spacing: 5) {
                Circle()
                    .fill(statusDotColor)
                    .frame(width: 7, height: 7)
                Text(statusLabel)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(KeyboardPalette.ink(scheme))
                    .lineLimit(1)
            }

            Spacer()

            if bridge.micOnStandby {
                Button {
                    bridge.endMicSession()
                } label: {
                    Text("關麥")
                        .font(.caption2.weight(.semibold))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(KeyboardPalette.subtle(scheme))
                        .foregroundStyle(KeyboardPalette.persimmon(scheme))
                        .clipShape(Capsule())
                }
                .accessibilityLabel("關閉待命中的麥克風")
            } else if bridge.state.telemetry.decodeTPS > 0 {
                Text("\(bridge.state.telemetry.backendUsed.components(separatedBy: " ").first ?? "") \(String(format: "%.0f", bridge.state.telemetry.decodeTPS))t/s")
                    .font(.caption2.monospaced())
                    .foregroundStyle(KeyboardPalette.muted(scheme))
            }

            Button {
                bridge.toggleNoiseSuppression()
            } label: {
                Text(bridge.state.noiseSuppressionActive ? "降噪ON" : "原音")
                    .font(.caption2.weight(.medium))
                    .padding(.horizontal, 7)
                    .padding(.vertical, 4)
                    .background(KeyboardPalette.subtle(scheme))
                    .foregroundStyle(bridge.state.noiseSuppressionActive ? KeyboardPalette.sage(scheme) : KeyboardPalette.muted(scheme))
                    .clipShape(Capsule())
            }

            Button {
                bridge.useDraftMode.toggle()
            } label: {
                Text(bridge.useDraftMode ? "草稿整合" : "逐句模式")
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(KeyboardPalette.subtle(scheme))
                    .foregroundStyle(KeyboardPalette.sage(scheme))
                    .clipShape(Capsule())
            }

            Button {
                bridge.dismissKeyboard()
            } label: {
                Image(systemName: "keyboard.chevron.compact.down")
                    .font(.caption)
                    .frame(width: 26, height: 24)
                    .background(KeyboardPalette.subtle(scheme))
                    .foregroundStyle(KeyboardPalette.muted(scheme))
                    .clipShape(Capsule())
            }
            .accessibilityLabel("收起鍵盤")
        }
    }

    private var preeditArea: some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                if !bridge.hasFullAccess {
                    Text("請至「設定 → 鍵盤 → Komorebi」開啟「允許完整取用」，才能與主 App 離線引擎連線。")
                        .font(.caption)
                        .foregroundStyle(KeyboardPalette.persimmon(scheme))
                        .lineLimit(2)
                } else if !bridge.state.partialText.isEmpty || !bridge.state.partialTail.isEmpty {
                    (Text(bridge.state.partialText)
                        .foregroundColor(KeyboardPalette.ink(scheme))
                     + Text(bridge.state.partialTail)
                        .foregroundColor(KeyboardPalette.muted(scheme)))
                        .font(.callout)
                        .lineLimit(2)
                        .truncationMode(.head)
                } else if isListening {
                    Text("邊說邊整理：整理好的字直接出現在輸入框（那就是草稿，⌫ 會同步），這裡淡色的是還沒整理的字")
                        .font(.caption)
                        .foregroundStyle(KeyboardPalette.muted(scheme))
                        .lineLimit(2)
                } else if bridge.micOnStandby {
                    Text("麥克風待命中：點「語音輸入」直接開始，不會跳到 App。「定稿」＝保留現有文字，之後另起新草稿")
                        .font(.caption)
                        .foregroundStyle(KeyboardPalette.muted(scheme))
                        .lineLimit(2)
                } else {
                    Text("點麥克風開始（第一次會切到 App 開麥克風再切回來）。說的話會整理成草稿放在輸入框")
                        .font(.caption)
                        .foregroundStyle(KeyboardPalette.muted(scheme))
                        .lineLimit(2)
                }
            }
            Spacer(minLength: 4)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity, minHeight: 58, alignment: .leading)
        .background(KeyboardPalette.card(scheme))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(KeyboardPalette.border(scheme), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var actionBar: some View {
        HStack(spacing: 8) {
            if needsInputModeSwitchKey {
                Button {
                    bridge.switchKeyboard()
                } label: {
                    Image(systemName: "globe")
                        .font(.body)
                        .frame(width: 40, height: 42)
                        .background(KeyboardPalette.subtle(scheme))
                        .foregroundStyle(KeyboardPalette.ink(scheme))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                }
            }

            Button {
                bridge.finalizeDraft()
            } label: {
                Text("定稿")
                    .font(.caption.weight(.medium))
                    .frame(width: 46, height: 42)
                    .background(KeyboardPalette.subtle(scheme))
                    .foregroundStyle(KeyboardPalette.ink(scheme))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            }
            .accessibilityLabel("定稿：輸入框的文字保留，之後說的話另起新草稿")

            Button {
                bridge.toggleListening()
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: isListening ? "stop.fill" : "mic.fill")
                    Text(isListening ? "停止聆聽" : "語音輸入")
                        .font(.callout.weight(.semibold))
                    if isListening {
                        Capsule()
                            .fill(.white.opacity(0.85))
                            .frame(width: max(6, CGFloat(bridge.state.micLevel) * 26), height: 6)
                    }
                }
                .frame(maxWidth: .infinity, minHeight: 42)
                .background(isListening ? KeyboardPalette.persimmon(scheme) : KeyboardPalette.sage(scheme))
                .foregroundStyle(.white)
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }

            if isListening {
                Button {
                    bridge.flushUtterance()
                } label: {
                    Text("斷句")
                        .font(.caption.weight(.semibold))
                        .frame(width: 44, height: 42)
                        .background(KeyboardPalette.subtle(scheme))
                        .foregroundStyle(KeyboardPalette.sage(scheme))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                }
                .accessibilityLabel("斷句：不等停頓，立刻整理剛剛說的話")
            } else {
                Button {
                    bridge.insertSpace()
                } label: {
                    Text("空白")
                        .font(.caption.weight(.medium))
                        .frame(width: 44, height: 42)
                        .background(KeyboardPalette.subtle(scheme))
                        .foregroundStyle(KeyboardPalette.ink(scheme))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                }
            }

            Button {
                bridge.deleteBackward()
            } label: {
                Image(systemName: "delete.left")
                    .font(.body)
                    .frame(width: 40, height: 42)
                    .background(KeyboardPalette.subtle(scheme))
                    .foregroundStyle(KeyboardPalette.ink(scheme))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            }

            Button {
                bridge.commitNewline()
            } label: {
                Image(systemName: "return")
                    .font(.body)
                    .frame(width: 40, height: 42)
                    .background(KeyboardPalette.subtle(scheme))
                    .foregroundStyle(KeyboardPalette.ink(scheme))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            }
        }
    }

    private var statusDotColor: Color {
        if !bridge.hasFullAccess { return KeyboardPalette.persimmon(scheme) }
        switch bridge.state.phase {
        case .listening, .transcribing, .refining:
            return KeyboardPalette.persimmon(scheme)
        case .error:
            return .red
        case .idle:
            return bridge.state.modelsReady ? KeyboardPalette.sage(scheme) : KeyboardPalette.muted(scheme)
        }
    }

    private var statusLabel: String {
        if !bridge.hasFullAccess { return "需要「允許完整取用」" }
        if !bridge.hostReachable && bridge.state.phase != .idle {
            return bridge.state.modelsReady ? "Komorebi 就緒" : "請先於主 App 下載模型"
        }
        switch bridge.state.phase {
        case .listening: return "聆聽中（Komorebi）"
        case .transcribing: return "SenseVoice 辨識中…"
        case .refining: return "Gemma 4 E2B 潤飾中…"
        case .error: return bridge.state.errorMessage ?? "發生錯誤"
        case .idle:
            if bridge.state.enginesLoading && bridge.hostReachable { return "載入模型中…" }
            if bridge.micOnStandby { return "麥克風待命中" }
            return bridge.state.modelsReady ? "Komorebi 就緒" : "請先於主 App 下載模型"
        }
    }
}
