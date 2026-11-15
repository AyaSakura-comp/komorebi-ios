import SwiftUI

/// Quiet Japanese Stationery Desk aesthetic ported verbatim from Komorebi `DESIGN.md`.
public enum KomorebiTheme {
    public static func paper(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(hex: 0x171B18) : Color(hex: 0xFAF9F6)
    }

    public static func card(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(hex: 0x1F2520) : Color(hex: 0xFFFFFF)
    }

    public static func subtle(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(hex: 0x262D27) : Color(hex: 0xF1F2EC)
    }

    public static func ink(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(hex: 0xE6E9E2) : Color(hex: 0x303B33)
    }

    public static func inkMuted(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(hex: 0x97A298) : Color(hex: 0x697268)
    }

    public static func border(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(hex: 0x333B34) : Color(hex: 0xDEDFD6)
    }

    public static func sage(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(hex: 0x8FAE8C) : Color(hex: 0x4F6550)
    }

    public static func persimmon(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(hex: 0xD98A6C) : Color(hex: 0xBD7254)
    }

    public static func freshTint(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(hex: 0x8FAE8C).opacity(0.24) : Color(hex: 0x4F6550).opacity(0.14)
    }

    public static func preeditTint(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(hex: 0xD98A6C).opacity(0.22) : Color(hex: 0xBD7254).opacity(0.14)
    }
}

public extension Color {
    init(hex: UInt32) {
        let r = Double((hex >> 16) & 0xFF) / 255.0
        let g = Double((hex >> 8) & 0xFF) / 255.0
        let b = Double(hex & 0xFF) / 255.0
        self.init(.sRGB, red: r, green: g, blue: b, opacity: 1.0)
    }
}
