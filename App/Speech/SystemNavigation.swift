import ObjectiveC
import UIKit

/// Goes back to the app that opened Komorebi, so the keyboard's tap only flashes Komorebi.
/// Strategies, in order (all private API, each skipped when unavailable):
/// 1. The status-bar breadcrumb ("◀ <App>"): `_systemNavigationAction` → `sendResponseForDestination:0`,
///    on the application and on each connected scene.
/// 2. Reopen the keyboard's host app by bundle id (passed in `komorebi://listen?return=<id>`) with
///    `LSApplicationWorkspace openApplicationWithBundleID:`.
@MainActor
enum SystemNavigation {
    /// Returns what happened, for the diagnostics log.
    /// iOS gives no breadcrumb when a keyboard opens the app (checked on iOS 27.2 even after 1.5 s), so by default
    /// this checks once and does not delay listening.
    @discardableResult
    static func returnToPreviousApp(hostBundleID: String?, wait: Double = 0) async -> String {
        var notes: [String] = []
        let deadline = Date().addingTimeInterval(wait)
        repeat {
            notes = []
            var targets: [NSObject] = [UIApplication.shared]
            targets.append(contentsOf: UIApplication.shared.connectedScenes.map { $0 as NSObject })
            for target in targets {
                let r = sendBreadcrumb(on: target)
                if r == "returned" { return "breadcrumb on \(type(of: target)): returned" }
                notes.append("\(type(of: target)): \(r)")
            }
            if wait > 0 { try? await Task.sleep(nanoseconds: 100_000_000) }
        } while Date() < deadline
        if let id = hostBundleID, !id.isEmpty {
            let r = openApplication(bundleID: id)
            if r.hasPrefix("opened") { return r }
            notes.append(r)
        } else {
            notes.append("no host bundle id from the keyboard")
        }
        return notes.joined(separator: "; ")
    }

    private static func sendBreadcrumb(on target: NSObject) -> String {
        let getter = NSSelectorFromString("_systemNavigationAction")
        guard target.responds(to: getter) else { return "no _systemNavigationAction" }
        guard let action = target.perform(getter)?.takeUnretainedValue() as? NSObject else { return "no breadcrumb" }
        var destinations: [Int] = []
        let destinationsSel = NSSelectorFromString("destinations")
        if action.responds(to: destinationsSel),
           let list = action.perform(destinationsSel)?.takeUnretainedValue() as? [NSNumber] {
            destinations = list.map(\.intValue)
        }
        // Destination 0 is the app that opened us (1 is a forward link).
        let sendSel = NSSelectorFromString("sendResponseForDestination:")
        guard destinations.contains(0), action.responds(to: sendSel) else {
            return "breadcrumb without a way back (destinations \(destinations))"
        }
        typealias SendResponse = @convention(c) (AnyObject, Selector, UInt) -> Bool
        let send = unsafeBitCast(action.method(for: sendSel), to: SendResponse.self)
        return send(action, sendSel, 0) ? "returned" : "breadcrumb refused"
    }

    /// `LSApplicationWorkspace.defaultWorkspace().openApplicationWithBundleID(_:)`.
    static func openApplication(bundleID: String) -> String {
        guard let cls = NSClassFromString("LSApplicationWorkspace") as? NSObject.Type else { return "no LSApplicationWorkspace" }
        let defaultSel = NSSelectorFromString("defaultWorkspace")
        guard cls.responds(to: defaultSel),
              let workspace = cls.perform(defaultSel)?.takeUnretainedValue() as? NSObject else { return "no defaultWorkspace" }
        let openSel = NSSelectorFromString("openApplicationWithBundleID:")
        guard workspace.responds(to: openSel) else { return "no openApplicationWithBundleID:" }
        typealias Open = @convention(c) (AnyObject, Selector, NSString) -> Bool
        let open = unsafeBitCast(workspace.method(for: openSel), to: Open.self)
        return open(workspace, openSel, bundleID as NSString) ? "opened \(bundleID)" : "LSApplicationWorkspace refused \(bundleID)"
    }

    /// Lists the navigation-related private API this iOS has (diagnostics, `--nav-probe`).
    static func probe() -> String {
        let keys = ["navigationaction", "breadcrumb", "backto", "returnto", "sourceapp", "originat"]
        var lines: [String] = []
        for name in ["UIApplication", "UIScene", "UIWindowScene", "UISystemNavigationAction", "LSApplicationWorkspace"] {
            guard let cls = NSClassFromString(name) else { lines.append("\(name): no class"); continue }
            var found: [String] = []
            for (c, prefix) in [(cls, ""), (object_getClass(cls)!, "+")] as [(AnyClass, String)] {
                var count: UInt32 = 0
                guard let list = class_copyMethodList(c, &count) else { continue }
                for i in 0..<Int(count) {
                    let sel = NSStringFromSelector(method_getName(list[i]))
                    let lower = sel.lowercased()
                    let wanted = name == "UISystemNavigationAction"
                        || (name == "LSApplicationWorkspace" && lower.contains("openapplication"))
                        || keys.contains { lower.contains($0) }
                    if wanted { found.append(prefix + sel) }
                }
                free(list)
            }
            lines.append("\(name): \(found.sorted().joined(separator: " "))")
        }
        return lines.joined(separator: "\n")
    }
}
