import UIKit

/// Goes back to the app that opened Komorebi: the same as tapping the status bar's "◀ <App>" breadcrumb.
/// Used after the keyboard's first tap, which has to open Komorebi to start the microphone, so the user lands back in
/// the app they were typing in. Uses UIKit's private `_systemNavigationAction`; does nothing when it is unavailable.
@MainActor
enum SystemNavigation {
    /// Returns what happened, for the diagnostics log.
    @discardableResult
    static func returnToPreviousApp() -> String {
        let app = UIApplication.shared
        let getter = NSSelectorFromString("_systemNavigationAction")
        guard app.responds(to: getter),
              let action = app.perform(getter)?.takeUnretainedValue() as? NSObject else {
            return "no breadcrumb"
        }
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
}
