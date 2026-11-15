import XCTest

final class KomorebiCrossAppKeyboardUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    private func findElement(_ app: XCUIApplication, matching predicateFormat: String, timeout: TimeInterval = 3) -> XCUIElement? {
        let pred = NSPredicate(format: predicateFormat)
        let queries: [XCUIElementQuery] = [
            app.buttons.matching(pred),
            app.cells.matching(pred),
            app.staticTexts.matching(pred),
        ]
        let deadline = Date().addingTimeInterval(timeout)
        repeat {
            for q in queries {
                let el = q.firstMatch
                if el.exists && el.isHittable {
                    return el
                }
            }
            for q in queries {
                let el = q.firstMatch
                if el.exists {
                    return el
                }
            }
            RunLoop.current.run(until: Date().addingTimeInterval(0.2))
        } while Date() < deadline
        return nil
    }

    private func findAndTap(_ app: XCUIApplication, matching predicateFormat: String, failureLabel: String) {
        if let el = findElement(app, matching: predicateFormat, timeout: 3) {
            el.tap()
            return
        }
        for _ in 0..<3 {
            app.swipeUp()
            if let el = findElement(app, matching: predicateFormat, timeout: 2) {
                el.tap()
                return
            }
        }
        for _ in 0..<4 {
            app.swipeDown()
            if let el = findElement(app, matching: predicateFormat, timeout: 2) {
                el.tap()
                return
            }
        }
        XCTFail("\(failureLabel) missing: \(app.debugDescription)")
    }

    @MainActor
    func testEnableAndUseKeyboardInOtherApp() throws {
        // 0. Ensure KomorebiVoice host app launches cleanly & loads models into App Group IPC
        let komorebi = XCUIApplication(bundleIdentifier: "dev.ayasakura.komorebi")
        komorebi.launch()
        XCTAssertTrue(komorebi.staticTexts["木漏れ日"].waitForExistence(timeout: 15))
        let readyLabel = komorebi.staticTexts.matching(NSPredicate(format: "label BEGINSWITH '就緒：'")).firstMatch
        XCTAssertTrue(readyLabel.waitForExistence(timeout: 20))
        Thread.sleep(forTimeInterval: 1.0)

        // 1. Open iOS Settings -> 一般 -> 鍵盤 -> 鍵盤 -> enable Komorebi 語音鍵盤 & 允許完整取用
        let settings = XCUIApplication(bundleIdentifier: "com.apple.Preferences")
        settings.terminate()
        settings.launch()
        Thread.sleep(forTimeInterval: 1.0)

        findAndTap(settings, matching: "identifier == 'com.apple.settings.general' OR label == '一般' OR label == 'General'", failureLabel: "Settings 一般")
        Thread.sleep(forTimeInterval: 0.7)

        findAndTap(settings, matching: "label == '鍵盤' OR label == 'Keyboard' OR identifier CONTAINS 'Keyboard'", failureLabel: "鍵盤")
        Thread.sleep(forTimeInterval: 0.7)

        findAndTap(settings, matching: "label BEGINSWITH '鍵盤' OR label BEGINSWITH 'Keyboards'", failureLabel: "鍵盤列表")
        Thread.sleep(forTimeInterval: 0.7)

        if findElement(settings, matching: "label CONTAINS 'Komorebi'", timeout: 2) == nil {
            findAndTap(settings, matching: "label CONTAINS '新增鍵盤' OR label CONTAINS 'Add New Keyboard'", failureLabel: "新增鍵盤")
            Thread.sleep(forTimeInterval: 0.8)
            findAndTap(settings, matching: "label CONTAINS 'Komorebi'", failureLabel: "Komorebi 第三方鍵盤")
            Thread.sleep(forTimeInterval: 0.8)
        }

        findAndTap(settings, matching: "label CONTAINS 'Komorebi'", failureLabel: "已加入的 Komorebi 鍵盤列")
        Thread.sleep(forTimeInterval: 0.7)

        let fullAccessSwitch = settings.switches.matching(NSPredicate(format: "label CONTAINS '允許完整取用' OR label CONTAINS 'Allow Full Access'")).firstMatch
        XCTAssertTrue(fullAccessSwitch.waitForExistence(timeout: 5), "Full Access switch missing: \(settings.debugDescription)")
        if (fullAccessSwitch.value as? String) != "1" {
            // Tap the toggle knob on the right side of the SwiftUI/UIKit row if the switch spans full width
            if fullAccessSwitch.frame.width > 100 {
                fullAccessSwitch.coordinate(withNormalizedOffset: CGVector(dx: 0.90, dy: 0.5)).tap()
            } else {
                fullAccessSwitch.tap()
            }
            Thread.sleep(forTimeInterval: 0.8)
            let allowCandidates: [XCUIElement] = [
                settings.alerts.buttons.matching(NSPredicate(format: "label == '允許' OR label == 'Allow'")).firstMatch,
                settings.buttons.matching(NSPredicate(format: "label == '允許' OR label == 'Allow'")).firstMatch,
                XCUIApplication(bundleIdentifier: "com.apple.springboard").alerts.buttons.matching(NSPredicate(format: "label == '允許' OR label == 'Allow'")).firstMatch,
            ]
            for allowBtn in allowCandidates {
                if allowBtn.waitForExistence(timeout: 1.5) && allowBtn.isHittable {
                    allowBtn.tap()
                    Thread.sleep(forTimeInterval: 0.6)
                    break
                }
            }
        }
        XCTAssertEqual(fullAccessSwitch.value as? String, "1", "允許完整取用 did not turn ON: \(settings.debugDescription)")

        // 2. Open Apple 提醒事項 (Reminders) - another stock Apple app!
        let reminders = XCUIApplication(bundleIdentifier: "com.apple.reminders")
        reminders.terminate()
        reminders.launch()
        Thread.sleep(forTimeInterval: 1.5)

        for label in ["繼續", "Continue", "稍後再說", "Not Now", "好", "OK"] {
            let btn = reminders.buttons[label]
            if btn.waitForExistence(timeout: 1.5) && btn.isHittable {
                btn.tap()
                Thread.sleep(forTimeInterval: 0.8)
            }
        }

        let newReminderBtn = reminders.buttons.matching(NSPredicate(format: "label CONTAINS '新增提醒事項' OR label CONTAINS 'New Reminder'")).firstMatch
        XCTAssertTrue(newReminderBtn.waitForExistence(timeout: 5), "New Reminder missing: \(reminders.debugDescription)")
        newReminderBtn.tap()
        Thread.sleep(forTimeInterval: 1.0)

        // Focus text field if keyboard isn't automatically shown yet
        if !reminders.keyboards.firstMatch.waitForExistence(timeout: 2) {
            if let field = findElement(reminders, matching: "elementType == %d OR elementType == %d", timeout: 2) {
                field.tap()
                Thread.sleep(forTimeInterval: 0.8)
            } else if reminders.textFields.firstMatch.exists {
                reminders.textFields.firstMatch.tap()
                Thread.sleep(forTimeInterval: 0.8)
            } else if reminders.textViews.firstMatch.exists {
                reminders.textViews.firstMatch.tap()
                Thread.sleep(forTimeInterval: 0.8)
            }
        }

        // Switch keyboard via Globe key to Komorebi 語音鍵盤
        for _ in 0..<5 {
            if reminders.buttons["語音輸入"].waitForExistence(timeout: 1.5) {
                break
            }
            let globe = reminders.buttons.matching(NSPredicate(format: "label == '下一個鍵盤' OR label == 'Next keyboard'")).firstMatch
            if globe.waitForExistence(timeout: 2) {
                globe.tap()
                Thread.sleep(forTimeInterval: 1.0)
            } else {
                break
            }
        }

        XCTAssertTrue(reminders.buttons["語音輸入"].waitForExistence(timeout: 5), "Komorebi keyboard did not appear in Reminders: \(reminders.debugDescription)")
        XCTAssertTrue(reminders.staticTexts["Komorebi 就緒"].waitForExistence(timeout: 5), "IPC did not report Komorebi 就緒 (Full Access / App Group): \(reminders.debugDescription)")
        Thread.sleep(forTimeInterval: 1.0)

        // Tap 語音輸入 button right inside Reminders' keyboard extension!
        reminders.buttons["語音輸入"].tap()
        Thread.sleep(forTimeInterval: 2.0)

        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        let micAllow = springboard.alerts.buttons.matching(NSPredicate(format: "label == '允許' OR label == 'Allow' OR label == '好' OR label == 'OK'")).firstMatch
        if micAllow.waitForExistence(timeout: 3) {
            micAllow.tap()
            Thread.sleep(forTimeInterval: 1.0)
        }

        // Verify Komorebi host app entered 聆聽中!
        let listeningLabel = komorebi.staticTexts.matching(NSPredicate(format: "label CONTAINS '聆聽中'")).firstMatch
        XCTAssertTrue(listeningLabel.waitForExistence(timeout: 6), "KomorebiVoice did not start listening from komorebi://listen: \(komorebi.debugDescription)")
        Thread.sleep(forTimeInterval: 1.2)

        // Switch back to Reminders while KomorebiVoice stays listening in background!
        reminders.activate()
        Thread.sleep(forTimeInterval: 1.2)

        // Reminders cancels an empty unsaved reminder when leaving the app; tap + if needed to show keyboard again
        if !reminders.buttons["停止聆聽"].waitForExistence(timeout: 2) {
            let addAgain = reminders.buttons.matching(NSPredicate(format: "label CONTAINS '新增提醒事項' OR label CONTAINS 'New Reminder'")).firstMatch
            if addAgain.waitForExistence(timeout: 3) {
                addAgain.tap()
                Thread.sleep(forTimeInterval: 1.0)
            }
            if !reminders.buttons["停止聆聽"].waitForExistence(timeout: 2) && !reminders.buttons["語音輸入"].exists {
                let globe = reminders.buttons.matching(NSPredicate(format: "label == '下一個鍵盤' OR label == 'Next keyboard'")).firstMatch
                if globe.waitForExistence(timeout: 2) {
                    globe.tap()
                    Thread.sleep(forTimeInterval: 1.0)
                }
            }
        }

        // Verify Komorebi keyboard inside Reminders shows live background listening status!
        XCTAssertTrue(reminders.buttons["停止聆聽"].waitForExistence(timeout: 6), "Keyboard inside Reminders should reflect 停止聆聽: \(reminders.debugDescription)")
        XCTAssertTrue(reminders.staticTexts["聆聽中（Komorebi）"].waitForExistence(timeout: 4))
        Thread.sleep(forTimeInterval: 1.0)

        // Toggle Noise Suppression & Draft Mode directly from inside Reminders' keyboard extension (IPC check)
        if reminders.buttons["降噪ON"].waitForExistence(timeout: 2) {
            reminders.buttons["降噪ON"].tap()
            Thread.sleep(forTimeInterval: 0.7)
            XCTAssertTrue(reminders.buttons["原音"].waitForExistence(timeout: 3))
            reminders.buttons["原音"].tap()
            Thread.sleep(forTimeInterval: 0.7)
            XCTAssertTrue(reminders.buttons["降噪ON"].waitForExistence(timeout: 3))
        }

        // Stop listening cleanly right from Reminders' keyboard!
        reminders.buttons["停止聆聽"].tap()
        Thread.sleep(forTimeInterval: 1.0)
        XCTAssertTrue(reminders.buttons["語音輸入"].waitForExistence(timeout: 4))
        XCTAssertTrue(reminders.staticTexts["Komorebi 就緒"].waitForExistence(timeout: 4))
        Thread.sleep(forTimeInterval: 1.0)
    }
}
