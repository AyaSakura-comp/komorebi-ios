import XCTest

final class KomorebiWalkthroughUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    @MainActor
    func testFullAppWalkthroughAndChecks() throws {
        let app = XCUIApplication(bundleIdentifier: "dev.ayasakura.komorebi")
        app.launch()

        // 1. 寫字桌 Tab Check
        XCTAssertTrue(app.staticTexts["木漏れ日"].waitForExistence(timeout: 15))
        let readyLabel = app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH '就緒：'")).firstMatch
        XCTAssertTrue(readyLabel.waitForExistence(timeout: 20))
        Thread.sleep(forTimeInterval: 0.8)

        let noiseButton = app.buttons.matching(NSPredicate(format: "label CONTAINS '降噪 ON'")).firstMatch
        if noiseButton.waitForExistence(timeout: 5) {
            noiseButton.tap()
            Thread.sleep(forTimeInterval: 0.7)
            let rawButton = app.buttons.matching(NSPredicate(format: "label CONTAINS '原始收音'")).firstMatch
            XCTAssertTrue(rawButton.waitForExistence(timeout: 3))
            rawButton.tap()
            Thread.sleep(forTimeInterval: 0.7)
        }

        // Mode segmented switch: 草稿整合 <-> 逐句模式
        let sentenceMode = app.buttons["逐句模式"]
        if sentenceMode.waitForExistence(timeout: 3) {
            sentenceMode.tap()
            Thread.sleep(forTimeInterval: 0.6)
            app.buttons["草稿整合"].tap()
            Thread.sleep(forTimeInterval: 0.6)
        }

        // Type into editor to check font rendering, char counter, and keyboard dismiss toolbar
        let textView = app.textViews.firstMatch
        XCTAssertTrue(textView.waitForExistence(timeout: 5))
        textView.tap()
        Thread.sleep(forTimeInterval: 0.5)
        textView.typeText("Komorebi iOS")
        Thread.sleep(forTimeInterval: 0.8)
        XCTAssertTrue(app.staticTexts["12 字"].waitForExistence(timeout: 3))

        // Dismiss keyboard via our new UIToolbar "收起鍵盤" button so TabBar is reachable
        let deskDismiss = app.buttons["dismissKeyboardButton"]
        if deskDismiss.waitForExistence(timeout: 3) {
            deskDismiss.tap()
        } else {
            let fallbackDone = app.buttons["收起鍵盤"]
            XCTAssertTrue(fallbackDone.waitForExistence(timeout: 2))
            fallbackDone.tap()
        }
        Thread.sleep(forTimeInterval: 0.8)

        // 2. 語音鍵盤設定 & Komorebi 語音鍵盤 Extension 預覽 Check
        let keyboardTab = app.tabBars.buttons["語音鍵盤"]
        XCTAssertTrue(keyboardTab.waitForExistence(timeout: 5))
        keyboardTab.tap()
        Thread.sleep(forTimeInterval: 0.8)

        XCTAssertTrue(app.staticTexts["啟用系統「Komorebi 語音鍵盤」三步驟"].waitForExistence(timeout: 5))

        // Scroll slightly if needed and focus sandbox editor (`axis: .vertical` TextField -> TextView or TextField)
        let sandboxInput = app.textViews.firstMatch.exists ? app.textViews.firstMatch : app.textFields.firstMatch
        XCTAssertTrue(sandboxInput.waitForExistence(timeout: 4))
        sandboxInput.tap()
        Thread.sleep(forTimeInterval: 1.0)

        // Switch via globe ("下一個鍵盤" / "Next keyboard") to showcase Komorebi Keyboard extension
        let globe = app.buttons.matching(NSPredicate(format: "label == '下一個鍵盤' OR label == 'Next keyboard'")).firstMatch
        if globe.waitForExistence(timeout: 3) {
            globe.tap()
            Thread.sleep(forTimeInterval: 2.0)
        }

        // Dismiss keyboard cleanly via sandboxCard's "收起鍵盤" header button
        let sandboxDismiss = app.buttons["sandboxDismissKeyboard"]
        if sandboxDismiss.waitForExistence(timeout: 3) {
            sandboxDismiss.tap()
        } else if app.buttons["收起鍵盤"].exists {
            app.buttons["收起鍵盤"].tap()
        } else {
            app.staticTexts["啟用系統「Komorebi 語音鍵盤」三步驟"].tap()
        }
        Thread.sleep(forTimeInterval: 0.8)

        // 3. 模型與效能 Tab Check
        let modelTab = app.tabBars.buttons["模型與效能"]
        XCTAssertTrue(modelTab.waitForExistence(timeout: 5))
        modelTab.tap()
        Thread.sleep(forTimeInterval: 0.8)

        XCTAssertTrue(app.staticTexts["iOS 內建降噪與語音隔離"].waitForExistence(timeout: 5))
        Thread.sleep(forTimeInterval: 0.8)

        // Toggle VoiceProcessingIO switch off & back on
        let vpSwitch = app.switches.firstMatch
        XCTAssertTrue(vpSwitch.waitForExistence(timeout: 3))
        vpSwitch.tap()
        Thread.sleep(forTimeInterval: 0.6)
        vpSwitch.tap()
        Thread.sleep(forTimeInterval: 0.8)

        // Scroll down to model checklist & benchmark button
        let scroll = app.scrollViews.firstMatch
        if scroll.exists {
            scroll.swipeUp()
            Thread.sleep(forTimeInterval: 0.8)
        }

        // Wait briefly if models are still loading, then tap Benchmark
        let benchBtn = app.buttons["執行實機 Benchmark"]
        XCTAssertTrue(benchBtn.waitForExistence(timeout: 5))
        if benchBtn.isEnabled {
            benchBtn.tap()
            let resultLabel = app.staticTexts.matching(NSPredicate(format: "label CONTAINS '整理結果：'")).firstMatch
            XCTAssertTrue(resultLabel.waitForExistence(timeout: 90))
            if scroll.exists {
                scroll.swipeUp()
            }
            Thread.sleep(forTimeInterval: 2.0)
        }

        // Return to 寫字桌 tab cleanly
        app.tabBars.buttons["寫字桌"].tap()
        Thread.sleep(forTimeInterval: 1.5)
    }
}
