# 部署指南

從一台全新的 Mac 到 iPhone 上能用 Komorebi 小鍵盤，照順序做下面幾步。用到的版本與相依套件見 [README 的「相依套件」](../README.md#相依套件)。

## 0. 需要準備

* 一台 Apple silicon Mac（macOS 27），裝好 Xcode 27.2（含 iOS 27 platform）。
* 一支有 Neural Engine 的 iPhone（實測 iPhone 14 Pro Max / A16 / iOS 27.2），至少 4 GB 可用空間。
* 一個 Apple ID（免費的就可以）。
* 約 3.5 GB 的模型下載量（建議 Wi-Fi，或先下載到 Mac 再用 USB 傳）。

## 1. 安裝開發工具

```bash
xcode-select -p                # 應指向 /Applications/Xcode.app/Contents/Developer
sudo xcode-select -s /Applications/Xcode.app   # 如果不是
brew install xcodegen          # 由 project.yml 產生 Xcode 專案
```

第一次開 Xcode 時讓它裝完 iOS platform（Xcode → Settings → Components）。

## 2. 取得原始碼

```bash
git clone git@github.com:AyaSakura-comp/komorebi-ios.git
cd komorebi-ios
```

## 3. 簽章設定

1. Xcode → Settings → Apple Accounts，登入你的 Apple ID（密碼與雙重認證要自己輸入）。
2. 找出你的 **Team ID**。最可靠的方法是看開發憑證的 `OU` 欄位：

   ```bash
   security find-certificate -c "Apple Development" -p | openssl x509 -noout -subject
   # subject=UID=…, CN=Apple Development: 你的名字 (XXXXXXXXXX), OU=ABCDE12345, …
   ```

   `OU=` 後面那串才是 Team ID；`CN` 括號裡的不是（拿它當 Team ID 會出現「No Account for Team」）。如果還沒有憑證，先在 Xcode 隨便用自動簽章建置一次任何 App 到手機上，Xcode 就會建立。
3. 把 `project.yml` 的 `DEVELOPMENT_TEAM` 換成你的 Team ID。

### 不是原作者的話：換掉 Bundle ID 與 App Group

Bundle ID 與 App Group 在 Apple 的帳號系統裡是全域唯一的，`dev.ayasakura.*` 已經被註冊，你要換成自己的前綴（例如 `com.yourname`）。需要改的地方：

| 檔案 | 內容 |
|---|---|
| `project.yml` | `bundleIdPrefix`、四個 target 的 `PRODUCT_BUNDLE_IDENTIFIER`、兩處 `com.apple.security.application-groups` |
| `Komorebi.entitlements`、`KeyboardExtension/KomorebiKeyboard.entitlements` | App Group |
| `Sources/KomorebiIPC/KomorebiIPC.swift` | `KomorebiAppConfig.appGroupID`（以及 Darwin 通知名稱，可不改） |

一次換完：

```bash
NEW=com.yourname
grep -rl "dev.ayasakura" project.yml *.entitlements KeyboardExtension Sources App \
  | xargs sed -i '' "s/dev\.ayasakura/${NEW}/g"
```

主 App 的 Bundle ID 會變成 `${NEW}.komorebi`、鍵盤是 `${NEW}.komorebi.keyboard`、App Group 是 `group.${NEW}.komorebi`；下面指令裡的 `dev.ayasakura.komorebi` 也要跟著換。

## 4. 準備 iPhone

1. 用 USB 接上 Mac，在 iPhone 上按「信任這部電腦」。
2. 打開開發者模式：設定 → 隱私權與安全性 → 開發者模式 → 開啟，重新開機後確認。
3. 確認 Mac 看得到手機：

   ```bash
   xcrun devicectl list devices
   # My-iPhone   00008120-XXXXXXXXXXXXXXXX (UDID)   connected   iPhone 14 Pro Max
   ```

   下面用 `UDID=` 代表這支手機的 UDID。

## 5. 建置與安裝

先產生 Xcode 專案（`project.yml` 有改就要重跑）：

```bash
xcodegen generate
```

**用 Xcode**：打開 `KomorebiVoice.xcodeproj`，scheme 選 `KomorebiVoice`、目標選你的 iPhone，按 Run。

**用命令列**：

```bash
UDID=00008120-XXXXXXXXXXXXXXXX
xcodebuild -project KomorebiVoice.xcodeproj -scheme KomorebiVoice -configuration Release \
  -destination "id=${UDID}" -derivedDataPath build/device -allowProvisioningUpdates build
xcrun devicectl device install app --device "${UDID}" build/device/Build/Products/Release-iphoneos/KomorebiVoice.app
xcrun devicectl device process launch --device "${UDID}" dev.ayasakura.komorebi
```

第一次裝免費帳號簽的 App 時，iPhone 會拒絕開啟：到 設定 → 一般 → VPN 與裝置管理 → 你的開發者 App → 信任。

## 6. 放模型

App 需要的檔案與它們在 `Models/` 底下的位置：

```text
Models/
├── sherpa-models/
│   ├── silero_vad.onnx
│   └── sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/
│       ├── model.int8.onnx
│       └── tokens.txt
├── gemma-4-e2b/
│   └── gemma-4-E2B_q4_0-it.gguf
└── gemma-4-e2b-assistant/          # MTP drafter，建議放（解碼快 1.1–2.7 倍）
    ├── config.json
    └── model.safetensors
```

三種放法擇一：

### A. App 內直接下載（最簡單）

App 底部「模型與效能」頁 → 「一鍵下載核心模型」，再按 MTP drafter 那一列的「下載」。來源是 Hugging Face／GitHub，全部公開免登入。

### B. 從 Mac 走區網（Mac 上已經有模型時）

先在 Mac 上把模型放成 `~/models/` 底下同樣的結構（下載指令見 C 的第一段），然後：

```bash
bash scripts/serve-models.sh        # 會印出 http://<Mac 的 IP>:8765
```

在 App「模型與效能」頁的「區網 Mac 伴隨伺服器」填入這個網址，再按下載。

### C. 用 USB 直接複製（最快）

先下載到 Mac：

```bash
M=~/models
mkdir -p $M/sherpa-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 $M/gemma-4-e2b $M/gemma-4-e2b-assistant
curl -L -o $M/sherpa-models/silero_vad.onnx \
  https://raw.githubusercontent.com/snakers4/silero-vad/v6.2/src/silero_vad/data/silero_vad.onnx
SV=https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main
curl -L -o $M/sherpa-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx $SV/model.int8.onnx
curl -L -o $M/sherpa-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt $SV/tokens.txt
curl -L -o $M/gemma-4-e2b/gemma-4-E2B_q4_0-it.gguf \
  https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B_q4_0-it.gguf
for f in config.json model.safetensors; do
  curl -L -o $M/gemma-4-e2b-assistant/$f https://huggingface.co/google/gemma-4-E2B-it-assistant/resolve/main/$f
done
```

再複製進 App Group 容器（App 至少要開過一次）：

```bash
UDID=00008120-XXXXXXXXXXXXXXXX
cd ~/models
for f in sherpa-models/silero_vad.onnx \
         sherpa-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx \
         sherpa-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
         gemma-4-e2b/gemma-4-E2B_q4_0-it.gguf \
         gemma-4-e2b-assistant/config.json gemma-4-e2b-assistant/model.safetensors; do
  xcrun devicectl device copy to --device "$UDID" \
    --domain-type appGroupDataContainer --domain-identifier group.dev.ayasakura.komorebi \
    --source "$f" --destination "Models/$f"
done
```

也可以用 Finder：iPhone → 檔案 → Komorebi，把整個 `Models` 資料夾拖進去（會放在 App 的 Documents，App 也會找那裡）。

## 7. 啟用小鍵盤

設定 → 一般 → 鍵盤 → 鍵盤 → 新增鍵盤 → Komorebi；再點 Komorebi → 打開「**允許完整取用**」。沒開完整取用的話，鍵盤和主 App 之間不能傳資料，鍵盤上會顯示提示。

## 8. 第一次啟動

打開 Komorebi App。上方狀態會依序顯示：

1. 「準備 ANE：ANE program n/35」：把 GGUF 的權重轉成 ANE 的 4-bit 查表格式並編譯 37 個程式。A16 上約 **72 秒**，只有第一次。
2. 「**就緒：ANE direct (LUT4, MTP×3)**」：完成。之後每次啟動約 2–2.6 秒：轉好的權重在 App 的 `Library/Application Support/komorebi-ane/`，編譯結果由系統的 aned 保留並直接載入，37 個程式平行載入。
3. 就緒後在背景做約 0.5 秒的 ANE 暖機（每個程式載入後第一次執行比較慢）；這段時間可以直接開始說話，第一句的整理會等暖機做完。

如果顯示「就緒：僅語音辨識（Gemma 未載入：…）」，代表 ANE 引擎沒有建起來（模型不完整、私有 API 不存在等），括號裡是原因；這時仍可語音輸入，但句子不經 Gemma 整理。

## 9. 使用

1. 在任何 App 的輸入框切到 Komorebi 鍵盤（地球鍵）。
2. 點「語音輸入」。iOS 不讓鍵盤擴充使用麥克風，也只允許 App 在前景開始錄音，所以麥克風關著時會切到 Komorebi App 開麥克風，再自動切回原本的 App（沒有切回來的話，點左上角「◀ 返回」）。
   停止聆聽後麥克風會**待命**（預設 10 分鐘，「模型與效能」頁可改成 1／5／10／30 分鐘或不待命）：這段時間內再點「語音輸入」直接開始，不會跳轉。待命時鍵盤狀態顯示「麥克風待命中」，按「關麥」可以馬上關掉；狀態列的麥克風指示燈會亮著，但聲音不會被處理。來電或其他 App 搶走麥克風時待命會結束。
3. 直接說話。**邊說邊整理**：說話時鍵盤的預覽列每隔約半秒就顯示 Gemma 先整理好的這一句（之後新聽到的字以淡色接在後面）；停頓後整句會合併進草稿，**草稿就顯示在輸入框裡**。之後說的話會合併進同一份草稿，說「不對，改成…」可以改前面的內容。「模型與效能」頁可以關掉邊說邊整理（關掉時預覽列顯示辨識原文）。
4. 鍵盤按鈕：
   * **⌫**：刪字，草稿會同步（刪掉的字不會被下一句帶回來）。
   * **斷句**（聆聽中）：不等停頓，立刻整理剛剛說的話。
   * **定稿**：輸入框的文字保留，之後說的話另起新草稿。在聊天 App 按送出、輸入框被清空時會自動另起新草稿。
   * **↵**：換行並另起新草稿。
   * **停止聆聽**：停止錄音。

## 10. 驗證與實機測速

手機要解鎖才能從 Mac 啟動 App：

```bash
UDID=00008120-XXXXXXXXXXXXXXXX
# 完整測速：一般解碼 vs MTP、prefill、每個 ANE 程式的時間；結束後自動關閉
xcrun devicectl device process launch --console --terminate-existing --device "$UDID" \
  dev.ayasakura.komorebi --ane-bench --exit-after
# 連續 60 秒測速，每行附上前景/背景、溫度、低耗電模式狀態
xcrun devicectl device process launch --console --terminate-existing --device "$UDID" \
  dev.ayasakura.komorebi --ane-bench --bg-loop 60 --exit-after
# 麥克風待命自我測試：前景開始→停止（待命）→看到 READY_FOR_BACKGROUND 後切到別的 App→
# 由 App Group 送出鍵盤的開始/停止/關麥指令，檢查背景也收得到聲音
xcrun devicectl device process launch --console --terminate-existing --device "$UDID" \
  dev.ayasakura.komorebi --mic-selftest --exit-after
```

邊說邊整理的自我測試（用合成語音照說話速度餵進辨識，列出預覽列每次的內容與草稿出現的時間）：

```bash
xcrun devicectl device process launch --console --terminate-existing --device "$UDID" \
  dev.ayasakura.komorebi --preview-selftest --exit-after
```

App 會把麥克風工作階段的事件（開始、自動返回結果、待命、背景開始、被中斷）寫進 App 的 `Documents/komorebi-diag.log`，可以這樣拿回 Mac：

```bash
xcrun devicectl device copy from --device "$UDID" --domain-type appDataContainer \
  --domain-identifier dev.ayasakura.komorebi --source Documents/komorebi-diag.log --destination ./komorebi-diag.log
```

正常結果（A16）大約是：`ANE_BENCH build: 1–2 s`、一般解碼 26–28 tok/s、`MTP draft` 約 70 tok/s。

Mac 上（Apple silicon）也能跑同一個 ANE 引擎的測試，模型放在 `~/models/`（結構同第 6 步）：

```bash
swift test -c release --filter KomorebiCoreTests        # 全部：tokenizer、ANE 引擎、長文 20K、MTP、refiner
swift test -c release --filter ANEDirectTests/testANEDirectLongContext   # 20K token prefill + 256 token decode
cd Tools/ane-direct && make && ./probe t1               # ANE 編譯器/頻寬探測工具（說明見該目錄 README）
```

第一次在 Mac 跑會花幾十秒轉權重，快取在 `~/Library/Caches/komorebi-ane/`。測試可用環境變數調整：`ANE_BITS`（LUT 位元數 4／6／8，預設 4）、`ANE_GROUP`（每幾列共用一張查表，預設 8）、`ANE_CHUNK`（prefill chunk，預設 64）、`ANE_LONG_TOKENS`／`ANE_DECODE_TOKENS`（長文測試的提示長度與解碼長度，預設 20000／256）、`MTP_MAX_NEW`（MTP 測試每題最多產生幾個 token，預設 128）、`FH_DIAG=1`（跑 FlashHead 分群診斷）。

## 11. 維護

* **7 天重簽**：免費 Apple ID 的描述檔 7 天到期，到期後 App 打不開；重新做第 5 步即可，模型與 ANE 快取都保留。
* **iOS 更新後**：系統的編譯快取可能被清掉，第一次啟動又會花約 72 秒重新編譯。
* **換模型或改 ANE 引擎的權重格式**：快取目錄名稱含 LUT 位元數、分組、chunk 與 GGUF 檔案大小，換了就會自動重建。

## 12. 疑難排解

| 狀況 | 原因與處理 |
|---|---|
| Xcode 顯示「No Accounts」或「No Account for Team」 | Xcode 沒登入，或 `DEVELOPMENT_TEAM` 填錯（填成憑證 `CN` 括號裡的 ID）。照第 3 步重查 `OU`。 |
| `devicectl` 回報 `device was not, or could not be, unlocked` | 手機鎖著，解鎖後再執行。 |
| `xcrun devicectl list devices` 顯示 `unavailable` | 手機沒接上或沒信任這台 Mac；重插 USB、解鎖、按信任。 |
| App 打不開，提示「未受信任的開發者」 | 第 5 步最後：VPN 與裝置管理 → 信任。 |
| 狀態「就緒：僅語音辨識（Gemma 未載入：…）」 | 看括號裡的原因：GGUF 不在、私有 ANE API 不可用（iOS 版本不同）、模擬器。 |
| 鍵盤顯示「需要『允許完整取用』」 | 第 7 步打開完整取用。 |
| 點「語音輸入」每次都會切到 App | 只有麥克風關著時才會切（iOS 只允許前景開始錄音）。檢查「模型與效能」的「停止後麥克風待命」不是「不待命」；來電、Siri 或其他 App 用了麥克風也會結束待命。 |
| 切到 App 後沒有自動切回原本的 App | 自動返回用的是私有 API，iOS 不支援時會記在 `komorebi-diag.log`（`return to the previous app: …`）；點左上角「◀ 返回」即可，之後待命期間不會再切。 |
| 第一次啟動很久 | 正常：正在轉權重與編譯（A16 約 72 秒），之後約 2–2.6 秒。 |
