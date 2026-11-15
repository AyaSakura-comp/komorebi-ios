# 木漏れ日 · Komorebi 語音輸入法 for iOS

完全離線的 iPhone 語音輸入法：在任何 App 切到 Komorebi 小鍵盤，講話就會出字，再由 Gemma 4 E2B 去贅字、套用改口（「不對，改成四點」會改到前面的句子）、補標點，整理好的草稿直接顯示在輸入框裡。

語音辨識跑在 CPU（Silero VAD + SenseVoice），**Gemma 4 E2B 只跑在 Apple Neural Engine 上**：這個 repo 自己實作了一個直接呼叫 ANE 的推論引擎（私有 `AppleNeuralEngine.framework` + MIL 程式，不經過 Core ML），前景、背景一樣快，也不用 GPU，手機不會燙。

> 用到 Apple 私有 API，只能自己簽章安裝（Xcode／免費 Apple ID 即可），不能上 App Store。

## 效能

iPhone 14 Pro Max（A16，iOS 27.2），LUT4 權重、小字典 LM head、MTP 推測解碼：

| 情境 | 一般解碼 | MTP 推測解碼 |
|---|---|---|
| Komorebi 草稿改寫（330 token 提示） | 26.1 tok/s | **69.3 tok/s**（×2.65，草稿接受率 82 %） |
| 英文說明 | 26.8 tok/s | 45.7 tok/s（×1.70） |
| 中文自由寫作 | 27.3 tok/s | 29.6 tok/s（×1.08） |

* Prefill：186–330 token 的提示約 **2100–2330 tok/s**。
* 啟動：第一次要把權重轉成 ANE 格式並編譯，約 72 秒；之後系統（aned）會保留編譯結果，App 啟動到可用約 6 秒。
* App 記憶體約 180 MB（ANE 權重放在 App 行程外）。
* 背景不變慢：在別的 App 用小鍵盤時（螢幕亮著），速度和前景相同。

Mac（M4）上同一個引擎：一般解碼 40.6 tok/s、prefill 3090 tok/s，MTP 草稿改寫最高 107.5 tok/s。詳細量測與 ANE 編譯器的實測限制見 [`Tools/ane-direct/README.md`](Tools/ane-direct/README.md)。

## 架構

```text
任何 App 的輸入框
   ▲  insertText / deleteBackward（草稿差異更新）
   │
KomorebiKeyboard（鍵盤擴充，不能用麥克風）
   │  App Group 檔案 + Darwin 通知（KomorebiIPC）
   ▼
KomorebiVoice（主 App，背景音訊模式保持錄音）
   麥克風 16 kHz → Silero VAD（斷句）→ SenseVoice-Small int8（sherpa-onnx，CPU）→ OpenCC 繁體
                → KomorebiRefiner：Komorebi 提示詞（整句整理 / 草稿合併 / 圈選改寫）
                → Gemma4ANEDirectEngine（Apple Neural Engine）
```

### ANE Gemma 引擎（`Sources/KomorebiANEDirect` + `Sources/KomorebiCore`）

* **權重**：從 Gemma 4 E2B 的 Q4_0 GGUF 讀出，每 8 列做一次 1-D k-means，轉成 4-bit 查表（MIL `constexpr_lut_to_dense`，ANE 執行時解壓）。結果快取在 App 的 Application Support／`komorebi-ane/`。
* **程式切法**：36 個「邊界」程式，每個包含第 l-1 層的後半（O 投影 + FFN + PLE）和第 l 層的前半（RMSNorm + QKV），各有三個入口（decode S=1、verify S=4、prefill S=64）共用一份權重。
* **LM head**：只保留中、英、日與控制字元等 170,748 個 token 的小字典在 ANE 上算（LUT4），前 16 名再用 CPU 對 Q6_K 原始權重精算，結果與完整 LM head 一致。
* **CPU 端**：KV cache 與 MQA 注意力（Accelerate）、PLE 嵌入、RoPE、取 argmax；prefill 時共用 KV 的第 15–34 層只算最後一個 chunk。
* **MTP 推測解碼**：`google/gemma-4-E2B-it-assistant`（4 層、共用主模型 13/14 層的 KV、centroid head）在 CPU 上猜 1–3 個 token，主模型用 S=4 入口一次驗證（貪婪驗證，輸出與一般解碼相同）。
* **編譯快取**：`_ANEInMemoryModel.compiledModelExists` 有結果就跳過編譯，只載入。

| 檔案 | 內容 |
|---|---|
| `Sources/KomorebiANEDirect/ane_rt.{h,m}` | 私有 ANE API 的 ObjC 包裝：MIL + 權重 blob 編譯、載入、多入口 procedure、IOSurface 輸入輸出、eval |
| `Sources/KomorebiANEDirect/komorebi_kernels.{h,c}` | NEON 小核心（int8 / bf16 內積、bf16 GEMV），給 LM head 精算與 MTP drafter 用 |
| `Sources/KomorebiCore/Gemma4ANEDirectEngine.swift` | MIL 產生、LUT 量化、程式建置與快取、forward、注意力、推測解碼迴圈、profiling |
| `Sources/KomorebiCore/Gemma4MTPDrafter.swift` | MTP drafter（safetensors mmap，bf16） |
| `Sources/KomorebiCore/FlashHead.swift` | 小字典建立；另含實驗性的 k-means FlashHead（實測不適合 Gemma 的 tied head，預設不用） |
| `Sources/KomorebiCore/GGUFLoader.swift`、`Gemma4Config.swift`、`Gemma4Tokenizer.swift` | GGUF v3 mmap 讀取、Q4_0 / Q6_K 解量化、模型設定、Gemma 4 tokenizer |
| `Sources/KomorebiCore/KomorebiRefiner.swift`、`KomorebiPrompts.swift` | 整句整理、草稿合併、圈選改寫的提示詞與接受條件 |
| `Sources/KomorebiIPC/KomorebiIPC.swift` | 主 App ↔ 鍵盤的狀態、指令、草稿同步 |
| `App/` | 主 App：錄音與 VAD/ASR 管線、模型下載、編輯器、`--ane-bench` 實機測速 |
| `KeyboardExtension/` | 小鍵盤 |
| `Tests/KomorebiCoreTests/` | macOS（Apple silicon）上的 ANE 引擎、MTP、tokenizer 測試 |
| `AppTests/`、`UITests/` | 模擬器測試（模擬器沒有 ANE，只測語音辨識與 IPC） |
| `Tools/ane-direct/` | ANE 探測工具 `probe` 與所有實測紀錄 |
| `scripts/serve-models.sh` | 在 Mac 上用區網把模型傳給 iPhone |

## 相依套件

### 開發工具

| 項目 | 版本（實測） | 用途 |
|---|---|---|
| macOS | 27.0.1 | 建置；在 Apple silicon 上也能跑 ANE 測試 |
| Xcode | 27.2（27B52），含 iOS 27 SDK | 編譯、簽章 |
| XcodeGen | 2.46.0（`brew install xcodegen`） | 由 `project.yml` 產生 `KomorebiVoice.xcodeproj`（xcodeproj 不進版控） |
| Apple ID | 免費個人團隊即可 | 自動簽章；免費團隊的描述檔 7 天到期 |

### Swift 套件（`project.yml` 固定版本）

| 套件 | 版本 | 用途 |
|---|---|---|
| [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | 2.0.1 | SenseVoice 語音辨識、Silero VAD（帶入 [onnxruntime-libs](https://github.com/csukuangfj/onnxruntime-libs) 1.28.2） |
| [ddddxxx/SwiftyOpenCC](https://github.com/ddddxxx/SwiftyOpenCC) | 1.0.1 | 簡轉繁（s2twp）；字典在 `App/Dictionary/` |
| KomorebiKit（本 repo 的 `Package.swift`） | — | `KomorebiANEDirect`、`KomorebiCore`、`KomorebiIPC` |

### 系統框架

`Accelerate`、`IOSurface`、`AVFoundation`、`SwiftUI` / `UIKit`；執行時 `dlopen` 私有的 `/System/Library/PrivateFrameworks/AppleNeuralEngine.framework`（`_ANEInMemoryModelDescriptor`、`_ANEInMemoryModel`、`_ANERequest`、`_ANEIOSurfaceObject`）。不需要 Metal 或 Core ML。

### 模型（不在 repo 裡，另外下載，共約 3.5 GB）

| 模型 | 來源 | 大小 | 放在 `Models/` 底下的路徑 |
|---|---|---|---|
| Silero VAD v6.2 | [snakers4/silero-vad](https://github.com/snakers4/silero-vad/blob/v6.2/src/silero_vad/data/silero_vad.onnx) | 2.3 MB | `sherpa-models/silero_vad.onnx` |
| SenseVoice-Small int8 | [csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17](https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17) | 228 MB | `sherpa-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/{model.int8.onnx,tokens.txt}` |
| Gemma 4 E2B it QAT Q4_0 | [google/gemma-4-E2B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf) | 3.12 GiB | `gemma-4-e2b/gemma-4-E2B_q4_0-it.gguf` |
| Gemma 4 E2B MTP drafter（建議） | [google/gemma-4-E2B-it-assistant](https://huggingface.co/google/gemma-4-E2B-it-assistant) | 150 MB | `gemma-4-e2b-assistant/{config.json,model.safetensors}` |

`Models/` 預設在 App Group 容器裡（`group.dev.ayasakura.komorebi`）；App 也會找 App 的 `Documents/Models/`，Mac 測試則找 `~/models/`。

### 裝置

* 有 Neural Engine 的 iPhone。實測機：iPhone 14 Pro Max（A16）、iOS 27.2。部署目標是 iOS 17，但私有 ANE API 只在 iOS 27 上驗證過，其他版本可能不同。
* 約 4 GB 可用空間（模型 3.5 GB + ANE 權重快取約 1.1 GB）。
* 模擬器沒有 Neural Engine：App 能跑語音辨識，但 Gemma 不會載入，句子以辨識原文送出。

## 快速開始

```bash
brew install xcodegen
git clone git@github.com:AyaSakura-comp/komorebi-ios.git && cd komorebi-ios
# 先把 project.yml 的 DEVELOPMENT_TEAM 換成你自己的 Team ID
xcodegen generate
open KomorebiVoice.xcodeproj   # 選 KomorebiVoice scheme 與你的 iPhone，Run
```

接著在 App 的「模型與效能」頁下載模型、到「設定 → 一般 → 鍵盤」加入 Komorebi 並開啟「允許完整取用」。完整步驟（簽章、模型的三種傳法、命令列安裝、實機測速、疑難排解）見 **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**。

## 已知限制

* 私有 API：iOS 更新可能讓 ANE 引擎失效；失效時 App 會退回「僅語音辨識」，句子以辨識原文送出。
* 鍵盤擴充不能使用麥克風，所以第一次點「語音輸入」會切到主 App 開始錄音，再回到原本的 App 說話。
* 免費 Apple ID 的描述檔 7 天到期，到期後重新建置安裝即可（模型與 ANE 快取都會保留）。

## 致謝與授權

* 直接呼叫 ANE 的做法參考 [maderix/ANE](https://github.com/maderix/ANE)（MIT）。
* 模型各自適用其授權：Gemma（[Gemma Terms of Use](https://ai.google.dev/gemma/terms)）、SenseVoice（FunASR 模型授權）、Silero VAD（MIT）。
* 提示詞與草稿流程移植自 Komorebi 語音輸入法的伺服器版（`app/realtime_service.py`）。
