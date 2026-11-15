# 木漏れ日 · Komorebi 語音輸入法 for iOS

完全離線的 iPhone 語音輸入法：在任何 App 切到 Komorebi 小鍵盤，講話就會出字，Gemma 4 E2B 邊聽邊整理（去贅字、套用改口、補標點；「不對，改成四點」會改到前面的句子），停頓後整句併進草稿，草稿直接顯示在輸入框裡。

App 裡還有一個「聊天」分頁：跟傲嬌的「小葉」語音聊天（Komorebi 聊天 demo 的 iPhone 版），對話有自己的 KV cache，每一句只 prefill 新的部分，上下文滿了會把最舊的對話整理成重點，KV cache 存在手機上，App 重開也不用重算。

語音辨識跟桌面版 Komorebi 一樣分兩段：講話時的即時字幕用 CPU 上的 SenseVoice（Silero VAD 斷句），一句講完後整句交給 **Qwen3-ASR-1.7B** 重新辨識（台灣口音中英夾雜的錯字約少一半）。**Qwen3-ASR 和 Gemma 4 E2B 都只跑在 Apple Neural Engine 上**：這個 repo 自己實作了直接呼叫 ANE 的推論引擎（私有 `AppleNeuralEngine.framework` + MIL 程式，不經過 Core ML），前景、背景一樣快，也不用 GPU，手機不會燙。

> 用到 Apple 私有 API，只能自己簽章安裝（Xcode／免費 Apple ID 即可），不能上 App Store。

## 效能

iPhone 14 Pro Max（A16，iOS 27.2），LUT4 權重、小字典 LM head、MTP 推測解碼：

| 情境 | 一般解碼 | MTP 推測解碼 |
|---|---|---|
| Komorebi 草稿改寫（330 token 提示） | 26.1 tok/s | **69.3 tok/s**（×2.65，草稿接受率 82 %） |
| 英文說明 | 26.8 tok/s | 45.7 tok/s（×1.70） |
| 中文自由寫作 | 27.3 tok/s | 29.6 tok/s（×1.08） |

* Prefill：186–330 token 的提示約 **2100–2330 tok/s**。
* 啟動：第一次要把權重轉成 ANE 格式並編譯，約 72 秒；之後直接載入系統（aned）保留的編譯結果，App 啟動到「就緒」約 2–2.6 秒，ANE 暖機（0.5 秒）在就緒之後背景進行。
* App 記憶體約 180 MB（ANE 權重放在 App 行程外）。
* 背景不變慢：在別的 App 用小鍵盤時（螢幕亮著），速度和前景相同。

Mac（M4）上同一個引擎：一般解碼 40.6 tok/s、prefill 3090 tok/s，MTP 草稿改寫最高 107.5 tok/s。

### 語音辨識（Qwen3-ASR-1.7B on ANE）

台灣口音中英夾雜測試集（[ky552/ML2021_ASR_ST](https://huggingface.co/datasets/ky552/ML2021_ASR_ST) test 隨機 150 句，375 秒；MER＝中文字＋英文字一起算的錯誤率，簡繁、他/她/它、數字先正規化）：

| 辨識器 | MER | 英文 WER | 速度 |
|---|---|---|---|
| SenseVoice-Small int8（即時字幕） | 15.97 % | 57.9 % | 每次約 50 ms（CPU） |
| Qwen3-ASR-1.7B，MLX 8-bit（Mac GPU，參考） | 8.50 % | 20.3 % | RTF 0.15（M4 GPU，指定中文；見下表） |
| **Qwen3-ASR-1.7B，本 repo 的 ANE 引擎**（解碼器 LUT6、編碼器 LUT8，預設） | **8.84 %** | **20.7 %** | RTF 0.12（M4）：編碼器 12 ms、prefill 35–85 ms、解碼約 34 tok/s |
| 同上，解碼器 LUT8 | 8.36 % | 21.1 % | RTF 0.15（M4）、解碼約 27 tok/s，ANE 權重多 0.35 GB |

Mac mini M4 上一句要多久：同一批句子，都指定中文、暖機後量。範圍是兩種量法的結果：先整批跑完一邊再跑另一邊，以及每句在 ANE 和 GPU 上輪流各跑一次（排除被背景工作卡住的句子）：

| 句長 | ANE（本 repo，預設 LUT6） | Metal GPU（MLX 8-bit） | GPU／ANE |
|---|---|---|---|
| 2.5 秒（約 8 個 token） | **0.29–0.31 秒** | 0.38–0.43 秒 | 1.34–1.37 倍 |
| 5 秒（約 17 個 token） | **0.61–0.64 秒** | 0.72–0.80 秒 | 1.13–1.31 倍 |
| 10 秒（約 33 個 token） | **1.2–1.4 秒** | 1.5–1.6 秒 | 1.09–1.23 倍 |

* 功耗（powermetrics，轉寫時比待機多出來的部分）：MLX 跑時 GPU 多 8–10 W，每句多耗 3.4 J（2.5 秒）到 12.7 J（10 秒）；ANE 引擎只多 0.2–1.5 W（ANE）。兩邊 CPU 的差異都在背景負載的波動範圍內，量不出來。
* 時間幾乎都花在解碼（暖機後約 30–36 tok/s），編碼器 11–22 ms、prefill 40–110 ms。解碼時 embedding 和重新評分用的列直接從 GGUF 讀，第一次碰到的分頁要從磁碟載入；GGUF 放在外接 exFAT 碟上時，第一輪只有 16–19 tok/s。
* 重跑方式：`Qwen3ASRTests/testInterleavedWithMLX`（每句輪流）、`testBakeoffSet` 加 `QWEN3_ASR_CONCAT=1,2,4`，以及 [`Tools/qwen3-asr/latency_mlx.py`](Tools/qwen3-asr/latency_mlx.py)，見 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) 第 10 節。

* iPhone 14 Pro Max（A16），Release 版，用測試集前 30 句（1.6–4.5 秒）：
  * 速度：每句平均 0.39 秒（p90 0.52 秒、最長 0.56 秒，RTF 0.15），編碼器 15 ms、prefill 45–85 ms、解碼約 26 tok/s。
  * 正確率：MER 5.2 %，SenseVoice 是 13.9 %。辨識出來的文字跟 Mac 上同一個引擎 30 句全部相同。
  * 實際說話流程（`--preview-selftest`）：Gemma 同時在 ANE 上整理草稿，所以解碼降到 18–19 tok/s。7.7 秒的句子要 1.6 秒，2.4 秒的句子要 0.53 秒。
  * 記憶體：App 從 500 MB 變成約 570–600 MB。
  * 第一次載入：每次重新安裝後第一次開啟，都要在手機上重新編譯 35 個 ANE 程式（約 90 秒，期間整句先用 SenseVoice；Gemma 的程式也要重編，就緒從 2.7 秒變成 14 秒）。之後從系統快取載入，Qwen3-ASR 約 4 秒就緒。
* 解碼器用 4-bit 查表會完全壞掉（MER > 100 %，一直重複同一個字），6-bit 是下限。
* 位址空間：免費 Apple ID 不能開 Extended Virtual Addressing，App 只有幾 GB 的位址空間。兩個模型的檔案和 ANE 權重全部 map 進來（約 8 GB）會配置失敗，所以 ANE 程式載入後就放掉權重 blob，Qwen3 的 GGUF 只 map 檔頭和 embedding 表，程式一次載入三個。
* 指定語言為中文（跟桌面版 Komorebi 對 TEA 的做法一樣）和讓模型自己判斷的 MER 相同，英文句子照樣輸出英文，每句少解碼 3 個 token。詳細量測與 ANE 編譯器的實測限制見 [`Tools/ane-direct/README.md`](Tools/ane-direct/README.md)。

## 架構

```text
任何 App 的輸入框
   ▲  insertText / deleteBackward（草稿差異更新）
   │
KomorebiKeyboard（鍵盤擴充，不能用麥克風）
   │  App Group 檔案 + Darwin 通知（KomorebiIPC）
   ▼
KomorebiVoice（主 App，背景音訊模式保持錄音）
   麥克風 16 kHz → Silero VAD（斷句）→ SenseVoice-Small int8（sherpa-onnx，CPU，講話中每 0.25 秒的即時字幕）
                → 一句講完：Qwen3ASRANEEngine（Apple Neural Engine）整句重新辨識 → OpenCC 繁體
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

### ANE Qwen3-ASR 引擎（`Sources/KomorebiCore/Qwen3ASR.swift`）

* **權重**：`Tools/qwen3-asr/convert_gguf.py` 把 `Qwen/Qwen3-ASR-1.7B`（bf16）轉成 Q8_0 GGUF（2.03 GiB）；第一次載入時再轉成 ANE 查表（解碼器與 LM head 6-bit、音訊編碼器 8-bit，共 1.6 GB），快取在 `komorebi-ane/qwen3asr-…/`。查表的 k-means 從「密度的立方根」分佈起始（高解析度量化的最佳分配），8-bit 的誤差從 2.5–4.7 % 降到 1.0–1.5 %，6-bit 從 3.9–7 % 降到 3.0–4.3 %。
* **CPU 端**：Whisper log-mel（400 點 DFT 用 GEMM 算，和 `WhisperFeatureExtractor` 誤差 2e-5）、token 嵌入（Q8_0）、解碼器的 GQA 注意力與 KV cache、LM head 前 16 名用 Q8_0 原始權重精算、byte-level BPE 解碼。
* **音訊編碼器**：卷積前端（3 層 stride-2 conv + 線性層 + 位置編碼）一個程式、24 層 transformer 分四個程式（最後一個含 ln_post 與投影到 2048 維）；一次處理一個 8 秒的注意力視窗（8 個 1 秒 chunk、104 個 token），不足的部分用遮罩蓋掉。
* **解碼器**：Qwen3 1.7B（28 層）切成 29 個邊界程式（第 l-1 層的 O 投影 + SwiGLU FFN，接第 l 層的 RMSNorm + QKV + q/k norm + RoPE），各有 decode（S=1）與 prefill（S=64）兩個入口；LM head 是完整 151,936 個 token。
* **提示詞**：跟原版一樣 `<|im_start|>system … <|audio_start|><|audio_pad|>×N<|audio_end|> … assistant\nlanguage Chinese<asr_text>`，音訊 token 位置直接放編碼器輸出。

| 檔案 | 內容 |
|---|---|
| `Sources/KomorebiCore/Qwen3ASR.swift` | Qwen3-ASR：設定、log-mel、BPE 解碼、編碼器與解碼器的 MIL 產生、建置與快取、轉寫流程、GQA 注意力 |
| `Sources/KomorebiCore/ANEMIL.swift` | 兩個引擎共用：MIL 文字產生器、權重 blob 與 LUT 查表、k-means 量化 |
| `Tools/qwen3-asr/` | `make_reference.py`（原版模型各階段的參考輸出）、`convert_gguf.py`（轉 GGUF） |
| `App/Speech/Qwen3FinalASR.swift` | App 端：整句辨識（指定中文、OpenCC 繁體），失敗時退回 SenseVoice |
| `Sources/KomorebiANEDirect/ane_rt.{h,m}` | 私有 ANE API 的 ObjC 包裝：MIL + 權重 blob 編譯、載入、多入口 procedure、IOSurface 輸入輸出、eval |
| `Sources/KomorebiANEDirect/komorebi_kernels.{h,c}` | NEON 小核心（int8 / bf16 內積、bf16 GEMV），給 LM head 精算與 MTP drafter 用 |
| `Sources/KomorebiCore/Gemma4ANEDirectEngine.swift` | MIL 產生、LUT 量化、程式建置與快取、forward、注意力、推測解碼迴圈、profiling |
| `Sources/KomorebiCore/KVSession.swift` | 一段對話的 KV cache：最長共同前綴重用、多個 session 共用一個引擎、fp16 存檔與讀回 |
| `Sources/KomorebiCore/Gemma4MTPDrafter.swift` | MTP drafter（safetensors mmap，bf16） |
| `Sources/KomorebiCore/FlashHead.swift` | 小字典建立；另含實驗性的 k-means FlashHead（實測不適合 Gemma 的 tied head，預設不用） |
| `Sources/KomorebiCore/GGUFLoader.swift`、`Gemma4Config.swift`、`Gemma4Tokenizer.swift` | GGUF v3 mmap 讀取、Q4_0 / Q6_K / Q8_0 解量化、模型設定、Gemma 4 tokenizer |
| `Sources/KomorebiCore/KomorebiRefiner.swift`、`KomorebiPrompts.swift` | 整句整理、草稿合併、圈選改寫的提示詞與接受條件 |
| `Sources/KomorebiIPC/KomorebiIPC.swift` | 主 App ↔ 鍵盤的狀態、指令、草稿同步 |
| `App/` | 主 App：錄音與 VAD/ASR 管線、模型下載、編輯器、聊天（`App/Chat/`）、`--ane-bench` 實機測速 |
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

### 模型（不在 repo 裡，另外下載，共約 3.5 GB；加上 Qwen3-ASR 約 5.5 GB）

| 模型 | 來源 | 大小 | 放在 `Models/` 底下的路徑 |
|---|---|---|---|
| Silero VAD v6.2 | [snakers4/silero-vad](https://github.com/snakers4/silero-vad/blob/v6.2/src/silero_vad/data/silero_vad.onnx) | 2.3 MB | `sherpa-models/silero_vad.onnx` |
| SenseVoice-Small int8 | [csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17](https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17) | 228 MB | `sherpa-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/{model.int8.onnx,tokens.txt}` |
| Gemma 4 E2B it QAT Q4_0 | [google/gemma-4-E2B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf) | 3.12 GiB | `gemma-4-e2b/gemma-4-E2B_q4_0-it.gguf` |
| Gemma 4 E2B MTP drafter（建議） | [google/gemma-4-E2B-it-assistant](https://huggingface.co/google/gemma-4-E2B-it-assistant) | 150 MB | `gemma-4-e2b-assistant/{config.json,model.safetensors}` |
| Qwen3-ASR-1.7B（建議，整句辨識） | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) 用 `Tools/qwen3-asr` 轉檔 | 2.03 GiB | `qwen3-asr/qwen3-asr-1.7b-q8_0.gguf` |

`Models/` 預設在 App Group 容器裡（`group.dev.ayasakura.komorebi`）；App 也會找 App 的 `Documents/Models/`，Mac 測試則找 `~/models/`。

### 裝置

* 有 Neural Engine 的 iPhone。實測機：iPhone 14 Pro Max（A16）、iOS 27.2。部署目標是 iOS 17，但私有 ANE API 只在 iOS 27 上驗證過，其他版本可能不同。
* 約 4 GB 可用空間（模型 3.5 GB + ANE 權重快取約 1.1 GB）；加上 Qwen3-ASR 再多約 3.6 GB（模型 2 GB + 查表快取 1.6 GB）。
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
* 鍵盤擴充不能使用麥克風，而 iOS 只允許 App 在前景開始錄音，所以麥克風關著時點「語音輸入」會切到主 App 開麥克風，再自動切回原本的 App（用私有的 `_systemNavigationAction`；不支援時要手動返回）。停止聆聽後麥克風會待命一段時間（預設 10 分鐘，可在「模型與效能」調整），期間再點「語音輸入」不會跳轉；待命時狀態列會顯示麥克風使用中，但聲音不會被處理。
* 免費 Apple ID 的描述檔 7 天到期，到期後重新建置安裝即可（模型與 ANE 快取都會保留）。

## 致謝與授權

* 直接呼叫 ANE 的做法參考 [maderix/ANE](https://github.com/maderix/ANE)（MIT）。
* 模型各自適用其授權：Gemma（[Gemma Terms of Use](https://ai.google.dev/gemma/terms)）、Qwen3-ASR（Apache-2.0）、SenseVoice（FunASR 模型授權）、Silero VAD（MIT）。
* `App/Dictionary/` 的簡繁轉換字典來自 [OpenCC](https://github.com/BYVoid/OpenCC)（Apache-2.0）。
* 提示詞與草稿流程移植自 Komorebi 語音輸入法的伺服器版（`app/realtime_service.py`）。
