# Direct ANE (no CoreML) for Gemma 4 E2B

Runtime: `Sources/KomorebiANEDirect` (ObjC; private `AppleNeuralEngine.framework`:
`_ANEInMemoryModelDescriptor` → `_ANEInMemoryModel compile/load` → `_ANERequest` + IOSurfaces).
Approach adapted from github.com/maderix/ANE (MIT). Engine: `Sources/KomorebiCore/Gemma4ANEDirectEngine.swift`.
Probe / roofline tool: `Tools/ane-direct/probe.m` (`make && ./probe t1|t2|t3|t4 [formats...]`).

## What the ANE compiler accepts (M4, macOS 27 / iOS 27)
| Weight format (MIL) | Compiles | Runtime decompression | Notes |
|---|---|---|---|
| fp16 `const` | yes | n/a | ~64 GB/s weight streaming |
| `constexpr_lut_to_dense` uint4/6/8, LUT per group of 8–64 output rows | yes | **yes** (time scales with bits) | the fast path |
| `constexpr_lut_to_dense`, LUT per row (group 1) | yes | no (4.3 ms vs 1.0 ms) | slow, long compile |
| `constexpr_lut_to_dense`, LUT per K-block (exact Q4_0) | **no** | – | InvalidMILProgram |
| `constexpr_blockwise_shift_scale` int4/uint4+offset/int8 | **no** | – | InvalidMILProgram |

Other facts measured with `probe`:
- dispatch latency of a tiny program: **0.07 ms**
- multiple inputs/outputs work (maderix's "single input" limit does not apply here)
- multi-function programs work and share one weight blob; each procedure uses *global* symbol indices
  (`ANEFModelProcedures[i].ANEFModelInputSymbolIndexArray`), not 0..n-1
- `[1, S, C]` activations + `linear` give correct results for S = 1 and S = 64
- S = 1 and S = 64 cost almost the same time → decode is weight-bandwidth bound; prefill batches for free
- first eval of each program after load is slow (one-time warm-up); later evals are steady

So Q4_0 cannot be represented exactly; weights are re-quantised per group of 8 rows with 1-D k-means.

## Gemma 4 E2B results on M4 (macOS, `swift test -c release --filter testANEDirectEngine`)
Real Komorebi refine prompt (186 tokens), greedy decoding, ANE programs warm.

| LUT bits (group 8) | ANE weights | decode tok/s | prefill tok/s (186 tok) | output vs Metal Q4_0 |
|---|---|---|---|---|
| 4 | 1.15 GB | **40.6** | **3090** | sentence 1 drops 「開會」, sentence 2 identical |
| 6 | 1.73 GB | 30.7 | 2616 | identical |
| 8 | 2.36 GB | 23.4 | 2145 | identical |

Reference on the same M4: CoreML ANE package (LUT4 g32, MLState) 33 tok/s decode / ~200 tok/s prefill;
Metal Q4_0 (after the simdgroup-matrix prefill rewrite) ~40 tok/s decode / ~1150–1330 tok/s prefill.
(Both of those engines, and a CPU fallback, were removed from the app once this one became the default; they
remain in the repository's first commit, `6b1f870`.)

Decode profile (LUT4): 36 layer programs ≈ 19.5 ms + LM head 3.4 ms per token ≈ 55 GB/s of LUT4 weights
(the ANE weight-streaming roofline measured by `probe t3`); CPU attention + PLE + argmax ≈ 1.5 ms.
Prefill per 64-token chunk: 0.35–0.85 ms per program; layers 15–34 only run for the final chunk.

## Long context: 20 000-token prefill + 256-token decode (M4, `testANEDirectLongContext`)
Prompt = this repo's Swift sources + a summary request (exactly 20 000 tokens); decode forced to 256 tokens.

| LUT | prefill tok/s | decode tok/s @ 20K ctx | prefill: ANE / CPU attention | decode per token: ANE layers / LM head / CPU attention |
|---|---|---|---|---|
| 4 | 2085 | 23.7 | 2.9 s / 6.0 s | 21.6 / 3.4 / 16.8 ms |
| 6 | 2054 | 19.8 | 3.4 s / 6.1 s | 28.6 / 4.9 / 16.8 ms |
| 8 | 1852 | 16.6 | 4.1 s / 6.2 s | 36.3 / 6.6 / 16.9 ms |

At 20K the bottleneck moves from the ANE to CPU attention (O(T²) prefill, ~574 MB of fp32 KV read per decoded token).
Microbenchmark (8 heads x 20K keys x 512): AMX sgemm 1.03 ms (40 GB/s, does not scale with threads),
multi-core NEON fp32 0.62 ms (66 GB/s) -> next steps: NEON decode attention, ANE flash-attention blocks for prefill.

## Decode speed-ups: small-dictionary LM head + Gemma 4 MTP drafter (M4, `ANEMTPTests`)
* **Small dictionary**: 170 748 tokens (ASCII, anything containing CJK / kana / bopomofo, CJK punctuation, emoji,
  control/byte tokens; other scripts and `<unusedN>` dropped). The ANE LM head scores only these rows (LUT4), then the
  best 16 are re-scored exactly against the Q6_K rows on the CPU. Audit vs the exact full Q6_K head: **0 / 264**
  disagreements (the plain full-vocabulary LUT4 ANE head disagreed **6.1 %** of the time).
* **FlashHead (k-means retrieval) does not fit Gemma 4 E2B's tied head**: with 4096 balanced clusters and 64 probes
  the exact argmax was found only 49–69 % of the time (best variant: query-aware spherical, 2048 clusters, 256 probes
  = 12.5 % of the dictionary scanned → 87.5 %). Kept as `FlashHeadLMHead` (experimental, off); study:
  `FH_DIAG=1 swift test --filter FlashHeadDiagTests`.
* **MTP drafter** (`google/gemma-4-E2B-it-assistant`, 4 Q-only layers sharing the target's KV of layers 13/14,
  centroid-masked head 2048×top-32) on the CPU; greedy verification of `drafts + 1` rows with a third ANE procedure
  (S = 4, same cost as S = 1). Outputs are identical to plain decoding.

| prompt | plain | MTP γ=2 | MTP γ=3 |
|---|---|---|---|
| Komorebi draft rewrite | 43.0 tok/s | 97.3 (×2.26) | **107.5 (×2.50)** |
| Komorebi sentence refine | 43.2 | 82.0 (×1.90) | 83.1 (×1.92) |
| English explanation | 44.0 | 77.9 (×1.77) | 77.1 (×1.75) |
| Japanese | 44.3 | 64.4 (×1.45) | 66.0 (×1.49) |
| zh creative writing | 44.4 | 46.9 (×1.06) | 45.9 (×1.03) |

## iPhone 14 Pro Max (A16, iOS 27.2) — `--ane-bench` (LUT4, small-dictionary head, MTP γ≤3 adaptive)
Measured 2026-11-15 with the bench as the only engine in the process (`iphone_clean` run):

| | plain decode | MTP decode | |
|---|---|---|---|
| Komorebi draft rewrite (330-tok prompt) | 26.1 tok/s | **69.3 tok/s (×2.65)** | 82 % accepted, 3.37 tok/step |
| English explanation | 26.8 | 45.7 (×1.70) | 46 %, 1.97 tok/step |
| zh creative writing | 27.3 | 29.6 (×1.08) | 14 %, 1.17 tok/step |
| Komorebi sentence refine (9 output tokens) | 25.8 | 21.6 | too short for drafting to pay off |

* Prefill (warm): **2100–2330 tok/s** on 186/330-token prompts (one or a few 64-token chunks), 370–490 tok/s on
  ~20-token prompts (fixed per-chunk cost).
* Per token: 36 layer programs 29.9 ms (0.5–1.1 ms per eval; verify S = 4 costs the same as S = 1) + LM head 3.1 ms;
  CPU attention/PLE/glue < 1 ms. The A16 ANE streams LUT4 weights at ≈ 30 GB/s (M4 ≈ 55 GB/s).
* Startup: 4.2–5.1 s when aned already has the compiled programs (first launch ever: 72 s compile). App footprint
  177–186 MB (ANE weights live outside the process).
* **Background is not slower**: `--ane-bench --bg-loop 80` while switching the phone to Settings and back
  (screen on, another app in front): foreground 28.9–29.1 tok/s plain / 72–74 tok/s MTP draft rewrite, background
  29.1 / 73.8, 0.89 ms per eval in both. Screen-off (locked phone) was not measured.
* The first iPhone numbers (6–9 tok/s plain, 474 tok/s prefill, ~2–3.6 ms per eval) were wrong: in bench mode the
  app also loaded its own engine at launch (a second 3 GB GGUF mapping and a second set of ANE programs, which that
  build always recompiled), so two engines competed for the ANE and CPU (footprint 477 MB instead of 177 MB). The
  run was at 04:34 and the phone was locked shortly after, so screen-off may also have contributed. `--ane-bench`
  now skips the app's own engine load (`ANEDirectBench.isActive`), and every run logs app/thermal/power state.
* **Compile cache**: `_ANEInMemoryModel.compiledModelExists` — aned keeps compiled programs keyed by the model hash, so
  `ane_rt` skips `compileWithQoS` when it is set. First launch on the iPhone compiled for 72 s; later launches only
  load (M4: 46 s → 1–4 s).
