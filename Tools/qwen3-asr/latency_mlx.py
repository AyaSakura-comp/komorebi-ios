"""Per-sentence latency of Qwen3-ASR-1.7B on the Metal GPU (mlx_audio, MLX 8-bit), to compare with the ANE engine
(Tests/KomorebiCoreTests/Qwen3ASRTests.swift testBakeoffSet with QWEN3_ASR_CONCAT=1,2,4 times the same sentences).

Run with the asr-bakeoff venv (it has mlx_audio):
  cd ~/ExtDev/asr-bakeoff && find .venv hf -name '._*' -delete
  HF_HOME=$PWD/hf .venv/bin/python ~/Developer/komorebi-ios/Tools/qwen3-asr/latency_mlx.py \
      --meta data/cs/meta.json --out /tmp/qwen3-1.7b-mlx8.json

--serve reads "<first clip> <clip count>" lines on stdin and answers each with a "RESULT {json}" line, for
testInterleavedWithMLX, which runs every sentence on the ANE and here back to back.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--meta", required=True, help="meta.json of the bake-off set (wav paths relative to the bake-off root)")
ap.add_argument("--repo", default="mlx-community/Qwen3-ASR-1.7B-8bit")
ap.add_argument("--language", default="Chinese", help='"auto" lets the model name the language')
ap.add_argument("--concat", default="1,2,4", help="also time longer sentences made of N clips joined")
ap.add_argument("--out", help="results JSON (not needed with --serve)")
ap.add_argument("--serve", action="store_true", help="answer sentence requests on stdin (see above)")
args = ap.parse_args()

meta_path = os.path.abspath(args.meta)
root = os.path.dirname(os.path.dirname(os.path.dirname(meta_path)))
meta = json.load(open(meta_path))


def load_wav(m):
    path = m["wav"] if os.path.isabs(m["wav"]) else os.path.join(root, m["wav"])
    a, sr = sf.read(path, dtype="float32")
    assert sr == 16000, path
    return a.mean(axis=1) if a.ndim > 1 else a


clips = [load_wav(m) for m in meta]

import mlx.core as mx  # noqa: E402
from mlx_audio.stt.utils import load  # noqa: E402

model = load(args.repo)
language = None if args.language == "auto" else args.language
for a in clips[:3]:
    model.generate(mx.array(a), language=language)

if args.serve:
    print("READY", flush=True)
    for line in sys.stdin:
        parts = line.split()
        if len(parts) != 2:
            continue
        g, n = int(parts[0]), int(parts[1])
        a = np.concatenate(clips[g:g + n])
        t0 = time.perf_counter()
        r = model.generate(mx.array(a), language=language)
        wall = time.perf_counter() - t0
        print("RESULT " + json.dumps({"wall": wall, "tokens": r.generation_tokens, "text": r.text}, ensure_ascii=False),
              flush=True)
    sys.exit(0)
assert args.out, "--out is required without --serve"

out = {"name": os.path.splitext(os.path.basename(args.out))[0], "backend": f"MLX GPU ({args.repo})",
       "language": args.language, "latency": []}
for n in [int(x) for x in args.concat.split(",") if x]:
    utts, walls, hyps = [], [], []
    start = time.time()
    for g in range(0, len(clips) - n + 1, n):
        a = np.concatenate(clips[g:g + n])
        t0 = time.perf_counter()
        r = model.generate(mx.array(a), language=language)
        wall = time.perf_counter() - t0
        hyps.append(r.text)
        walls.append(wall)
        utts.append({"dur": len(a) / 16000, "wall": wall, "tokens": r.generation_tokens,
                     "prompt_tokens": r.prompt_tokens, "text": r.text})
    end = time.time()
    if n == 1:
        out["hyps"] = hyps
    out["latency"].append({"concat": n, "start": start, "end": end, "utts": utts})
    s = sorted(walls)
    audio = sum(u["dur"] for u in utts)
    print(f"MLX GPU x{n}: {len(walls)} sentences of {audio / len(walls):.2f} s, wall mean {1000 * sum(walls) / len(walls):.0f} ms, "
          f"p50 {1000 * s[len(s) // 2]:.0f}, p90 {1000 * s[int((len(s) - 1) * 0.9)]:.0f}, max {1000 * s[-1]:.0f}, "
          f"RTF {sum(walls) / audio:.3f}", flush=True)
json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
print("->", args.out)
