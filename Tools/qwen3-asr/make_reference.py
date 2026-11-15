"""Reference tensors for the Swift/ANE port of Qwen3-ASR (run on a Mac with mlx_audio).

Loads the original Qwen/Qwen3-ASR-1.7B weights with mlx_audio, casts them to float32 and, for a few
utterances, writes what each stage produces: log-mel features, audio-encoder output (also after the
conv front end and after encoder layer 0), prompt ids, greedy output ids and text. The Swift tests
compare the ANE engine with these files.

usage: python make_reference.py <model_dir> <out_dir> <wav>...
"""
import json, os, sys
import numpy as np
import soundfile as sf
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from mlx_audio.stt.utils import load

model_dir, out_dir, wavs = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(out_dir, exist_ok=True)
model = load(model_dir)
inner = model.model if hasattr(model, "model") and hasattr(model.model, "audio_tower") else model
# The audio tower in float32 (compared tensor by tensor); the decoder stays in its original bf16
# (only its tokens are compared, and float32 would not fit next to other jobs on a 16 GB Mac).
inner.audio_tower.update(tree_map(lambda p: p.astype(mx.float32) if isinstance(p, mx.array) else p, inner.audio_tower.parameters()))
mx.eval(inner.parameters())
fe = model._feature_extractor

def save(name, arr):
    a = np.asarray(arr, dtype=np.float32)
    a.tofile(os.path.join(out_dir, name + ".f32"))
    return list(a.shape)

manifest = {"mel_filters": save("mel_filters", fe.mel_filters), "utterances": []}
eos = {151643, 151645}
for wi, path in enumerate(wavs):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == 16000, sr
    feats, mask, n_audio = model._preprocess_audio(audio)
    tag = f"u{wi}"
    entry = {"wav": os.path.abspath(path), "samples": int(len(audio)), "audio_tokens": n_audio,
             "features": save(f"{tag}_features", feats[0])}
    at = inner.audio_tower
    # Conv front end and layer 0 for the single-window case (all test clips are < 8 s).
    T = feats.shape[-1]
    chunks = []
    for s in range(0, T, 100):
        c = feats[0, :, s:s + 100]
        if c.shape[-1] < 100 and T > 100:
            c = mx.pad(c, [(0, 0), (0, 100 - c.shape[-1])])
        chunks.append(c)
    x = mx.stack(chunks, axis=0)[:, :, :, None]
    x = nn.gelu(at.conv2d1(x)); x = nn.gelu(at.conv2d2(x)); x = nn.gelu(at.conv2d3(x))
    b, f, t, c = x.shape
    x = x.transpose(0, 2, 3, 1).reshape(b, t, c * f)
    x = at.conv_out(x) + at.positional_embedding(t)[None]
    front = x.reshape(-1, x.shape[-1])[:n_audio]
    entry["front"] = save(f"{tag}_front", front)
    h0 = at.layers[0](front[None])[0]
    entry["layer0"] = save(f"{tag}_layer0", h0)
    enc = inner.get_audio_features(feats, mask)
    entry["encoder"] = save(f"{tag}_encoder", enc)
    mx.clear_cache()
    for lang in (None, "Chinese"):
        ids = model._build_prompt(n_audio, language=lang)
        cache = model.make_cache()
        logits = inner(ids, input_features=feats, feature_attention_mask=mask, cache=cache)
        out = []
        first = np.array(logits[0, -1].astype(mx.float32))
        top = np.argsort(-first)[:5]
        tok = int(top[0])
        while tok not in eos and len(out) < 120:
            out.append(tok)
            logits = inner(mx.array([[tok]]), cache=cache)
            tok = int(mx.argmax(logits[0, -1]).item())
        key = "auto" if lang is None else lang
        entry[f"prompt_{key}"] = np.array(ids)[0].tolist()
        entry[f"out_{key}"] = out
        entry[f"text_{key}"] = model._tokenizer.decode(out)
        entry[f"top5_{key}"] = [[int(i), float(first[i])] for i in top]
    print(tag, entry["text_auto"], "|", entry["text_Chinese"], flush=True)
    manifest["utterances"].append(entry)
json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), ensure_ascii=False, indent=1)
