"""Qwen3-ASR (Hugging Face safetensors, bf16) -> one GGUF file for Komorebi's ANE engine.

Big matrices are stored as Q8_0 (the engine re-quantises them into ANE LUT palettes, so Q8_0 only
has to be a faithful source), the audio front end's convolutions as F16, norms and biases as F32.
The prompt pieces (token ids around the audio) and the Whisper mel filter bank come from the
reference dump of make_reference.py, so they are exactly what the original pipeline uses.

usage: uv run --no-project --with gguf --with numpy python convert_gguf.py \
           <model_dir> <reference_dir> <out.gguf>
"""
import json
import os
import struct
import sys

import numpy as np
import gguf
from gguf import GGMLQuantizationType as Q

model_dir, ref_dir, out_path = sys.argv[1:4]
cfg = json.load(open(os.path.join(model_dir, "config.json")))["thinker_config"]
tc, ac = cfg["text_config"], cfg["audio_config"]


class Safetensors:
    """Lazy reader for a sharded safetensors checkpoint (bf16/f16/f32 -> float32)."""

    def __init__(self, folder):
        index = json.load(open(os.path.join(folder, "model.safetensors.index.json")))["weight_map"]
        self.where = {}
        self.headers = {}
        for name, shard in index.items():
            path = os.path.join(folder, shard)
            if path not in self.headers:
                with open(path, "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    self.headers[path] = (8 + n, json.loads(f.read(n)))
            self.where[name] = path
        self.maps = {}

    def shape(self, name):
        path = self.where[name]
        return self.headers[path][1][name]["shape"]

    def get(self, name):
        path = self.where[name]
        base, header = self.headers[path]
        if path not in self.maps:
            self.maps[path] = np.memmap(path, dtype=np.uint8, mode="r")
        info = header[name]
        s, e = info["data_offsets"]
        raw = np.asarray(self.maps[path][base + s:base + e])
        if info["dtype"] == "BF16":
            a = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif info["dtype"] == "F16":
            a = raw.view(np.float16).astype(np.float32)
        elif info["dtype"] == "F32":
            a = raw.view(np.float32).copy()
        else:
            raise ValueError(f"{name}: dtype {info['dtype']}")
        return a.reshape(info["shape"])


st = Safetensors(model_dir)

# (gguf name, checkpoint name, storage) in file order. storage: q8 | f16 | f32
plan = []
T = "thinker.model."
plan.append(("token_embd.weight", T + "embed_tokens.weight", "q8"))
for l in range(tc["num_hidden_layers"]):
    p = f"{T}layers.{l}."
    plan += [
        (f"blk.{l}.attn_norm.weight", p + "input_layernorm.weight", "f32"),
        (f"blk.{l}.attn_q.weight", p + "self_attn.q_proj.weight", "q8"),
        (f"blk.{l}.attn_k.weight", p + "self_attn.k_proj.weight", "q8"),
        (f"blk.{l}.attn_v.weight", p + "self_attn.v_proj.weight", "q8"),
        (f"blk.{l}.attn_q_norm.weight", p + "self_attn.q_norm.weight", "f32"),
        (f"blk.{l}.attn_k_norm.weight", p + "self_attn.k_norm.weight", "f32"),
        (f"blk.{l}.attn_output.weight", p + "self_attn.o_proj.weight", "q8"),
        (f"blk.{l}.ffn_norm.weight", p + "post_attention_layernorm.weight", "f32"),
        (f"blk.{l}.ffn_gate.weight", p + "mlp.gate_proj.weight", "q8"),
        (f"blk.{l}.ffn_up.weight", p + "mlp.up_proj.weight", "q8"),
        (f"blk.{l}.ffn_down.weight", p + "mlp.down_proj.weight", "q8"),
    ]
plan.append(("output_norm.weight", T + "norm.weight", "f32"))

A = "thinker.audio_tower."
for i in (1, 2, 3):
    plan += [(f"a.conv{i}.weight", f"{A}conv2d{i}.weight", "f16"), (f"a.conv{i}.bias", f"{A}conv2d{i}.bias", "f32")]
plan.append(("a.conv_out.weight", A + "conv_out.weight", "q8"))
for l in range(ac["encoder_layers"]):
    p = f"{A}layers.{l}."
    for dst, src in [("attn_norm", "self_attn_layer_norm"), ("ffn_norm", "final_layer_norm")]:
        plan += [(f"a.blk.{l}.{dst}.weight", p + src + ".weight", "f32"), (f"a.blk.{l}.{dst}.bias", p + src + ".bias", "f32")]
    for dst, src in [("attn_q", "self_attn.q_proj"), ("attn_k", "self_attn.k_proj"), ("attn_v", "self_attn.v_proj"),
                     ("attn_out", "self_attn.out_proj"), ("ffn_up", "fc1"), ("ffn_down", "fc2")]:
        plan += [(f"a.blk.{l}.{dst}.weight", p + src + ".weight", "q8"), (f"a.blk.{l}.{dst}.bias", p + src + ".bias", "f32")]
for dst, src in [("ln_post", "ln_post"), ("proj1", "proj1"), ("proj2", "proj2")]:
    plan += [(f"a.{dst}.weight", f"{A}{src}.weight", "q8" if dst != "ln_post" else "f32"),
             (f"a.{dst}.bias", f"{A}{src}.bias", "f32")]

# The checkpoint carries lm_head next to the tied embeddings; keep it only if it differs.
emb = st.get(T + "embed_tokens.weight")
head = st.get("thinker.lm_head.weight")
tied = bool(np.array_equal(emb, head))
print("lm_head tied to embed_tokens:", tied, flush=True)
if not tied:
    plan.append(("output.weight", "thinker.lm_head.weight", "q8"))
del emb, head

# ---- metadata ----
w = gguf.GGUFWriter(out_path, "qwen3asr")
w.add_name("Qwen3-ASR-1.7B")
k = "qwen3asr"
w.add_uint32(f"{k}.block_count", tc["num_hidden_layers"])
w.add_uint32(f"{k}.embedding_length", tc["hidden_size"])
w.add_uint32(f"{k}.feed_forward_length", tc["intermediate_size"])
w.add_uint32(f"{k}.attention.head_count", tc["num_attention_heads"])
w.add_uint32(f"{k}.attention.head_count_kv", tc["num_key_value_heads"])
w.add_uint32(f"{k}.attention.key_length", tc["head_dim"])
w.add_float32(f"{k}.attention.layer_norm_rms_epsilon", tc["rms_norm_eps"])
w.add_float32(f"{k}.rope.freq_base", float(tc["rope_theta"]))
w.add_uint32(f"{k}.context_length", tc["max_position_embeddings"])
w.add_uint32(f"{k}.vocab_size", tc["vocab_size"])
w.add_bool(f"{k}.tied_lm_head", tied)
w.add_uint32(f"{k}.audio.block_count", ac["encoder_layers"])
w.add_uint32(f"{k}.audio.embedding_length", ac["d_model"])
w.add_uint32(f"{k}.audio.feed_forward_length", ac["encoder_ffn_dim"])
w.add_uint32(f"{k}.audio.head_count", ac["encoder_attention_heads"])
w.add_uint32(f"{k}.audio.num_mel_bins", ac["num_mel_bins"])
w.add_uint32(f"{k}.audio.downsample_hidden_size", ac["downsample_hidden_size"])
w.add_uint32(f"{k}.audio.n_window", ac["n_window"])
w.add_uint32(f"{k}.audio.n_window_infer", ac["n_window_infer"])
w.add_uint32(f"{k}.audio.output_dim", ac["output_dim"])
w.add_uint32(f"{k}.audio.max_source_positions", ac["max_source_positions"])
w.add_float32(f"{k}.audio.layer_norm_epsilon", 1e-5)
w.add_array(f"{k}.support_languages", json.load(open(os.path.join(model_dir, "config.json"))).get("support_languages", []))

mel = np.fromfile(os.path.join(ref_dir, "mel_filters.f32"), dtype=np.float32)
w.add_array(f"{k}.audio.mel_filters", mel.tolist())   # [201 * 128], row = frequency bin
w.add_uint32(f"{k}.audio.n_fft", 400)
w.add_uint32(f"{k}.audio.hop_length", 160)

ref = json.load(open(os.path.join(ref_dir, "manifest.json")))["utterances"][0]
auto, zh = ref["prompt_auto"], ref["prompt_Chinese"]
PAD, AEND = cfg["audio_token_id"], cfg["audio_end_token_id"]
first, last = auto.index(PAD), len(auto) - 1 - auto[::-1].index(PAD)
assert zh[:len(auto)] == auto, "forced-language prompt must extend the auto prompt"
w.add_array(f"{k}.prompt.prefix", auto[:first])          # ... <|audio_start|>
w.add_array(f"{k}.prompt.suffix", auto[last + 1:])        # <|audio_end|> ... assistant\n
w.add_array(f"{k}.prompt.language_chinese", zh[len(auto):])  # language Chinese<asr_text>
w.add_uint32(f"{k}.audio_token_id", PAD)
w.add_uint32(f"{k}.audio_start_token_id", cfg["audio_start_token_id"])
w.add_uint32(f"{k}.audio_end_token_id", AEND)
w.add_uint32(f"{k}.asr_text_token_id", 151704)
w.add_array(f"{k}.eos_token_ids", [151643, 151645])

vocab = json.load(open(os.path.join(model_dir, "vocab.json"), encoding="utf-8"))
added = json.load(open(os.path.join(model_dir, "tokenizer_config.json"), encoding="utf-8"))["added_tokens_decoder"]
n_vocab = tc["vocab_size"]
tokens = [f"[PAD{i}]" for i in range(n_vocab)]
types = [gguf.TokenType.UNUSED] * n_vocab
for t, i in vocab.items():
    tokens[i] = t
    types[i] = gguf.TokenType.NORMAL
for i, info in added.items():
    i = int(i)
    tokens[i] = info["content"]
    types[i] = gguf.TokenType.CONTROL if info.get("special") else gguf.TokenType.USER_DEFINED
w.add_tokenizer_model("gpt2")
w.add_token_list(tokens)
w.add_token_types(types)
merges = [line for line in open(os.path.join(model_dir, "merges.txt"), encoding="utf-8").read().split("\n")
          if line and not line.startswith("#version")]
w.add_token_merges(merges)
w.add_eos_token_id(151645)

# ---- tensor infos, then data in the same order ----
def stored(name, kind):
    shape = st.shape(name)
    if kind == "q8":
        n, kk = shape[0], int(np.prod(shape[1:]))
        assert kk % 32 == 0, (name, shape)
        return (n, kk // 32 * 34), np.uint8, n * (kk // 32) * 34, Q.Q8_0
    count = int(np.prod(shape))
    if kind == "f16":
        return tuple(shape), np.float16, count * 2, None
    return tuple(shape), np.float32, count * 4, None

for dst, src, kind in plan:
    shape, dt, nbytes, raw = stored(src, kind)
    w.add_tensor_info(dst, shape, np.dtype(dt), nbytes, raw_dtype=raw)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
total = 0
for i, (dst, src, kind) in enumerate(plan):
    a = st.get(src)
    if kind == "q8":
        a = a.reshape(a.shape[0], -1)
        data = gguf.quants.quantize(a, Q.Q8_0)
    elif kind == "f16":
        data = a.astype(np.float16)
    else:
        data = a.astype(np.float32)
    w.write_tensor_data(data)
    total += data.nbytes
    if i % 50 == 0:
        print(f"{i}/{len(plan)} {dst} {total / 1e9:.2f} GB", flush=True)
w.close()
print("wrote", out_path, f"{total / 1e9:.2f} GB in {len(plan)} tensors")
