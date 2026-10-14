#!/usr/bin/env python3
"""Convert a llama2.c checkpoint into the .tlm format read by the C++ engine.

.tlm is a single little-endian file holding the config, int8 group-quantized
weights and the tokenizer, so the browser fetches exactly one asset.

Layout (every section starts on a 64-byte boundary):
    header   64 bytes   magic "TLM1", version, model config, tokenizer location
    f32      rms_att[L*dim], rms_ffn[L*dim], rms_final[dim]
    q8       tok_embedding[vocab, dim]
    q8 x L   wq, wk, wv, wo, w1, w2, w3   (row-major [out, in])
    q8       wcls[vocab, dim]             (only when not shared with embedding)
    bytes    tokenizer in llama2.c tokenizer.bin format
A q8 tensor is int8 values followed by one f32 scale per group of GROUP_SIZE.

hidden_dim is zero-padded up to a multiple of GROUP_SIZE so every row splits
into whole groups. Padding rows of w1/w3 produce silu(0) * 0 = 0, so the
padded model computes exactly the same function.

Usage:
    python3 tools/export.py --bin stories15M.bin --tokenizer tokenizer.bin -o stories15M.tlm
    python3 tools/export.py --npz stories15M.model.npz --tokenizer tokenizer.bin -o stories15M.tlm
"""
import argparse
import struct
import sys

import numpy as np

MAGIC = b"TLM1"
VERSION = 1
GROUP_SIZE = 32
ALIGN = 64
HEADER_FMT = "<4sI11i"  # magic, version, 11 int32 fields; padded to 64 bytes


def read_llama2c_bin(path):
    """Read Karpathy's legacy llama2.c export (fp32, 7-int header)."""
    with open(path, "rb") as f:
        dim, hidden, n_layers, n_heads, n_kv_heads, vocab, seq_len = struct.unpack("<7i", f.read(28))
        data = np.frombuffer(f.read(), dtype=np.float32)
    shared = vocab > 0
    vocab = abs(vocab)
    head_size = dim // n_heads
    kv_dim = n_kv_heads * head_size
    pos = 0

    def take(*shape):
        nonlocal pos
        n = int(np.prod(shape))
        out = data[pos:pos + n].reshape(shape)
        pos += n
        return out

    w = {}
    w["tok_embedding"] = take(vocab, dim)
    w["rms_att"] = take(n_layers, dim)
    w["wq"] = take(n_layers, dim, dim)
    w["wk"] = take(n_layers, kv_dim, dim)
    w["wv"] = take(n_layers, kv_dim, dim)
    w["wo"] = take(n_layers, dim, dim)
    w["rms_ffn"] = take(n_layers, dim)
    w["w1"] = take(n_layers, hidden, dim)
    w["w2"] = take(n_layers, dim, hidden)
    w["w3"] = take(n_layers, hidden, dim)
    w["rms_final"] = take(dim)
    take(seq_len * head_size // 2)  # legacy freq_cis_real, recomputed at runtime
    take(seq_len * head_size // 2)  # legacy freq_cis_imag
    w["wcls"] = w["tok_embedding"] if shared else take(vocab, dim)
    cfg = dict(dim=dim, hidden_dim=hidden, n_layers=n_layers, n_heads=n_heads,
               n_kv_heads=n_kv_heads, vocab_size=vocab, seq_len=seq_len)
    return cfg, w, shared


def read_hf_npz(path, n_heads, n_kv_heads, seq_len):
    """Read a Hugging Face style npz export of the same llama2.c checkpoint."""
    z = np.load(path, allow_pickle=False)
    emb = z["model.embed_tokens.weight"]
    vocab, dim = emb.shape
    n_layers = 1 + max(int(k.split(".")[2]) for k in z.files if k.startswith("model.layers."))
    hidden = z["model.layers.0.mlp.gate_proj.weight"].shape[0]

    def stack(name):
        return np.stack([z[f"model.layers.{i}.{name}"] for i in range(n_layers)])

    w = {
        "tok_embedding": emb,
        "rms_att": stack("input_layernorm.weight"),
        "wq": stack("self_attn.q_proj.weight"),
        "wk": stack("self_attn.k_proj.weight"),
        "wv": stack("self_attn.v_proj.weight"),
        "wo": stack("self_attn.o_proj.weight"),
        "rms_ffn": stack("post_attention_layernorm.weight"),
        "w1": stack("mlp.gate_proj.weight"),
        "w2": stack("mlp.down_proj.weight"),
        "w3": stack("mlp.up_proj.weight"),
        "rms_final": z["model.norm.weight"],
        "wcls": z["lm_head.weight"],
    }
    shared = np.array_equal(w["wcls"], emb)
    cfg = dict(dim=dim, hidden_dim=hidden, n_layers=n_layers, n_heads=n_heads,
               n_kv_heads=n_kv_heads, vocab_size=vocab, seq_len=seq_len)
    return cfg, w, shared


def pad_hidden(cfg, w):
    hidden = cfg["hidden_dim"]
    padded = -(-hidden // GROUP_SIZE) * GROUP_SIZE
    if padded != hidden:
        extra = padded - hidden
        w["w1"] = np.pad(w["w1"], ((0, 0), (0, extra), (0, 0)))
        w["w3"] = np.pad(w["w3"], ((0, 0), (0, extra), (0, 0)))
        w["w2"] = np.pad(w["w2"], ((0, 0), (0, 0), (0, extra)))
        cfg["hidden_dim"] = padded


def quantize_q8(x):
    """Symmetric int8 quantization in groups of GROUP_SIZE along the flattened tensor."""
    g = x.astype(np.float32).reshape(-1, GROUP_SIZE)
    scale = np.abs(g).max(axis=1) / 127.0
    safe = np.where(scale == 0, 1.0, scale)
    q = np.round(g / safe[:, None]).clip(-127, 127).astype(np.int8)
    err = np.abs(q.astype(np.float32) * scale[:, None] - g).max()
    return q.reshape(-1), scale.astype(np.float32), float(err)


class Writer:
    def __init__(self, f):
        self.f = f

    def align(self):
        pad = (-self.f.tell()) % ALIGN
        self.f.write(b"\0" * pad)

    def f32(self, arr):
        self.align()
        self.f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())

    def q8(self, arr):
        q, s, err = quantize_q8(arr)
        self.align()
        self.f.write(q.tobytes())
        self.align()
        self.f.write(s.astype("<f4").tobytes())
        return err


def write_tlm(path, cfg, w, shared, tokenizer_bytes):
    dim, L = cfg["dim"], cfg["n_layers"]
    if dim % GROUP_SIZE:
        sys.exit(f"dim={dim} must be a multiple of {GROUP_SIZE}")
    pad_hidden(cfg, w)

    with open(path, "wb") as f:
        out = Writer(f)
        f.write(b"\0" * ALIGN)  # header placeholder
        out.f32(w["rms_att"])
        out.f32(w["rms_ffn"])
        out.f32(w["rms_final"])
        max_err = out.q8(w["tok_embedding"])
        for l in range(L):
            for name in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
                max_err = max(max_err, out.q8(w[name][l]))
        if not shared:
            max_err = max(max_err, out.q8(w["wcls"]))
        out.align()
        tok_offset = f.tell()
        f.write(tokenizer_bytes)
        size = f.tell()

        header = struct.pack(
            HEADER_FMT, MAGIC, VERSION,
            cfg["dim"], cfg["hidden_dim"], cfg["n_layers"], cfg["n_heads"], cfg["n_kv_heads"],
            cfg["vocab_size"], cfg["seq_len"], GROUP_SIZE, int(shared), tok_offset, len(tokenizer_bytes))
        f.seek(0)
        f.write(header.ljust(ALIGN, b"\0"))
    return size, max_err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bin", help="llama2.c legacy fp32 checkpoint (e.g. stories15M.bin)")
    src.add_argument("--npz", help="Hugging Face style npz of a llama2.c checkpoint")
    ap.add_argument("--tokenizer", required=True, help="llama2.c tokenizer.bin")
    ap.add_argument("--n-heads", type=int, default=6, help="npz only")
    ap.add_argument("--n-kv-heads", type=int, default=6, help="npz only")
    ap.add_argument("--seq-len", type=int, default=256, help="npz only")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    if args.bin:
        cfg, w, shared = read_llama2c_bin(args.bin)
    else:
        cfg, w, shared = read_hf_npz(args.npz, args.n_heads, args.n_kv_heads, args.seq_len)
    with open(args.tokenizer, "rb") as f:
        tok = f.read()

    size, err = write_tlm(args.out, cfg, w, shared, tok)
    print(f"{args.out}: {size / 1e6:.1f} MB  config={cfg}  shared_cls={shared}  max_abs_quant_err={err:.4g}")


if __name__ == "__main__":
    main()
