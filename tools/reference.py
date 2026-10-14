#!/usr/bin/env python3
"""fp32 numpy reference forward pass, for checking the int8 C++ engine.

Greedy-decodes a prompt with the unquantized weights and prints the text, so
it can be compared with `tlm-run model.tlm -p PROMPT -n N` (temperature 0).

    python3 tools/reference.py --bin stories15M.bin --tokenizer tokenizer.bin -n 24
"""
import argparse
import struct

import numpy as np

from export import read_hf_npz, read_llama2c_bin


def load_tokenizer(path, vocab_size):
    vocab, scores = [], []
    with open(path, "rb") as f:
        struct.unpack("<i", f.read(4))
        for _ in range(vocab_size):
            score, n = struct.unpack("<fi", f.read(8))
            scores.append(score)
            vocab.append(f.read(n))
    return vocab, scores


def encode(text, vocab, scores):
    index = {}
    for i, v in enumerate(vocab):
        index.setdefault(v, i)
    tokens = [index[b" "]] if b" " in index else []
    for ch in text:
        b = ch.encode()
        tokens.extend([index[b]] if b in index else [x + 3 for x in b])
    while True:
        best = None
        for i in range(len(tokens) - 1):
            j = index.get(vocab[tokens[i]] + vocab[tokens[i + 1]])
            if j is not None and (best is None or scores[j] > scores[best[1]]):
                best = (i, j)
        if best is None:
            return [1] + tokens
        tokens[best[0]:best[0] + 2] = [best[1]]


def rmsnorm(x, w):
    return w * x / np.sqrt(np.mean(x * x) + 1e-5)


def rope(v, head_size, pos):
    out = v.copy()
    i = np.arange(0, v.size, 2)
    freq = 1.0 / 10000.0 ** ((i % head_size) / head_size)
    c, s = np.cos(pos * freq), np.sin(pos * freq)
    out[i] = v[i] * c - v[i + 1] * s
    out[i + 1] = v[i] * s + v[i + 1] * c
    return out


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bin")
    src.add_argument("--npz")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("-p", "--prompt", default="Once upon a time")
    ap.add_argument("-n", type=int, default=24)
    args = ap.parse_args()

    cfg, w, _ = read_llama2c_bin(args.bin) if args.bin else read_hf_npz(args.npz, 6, 6, 256)
    vocab, scores = load_tokenizer(args.tokenizer, cfg["vocab_size"])
    dim, L, H, KV = cfg["dim"], cfg["n_layers"], cfg["n_heads"], cfg["n_kv_heads"]
    hs = dim // H
    kcache = np.zeros((L, cfg["seq_len"], KV * hs), np.float32)
    vcache = np.zeros_like(kcache)

    def forward(tok, pos):
        x = w["tok_embedding"][tok].astype(np.float32)
        for l in range(L):
            xb = rmsnorm(x, w["rms_att"][l])
            q = rope(w["wq"][l] @ xb, hs, pos)
            kcache[l, pos] = rope(w["wk"][l] @ xb, hs, pos)
            vcache[l, pos] = w["wv"][l] @ xb
            out = np.empty(dim, np.float32)
            for h in range(H):
                kh = (h // (H // KV)) * hs
                att = kcache[l, :pos + 1, kh:kh + hs] @ q[h * hs:(h + 1) * hs] / np.sqrt(hs)
                att = np.exp(att - att.max())
                att /= att.sum()
                out[h * hs:(h + 1) * hs] = att @ vcache[l, :pos + 1, kh:kh + hs]
            x = x + w["wo"][l] @ out
            xb = rmsnorm(x, w["rms_ffn"][l])
            g = w["w1"][l] @ xb
            x = x + w["w2"][l] @ (g / (1 + np.exp(-g)) * (w["w3"][l] @ xb))
        return w["wcls"] @ rmsnorm(x, w["rms_final"])

    tokens = encode(args.prompt, vocab, scores)
    pos = 0
    for t in tokens:
        logits = forward(t, pos)
        pos += 1
    prev, out = tokens[-1], b""
    for _ in range(args.n):
        tok = int(np.argmax(logits))
        if tok in (1, 2):
            break
        piece = vocab[tok]
        if prev == 1 and piece.startswith(b" "):
            piece = piece[1:]
        if len(piece) == 6 and piece.startswith(b"<0x"):
            piece = bytes([int(piece[3:5], 16)])
        out += piece
        prev = tok
        logits = forward(tok, pos)
        pos += 1
    print(out.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
