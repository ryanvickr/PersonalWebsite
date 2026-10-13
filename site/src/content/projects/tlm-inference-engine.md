---
title: "tlm: transformer inference in C++ → WebAssembly"
summary: "A dependency-free C++20 engine for Llama-2-architecture models with int8 group quantization and hand-written SIMD kernels (AVX2 and wasm simd128). It powers the live demo on this site."
stack: ["C++20", "WASM SIMD", "AVX2", "Emscripten", "Python/NumPy"]
metric: { value: "251 tok/s", label: "stories15M in WASM, one core" }
repo: "https://github.com/ryanvickr/PersonalWebsite/tree/main/engine"
order: 1
featured: true
---

## What it is

`tlm` runs small Llama-2 style language models (the TinyStories checkpoints
from [llama2.c](https://github.com/karpathy/llama2.c)) one token at a time.
The same ~1,000 lines of C++ build natively for benchmarking and to WebAssembly
for the browser demo, where the binary is **70 KB**.

## Design

- **One file, zero copies.** `tools/export.py` packs config, int8 weights
  and tokenizer into a 64-byte-aligned `.tlm` file. The engine parses the
  header and points straight into the buffer, so loading is a single `memcpy`
  into WASM memory.
- **int8 group quantization.** Weights are stored in groups of 32 int8 values
  with one fp32 scale each. Activations are quantized the same way just before
  every matmul, so the inner loop is pure integer arithmetic. Weights shrink
  from 60 MB to 17.5 MB.
- **SIMD kernels.** WASM has no int8 dot-product instruction, so the kernel
  widens to i16 and uses `i32x4.dot_i16x8_s`. It computes two output rows per
  iteration so each activation load is reused. On x86 the same function uses
  AVX2 `maddubs`/`madd`. A scalar reference stays in the build, and the tests
  check the SIMD paths against it.
- **Exact padding.** stories260K's FFN width (172) isn't a multiple of the
  group size, so the exporter zero-pads it. Padded rows contribute
  `silu(0) · 0 = 0`, which makes the padded model compute exactly the same
  function.
- **Streaming in the browser.** The engine runs in a Web Worker. It generates
  in ~24 ms bursts and posts batches of tokens with their top-5
  alternatives, so the page stays smooth and "stop" responds within a frame.
  Results cross the JS/WASM boundary through a fixed struct in linear memory,
  so no strings are allocated per token.

## Measurements

Single core, x86-64 cloud VM, stories15M, greedy decoding:

| build                | tok/s | matmul 32000×288 |
|----------------------|------:|-----------------:|
| native, AVX2         |   326 | 0.76 ms          |
| native, scalar       |     — | 2.35 ms          |
| WASM simd128 (V8)    |   251 | —                |
| WASM scalar (V8)     |    77 | —                |

## Correctness

- Unit tests build a random model in memory (with grouped-query attention)
  and check quantization error bounds, SIMD against scalar, tokenizer merges
  and byte fallback, determinism and context limits.
- `tools/reference.py` is an fp32 NumPy implementation of the same forward
  pass. On test prompts the int8 engine's greedy output matches it token for
  token. The first divergence seen was a genuine near-tie
  (p = 0.213 vs 0.212).
