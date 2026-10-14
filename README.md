# PersonalWebsite

My portfolio site. Its centerpiece is **`tlm`**, a dependency-free C++20
transformer inference engine compiled to WebAssembly. It runs a 15M-parameter
Llama-2-style language model live in the visitor's browser, streaming tokens
with their probabilities.

```
engine/   C++20 inference engine (native + WebAssembly builds)
tools/    export.py (checkpoint → int8 .tlm), reference.py (fp32 NumPy check), build.sh
site/     Astro static site (terminal / quant theme)
.github/  CI: export models → test → build WASM → build site → deploy to GitHub Pages
```

## Engine

| build             | stories15M, greedy, 1 core |
|-------------------|---------------------------:|
| native AVX2       | 326 tok/s                  |
| WASM simd128 (V8) | 251 tok/s                  |
| WASM scalar (V8)  |  77 tok/s                  |

- int8 group quantization (32 values per fp32 scale) for weights and activations; 60 MB → 17.5 MB
- SIMD int8 matmul kernels for `wasm_simd128` and AVX2, plus a scalar reference that the tests check them against
- KV cache, RoPE, grouped-query attention, SwiGLU, SentencePiece BPE with byte fallback
- one self-describing `.tlm` file (config + weights + tokenizer), parsed zero-copy

### Build and test (native)

```sh
cmake -S engine -B engine/build && cmake --build engine/build -j
ctest --test-dir engine/build                       # unit tests, no weights needed
engine/build/tlm-run stories15M.tlm -p "Once upon a time" -t 0.8
engine/build/tlm-bench stories15M.tlm
```

### Get a model

```sh
curl -LO https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
curl -LO https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin
pip install numpy
python3 tools/export.py --bin stories15M.bin --tokenizer tokenizer.bin -o stories15M.tlm
python3 tools/reference.py --bin stories15M.bin --tokenizer tokenizer.bin   # fp32 cross-check
```

## Site

Needs [emsdk](https://emscripten.org/docs/getting_started/downloads.html) for the WASM build.

```sh
source /path/to/emsdk/emsdk_env.sh
MODELS_DIR=/dir/with/tlm/files tools/build.sh   # builds + tests the engine, stages wasm and models into site/public
cd site && npm install && npm run dev
```

Personal details live in `site/src/site.config.ts`. Projects and experience are
Markdown files in `site/src/content/`. Replace `site/public/resume.pdf` with
your résumé.

## Deploy

Pushes to `main` build and deploy to GitHub Pages through
`.github/workflows/deploy.yml`. One-time setup: **Settings → Pages → Source:
GitHub Actions**. For a custom domain, add `site/public/CNAME` and set the
`SITE_URL` / `BASE_PATH=/` env vars in the workflow's site build step.

## Credits

Model checkpoints and the tokenizer come from Andrej Karpathy's
[llama2.c](https://github.com/karpathy/llama2.c) (MIT), trained on
[TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023). The engine
is an independent C++ implementation of the same architecture.
