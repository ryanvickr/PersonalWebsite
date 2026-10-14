#!/usr/bin/env bash
# Builds the engine (native tests + WASM) and stages artifacts into site/public.
#   tools/build.sh                 # needs emcc on PATH (source emsdk_env.sh)
#   MODELS_DIR=/path tools/build.sh   # also copy *.tlm models from MODELS_DIR
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"

cmake -S "$root/engine" -B "$root/engine/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$root/engine/build" -j
ctest --test-dir "$root/engine/build" --output-on-failure

emcmake cmake -S "$root/engine" -B "$root/engine/build-wasm" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$root/engine/build-wasm" -j

mkdir -p "$root/site/public/wasm" "$root/site/public/models"
cp "$root"/engine/build-wasm/tlm-{scalar,simd}.{mjs,wasm} "$root/site/public/wasm/"
if [[ -n "${MODELS_DIR:-}" ]]; then
  cp "$MODELS_DIR"/*.tlm "$root/site/public/models/"
fi
echo "staged: $(ls "$root/site/public/wasm" "$root/site/public/models" | tr '\n' ' ')"
