// Smoke test for the WASM builds under Node: node tests/wasm_smoke.mjs build-wasm model.tlm
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const [dir, modelPath] = process.argv.slice(2);
const model = readFileSync(modelPath);
const enc = new TextEncoder();
const dec = new TextDecoder();

for (const variant of ["scalar", "simd"]) {
  const { default: create } = await import(resolve(dir, `tlm-${variant}.mjs`));
  const m = await create();
  const ptr = m._tlm_alloc(model.length);
  m.HEAPU8.set(model, ptr);
  const err = m._tlm_load(ptr, model.length);
  if (err) throw new Error(`${variant}: load failed`);

  const prompt = enc.encode("Once upon a time");
  const pp = m._tlm_alloc(prompt.length);
  m.HEAPU8.set(prompt, pp);
  m._tlm_start(pp, prompt.length, 0, 0.9, 1);

  let text = "";
  const t0 = performance.now();
  let n = 0;
  for (; n < 64; n++) {
    const s = m._tlm_step() >> 2;
    if (m.HEAP32[s] < 0) break;
    const ptrPiece = m.HEAP32[s + 2], len = m.HEAP32[s + 3];
    text += dec.decode(m.HEAPU8.subarray(ptrPiece, ptrPiece + len), { stream: true });
  }
  const tps = n / ((performance.now() - t0) / 1000);
  const backend = m.UTF8ToString ? m.UTF8ToString(m._tlm_backend()) : "";
  console.log(`${variant.padEnd(6)} ${tps.toFixed(1)} tok/s ${backend}\n  ${text.slice(0, 120)}`);
  if (!text.startsWith(", there was a little girl named Lily")) throw new Error(`${variant}: unexpected output`);
}
