/// <reference lib="webworker" />
// Runs the C++ engine (compiled to WASM) off the main thread. Tokens are
// generated in short bursts and posted in batches so the page stays smooth
// and a "stop" message is handled within one frame.
import type { FromWorker, ToWorker, TokenEvent } from "./protocol";

declare const self: DedicatedWorkerGlobalScope;

interface TlmModule {
  HEAPU8: Uint8Array;
  HEAP32: Int32Array;
  HEAPF32: Float32Array;
  _tlm_alloc(n: number): number;
  _tlm_free(p: number): void;
  _tlm_load(p: number, n: number): number;
  _tlm_start(p: number, n: number, temperature: number, topP: number, seed: number): number;
  _tlm_step(): number;
  _tlm_piece(token: number, lenPtr: number): number;
  _tlm_position(): number;
  _tlm_seq_len(): number;
  _tlm_n_params(): number;
  _tlm_backend(): number;
}

// Byte layout of StepOut in engine/src/wasm_api.cpp, in 4-byte words.
const STEP = { token: 0, prob: 1, piecePtr: 2, pieceLen: 3, nTop: 4, topToken: 5, topProb: 10 } as const;
const CACHE = "tlm-models-v1";
const BURST_MS = 24;

// Smallest valid module using a SIMD opcode (from wasm-feature-detect).
const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
]);

let mod: TlmModule | null = null;
let modelPtr = 0;
let scratch = 0; // 4 bytes for tlm_piece's out-length
let stopRequested = false;
let running = false;

const post = (m: FromWorker) => self.postMessage(m);

function cString(ptr: number): string {
  const heap = mod!.HEAPU8;
  let end = ptr;
  while (heap[end]) end++;
  return new TextDecoder().decode(heap.subarray(ptr, end));
}

async function loadModule(wasmBase: string): Promise<TlmModule> {
  const variant = WebAssembly.validate(SIMD_PROBE) ? "simd" : "scalar";
  const url = `${wasmBase}tlm-${variant}.mjs`;
  const { default: create } = await import(/* @vite-ignore */ url);
  return create({ locateFile: (f: string) => `${wasmBase}${f}` });
}

// Downloads with progress, serving from the Cache API on repeat visits.
async function fetchModel(url: string): Promise<Uint8Array> {
  let cache: Cache | null = null;
  try {
    cache = await caches.open(CACHE);
    const hit = await cache.match(url);
    if (hit) {
      const buf = new Uint8Array(await hit.arrayBuffer());
      post({ type: "progress", loaded: buf.length, total: buf.length, cached: true });
      return buf;
    }
  } catch {
    cache = null; // Cache API unavailable (e.g. private mode)
  }

  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`model download failed (HTTP ${res.status})`);
  const total = Number(res.headers.get("content-length")) || 0;
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    post({ type: "progress", loaded, total, cached: false });
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  if (cache) {
    cache.put(url, new Response(buf, { headers: { "content-type": "application/octet-stream" } })).catch(() => {});
  }
  return buf;
}

async function load(url: string, wasmBase: string) {
  const t0 = performance.now();
  mod ??= await loadModule(wasmBase);
  const bytes = await fetchModel(url);

  if (modelPtr) mod._tlm_free(modelPtr);
  modelPtr = mod._tlm_alloc(bytes.length);
  mod.HEAPU8.set(bytes, modelPtr);
  const err = mod._tlm_load(modelPtr, bytes.length);
  if (err) throw new Error(cString(err));
  scratch ||= mod._tlm_alloc(4);

  post({
    type: "ready",
    backend: cString(mod._tlm_backend()),
    params: mod._tlm_n_params(),
    seqLen: mod._tlm_seq_len(),
    loadMs: performance.now() - t0,
  });
}

function pieceText(token: number): string {
  const m = mod!;
  const p = m._tlm_piece(token, scratch);
  const len = m.HEAP32[scratch >> 2];
  return new TextDecoder().decode(m.HEAPU8.subarray(p, p + len));
}

async function generate(msg: Extract<ToWorker, { type: "generate" }>) {
  const m = mod!;
  running = true;
  stopRequested = false;

  const prompt = new TextEncoder().encode(msg.prompt);
  const pp = m._tlm_alloc(Math.max(1, prompt.length));
  m.HEAPU8.set(prompt, pp);
  const tp = performance.now();
  const n = m._tlm_start(pp, prompt.length, msg.temperature, msg.topP, msg.seed);
  m._tlm_free(pp);
  post({ type: "prompt", tokens: n, ms: performance.now() - tp });

  const decoder = new TextDecoder();
  const altCache = new Map<number, string>();
  const alt = (id: number) => {
    let s = altCache.get(id);
    if (s === undefined) altCache.set(id, (s = pieceText(id)));
    return s;
  };

  const t0 = performance.now();
  let generated = 0;
  let reason: "eos" | "length" | "context" | "stopped" = "length";

  let finished = false;
  while (!finished) {
    const items: TokenEvent[] = [];
    const burstEnd = performance.now() + BURST_MS;
    while (performance.now() < burstEnd) {
      if (generated >= msg.maxTokens) {
        finished = true;
        break;
      }
      const s = m._tlm_step() >> 2;
      // Re-read heap views each step: memory growth replaces the buffer.
      const i32 = m.HEAP32,
        f32 = m.HEAPF32;
      const token = i32[s + STEP.token];
      if (token < 0) {
        reason = m._tlm_position() >= m._tlm_seq_len() ? "context" : "eos";
        finished = true;
        break;
      }
      const ptr = i32[s + STEP.piecePtr],
        len = i32[s + STEP.pieceLen];
      const top = [];
      for (let k = 0; k < i32[s + STEP.nTop]; k++) {
        top.push({ text: alt(i32[s + STEP.topToken + k]), prob: f32[s + STEP.topProb + k] });
      }
      items.push({
        id: token,
        text: decoder.decode(m.HEAPU8.subarray(ptr, ptr + len), { stream: true }),
        prob: f32[s + STEP.prob],
        top,
      });
      generated++;
    }
    if (items.length) flush(items);
    if (finished) break;
    // Yield so "stop" messages can arrive between bursts.
    await new Promise((r) => setTimeout(r, 0));
    if (stopRequested) {
      reason = "stopped";
      break;
    }
  }

  function flush(items: TokenEvent[]) {
    post({
      type: "tokens",
      items,
      position: m._tlm_position(),
      generated,
      elapsedMs: performance.now() - t0,
    });
  }

  running = false;
  post({ type: "done", reason, generated, elapsedMs: performance.now() - t0 });
}

self.onmessage = async (e: MessageEvent<ToWorker>) => {
  const msg = e.data;
  try {
    if (msg.type === "load") await load(msg.url, msg.wasmBase);
    else if (msg.type === "generate" && mod && !running) await generate(msg);
    else if (msg.type === "stop") stopRequested = true;
  } catch (err) {
    running = false;
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
