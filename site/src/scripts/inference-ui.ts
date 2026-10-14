// UI controller for the in-browser LLM demo. All inference happens in
// llm-worker.ts; this file only renders what the worker reports.
import type { FromWorker, ToWorker, TokenEvent } from "./protocol";

interface ModelInfo {
  id: string;
  file: string;
  label: string;
  detail: string;
  sizeMB: number;
}

const $ = <T extends HTMLElement>(root: HTMLElement, sel: string) => root.querySelector<T>(sel)!;

const fmtPct = (p: number) => (p >= 0.995 ? "1.00" : p.toFixed(2));
// Show spaces and newlines in token strings so " the" and "the" are distinguishable.
const visible = (s: string) => (s === "" ? "∅" : s.replace(/ /g, "␣").replace(/\n/g, "↵"));

export function initDemo(root: HTMLElement) {
  const base = root.dataset.base ?? "/";
  const models: ModelInfo[] = JSON.parse(root.dataset.models ?? "[]");

  const el = {
    status: $(root, ".status"),
    statusText: $(root, "#d-status"),
    meta: $(root, "#d-meta"),
    model: $<HTMLSelectElement>(root, "#d-model"),
    form: $<HTMLFormElement>(root, "#d-form"),
    prompt: $<HTMLInputElement>(root, "#d-prompt"),
    go: $<HTMLButtonElement>(root, "#d-go"),
    out: $(root, "#d-out"),
    gate: $(root, "#d-gate"),
    load: $<HTMLButtonElement>(root, "#d-load"),
    size: $(root, "#d-size"),
    progress: $(root, "#d-progress"),
    err: $(root, "#d-err"),
    heat: $<HTMLInputElement>(root, "#d-heat"),
    tps: $(root, "#s-tps"),
    ms: $(root, "#s-ms"),
    n: $(root, "#s-n"),
    backend: $(root, "#s-backend"),
    ctx: $(root, "#s-ctx"),
    ctxBar: $(root, "#s-ctx-bar"),
    topk: $(root, "#d-topk"),
    temp: $<HTMLInputElement>(root, "#c-temp"),
    topp: $<HTMLInputElement>(root, "#c-topp"),
    max: $<HTMLInputElement>(root, "#c-max"),
    tip: $(root, "#d-tip"),
  };

  let worker: Worker | null = null;
  let loaded = false;
  let generating = false;
  let seqLen = 256;
  let tokens: TokenEvent[] = [];
  let cursor: HTMLElement | null = null;
  let stickToBottom = true;

  // Phones get the tiny model by default: a 0.3 MB download instead of 17.5 MB.
  if (window.matchMedia("(max-width: 560px)").matches && models.length > 1) {
    el.model.value = models[models.length - 1].id;
  }
  const current = () => models.find((m) => m.id === el.model.value) ?? models[0];
  const syncSize = () => (el.size.textContent = `(${current().sizeMB} MB)`);
  syncSize();

  function setStatus(state: "idle" | "busy" | "ready" | "error", text: string) {
    el.status.dataset.state = state;
    el.statusText.textContent = text;
  }

  function setGenerating(on: boolean) {
    generating = on;
    el.go.textContent = on ? "■ Stop" : "Generate";
    el.go.classList.toggle("primary", !on);
    el.prompt.disabled = on;
  }

  function send(msg: ToWorker) {
    worker!.postMessage(msg);
  }

  function startWorker() {
    worker?.terminate();
    worker = new Worker(new URL("./llm-worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (e: MessageEvent<FromWorker>) => onMessage(e.data);
    worker.onerror = (e) => fail(e.message || "worker crashed");
  }

  function load() {
    const m = current();
    loaded = false;
    setGenerating(false);
    el.go.disabled = true;
    el.gate.hidden = false;
    el.load.hidden = true;
    el.err.hidden = true;
    el.progress.hidden = false;
    setStatus("busy", "LOADING");
    startWorker();
    send({ type: "load", url: new URL(base + m.file, location.href).href, wasmBase: new URL(base + "wasm/", location.href).href });
  }

  function fail(message: string) {
    setGenerating(false);
    setStatus("error", "ERROR");
    if (!loaded) {
      el.gate.hidden = false;
      el.load.hidden = false;
      el.load.textContent = "Retry";
      el.progress.hidden = true;
    }
    el.err.hidden = false;
    el.err.textContent = message;
  }

  function generate() {
    if (!loaded) return;
    tokens = [];
    el.out.textContent = "";
    const pr = document.createElement("span");
    pr.className = "pr";
    pr.textContent = el.prompt.value;
    el.out.append(pr);
    cursor = document.createElement("span");
    cursor.className = "cur";
    cursor.textContent = "▌";
    el.out.append(cursor);
    stickToBottom = true;
    setGenerating(true);
    setStatus("busy", "GENERATING");
    send({
      type: "generate",
      prompt: el.prompt.value,
      temperature: Number(el.temp.value),
      topP: Number(el.topp.value),
      maxTokens: Number(el.max.value),
      seed: (Math.random() * 2 ** 31) | 0,
    });
  }

  function renderTopK(top: TokenEvent["top"]) {
    const rows = el.topk.children;
    for (let i = 0; i < rows.length; i++) {
      const li = rows[i] as HTMLElement;
      const a = top[i];
      (li.children[0] as HTMLElement).textContent = a ? visible(a.text) : "·";
      ((li.children[1] as HTMLElement).firstElementChild as HTMLElement).style.width = a ? `${a.prob * 100}%` : "0";
      (li.children[2] as HTMLElement).textContent = a ? fmtPct(a.prob) : "—";
    }
  }

  function appendTokens(items: TokenEvent[]) {
    const frag = document.createDocumentFragment();
    for (const t of items) {
      const span = document.createElement("span");
      span.className = "tk";
      span.dataset.i = String(tokens.length);
      span.textContent = t.text;
      span.style.setProperty("--u", ((1 - t.prob) ** 2).toFixed(3));
      tokens.push(t);
      frag.append(span);
    }
    el.out.insertBefore(frag, cursor);
    if (stickToBottom) el.out.scrollTop = el.out.scrollHeight;
  }

  function onMessage(msg: FromWorker) {
    switch (msg.type) {
      case "progress": {
        const pct = msg.total ? (msg.loaded / msg.total) * 100 : 50;
        (el.progress.firstElementChild as HTMLElement).style.width = `${pct}%`;
        setStatus("busy", msg.cached ? "CACHED" : `LOADING ${pct.toFixed(0)}%`);
        break;
      }
      case "ready": {
        loaded = true;
        seqLen = msg.seqLen;
        el.gate.hidden = true;
        el.go.disabled = false;
        el.backend.textContent = msg.backend;
        el.meta.textContent = `${current().label} · ${(msg.params / 1e6).toFixed(1)}M params · int8 · ${msg.backend} · loaded in ${Math.round(msg.loadMs)} ms`;
        el.ctx.textContent = `0 / ${seqLen}`;
        setStatus("ready", "READY");
        generate();
        break;
      }
      case "prompt":
        el.ctx.textContent = `${msg.tokens + 1} / ${seqLen}`;
        break;
      case "tokens": {
        appendTokens(msg.items);
        const last = msg.items[msg.items.length - 1];
        if (last) renderTopK(last.top);
        const secs = msg.elapsedMs / 1000;
        el.tps.textContent = secs > 0 ? (msg.generated / secs).toFixed(0) : "—";
        el.ms.textContent = msg.generated ? (msg.elapsedMs / msg.generated).toFixed(2) : "—";
        el.n.textContent = String(msg.generated);
        el.ctx.textContent = `${msg.position} / ${seqLen}`;
        el.ctxBar.style.width = `${(msg.position / seqLen) * 100}%`;
        break;
      }
      case "done": {
        setGenerating(false);
        setStatus("ready", "READY");
        cursor?.remove();
        cursor = null;
        const end = document.createElement("span");
        end.className = "end";
        const why = { eos: "end of story", length: "max tokens", context: "context full", stopped: "stopped" }[msg.reason];
        end.textContent = `── ${why} · ${msg.generated} tokens in ${(msg.elapsedMs / 1000).toFixed(2)} s`;
        el.out.append(end);
        break;
      }
      case "error":
        fail(msg.message);
        break;
    }
  }

  // Tooltip: what the model considered at each generated token.
  let active: HTMLElement | null = null;
  function showTip(span: HTMLElement) {
    const t = tokens[Number(span.dataset.i)];
    if (!t) return;
    active?.classList.remove("active");
    active = span;
    span.classList.add("active");
    const rows = t.top
      .map((a) => {
        const chosen = a.text === t.text ? " chosen" : "";
        return `<div class="row${chosen}"><span>${escapeHtml(visible(a.text))}</span><span class="mini"><span style="width:${a.prob * 100}%"></span></span><span>${fmtPct(a.prob)}</span></div>`;
      })
      .join("");
    el.tip.innerHTML = `<div class="hdr">token ${Number(span.dataset.i) + 1} · p = ${t.prob.toFixed(3)}</div>${rows}`;
    el.tip.hidden = false;
    const r = span.getBoundingClientRect();
    const host = root.getBoundingClientRect();
    const w = el.tip.offsetWidth;
    const left = Math.min(Math.max(8, r.left - host.left), host.width - w - 8);
    el.tip.style.left = `${left}px`;
    el.tip.style.top = `${r.bottom - host.top + 6}px`;
    renderTopK(t.top);
  }
  function hideTip() {
    el.tip.hidden = true;
    active?.classList.remove("active");
    active = null;
  }

  el.out.addEventListener("mouseover", (e) => {
    const span = (e.target as HTMLElement).closest<HTMLElement>(".tk");
    if (span) showTip(span);
  });
  el.out.addEventListener("mouseleave", hideTip);
  el.out.addEventListener("click", (e) => {
    const span = (e.target as HTMLElement).closest<HTMLElement>(".tk");
    if (span) showTip(span);
    else hideTip();
  });
  el.out.addEventListener("scroll", () => {
    hideTip();
    stickToBottom = el.out.scrollTop + el.out.clientHeight >= el.out.scrollHeight - 24;
  });

  el.load.addEventListener("click", load);
  el.model.addEventListener("change", () => {
    syncSize();
    if (loaded || worker) load();
  });
  el.form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (generating) send({ type: "stop" });
    else generate();
  });
  root.querySelectorAll<HTMLButtonElement>(".chip").forEach((chip) =>
    chip.addEventListener("click", () => {
      if (generating) return;
      el.prompt.value = chip.dataset.prompt ?? "";
      if (loaded) generate();
      else el.prompt.focus();
    }),
  );
  el.heat.addEventListener("change", () => (el.out.dataset.heat = el.heat.checked ? "on" : "off"));

  const bindRange = (input: HTMLInputElement, id: string, digits: number) => {
    const out = $(root, id);
    const sync = () => (out.textContent = Number(input.value).toFixed(digits));
    input.addEventListener("input", sync);
    sync();
  };
  bindRange(el.temp, "#o-temp", 2);
  bindRange(el.topp, "#o-topp", 2);
  bindRange(el.max, "#o-max", 0);
}

function escapeHtml(s: string) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
}
