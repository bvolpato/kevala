// Small UI helpers shared by the views: formatting, the answer bars, syntax highlighting and
// the "load a model" gate. Nothing here talks to the network.

import { session, MODELS } from "./session.js";

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const HTML_ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

export function esc(s) {
  return String(s).replace(/[&<>"']/g, (ch) => HTML_ESCAPES[ch]);
}

export function fmtBytes(n) {
  if (!n) return "0 B";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  if (n >= 1e3) return `${Math.round(n / 1e3)} KB`;
  return `${n} B`;
}

export function fmtMs(ms) {
  if (ms == null || !isFinite(ms)) return "–";
  if (ms >= 10000) return `${(ms / 1000).toFixed(1)} s`;
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  if (ms >= 100) return `${Math.round(ms)} ms`;
  return `${ms.toFixed(1)} ms`;
}

export function debounce(fn, ms) {
  let timer = 0;
  const debounced = (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
  debounced.cancel = () => clearTimeout(timer);
  return debounced;
}

export function backendLabel(info) {
  if (!info) return "";
  if (info.backend === "webgpu") return "WebGPU";
  return `CPU · ${info.threads > 1 ? `${info.threads} workers` : "1 thread"}`;
}

/**
 * Why a model that could use WebGPU runs on the CPU, and how this browser can turn WebGPU on; ""
 * when it runs on WebGPU or the CPU was asked for.
 */
export function cpuReason(info) {
  if (!info || info.backend === "webgpu" || !info.gpuUnavailable) return "";
  const ua = navigator.userAgent;
  const firefox = /Firefox\//.test(ua);
  const chrome = /Chrome\//.test(ua);
  const safari = /Safari\//.test(ua) && !chrome;
  const linux = /Linux/.test(ua) && !/Android/.test(ua);
  let fix = "";
  if (navigator.gpu && /worker/.test(info.gpuUnavailable)) {
    fix = firefox
      ? "Firefox has WebGPU on this page but not in workers, where kevala runs: set dom.webgpu.workers.enabled to true in about:config and reload."
      : "This browser has WebGPU on pages but not in workers, where kevala runs.";
  } else if (firefox) {
    fix = "In about:config, set dom.webgpu.enabled and dom.webgpu.workers.enabled to true (and gfx.webgpu.ignore-blocklist if the GPU is blocklisted), then reload.";
  } else if (chrome && linux) {
    fix = "Turn on chrome://flags/#enable-unsafe-webgpu and chrome://flags/#enable-vulkan, then restart the browser.";
  } else if (safari) {
    fix = "Safari has WebGPU from version 26; before that it is under Develop > Feature Flags.";
  }
  return `Running on the CPU because ${info.gpuUnavailable}. ${fix}`.trim();
}

export function backendBadge(info) {
  if (!info) return `<span class="badge"><span class="dot"></span>not loaded</span>`;
  const kind = info.backend === "webgpu" ? "gpu" : "cpu";
  const title = `${esc(info.backend)}${info.gpu ? ` · ${esc(info.gpu)}` : ""}`;
  return `<span class="badge ${kind}" title="${title}"><span class="dot"></span>${esc(backendLabel(info))}</span>`;
}

// Syntax highlighting

// groups: 1 a string (a key when group 2, its colon, follows), 3 a number, 4 a literal, 5 punctuation
const JSON_TOKENS = /("(?:[^"\\\n]|\\.)*"?)(\s*:)?|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|\b(true|false|null)\b|([{}[\],])/g;

/**
 * Highlights JSON (or JSON-like text being typed). Every character of the input is kept, so the
 * result can sit under a textarea as its highlighted twin.
 */
export function highlightJSON(text) {
  let out = "";
  let last = 0;
  for (const match of text.matchAll(JSON_TOKENS)) {
    const [token, string, colon, number, literal, punct] = match;
    out += esc(text.slice(last, match.index));
    if (string !== undefined) {
      out += colon ? `<span class="j-k">${esc(string)}</span>${esc(colon)}` : `<span class="j-s">${esc(string)}</span>`;
    } else if (number !== undefined) out += `<span class="j-n">${number}</span>`;
    else if (literal !== undefined) out += `<span class="j-b">${literal}</span>`;
    else out += `<span class="j-p">${esc(punct)}</span>`;
    last = match.index + token.length;
  }
  return out + esc(text.slice(last));
}

const JS_TOKENS = new RegExp(
  [
    /(\/\/[^\n]*)/.source,
    /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)/.source,
    /\b(import|from|const|let|await|async|return|if|else|new|export|function|for|of)\b/.source,
    /\b(\d+(?:\.\d+)?)\b/.source,
    /\b([A-Za-z_$][\w$]*)(?=\()/.source,
  ].join("|"),
  "g",
);

/** Tiny highlighter for the JS snippets on the site: comments, strings, keywords, numbers, calls. */
export function highlight(code) {
  let out = "";
  let last = 0;
  for (const match of code.matchAll(JS_TOKENS)) {
    const [token, comment, string, keyword, number] = match;
    out += esc(code.slice(last, match.index));
    const kind = comment ? "c" : string ? "s" : keyword ? "k" : number ? "n" : "f";
    out += `<span class="tok-${kind}">${esc(token)}</span>`;
    last = match.index + token.length;
  }
  return out + esc(code.slice(last));
}

/** Adds copy buttons to every `.code` block under `root`. */
export function wireCopy(root = document) {
  for (const block of root.querySelectorAll(".code")) {
    if (block.querySelector(".copy")) continue;
    const button = document.createElement("button");
    button.className = "btn small ghost copy";
    button.type = "button";
    button.textContent = "Copy";
    button.addEventListener("click", async () => {
      const text = block.querySelector("pre").innerText;
      try {
        await navigator.clipboard.writeText(text);
        button.textContent = "Copied";
      } catch {
        button.textContent = "Select and copy";
      }
      setTimeout(() => (button.textContent = "Copy"), 1400);
    });
    block.appendChild(button);
  }
}


/**
 * Runs `items` ({ state, questions }) in growing batches (1, 2, 4, 8, ...) and calls
 * `onEach(response, index)` as soon as each batch returns, so the first answers show within
 * one short pass instead of after the whole set; `onBatch(done)` follows each batch. Resolves
 * to all responses in order, or null once `isStale()` says the run is no longer wanted.
 */
export async function decideStream(kevala, items, onEach, { isStale = () => false, onBatch } = {}) {
  const out = new Array(items.length);
  let start = 0;
  let batchSize = 1;
  while (start < items.length) {
    if (isStale()) return null;
    const batch = items.slice(start, start + batchSize);
    const responses =
      batch.length === 1 ? [await kevala.decide(batch[0].state, batch[0].questions)] : await kevala.decideMany(batch);
    if (isStale()) return null;
    responses.forEach((response, j) => {
      out[start + j] = response;
      onEach?.(response, start + j);
    });
    start += batch.length;
    onBatch?.(start);
    batchSize *= 2;
  }
  return out;
}

// The model gate: shown in a view until the session's model is ready

function gateLoadingHTML(name) {
  return [
    `<div class="gate-row">`,
    `<span class="spin"></span>`,
    `<div class="gate-t"><b>Loading ${name}…</b><span data-f="label"></span></div>`,
    `<button type="button" class="btn small ghost" data-act="cancel">Cancel</button>`,
    `</div>`,
    `<div class="progress"><i></i></div>`,
  ].join("");
}

function gateIdleHTML(s, name, what) {
  const failed = s.status === "error";
  return [
    `<div class="gate-row">`,
    `<div class="gate-t"><b>Load ${name} to ${esc(what)}</b><span>${esc(s.costLine())}</span></div>`,
    `<button type="button" class="btn primary" data-act="load">${failed ? "Retry" : `Load ${name}`}</button>`,
    `</div>`,
    failed ? `<p class="gate-err">Could not load: ${esc(s.error || "unknown error")}</p>` : "",
  ].join("");
}

/**
 * Fills `el` with a compact "load the model" card that follows the session: a load button,
 * then live progress, then it hides itself. `what` says what the model is needed for.
 * Returns the unsubscribe function.
 */
export function modelGate(el, what = "run this demo") {
  el.classList.add("gate");
  let builtFor = "";
  const render = (s) => {
    el.classList.toggle("hidden", s.ready);
    if (s.ready) return (builtFor = "ready");
    // rebuild only when the card's content changes, so progress updates do not flicker
    const key = `${s.status}|${s.model}|${s.cached[s.model]}|${s.error}`;
    if (key !== builtFor) {
      builtFor = key;
      const name = esc(s.nameOf());
      el.innerHTML = s.status === "loading" ? gateLoadingHTML(name) : gateIdleHTML(s, name, what);
    }
    if (s.status === "loading") {
      const progress = s.progress || {};
      el.querySelector('[data-f="label"]').textContent = progress.label || "";
      const bar = el.querySelector(".progress");
      bar.classList.toggle("indet", !!progress.indet);
      bar.firstElementChild.style.width = `${Math.round((progress.frac || 0) * 100)}%`;
    }
  };
  el.addEventListener("click", (e) => {
    const action = e.target.closest("[data-act]")?.dataset.act;
    if (action === "load") session.load();
    if (action === "cancel") session.cancel();
  });
  render(session);
  return session.on(render);
}

/** Adds a stylesheet (resolved against the calling module's URL) once. */
export function css(href) {
  if (document.querySelector(`link[data-css="${href}"]`)) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = href;
  link.dataset.css = href;
  document.head.appendChild(link);
}

export const REPO = "https://github.com/bvolpato/kevala";

/** The kevala mark; `id` names its gradient, so two marks on one page need different ids. */
export function logo(id) {
  return [
    `<svg viewBox="0 0 32 32" aria-hidden="true">`,
    `<defs><linearGradient id="${id}" x1="0" y1="0" x2="1" y2="1">`,
    `<stop offset="0" stop-color="#7aa2ff"/><stop offset=".55" stop-color="#45e0c0"/><stop offset="1" stop-color="#c592ff"/>`,
    `</linearGradient></defs>`,
    `<rect x="1" y="1" width="30" height="30" rx="9" fill="#0f131b" stroke="url(#${id})" stroke-width="2"/>`,
    `<path d="M11.5 7.5v17M21 12.5l-9 6.5M15.6 16.4l5.9 8.1" fill="none" stroke="url(#${id})" stroke-width="2.6"` +
      ` stroke-linecap="round" stroke-linejoin="round"/>`,
    `</svg>`,
  ].join("");
}

/** Where pages import the library from: the npm package on jsDelivr. */
export const CDN = "https://cdn.jsdelivr.net/npm/kevala@latest/js/src/index.js";

/**
 * Formats a value as JavaScript source for the code snippets on the site: unquoted keys where
 * they are identifiers, short arrays and objects on one line, 2-space indentation.
 */
export function js(value, indent = "") {
  const inner = indent + "  ";
  const fitsOneLine = (s) => s.length < 80 && !s.includes("\n");
  if (Array.isArray(value)) {
    const items = value.map((x) => js(x, inner));
    const oneLine = `[${items.join(", ")}]`;
    return fitsOneLine(oneLine) ? oneLine : `[\n${items.map((x) => inner + x).join(",\n")},\n${indent}]`;
  }
  if (value && typeof value === "object") {
    const key = (k) => (/^[A-Za-z_$][\w$]*$/.test(k) ? k : JSON.stringify(k));
    const entries = Object.entries(value).map(([k, x]) => `${key(k)}: ${js(x, inner)}`);
    const oneLine = `{ ${entries.join(", ")} }`;
    return fitsOneLine(oneLine) ? oneLine : `{\n${entries.map((x) => inner + x).join(",\n")},\n${indent}}`;
  }
  return JSON.stringify(value);
}

export { session, MODELS };
