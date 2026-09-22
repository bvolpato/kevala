// Small UI helpers shared by the views: formatting, the answer bars, syntax highlighting and
// the "load a model" gate. Nothing here talks to the network.

import { session, MODELS } from "./session.js";

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
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
  let t = 0;
  const d = (...a) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...a), ms);
  };
  d.cancel = () => clearTimeout(t);
  return d;
}

export function backendLabel(info) {
  if (!info) return "";
  if (info.backend === "webgpu") return "WebGPU";
  return `CPU · ${info.threads > 1 ? `${info.threads} workers` : "1 thread"}`;
}

export function backendBadge(info) {
  if (!info) return `<span class="badge"><span class="dot"></span>not loaded</span>`;
  const gpu = info.backend === "webgpu";
  return `<span class="badge ${gpu ? "gpu" : "cpu"}" title="${esc(info.backend)}${info.gpu ? ` · ${esc(info.gpu)}` : ""}"><span class="dot"></span>${esc(backendLabel(info))}</span>`;
}

// ---------------------------------------------------------------------------------------------
// syntax highlighting

/**
 * Highlights JSON (or JSON-like text being typed). Every character of the input is kept, so the
 * result can sit under a textarea as its highlighted twin.
 */
export function highlightJSON(text) {
  const re = /("(?:[^"\\\n]|\\.)*"?)(\s*:)?|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|\b(true|false|null)\b|([{}[\],])/g;
  let out = "";
  let last = 0;
  let m;
  while ((m = re.exec(text))) {
    out += esc(text.slice(last, m.index));
    if (m[1] !== undefined) out += m[2] ? `<span class="j-k">${esc(m[1])}</span>${esc(m[2])}` : `<span class="j-s">${esc(m[1])}</span>`;
    else if (m[3] !== undefined) out += `<span class="j-n">${m[3]}</span>`;
    else if (m[4] !== undefined) out += `<span class="j-b">${m[4]}</span>`;
    else out += `<span class="j-p">${esc(m[5])}</span>`;
    last = m.index + m[0].length;
  }
  return out + esc(text.slice(last));
}

/** Tiny highlighter for the JS snippets on the site (keywords, strings, comments). */
export function highlight(code) {
  const out = [];
  const re = /(\/\/[^\n]*)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)|\b(import|from|const|let|await|async|return|if|else|new|export|function|for|of)\b|\b(\d+(?:\.\d+)?)\b|\b([A-Za-z_$][\w$]*)(?=\()/g;
  let last = 0;
  let m;
  while ((m = re.exec(code))) {
    out.push(esc(code.slice(last, m.index)));
    const cls = m[1] ? "c" : m[2] ? "s" : m[3] ? "k" : m[4] ? "n" : "f";
    out.push(`<span class="tok-${cls}">${esc(m[0])}</span>`);
    last = m.index + m[0].length;
  }
  out.push(esc(code.slice(last)));
  return out.join("");
}

/** Adds copy buttons to every `.code` block under `root`. */
export function wireCopy(root = document) {
  for (const block of root.querySelectorAll(".code")) {
    if (block.querySelector(".copy")) continue;
    const b = document.createElement("button");
    b.className = "btn small ghost copy";
    b.type = "button";
    b.textContent = "Copy";
    b.addEventListener("click", async () => {
      const text = block.querySelector("pre").innerText;
      try {
        await navigator.clipboard.writeText(text);
        b.textContent = "Copied";
      } catch {
        b.textContent = "Select and copy";
      }
      setTimeout(() => (b.textContent = "Copy"), 1400);
    });
    block.appendChild(b);
  }
}

// ---------------------------------------------------------------------------------------------
// answers: one card per question, bars keyed by option so updates animate in place

function optionsOf(a) {
  if (a.type === "noul") return [["true", a.noul], ["false", 1 - a.noul]];
  return Object.entries(a.probabilities || {});
}

function argmax(probs) {
  let best = null;
  for (const [k, p] of Object.entries(probs || {})) if (best === null || p > probs[best]) best = k;
  return best;
}

function verdictOf(a) {
  if (a.type === "noul") return a.noul >= 0.5 ? "yes" : "no";
  if (a.type === "choice") return a.choice;
  if (a.type === "score") {
    const k = argmax(a.probabilities);
    const label = a.legend && typeof a.legend === "object" ? a.legend[k] : a.legend;
    return `${label ?? k}${typeof a.score === "number" ? `  ·  ${a.score.toFixed(2)}` : ""}`;
  }
  return "";
}

/** Renders or updates answers into `el`. `labels` can rename options per question. */
export function renderAnswers(el, response, questions = {}, { labels = {}, max = 0 } = {}) {
  el.classList.add("answers");
  const answers = response?.answers || {};
  const seen = new Set();
  for (const id of Object.keys(answers)) {
    seen.add(id);
    const a = answers[id];
    let card = el.querySelector(`[data-q="${CSS.escape(id)}"]`);
    if (!card) {
      card = document.createElement("div");
      card.className = "answer";
      card.dataset.q = id;
      card.innerHTML = `<div class="head"><span class="qid"></span><span class="qtype"></span></div><div class="instr"></div><div class="row" style="justify-content:space-between"><span class="verdict"></span></div><div class="bars"></div><div class="meta"></div>`;
      el.appendChild(card);
    }
    card.querySelector(".qid").textContent = id;
    card.querySelector(".qtype").textContent = a.type;
    card.querySelector(".instr").textContent = questions[id]?.instructions || "";
    let opts = optionsOf(a);
    const top = opts.reduce((b, o) => (o[1] > b[1] ? o : b), opts[0] || ["", 0]);
    const hidden = max && opts.length > max ? opts.length - max : 0;
    if (hidden) opts = opts.slice().sort((x, y) => y[1] - x[1]).slice(0, max);
    const v = card.querySelector(".verdict");
    v.textContent = verdictOf(a);
    v.style.color = a.type === "noul" ? (a.noul >= 0.5 ? "var(--accent-2)" : "var(--text-2)") : "var(--text)";
    const bars = card.querySelector(".bars");
    const keep = new Set();
    for (const [k, p] of opts) {
      keep.add(k);
      let b = bars.querySelector(`[data-k="${CSS.escape(k)}"]`);
      if (!b) {
        b = document.createElement("div");
        b.className = "bar";
        b.dataset.k = k;
        b.innerHTML = `<span class="k"></span><span class="track"><span class="fill"></span></span><span class="v"></span>`;
        bars.appendChild(b);
      }
      const label = labels[id]?.[k] ?? (a.type === "score" && a.legend && typeof a.legend === "object" ? a.legend[k] ?? k : k);
      b.querySelector(".k").textContent = label;
      b.querySelector(".k").title = label;
      b.querySelector(".v").textContent = `${(p * 100).toFixed(p >= 0.995 || p < 0.005 ? 0 : 1)}%`;
      b.classList.toggle("top", k === top[0]);
      requestAnimationFrame(() => (b.querySelector(".fill").style.width = `${(p * 100).toFixed(2)}%`));
    }
    for (const b of [...bars.children]) if (!keep.has(b.dataset.k)) b.remove();
    if (hidden) for (const [k] of opts) bars.appendChild(bars.querySelector(`[data-k="${CSS.escape(k)}"]`));
    const meta = [];
    if (hidden) meta.push(`+${hidden} more`);
    if (a.confidence != null) meta.push(`confidence ${a.confidence.toFixed(2)}`);
    if (a.action?.act_probability != null) meta.push(`act ${a.action.act_probability.toFixed(2)}`);
    card.querySelector(".meta").textContent = meta.join("  ·  ");
  }
  for (const c of [...el.children]) if (c.dataset.q && !seen.has(c.dataset.q)) c.remove();
}

/**
 * Runs `items` ({ state, questions }) in growing batches (1, 2, 4, 8, ...) and calls
 * `onEach(response, index)` as soon as each batch returns, so the first answers show within
 * one short pass instead of after the whole set. Resolves to all responses in order.
 */
export async function decideStream(kevala, items, onEach, { isStale = () => false } = {}) {
  const out = new Array(items.length);
  let i = 0;
  let n = 1;
  while (i < items.length) {
    if (isStale()) return null;
    const part = items.slice(i, i + n);
    const rs = part.length === 1 ? [await kevala.decide(part[0].state, part[0].questions)] : await kevala.decideMany(part);
    if (isStale()) return null;
    rs.forEach((r, j) => {
      out[i + j] = r;
      onEach?.(r, i + j);
    });
    i += part.length;
    n *= 2;
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// the model gate: shown in a view until the session's model is ready

/**
 * Fills `el` with a compact "load the model" card that follows the session: a load button,
 * then live progress, then it hides itself. `what` says what the model is needed for.
 */
export function modelGate(el, what = "run this demo") {
  el.classList.add("gate");
  let mode = "";
  const build = (s) => {
    const name = esc(s.nameOf());
    if (s.status === "loading") {
      el.innerHTML = `<div class="gate-row"><span class="spin"></span><div class="gate-t"><b>Loading ${name}…</b><span data-f="label"></span></div><button type="button" class="btn small ghost" data-act="cancel">Cancel</button></div>
        <div class="progress"><i></i></div>
        <p class="tiny faint">Keep browsing: the download continues when you switch tabs.</p>`;
    } else {
      const err = s.status === "error" ? `<p class="gate-err">Could not load: ${esc(s.error || "unknown error")}</p>` : "";
      el.innerHTML = `<div class="gate-row"><div class="gate-t"><b>Load ${name} to ${esc(what)}</b><span>${esc(s.costLine())}</span></div><button type="button" class="btn primary" data-act="load">${s.status === "error" ? "Retry" : `Load ${name}`}</button></div>${err}`;
    }
  };
  const render = (s) => {
    el.classList.toggle("hidden", s.ready);
    if (s.ready) return (mode = "ready");
    const m = `${s.status}|${s.model}|${s.cached[s.model]}|${s.error}`;
    if (m !== mode) {
      mode = m;
      build(s);
    }
    if (s.status === "loading") {
      const p = s.progress || {};
      el.querySelector('[data-f="label"]').textContent = p.label || "";
      const bar = el.querySelector(".progress");
      bar.classList.toggle("indet", !!p.indet);
      bar.firstElementChild.style.width = `${Math.round((p.frac || 0) * 100)}%`;
    }
  };
  el.addEventListener("click", (e) => {
    const b = e.target.closest("[data-act]");
    if (!b) return;
    if (b.dataset.act === "load") session.load();
    if (b.dataset.act === "cancel") session.cancel();
  });
  render(session);
  return session.on(render);
}

/** Adds a stylesheet (resolved against the calling module's URL) once. */
export function css(href) {
  if (document.querySelector(`link[data-css="${href}"]`)) return;
  const l = document.createElement("link");
  l.rel = "stylesheet";
  l.href = href;
  l.dataset.css = href;
  document.head.appendChild(l);
}

export const REPO = "https://github.com/bvolpato/kevala";

/** Where pages import the library from: the npm package on jsDelivr. */
export const CDN = "https://cdn.jsdelivr.net/npm/kevala@0.1/js/src/index.js";

/**
 * Formats a value as JavaScript source for the code snippets on the site: unquoted keys where
 * they are identifiers, short arrays and objects on one line, 2-space indentation.
 */
export function js(v, indent = "") {
  const inner = indent + "  ";
  if (Array.isArray(v)) {
    const one = `[${v.map((x) => js(x, inner)).join(", ")}]`;
    return one.length < 80 && !one.includes("\n") ? one : `[\n${v.map((x) => inner + js(x, inner)).join(",\n")},\n${indent}]`;
  }
  if (v && typeof v === "object") {
    const key = (k) => (/^[A-Za-z_$][\w$]*$/.test(k) ? k : JSON.stringify(k));
    const parts = Object.entries(v).map(([k, x]) => `${key(k)}: ${js(x, inner)}`);
    const one = `{ ${parts.join(", ")} }`;
    return one.length < 80 && !one.includes("\n") ? one : `{\n${parts.map((x) => inner + x).join(",\n")},\n${indent}}`;
  }
  return JSON.stringify(v);
}
export { session, MODELS };
