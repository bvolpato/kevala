// One model for the whole site. The header picks and loads it; every view reads it from here.
// Loading lives in this module, not in a view, so switching tabs never interrupts a download.
// The choice sticks for the browser session (sessionStorage): after a reload, a model that was
// loaded comes back by itself when its pack is cached.

import { Kevala, MODELS, cacheInfo, clearCache, isCached } from "../js/src/index.js";

export { MODELS };

export const params = new URLSearchParams(location.search);

/** `?pack=local` loads the dev packs from /tmp instead of converting from Hugging Face. */
export const LOCAL = params.get("pack") === "local";
const LOCAL_PACKS = {
  laya: new URL("../tmp/laya-q8.kevala", import.meta.url).href,
  "kev-0.8b": new URL("../tmp/kev-0.8b-q8.kevala", import.meta.url).href,
};
/** `?shot=1` hides dev-only chrome, for README screenshots taken with dev packs. */
export const SHOT = params.get("shot") === "1";

/** What the site says about each model. */
export const MODEL_NOTES = {
  laya: {
    name: "Laya",
    short: "ModernBERT-large encoder + decision head, 421M",
    note: "WebGPU: tens of ms per decision. CPU fallback everywhere.",
  },
  "kev-0.8b": {
    name: "Kev-0.8B",
    short: "Qwen3.5-0.8B hybrid decoder + pointer head",
    note: "WebGPU: tens of ms per request, less for repeated states. The CPU fallback takes seconds.",
  },
};

const KEY = "kevala.session";
function readSaved() {
  try {
    return JSON.parse(sessionStorage.getItem(KEY)) || JSON.parse(localStorage.getItem("kevala.pref")) || {};
  } catch {
    return {};
  }
}

function fmtBytes(n) {
  if (!n) return "0 B";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  return `${Math.round(n / 1e3)} KB`;
}

class Session extends EventTarget {
  constructor() {
    super();
    const saved = readSaved();
    this.saved = saved;
    this.model = MODELS[params.get("model")] ? params.get("model") : MODEL_NOTES[saved.model] || saved.model === "custom" ? saved.model : "laya";
    this.backend = ["auto", "webgpu", "wasm"].includes(params.get("backend")) ? params.get("backend") : saved.backend || "auto";
    this.customUrl = saved.customUrl || "";
    /** idle | loading | ready | error */
    this.status = "idle";
    /** { frac, indet, label } while loading */
    this.progress = null;
    this.kevala = null;
    this.error = null;
    this.cached = {};
    this.ctrl = null;
    this.waiters = [];
  }

  get ready() {
    return this.status === "ready";
  }

  get info() {
    return this.kevala?.info || null;
  }

  nameOf(m = this.model) {
    return MODEL_NOTES[m]?.name || "Custom pack";
  }

  /** Calls `fn(session)` on every change; returns the unsubscribe function. */
  on(fn) {
    const h = () => fn(this);
    this.addEventListener("change", h);
    return () => this.removeEventListener("change", h);
  }

  emit() {
    this.dispatchEvent(new Event("change"));
  }

  persist() {
    const v = JSON.stringify({ model: this.model, backend: this.backend, customUrl: this.customUrl, active: this.status === "ready" || this.status === "loading" });
    try {
      sessionStorage.setItem(KEY, v);
      localStorage.setItem("kevala.pref", JSON.stringify({ model: this.model, backend: this.backend, customUrl: this.customUrl }));
    } catch {}
  }

  /** The value to pass as `model` to Kevala.load. */
  source(m = this.model) {
    if (m === "custom") return this.customUrl || null;
    if (LOCAL && LOCAL_PACKS[m]) return LOCAL_PACKS[m];
    return m;
  }

  async refreshCache() {
    for (const m of Object.keys(MODEL_NOTES)) this.cached[m] = await isCached(this.source(m)).catch(() => false);
    const info = await cacheInfo().catch(() => ({ available: false, bytes: 0, entries: [] }));
    this.storage = info;
    this.emit();
  }

  /** Reloads the model of the last visit when it was loaded and is cached (or local). */
  async boot() {
    await this.refreshCache();
    if (params.get("autoload") === "1" || (this.saved.active && (LOCAL || this.cached[this.model]))) this.load();
  }

  /** Resolves with the loaded Kevala, or null when loading fails or is cancelled. */
  whenReady() {
    if (this.status === "ready") return Promise.resolve(this.kevala);
    if (this.status !== "loading") return Promise.resolve(null);
    return new Promise((r) => this.waiters.push(r));
  }

  settle(v) {
    for (const r of this.waiters.splice(0)) r(v);
  }

  select(m) {
    if (m === this.model) return;
    this.cancel();
    this.unload();
    this.model = m;
    this.error = null;
    this.persist();
    this.emit();
  }

  setBackend(b) {
    if (b === this.backend) return;
    const was = this.status;
    this.backend = b;
    this.persist();
    if (was === "ready" || was === "loading") {
      this.cancel();
      this.unload();
      this.load();
    } else this.emit();
  }

  setCustomUrl(u) {
    this.customUrl = u.trim();
    this.persist();
    this.emit();
  }

  async load(m) {
    if (m && m !== this.model) this.select(m);
    if (this.kevala || this.ctrl) return this.whenReady();
    const src = this.source();
    if (!src) {
      this.status = "error";
      this.error = "Paste the URL of a .kevala pack first.";
      this.emit();
      return null;
    }
    const ctrl = new AbortController();
    this.ctrl = ctrl;
    this.status = "loading";
    this.error = null;
    this.progress = { frac: 0, indet: true, label: "Starting…" };
    this.persist();
    this.emit();
    const t0 = performance.now();
    let conv = "";
    let dl = 0;
    const set = (frac, indet, label) => {
      this.progress = { frac, indet, label };
      this.emit();
    };
    try {
      const kevala = await Kevala.load({
        model: src,
        backend: this.backend,
        signal: ctrl.signal,
        onProgress: (p) => {
          if (ctrl.signal.aborted) return;
          if (p.phase === "download" && p.total) {
            dl = p.total;
            const secs = (performance.now() - t0) / 1000;
            const rate = secs > 0.5 ? ` · ${fmtBytes(p.loaded / secs)}/s` : "";
            set(p.loaded / p.total, false, `Downloading ${fmtBytes(p.loaded)} of ${fmtBytes(p.total)}${rate}${conv}`);
          } else if (p.phase === "download") {
            set(0, true, `Fetching ${p.file || "files"}…`);
          } else if (p.phase === "convert") {
            conv = ` · ${p.loaded}/${p.total} tensors converted`;
            if (!dl) set(p.loaded / p.total, false, `Converting to int8: ${p.loaded}/${p.total} tensors`);
          } else if (p.phase === "cache") {
            set(1, !p.total, p.loaded ? `Reading ${fmtBytes(p.total)} from browser storage…` : "Saving the pack to browser storage…");
          } else if (p.phase === "init") {
            set(1, true, p.message || "Initializing…");
          } else if (p.phase === "warmup") {
            set(1, true, "Warming up…");
          }
        },
      });
      if (this.ctrl !== ctrl) {
        kevala.dispose();
        return null;
      }
      this.ctrl = null;
      this.kevala = kevala;
      this.status = "ready";
      this.progress = null;
      this.persist();
      this.emit();
      this.settle(kevala);
      this.refreshCache();
      return kevala;
    } catch (e) {
      if (this.ctrl !== ctrl) return null;
      this.ctrl = null;
      const aborted = e?.name === "AbortError" || /abort|cancel/i.test(e?.message || "");
      this.status = aborted ? "idle" : "error";
      this.error = aborted ? null : e?.message || String(e);
      this.progress = null;
      this.persist();
      this.emit();
      this.settle(null);
      return null;
    }
  }

  cancel() {
    if (!this.ctrl) return;
    this.ctrl.abort();
    this.ctrl = null;
    this.status = "idle";
    this.progress = null;
    this.persist();
    this.emit();
    this.settle(null);
  }

  unload() {
    if (!this.kevala) return;
    this.kevala.dispose();
    this.kevala = null;
    this.status = "idle";
    this.persist();
    this.emit();
  }

  async clearCache() {
    await clearCache();
    await this.refreshCache();
  }

  /** A one-line description of what loading the current choice costs. */
  costLine(m = this.model) {
    const spec = MODELS[m];
    if (LOCAL && !SHOT) return "Dev mode: loads the local pack from this server.";
    if (this.cached[m]) return "Cached in this browser: loads in about a second.";
    if (!spec) return "Loads the pack from its URL and caches it.";
    return `First load: ${fmtBytes(spec.download)} from Hugging Face, converted to a ${fmtBytes(spec.pack)} int8 pack and cached.`;
  }
}

export const session = new Session();
