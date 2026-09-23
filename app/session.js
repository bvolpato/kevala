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
  "kev-4b": new URL("../tmp/kev-4b-q8.kevala", import.meta.url).href,
  "kev-9b": new URL("../tmp/kev-9b-q8.kevala", import.meta.url).href,
  "semif-qwen3.5-0.8b": new URL("../tmp/semif-qwen3.5-0.8b-q8.kevala", import.meta.url).href,
  "semif-qwen3.5-2b": new URL("../tmp/semif-qwen3.5-2b-q8.kevala", import.meta.url).href,
  "semif-qwen3.5-4b": new URL("../tmp/semif-qwen3.5-4b-q8.kevala", import.meta.url).href,
  "gemma-4-e2b": new URL("../tmp/gemma-4-e2b-q8.kevala", import.meta.url).href,
  "gemma-4-e4b": new URL("../tmp/gemma-4-e4b-q8.kevala", import.meta.url).href,
};
/** `?from=checkpoint` converts the original weights in the browser instead of downloading the pack. */
export const FROM = params.get("from") === "checkpoint" ? "checkpoint" : "pack";
/** `?shot=1` hides dev-only chrome, for README screenshots taken with dev packs. */
export const SHOT = params.get("shot") === "1";

/** What the model menu says about each model. */
export const MODEL_NOTES = {
  laya: {
    name: "Laya",
    short: "ModernBERT-large encoder + decision head, 421M",
  },
  "kev-0.8b": {
    name: "Kev-0.8B",
    short: "Qwen3.5-0.8B hybrid decoder + pointer head",
  },
  "kev-4b": {
    name: "Kev-4B",
    short: "Qwen3.5-4B hybrid decoder + pointer head",
    large: true,
  },
  "kev-9b": {
    name: "Kev-9B",
    short: "Qwen3.5-9B hybrid decoder + pointer head",
    large: true,
  },
  "semif-qwen3.5-0.8b": {
    name: "SemIf-0.8B",
    short: "Direct option scores from a frozen Qwen3.5-0.8B model",
    large: true,
  },
  "semif-qwen3.5-2b": {
    name: "SemIf-2B",
    short: "Direct option scores from a frozen Qwen3.5-2B model",
    large: true,
  },
  "semif-qwen3.5-4b": {
    name: "SemIf-4B",
    short: "Direct option scores from a frozen Qwen3.5-4B model",
    large: true,
  },
  "gemma-4-e2b": {
    name: "Gemma 4 E2B",
    short: "Direct option scores from Gemma 4 E2B instruction weights; text only",
    large: true,
    requiresWebGPU: true,
  },
  "gemma-4-e4b": {
    name: "Gemma 4 E4B",
    short: "Direct option scores from Gemma 4 E4B instruction weights; text only",
    large: true,
    requiresWebGPU: true,
  },
};

const BACKENDS = ["auto", "webgpu", "wasm"];
const SESSION_KEY = "kevala.session";
const PREF_KEY = "kevala.pref";

/** The choice of this browser session, or else the last one made in this browser. */
function readSaved() {
  try {
    return JSON.parse(sessionStorage.getItem(SESSION_KEY)) || JSON.parse(localStorage.getItem(PREF_KEY)) || {};
  } catch {
    return {};
  }
}

/** The URL's `?model=` wins, then the saved choice, then Laya. */
function initialModel(saved) {
  const fromUrl = params.get("model");
  if (MODELS[fromUrl]) return fromUrl;
  if (MODEL_NOTES[saved.model] || saved.model === "custom") return saved.model;
  return "laya";
}

function initialBackend(saved) {
  const fromUrl = params.get("backend");
  return BACKENDS.includes(fromUrl) ? fromUrl : saved.backend || "auto";
}

// a copy of ui.js's fmtBytes: ui.js imports this module, so this one cannot import it back
function fmtBytes(n) {
  if (!n) return "0 B";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  return `${Math.round(n / 1e3)} KB`;
}

/** The progress line for one `onProgress` event of Kevala.load, or null to leave it as is. */
function progressOf(event, { startedAt, convertNote, downloadTotal }) {
  const { phase, loaded, total } = event;
  if (phase === "download" && total) {
    const secs = (performance.now() - startedAt) / 1000;
    const rate = secs > 0.5 ? ` · ${fmtBytes(loaded / secs)}/s` : "";
    const label = `Downloading ${fmtBytes(loaded)} of ${fmtBytes(total)}${rate}${convertNote}`;
    return { frac: loaded / total, indet: false, label };
  }
  if (phase === "download") return { frac: 0, indet: true, label: `Fetching ${event.file || "files"}…` };
  if (phase === "convert") {
    // conversion runs while the download streams; it gets its own bar only when nothing downloads
    if (downloadTotal) return null;
    return { frac: loaded / total, indet: false, label: `Converting to int8: ${loaded}/${total} tensors` };
  }
  if (phase === "cache") {
    const label = loaded ? `Reading ${fmtBytes(total)} from browser storage…` : "Saving the pack to browser storage…";
    return { frac: 1, indet: !total, label };
  }
  if (phase === "init") return { frac: 1, indet: true, label: event.message || "Initializing…" };
  if (phase === "warmup") return { frac: 1, indet: true, label: "Warming up…" };
  return null;
}

class Session extends EventTarget {
  #saved;
  #abort = null;
  #waiters = [];

  constructor() {
    super();
    this.#saved = readSaved();
    this.model = initialModel(this.#saved);
    this.backend = initialBackend(this.#saved);
    if (this.backend === "wasm" && MODEL_NOTES[this.model]?.requiresWebGPU) this.backend = "auto";
    this.customUrl = this.#saved.customUrl || "";
    /** idle | loading | ready | error */
    this.status = "idle";
    /** { frac, indet, label } while loading */
    this.progress = null;
    this.kevala = null;
    this.loadedOptions = null;
    this.error = null;
    this.cached = {};
  }

  get ready() {
    return this.status === "ready";
  }

  get info() {
    return this.kevala?.info || null;
  }

  get from() {
    return FROM;
  }

  nameOf(model = this.model) {
    return MODEL_NOTES[model]?.name || "Custom pack";
  }

  /** Calls `fn(session)` on every change; returns the unsubscribe function. */
  on(fn) {
    const handler = () => fn(this);
    this.addEventListener("change", handler);
    return () => this.removeEventListener("change", handler);
  }

  #emit() {
    this.dispatchEvent(new Event("change"));
  }

  #persist() {
    const choice = { model: this.model, backend: this.backend, customUrl: this.customUrl };
    const active = this.status === "ready" || this.status === "loading";
    try {
      sessionStorage.setItem(SESSION_KEY, JSON.stringify({ ...choice, active }));
      localStorage.setItem(PREF_KEY, JSON.stringify(choice));
    } catch {}
  }

  /** The value to pass as `model` to Kevala.load. */
  source(model = this.model) {
    if (model === "custom") return this.customUrl || null;
    if (LOCAL && LOCAL_PACKS[model]) return LOCAL_PACKS[model];
    return model;
  }

  async refreshCache() {
    for (const model of Object.keys(MODEL_NOTES)) {
      this.cached[model] = await isCached(this.source(model)).catch(() => false);
    }
    this.storage = await cacheInfo().catch(() => ({ available: false, bytes: 0, entries: [] }));
    this.#emit();
  }

  /** Reloads the model of the last visit when it was loaded and is cached (or local). */
  async boot() {
    await this.refreshCache();
    const wasActive = this.#saved.active && (LOCAL || this.cached[this.model]);
    if (params.get("autoload") === "1" || wasActive) this.load();
  }

  /** Resolves with the loaded Kevala, or null when loading fails or is cancelled. */
  whenReady() {
    if (this.status === "ready") return Promise.resolve(this.kevala);
    if (this.status !== "loading") return Promise.resolve(null);
    return new Promise((resolve) => this.#waiters.push(resolve));
  }

  #settle(kevala) {
    for (const resolve of this.#waiters.splice(0)) resolve(kevala);
  }

  select(model) {
    if (model === this.model) return;
    this.cancel();
    this.unload();
    this.model = model;
    if (this.backend === "wasm" && MODEL_NOTES[model]?.requiresWebGPU) this.backend = "auto";
    this.error = null;
    this.#persist();
    this.#emit();
  }

  /**
   * Switching backend never downloads again: a loaded model reopens from browser storage in about
   * a second, and a model still downloading finishes first (its pack is stored), then reopens.
   */
  setBackend(backend) {
    if (backend === "wasm" && MODEL_NOTES[this.model]?.requiresWebGPU) return;
    if (backend === this.backend) return;
    this.backend = backend;
    this.#persist();
    if (this.status === "ready") {
      this.unload();
      this.load();
    } else this.#emit();
  }

  setCustomUrl(url) {
    this.customUrl = url.trim();
    this.#persist();
    this.#emit();
  }

  async load(model) {
    if (model && model !== this.model) this.select(model);
    if (this.kevala || this.#abort) return this.whenReady();
    const src = this.source();
    if (!src) {
      this.status = "error";
      this.error = "Paste the URL of a .kevala pack first.";
      this.#emit();
      return null;
    }
    const abort = new AbortController();
    this.#abort = abort;
    this.status = "loading";
    this.error = null;
    this.progress = { frac: 0, indet: true, label: "Starting…" };
    this.#persist();
    this.#emit();
    const tracking = { startedAt: performance.now(), convertNote: "", downloadTotal: 0 };
    this.storageError = null;
    const onProgress = (event) => {
      if (abort.signal.aborted) return;
      if (event.phase === "cache-failed") {
        // the model still loads; the next visit will have to download it again
        this.storageError = event.message;
        return;
      }
      if (event.phase === "download" && event.total) tracking.downloadTotal = event.total;
      if (event.phase === "convert") tracking.convertNote = ` · ${event.loaded}/${event.total} tensors converted`;
      const progress = progressOf(event, tracking);
      if (!progress) return;
      this.progress = progress;
      this.#emit();
    };
    const backend = this.backend;
    try {
      const kevala = await Kevala.load({ model: src, from: FROM, backend, signal: abort.signal, onProgress });
      if (this.#abort !== abort) {
        kevala.dispose();
        return null;
      }
      this.#abort = null;
      if (this.backend !== backend) {
        // the backend changed while the pack downloaded: it is stored now, so reopen from disk
        kevala.dispose();
        return this.load();
      }
      this.kevala = kevala;
      this.loadedOptions = { model: src, from: FROM, backend };
      this.status = "ready";
      this.progress = null;
      this.#persist();
      this.#emit();
      this.#settle(kevala);
      this.refreshCache();
      return kevala;
    } catch (e) {
      if (this.#abort !== abort) return null;
      this.#abort = null;
      const aborted = e?.name === "AbortError" || /abort|cancel/i.test(e?.message || "");
      this.status = aborted ? "idle" : "error";
      this.error = aborted ? null : e?.message || String(e);
      this.progress = null;
      this.#persist();
      this.#emit();
      this.#settle(null);
      return null;
    }
  }

  cancel() {
    if (!this.#abort) return;
    this.#abort.abort();
    this.#abort = null;
    this.status = "idle";
    this.progress = null;
    this.#persist();
    this.#emit();
    this.#settle(null);
  }

  unload() {
    if (!this.kevala) return;
    this.kevala.dispose();
    this.kevala = null;
    this.loadedOptions = null;
    this.status = "idle";
    this.#persist();
    this.#emit();
  }

  async clearCache() {
    await clearCache();
    await this.refreshCache();
  }

  /** A one-line description of what loading the current choice costs. */
  costLine(model = this.model) {
    const spec = MODELS[model];
    if (LOCAL && !SHOT) return "Dev mode: loads the local pack from this server.";
    if (this.cached[model]) {
      return MODEL_NOTES[model]?.large ? "Cached in this browser: reopens locally." : "Cached in this browser: loads in about a second.";
    }
    if (!spec) return "Loads the pack from its URL and caches it.";
    if (spec.hosted && FROM === "pack") {
      return spec.pack
        ? `First load: a ${fmtBytes(spec.pack)} int8 pack from Hugging Face, then cached in this browser.`
        : "First load: an int8 pack from Hugging Face, then cached in this browser.";
    }
    return `First load: ${fmtBytes(spec.download)} from Hugging Face, converted to a ${fmtBytes(spec.pack)} int8 pack and cached.`;
  }
}

export const session = new Session();
