// One model for the whole site. The header picks and loads it; every view reads it from here.
// Loading lives in this module, not in a view, so switching tabs never interrupts a download.
// The choice sticks for the browser session (sessionStorage): after a reload, a model that was
// selected stays unloaded until the user presses Load.

import { Kevala, MODELS as CATALOG_MODELS, cacheInfo, clearCache, defaultPrecision, isCached, precisionOptions } from "../js/src/index.js";
import { assertWasmPackSize } from "../js/src/pack-layout.js";
import { fmtBytes } from "./format.js";
import { discoverLocalPacks } from "./model-packs.js";
import { PREVIEW_MODEL_IDS } from "./model-policy.js";

export const MODELS = Object.fromEntries(Object.entries(CATALOG_MODELS).filter(([id]) => PREVIEW_MODEL_IDS.has(id)));
const IMAGE_MODEL = "gemma-4-e2b";

export const params = new URLSearchParams(location.search);

/** `?from=checkpoint` converts the original weights in the browser instead of downloading the pack. */
export const FROM = params.get("from") === "checkpoint" ? "checkpoint" : "pack";
const explicitLocal = params.get("pack") === "local";
const localPacks = FROM === "pack" ? await discoverLocalPacks(import.meta.url) : {};
/** The preview server advertises its packs automatically; the old query flag also works. */
export const LOCAL = explicitLocal || Object.keys(localPacks).length > 0;
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
  "h2o-lightning-4b": {
    name: "H2O-Lightning-4B",
    short: "Qwen3.5-4B with H2O's decision prompt and calibrated label scores; text only",
    large: true,
  },
  "clef-flash": {
    name: "Clef-flash",
    short: "Qwen3.5-9B with Cloudflare's joint decision head; text only",
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
    short: "Direct option scores; optional BF16 image companion for the Reaction Recorder",
    large: true,
  },
  "gemma-4-e4b": {
    name: "Gemma 4 E4B",
    short: "Direct option scores from Gemma 4 E4B instruction weights; text only",
    large: true,
  },
};

/** Whether a model's pack is larger than one WebAssembly instance can hold, so only WebGPU runs it. */
export function requiresWebGPU(model, precision = defaultPrecision(model)) {
  const spec = MODELS[model];
  const bytes = spec?.precisionPackBytes?.[precision] ?? spec?.pack;
  if (!bytes) return false;
  try {
    assertWasmPackSize(bytes);
    return false;
  } catch {
    return true;
  }
}

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
  if (MODELS[saved.model]) return saved.model;
  return "laya";
}

function initialBackend(saved) {
  const fromUrl = params.get("backend");
  return BACKENDS.includes(fromUrl) ? fromUrl : saved.backend || "auto";
}

function initialPrecision(saved, model) {
  if (saved.model !== model) return defaultPrecision(model);
  const choices = pickerPrecisionOptions(model);
  return choices.includes(saved.precision) ? saved.precision : defaultPrecision(model);
}

function pickerPrecisionOptions(model) {
  return precisionOptions(model).filter((precision) => precision === "bf16" || precision === "q8");
}

/** The progress line for one `onProgress` event of Kevala.load, or null to leave it as is. */
function progressOf(event, { startedAt, convertNote, downloadTotal, precision, local }) {
  const { phase, loaded, total } = event;
  if (phase === "download" && total) {
    const secs = (performance.now() - startedAt) / 1000;
    const rate = secs > 0.5 ? ` · ${fmtBytes(loaded / secs)}/s` : "";
    const label = `${local ? "Loading from this server:" : "Downloading"} ${fmtBytes(loaded)} of ${fmtBytes(total)}${rate}${convertNote}`;
    return { frac: loaded / total, indet: false, label };
  }
  if (phase === "download") return { frac: 0, indet: true, label: `Fetching ${event.file || "files"}…` };
  if (phase === "convert") {
    // conversion runs while the download streams; it gets its own bar only when nothing downloads
    if (downloadTotal) return null;
    const format = precision === "q8" ? "Q8 (quantized)" : precision?.toUpperCase() || "the selected";
    return { frac: loaded / total, indet: false, label: `Converting to ${format}: ${loaded}/${total} tensors` };
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
  #cacheGeneration = 0;
  #waiters = [];

  constructor() {
    super();
    this.#saved = readSaved();
    this.model = initialModel(this.#saved);
    this.backend = initialBackend(this.#saved);
    this.precision = initialPrecision(this.#saved, this.model);
    if (this.backend === "wasm" && requiresWebGPU(this.model, this.precisionFor(this.model))) this.backend = "auto";
    this.customUrl = this.#saved.customUrl || "";
    this.customUrlModel = this.#saved.customUrlModel || this.model;
    this.visionUrl = typeof this.#saved.visionUrl === "string" ? this.#saved.visionUrl : "";
    this.visionUrlModel = this.#saved.visionUrlModel || this.#saved.model || "";
    this.visionEnabled = this.model === IMAGE_MODEL && this.#saved.model === this.model
      && typeof this.#saved.visionEnabled === "boolean"
      ? this.#saved.visionEnabled
      : !!this.localVisionPackFor(this.model);
    /** idle | loading | ready | error */
    this.status = "idle";
    /** { frac, indet, label } while loading */
    this.progress = null;
    this.kevala = null;
    this.loadedOptions = null;
    this.error = null;
    this.errorKind = null;
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

  get localOnly() {
    return explicitLocal;
  }

  precisionFor(model = this.model) {
    if (model !== this.model) return defaultPrecision(model);
    const choices = pickerPrecisionOptions(model);
    return choices.includes(this.precision) ? this.precision : defaultPrecision(model);
  }

  precisionChoices(model = this.model) {
    return pickerPrecisionOptions(model);
  }

  customPackUrl(model = this.model) {
    return !explicitLocal && MODELS[model]?.nativeConversion && this.customUrlModel === model ? this.customUrl : "";
  }

  localPackFor(model = this.model, precision = this.precisionFor(model)) {
    if (this.customPackUrl(model)) return null;
    return localPacks[model]?.[precision] || null;
  }

  localVisionPackFor(model = this.model) {
    if (!MODELS[model]?.modalities?.includes("image")) return null;
    return localPacks[model]?.vision || null;
  }

  visionEnabledFor(model = this.model) {
    return model === this.model && !!MODELS[model]?.modalities?.includes("image") && this.visionEnabled;
  }

  visionUrlFor(model = this.model) {
    if (!MODELS[model]?.modalities?.includes("image")) return "";
    const local = this.localVisionPackFor(model);
    if (local) return local.url;
    return this.visionUrlModel === model ? this.visionUrl.trim() : "";
  }

  visionUrlError(url = this.visionUrlFor()) {
    if (!url) return "";
    try {
      const resolved = new URL(url, globalThis.location?.href || "http://localhost/");
      if (!["http:", "https:"].includes(resolved.protocol) || resolved.username || resolved.password) {
        return "Use an HTTP or HTTPS vision-pack URL without embedded credentials.";
      }
    } catch {
      return "Enter a valid HTTP or HTTPS vision-pack URL.";
    }
    return "";
  }

  loadUnavailable() {
    const spec = MODELS[this.model];
    if (explicitLocal && MODEL_NOTES[this.model] && !this.localPackFor()) {
      return `No matching ${this.precisionFor().toUpperCase()} pack is available in this local preview.`;
    }
    const hostedPackAvailable = FROM === "pack" && this.precisionFor() === "q8" && spec?.hosted;
    if (spec?.nativeConversion && !this.customPackUrl() && !this.localPackFor() && !hostedPackAvailable) {
      return "A converted .kevala URL is required; no matching pack is available for this precision.";
    }
    if (this.visionEnabledFor()) {
      if (!this.visionUrlFor()) return "An image vision-pack URL is required; paste a BF16 vision pack URL or turn off Enable images to load text only.";
      const visionUrlError = this.visionUrlError();
      if (visionUrlError) return visionUrlError;
    }
    const needsGpu = this.backend === "webgpu" || requiresWebGPU(this.model, this.precisionFor());
    if (needsGpu && this.gpuAvailable === undefined) return "Checking WebGPU availability…";
    if (needsGpu && this.gpuAvailable === false) {
      return "This model needs WebGPU. Open this page in a browser with WebGPU enabled, or choose Laya or Kev-0.8B for CPU use.";
    }
    return "";
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
    const choice = {
      model: this.model, backend: this.backend, precision: this.precision,
      customUrl: this.customUrl, customUrlModel: this.customUrlModel,
      visionEnabled: this.visionEnabledFor(), visionUrl: this.visionUrl, visionUrlModel: this.visionUrlModel,
    };
    const active = this.status === "ready" || this.status === "loading";
    try {
      sessionStorage.setItem(SESSION_KEY, JSON.stringify({ ...choice, active }));
      localStorage.setItem(PREF_KEY, JSON.stringify(choice));
    } catch {}
  }

  /** The value to pass as `model` to Kevala.load. */
  source(model = this.model) {
    if (model === "custom") return this.customUrl || null;
    if (MODELS[model]?.nativeConversion && this.customPackUrl(model)) return { name: model, url: this.customPackUrl(model) };
    const local = this.localPackFor(model);
    if (local) return { name: model, url: local.url };
    if (explicitLocal && MODEL_NOTES[model]) return { name: model, url: new URL(`../tmp/${model}-${this.precisionFor(model)}.kevala`, import.meta.url).href };
    return model;
  }

  async refreshCache() {
    const generation = ++this.#cacheGeneration;
    const requests = Object.keys(MODELS).map((model) => ({
      model,
      precision: this.precisionFor(model),
      source: this.source(model),
    }));
    const values = await Promise.all(requests.map(async ({ model, source, precision }) => {
      if (model !== this.model || !this.visionEnabledFor(model)) return isCached(source, precision).catch(() => false);
      const vision = this.visionUrlFor(model);
      if (!vision || this.localVisionPackFor(model) || this.localPackFor(model, precision)) return false;
      const [decoderCached, visionCached] = await Promise.all([
        isCached(source, precision).catch(() => false),
        isCached(vision, "bf16").catch(() => false),
      ]);
      return decoderCached && visionCached;
    }));
    const storage = await cacheInfo().catch(() => ({ available: false, bytes: 0, entries: [] }));
    if (generation !== this.#cacheGeneration) return;
    this.cached = Object.fromEntries(requests.map(({ model }, index) => [model, values[index]]));
    this.storage = storage;
    this.#emit();
  }

  #invalidateCache(...models) {
    this.#cacheGeneration += 1;
    for (const model of models) if (model) this.cached[model] = false;
  }

  /** Check capabilities and storage without loading model weights. */
  async boot() {
    this.gpuAvailable = !!(await navigator.gpu?.requestAdapter().catch(() => null));
    await this.refreshCache();
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
    if (!MODELS[model]) return;
    if (model === this.model) return;
    const previousModel = this.model;
    this.cancel();
    this.unload();
    this.model = model;
    this.precision = defaultPrecision(model);
    this.visionEnabled = !!this.localVisionPackFor(model);
    this.#invalidateCache(previousModel, model);
    if (this.backend === "wasm" && requiresWebGPU(model, this.precision)) this.backend = "auto";
    this.error = null;
    this.errorKind = null;
    this.#persist();
    this.#emit();
    this.refreshCache();
  }

  /** Changing backend releases the current model; press Load to reopen it. */
  setBackend(backend) {
    if (backend === "wasm" && requiresWebGPU(this.model, this.precisionFor())) return;
    if (backend === this.backend) return;
    this.backend = backend;
    this.#persist();
    if (this.status === "ready") {
      this.unload();
    } else this.#emit();
  }

  setPrecision(precision) {
    if (this.customPackUrl() || !this.precisionChoices().includes(precision) || precision === this.precisionFor()) return;
    this.cancel();
    this.unload();
    this.precision = precision;
    this.#invalidateCache(this.model);
    if (this.backend === "wasm" && requiresWebGPU(this.model, precision)) this.backend = "auto";
    this.error = null;
    this.errorKind = null;
    this.#persist();
    this.#emit();
    this.refreshCache();
  }

  setCustomUrl(url, model = this.model) {
    const next = url.trim();
    if (next === this.customPackUrl(model)) return;
    if (model === this.model && (this.status === "loading" || this.status === "ready")) {
      this.cancel();
      this.unload();
    }
    this.customUrl = next;
    this.customUrlModel = model;
    this.#invalidateCache(model);
    this.#persist();
    this.#emit();
  }

  setVisionEnabled(enabled) {
    if (!MODELS[this.model]?.modalities?.includes("image")) return;
    const next = !!enabled;
    if (next === this.visionEnabled) return;
    if (this.status === "loading" || this.status === "ready") {
      this.cancel();
      this.unload();
    }
    this.visionEnabled = next;
    if (this.status === "error") {
      this.status = "idle";
      this.error = null;
      this.errorKind = null;
    }
    this.cached[this.model] = false;
    this.#persist();
    this.#emit();
    this.refreshCache();
  }

  setVisionUrl(url, model = this.model) {
    if (!MODELS[model]?.modalities?.includes("image")) return;
    const next = url.trim();
    if (next === this.visionUrlFor(model) && this.visionUrlModel === model) return;
    if (model === this.model && this.visionEnabledFor(model) && (this.status === "loading" || this.status === "ready")) {
      this.cancel();
      this.unload();
    }
    this.visionUrl = next;
    this.visionUrlModel = model;
    if (model === this.model && this.status === "error") {
      this.status = "idle";
      this.error = null;
      this.errorKind = null;
    }
    this.cached[model] = false;
    this.#persist();
    this.#emit();
  }

  async load(model) {
    if (model && !MODELS[model]) return null;
    if (model && model !== this.model) this.select(model);
    if (this.kevala || this.#abort) return this.whenReady();
    const unavailable = this.loadUnavailable();
    if (unavailable) {
      this.status = "error";
      this.error = unavailable;
      this.errorKind = "load";
      this.#emit();
      return null;
    }
    const src = this.source();
    if (!src) {
      this.status = "error";
      this.error = "Paste the URL of a .kevala pack first.";
      this.errorKind = "load";
      this.#emit();
      return null;
    }
    const abort = new AbortController();
    this.#abort = abort;
    this.status = "loading";
    this.error = null;
    this.errorKind = null;
    this.progress = { frac: 0, indet: true, label: "Starting…" };
    this.#persist();
    this.#emit();
    const customPack = this.model === "custom" || (MODELS[this.model]?.nativeConversion && this.customPackUrl());
    const selectedPrecision = customPack ? undefined : this.precisionFor();
    const tracking = { startedAt: performance.now(), convertNote: "", downloadTotal: 0, precision: selectedPrecision, local: !!this.localPackFor() };
    this.storageError = null;
    const onProgress = (event) => {
      if (abort.signal.aborted) return;
      if (event.phase === "cache-failed") {
        // the model still loads; the next visit will have to download it again
        this.storageError = event.message;
        return;
      }
      if (event.phase === "download" && event.total) tracking.downloadTotal = event.total;
      if (event.phase === "convert") tracking.convertNote = ` · ${tracking.precision.toUpperCase()} conversion ${event.loaded}/${event.total} tensors`;
      const progress = progressOf(event, tracking);
      if (!progress) return;
      this.progress = progress;
      this.#emit();
    };
    const backend = this.backend;
    const vision = this.visionEnabledFor() ? this.visionUrlFor() : "";
    try {
      const localWeights = !!this.localPackFor() || (!!vision && !!this.localVisionPackFor());
      const kevala = await Kevala.load({
        model: src, from: FROM,
        ...(selectedPrecision ? { precision: selectedPrecision } : {}),
        ...(vision ? { vision } : {}),
        ...(localWeights ? { cache: false } : {}),
        backend, signal: abort.signal, onProgress,
      });
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
      this.loadedOptions = { model: src, from: FROM, precision: selectedPrecision, ...(vision ? { vision } : {}), backend };
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
      this.errorKind = aborted ? null : "load";
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

  /** Unload a model after a runtime failure, preserving that failure for the UI. */
  failRuntime(error, expectedModel = this.kevala) {
    if (!expectedModel || this.kevala !== expectedModel) return false;
    const message = error?.message || String(error);
    const loaded = this.kevala;
    this.kevala = null;
    this.loadedOptions = null;
    this.status = "error";
    this.error = message;
    this.errorKind = "runtime";
    this.progress = null;
    try {
      loaded.dispose?.();
    } catch {}
    this.#persist();
    this.#emit();
    this.#settle(null);
    return true;
  }

  async clearCache() {
    this.#invalidateCache(...Object.keys(MODELS));
    await clearCache();
    await this.refreshCache();
  }

  /** A one-line description of what loading the current choice costs. */
  costLine(model = this.model, precision = this.precisionFor(model)) {
    const spec = MODELS[model];
    const local = this.localPackFor(model, precision);
    if (local) {
      const gpuNote = requiresWebGPU(model, precision) ? " Requires WebGPU." : " Runs on WebGPU or CPU.";
      const memoryNote = model === "clef-flash" && precision === "bf16" ? " The weights exceed a 16 GB GPU; choose Q8 if you need a smaller pack." : "";
      return `Ready on this server: ${fmtBytes(local.bytes)} ${precision.toUpperCase()}. Select and press Load; no conversion needed.${gpuNote}${memoryNote}`;
    }
    if (explicitLocal && !SHOT) return "Loads the local pack from this server.";
    if (spec?.nativeConversion && this.customPackUrl(model)) {
      return "Loads your converted .kevala URL and caches it in this browser. Its precision is embedded in the pack.";
    }
    if (this.cached[model]) {
      return MODEL_NOTES[model]?.large ? "Cached in this browser: reopens locally." : "Cached in this browser: loads in about a second.";
    }
    if (!spec) return "Loads the pack from its URL and caches it.";
    const outputBytes = spec.precisionPackBytes?.[precision] ?? spec.pack;
    const outputApprox = spec.precisionPackBytes?.[precision] && spec.precisionPackApprox?.[precision] !== false;
    const outputSize = outputBytes ? `${outputApprox ? "about " : ""}${fmtBytes(outputBytes)} ${precision.toUpperCase()}` : `${precision.toUpperCase()} pack`;
    const gpuNote = requiresWebGPU(model, precision) ? " WebGPU is required at this size and precision." : "";
    if (spec.hosted && FROM === "pack" && precision === "q8") {
      return spec.pack
        ? `First load: a ${fmtBytes(spec.pack)} Q8 pack from Hugging Face, then cached in this browser.`
        : "First load: a Q8 pack from Hugging Face, then cached in this browser.";
    }
    if (spec.browserConvert === false) {
      return `No published ${precision.toUpperCase()} pack yet. Convert the pinned checkpoint with the native CLI, then load its .kevala URL.${gpuNote}`;
    }
    const sourceBytes = spec.download ? `${spec.downloadApprox ? "about " : ""}${fmtBytes(spec.download)}` : "the pinned checkpoint";
    return `First load: ${sourceBytes} from Hugging Face, converted to an estimated ${outputSize} pack and cached.${gpuNote}`;
  }
}

export const session = new Session();
