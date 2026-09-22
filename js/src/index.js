// kevala: System 1 decision models (Laya, Kev, ...) in the browser, no server, no dependencies.
//
//   import { Kevala } from "kevala";
//   const kevala = await Kevala.load({ onProgress: (p) => console.log(p) });
//   const r = await kevala.decide("Refund the duplicate charge or we cancel.", {
//     churn: { type: "noul", instructions: "Does the customer threaten to leave?" },
//   });
//   r.answers.churn.noul; // P(true)

import { MODELS } from "./source.js";
export { cacheInfo, clearCache, isCached, MODELS } from "./source.js";
export * as presets from "./presets.js";

export const VERSION = "0.1.3";

function spawn(file) {
  const url = new URL(file, import.meta.url);
  // workers must be same-origin; from a CDN, start a blob worker that imports the real module
  if (typeof location !== "undefined" && url.origin === location.origin) return new Worker(url, { type: "module" });
  const src = `import ${JSON.stringify(url.href)};`;
  return new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })), { type: "module" });
}

/** A loaded model. Create with `Kevala.load()`. */
export class Kevala {
  /**
   * Loads the model.
   *
   * options:
   *   model       a known model name ("laya", "kev-0.8b"; see MODELS); or the URL of a .kevala
   *               pack; or an ArrayBuffer/Blob of one (default: "laya")
   *   from        for a known model: "pack" downloads its pinned int8 pack from Hugging Face, and
   *               converts the original weights when the pack is unreachable (default);
   *               "checkpoint" always downloads the original weights and converts them here
   *   backend     "auto" (WebGPU when available, else WebAssembly), "webgpu", or "wasm"
   *   threads     WebAssembly workers for the wasm backend (default: cores, at most 8)
   *   cache       keep the pack in the Cache API (default true)
   *   onProgress  receives { phase, file, loaded, total, message }
   *   signal      AbortSignal that cancels loading
   *   wasmBase    where the .wasm files live (default: next to this module)
   *   plugins     URLs of extra architecture plugin modules (see js/src/archs/index.js)
   */
  static async load(options = {}) {
    const w = new Kevala();
    await w.#start(options);
    return w;
  }

  #worker = null;
  #shards = [];
  #pending = new Map();
  #seq = 0;
  #ready = null;
  info = null;
  lastTiming = null;

  async #start(o) {
    const { onProgress, signal, ...opts } = o;
    // persistent storage is not evicted under disk pressure; browsers grant or ignore the request
    // (only pages can ask, not the worker that stores the pack)
    if (opts.cache !== false && typeof document !== "undefined") navigator.storage?.persist?.()?.catch(() => {});
    this.#worker = spawn("./engine-worker.js");
    const ready = new Promise((resolve, reject) => {
      this.#ready = { resolve, reject };
    });
    this.#worker.onmessage = (ev) => this.#onMessage(ev.data, onProgress);
    this.#worker.onerror = (ev) => {
      const err = new Error(ev.message || "kevala worker failed to start");
      this.#ready?.reject(err);
      for (const p of this.#pending.values()) p.reject(err);
    };
    const abort = () => {
      this.dispose();
      this.#ready?.reject(signal.reason ?? new DOMException("Loading was cancelled", "AbortError"));
    };
    if (signal) {
      if (signal.aborted) abort();
      signal.addEventListener("abort", abort, { once: true });
    }
    // a relative pack URL means relative to the page, not to the worker script
    const model = typeof opts.model === "string" && !MODELS[opts.model] && typeof document !== "undefined" ? new URL(opts.model, document.baseURI).href : opts.model;
    const plugins = (opts.plugins || []).map((u) => (typeof document !== "undefined" ? new URL(u, document.baseURI).href : u));
    this.#worker.postMessage({ type: "load", options: { ...opts, model, plugins, wasmBase: opts.wasmBase || new URL("./", import.meta.url).href } });
    try {
      this.info = await ready;
    } finally {
      signal?.removeEventListener("abort", abort);
    }
  }

  #onMessage(m, onProgress) {
    switch (m.type) {
      case "progress":
        onProgress?.(m);
        break;
      case "need-shards": {
        const ports = [];
        for (let i = 0; i < m.n; i++) {
          const w = spawn("./shard-worker.js");
          const ch = new MessageChannel();
          w.postMessage({ type: "port", port: ch.port1 }, [ch.port1]);
          ports.push(ch.port2);
          this.#shards.push(w);
        }
        this.#worker.postMessage({ type: "shards", ports }, ports);
        break;
      }
      case "ready":
        this.#ready.resolve(m.info);
        this.#ready = null;
        break;
      case "result": {
        const p = this.#pending.get(m.id);
        this.#pending.delete(m.id);
        this.lastTiming = m.timing;
        p?.resolve(m);
        break;
      }
      case "error": {
        if (m.id === undefined && this.#ready) {
          this.#ready.reject(new Error(m.message));
          this.#ready = null;
          break;
        }
        const p = this.#pending.get(m.id);
        this.#pending.delete(m.id);
        p?.reject(new Error(m.message));
        break;
      }
    }
  }

  #send(requests) {
    if (!this.#worker) return Promise.reject(new Error("kevala was disposed"));
    const id = ++this.#seq;
    const norm = requests.map((r) => ({ state: r.state ?? "", ...(r.parts ? { parts: r.parts } : {}), questions: r.questions }));
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#worker.postMessage({ type: "decide", id, requests: norm });
    });
  }

  /**
   * Answers typed questions about a state in one forward pass (System One shape, like
   * `agent.predict(state, questions)`). `state` is text or any JSON value. `options.parts` adds
   * typed content (`{ type: "text", text }`, and `image` / `audio` parts for packs whose
   * `info.modalities` list them); a part the model cannot read is an error, never dropped.
   */
  async decide(state, questions, options = {}) {
    const m = await this.#send([{ state, questions, parts: options.parts }]);
    return { ...m.responses[0], timing: m.timing };
  }

  /** Many `{ state, questions, parts? }` at once, packed into one forward pass; one response each. */
  async decideMany(items) {
    const m = await this.#send(items);
    return m.responses.map((r) => ({ ...r, timing: m.timing }));
  }

  /**
   * Turns per-kernel GPU timing on or off. While it is on, every response's `timing.gpu` maps
   * each kernel (matmul by weight, attention, norms...) to its milliseconds. Each kernel then
   * runs in its own pass with timestamps, so requests are slower than usual: profile, then turn
   * it off. Resolves to whether the backend can profile (WebGPU with timestamp queries).
   */
  async profile(on = true) {
    if (!this.#worker) throw new Error("kevala was disposed");
    const id = ++this.#seq;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve: (m) => resolve(m.supported), reject });
      this.#worker.postMessage({ type: "profile", id, on });
    });
  }

  /** Stops every worker and frees the model. */
  dispose() {
    this.#worker?.terminate();
    for (const s of this.#shards) s.terminate();
    this.#worker = null;
    this.#shards = [];
    for (const p of this.#pending.values()) p.reject(new Error("kevala was disposed"));
    this.#pending.clear();
  }
}

export const load = (options) => Kevala.load(options);
