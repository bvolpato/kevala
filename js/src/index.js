// kevala: System 1 decision models (Laya, Kev, SemIf, Gemma 4) in the browser, no server, no dependencies.
//
//   import { Kevala } from "kevala";
//   const kevala = await Kevala.load({ onProgress: (p) => console.log(p) });
//   const r = await kevala.decide("Refund the duplicate charge or we cancel.", {
//     churn: { type: "noul", instructions: "Does the customer threaten to leave?" },
//   });
//   r.answers.churn.noul; // P(true)

import { MODELS } from "./source.js";
import { cpuOptions } from "./cpu-policy.js";
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

/** Whether WebGPU is available inside `worker`; startup errors reject the load immediately. */
function workerHasGpu(worker, signal) {
  return new Promise((resolve, reject) => {
    const fail = (error) => {
      signal?.removeEventListener("abort", abort);
      reject(error);
    };
    const abort = () => fail(signal.reason ?? new DOMException("Loading was cancelled", "AbortError"));
    const done = (gpu) => {
      signal?.removeEventListener("abort", abort);
      resolve(gpu);
    };
    signal?.addEventListener("abort", abort, { once: true });
    worker.onmessage = (ev) => ev.data.type === "probe" && done(ev.data.gpu);
    worker.onerror = (ev) => fail(new Error(ev.message || "kevala worker failed to start"));
    try {
      worker.postMessage({ type: "probe" });
    } catch (e) {
      fail(e);
    }
  });
}

let pageEngines = 0;

/**
 * The engine on this thread, behind a MessagePort that stands in for its worker. Each call
 * imports a fresh module instance, since the engine keeps its model in module state.
 */
async function pageEngine() {
  const engine = await import(new URL(`./engine-worker.js?page=${++pageEngines}`, import.meta.url).href);
  const { port1, port2 } = new MessageChannel();
  engine.serve(port2);
  port1.terminate = () => {
    engine.dispose();
    port1.close();
    port2.close();
  };
  return port1;
}

/** A loaded model. Create with `Kevala.load()`. */
export class Kevala {
  /**
   * Loads the model.
   *
   * options:
   *   model       a known model name ("laya", "kev-0.8b", "kev-4b", "kev-9b", or a
   *               "semif-qwen3.5-*" or "gemma-4-e*" name; see MODELS); or the URL of a .kevala
   *               pack; or an ArrayBuffer/Blob of one (default: "laya")
   *   from        for a known model: "pack" downloads its pinned int8 pack from Hugging Face, and
   *               converts the original weights when the pack is unreachable (default);
   *               "checkpoint" always downloads the original weights and converts them here
   *   backend     "auto" (WebGPU when available, else WebAssembly), "webgpu", or "wasm"
   *   onPage      run the engine on the page instead of in a worker (default: only when the
   *               browser offers WebGPU to pages but not to workers)
   *   threads     "auto" measures CPU worker counts (default); 1..16 overrides, capped by model
   *   cpuKernel   "auto" measures CPU register tiles (default), or "2x4" / "4x4"
   *   gpuKernel   GPU matrices: "auto" measures kernels (default), "generic", or "wide"
   *   stateCache  keep Kev/SemIf cross-request GPU state caches (default true); false disables
   *               cross-request reuse while retaining duplicate sharing within one pass. It is
   *               ignored by model families without a state cache; Kev/SemIf WASM rejects false
   *   retune      ignore the cached CPU tuning profile and measure again (default false)
   *   submit      WebGPU scheduling: "await" drains each long-pass chunk (default);
   *               "split" queues separate chunks without waiting, reducing latency at a
   *               possible cost to UI responsiveness; "none" uses one command buffer
   *   cache       keep the pack in the Cache API (default true)
   *   onProgress  receives { phase, file, loaded, total, message }
   *   signal      AbortSignal that cancels loading
   *   wasmBase    where the .wasm files live (default: next to this module)
   *   plugins     URLs of extra architecture plugin modules (see js/src/archs/index.js)
   */
  static async load(options = {}) {
    cpuOptions(options);
    if (options.gpuKernel !== undefined && !["auto", "generic", "wide"].includes(options.gpuKernel)) throw new Error('gpuKernel must be "auto", "generic", or "wide"');
    if (options.stateCache !== undefined && typeof options.stateCache !== "boolean") throw new TypeError("stateCache must be a boolean");
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
    const cancelled = () => signal.reason ?? new DOMException("Loading was cancelled", "AbortError");
    if (signal?.aborted) throw cancelled();
    const abort = () => this.dispose(cancelled());
    signal?.addEventListener("abort", abort, { once: true });
    // a relative pack URL means relative to the page, not to the worker script
    const model = typeof opts.model === "string" && !MODELS[opts.model] && typeof document !== "undefined" ? new URL(opts.model, document.baseURI).href : opts.model;
    const plugins = (opts.plugins || []).map((u) => (typeof document !== "undefined" ? new URL(u, document.baseURI).href : u));
    const options = { ...opts, model, plugins, wasmBase: opts.wasmBase || new URL("./", import.meta.url).href };
    try {
      let worker = spawn("./engine-worker.js");
      this.#worker = worker;
      const pageGpu = typeof navigator !== "undefined" && !!navigator.gpu && opts.backend !== "wasm";
      const onPage = pageGpu && (opts.onPage || !(await workerHasGpu(worker, signal)));
      if (onPage) {
        worker.terminate();
        this.#worker = worker = null;
      }
      const failures = [];
      for (const gpuPowerPreference of ["high-performance", "low-power"]) {
        if (signal?.aborted) throw cancelled();
        const engine = onPage ? await pageEngine() : worker || spawn("./engine-worker.js");
        worker = null;
        if (signal?.aborted) {
          engine.terminate();
          throw cancelled();
        }
        try {
          this.info = await this.#connect(engine, { ...options, ...(onPage ? { backend: "webgpu" } : {}), gpuPowerPreference }, onProgress);
          if (signal?.aborted) throw cancelled();
          if (onPage) this.info.onPage = true;
          return;
        } catch (e) {
          this.dispose();
          if (signal?.aborted) throw cancelled();
          if (e.code !== "WEBGPU_INIT" || opts.backend === "wasm") throw e;
          failures.push(`${gpuPowerPreference}: ${String(e.message).replace(/^WebGPU: /, "")}`);
          if (gpuPowerPreference === "high-performance") onProgress?.({ phase: "init", message: "WebGPU failed; trying the low-power adapter" });
        }
      }
      if (signal?.aborted) throw cancelled();
      const gpuUnavailable = failures.join("; ");
      if (opts.backend === "webgpu") throw Object.assign(new Error(`WebGPU: ${gpuUnavailable}`), { code: "WEBGPU_INIT" });
      onProgress?.({ phase: "init", message: "WebGPU unavailable; loading on the CPU" });
      if (signal?.aborted) throw cancelled();
      this.info = { ...(await this.#connect(spawn("./engine-worker.js"), { ...options, backend: "wasm" }, onProgress)), gpuUnavailable };
      if (signal?.aborted) throw cancelled();
    } catch (e) {
      this.dispose();
      throw e;
    } finally {
      signal?.removeEventListener("abort", abort);
    }
  }

  /** Talks to an engine (a worker, or a port to one on this page): loads the model, resolves to its info. */
  #connect(engine, options, onProgress) {
    this.#worker = engine;
    const ready = new Promise((resolve, reject) => {
      this.#ready = { resolve, reject };
    });
    engine.onmessage = (ev) => {
      if (this.#worker === engine) this.#onMessage(ev.data, onProgress);
    };
    engine.onerror = (ev) => {
      if (this.#worker !== engine) return;
      const err = new Error(ev.message || "kevala worker failed to start");
      this.#ready?.reject(err);
      for (const p of this.#pending.values()) p.reject(err);
    };
    try {
      engine.postMessage({ type: "load", options });
    } catch (e) {
      this.#ready.reject(e);
    }
    return ready;
  }

  #onMessage(m, onProgress) {
    switch (m.type) {
      case "progress":
        onProgress?.(m);
        break;
      case "need-shards": {
        const ports = [];
        try {
          for (let i = 0; i < m.n; i++) {
            const w = spawn("./shard-worker.js");
            this.#shards.push(w);
            w.onerror = (ev) => {
              if (this.#shards.includes(w)) this.dispose(new Error(ev.message || "kevala shard worker failed to start"));
            };
            const ch = new MessageChannel();
            ports.push(ch.port2);
            w.postMessage({ type: "port", port: ch.port1 }, [ch.port1]);
          }
          this.#worker.postMessage({ type: "shards", ports, generation: m.generation }, ports);
        } catch (error) {
          for (const p of ports) p.close();
          this.dispose(error);
        }
        break;
      }
      case "release-shards":
        for (const w of this.#shards) w.terminate();
        this.#shards = [];
        break;
      case "ready":
        this.#ready?.resolve(m.info);
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
        const err = Object.assign(new Error(m.message), { code: m.code });
        if (m.id === undefined && this.#ready) {
          this.#ready.reject(err);
          this.#ready = null;
          break;
        }
        const p = this.#pending.get(m.id);
        this.#pending.delete(m.id);
        p?.reject(err);
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
  dispose(reason = new Error("kevala was disposed")) {
    this.#worker?.terminate();
    for (const s of this.#shards) s.terminate();
    this.#worker = null;
    this.#shards = [];
    this.#ready?.reject(reason);
    this.#ready = null;
    for (const p of this.#pending.values()) p.reject(reason);
    this.#pending.clear();
  }
}

export const load = (options) => Kevala.load(options);
