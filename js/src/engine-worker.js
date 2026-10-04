// The engine worker: owns the model, picks a backend, and answers requests.
//
// Everything here is family-agnostic. The pack's `config.arch` selects an architecture plugin
// (js/src/archs), which a pack must have. The plugin says how the family runs on WebGPU or across
// shard workers; a family whose plugin says neither runs on the CPU in one WebAssembly instance.
//
// It normally runs as a module worker. A browser can offer WebGPU to pages but not to workers;
// then the page imports this module and talks to it over a MessagePort (serve()).
//
// Backends:
//   webgpu  the coordinator (WebAssembly) tokenizes, embeds and scores; the GPU runs the layers
//   wasm    one instance runs everything, or the layers are split across N shard workers
//           (tensor parallel), shard 0 running in this worker

import { Wasm, compile, parseLayouts, wasmFlavor } from "./wasm.js";
import { openPack, PieceSink, Batcher } from "./source.js";
import { Profiler, requestDevice, SUBMIT, withGpuErrors } from "./gpu.js";
import { archPlugin, registerArch, validatePackArchitecture } from "./archs/index.js";
import { RemoteShard } from "./shard-client.js";
import { CPU_KERNELS, cpuOptions, readTuning, threadCandidates, tuningKey, writeTuning } from "./cpu-policy.js";
import { calibrateThreads } from "./cpu-calibrate.js";
import { assertWasmPackSize, packSize, parsePackHeader, readPackHead } from "./pack-layout.js";
import { requestQueue } from "./request-queue.js";

const enc = new TextEncoder();
let E = null; // the loaded engine
let loadController = null;
let activeGpu = null;
let packIterator = null;
const shardPorts = [];
const shardClients = [];
let shardWaiters = null;
let shardGeneration = 0;

// the page on the other end: this worker's scope, or the port serve() was given
let port = self;
const post = (m, t) => port.postMessage(m, t || []);
const progress = (p) => post({ type: "progress", ...p });
// Requests that arrive while a pass runs are packed into the next pass together.
const decideQueue = requestQueue((batch) => run(batch), post);

async function onMessage(ev) {
  const m = ev.data;
  try {
    if (m.type === "probe") post({ type: "probe", gpu: typeof navigator !== "undefined" && !!navigator.gpu });
    else if (m.type === "load") await load(m.options);
    else if (m.type === "decide") decideQueue.push(m);
    else if (m.type === "profile") setProfiling(m);
    else if (m.type === "shards") {
      if (m.generation !== shardGeneration) {
        for (const p of m.ports) p.close();
        return;
      }
      shardPorts.push(...m.ports);
      shardWaiters?.();
    }
  } catch (e) {
    if (m.type === "load") dispose();
    post({ type: "error", id: m.id, message: String(e?.message || e), code: e?.code });
  }
}

/** Runs the engine on the importing thread, answering on `p` (a MessagePort) instead of a worker scope. */
export function serve(p) {
  port = p;
  port.onmessage = onMessage;
}

/** Frees the model; a worker gets the same by being terminated. */
export function dispose() {
  loadController?.abort();
  activeGpu?.device.destroy();
  activeGpu = null;
  packIterator?.return?.().catch(() => {});
  packIterator = null;
  releaseShards();
  E = null;
  decideQueue.clear();
}

if (typeof WorkerGlobalScope !== "undefined") self.onmessage = onMessage;

function now() {
  return performance.now();
}

function releaseShards() {
  shardGeneration++;
  for (const r of shardClients.splice(0)) r.close();
  for (const p of shardPorts.splice(0)) p.close();
  post({ type: "release-shards" });
}

async function waitShards(n, signal) {
  signal.throwIfAborted();
  if (shardPorts.length >= n) return shardPorts.slice(0, n);
  await new Promise((resolve, reject) => {
    const finish = (error) => {
      clearTimeout(timer);
      signal.removeEventListener("abort", abort);
      shardWaiters = null;
      if (error) reject(error);
      else resolve();
    };
    const abort = () => finish(signal.reason);
    const timer = setTimeout(() => finish(new Error("CPU shard workers failed to start")), 10000);
    signal.addEventListener("abort", abort, { once: true });
    shardWaiters = () => shardPorts.length >= n && finish();
    post({ type: "need-shards", n: n - shardPorts.length, generation: shardGeneration });
  });
  return shardPorts.slice(0, n);
}

function remoteShard(port) {
  const client = new RemoteShard(port);
  shardClients.push(client);
  return client;
}

async function tuneCpu(coord, module, header, headerBytes, plugin, flavor, base, o, signal) {
  const started = now();
  const options = cpuOptions(o);
  const hardware = navigator.hardwareConcurrency || 4;
  const maxShards = plugin.run ? Math.max(1, Math.min(16, plugin.maxShards?.(header) || 1)) : 1;
  let counts = options.threads === "auto" ? threadCandidates(hardware, maxShards) : [Math.min(options.threads, maxShards)];
  if (options.threads === "auto" && counts.length > 1) {
    // Custom packs can impose alignment constraints beyond their attention-head count.
    counts = coord.withInput(headerBytes, (p, l) => counts.filter((n) => n === 1 || coord.x.kevala_layouts(p, l, n) === 0));
  }
  let key;
  if (o.cache !== false) {
    try {
      key = await tuningKey({ flavor, base, config: header.config, options, hardware, userAgent: navigator.userAgent });
    } catch {} // A non-secure context can run WASM without persistent tuning.
  }
  if (key && !options.retune) {
    const cached = await readTuning(key, counts, options);
    signal.throwIfAborted();
    if (cached) {
      coord.tile = CPU_KERNELS.indexOf(cached.kernel);
      coord.x.kevala_set_tile(coord.tile);
      return { ...cached, source: "cache", tuningMs: now() - started };
    }
  }
  progress({ phase: "init", message: "selecting CPU kernel and worker count" });
  coord.tile = options.cpuKernel === "auto" ? coord.tune() : CPU_KERNELS.indexOf(options.cpuKernel);
  coord.x.kevala_set_tile(coord.tile);
  signal.throwIfAborted();
  const profile = {
    kernel: CPU_KERNELS[coord.tile], threads: counts.at(-1), source: options.cpuKernel !== "auto" && options.threads !== "auto" ? "override" : "measured",
    kernelSource: options.cpuKernel === "auto" ? "measured" : "override",
    threadSource: options.threads !== "auto" ? "override" : counts.length === 1 ? "model-limit" : "measured",
    measurements: [], complete: true,
  };
  const config = plugin.cpuProbe?.(header);
  if (options.threads === "auto" && counts.length > 1) {
    if (config) {
      try {
        const remotes = (await waitShards(counts.at(-1) - 1, signal)).map(remoteShard);
        Object.assign(profile, await calibrateThreads(coord, remotes, module, config, counts, signal));
      } catch (error) {
        signal.throwIfAborted();
        Object.assign(profile, { threads: 1, threadSource: "fallback", complete: false, reason: String(error.message || error) });
      } finally {
        // Throw away synthetic allocations and ports before loading the production shards.
        releaseShards();
      }
    } else {
      Object.assign(profile, { threads: counts.filter((n) => n <= 8).at(-1), threadSource: "hardware-limit", complete: false });
    }
  }
  profile.tuningMs = now() - started;
  if (key && profile.complete) await writeTuning(key, profile);
  signal.throwIfAborted();
  return profile;
}

async function load(o) {
  loadController = new AbortController();
  const signal = loadController.signal;
  const checkCancelled = () => signal.throwIfAborted();
  const t0 = now();
  const base = o.wasmBase || new URL("./", import.meta.url).href;
  for (const url of o.plugins || []) registerArch((await import(url)).default);
  checkCancelled();
  const flavor = wasmFlavor(o.flavor);
  progress({ phase: "init", message: `compiling kevala-${flavor}.wasm` });
  const module = await compile(base, flavor);
  checkCancelled();
  const pack = await openPack(o.model ?? null, {
    cache: o.cache !== false,
    from: o.from,
    signal,
    onProgress: progress,
    convert: (spec, opts) => {
      const plugin = archPlugin(spec.arch);
      if (!plugin?.convert) throw new Error(`no in-browser converter for ${spec.arch}: pass the URL of a .kevala pack`);
      return plugin.convert(module, spec, opts);
    },
  });
  checkCancelled();
  const it = (packIterator = pack.chunks());
  const { head, header, headerBytes } = await readPackHead(it);
  const { arch, plugin } = validatePackArchitecture(header, pack.expectedArch);
  const fullSize = packSize(header, headerBytes.byteLength);
  checkCancelled();
  const requestedStateCache = o.stateCache !== false;

  // backend choice: WebGPU when the family has a GPU trunk, else shards when it can split
  let gpu = null;
  let gpuUnavailable = plugin.createGpu ? null : `no WebGPU backend for ${arch} yet`;
  if (plugin.createGpu && (o.backend === "webgpu" || o.backend === "auto" || !o.backend)) {
    if (!navigator.gpu && o.backend !== "webgpu") {
      gpuUnavailable = "this browser has no WebGPU in a worker (navigator.gpu is missing)";
    } else {
      try {
        gpu = await requestDevice({ baseline: o.gpuBaseline, features: o.gpuFeatures, powerPreference: o.gpuPowerPreference });
        gpu.kernel = o.gpuKernel || "auto";
        activeGpu = gpu;
      } catch (e) {
        throw Object.assign(new Error(`WebGPU: ${e.message}`, { cause: e }), { code: "WEBGPU_INIT" });
      }
      checkCancelled();
    }
  }
  if (!gpu && o.backend === "webgpu") throw Object.assign(new Error(`WebGPU: ${gpuUnavailable}`), { code: "WEBGPU_INIT" });
  if (!gpu && plugin.stateCache && !requestedStateCache) {
    throw Object.assign(new Error("stateCache: false requires WebGPU for Kev and SemIf; their WASM backend cannot disable the cache"), { code: "STATE_CACHE" });
  }
  if (!gpu && (plugin.maxShards?.(header) || 1) === 1) assertWasmPackSize(fullSize);
  const coord = await Wasm.create(module, 0);
  const cpuTuning = gpu ? null : await tuneCpu(coord, module, header, headerBytes, plugin, flavor, base, o, signal);
  const threads = cpuTuning?.threads || 1;
  checkCancelled();
  // the GPU kernels are WGSL sources in the Rust crate; the binary hands them out specialized
  if (gpu) gpu.wgsl = (kernel, spec = {}) => coord.withInput(JSON.stringify({ kernel, ...spec }), (p, l) => (coord.check(coord.x.kevala_wgsl(p, l)), coord.outText()));
  const external = gpu || threads > 1;
  let layouts = [];
  if (gpu && plugin.gpuLayouts) layouts = plugin.gpuLayouts(header, headerBytes.byteLength);
  else if (external) layouts = parseLayouts(coord.withInput(headerBytes, (p, l) => (coord.check(coord.x.kevala_layouts(p, l, gpu ? 1 : threads)), coord.out().slice())));
  const sinks = [];
  const engine = { arch, plugin, header, coord, gpu: null, shards: [], local: null, flavor, threads, pack: { bytes: pack.size, cached: pack.cached } };

  if (!external) {
    // one instance holds the whole pack
    const total = fullSize;
    assertWasmPackSize(total);
    engine.fullPtr = coord.alloc(total);
    engine.fullTotal = total;
    sinks.push({
      push: (chunk, at) => {
        const n = Math.min(chunk.byteLength, total - at);
        if (n > 0) coord.bytes(engine.fullPtr + at, n).set(chunk.subarray(0, n));
      },
    });
  } else {
    const cl = layouts[0];
    engine.coordPtr = coord.alloc(cl.total);
    engine.coordTotal = cl.total;
    engine.coordHeader = parsePackHeader(cl.prefix);
    sinks.push(new PieceSink(cl, (dst, bytes) => coord.bytes(engine.coordPtr + dst, bytes.byteLength).set(bytes)));
    if (gpu) {
      progress({ phase: "init", message: `WebGPU: ${gpu.name}` });
      engine.gpu = await withGpuErrors(gpu, () => plugin.createGpu(gpu, layouts[1], header), "creating the model");
      engine.gpu.stateCacheEnabled = requestedStateCache;
      checkCancelled();
      sinks.push(new PieceSink(layouts[1], (dst, bytes) => engine.gpu.write(dst, bytes)));
    } else {
      const ports = await waitShards(threads - 1, signal);
      // shard 0 lives here, next to the coordinator
      const local = await Wasm.create(module, coord.tile);
      const l0 = layouts[1];
      const p0 = local.alloc(l0.total);
      engine.local = { w: local, ptr: p0, total: l0.total };
      sinks.push(new PieceSink(l0, (dst, bytes) => local.bytes(p0 + dst, bytes.byteLength).set(bytes)));
      for (let i = 1; i < threads; i++) {
        const r = remoteShard(ports[i - 1]);
        const l = layouts[i + 1];
        const init = await r.call({ type: "init", module, tile: coord.tile, total: l.total }, [], { signal, timeout: 10000 });
        r.tile = init.tile;
        const b = new Batcher((dst, view) => r.send({ type: "data", dst, bytes: view }, [view.buffer]));
        const sink = new PieceSink(l, (dst, bytes) => b.write(dst, bytes));
        sink.batcher = b;
        sinks.push(sink);
        engine.shards.push(r);
      }
    }
  }

  // stream the pack through every sink
  let at = 0;
  const feed = (chunk) => {
    for (const s of sinks) s.push(chunk, at);
    at += chunk.byteLength;
  };
  const upload = async () => {
    feed(head);
    let pendingBytes = head.byteLength;
    const drain = async () => {
      if (!gpu) return;
      gpu.device.queue.submit([]);
      await gpu.device.queue.onSubmittedWorkDone();
      checkCancelled();
      pendingBytes = 0;
    };
    for (;;) {
      const { done, value } = await it.next();
      checkCancelled();
      if (done) break;
      feed(value);
      // Cached/local packs can arrive faster than transfers finish. Bound the staging
      // allocations so loading a large model does not temporarily double its GPU memory.
      pendingBytes += value.byteLength;
      if (gpu && pendingBytes >= 64 * 1024 * 1024) await drain();
    }
    await drain();
    if (at < fullSize) throw new Error(`pack is truncated: ${at} of ${fullSize} bytes`);
    packIterator = null;
    for (const s of sinks) s.batcher?.flush();
    // Finish a completed download's cache write before retrying a failed GPU load.
    if (pack.saving) {
      progress({ phase: "cache", message: "saving the pack for next time" });
      await pack.saving;
    }
  };
  if (gpu) await withGpuErrors(gpu, upload, "uploading model weights");
  else await upload();
  checkCancelled();
  progress({ phase: "init", message: "building the model" });

  const loaded = external ? [engine.coordPtr, engine.coordTotal] : [engine.fullPtr, engine.fullTotal];
  coord.call(() => coord.check(coord.x.kevala_engine_load(...loaded)));
  const meta = JSON.parse(coord.outText());
  if (engine.gpu) {
    await withGpuErrors(gpu, async () => {
      await plugin.initGpu(engine);
      if (o.profile) engine.gpu.profiler = new Profiler(gpu.device);
    }, "initializing GPU kernels");
    checkCancelled();
    // module state: a second load on this page must not inherit the first one's choice
    SUBMIT.mode = o.submit || "await";
  } else if (engine.local) {
    const { w, ptr, total } = engine.local;
    w.call(() => w.check(w.x.kevala_shard_load(ptr, total, 1)));
    await Promise.all(engine.shards.map((r) => r.call({ type: "load", primary: 0 })));
  }
  E = engine;
  const backend = engine.gpu ? "webgpu" : threads > 1 ? `wasm-${flavor} x${threads}` : `wasm-${flavor}`;
  engine.backend = backend;
  progress({ phase: "warmup" });
  const tw = now();
  const warmup = () => run([{ state: "warm up", questions: { q: { type: "noul", instructions: "Is this a warm up?" } } }]);
  if (gpu) await withGpuErrors(gpu, warmup, "warming up the model");
  else await warmup();
  checkCancelled();
  const warm = now() - tw;
  const cpuTiles = engine.gpu ? null : engine.local ? [engine.local.w.tile, ...engine.shards.map((r) => r.tile)] : [coord.tile];
  const cachesState = !!plugin.stateCache;
  const stateCache = {
    requested: requestedStateCache,
    supported: cachesState,
    enabled: cachesState ? (engine.gpu ? engine.gpu.stateCacheEnabled : true) : false,
    backend: cachesState ? (engine.gpu ? "webgpu" : "wasm") : null,
  };
  post({
    type: "ready",
    info: {
      arch,
      backend,
      gpu: engine.gpu?.name || null,
      gpuPowerPreference: engine.gpu ? o.gpuPowerPreference || "high-performance" : null,
      gpuUnavailable: engine.gpu ? null : gpuUnavailable,
      threads,
      flavor,
      cpuTiles,
      cpuTuning,
      gpuTuning: engine.gpu?.tuning?.diagnostics || null,
      stateCache,
      modalities: meta.modalities,
      model: header.model,
      config: header.config,
      pack: engine.pack,
      loadMs: now() - t0,
      warmupMs: warm,
    },
  });
}

/** One forward pass over a batch of requests; returns one response per request. */
async function run(requests) {
  const e = E;
  if (e.gpu || e.local) {
    const r = await e.plugin.run(e, requests);
    if (e.gpu?.lastProfile) r.timing.gpu = e.gpu.lastProfile;
    return r;
  }
  const c = e.coord;
  const t0 = now();
  c.withInput(enc.encode(JSON.stringify({ requests })), (p, l) => c.check(c.x.kevala_decide(p, l)));
  return { responses: JSON.parse(c.outText()), timing: { forward: now() - t0 } };
}

/** Starts or stops per-kernel GPU timing; answers whether this backend supports it. */
function setProfiling(m) {
  const gpu = E?.gpu;
  const supported = !!gpu && gpu.device.features.has("timestamp-query");
  if (supported) gpu.profiler = m.on ? new Profiler(gpu.device) : null;
  if (gpu && !m.on) gpu.lastProfile = null;
  post({ type: "result", id: m.id, supported, responses: [] });
}
