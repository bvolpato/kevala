// The engine worker: owns the model, picks a backend, and answers requests.
//
// Everything here is family-agnostic. The pack's `config.arch` selects an architecture plugin
// (js/src/archs) that says how the family runs on WebGPU or across shard workers; without one,
// the family still runs on the CPU in a single WebAssembly instance.
//
// Backends:
//   webgpu  the coordinator (WebAssembly) tokenizes, embeds and scores; the GPU runs the layers
//   wasm    one instance runs everything, or the layers are split across N shard workers
//           (tensor parallel), shard 0 running in this worker

import { Wasm, compile, parseLayouts, wasmFlavor } from "./wasm.js";
import { openPack, PieceSink, Batcher } from "./source.js";
import { Profiler, requestDevice, SUBMIT } from "./gpu.js";
import { archPlugin, registerArch } from "./archs/index.js";

const enc = new TextEncoder();
let E = null; // the loaded engine
let queue = [];
let busy = false;
const shardPorts = [];
let shardWaiters = null;

const post = (m, t) => self.postMessage(m, t || []);
const progress = (p) => post({ type: "progress", ...p });

self.onmessage = async (ev) => {
  const m = ev.data;
  try {
    if (m.type === "load") await load(m.options);
    else if (m.type === "decide") enqueue(m);
    else if (m.type === "shards") {
      shardPorts.push(...m.ports);
      shardWaiters?.();
    }
  } catch (e) {
    post({ type: "error", id: m.id, message: String(e?.message || e), stack: e?.stack });
  }
};

function now() {
  return performance.now();
}

/** Reads the pack header from the front of the stream, keeping the bytes it consumed. */
async function readHead(iter) {
  let buf = new Uint8Array(0);
  const need = () => {
    if (buf.byteLength < 16) return 16;
    return 16 + new DataView(buf.buffer, buf.byteOffset).getUint32(8, true);
  };
  while (buf.byteLength < need()) {
    const { done, value } = await iter.next();
    if (done) throw new Error("pack ended before its header");
    const n = new Uint8Array(buf.byteLength + value.byteLength);
    n.set(buf);
    n.set(value, buf.byteLength);
    buf = n;
  }
  if (new TextDecoder().decode(buf.subarray(0, 4)) !== "KVLA") throw new Error("not a .kevala pack");
  const hlen = new DataView(buf.buffer, buf.byteOffset).getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(buf.subarray(16, 16 + hlen)));
  return { head: buf, header, headerBytes: buf.subarray(0, 16 + hlen) };
}

async function waitShards(n) {
  if (shardPorts.length >= n) return shardPorts.slice(0, n);
  post({ type: "need-shards", n: n - shardPorts.length });
  await new Promise((r) => (shardWaiters = () => shardPorts.length >= n && r()));
  return shardPorts.slice(0, n);
}

class RemoteShard {
  constructor(port) {
    this.port = port;
    this.waits = new Map();
    this.seq = 0;
    port.onmessage = (ev) => {
      const m = ev.data;
      const w = this.waits.get(m.seq);
      if (!w) return;
      this.waits.delete(m.seq);
      if (m.type === "error") w.reject(new Error(m.message));
      else w.resolve(m);
    };
  }
  call(msg, transfer) {
    const seq = ++this.seq;
    return new Promise((resolve, reject) => {
      this.waits.set(seq, { resolve, reject });
      this.port.postMessage({ ...msg, seq }, transfer || []);
    });
  }
  send(msg, transfer) {
    this.port.postMessage(msg, transfer || []);
  }
}

function subHeader(prefix) {
  const n = new DataView(prefix.buffer, prefix.byteOffset).getUint32(8, true);
  return JSON.parse(new TextDecoder().decode(prefix.subarray(16, 16 + n)));
}

async function load(o) {
  const t0 = now();
  const base = o.wasmBase || new URL("./", import.meta.url).href;
  for (const url of o.plugins || []) registerArch((await import(url)).default);
  const flavor = wasmFlavor(o.flavor);
  progress({ phase: "init", message: `compiling kevala-${flavor}.wasm` });
  const module = await compile(base, flavor);
  const pack = await openPack(o.model ?? {}, {
    cache: o.cache !== false,
    onProgress: progress,
    convert: (spec, opts) => {
      const plugin = archPlugin(spec.arch);
      if (!plugin?.convert) throw new Error(`no in-browser converter for ${spec.arch}: pass the URL of a .kevala pack`);
      return plugin.convert(module, spec, opts);
    },
  });
  const it = pack.chunks();
  const { head, header, headerBytes } = await readHead(it);
  const arch = header.config.arch || "laya";
  const plugin = archPlugin(arch);

  // backend choice: WebGPU when the family has a GPU trunk, else shards when it can split
  let gpu = null;
  if (plugin?.createGpu && (o.backend === "webgpu" || o.backend === "auto" || !o.backend)) {
    try {
      gpu = await requestDevice();
    } catch (e) {
      if (o.backend === "webgpu") throw e;
    }
  }
  if (!gpu && o.backend === "webgpu") throw new Error(plugin?.createGpu ? "WebGPU is not available in this browser" : `no WebGPU backend for ${arch} yet`);
  const hw = Math.max(1, Math.min(16, navigator.hardwareConcurrency || 4));
  const maxShards = gpu || !plugin?.run ? 1 : plugin.maxShards?.(header) || 1;
  const threads = Math.max(1, Math.min(o.threads || Math.min(hw, 8), maxShards));

  const coord = await Wasm.create(module);
  const external = gpu || threads > 1;
  const layouts = external ? parseLayouts(coord.withInput(headerBytes, (p, l) => (coord.check(coord.x.kevala_layouts(p, l, gpu ? 1 : threads)), coord.out().slice()))) : [];
  const sinks = [];
  const engine = { arch, plugin, header, coord, gpu: null, shards: [], local: null, flavor, threads, pack: { bytes: pack.size, cached: pack.cached } };

  if (!external) {
    // one instance holds the whole pack
    const total = header.tensors.reduce((m, t) => Math.max(m, t.offset + t.size, (t.scales_offset || 0) + (t.scales_size || 0)), 0);
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
    engine.coordHeader = subHeader(cl.prefix);
    sinks.push(new PieceSink(cl, (dst, bytes) => coord.bytes(engine.coordPtr + dst, bytes.byteLength).set(bytes)));
    if (gpu) {
      progress({ phase: "init", message: `WebGPU: ${gpu.name}` });
      engine.gpu = plugin.createGpu(gpu, layouts[1], header);
      sinks.push(new PieceSink(layouts[1], (dst, bytes) => engine.gpu.write(dst, bytes)));
    } else {
      const ports = await waitShards(threads - 1);
      // shard 0 lives here, next to the coordinator
      const local = await Wasm.create(module);
      const l0 = layouts[1];
      const p0 = local.alloc(l0.total);
      engine.local = { w: local, ptr: p0, total: l0.total };
      sinks.push(new PieceSink(l0, (dst, bytes) => local.bytes(p0 + dst, bytes.byteLength).set(bytes)));
      for (let i = 1; i < threads; i++) {
        const r = new RemoteShard(ports[i - 1]);
        const l = layouts[i + 1];
        await r.call({ type: "init", module, base, total: l.total });
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
  feed(head);
  for (;;) {
    const { done, value } = await it.next();
    if (done) break;
    feed(value);
  }
  for (const s of sinks) s.batcher?.flush();
  // finish storing the pack before reporting ready, so a page that closes right away keeps it
  if (pack.saving) {
    progress({ phase: "cache", message: "saving the pack for next time" });
    await pack.saving;
  }
  progress({ phase: "init", message: "building the model" });

  const loaded = external ? [engine.coordPtr, engine.coordTotal] : [engine.fullPtr, engine.fullTotal];
  coord.call(() => coord.check(coord.x.kevala_engine_load(...loaded)));
  const meta = JSON.parse(coord.outText());
  if (engine.gpu) {
    await plugin.initGpu(engine);
    if (o.profile) engine.gpu.profiler = new Profiler(gpu.device);
    if (["await", "split", "none"].includes(o.submit)) SUBMIT.mode = o.submit;
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
  await run([{ state: "warm up", questions: { q: { type: "noul", instructions: "Is this a warm up?" } } }]);
  const warm = now() - tw;
  post({
    type: "ready",
    info: {
      arch,
      backend,
      gpu: engine.gpu?.name || null,
      threads,
      flavor,
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

function enqueue(m) {
  queue.push(m);
  if (!busy) drain();
}

// Requests that arrive while a pass runs are packed into the next pass together.
async function drain() {
  busy = true;
  while (queue.length) {
    const batch = [];
    let n = 0;
    while (queue.length && (batch.length === 0 || n + queue[0].requests.length <= 64)) {
      const m = queue.shift();
      batch.push(m);
      n += m.requests.length;
    }
    const reqs = batch.flatMap((m) => m.requests);
    try {
      const { responses, timing } = await run(reqs);
      let i = 0;
      for (const m of batch) {
        post({ type: "result", id: m.id, responses: responses.slice(i, i + m.requests.length), timing: { ...timing, batched: reqs.length } });
        i += m.requests.length;
      }
    } catch (e) {
      // one bad request should not fail the others packed with it: retry them one by one
      if (batch.length > 1) {
        for (const m of batch) queue.unshift(m);
        const first = queue.shift();
        try {
          const { responses, timing } = await run(first.requests);
          post({ type: "result", id: first.id, responses, timing });
        } catch (err) {
          post({ type: "error", id: first.id, message: String(err?.message || err) });
        }
      } else {
        post({ type: "error", id: batch[0].id, message: String(e?.message || e) });
      }
    }
  }
  busy = false;
}
