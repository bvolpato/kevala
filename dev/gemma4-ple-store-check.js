import { createGemma4PleResource, createGemma4PleStore } from "../js/src/gemma4-ple-store.js";

const ROWS = 4096;
const SELECTED = 256;
const WARM_SAMPLES = 5;
const WRITE_CHUNK = 65537;
const FLUSH_BYTES = 2 * 1024 * 1024;
const LIMIT_BYTES = 32 * 1024 * 1024;
const TABLES = [...Array.from({ length: 6 }, (_, i) => ({ name: `ple.${i}.embed`, width: 256, seed: i, scale: 1.123456789 })), { name: "emb", width: 512, seed: 6, scale: 1.001234567 }];
const IDS = Int32Array.from({ length: SELECTED }, (_, i) => i % 13 === 0 || i % 64 === 0 ? 0 : 1 + (i % 64) * 61);
const TABLE_BYTES = TABLES.reduce((sum, t) => sum + ROWS * t.width * 9 / 8, 0);
const OUTPUT_FLOATS = 5 + SELECTED * TABLES.reduce((sum, t) => sum + t.width, 0) + 3;
const BUFFER_BOUND_BYTES = OUTPUT_FLOATS * 8 + FLUSH_BYTES + WRITE_CHUNK * 2 + 1024 * 1024;

const q8 = (t, row, col) => ((row * 17 + col * 13 + t.seed * 29) & 255) - 128;
const factor = (t, row, block) => row === 0 ? 0 : Math.fround((t.seed + 1) * ((row + block * 7) % 19 + 1) / 131.1234567 * (block % 3 === 0 ? -1 : 1));
const median = (values) => values.toSorted((a, b) => a - b)[Math.floor(values.length / 2)];

function traceStorage() {
  const calls = {};
  const restore = [];
  for (const [prototype, names] of [
    [globalThis.FileSystemFileHandle?.prototype, ["createWritable", "createSyncAccessHandle", "getFile"]],
    [globalThis.FileSystemSyncAccessHandle?.prototype, ["read", "write", "flush", "getSize", "close"]],
    [globalThis.Blob?.prototype, ["arrayBuffer"]],
  ]) {
    for (const name of names) {
      const descriptor = prototype && Object.getOwnPropertyDescriptor(prototype, name);
      if (typeof descriptor?.value !== "function") continue;
      Object.defineProperty(prototype, name, { ...descriptor, value(...args) {
        calls[name] = (calls[name] || 0) + 1;
        return descriptor.value.apply(this, args);
      } });
      restore.push(() => Object.defineProperty(prototype, name, descriptor));
    }
  }
  return { calls, restore: () => { for (const fn of restore.reverse()) fn(); } };
}

function reference() {
  const output = new Float32Array(OUTPUT_FLOATS).fill(100);
  let at = 5;
  for (const t of TABLES) {
    for (const id of IDS) for (let col = 0; col < t.width; col++) {
      output[at++] = Math.fround(Math.fround(q8(t, id, col) * factor(t, id, Math.floor(col / 32))) * Math.fround(t.scale));
    }
  }
  return output;
}

function compare(actual, expected) {
  const a = new Uint32Array(actual.buffer);
  const b = new Uint32Array(expected.buffer);
  let hash = 0x811c9dc5;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) throw new Error(`PLE F32 bits differ at ${i}: ${a[i].toString(16)} != ${b[i].toString(16)}`);
    hash = Math.imul(hash ^ a[i], 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

async function absent(resource) {
  try {
    const directory = await navigator.storage.getDirectory();
    await directory.getFileHandle(resource.name);
  } catch (error) {
    if (error.name === "NotFoundError") return true;
    throw error;
  }
  throw new Error(`owned PLE tempfile remains: ${resource.name}`);
}

/** Timings include store gathers and dequantization only. Validation runs outside each sample. */
export async function runPleStoreBenchmark(scope, onResource = () => {}, onProgress = () => {}, resource = undefined) {
  if (TABLE_BYTES > LIMIT_BYTES || BUFFER_BOUND_BYTES * 2 > LIMIT_BYTES) throw new Error("PLE benchmark exceeds its buffer or storage limit");
  const trace = traceStorage();
  let store, result;
  try {
    const metadata = TABLES.map((t) => ({ name: t.name, dtype: "q8", shape: [ROWS, t.width], size: ROWS * t.width, block: 32, scales_size: ROWS * t.width / 8 }));
    const started = performance.now();
    store = await createGemma4PleStore([metadata[3], metadata[6], metadata[1], metadata[5], metadata[0], metadata[4], metadata[2]], resource);
    if (!store) throw new Error("OPFS PLE storage is unavailable in this context");
    onResource(store.resource);
    let queued = 0;
    const write = async (name, part, at, bytes) => {
      store.write(name, part, at, bytes);
      queued += bytes.byteLength;
      if (queued >= FLUSH_BYTES) { await store.flush(); queued = 0; }
    };
    for (const t of TABLES.toReversed()) {
      onProgress(`${scope}: write ${t.name}`);
      for (let at = 0; at < ROWS * t.width; at += WRITE_CHUNK) {
        const bytes = new Uint8Array(Math.min(WRITE_CHUNK, ROWS * t.width - at));
        for (let i = 0; i < bytes.length; i++) {
          const index = at + i;
          bytes[i] = q8(t, Math.floor(index / t.width), index % t.width);
        }
        await write(t.name, "data", at, bytes);
      }
      const blocks = t.width / 32;
      for (let at = 0; at < ROWS * blocks; at += 8192) {
        const scales = new Float32Array(Math.min(8192, ROWS * blocks - at));
        for (let i = 0; i < scales.length; i++) scales[i] = factor(t, Math.floor((at + i) / blocks), (at + i) % blocks);
        await write(t.name, "scales", at * 4, new Uint8Array(scales.buffer));
      }
    }
    await store.finish();
    const setupMs = performance.now() - started;
    const storagePath = trace.calls.createSyncAccessHandle ? "opfs-sync-access-handle" : trace.calls.createWritable ? "opfs-writable-stream-and-blob" : "unknown";
    const expectedPath = scope === "window" ? "opfs-writable-stream-and-blob" : "opfs-sync-access-handle";
    if (storagePath !== expectedPath) throw new Error(`${scope} selected ${storagePath}; expected ${expectedPath}`);
    const expected = reference();
    const output = new Float32Array(OUTPUT_FLOATS);
    const sample = async (label) => {
      onProgress(`${scope}: ${label}`);
      output.fill(100);
      const before = { ...trace.calls };
      const started = performance.now();
      let offset = 5;
      for (const t of TABLES) {
        await store.gather(t.name, IDS, t.scale, output, offset);
        offset += IDS.length * t.width;
      }
      const gatherMs = performance.now() - started;
      const outputBitsFingerprint = compare(output, expected);
      const calls = Object.fromEntries(Object.keys(trace.calls).map((key) => [key, trace.calls[key] - (before[key] || 0)]));
      return { label, gatherMs, bitExact: true, outputBitsFingerprint, calls };
    };
    const cold = await sample("first-gather");
    const warm = [];
    for (let i = 0; i < WARM_SAMPLES; i++) warm.push(await sample(`repeat-${i + 1}`));
    result = {
      scope, storagePath, setupMs, cold, warm, warmMedianMs: median(warm.map((row) => row.gatherMs)),
      bitExact: true, outputBitsFingerprint: cold.outputBitsFingerprint,
      totalCalls: trace.calls, crossOriginIsolated: globalThis.crossOriginIsolated,
      dataset: {
        tables: TABLES.map(({ name, width }) => ({ name, rows: ROWS, width })), tableBytes: TABLE_BYTES,
        selectedRowsPerTable: IDS.length, uniqueRowsPerTable: new Set(IDS).size,
        padId: 0, padSelectionsPerTable: IDS.filter((id) => id === 0).length,
        outputFloats: OUTPUT_FLOATS, conservativeApplicationBufferBoundBytes: BUFFER_BOUND_BYTES,
        conservativeCombinedContextBufferBoundBytes: BUFFER_BOUND_BYTES * 2, limitBytes: LIMIT_BYTES,
      },
    };
    return result;
  } finally {
    try {
      if (store) {
        await store.dispose();
        const removed = await absent(store.resource);
        if (result) result.cleanup = { removed, name: store.resource.name };
      }
    } finally { trace.restore(); }
  }
}

if (typeof document !== "undefined") {
  const button = document.getElementById("run");
  const log = (value) => { document.getElementById("log").textContent += `${value}\n`; };
  const workerRun = async () => {
    const worker = new Worker(new URL("./gemma4-ple-store-check-worker.js", import.meta.url), { type: "module" });
    let resource, timer;
    try {
      resource = await createGemma4PleResource();
      if (!resource) throw new Error("OPFS PLE storage is unavailable on the owner page");
      return await new Promise((resolve, reject) => {
        timer = setTimeout(() => reject(new Error("PLE worker benchmark exceeded 60 seconds")), 60000);
        worker.onmessage = ({ data }) => {
          if (data.type === "resource") resource = data.resource;
          else if (data.type === "progress") log(data.value);
          else if (data.type === "result") resolve(data.result);
          else if (data.type === "error") reject(new Error(data.error));
        };
        worker.onerror = (event) => reject(new Error(event.message || "PLE benchmark worker failed"));
        worker.postMessage({ type: "run", resource: { name: resource.name } });
      });
    } finally {
      clearTimeout(timer);
      worker.terminate();
      if (resource) {
        try { await (await navigator.storage.getDirectory()).removeEntry(resource.name); }
        catch (error) { if (error.name !== "NotFoundError") throw error; }
        await absent(resource);
      }
    }
  };
  async function main() {
    if (button.disabled) return;
    button.disabled = true;
    window.pleStoreBench = { status: "running" };
    try {
      const windowResult = await runPleStoreBenchmark("window", undefined, log);
      const workerResult = await workerRun();
      if (windowResult.outputBitsFingerprint !== workerResult.outputBitsFingerprint) throw new Error("Window and worker output fingerprints differ despite reference comparisons");
      window.pleStoreBench = {
        status: "done", done: true, measuredAt: new Date().toISOString(), userAgent: navigator.userAgent,
        method: "wall-clock batch of seven direct OPFS store gathers, including Q8 dequantization; validation and writes excluded",
        coldDefinition: "first gather after writing and finishing a new tempfile; OS storage cache was not evicted",
        warmDefinition: "five repeated gathers with identical token IDs; no assembled GPU upload cache is involved",
        instrumentation: "native API wrappers count each storage call; these counters remain active during timed gathers",
        correctness: { bitExactReference: true, bitExactAcrossPaths: true, duplicateIds: true, padId: 0 },
        window: windowResult, worker: workerResult,
        firstGatherRatio: windowResult.cold.gatherMs / workerResult.cold.gatherMs,
        repeatedGatherMedianRatio: windowResult.warmMedianMs / workerResult.warmMedianMs,
      };
      log(JSON.stringify(window.pleStoreBench, null, 2));
    } catch (error) {
      window.pleStoreBench = { status: "error", done: true, error: String(error.stack || error) };
      log(window.pleStoreBench.error);
    } finally { button.disabled = false; }
  }
  window.pleStoreBench = { status: "idle" };
  button.addEventListener("click", main);
  if (new URLSearchParams(location.search).get("autorun") === "1") main();
}
