// Laya attention kernels on synthetic inputs in the runtime's layout: QKV [T, 3 W] with W = heads *
// 64, query blocks of (segment start, segment length, first query), and CTX [T, W]. Each case is
// checked against a CPU reference and timed with WebGPU timestamps, the variants taking turns.
//
//   attn-bench.html?cases=512:0,512:64,140:64,47:0,150/40:64&kernels=attention,attention_subgroup
//
// A case is T:window, one segment of T tokens (window 0 means global attention); T/T2:window
// packs several segments, as a batched request does. The result has the same shape as
// gpu-bench.js (window.gpuBench), so scripts/bench-gpu.py --result gpu validates it.

import { requestDevice } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const HEADS = 16;
const DIM = 64;
// queries per workgroup and workgroup-memory needs of each kernel
const KERNELS = {
  attention: { queries: 16 },
  attention_subgroup: { queries: 16, subgroup32: true },
  attention_tile: { queries: 64, subgroup32: true },
};

const query = new URLSearchParams(location.search);
const log = (s) => (document.getElementById("log").textContent += s + "\n");
const state = { status: "running" };
window.gpuBench = state;

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32) * 2 - 1;
}

function parseCases() {
  return (query.get("cases") || "512:0,512:64,140:64,47:0").split(",").map((item) => {
    const [lens, window] = item.split(":");
    return { label: item, lens: lens.split("/").map(Number), window: Number(window || 0) };
  });
}

/** CPU attention for query i of head h (the reference the kernels must match). */
function reference(qkv, tok, i, h, window, W, stride) {
  const [s0, L, pos] = [tok[i * 4], tok[i * 4 + 1], tok[i * 4 + 2]];
  const lo = window ? Math.max(0, pos - window) : 0;
  const hi = window ? Math.min(L, pos + window + 1) : L;
  const scores = [];
  for (let j = lo; j < hi; j++) {
    let s = 0;
    for (let c = 0; c < DIM; c++) s += qkv[i * stride + h * DIM + c] * qkv[(s0 + j) * stride + W + h * DIM + c];
    scores.push(s / 8);
  }
  const m = Math.max(...scores);
  const e = scores.map((v) => Math.exp(v - m));
  const z = e.reduce((a, b) => a + b, 0);
  const out = new Float64Array(DIM);
  for (let j = lo; j < hi; j++) for (let c = 0; c < DIM; c++) out[c] += (e[j - lo] / z) * qkv[(s0 + j) * stride + 2 * W + h * DIM + c];
  return out;
}

async function main() {
  const gpu = await requestDevice();
  const d = gpu.device;
  const wgsl = await kernelSource();
  const timestamps = d.features.has("timestamp-query");
  const names = (query.get("kernels") || "attention,attention_subgroup").split(",");
  const samples = Number(query.get("samples") || 7);
  const warmups = Number(query.get("warmups") || 2);
  const usable = names.filter((n) => !KERNELS[n].subgroup32 || gpu.subgroup32);
  log(`device: ${gpu.name}; kernels ${usable.join(", ")}${usable.length < names.length ? " (others need 32-lane subgroups)" : ""}`);
  const layout = d.createBindGroupLayout({
    entries: ["uniform", "uniform", "read-only-storage", "read-only-storage", "storage"].map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })),
  });
  const pipes = {};
  for (const name of usable) {
    const module = d.createShaderModule({ code: wgsl(name, { subgroups: !!KERNELS[name].subgroup32 }) });
    pipes[name] = await d.createComputePipelineAsync({ layout: d.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint: "main" } });
  }
  const querySet = timestamps ? d.createQuerySet({ type: "timestamp", count: 2 }) : null;
  const resolve = d.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const readTs = d.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const W = HEADS * DIM;
  const stride = 3 * W;
  const cases = [];
  let ok = true;
  for (const [ci, c] of parseCases().entries()) {
    const T = c.lens.reduce((a, b) => a + b, 0);
    const r = rng(7 + ci);
    const qkv = new Float32Array(T * stride).map(() => r() * 1.5);
    const tok = new Uint32Array(T * 4);
    let start = 0;
    for (const L of c.lens) {
      for (let p = 0; p < L; p++) tok.set([start, L, p, 0], (start + p) * 4);
      start += L;
    }
    const buffer = (data, usage) => {
      const b = d.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      d.queue.writeBuffer(b, 0, data);
      return b;
    };
    const QKV = buffer(qkv, GPUBufferUsage.STORAGE);
    const CTX = buffer(new Float32Array(T * W), GPUBufferUsage.STORAGE);
    const params = buffer(new Uint32Array([W, stride, c.window, 0]), GPUBufferUsage.UNIFORM);
    // query samples for the CPU check: every query of small cases, a spread of them otherwise
    const checks = [];
    const step = Math.max(1, Math.floor((T * HEADS) / 96));
    for (let n = 0; n < T * HEADS; n += step) checks.push([Math.floor(n / HEADS), n % HEADS]);
    checks.push([T - 1, HEADS - 1]);
    const want = checks.map(([i, h]) => reference(qkv, tok, i, h, c.window, W, stride));

    const runs = usable.map((name) => {
      const qpb = KERNELS[name].queries;
      const blocks = [];
      let s0 = 0;
      for (const L of c.lens) {
        for (let q0 = 0; q0 < L; q0 += qpb) blocks.push(s0, L, q0, 0);
        s0 += L;
      }
      const n = blocks.length / 4;
      const globals = buffer(new Uint32Array([T, 0, n, 0]), GPUBufferUsage.UNIFORM);
      const group = d.createBindGroup({ layout, entries: [globals, params, QKV, buffer(new Uint32Array(blocks), GPUBufferUsage.STORAGE), CTX].map((b, binding) => ({ binding, resource: { buffer: b } })) });
      // keys each query sees, for the work estimate that sizes a timed batch
      const keys = c.window ? Math.min(T, 2 * c.window + 1) : Math.max(...c.lens);
      const reps = Math.max(1, Math.min(256, Math.round(4e9 / (4 * T * keys * DIM * HEADS))));
      return { name, group, blocks: n, reps, samples: [] };
    });
    const timed = async (run) => {
      const e = d.createCommandEncoder();
      const pass = e.beginComputePass(timestamps ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
      pass.setPipeline(pipes[run.name]);
      pass.setBindGroup(0, run.group);
      for (let i = 0; i < run.reps; i++) pass.dispatchWorkgroups(run.blocks, HEADS);
      pass.end();
      if (timestamps) {
        e.resolveQuerySet(querySet, 0, 2, resolve, 0);
        e.copyBufferToBuffer(resolve, 0, readTs, 0, 16);
      }
      const t0 = performance.now();
      d.queue.submit([e.finish()]);
      if (!timestamps) {
        await d.queue.onSubmittedWorkDone();
        return (performance.now() - t0) / run.reps;
      }
      await readTs.mapAsync(GPUMapMode.READ);
      const [a, b] = new BigInt64Array(readTs.getMappedRange().slice(0));
      readTs.unmap();
      return Number(b - a) / 1e6 / run.reps;
    };
    const turn = (round) => runs.map((_, i) => runs[(i + round) % runs.length]);
    for (let i = 0; i < warmups; i++) for (const run of turn(i)) await timed(run);
    for (let i = 0; i < samples; i++) for (const run of turn(i)) run.samples.push(await timed(run));

    const timings = [];
    const errors = {};
    for (const run of runs) {
      d.queue.writeBuffer(CTX, 0, new Float32Array(T * W));
      const e = d.createCommandEncoder();
      const pass = e.beginComputePass();
      pass.setPipeline(pipes[run.name]);
      pass.setBindGroup(0, run.group);
      pass.dispatchWorkgroups(run.blocks, HEADS);
      pass.end();
      const out = d.createBuffer({ size: T * W * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      e.copyBufferToBuffer(CTX, 0, out, 0, T * W * 4);
      d.queue.submit([e.finish()]);
      await out.mapAsync(GPUMapMode.READ);
      const got = new Float32Array(out.getMappedRange().slice(0));
      out.destroy();
      let err = 0;
      checks.forEach(([i, h], k) => {
        for (let x = 0; x < DIM; x++) err = Math.max(err, Math.abs(got[i * W + h * DIM + x] - want[k][x]));
      });
      if (!(err < 2e-3)) ok = false;
      errors[run.name] = err;
      const sorted = [...run.samples].sort((a, b) => a - b);
      timings.push({ kernel: run.name, reps: run.reps, samples: run.samples, medianMs: sorted[sorted.length >> 1] });
    }
    cases.push({ shape: c.label, T, window: c.window, timings, errors, medianMs: timings[0].medianMs });
    log(`${c.label}: ${timings.map((t) => `${t.kernel}=${t.medianMs.toFixed(3)}ms (err ${errors[t.kernel].toExponential(1)})`).join(", ")}`);
  }
  const metricMs = Math.exp(cases.reduce((s, c) => s + Math.log(c.medianMs), 0) / cases.length);
  const info = gpu.adapter.info || {};
  Object.assign(state, {
    status: "done",
    backend: "webgpu",
    metricMs,
    method: timestamps ? "WebGPU timestamp-query" : "wall clock",
    kernels: usable,
    cases,
    adapter: { vendor: info.vendor, architecture: info.architecture, isFallbackAdapter: !!info.isFallbackAdapter },
    correctness: { ok, limit: 2e-3 },
  });
  log(`metric ${metricMs.toFixed(3)} ms (first kernel), correctness ${ok ? "ok" : "FAILED"}`);
}

main().catch((e) => {
  log("ERROR " + (e.stack || e));
  Object.assign(state, { status: "error", error: String(e.message || e) });
});
