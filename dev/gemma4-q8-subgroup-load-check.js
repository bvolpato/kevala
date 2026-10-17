// DEV-only exact subgroup sharing of redundant Q8 loader weights and scales.
import { matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { selectMatmulKernel } from "../js/src/gpu-tuning.js";
import { kernelSource } from "./wgsl.js";

const SENTINEL = 12345.25, LIMIT = 64 * 1024 * 1024;
const samples = [], params = new URLSearchParams(location.search);
const log = (row) => { document.getElementById("log").textContent += JSON.stringify(row) + "\n"; };
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((b) => b.toString(16).padStart(2, "0")).join("");

function subgroupLoads(code) {
  const signature = "@builtin(local_invocation_index) li: u32)";
  const begin = "      if (wrow < N) {\n        let v = W[(wrow * K + kb * 32u) / 16u + lq / 2u];\n        let s = S[wrow * nb + kb];\n        let lo = select(v.xy, v.zw, (lq & 1u) == 1u);\n        wv[2u * gg] = sx(lo.x) * s;\n        wv[2u * gg + 1u] = sx(lo.y) * s;\n      }";
  for (const marker of [signature, begin]) if (code.split(marker).length !== 2) throw new Error("subgroup loader marker mismatch: " + marker);
  const scaleOnly = params.get("mode") === "scale";
  const loads = scaleOnly ? "var raw = vec4<u32>(0u); if (wrow < N) { raw = W[(wrow * K + kb * 32u) / 16u + lq / 2u]; }" : "var loaded = vec4<u32>(0u); if (wrow < N && (lq & 1u) == 0u) { loaded = W[(wrow * K + kb * 32u) / 16u + lq / 2u]; } let raw = subgroupShuffle(loaded, lane & ~1u);";
  return "enable subgroups;\n" + code.replace(signature, "@builtin(local_invocation_index) li: u32, @builtin(subgroup_invocation_id) lane: u32)")
    .replace(begin, `      var scale = 0.0;
      if (wrow < N && lq == 0u) { scale = S[wrow * nb + kb]; }
      let s = subgroupShuffle(scale, lane & ~3u);
      ${loads}
      let lo = select(raw.xy, raw.zw, (lq & 1u) == 1u);
      wv[2u * gg] = sx(lo.x) * s;
      wv[2u * gg + 1u] = sx(lo.y) * s;`);
}

function compare(a, b) {
  const x = new Uint32Array(a.buffer, a.byteOffset, a.length), y = new Uint32Array(b.buffer, b.byteOffset, b.length);
  let differences = 0, nonfinite = 0, maxAbs = 0;
  for (let i = 0; i < a.length; i++) {
    if (x[i] !== y[i]) differences++;
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) nonfinite++;
    else maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
  }
  return { ok: a.length === b.length && !differences && !nonfinite, differences, nonfinite, maxAbs, values: a.length, signedZeroChecked: true };
}

async function check(runtime, source, c, kind, f16) {
  const { device } = runtime, U = GPUBufferUsage;
  const { T, N, K, rows, splitT, policy, row56 = false } = c;
  const splits = mmSplits(splitT, N, K, 256, rows * 16, 64);
  const yLength = T * N, partLength = splits > 1 ? splits * yLength : 0;
  const yBytes = (yLength + 4) * 4, partBytes = (partLength + 4) * 4;
  const stride = yBytes + partBytes;
  const scratchBytes = 6 * stride + 2 * T * K * 4 + 2048;
  if (scratchBytes > LIMIT) return { ...c, skipped: true, reason: "temporary-memory-limit", scratchBytes };
  const kernel = kind === "wide" ? "matmul_wide" : "matmul";
  const spec = { weights: "q8", f16, rows, row56, groups: 1, splitTarget: 256 };
  const original = source(kernel + policy, spec), candidate = subgroupLoads(original);
  const mmLayout = matmulLayout(device), rLayout = reduceLayout(device);
  const pl = device.createPipelineLayout({ bindGroupLayouts: [mmLayout] });
  const [base, fast, reduce] = await Promise.all([
    pipeline(device, original, "q8_load16", pl), pipeline(device, candidate, "q8_subgroup_load", pl),
    pipeline(device, source("reduce" + policy, spec), "q8_load_reduce", device.createPipelineLayout({ bindGroupLayouts: [rLayout] })),
  ]);
  const owned = [], deadline = performance.now() + 30000;
  let query, read, mapped = false;
  const make = (size, usage, data) => {
    const b = device.createBuffer({ size: Math.max(16, size), usage }); owned.push(b);
    if (data) device.queue.writeBuffer(b, 0, data);
    return b;
  };
  const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  try {
    const weights = Int8Array.from({ length: N * K }, (_, i) => (i * 31 + (i >>> 5) * 17 + (i >>> 10) * 7 + 19) % 256 - 128);
    const scales = Float32Array.from({ length: N * K / 32 }, (_, i) => i % 257 === 0 ? -0 : (i % 2 ? -1 : 1) * (0.0002 + i % 11 * 0.000037));
    const w = make(weights.byteLength, U.STORAGE | U.COPY_DST, weights), s = make(scales.byteLength, U.STORAGE | U.COPY_DST, scales);
    const x = make(T * K * 4, U.STORAGE | U.COPY_DST, Float32Array.from({ length: T * K }, (_, i) => Math.fround(.04 * Math.sin(i * .127 + Math.floor(i / K) * .13 + .3))));
    const g = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([T, splitT, splitT, 0]));
    const p = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([N, K, 0, 0]));
    const zero = make(16, U.STORAGE);
    const poisoned = (length) => { const a = new Float32Array(length + 4).fill(NaN); a.fill(SENTINEL, length); return a; };
    const ys = [0, 1].map(() => make(yBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, poisoned(yLength)));
    const parts = [0, 1].map(() => make(partBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, poisoned(partLength)));
    const ops = ys.map((y, i) => ({ group: group(mmLayout, [g, p, x, w, s, zero, y, parts[i]]), reduce: group(rLayout, [g, p, parts[i], zero, y]) }));
    const dispatch = (pass, index) => {
      pass.setPipeline(index ? fast : base); pass.setBindGroup(0, ops[index].group);
      pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(T / (row56 ? 56 : rows * 16)), splits);
      if (splits > 1) {
        pass.setPipeline(reduce); pass.setBindGroup(0, ops[index].reduce);
        const n = Math.ceil(yLength / 256); pass.dispatchWorkgroups(Math.min(n, 65535), Math.ceil(n / 65535));
      }
    };
    read = make(Math.max(256, 2 * stride), U.MAP_READ | U.COPY_DST);
    const verify = async () => {
      for (const y of ys) device.queue.writeBuffer(y, 0, poisoned(yLength));
      for (const part of parts) device.queue.writeBuffer(part, 0, poisoned(partLength));
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    dispatch(pass, 0); dispatch(pass, 1); pass.end();
    for (let i = 0; i < 2; i++) { encoder.copyBufferToBuffer(ys[i], 0, read, i * stride, yBytes); encoder.copyBufferToBuffer(parts[i], 0, read, i * stride + yBytes, partBytes); }
    device.queue.submit([encoder.finish()]);
    if (!await mapBeforeDeadline(read, 2 * stride, deadline)) { read.unmap(); throw new Error("Q8 load guard deadline"); }
    mapped = true;
    const data = new Float32Array(read.getMappedRange(0, 2 * stride).slice(0)); read.unmap(); mapped = false;
    const output = compare(data.subarray(0, yLength), data.subarray(stride / 4, stride / 4 + yLength));
    const partials = partLength ? compare(data.subarray(yBytes / 4, yBytes / 4 + partLength), data.subarray((stride + yBytes) / 4, (stride + yBytes) / 4 + partLength)) : null;
    const guards = [yLength, stride / 4 - 4, stride / 4 + yLength, stride / 2 - 4].every((i) => data.subarray(i, i + 4).every((v) => v === SENTINEL));
    const ok = output.ok && (!partials || partials.ok) && guards;
      return { output, partials, guards, ok };
    };
    const before = await verify(), { output, partials, guards, ok } = before;
    let timing;
    if (ok) {
      query = device.createQuerySet({ type: "timestamp", count: 12 });
      const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC);
      const e = device.createCommandEncoder(), warm = e.beginComputePass(); dispatch(warm, 0); dispatch(warm, 1); warm.end();
      const order = { baseline: [], candidate: [] }; let index = 0;
      for (let pair = 0; pair < 3; pair++) for (const kind of pair % 2 ? [1, 0] : [0, 1]) {
        order[kind ? "candidate" : "baseline"].push(index);
        const p = e.beginComputePass({ timestampWrites: { querySet: query, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
        for (let rep = 0; rep < 4; rep++) dispatch(p, kind); p.end(); index += 2;
      }
      e.resolveQuerySet(query, 0, 12, resolved, 0); e.copyBufferToBuffer(resolved, 0, read, 0, 96); device.queue.submit([e.finish()]);
      if (!await mapBeforeDeadline(read, 96, deadline)) { read.unmap(); throw new Error("Q8 load timing deadline"); }
      mapped = true;
      const stamps = new BigInt64Array(read.getMappedRange(0, 96).slice(0)); read.unmap(); mapped = false;
      const times = Object.fromEntries(Object.entries(order).map(([k, indices]) => [k, indices.map(i => Number(stamps[i + 1] - stamps[i]) / 1e6 / 4)]));
      const selection = selectMatmulKernel({ generic: times.baseline, wide: times.candidate });
      timing = { ...times, baselineMs: selection.genericMs, candidateMs: selection.wideMs, benefit: selection.benefit, selected: selection.kernel === "wide", stable: selection.stable, reason: selection.reason };
    }
    const after = await verify();
    return { ...c, kind, f16, splits, mode: params.get("mode") || "both", ok: ok && after.ok, before, after, output, partials, guards, timing, scratchBytes, sharedWeightBytes: weights.byteLength + scales.byteLength, shaderHashes: { baseline: await hash(original), candidate: await hash(candidate) } };
  } finally { if (mapped) read.unmap(); query?.destroy(); for (const b of owned) b.destroy(); }
}

async function main() {
  const runtime = await requestDevice({ features: ["timestamp-query", "subgroups"] });
  try {
    const f16 = runtime.device.features.has("shader-f16"), kind = params.get("kernel") || "generic";
    if (!runtime.subgroup32) throw new Error("fixed subgroup32 unavailable");
    if (!runtime.device.features.has("timestamp-query")) throw new Error("timestamp-query unavailable");
    if (params.get("f16") === "1" && !f16) throw new Error("shader-f16 unavailable");
    if (!["generic", "wide"].includes(kind)) throw new Error("invalid kernel");
    const source = await kernelSource();
    const configurations = [{ T: 97, splitT: 1, rows: 4, row56: true, policy: "_gemma_batch" }];
    for (const config of configurations) for (const [N, K] of [[6144, 1536], [12288, 1536], [1536, 6144], [2048, 1536]]) {
      const result = await withGpuErrors(runtime, () => check(runtime, source, { ...config, N, K }, kind, f16), "Q8 narrow load guard"); samples.push(result); log(result);
      if (!result.skipped && !result.ok) throw new Error("Q8 narrow load output differs");
    }
    const measured = samples.filter(s => s.timing);
    window.gpuBench = { status: "done", done: true, backend: "webgpu", adapter: { name: runtime.name, features: [...runtime.device.features] },
      correctness: { ok: true, bitExact: true, cases: samples }, metricMs: Math.exp(measured.reduce((a, s) => a + Math.log(s.timing.candidateMs), 0) / measured.length), method: "GPU timestamps; subgroup32 shares identical packed Q8 weights/scales among loader lanes; complete Y/PART/signed-zero/finiteness/tails before and after timing" };
  } finally { runtime.device.destroy(); }
}
window.gpuBench = { status: "running", backend: "webgpu" };
main().catch(error => { window.gpuBench = { status: "error", done: true, backend: "webgpu", error: String(error?.message || error), correctness: { ok: false, cases: samples } }; log(error.stack || String(error)); });
