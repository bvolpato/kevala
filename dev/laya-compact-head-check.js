// Direct Laya trunk guard for the final decision-head compact FFN. The page loads the production
// GPU module and a pre-change module supplied as ?baseline=...; it does not use a public override.
// The synthetic pack keeps this check small while exercising the actual bind groups, matmuls,
// attention, segment table, row gather, residual add, and readback path.

import { kernelSource } from "./wgsl.js";

const D = 1024;
const F = new URLSearchParams(location.search).get("small") === "1" ? 64 : 4096;
const HEAD_LAYERS = 2;
const HIDDEN_LIMIT = 2e-5;
const logNode = document.getElementById("log");
const log = (line) => { logNode.textContent += `${line}\n`; };

function tensor(name, dtype, shape) {
  const elements = shape.reduce((a, b) => a * b, 1);
  const q8 = dtype === "q8";
  return {
    name,
    dtype,
    shape,
    size: q8 ? elements : elements * 4,
    ...(q8 ? { block: 32, scales_size: shape[0] * (shape[1] / 32) * 4 } : {}),
  };
}

function makePack() {
  const template = [
    tensor("head.0.norm1.w", "f32", [D]),
    tensor("head.0.norm1.b", "f32", [D]),
    tensor("head.0.in_proj", "q8", [3 * D, D]),
    tensor("head.0.in_proj.b", "f32", [3 * D]),
    tensor("head.0.out_proj", "q8", [D, D]),
    tensor("head.0.out_proj.b", "f32", [D]),
    tensor("head.0.norm2.w", "f32", [D]),
    tensor("head.0.norm2.b", "f32", [D]),
    tensor("head.0.lin1", "q8", [F, D]),
    tensor("head.0.lin1.b", "f32", [F]),
    tensor("head.0.lin2", "q8", [D, F]),
    tensor("head.0.lin2.b", "f32", [D]),
  ];
  const tensors = Array.from({ length: HEAD_LAYERS }, (_, i) => template.map((t) => ({ ...t, name: t.name.replace("head.0.", `head.${i}.`) }))).flat();
  const dataStart = 4096;
  let offset = dataStart;
  const metadata = tensors.map((item) => {
    const out = { ...item, offset };
    offset += item.size;
    if (item.dtype === "q8") {
      out.scales_offset = offset;
      offset += item.scales_size;
    }
    return out;
  });
  const header = new TextEncoder().encode(JSON.stringify({ tensors: metadata }));
  if (header.byteLength + 16 > dataStart) throw new Error("synthetic pack metadata overlaps weights");
  const prefix = new Uint8Array(16 + header.byteLength);
  prefix.set([0x4b, 0x56, 0x4c, 0x41]); // KVLA
  const view = new DataView(prefix.buffer);
  view.setUint32(4, 1, true);
  view.setUint32(8, header.byteLength, true);
  prefix.set(header, 16);
  const bytes = new Uint8Array(offset);
  bytes.set(prefix);
  const raw = new DataView(bytes.buffer);
  metadata.forEach((item, ti) => {
    if (item.dtype === "q8") {
      for (let i = 0; i < item.size; i++) {
        const value = ((i * 19 + ti * 7) % 17) - 8;
        bytes[item.offset + i] = value & 0xff;
      }
      for (let i = 0; i < item.scales_size / 4; i++) raw.setFloat32(item.scales_offset + i * 4, 0.02 + (i % 3) * 0.003, true);
    } else {
      for (let i = 0; i < item.size / 4; i++) {
        const value = item.name.endsWith(".w") ? 1 + (i % 7) * 0.003 : ((i + ti) % 9 - 4) * 0.002;
        raw.setFloat32(item.offset + i * 4, value, true);
      }
    }
  });
  return { layout: { prefix, total: bytes.byteLength }, bytes };
}

function config() {
  return {
    hidden: D,
    heads: D / 64,
    layers: 0,
    intermediate: F,
    global_every: 1,
    window: 0,
    rope_global: 10000,
    rope_local: 10000,
    norm_eps: 1e-5,
    head_layers: HEAD_LAYERS,
    head_ff: F,
    head_heads: D / 64,
    head_norm_eps: 1e-5,
    steps: 2,
  };
}

function bridgeTensors() {
  const finalNorm = new Float32Array(D).fill(1);
  const typeEmb = new Float32Array(4 * D);
  for (let i = 0; i < typeEmb.length; i++) typeEmb[i] = ((i % 11) - 5) * 0.001;
  return { finalNorm, typeEmb };
}

function inputFor(lengths) {
  const segs = [];
  let start = 0;
  for (const [index, len] of lengths.entries()) {
    segs.push({ start, len, qtype: index % 2 });
    start += len;
  }
  const x = new Float32Array(start * D);
  for (let i = 0; i < x.length; i++) x[i] = Math.fround(Math.sin((i + 3) * 0.17) * 0.2 + Math.cos((i + 11) * 0.031) * 0.05);
  return { x, segs, tokens: start };
}

function allRows(tokens) {
  return Uint32Array.from({ length: tokens }, (_, i) => i);
}

function rowError(full, picked, rows, width) {
  if (picked.length !== rows.length * width) return { ok: false, maxAbs: Infinity, message: `readback length ${picked.length} != ${rows.length * width}` };
  let maxAbs = 0;
  let worst = null;
  for (let r = 0; r < rows.length; r++) for (let d = 0; d < width; d++) {
    const actual = picked[r * width + d];
    const expected = full[rows[r] * width + d];
    if (!Number.isFinite(actual) || !Number.isFinite(expected)) {
      return {
        ok: false,
        maxAbs: Infinity,
        worst: { row: rows[r], dim: d, expected, actual },
        message: `non-finite value at row ${rows[r]}, dim ${d}`,
      };
    }
    const abs = Math.abs(actual - expected);
    if (abs > maxAbs) { maxAbs = abs; worst = { row: rows[r], dim: d, expected, actual }; }
  }
  return { ok: Number.isFinite(maxAbs) && maxAbs <= HIDDEN_LIMIT, maxAbs, worst };
}

async function makeModel(Module, runtime, pack, cfg, bridge) {
  const model = new Module.GpuTrunk(runtime, pack.layout, cfg);
  model.write(0, pack.bytes);
  await model.init(bridge.finalNorm, bridge.typeEmb);
  return model;
}

// Record pipelines and grids through the existing profiler hook. Hidden outputs still come
// from GpuTrunk's storage-buffer readback; the baseline uses its original graph.
async function tracedForward(model, input, rows) {
  const trace = [];
  model.profiler = {
    pass(encoder, label) {
      const pass = encoder.beginComputePass();
      const entry = { label, pipelines: [], dispatches: [] };
      trace.push(entry);
      return {
        setPipeline(p) { entry.pipelines.push(p === model.mm.mm[1] ? "matmul_r1" : p === model.mm.reduce[1] ? "reduce_r1" : "other"); pass.setPipeline(p); },
        setBindGroup(...args) { pass.setBindGroup(...args); },
        dispatchWorkgroups(...args) { entry.dispatches.push(args); pass.dispatchWorkgroups(...args); },
        end() { pass.end(); },
      };
    },
    finish() {},
    async collect() { return trace; },
  };
  return { hidden: await model.forward(input.x, input.segs, rows), trace };
}

async function main() {
  const query = new URLSearchParams(location.search);
  const baseline = query.get("baseline");
  if (!baseline) throw new Error("baseline=URL-to-pre-change-gpu.js is required");
  const [candidateModule, baselineModule] = await Promise.all([
    import(new URL("../js/src/gpu.js", import.meta.url).href),
    import(new URL(baseline, location.href).href),
  ]);
  const source = await kernelSource();
  // Keep this guard about the final FFN row contract. Generic attention and FP32 matmul are
  // sufficient here, and make the page usable on browsers without optional subgroup or f16 paths.
  const runtime = await candidateModule.requestDevice({ features: ["timestamp-query"] });
  runtime.wgsl = source;
  runtime.kernel = "generic";
  runtime.subgroup32 = false;
  runtime.attentionTile = false;
  runtime.attentionTileShared = false;
  const cfg = config();
  const pack = makePack();
  const bridge = bridgeTensors();
  const candidate = await makeModel(candidateModule, runtime, pack, cfg, bridge);
  const reference = await makeModel(baselineModule, runtime, pack, cfg, bridge);
  const cases = [
    { name: "reordered-duplicates", lengths: [3, 5], rows: [7, 0, 7, 4] },
    { name: "single-row", lengths: [6], rows: [3] },
    { name: "multiple-rows", lengths: [2, 4, 3], rows: [8, 1, 5] },
    { name: "all-rows-fallback", lengths: [3, 2], rows: [0, 1, 2, 3, 4] },
    { name: "segmented-tail-duplicates", lengths: [17, 1, 5], rows: [22, 17, 22, 0] },
    { name: "sixteen-selected", lengths: [17, 16], rows: Array.from({ length: 16 }, (_, i) => 32 - i * 2) },
    { name: "seventeen-selected-fallback", lengths: [17, 16], rows: Array.from({ length: 17 }, (_, i) => 32 - i) },
    { name: "short-R3", lengths: [47], rows: [46, 0, 21, 46] },
    { name: "medium-R3", lengths: [65, 75], rows: [139, 0, 65, 139] },
    { name: "different-split-counts", lengths: [257, 64], rows: [320, 0, 256, 320] },
    { name: "long-R4", lengths: [512], rows: [511, 0, 256, 511] },
  ];
  const results = [];
  let ok = true;
  const started = performance.now();
  const checkCase = async (item, candidate, reference, cfg) => {
    const input = inputFor(item.lengths);
    const rows = Uint32Array.from(item.rows);
    const fullRows = allRows(input.tokens);
    const candidateFull = await tracedForward(candidate, input, fullRows);
    const referenceFull = await reference.forward(input.x, input.segs, fullRows);
    const candidatePicked = await tracedForward(candidate, input, rows);
    const referencePicked = await reference.forward(input.x, input.segs, rows);
    const candidateRows = rowError(candidateFull.hidden, candidatePicked.hidden, rows, D);
    const referenceRows = rowError(referenceFull, referencePicked, rows, D);
    const crossFull = rowError(referenceFull, candidateFull.hidden, fullRows, D);
    // Preserve original row IDs when comparing the selected output with the full baseline.
    const crossPicked = rowError(referenceFull, candidatePicked.hidden, rows, D);
    const selectedPair = rowError(referencePicked, candidatePicked.hidden, allRows(rows.length), D);
    const compact = cfg.head_layers > 0 && rows.length > 0 && rows.length <= 16 && rows.length < input.tokens;
    const originalR = candidateModule.rowsPerThread(input.tokens, candidate.mm);
    const virtualT = Math.min(input.tokens, Math.ceil(input.tokens / (16 * originalR)) * 16);
    const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
    const tail = candidatePicked.trace.filter((p) => p.label.includes("head.compact"));
    const plans = [["lin1", F, D], ["lin2", D, F]].map(([name, N, K]) => {
      const originalSplits = candidateModule.mmSplits(input.tokens, N, K, candidate.mm.splitTarget, 16 * originalR, 64);
      const compactSplits = candidateModule.mmSplits(virtualT, N, K, candidate.mm.splitTarget, 16, 64);
      const expectedDispatches = [[Math.ceil(N / 64), Math.ceil(rows.length / 16), originalSplits]];
      const expectedPipelines = ["matmul_r1"];
      if (originalSplits > 1) {
        expectedDispatches.push([Math.ceil(rows.length * N / 256), 1]);
        expectedPipelines.push("reduce_r1");
      }
      const recorded = tail.find((p) => p.label === `mm.head.compact.${name}`);
      return { name, originalSplits, compactSplits, recorded, ok: !compact || (originalSplits === compactSplits && same(recorded?.dispatches, expectedDispatches) && same(recorded?.pipelines, expectedPipelines)) };
    });
    const executionOk = candidateFull.trace.every((p) => !p.label.includes("head.compact")) &&
      (compact ? tail.length === 4 && plans.every((p) => p.ok) &&
        same(tail.find((p) => p.label === "gather.head.compact")?.dispatches, [[rows.length]]) &&
        same(tail.find((p) => p.label === "norm.head.compact")?.dispatches, [[rows.length]]) : tail.length === 0);
    const result = {
      name: item.name,
      tokens: input.tokens,
      segments: item.lengths,
      rows: item.rows,
      headLayers: cfg.head_layers,
      compact,
      originalR,
      virtualT,
      candidate: { hidden: candidateRows },
      baseline: { hidden: referenceRows },
      crossModule: { full: crossFull, selected: crossPicked, selectedPair },
      execution: { ok: executionOk, plans },
    };
    results.push(result);
    ok &&= candidateRows.ok && referenceRows.ok && crossFull.ok && crossPicked.ok && selectedPair.ok && executionOk;
    log(`${item.name}: compact=${compact} virtualT=${virtualT} hidden=${crossPicked.maxAbs.toExponential(2)} dispatch=${executionOk}`);
  };
  for (const item of cases) await checkCase(item, candidate, reference, cfg);
  const noHead = { ...cfg, head_layers: 0 };
  const candidateNoHead = await makeModel(candidateModule, runtime, pack, noHead, bridge);
  const referenceNoHead = await makeModel(baselineModule, runtime, pack, noHead, bridge);
  await checkCase({ name: "zero-head-layers-fallback", lengths: [17, 6], rows: [22, 0, 17, 22] }, candidateNoHead, referenceNoHead, noHead);
  const info = runtime.adapter.info || {};
  window.gpuBench = {
    status: ok ? "done" : "error",
    done: true,
    backend: "webgpu",
    check: "laya-final-head-compact",
    baseline,
    dimensions: { hidden: D, intermediate: F, headLayers: HEAD_LAYERS },
    metricMs: performance.now() - started,
    method: "Correctness suite elapsed wall time; not model latency",
    adapter: {
      name: runtime.name,
      vendor: info.vendor || null,
      architecture: info.architecture || null,
      device: info.device || null,
      features: [...runtime.device.features].sort(),
    },
    correctness: {
      ok,
      hiddenLimit: HIDDEN_LIMIT,
      reorderedDuplicates: true,
      singleRows: true,
      multipleRows: true,
      allRowsFallback: true,
      zeroHeadLayersFallback: true,
      moreThanSixteenRowsFallback: true,
      originalSplitPartitions: true,
      segmentedInputs: true,
      cases: results,
    },
  };
  if (!ok) throw new Error("final-head compact guard failed");
  log(`done: ${results.length} full-vs-compact cases passed`);
}

window.gpuBench = { status: "running", done: false, check: "laya-final-head-compact" };
main().catch((error) => {
  window.gpuBench = { ...window.gpuBench, status: "error", done: true, error: String(error?.message || error) };
  log(`ERROR ${error?.stack || error}`);
});
