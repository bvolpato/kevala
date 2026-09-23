// Direct Laya trunk guard for the final decision-head compact FFN. The page loads the production
// GPU module and a pre-change module supplied as ?baseline=...; it does not use a public override.
// The synthetic pack keeps this check small while exercising the actual bind groups, matmuls,
// attention, segment table, row gather, residual add, and readback path.

import { kernelSource } from "./wgsl.js";

const large = new URLSearchParams(location.search).get("size") === "full";
// Laya gather uses the supported 1024-wide hidden layout.
const D = 1024;
const F = large ? 4096 : 64;
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
  const tensors = [
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
    head_layers: 1,
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
  ];
  if (large) cases.push({ name: "different-split-counts", lengths: [257, 64], rows: [320, 0, 256, 320] });
  const started = performance.now();
  const results = [];
  let ok = true;
  for (const item of cases) {
    const input = inputFor(item.lengths);
    const rows = Uint32Array.from(item.rows);
    const fullRows = allRows(input.tokens);
    const candidateFull = await candidate.forward(input.x, input.segs, fullRows);
    const referenceFull = await reference.forward(input.x, input.segs, fullRows);
    const candidatePicked = await candidate.forward(input.x, input.segs, rows);
    const referencePicked = await reference.forward(input.x, input.segs, rows);
    const candidateRows = rowError(candidateFull, candidatePicked, rows, D);
    const referenceRows = rowError(referenceFull, referencePicked, rows, D);
    const crossFull = rowError(referenceFull, candidateFull, fullRows, D);
    const compactRows = Uint32Array.from({ length: rows.length }, (_, i) => i);
    const crossPicked = rowError(referencePicked, candidatePicked, compactRows, D);
    const result = {
      name: item.name,
      tokens: input.tokens,
      segments: item.lengths,
      rows: item.rows,
      compact: rows.length > 0 && rows.length < input.tokens,
      candidate: { hidden: candidateRows },
      baseline: { hidden: referenceRows },
      crossModule: { full: crossFull, selected: crossPicked },
    };
    results.push(result);
    ok &&= candidateRows.ok && referenceRows.ok && crossFull.ok && crossPicked.ok;
    log(`${item.name}: compact=${result.compact} hidden=${candidateRows.maxAbs.toExponential(2)} baseline=${referenceRows.maxAbs.toExponential(2)}`);
  }
  const info = runtime.adapter.info || {};
  window.gpuBench = {
    status: ok ? "done" : "error",
    done: true,
    backend: "webgpu",
    check: "laya-final-head-compact",
    metricMs: performance.now() - started,
    method: "Correctness suite elapsed wall time; not a model latency measurement",
    dimensions: { hidden: D, intermediate: F },
    baseline,
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
