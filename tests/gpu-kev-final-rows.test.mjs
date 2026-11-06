import assert from "node:assert/strict";
import test from "node:test";
import { GpuKev, encodeFinalRowMatmul, finalRowPlan } from "../js/src/gpu-kev.js";
import { mmSplits, rowsPerThread } from "../js/src/gpu.js";

test("final rows preserve sparse order and duplicates while all-row consumers keep their graph", () => {
  assert.deepEqual(finalRowPlan(97, [96, 4, 96]), { active: true, sourceRows: 97, scorerRows: 3, reason: "selected-rows" });
  assert.equal(finalRowPlan(3, [0, 1, 2]).reason, "all-rows");
  assert.equal(finalRowPlan(3, [2, 0, 1]).active, false);
  assert.equal(finalRowPlan(3, [2, 2, 2, 2]).active, false);
  assert.equal(finalRowPlan(97, [96], false).reason, "disabled");
  for (const rows of [[], [-1], [97], [0.5]]) assert.equal(finalRowPlan(97, rows).reason, "invalid-rows");
});

test("compact matrices retain the original row tile, padding policy and split-K reduction", () => {
  for (const sourceT of [7, 47, 65, 97, 129, 512]) for (const f16 of [false, true]) for (const kernel of ["matmul", "matmul_wide"]) {
    const pipes = { f16, kernel, groups: 1, splitTarget: 256,
      mm: [null, "r1", "r2", "r3", "r4"], mm56: "row56", reduce: [null, "reduce1", "reduce2", "reduce3", "reduce4"] };
    const ops = [], op = { N: 256, K: 1536, group: "matrix", reduce: "reduce" };
    const pass = { setPipeline: (value) => ops.push(["pipeline", value]), setBindGroup: (_, value) => ops.push(["group", value]), dispatchWorkgroups: (...value) => ops.push(["grid", ...value]) };
    encodeFinalRowMatmul(pass, pipes, op, 1, sourceT);
    const rows = rowsPerThread(sourceT, pipes);
    const row56 = rows === 4 && !f16 && Math.ceil(sourceT / 56) === Math.ceil(sourceT / 64);
    assert.equal(ops[0][1], row56 ? pipes.mm56 : pipes.mm[rows]);
    assert.deepEqual(ops[2], ["grid", 4, 1, mmSplits(sourceT, 256, 1536, 256, 16 * rows, 64)]);
    if (ops[2][3] > 1) {
      assert.equal(ops[3][1], pipes.reduce[rows]);
      assert.deepEqual(ops.at(-1), ["grid", 1, 1]);
    }
  }
});

test("one compact scorer row cannot activate the model's single-row matvec policy", () => {
  const gpu = Object.create(GpuKev.prototype), pipelines = [], grids = [];
  const pipes = { kernel: "matmul", f16: true, groups: 1, splitTarget: 256,
    mm: [null, "source1", "source2", "source3", "source4"], reduce: [null, "reduce1", "reduce2", "reduce3", "reduce4"] };
  gpu.precisionTuning = new Map([["q8", { selection: new Map(), pipelines: { generic: pipes } }]]);
  gpu.precisionMatvec = new Map([["q8", { selection: new Set(["256:1536"]), pipelines: { pipeline: "matvec", columns: 4 } }]]);
  gpu.gpuKernel = "auto";
  const pass = { setPipeline: (value) => pipelines.push(value), setBindGroup() {}, dispatchWorkgroups: (...value) => grids.push(value), end() {} };
  gpu.encode({ beginComputePass: () => pass }, [{ k: "mm", finalRows: true, finalPipes: pipes, dtype: "q8", N: 256, K: 1536 }], 1, 1, 97);
  assert.deepEqual(pipelines, ["source4", "reduce4"]);
  assert.deepEqual(grids, [[4, 1, mmSplits(97, 256, 1536, 256, 64, 64)], [1, 1]]);
});

test("a compact 57-row tail covers the physical BM56 boundary without changing its virtual split", () => {
  const pipes = { kernel: "matmul", f16: false, groups: 1, splitTarget: 256,
    mm: [null, "r1", "r2", "r3", "r4"], mm56: "row56", reduce: [null, "reduce1", "reduce2", "reduce3", "reduce4"] };
  const grids = [], pipelines = [], pass = { setPipeline: (value) => pipelines.push(value), setBindGroup() {}, dispatchWorkgroups: (...value) => grids.push(value) };
  encodeFinalRowMatmul(pass, pipes, { N: 2048, K: 1536 }, 57, 161);
  assert.deepEqual(pipelines, ["row56", "reduce4"]);
  assert.deepEqual(grids[0], [32, 2, mmSplits(161, 2048, 1536, 256, 64, 64)]);
  assert.deepEqual(grids[1], [456, 1]);
});

test("same physical row-tile work falls back before creating a compact pipeline", async () => {
  const gpu = Object.create(GpuKev.prototype), original = { kernel: "matmul", f16: false, groups: 1, mm56: "row56" };
  gpu.gpuKernel = "auto";
  gpu.finalRowsEnabled = true;
  gpu.finalRows = { tail: [{ k: "mm", N: 256, K: 1536, dtype: "q8", pipes: original, label: "mm.down" }] };
  gpu.precisionTuning = new Map([["q8", { pipelines: { generic: original }, selection: new Map() }]]);
  assert.equal(await gpu.prepareFinalRows(47, [46, 0, 46]), false);
  assert.equal(gpu.lastFinalRows.reason, "same-row-tiles");
  assert.deepEqual(gpu.lastFinalRows.tiles, [{ projection: "mm.down", source: 1, compact: 1 }]);
  assert.equal(await gpu.prepareFinalRows(97, Array.from({ length: 57 }, (_, i) => i)), false);
  assert.equal(gpu.lastFinalRows.reason, "same-row-tiles");
  assert.deepEqual(gpu.lastFinalRows.tiles, [{ projection: "mm.down", source: 2, compact: 2 }]);
});

test("unavailable source pipelines fall back without disabling the full graph", async () => {
  const gpu = Object.create(GpuKev.prototype), original = { kernel: "matmul", f16: true };
  gpu.gpuKernel = "auto";
  gpu.finalRowsEnabled = true;
  gpu.finalRows = { tail: [{ k: "mm", N: 256, K: 1536, dtype: "bf16", pipes: original }] };
  gpu.precisionTuning = new Map([["bf16", { pipelines: { generic: original }, selection: new Map() }]]);
  gpu.finalRowPipelineErrors = new Map([["bf16:matmul:r4:standard", "source-pipeline-unavailable: test device limit"]]);
  gpu.finalRowPipelines = new Map();
  assert.equal(await gpu.prepareFinalRows(97, [96]), false);
  assert.equal(gpu.lastFinalRows.active, false);
  assert.match(gpu.lastFinalRows.reason, /source-pipeline-unavailable/);
  assert.equal(gpu.finalRowsEnabled, true);
  assert.equal(await gpu.prepareFinalRows(3, [0, 1, 2]), false);
  assert.equal(gpu.lastFinalRows.reason, "all-rows");
});

test("forward bypasses async planning for short, disabled and all-row consumers", async (t) => {
  const prior = Object.getOwnPropertyDescriptor(globalThis, "GPUMapMode");
  Object.defineProperty(globalThis, "GPUMapMode", { configurable: true, value: { READ: 1 } });
  t.after(() => prior ? Object.defineProperty(globalThis, "GPUMapMode", prior) : delete globalThis.GPUMapMode);
  const forward = async ({ tokens, rows, enabled = true, compact = false, bypass = false }) => {
    const gpu = Object.create(GpuKev.prototype), encoded = [];
    gpu.cfg = { hidden: 1 };
    gpu.cap = { S: 1 };
    gpu.carry = [];
    gpu.ops = [[], [{ k: "full-branch" }]];
    gpu.finalRows = { prefix: [], gather: { k: "gather" }, tail: [{ k: "compact-tail" }] };
    gpu.finalRowsEnabled = enabled;
    gpu.lru = [];
    gpu.ensureCarries = gpu.ensure = () => {};
    gpu.plan = () => ({ segs: [], carries: [null], copies: [] });
    gpu.encode = (_, ops, activeT, segments, sourceT) => encoded.push({ ops, activeT, segments, sourceT });
    let plans = 0;
    gpu.prepareFinalRows = async () => {
      plans++;
      if (bypass) throw new Error("no-op forward awaited final-row planning");
      gpu.lastFinalRows = { active: compact, reason: compact ? "selected-rows" : "same-row-tiles" };
      return compact;
    };
    gpu.device = { pushErrorScope() {}, popErrorScope: async () => null,
      queue: { writeBuffer() {}, submit() {} },
      createCommandEncoder: () => ({ copyBufferToBuffer() {}, finish() {} }) };
    gpu.readback = { mapAsync: async () => {}, getMappedRange: (_, bytes) => new ArrayBuffer(bytes), unmap() {} };
    const batch = { T1: 0, T2: tokens, states: [[0, 0]], branches: [[0, tokens, 0]], rows };
    const result = await gpu.forward(new Float32Array(), new Float32Array(tokens), batch, new Uint32Array());
    assert.equal(result.length, rows.length);
    return { gpu, plans, encoded };
  };
  for (const tokens of [1, 7, 47, 64]) {
    const { gpu, plans } = await forward({ tokens, rows: [tokens - 1], bypass: true });
    assert.equal(plans, 0);
    assert.equal(gpu.lastFinalRows.reason, tokens === 1 ? "all-rows" : "same-row-tiles");
  }
  const all = await forward({ tokens: 97, rows: Array.from({ length: 97 }, (_, i) => i), bypass: true });
  assert.equal(all.plans, 0);
  assert.equal(all.gpu.lastFinalRows.reason, "all-rows");
  const disabled = await forward({ tokens: 97, rows: [96], enabled: false, bypass: true });
  assert.equal(disabled.plans, 0);
  assert.equal(disabled.gpu.lastFinalRows.reason, "disabled");
  const selected = await forward({ tokens: 65, rows: [64, 0, 64], compact: true });
  assert.equal(selected.plans, 1);
  assert.deepEqual(selected.encoded.at(-1), { ops: selected.gpu.finalRows.tail, activeT: 3, segments: 1, sourceT: 65 });
  const fallback = await forward({ tokens: 65, rows: [64], compact: false });
  assert.equal(fallback.plans, 1);
  assert.equal(fallback.gpu.lastFinalRows.reason, "same-row-tiles");
  assert.equal(fallback.encoded.at(-1).ops, fallback.gpu.ops[1]);
});
