import assert from "node:assert/strict";
import test from "node:test";

import { GpuGemma4 } from "../js/src/gpu-gemma4.js";
import { SPLIT_SCRATCH, mmSplits } from "../js/src/gpu.js";
import { calibrateGemma4BatchMlp, encodeGemma4BatchMlp, gemma4BatchMlpEligible, gemma4BatchMlpPartBytes, gemma4BatchMlpTallEligible } from "../js/src/gpu-gemma4-batch-mlp.js";

const GATE = { kind: "mm", N: 6144, K: 1536 };
const BASE_BYTES = SPLIT_SCRATCH * Float32Array.BYTES_PER_ELEMENT;
const E2B_BATCH_BYTES = 15_851_520;

function receiver({ maxBufferSize = 128 * 1024 * 1024, maxStorageBufferBindingSize = 128 * 1024 * 1024 } = {}) {
  const allocations = [];
  const builds = [];
  const value = {
    batchPrefixSuffixOps: [GATE],
    batchPart: null,
    part: { size: BASE_BYTES, destroy() { throw new Error("the baseline scratch must remain owned"); } },
    dynamic: [],
    device: { limits: { maxBufferSize, maxStorageBufferBindingSize } },
    buffer(floats, _extra, label) {
      const allocation = {
        size: floats * Float32Array.BYTES_PER_ELEMENT,
        label,
        destroyed: false,
        destroy() { this.destroyed = true; },
      };
      allocations.push(allocation);
      return allocation;
    },
    build() {
      builds.push(this.batchPart);
    },
  };
  return { value, allocations, builds };
}

test("batched Gemma 4 scratch grows for the split-producing E2B gate and reuses capacity", () => {
  assert.equal(mmSplits(1, GATE.N, GATE.K, 256, 16, 64), 3);
  assert.equal(E2B_BATCH_BYTES, 3 * 215 * GATE.N * Float32Array.BYTES_PER_ELEMENT);
  assert.ok(E2B_BATCH_BYTES > BASE_BYTES);

  const { value, allocations, builds } = receiver();
  assert.equal(GpuGemma4.prototype.ensureBatchPart.call(value, 215), true);
  const first = value.batchPart;
  assert.ok(first.size >= E2B_BATCH_BYTES);
  assert.equal(first.destroyed, false);
  assert.deepEqual(value.dynamic, [first]);
  assert.deepEqual(builds, [first]);
  assert.equal(allocations.length, 1);

  assert.equal(GpuGemma4.prototype.ensureBatchPart.call(value, 215), true);
  assert.equal(value.batchPart, first);
  assert.equal(allocations.length, 1);
  assert.deepEqual(builds, [first]);
});

test("larger batched suffix capacity replaces and releases the previously owned buffer", () => {
  const { value, allocations, builds } = receiver();
  assert.equal(GpuGemma4.prototype.ensureBatchPart.call(value, 215), true);
  const previous = value.batchPart;

  assert.equal(GpuGemma4.prototype.ensureBatchPart.call(value, 400), true);
  const replacement = value.batchPart;
  assert.notEqual(replacement, previous);
  assert.ok(replacement.size > previous.size);
  assert.equal(previous.destroyed, true);
  assert.equal(replacement.destroyed, false);
  assert.deepEqual(value.dynamic, [replacement]);
  assert.deepEqual(builds, [previous, replacement]);
  assert.equal(allocations.length, 2);
});

test("batch scratch over the 64 MiB policy cap is rejected without changing ownership", () => {
  const { value, allocations, builds } = receiver();
  assert.equal(GpuGemma4.prototype.ensureBatchPart.call(value, 215), true);
  const previous = value.batchPart;
  const dynamicBefore = [...value.dynamic];
  const allocationCount = allocations.length;
  const buildCount = builds.length;

  assert.equal(GpuGemma4.prototype.ensureBatchPart.call(value, 1000), false);
  assert.equal(value.batchPart, previous);
  assert.equal(previous.destroyed, false);
  assert.deepEqual(value.dynamic, dynamicBefore);
  assert.equal(allocations.length, allocationCount);
  assert.equal(builds.length, buildCount);
});

test("batch scratch over the device buffer limit is rejected without allocation", () => {
  const { value, allocations, builds } = receiver({ maxBufferSize: 16 * 1024 * 1024 });

  assert.equal(GpuGemma4.prototype.ensureBatchPart.call(value, 300), false);
  assert.equal(value.batchPart, null);
  assert.deepEqual(value.dynamic, []);
  assert.equal(allocations.length, 0);
  assert.equal(builds.length, 0);
});

function pairedReceiver(tokens, limits) {
  const value = receiver(limits);
  Object.assign(value.value, {
    fuseBatchPrefillGelu: true,
    batchSuffixes: true,
    batchMlp: { tokens, selection: new Set(["6144:1536"]) },
    batchPrefixSuffixOps: [{ ...GATE, batched: true, batchMlpFused: {} }],
    tuning: { diagnostics: { batchMlp: {} } },
  });
  return value;
}

test("paired batch MLP keeps logical splitT=1 while sizing two physical-row partial planes", () => {
  for (const tokens of [97, 241, 256]) assert.equal(gemma4BatchMlpEligible(tokens), true);
  for (const tokens of [96, 257, 97.5, NaN, "97"]) assert.equal(gemma4BatchMlpEligible(tokens), false);
  assert.equal(gemma4BatchMlpPartBytes(97, 6144, 1536), 14_303_232);
  assert.equal(gemma4BatchMlpPartBytes(241, 6144, 1536), 35_536_896);
  assert.equal(gemma4BatchMlpPartBytes(241, 12288, 1536), 16);
  const dispatches = [];
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups(...counts) { dispatches.push(counts); } };
  encodeGemma4BatchMlp(pass, {}, { N: 6144, K: 1536, batched: true }, 241);
  assert.deepEqual(dispatches, [[96, 4, 3], [5784, 1]]);
  assert.throws(() => encodeGemma4BatchMlp(pass, {}, { N: 6144, K: 1536, batched: true, sourcePolicy: true }, 241));
});

test("tall batch MLP selection is confined to its exact calibrated shape and T97..104 context", () => {
  for (const tokens of [97, 104]) assert.equal(gemma4BatchMlpTallEligible(tokens), true);
  for (const tokens of [96, 105, 112, 113, 241, 97.5, NaN, "97"]) assert.equal(gemma4BatchMlpTallEligible(tokens), false);
  const tall = { enabled: true, tokens: 97, pipeline: "tall", selection: new Set(["12288:1536"]) };
  const pipes = { matmul: "paired", reduce: "reduce", tall };
  const run = (tokens = 97, N = 12288) => {
    const calls = [], dispatches = [];
    encodeGemma4BatchMlp({ setPipeline(value) { calls.push(value); }, setBindGroup() {}, dispatchWorkgroups(...value) { dispatches.push(value); } }, pipes, { N, K: 1536, batched: true }, tokens);
    return { calls, dispatches };
  };
  assert.deepEqual(run(), { calls: ["tall"], dispatches: [[192, 1, 1]] });
  assert.deepEqual(run(112), { calls: ["paired"], dispatches: [[192, 2, 1]] });
  assert.deepEqual(run(97, 6144), { calls: ["paired", "reduce"], dispatches: [[96, 2, 3], [2328, 1]] });
  tall.enabled = false;
  assert.deepEqual(run(), { calls: ["paired"], dispatches: [[192, 2, 1]] });
  tall.enabled = true; tall.tokens = 104;
  assert.deepEqual(run(104), { calls: ["tall"], dispatches: [[192, 1, 1]] });
  tall.tokens = 113;
  assert.deepEqual(run(113), { calls: ["paired"], dispatches: [[192, 2, 1]] });
  tall.tokens = 97; tall.selection.add("6144:1536");
  assert.deepEqual(run(97, 6144), { calls: ["tall", "reduce"], dispatches: [[96, 1, 3], [2328, 1]] });
});

test("paired batch scratch replaces one-projection storage and reuses the larger capacity", async () => {
  const { value, allocations, builds } = pairedReceiver(241);
  assert.equal(GpuGemma4.prototype.ensureBatchPart.call(value, 241), true);
  const previous = value.batchPart;
  assert.equal(await GpuGemma4.prototype.ensureBatchMlpPart.call(value, 241), true);
  const paired = value.batchPart;
  assert.equal(paired.size, 35_536_896);
  assert.equal(previous.destroyed, true);
  assert.equal(value.batchMlpScratchReady, true);
  assert.deepEqual(value.dynamic, [paired]);
  assert.deepEqual(builds, [previous, paired]);
  assert.equal(await GpuGemma4.prototype.ensureBatchMlpPart.call(value, 241), true);
  assert.equal(allocations.length, 2);
  assert.equal(value.batchPart, paired);
});

test("optional paired scratch limit retains ordinary batch storage and ownership", async () => {
  const { value, allocations, builds } = pairedReceiver(241, { maxBufferSize: 24 * 1024 * 1024 });
  assert.equal(GpuGemma4.prototype.ensureBatchPart.call(value, 241), true);
  const previous = value.batchPart;
  assert.equal(await GpuGemma4.prototype.ensureBatchMlpPart.call(value, 241), false);
  assert.equal(value.batchSuffixes, true);
  assert.equal(value.batchPart, previous);
  assert.equal(previous.destroyed, false);
  assert.equal(value.batchMlpScratchReady, false);
  assert.deepEqual(value.dynamic, [previous]);
  assert.equal(allocations.length, 1);
  assert.deepEqual(builds, [previous]);
});

test("optional paired allocation failure retains ordinary batch storage", async () => {
  const { value, builds } = pairedReceiver(241);
  assert.equal(GpuGemma4.prototype.ensureBatchPart.call(value, 241), true);
  const previous = value.batchPart;
  value.buffer = () => { throw new Error("simulated paired allocation failure"); };
  assert.equal(await GpuGemma4.prototype.ensureBatchMlpPart.call(value, 241), false);
  assert.equal(value.batchPart, previous);
  assert.equal(previous.destroyed, false);
  assert.equal(value.batchSuffixes, true);
  assert.deepEqual(value.dynamic, [previous]);
  assert.deepEqual(builds, [previous]);
  assert.equal(value.tuning.diagnostics.batchMlp.scratchReason, "allocation-failed");
});

test("paired batch MLP requires its exact calibrated physical length and skips adjacent gate/up/GELU", () => {
  const run = (calibratedTokens, sourcePolicy = false, split = false) => {
    const calls = [];
    const pass = { setPipeline(value) { calls.push(value); }, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
    const pipes = { f16: true, groups: 1, splitTarget: 256, kernel: "matmul_wide", mm: { 4: "baseline-mm" }, reduce: { 4: "baseline-reduce" } };
    const value = {
      fuseBatchPrefillGelu: true, batchMlpScratchReady: true, fuseRms: false, gpuKernel: "wide",
      batchMlp: { tokens: calibratedTokens, pipelines: { matmul: "paired-mm", reduce: "paired-reduce" } },
      batchMatmul: { wide: pipes },
      encodeSourceMatmul() { calls.push("source-policy"); },
    };
    const gate = { ...GATE, dtype: "q8", batched: true, sourcePolicy,
      batchMlpFused: { N: 6144, K: 1536, batched: true } };
    const up = { ...GATE, dtype: "q8", batched: true, sourcePolicy };
    const gelu = { kind: "GELU", pipeline: "gelu", dispatch() {} };
    gate.mlpPartners = [up, gelu];
    const encoder = { beginComputePass() { return pass; } };
    for (const ops of split ? [[gate], [up, gelu]] : [[gate, up, gelu]]) GpuGemma4.prototype.encode.call(value, encoder, ops, 97, null);
    return calls;
  };
  assert.deepEqual(run(97), ["paired-mm", "paired-reduce"]);
  assert.deepEqual(run(241), ["baseline-mm", "baseline-reduce", "baseline-mm", "baseline-reduce", "gelu"]);
  assert.deepEqual(run(97, true), ["source-policy", "source-policy", "gelu"]);
  assert.deepEqual(run(97, false, true), ["baseline-mm", "baseline-reduce", "baseline-mm", "baseline-reduce", "gelu"]);
});

test("two-row kernels keep unselected, non-batched, and moved source operations on their original paths", () => {
  const run = ({ T = 2, share = true, pair = true, oldPair = true, selected = true, batched = true, sourcePolicy = false } = {}) => {
    const calls = [];
    const pass = { setPipeline(value) { calls.push(value); }, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
    const pipe = (pipeline) => ({ pipeline, columns: 4 });
    const baseline = { f16: false, groups: 1, splitTarget: 256, kernel: "matmul", mm: { 1: "mm" }, reduce: { 1: "reduce" } };
    const value = {
      shareMatvecRows: share, lastMatvecTwoDispatches: 0, fuseRms: false, useK64: false, gpuKernel: "generic",
      matvecTwo: { pipelines: { plain: pipe("two-plain"), gelu: pipe("two-gelu") }, selection: { plain: new Set(selected ? ["6144:1536"] : []) } },
      matvecGelu: { pipelines: pipe("old-gelu") }, matvec: { pipelines: pipe("old-plain"), selection: new Set(["6144:1536"]) },
      tuning: { pipelines: { generic: baseline } }, batchMatmul: { generic: baseline },
      encodeSourceMatmul() { calls.push("source"); },
    };
    const gate = { ...GATE, dtype: "q8", batched, sourcePolicy,
      fused: pair && oldPair ? { N: 6144, group: {} } : null, twoFused: pair ? { N: 6144, group: {} } : null };
    const up = { ...GATE, dtype: "q8", batched, sourcePolicy };
    const gelu = { kind: "GELU", pipeline: "gelu", dispatch() {} };
    gate.mlpPartners = [up, gelu];
    GpuGemma4.prototype.encode.call(value, { beginComputePass() { return pass; } }, [gate, up, gelu], T, null);
    return { calls, count: value.lastMatvecTwoDispatches };
  };
  assert.deepEqual(run(), { calls: ["two-gelu"], count: 1 });
  assert.deepEqual(run({ oldPair: false }), { calls: ["two-gelu"], count: 1 });
  assert.deepEqual(run({ share: false }), { calls: ["old-gelu"], count: 0 });
  assert.deepEqual(run({ T: 5 }), { calls: ["old-gelu"], count: 0 });
  assert.deepEqual(run({ pair: false }), { calls: ["two-plain", "two-plain", "gelu"], count: 2 });
  assert.deepEqual(run({ pair: false, selected: false }), { calls: ["old-plain", "old-plain", "gelu"], count: 0 });
  assert.deepEqual(run({ sourcePolicy: true }), { calls: ["source", "source", "gelu"], count: 0 });
  assert.deepEqual(run({ batched: false }), { calls: ["mm", "reduce", "mm", "reduce", "gelu"], count: 0 });
});

test("paired batch scratch excludes the moved source-policy layer", async () => {
  const { value, allocations, builds } = pairedReceiver(97);
  value.batchPrefixSuffixOps[0].sourcePolicy = true;
  assert.equal(await GpuGemma4.prototype.ensureBatchMlpPart.call(value, 97), false);
  assert.equal(value.batchMlpScratchReady, false);
  assert.equal(allocations.length, 0);
  assert.equal(builds.length, 0);
});

test("paired batch calibration reuses the exact-length context without rebuilding stable graphs", async () => {
  const context = (tokens) => ({ tokens, pipelines: {}, selection: new Set(["6144:1536"]), diagnostics: { tokens, reason: "calibrated" } });
  const first = context(97), second = context(241), builds = [];
  const value = {
    fuseBatchPrefillGelu: true, batchMlpCache: new Map([["97|6144:1536", first], ["241|6144:1536", second]]), batchMlp: first,
    batchPrefixSuffixOps: [{ kind: "mm", label: "mm.gate", N: 6144, K: 1536 }],
    batchMlpScratchReady: true, capacity: 512, p: {}, tuning: { diagnostics: {} },
    build() { builds.push(this.batchMlp); },
  };
  assert.equal(await GpuGemma4.prototype.prepareBatchMlp.call(value, 97), first);
  assert.deepEqual(builds, []);
  assert.equal(value.batchMlpScratchReady, true);
  assert.equal(await GpuGemma4.prototype.prepareBatchMlp.call(value, 241), second);
  assert.deepEqual(builds, [second]);
  assert.equal(value.batchMlpScratchReady, false);
  value.fuseBatchPrefillGelu = false;
  assert.equal(await GpuGemma4.prototype.prepareBatchMlp.call(value, 241), null);
  assert.deepEqual(builds, [second, null]);
  assert.equal(value.batchMlpCache.size, 2);
  assert.equal(value.tuning.diagnostics.batchMlp.reason, "disabled");
  assert.deepEqual(structuredClone(value.tuning.diagnostics.batchMlp), value.tuning.diagnostics.batchMlp);
});

test("paired scratch follows the active compact graph when its source MLP is omitted", async () => {
  const { value, allocations, builds } = pairedReceiver(97);
  value.sourceBatchPrefixSuffixOps = [];
  assert.equal(await GpuGemma4.prototype.ensureBatchMlpPart.call(value, 97, true), false);
  assert.equal(value.batchMlpScratchReady, false);
  assert.equal(allocations.length, 0);
  assert.equal(builds.length, 0);
});

test("compact source dispatch covers 57–64 active rows with a 56-row kernel", () => {
  const calls = [];
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups(...groups) { calls.push(groups); } };
  const pipes = { mm56: {}, reduce: { 4: {} }, f16: false, groups: 1, splitTarget: 256, kernel: "matmul" };
  const value = { sourceTokens: 97, gpuKernel: "generic", useK64: false, sourceMatmul: { generic: pipes },
    tuning: { pipelines: { generic: pipes } }, matvec: { selection: new Set() } };
  GpuGemma4.prototype.encodeSourceMatmul.call(value, pass, { N: 1536, K: 6144 }, 60);
  assert.deepEqual(calls, [[24, 2, 6], [360, 1]]);
});

test("paired batch calibration fails closed without F16 or timestamps", async () => {
  const unavailable = () => { throw new Error("GPU compilation must not run"); };
  for (const [features, reason] of [[[], "no-shader-f16"], [["shader-f16"], "no-timestamp-query"]]) {
    const result = await calibrateGemma4BatchMlp({ features: new Set(features) }, unavailable, null, null, { tokens: 97 });
    assert.equal(result.selection.size, 0);
    assert.equal(result.pipelines, null);
    assert.equal(result.diagnostics.reason, reason);
    assert.deepEqual(structuredClone(result.diagnostics), result.diagnostics);
  }
});

test("paired batch calibration skips weight shapes absent from the active graph", async () => {
  const unavailable = () => { throw new Error("GPU compilation must not run"); };
  const tensor = { info: { dtype: "q8", shape: [6144, 1536] }, buf: {}, sbuf: {} };
  const weights = { tensors: new Map([["l.0.gate", tensor], ["l.0.up", tensor]]) };
  for (const eligibleShapes of [new Set(), new Set(["12288:1536"])]) {
    const result = await calibrateGemma4BatchMlp({ features: new Set(["shader-f16", "timestamp-query"]) }, unavailable, weights, {}, {
      tokens: 97, batchPipelines: { generic: {} }, gelu: {}, eligibleShapes,
    });
    assert.equal(result.selection.size, 0);
    assert.equal(result.pipelines, null);
    assert.equal(result.diagnostics.reason, "no-eligible-shapes");
  }
});
