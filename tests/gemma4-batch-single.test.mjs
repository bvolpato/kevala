import assert from "node:assert/strict";
import test from "node:test";
import { calibrateGemma4BatchSingle, encodeGemma4BatchSingle, gemma4BatchSingleEligible, gemma4BatchSingleMemory } from "../js/src/gpu-gemma4-batch-single.js";
import { GpuGemma4 } from "../js/src/gpu-gemma4.js";
import { mmSplits } from "../js/src/gpu.js";

const q = { kind: "mm", label: "mm.q", N: 2048, K: 1536, dtype: "q8", batched: true, sourcePolicy: false, mode: 0, bias: 0, group: {}, reduce: {} };
const down = { ...q, label: "mm.down", N: 1536, K: 6144 };
const baseline = { f16: true, groups: 1, splitTarget: 256, kernel: "matmul", layout: {}, reduceLayout: {}, mm: { 4: "old-mm" }, reduce: { 4: "old-reduce" } };

test("single tall suffix eligibility excludes other row counts, modes and projections", () => {
  for (const T of [97, 100, 104]) for (const op of [q, down]) assert.equal(gemma4BatchSingleEligible(T, op), true);
  for (const T of [1, 2, 96, 105, 112, 241, NaN]) assert.equal(gemma4BatchSingleEligible(T, q), false);
  for (const change of [{ batched: false }, { sourcePolicy: true }, { dtype: "bf16" }, { mode: 1 }, { bias: 1 }, { label: "mm.ple.proj" }, { N: 1536 }, { K: 6144 }]) assert.equal(gemma4BatchSingleEligible(97, { ...q, ...change }), false);
});

test("single tall dispatch preserves original split-K partition and R4 reducer", () => {
  for (const op of [q, down]) {
    const calls = [];
    const pass = { setPipeline(value) { calls.push(value); }, setBindGroup() {}, dispatchWorkgroups(...groups) { calls.push(groups); } };
    encodeGemma4BatchSingle(pass, { pipeline: "tall" }, baseline, op, 97);
    assert.deepEqual(calls, ["tall", [Math.ceil(op.N / 64), 1, mmSplits(1, op.N, op.K, 256, 64, 64)], "old-reduce", [Math.ceil(97 * op.N / 256), 1]]);
    assert.throws(() => encodeGemma4BatchSingle(pass, {}, { ...baseline, f16: false }, op, 97), /F16 R4/);
    assert.throws(() => encodeGemma4BatchSingle(pass, {}, { ...baseline, groups: 2 }, op, 97), /F16 R4/);
  }
});

test("calibration accounts for separate output and partial planes, padding, host seed and readback", () => {
  for (const T of [97, 104]) for (const { N, K } of [q, down]) {
    const bytes = gemma4BatchSingleMemory(T, N, K);
    assert.equal(bytes.x, 128 * K * 4);
    assert.equal(bytes.y, 128 * N * 4 + 16);
    assert.equal(bytes.part, mmSplits(1, N, K, 256, 64, 64) * T * N * 4 + 16);
    assert.equal(bytes.gpuBytes, bytes.x + bytes.y * 2 + bytes.part * 2 + 624);
    assert.ok(bytes.hostBytes >= 2 * T * K * 4 + 96);
    assert.equal(bytes.totalBytes, bytes.gpuBytes + bytes.hostBytes);
    assert.ok(bytes.totalBytes < 64 * 1024 * 1024);
  }
  assert.equal(gemma4BatchSingleMemory(105, 2048, 1536), null);
  assert.equal(gemma4BatchSingleMemory(97, 6144, 1536), null);
});

test("optional calibration rejects unsupported contexts before touching GPU compilation", async () => {
  const unavailable = () => { throw new Error("GPU must not run"); };
  const features = new Set(["shader-f16", "timestamp-query"]);
  const tensor = { info: { dtype: "q8", shape: [q.N, q.K] }, buf: {}, sbuf: {} };
  const weights = { tensors: new Map([["l.0.q", tensor]]) };
  for (const [device, options, reason] of [
    [{ features }, { kernel: "generic" }, "kernel-override"],
    [{ features }, { kernel: "wide" }, "kernel-override"],
    [{ features }, { tokens: 105 }, "unsupported-physical-tokens"],
    [{ features: new Set() }, {}, "no-shader-f16"],
    [{ features: new Set(["shader-f16"]) }, {}, "no-timestamp-query"],
    [{ features }, { eligibleShapes: new Set() }, "no-eligible-shapes"],
    [{ features }, { batchPipelines: { generic: { ...baseline, groups: 2 } } }, "no-eligible-shapes"],
    [{ features }, { batchPipelines: { generic: { ...baseline, f16: false } } }, "no-eligible-shapes"],
  ]) {
    const result = await calibrateGemma4BatchSingle(device, unavailable, weights, { selection: new Map() }, { tokens: 97, batchPipelines: { generic: baseline }, ...options });
    assert.equal(result.selection.size, 0);
    assert.equal(result.diagnostics.reason, reason);
    assert.deepEqual(structuredClone(result.diagnostics), result.diagnostics);
  }
});

test("optional pipeline failure returns a cloneable baseline result", async () => {
  const device = { features: new Set(["shader-f16", "timestamp-query"]), pushErrorScope() {}, async popErrorScope() { return null; } };
  const tensor = { info: { dtype: "q8", shape: [q.N, q.K] }, buf: {}, sbuf: {} };
  const result = await calibrateGemma4BatchSingle(device, () => { throw new Error("pipeline unavailable"); }, { tensors: new Map([["l.0.q", tensor]]) }, { selection: new Map() }, { tokens: 97, batchPipelines: { generic: baseline } });
  assert.equal(result.selection.size, 0);
  assert.equal(result.diagnostics.reason, "pipeline-failed");
  assert.match(result.diagnostics.error, /pipeline unavailable/);
  assert.deepEqual(structuredClone(result.diagnostics), result.diagnostics);
});

test("dispatch toggle keeps the calibrated context and all baseline fallbacks", () => {
  const run = (change = {}, opChange = {}, T = 97) => {
    const calls = [];
    const pass = { setPipeline(value) { calls.push(value); }, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
    const context = { tokens: 97, pipelines: { pipeline: "tall" }, selection: new Set(["2048:1536", "1536:6144"]) };
    const value = { batchSingleEnabled: true, batchSingle: context, lastBatchSingleDispatches: 0, gpuKernel: "auto", fuseRms: false,
      tuning: { selection: new Map(), pipelines: { generic: baseline } }, batchMatmul: { generic: baseline },
      encodeSourceMatmul() { calls.push("source"); }, ...change };
    GpuGemma4.prototype.encode.call(value, { beginComputePass() { return pass; } }, [{ ...q, ...opChange }], T, null);
    return { calls, count: value.lastBatchSingleDispatches, context: value.batchSingle };
  };
  assert.deepEqual(run().calls, ["tall", "old-reduce"]);
  assert.equal(run().count, 1);
  for (const change of [{ batchSingleEnabled: false }, { gpuKernel: "generic" }, { batchSingle: { tokens: 98, selection: new Set(["2048:1536"]) } }]) assert.deepEqual(run(change).calls, ["old-mm", "old-reduce"]);
  for (const opChange of [{ mode: 1 }, { bias: 1 }, { dtype: "bf16" }, { label: "mm.ple.proj" }]) assert.deepEqual(run({}, opChange).calls, ["old-mm", "old-reduce"]);
  assert.deepEqual(run({}, {}, 105).calls, ["old-mm", "old-reduce"]);
  assert.deepEqual(run({}, { sourcePolicy: true }).calls, ["source"]);
});

test("off/on/off reuses calibrated state without rebuilding or changing prefix policy", async () => {
  const context = { tokens: 97, pipelines: {}, selection: new Set(["2048:1536"]), diagnostics: { tokens: 97, reason: "calibrated" } };
  const key = JSON.stringify([97, "auto", [["2048:1536", "generic", true, 1, 256]]]);
  const prefix = {};
  const value = { batchSingleEnabled: false, batchSingleCache: new Map([[key, context]]), batchPrefixSuffixOps: [q], batchMatmul: { generic: baseline },
    gpuKernel: "auto", tuning: { selection: new Map(), diagnostics: {} }, prefixRetile: prefix };
  for (const enabled of [false, true, false]) {
    value.batchSingleEnabled = enabled;
    assert.equal(await GpuGemma4.prototype.prepareBatchSingle.call(value, 97), context);
    assert.equal(value.batchSingle, context);
    assert.equal(value.prefixRetile, prefix);
    assert.equal(value.tuning.diagnostics.batchSingle.enabled, enabled);
    assert.deepEqual(structuredClone(value.tuning.diagnostics.batchSingle), value.tuning.diagnostics.batchSingle);
  }
});

test("new token contexts are bounded and retain the most recent calibrated fallback", async () => {
  const context = (tokens) => ({ tokens, selection: new Set(["2048:1536"]), diagnostics: { tokens, reason: "calibrated" } });
  const key = (tokens) => JSON.stringify([tokens, "auto", [["2048:1536", "generic", true, 1, 256]]]);
  const first = context(97), second = context(98);
  const value = { batchSingleEnabled: false, batchSingleCache: new Map([[key(97), first], [key(98), second]]), batchPrefixSuffixOps: [q], batchMatmul: { generic: baseline },
    gpuKernel: "auto", tuning: { selection: new Map(), diagnostics: {} }, device: { features: new Set() } };
  const fallback = await GpuGemma4.prototype.prepareBatchSingle.call(value, 99);
  assert.equal(fallback.selection.size, 0);
  assert.equal(value.batchSingleCache.size, 2);
  assert.equal(value.batchSingleCache.has(key(97)), false);
  assert.equal(await GpuGemma4.prototype.prepareBatchSingle.call(value, 98), second);
});
