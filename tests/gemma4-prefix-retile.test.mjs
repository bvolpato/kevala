import assert from "node:assert/strict";
import test from "node:test";
import { mmSplits } from "../js/src/gpu.js";
import { GpuGemma4 } from "../js/src/gpu-gemma4.js";
import { calibrateGemma4PrefixRetile, compareGemma4PrefixRetileOutputs, gemma4PrefixRetileConfig } from "../js/src/gpu-gemma4-prefix-retile.js";

const pipes = (f16 = false, kernel = "matmul") => ({ f16, kernel, groups: 1, splitTarget: 256, mm: [null, {}, {}, {}, {}], reduce: [null, {}, {}, {}, {}], mm56: !f16 && kernel === "matmul" ? {} : null });
const plan = (logicalRows, staticRows = 38) => ({ logicalRows, staticRows, activeRows: logicalRows - staticRows });

test("Fast R2 retile preserves the original R3 split partition", () => {
  const config = gemma4PrefixRetileConfig(plan(66), pipes());
  assert.deepEqual(config, { kernel: "generic", rows: 2, row56: false, splitTokens: 64 });
  assert.equal(mmSplits(66, 1536, 6144, 256, 48, 64), 6);
  assert.equal(mmSplits(66, 1536, 6144, 256, 32, 64), 4);
  for (const [N, K] of [[6144, 1536], [1536, 6144], [2048, 1536], [1536, 2048], [256, 1536]]) {
    assert.equal(mmSplits(config.splitTokens, N, K, 256, 32, 64), mmSplits(66, N, K, 256, 48, 64));
  }
});

test("Balanced selects BM56 for generic F32 and preserves F16 kernel policy", () => {
  assert.deepEqual(gemma4PrefixRetileConfig(plan(89), pipes()), { kernel: "generic", rows: 4, row56: true, splitTokens: 128 });
  assert.deepEqual(gemma4PrefixRetileConfig(plan(89), pipes(true)), { kernel: "generic", rows: 4, row56: false, splitTokens: 128 });
  assert.equal(gemma4PrefixRetileConfig(plan(89), pipes(true, "matmul_wide"), "wide"), null);
  assert.deepEqual(gemma4PrefixRetileConfig(plan(66), pipes(true, "matmul_wide"), "wide"), { kernel: "wide", rows: 2, row56: false, splitTokens: 64 });
});

test("retile rejects invalid plans, missing pipelines, and increased padding", () => {
  for (const invalid of [null, plan(257), plan(120), { ...plan(89), activeRows: 50 }, plan(89, 0)]) assert.equal(gemma4PrefixRetileConfig(invalid, pipes()), null);
  assert.equal(gemma4PrefixRetileConfig(plan(80), pipes()), null);
  assert.equal(gemma4PrefixRetileConfig(plan(66), { ...pipes(), mm: [] }), null);
  assert.equal(gemma4PrefixRetileConfig(plan(66), { ...pipes(), groups: 2 }), null);
  assert.equal(gemma4PrefixRetileConfig(plan(66), pipes(), "k64"), null);
});

test("retile parity rejects signed zero differences and nonfinite output", () => {
  assert.equal(compareGemma4PrefixRetileOutputs(new Float32Array([1, -0]), new Float32Array([1, -0])).ok, true);
  assert.equal(compareGemma4PrefixRetileOutputs(new Float32Array([0]), new Float32Array([-0])).ok, false);
  assert.equal(compareGemma4PrefixRetileOutputs(new Float32Array([NaN]), new Float32Array([NaN])).ok, false);
  assert.equal(compareGemma4PrefixRetileOutputs(new Float32Array([1]), new Float32Array([1, 2])).ok, false);
});

test("disabled and unsupported calibration returns a cloneable fallback without GPU allocations", async () => {
  const device = { features: new Set(), createBuffer() { throw new Error("unexpected allocation"); } };
  const disabled = await calibrateGemma4PrefixRetile(device, null, null, { enabled: false });
  assert.equal(disabled.diagnostics.reason, "disabled");
  assert.equal(disabled.selection.size, 0);
  const unsupported = await calibrateGemma4PrefixRetile(device, null, null, { plan: plan(66) });
  assert.equal(unsupported.diagnostics.reason, "no-timestamp-query");
  assert.deepEqual(structuredClone(unsupported.diagnostics), unsupported.diagnostics);
});

test("compact prefix dispatch uses the calibrated tile and original split count", () => {
  const calls = [];
  const pass = { setPipeline(pipeline) { calls.push(pipeline); }, setBindGroup() {}, dispatchWorkgroups(...groups) { calls.push(groups); } };
  const config = { ...gemma4PrefixRetileConfig(plan(66), pipes()), N: 1536, K: 6144, group: {}, reduce: {} };
  const source = { mm: { 2: "r2" }, reduce: { 2: "reduce-r2" }, groups: 1, splitTarget: 256 };
  const value = { sourceTokens: 66, retileTextPrefix: true, lastPrefixRetileDispatches: 0, sourceMatmul: { generic: source } };
  GpuGemma4.prototype.encodeSourceMatmul.call(value, pass, { logicalRows: 66, prefixRetile: config }, 28);
  assert.deepEqual(calls, ["r2", [24, 1, 6], "reduce-r2", [168, 1]]);
  assert.equal(value.lastPrefixRetileDispatches, 1);
  calls.length = 0;
  config.bn32 = "bn32";
  value.narrowPrefixColumns = true;
  value.lastPrefixBn32Dispatches = 0;
  GpuGemma4.prototype.encodeSourceMatmul.call(value, pass, { logicalRows: 66, prefixRetile: config }, 28);
  assert.deepEqual(calls, ["bn32", [48, 1, 6], "reduce-r2", [168, 1]]);
  assert.equal(value.lastPrefixBn32Dispatches, 1);
  assert.equal(value.lastPrefixRetileDispatches, 2);
});

test("prefix retile context changes release only the dynamic graph and reuse stable calibration", async () => {
  const first = { key: "first", selection: new Map([["1536:6144", {}]]), diagnostics: { reason: "calibrated" } };
  const second = { key: "second", selection: new Map(), diagnostics: { reason: "no-stable-win" } };
  let clears = 0;
  const value = { retileTextPrefix: true, prefixRetile: first, prefixRetileCache: new Map([["first|bn32:false|gelu:false", first], ["second|bn32:false|gelu:false", second]]),
    tuning: { diagnostics: {} }, clearTextPrefixGraph() { clears++; } };
  await GpuGemma4.prototype.preparePrefixRetile.call(value, { key: "first", activeRows: 28 });
  assert.equal(clears, 0);
  await GpuGemma4.prototype.preparePrefixRetile.call(value, { key: "second", activeRows: 51 });
  assert.equal(clears, 1);
  assert.equal(value.prefixRetile, second);
  value.retileTextPrefix = false;
  await GpuGemma4.prototype.preparePrefixRetile.call(value, { key: "second", activeRows: 51 });
  assert.equal(clears, 2);
  assert.equal(value.prefixRetile, null);
  assert.equal(value.prefixRetileCache.size, 2);
  assert.equal(value.tuning.diagnostics.prefixRetile.reason, "disabled");
  assert.deepEqual(structuredClone(value.tuning.diagnostics.prefixRetile), value.tuning.diagnostics.prefixRetile);
});

test("prefix column policy invalidates a cached dynamic graph", async () => {
  const wide = { key: "same", selection: new Map(), diagnostics: { reason: "wide" } };
  const narrow = { key: "same", selection: new Map(), diagnostics: { reason: "narrow" } };
  let clears = 0;
  const value = { retileTextPrefix: true, narrowPrefixColumns: false, prefixRetile: wide,
    prefixRetileCache: new Map([["same|bn32:false|gelu:false", wide], ["same|bn32:true|gelu:false", narrow]]),
    tuning: { diagnostics: {} }, clearTextPrefixGraph() { clears++; } };
  const plan = { key: "same", activeRows: 28 };
  await GpuGemma4.prototype.preparePrefixRetile.call(value, plan);
  assert.equal(clears, 0);
  value.narrowPrefixColumns = true;
  await GpuGemma4.prototype.preparePrefixRetile.call(value, plan);
  assert.equal(clears, 1);
  assert.equal(value.prefixRetile, narrow);
});

test("PLE toggles preserve matrix selections and rebind only the active numerical graph key", async () => {
  const calibrated = { key: "ple-off", selection: new Map([["256:1536", { rows: 2 }]]),
    bn32: { selection: new Set(["256:1536"]), pipeline: {} }, prefixGelu: { selection: new Set(["6144:1536"]), pipelines: {} }, diagnostics: { reason: "calibrated" } };
  const other = { key: "other-rms", selection: new Map(), diagnostics: { reason: "other-policy" } };
  let clears = 0;
  const value = { retileTextPrefix: true, prefixRetile: calibrated, prefixRetileCache: new Map([["stable|bn32:false|gelu:false", calibrated], ["other|bn32:false|gelu:false", other]]),
    tuning: { diagnostics: {} }, clearTextPrefixGraph() { clears++; } };
  const context = (key, calibrationKey = "stable") => ({ key, calibrationKey, activeRows: 28 });
  await GpuGemma4.prototype.preparePrefixRetile.call(value, context("ple-off"));
  assert.equal(value.prefixRetile, calibrated);
  assert.equal(clears, 0);
  await GpuGemma4.prototype.preparePrefixRetile.call(value, context("ple-on"));
  const on = value.prefixRetile;
  assert.equal(on.key, "ple-on");
  assert.equal(on.selection, calibrated.selection);
  assert.equal(on.bn32, calibrated.bn32);
  assert.equal(on.prefixGelu, calibrated.prefixGelu);
  assert.equal(on.diagnostics, calibrated.diagnostics);
  assert.equal(clears, 1);
  await GpuGemma4.prototype.preparePrefixRetile.call(value, context("ple-on"));
  assert.equal(value.prefixRetile, on);
  assert.equal(clears, 1);
  await GpuGemma4.prototype.preparePrefixRetile.call(value, context("ple-off"));
  assert.equal(value.prefixRetile.key, "ple-off");
  assert.equal(value.prefixRetile.selection, calibrated.selection);
  assert.equal(value.prefixRetileCache.size, 2);
  assert.equal(clears, 2);
  await GpuGemma4.prototype.preparePrefixRetile.call(value, context("other-rms", "other"));
  assert.equal(value.prefixRetile, other);
  assert.equal(clears, 3);
});
