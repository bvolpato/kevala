import assert from "node:assert/strict";
import test from "node:test";
import { GpuGemma4 } from "../js/src/gpu-gemma4.js";
import { createGemma4PleReduce, gemma4PleReduceAdjacent, gemma4PleReduceChoice, gemma4PleReduceGroup } from "../js/src/gpu-gemma4-ple.js";

function fixture({ N = 256, K = 1536, source = false, batch = false } = {}) {
  const partner = { kind: "GELU", label: "gelu.ple", pipeline: "gelu", group: {}, dispatch(pass) { pass.dispatchWorkgroups(1); } };
  const op = { kind: "mm", label: "mm.ple.gate", N, K, dtype: "q8", sourcePolicy: source, batched: batch, group: {}, reduce: {},
    pleReduce: { N, K, partner, group: {}, retiledGroup: {} } };
  const pipes = { kernel: "matmul", f16: false, groups: 1, splitTarget: 256, mm: [null, "mm1", "mm2", "mm3", "mm4"], reduce: [null, "r1", "r2", "r3", "r4"], mm56: "mm56" };
  const variants = Object.fromEntries(["plain", "batch", "source"].map((policy) => [policy, [null, ...[1, 2, 3, 4].map((rows) => `${policy}${rows}`)]]));
  return { op, partner, pipes, fusion: { variants } };
}

test("PLE reduction requires the exact adjacent GELU and a real split projection", () => {
  const { op, partner, pipes, fusion } = fixture();
  assert.equal(gemma4PleReduceAdjacent(op, partner, 28, true), true);
  for (const [next, T, enabled] of [[undefined, 28, true], [{ ...partner }, 28, true], [partner, 8, true], [partner, 257, true], [partner, 28, false]]) {
    assert.equal(gemma4PleReduceAdjacent(op, next, T, enabled), false);
  }
  const choice = gemma4PleReduceChoice(fusion, op, partner, 28, pipes, 2, 28, "plain");
  assert.equal(choice.pipeline, "plain2");
  assert.equal(choice.group, op.pleReduce.group);
  assert.equal(gemma4PleReduceChoice(fusion, op, partner, 28, pipes, 2, 28, "plain"), choice);
  for (const [config, rows, splitTokens, policy] of [[{ ...pipes, splitTarget: 128 }, 2, 28, "plain"], [{ ...pipes, groups: 2 }, 2, 28, "plain"],
    [pipes, 0, 28, "plain"], [pipes, 2, 0, "plain"], [pipes, 2, 28, "missing"]]) {
    assert.equal(gemma4PleReduceChoice(fusion, op, partner, 28, config, rows, splitTokens, policy), null);
  }
  const unsplit = fixture({ N: 16384 });
  assert.equal(gemma4PleReduceChoice(unsplit.fusion, unsplit.op, unsplit.partner, 256, unsplit.pipes, 4, 256, "plain"), null);
});

function run({ T = 28, source = false, batch = false, enabled = true, chunks = false, retile = null, bn32 = false, original = T, missing = false, N = 256, K = 1536, k64 = false } = {}) {
  const { op, partner, pipes, fusion } = fixture({ source, batch, N, K });
  if (retile) op.prefixRetile = { ...retile, N, K, group: {}, reduce: {}, bn32: bn32 ? "bn32" : null };
  if (missing) delete fusion.variants[source ? "source" : batch ? "batch" : "plain"];
  const calls = [], pass = { setPipeline(value) { calls.push(["pipeline", value]); }, setBindGroup(_i, value) { calls.push(["group", value]); }, dispatchWorkgroups(...values) { calls.push(["dispatch", ...values]); }, end() {} };
  const value = { gpuKernel: "generic", pleReduceEnabled: enabled, pleReduce: fusion, lastPleReduceDispatches: 0, fuseRms: false,
    matvec: { selection: new Set() }, k64: { pipelines: pipes, selection: new Set(k64 ? [`${N}:${K}`] : []) }, useK64: k64,
    tuning: { pipelines: { generic: pipes } }, batchMatmul: { generic: pipes }, sourceMatmul: { generic: pipes, k64: pipes }, sourceTokens: original,
    retileTextPrefix: true, narrowPrefixColumns: bn32, lastPrefixRetileDispatches: 0, lastPrefixBn32Dispatches: 0,
    encodeSourceMatmul: GpuGemma4.prototype.encodeSourceMatmul };
  const encoder = { beginComputePass() { return pass; } };
  for (const ops of chunks ? [[op], [partner]] : [[op, partner]]) GpuGemma4.prototype.encode.call(value, encoder, ops, T, null);
  return { calls, count: value.lastPleReduceDispatches, op };
}

test("PLE fusion preserves selected MM dispatches for normal, batched, K64 and source rows", () => {
  for (const options of [{}, { T: 97, batch: true }, { T: 70, k64: true }, { T: 129 }, { T: 17, source: true, original: 241, N: 1024 },
    { T: 28, source: true, original: 117, retile: { kernel: "generic", rows: 2, row56: false, splitTokens: 64 } },
    { T: 28, source: true, original: 117, retile: { kernel: "generic", rows: 2, row56: false, splitTokens: 64 }, bn32: true }]) {
    const fused = run(options), baseline = run({ ...options, enabled: false });
    assert.equal(fused.count, 1, JSON.stringify(options));
    assert.equal(baseline.count, 0);
    const mm = (calls) => calls.slice(0, 3);
    assert.deepEqual(mm(fused.calls), mm(baseline.calls));
    assert.equal(fused.calls.filter(([kind]) => kind === "dispatch").length, 2);
    assert.equal(baseline.calls.filter(([kind]) => kind === "dispatch").length, 3);
    assert.equal(fused.calls.some(([kind, name]) => kind === "pipeline" && name === "gelu"), false);
  }
});

test("missing pipelines, chunk boundaries, tail rows and 257-token chunks retain GELU once", () => {
  for (const options of [{ missing: true }, { chunks: true }, { chunks: true, source: true }, { T: 2, batch: true }, { T: 257 }, { T: 256, N: 16384 }, { enabled: false }]) {
    const result = run(options);
    assert.equal(result.count, 0, JSON.stringify(options));
    assert.equal(result.calls.filter(([kind, name]) => kind === "pipeline" && name === "gelu").length, 1);
  }
});

test("PLE bind groups preserve prefix row-view offsets and ranges", () => {
  const globals = {}, params = {}, partials = {}, bias = {}, raw = { buffer: {}, offset: 256, size: 512 }, ple = { buffer: {}, offset: 512, size: 512 }, output = { buffer: {}, offset: 768, size: 512 };
  const resources = { globals, params, partials, bias, raw, ple, output };
  const device = { createBindGroup(value) { return value; } }, layout = {};
  const group = gemma4PleReduceGroup(device, { layout }, resources);
  assert.equal(group.layout, layout);
  assert.deepEqual(group.entries.slice(4).map((e) => e.resource), [raw, ple, output]);
  assert.equal(group.entries[0].resource.buffer, globals);
  assert.equal(gemma4PleReduceGroup(device, null, resources), null);
});

test("the effective PLE policy separates cached static-prefix numerical paths", () => {
  const selection = new Set(), value = { tuning: { selection: new Map(), pipelines: {} }, matvec: { selection }, k64: { selection }, mlp: { selection },
    pleReduceEnabled: true, pleReduce: { diagnostics: { enabled: true } } };
  const active = GpuGemma4.prototype.textPrefixPolicy.call(value, true);
  const calibration = GpuGemma4.prototype.textPrefixPolicy.call(value, true, false);
  const { pleReduce, ...otherPolicies } = JSON.parse(active);
  assert.equal(pleReduce, true);
  assert.deepEqual(JSON.parse(calibration), otherPolicies);
  value.pleReduceEnabled = false;
  const inactive = GpuGemma4.prototype.textPrefixPolicy.call(value, true);
  assert.notEqual(active, inactive);
  assert.equal(GpuGemma4.prototype.textPrefixPolicy.call(value, true, false), calibration);
  value.pleReduceEnabled = true;
  assert.equal(GpuGemma4.prototype.textPrefixPolicy.call(value, true), active);
  value.pleReduce = null;
  assert.equal(GpuGemma4.prototype.textPrefixPolicy.call(value, true), inactive);
  for (const patch of [{ fusePrefixGelu: true }, { narrowPrefixColumns: true }, { useK64: false }, { gpuKernel: "wide" }]) {
    assert.notEqual(GpuGemma4.prototype.textPrefixPolicy.call({ ...value, ...patch }, true, false), calibration);
  }
});

test("optional PLE compilation fails closed and leaves cloneable diagnostics", async (t) => {
  const old = globalThis.GPUShaderStage; globalThis.GPUShaderStage = { COMPUTE: 1 };
  t.after(() => old === undefined ? delete globalThis.GPUShaderStage : globalThis.GPUShaderStage = old);
  const disabled = await createGemma4PleReduce({}, () => { throw Error("unexpected compile"); }, { enabled: false });
  assert.equal(disabled.diagnostics.reason, "disabled");
  const device = { createBindGroupLayout: () => ({}), createPipelineLayout: () => ({}), createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [{ type: "error", lineNum: 1, linePos: 1, message: "synthetic PLE compile error" }] }) }) };
  const failed = await createGemma4PleReduce(device, () => "invalid");
  assert.equal(failed.layout, null);
  assert.deepEqual(failed.variants, {});
  assert.equal(failed.diagnostics.reason, "pipeline-failed");
  assert.match(failed.diagnostics.error, /synthetic PLE compile error/);
  assert.deepEqual(structuredClone(failed.diagnostics), failed.diagnostics);
});
