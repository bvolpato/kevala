import assert from "node:assert/strict";
import test from "node:test";
import { calibrateGemma4PrefixGelu, comparePrefixGeluSnapshot } from "../js/src/gpu-gemma4-prefix-gelu.js";
import { GpuGemma4 } from "../js/src/gpu-gemma4.js";

const SENTINEL = 12345.25;
const PLAN = { logicalRows: 66, staticRows: 38, activeRows: 28, layerWidths: Array(15).fill(256) };
const CONFIG = { kernel: "generic", rows: 2, row56: false, splitTokens: 64 };
const tensor = (shape = [6144, 1536], dtype = "q8", block = 32) => ({ info: { shape, dtype, block }, buf: {}, sbuf: {} });

test("prefix pair dispatch preserves splits and refuses incomplete or disabled groups", () => {
  const run = ({ N = 6144, K = 1536, T = 28, enabled = true, split = false } = {}) => {
    const calls = [], pass = { setPipeline(p) { calls.push(p); }, setBindGroup() {}, dispatchWorkgroups(...groups) { calls.push(groups); }, end() {} };
    const value = { fusePrefixGelu: enabled, fuseRms: false, lastPrefixGeluDispatches: 0, encodeSourceMatmul() { calls.push("source"); } };
    const up = { kind: "mm", sourcePolicy: true }, gelu = { kind: "GELU", pipeline: "gelu", dispatch() {} };
    const gate = { kind: "mm", logicalRows: 66, sourcePolicy: true, mlpPartners: [up, gelu],
      prefixGelu: { N, K, splitTokens: 64, pipelines: { matmul: "paired", reduce: "reduce" }, group: {}, reduce: {} } };
    const encoder = { beginComputePass() { return pass; } };
    for (const ops of split ? [[gate], [up, gelu]] : [[gate, up, gelu]]) GpuGemma4.prototype.encode.call(value, encoder, ops, T, null);
    return { calls, count: value.lastPrefixGeluDispatches };
  };
  assert.deepEqual(run(), { calls: ["paired", [192, 1, 1]], count: 1 });
  assert.deepEqual(run({ N: 65, K: 1056, T: 17 }), { calls: ["paired", [3, 1, 8], "reduce", [5, 1]], count: 1 });
  for (const options of [{ enabled: false }, { split: true }]) assert.deepEqual(run(options), { calls: ["source", "source", "gelu"], count: 0 });
});
const weightsFor = (gate = tensor(), up = tensor(), layer = 0) => ({ tensors: new Map([[`l.${layer}.gate`, gate], [`l.${layer}.up`, up]]) });
const OPTIONS = { plan: PLAN, retile: { selection: new Map([["6144:1536", CONFIG]]) },
  bn32: { pipeline: {}, selection: new Set(["6144:1536"]), diagnostics: { f16: false, rows: 2, candidateBN: 32 } }, gelu: {} };
const DEVICE = { features: new Set(["timestamp-query"]) };

function snapshot(splits = 2) {
  const tokens = 2, N = 3, length = tokens * N, y = 32 * N + 4, p = splits > 1 ? splits * length : 0;
  const values = new Float32Array(4 * y + 4 * p + 12).fill(SENTINEL);
  for (let output = 0; output < 4; output++) for (let i = 0; i < length; i++) values[output * y + i] = i === 0 ? -0 : i / 16;
  const gateAt = 4 * y, upAt = gateAt + p + 4, pairAt = upAt + p + 4;
  for (let i = 0; i < p; i++) {
    values[gateAt + i] = values[pairAt + i] = i === 0 ? -0 : i / 32;
    values[upAt + i] = values[pairAt + p + i] = i === 0 ? -0 : -i / 64;
  }
  return { values, config: { tokens, N, splits }, length, y, p, gateAt, upAt, pairAt };
}

test("prefix GELU snapshot requires exact activation and both partial planes, including signed zero", () => {
  const clean = snapshot();
  assert.equal(comparePrefixGeluSnapshot(clean.values, clean.config).ok, true);
  for (const at of [3 * clean.y, clean.pairAt, clean.pairAt + clean.p]) {
    const changed = clean.values.slice(); changed[at] = 0;
    assert.equal(comparePrefixGeluSnapshot(changed, clean.config).ok, false);
  }
  for (const at of [2 * clean.y + 1, clean.gateAt + 1, clean.upAt + 1]) {
    const changed = clean.values.slice(); changed[at] = NaN;
    assert.equal(comparePrefixGeluSnapshot(changed, clean.config).ok, false);
  }
  assert.equal(comparePrefixGeluSnapshot(clean.values.subarray(1), clean.config).ok, false);
});

test("prefix GELU guards all inactive rows, partial tails, and unused unsplit partial storage", () => {
  for (const splits of [1, 2]) {
    const clean = snapshot(splits);
    assert.equal(comparePrefixGeluSnapshot(clean.values, clean.config).ok, true);
    for (const at of [clean.length, clean.y + clean.length, 3 * clean.y + clean.length, clean.gateAt + clean.p, clean.upAt + clean.p, clean.pairAt + 2 * clean.p]) {
      const changed = clean.values.slice(); changed[at] = 0;
      assert.equal(comparePrefixGeluSnapshot(changed, clean.config).ok, false);
    }
  }
});

test("prefix GELU rejects precision, suffix, and unselected BN32 contexts before compilation", async () => {
  const source = () => { throw new Error("unexpected GPU compilation"); };
  for (const override of [{ enabled: false }, { plan: { ...PLAN, activeRows: 33 } },
    { bn32: { ...OPTIONS.bn32, diagnostics: { ...OPTIONS.bn32.diagnostics, f16: true } } },
    { bn32: { ...OPTIONS.bn32, selection: new Set() } },
    { retile: { selection: new Map([["6144:1536", { ...CONFIG, rows: 4 }]]) } },
    { retile: { selection: new Map([["6144:1536", { ...CONFIG, kernel: "wide" }]]) } }]) {
    const result = await calibrateGemma4PrefixGelu(DEVICE, source, weightsFor(), { ...OPTIONS, ...override });
    assert.equal(result.selection.size, 0); assert.equal(result.pipelines, null);
  }
  const noTimestamp = await calibrateGemma4PrefixGelu({ features: new Set() }, source, weightsFor(), OPTIONS);
  assert.equal(noTimestamp.diagnostics.reason, "no-timestamp-query");
});

test("prefix GELU requires matching loaded block-32 Q8 gate/up tensors before the shared tail", async () => {
  const source = () => { throw new Error("unexpected GPU compilation"); };
  for (const weights of [weightsFor(tensor(), tensor([6144, 1568])), weightsFor(tensor(), tensor([6144, 1536], "bf16")),
    weightsFor(tensor(), tensor([6144, 1536], "q8", 64)), weightsFor(tensor(), { ...tensor(), sbuf: null }),
    weightsFor(tensor(), tensor(), 15)]) {
    const result = await calibrateGemma4PrefixGelu(DEVICE, source, weights, OPTIONS);
    assert.equal(result.diagnostics.reason, "no-eligible-pairs");
    assert.equal(result.selection.size, 0); assert.equal(result.pipelines, null);
    assert.deepEqual(structuredClone(result.diagnostics), result.diagnostics);
  }
});

test("a shader failure leaves no selected prefix fusion", async (t) => {
  const previous = globalThis.GPUShaderStage; globalThis.GPUShaderStage = { COMPUTE: 1 };
  t.after(() => previous === undefined ? delete globalThis.GPUShaderStage : globalThis.GPUShaderStage = previous);
  const device = { ...DEVICE, createBindGroupLayout: () => ({}), createPipelineLayout: () => ({}),
    createShaderModule: () => ({ async getCompilationInfo() { return { messages: [{ type: "error", lineNum: 1, linePos: 1, message: "synthetic compile error" }] }; } }) };
  const result = await calibrateGemma4PrefixGelu(device, () => "invalid", weightsFor(), OPTIONS);
  assert.equal(result.diagnostics.reason, "pipeline-failed");
  assert.equal(result.selection.size, 0); assert.equal(result.pipelines, null);
  assert.match(result.diagnostics.error, /synthetic compile error/);
});
