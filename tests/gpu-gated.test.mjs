import assert from "node:assert/strict";
import test from "node:test";

import {
  canFuse,
  calibrateGateUp,
  fusionEligibility,
  selectGateUpCandidate,
  selectionKey,
} from "../js/src/gpu-gated.js";

function q8(name, shape, { buf = {}, sbuf = {}, block = 32, dtype = "q8" } = {}) {
  return { info: { name, shape, dtype, block }, buf, sbuf };
}

test("gated contract keeps logical half shape, packed aliasing, and runtime split guards", () => {
  const separate = {
    gate: q8("enc.0.gate", [4096, 1024]),
    up: q8("enc.0.up", [4096, 1024]),
    activation: 0,
    packed: false,
  };
  assert.equal(canFuse(separate), true);
  assert.equal(canFuse(separate, { tokens: 192 }), true);
  assert.equal(canFuse(separate, { tokens: 32 }), false);
  assert.equal(fusionEligibility(separate, { tokens: 32 }).reason, "short-input");

  const packedBuffer = {};
  const packedScales = {};
  const packed = {
    gate: q8("L.0.gate_up", [8192, 1024], { buf: packedBuffer, sbuf: packedScales }),
    up: q8("L.0.gate_up", [8192, 1024], { buf: packedBuffer, sbuf: packedScales }),
    activation: 1,
    packed: true,
  };
  assert.equal(canFuse(packed, { tokens: 192 }), true);
  assert.deepEqual(fusionEligibility(packed), {
    ok: true,
    N: 4096,
    K: 1024,
    packedN: 8192,
    upOffset: 4096,
    activation: 1,
    packed: true,
    gate: packed.gate,
    up: packed.up,
  });
  assert.equal(canFuse({ ...packed, up: q8("different", [8192, 1024]) }), false);
});

test("selection keys retain activation, packing, and mid/long buckets", () => {
  const shape = { N: 4096, K: 1024, activation: 1, packed: true };
  assert.equal(selectionKey(shape, 192), "4096:1024:a1:p1:mid");
  assert.equal(selectionKey(shape, 512), "4096:1024:a1:p1:long");
  assert.notEqual(selectionKey(shape, 192), selectionKey({ ...shape, packed: false, activation: 0 }, 192));
  assert.equal(selectionKey({ N: 100, K: 31, activation: 0, packed: false }, 192), null);
});

test("paired selection is independent for eligible token buckets", () => {
  const winner = selectGateUpCandidate({
    points: [
      { tokens: 192, baseline: [10, 10.1, 9.9], fused: [9, 9.1, 8.9] },
      { tokens: 512, baseline: [20, 20.1, 19.9], fused: [18, 18.1, 17.9] },
    ],
  });
  assert.equal(winner.choice, "fused");
  assert.equal(winner.reason, "stable-win-both-points");
  assert.ok(winner.benefit > 0.09);

  const longOnlyWinner = selectGateUpCandidate({
    points: [
      { tokens: 192, eligible: false, reason: "fused-split-k" },
      { tokens: 512, baseline: [20, 20.1, 19.9], fused: [18, 18.1, 17.9] },
    ],
  });
  assert.equal(longOnlyWinner.choice, "fused");
  assert.equal(longOnlyWinner.reason, "stable-win-with-ineligible-point");
  assert.equal(longOnlyWinner.points[0].choice, "generic");
  assert.equal(longOnlyWinner.points[1].choice, "fused");

  const midOnlyWinner = selectGateUpCandidate({
    points: [
      { tokens: 192, baseline: [10, 10, 10], fused: [9, 9, 9] },
      { tokens: 512, baseline: [20, 20, 20], fused: [20.1, 20.2, 20.1] },
    ],
  });
  assert.equal(midOnlyWinner.choice, "fused");
  assert.equal(midOnlyWinner.reason, "stable-win-eligible-points");
  assert.equal(midOnlyWinner.points[0].choice, "fused");
  assert.equal(midOnlyWinner.points[1].choice, "generic");
});

test("a shape remains eligible when only the long-token bucket avoids split-K", () => {
  const descriptor = {
    gate: q8("Laya.gate", [2624, 1024]),
    up: q8("Laya.up", [2624, 1024]),
    activation: 0,
    packed: false,
  };
  assert.equal(canFuse(descriptor, { tokens: 192 }), false);
  assert.equal(fusionEligibility(descriptor, { tokens: 192 }).reason, "fused-split-k");
  assert.equal(canFuse(descriptor, { tokens: 512 }), true);
});

test("automatic calibration keeps the existing shader-f16 path", async () => {
  const generic = { splitTarget: 256 };
  const descriptor = {
    gate: q8("enc.0.gate", [4096, 1024]),
    up: q8("enc.0.up", [4096, 1024]),
    activation: 0,
    packed: false,
  };
  let compiled = false;
  const result = await calibrateGateUp(
    { features: new Set(["shader-f16", "timestamp-query"]), limits: { maxComputeWorkgroupStorageSize: 32768 } },
    () => { compiled = true; return "unused"; },
    descriptor,
    { kernel: "auto", pipelines: { generic } },
  );
  assert.equal(compiled, false);
  assert.equal(result.pipeline, null);
  assert.equal(result.pipelines.generic, generic);
  assert.equal(result.pipelines.fused, null);
  assert.equal(result.diagnostics.reason, "f16-existing-path");
  assert.equal(result.selection.get(selectionKey(descriptor, 512)), "generic");
});

test("automatic calibration falls back without timestamp queries", async () => {
  const generic = { splitTarget: 256 };
  const result = await calibrateGateUp(
    { features: new Set(), limits: { maxComputeWorkgroupStorageSize: 32768 } },
    () => { throw new Error("must not compile"); },
    null,
    { pipelines: { generic } },
  );
  assert.equal(result.pipeline, null);
  assert.equal(result.pipelines.generic, generic);
  assert.equal(result.diagnostics.reason, "no-timestamp-query");
});

test("non automatic requests never compile or measure an optional candidate", async () => {
  let called = false;
  const generic = { splitTarget: 256 };
  const result = await calibrateGateUp(
    { features: new Set(["timestamp-query"]), limits: { maxComputeWorkgroupStorageSize: 32768 } },
    () => { called = true; throw new Error("must not compile"); },
    null,
    { kernel: "generic", pipelines: { generic } },
  );
  assert.equal(called, false);
  assert.equal(result.pipeline, null);
  assert.equal(result.pipelines.generic, generic);
  assert.equal(result.diagnostics.reason, "automatic-only");
});

test("optional pipeline compilation errors return the generic bundle without a pipeline value", async (t) => {
  const previous = globalThis.GPUShaderStage;
  globalThis.GPUShaderStage = { COMPUTE: 4 };
  t.after(() => {
    if (previous === undefined) delete globalThis.GPUShaderStage;
    else globalThis.GPUShaderStage = previous;
  });
  const generic = { layout: {}, reduceLayout: {}, mm: [], reduce: [], splitTarget: 256 };
  const layouts = [];
  let scopes = 0;
  const device = {
    features: new Set(["timestamp-query"]),
    limits: { maxComputeWorkgroupStorageSize: 32768 },
    pushErrorScope() { scopes++; },
    async popErrorScope() { scopes--; return null; },
    createBindGroupLayout({ entries }) { layouts.push(entries); return { entries }; },
    createPipelineLayout({ bindGroupLayouts }) { return { bindGroupLayouts }; },
  };
  const descriptor = {
    gate: q8("enc.0.gate", [4096, 1024]),
    up: q8("enc.0.up", [4096, 1024]),
    activation: 0,
    packed: false,
  };
  const result = await calibrateGateUp(
    device,
    (kernel) => { throw new Error(`${kernel} compilation failed`); },
    descriptor,
    { pipelines: { generic } },
  );
  assert.equal(result.pipeline, null);
  assert.equal(result.pipelines.generic, generic);
  assert.equal(result.pipelines.fused, null);
  assert.equal(result.diagnostics.reason, "benchmark-failed");
  assert.equal(result.diagnostics.error, "matmul_gate_up compilation failed");
  assert.equal(scopes, 0);
  assert.deepEqual(layouts[0].map(({ binding, buffer }) => [binding, buffer.type]), [
    [0, "uniform"],
    [1, "uniform"],
    [2, "read-only-storage"],
    [3, "read-only-storage"],
    [4, "read-only-storage"],
    [5, "read-only-storage"],
    [6, "read-only-storage"],
    [7, "storage"],
  ]);
  assert.equal("fallbackPipeline" in result, false);
});
