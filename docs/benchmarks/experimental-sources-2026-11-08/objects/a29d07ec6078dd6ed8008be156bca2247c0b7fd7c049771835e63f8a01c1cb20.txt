import assert from "node:assert/strict";
import test from "node:test";
import {
  calibrateMatmul,
  matmulShapes,
  matmulTokenBucket,
  matmulTuningKey,
  selectedMatmulKernel,
  selectMatmulKernel,
} from "../js/src/gpu-tuning.js";

test("GPU tuning keeps generic when optional features are unavailable", async () => {
  const generic = {};
  for (const [features, reason] of [[[], "no-shader-f16"], [["shader-f16"], "no-timestamp-query"]]) {
    const result = await calibrateMatmul({ features: new Set(features) }, null, null, { pipelines: { generic } });
    assert.equal(result.pipelines.generic, generic);
    assert.equal(result.pipelines.wide, null);
    assert.equal(result.diagnostics.reason, reason);
  }
  await assert.rejects(calibrateMatmul({ features: new Set() }, null, null, { kernel: "wide" }), /requires.*shader-f16/);
});

test("a wide pipeline failure falls back only for automatic selection", async (t) => {
  const previous = globalThis.GPUShaderStage;
  globalThis.GPUShaderStage = { COMPUTE: 4 };
  t.after(() => {
    if (previous === undefined) delete globalThis.GPUShaderStage;
    else globalThis.GPUShaderStage = previous;
  });
  const generic = {};
  let scopes = 0;
  const device = {
    features: new Set(["shader-f16", "timestamp-query"]),
    pushErrorScope() { scopes++; },
    async popErrorScope() { scopes--; return null; },
    createBindGroupLayout() { throw new Error("test pipeline failure"); },
  };
  const options = { pipelines: { generic } };
  const result = await calibrateMatmul(device, null, null, options);
  assert.equal(result.pipelines.generic, generic);
  assert.equal(result.pipelines.wide, null);
  assert.equal(result.diagnostics.reason, "wide-pipeline-failed");
  assert.equal(scopes, 0);
  await assert.rejects(calibrateMatmul(device, null, null, { ...options, kernel: "wide" }), /test pipeline failure/);
  assert.equal(scopes, 0);
});

test("GPU tuning buckets follow the runtime token thresholds", () => {
  assert.equal(matmulTokenBucket(16), "r1");
  assert.equal(matmulTokenBucket(32), "r2");
  assert.equal(matmulTokenBucket(48), "r3");
  assert.equal(matmulTokenBucket(49), "mid");
  assert.equal(matmulTokenBucket(63), "mid");
  assert.equal(matmulTokenBucket(128), "mid");
  assert.equal(matmulTokenBucket(512), "long");
  const choices = new Map([["4096:2048:r1", "wide"], ["4096:2048:r2", "generic"], ["4096:2048:r3", "wide"], ["4096:2048:mid", "wide"], ["4096:2048:long", "generic"]]);
  assert.equal(matmulTuningKey(4096, 2048, 128), "4096:2048:mid");
  assert.equal(selectedMatmulKernel(choices, 4096, 2048, 8), "wide");
  assert.equal(selectedMatmulKernel(choices, 4096, 2048, 32), "generic");
  assert.equal(selectedMatmulKernel(choices, 4096, 2048, 48), "wide");
  assert.equal(selectedMatmulKernel(choices, 4096, 2048, 63), "wide");
  assert.equal(selectedMatmulKernel(choices, 4096, 2048, 128), "wide");
  assert.equal(selectedMatmulKernel(new Map(), 4096, 2048, 512), "generic");
});

test("GPU tuning requires a stable five percent wide win", () => {
  const win = selectMatmulKernel({ generic: [100, 101, 99], wide: [93, 95, 92] });
  assert.equal(win.kernel, "wide");
  assert.equal(win.reason, "stable-wide-win");
  assert.equal(selectMatmulKernel({ generic: [100, 100, 100], wide: [96, 96, 96] }).kernel, "generic");

  const noisy = selectMatmulKernel({ generic: [100, 100, 100], wide: [94, 105, 94] });
  assert.equal(noisy.kernel, "generic");
  assert.equal(noisy.reason, "noisy");

  const generic = selectMatmulKernel({ generic: [90, 91, 89], wide: [100, 101, 99] });
  assert.equal(generic.kernel, "generic");
  assert.equal(generic.reason, "stable-generic-win");

  assert.equal(selectMatmulKernel({ generic: [1, 2], wide: [1] }).reason, "insufficient-samples");
});

test("GPU tuning collapses loaded Q8 projection shapes in priority order", () => {
  const buffer = {};
  const scales = {};
  const weights = { tensors: new Map([
    ["o", { info: { name: "L.0.o", dtype: "q8", shape: [1024, 1024] }, buf: buffer, sbuf: scales }],
    ["gate_up", { info: { name: "L.0.gate_up", dtype: "q8", shape: [4096, 1024] }, buf: buffer, sbuf: scales }],
    ["gate_up-duplicate", { info: { name: "L.1.gate_up", dtype: "q8", shape: [4096, 1024] }, buf: buffer, sbuf: scales }],
    ["qkvz", { info: { name: "L.0.qkvz", dtype: "q8", shape: [2048, 1024] }, buf: buffer, sbuf: scales }],
    ["embedding", { info: { name: "L.0.embedding", dtype: "q8", shape: [32000, 1024] }, buf: buffer, sbuf: scales }],
    ["bad", { info: { name: "L.0.bad", dtype: "q8", shape: [100, 31] }, buf: buffer, sbuf: scales }],
  ]) };
  assert.deepEqual(matmulShapes(weights).map(({ suffix, N, K }) => `${suffix}:${N}x${K}`), ["gate_up:4096x1024", "qkvz:2048x1024", "o:1024x1024"]);
});

test("GPU tuning recognizes Laya projection suffixes without changing the six-shape cap", () => {
  const buffer = {};
  const scales = {};
  const suffixes = ["wi", "wqkv", "wo2", "wo", "in_proj", "out_proj", "lin1", "lin2"];
  const tensors = new Map(suffixes.map((suffix, i) => [suffix, {
    info: { name: `enc.0.${suffix}`, dtype: "q8", shape: [1024 + i * 32, 1024] }, buf: buffer, sbuf: scales,
  }]));
  assert.deepEqual(matmulShapes({ tensors }).map(({ suffix }) => suffix), suffixes.slice(0, 6));
});
