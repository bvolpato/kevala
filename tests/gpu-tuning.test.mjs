import assert from "node:assert/strict";
import test from "node:test";
import { encodeMatmul, mmSplits, rowsPerThread } from "../js/src/gpu.js";
import {
  calibrateMatmul,
  matmulShapes,
  matmulTokenBucket,
  matmulTuningKey,
  selectedMatmulKernel,
  selectMatmulKernel,
} from "../js/src/gpu-tuning.js";

test("FP32 row tiles reduce padding while preserving the grid and split count", () => {
  const config = { f16: false, groups: 1 };
  for (const tokens of [65, 80, 96, 129, 140, 144]) {
    const rows = rowsPerThread(tokens, config);
    assert.equal(rows, 3);
    assert.equal(Math.ceil(tokens / (16 * rows)), Math.ceil(tokens / 64));
    for (const [N, K] of [[70, 160], [1024, 2624], [12288, 2048]]) {
      assert.equal(mmSplits(tokens, N, K, 256, 16 * rows, 64), mmSplits(tokens, N, K, 256, 64, 64));
    }
    assert.equal(rowsPerThread(tokens, { f16: true, groups: 1 }), 4);
    assert.equal(rowsPerThread(tokens, { f16: false, groups: 2 }), 4);
    assert.equal(rowsPerThread(tokens), 4);
  }
  for (const [tokens, rows] of [[1, 1], [16, 1], [17, 2], [32, 2], [33, 3], [48, 3], [49, 4], [64, 4], [97, 4], [128, 4], [145, 4], [512, 4]]) {
    assert.equal(rowsPerThread(tokens, config), rows);
  }
});

test("GPU tuning keeps generic when optional features are unavailable", async () => {
  const generic = {};
  for (const features of [[], ["shader-f16"]]) {
    const reason = "no-timestamp-query";
    const result = await calibrateMatmul({ features: new Set(features) }, null, null, { pipelines: { generic } });
    assert.equal(result.pipelines.generic, generic);
    assert.equal(result.pipelines.wide, null);
    assert.equal(result.diagnostics.reason, reason);
  }
  await assert.rejects(calibrateMatmul({ features: new Set() }, null, null, { kernel: "wide" }), /requires.*shader-f16/);
});

function genericPipelines() {
  const mm = [];
  const reduce = [];
  mm[4] = { label: "generic-matmul", splitTarget: 256 };
  reduce[4] = { label: "generic-reduce", splitTarget: 256 };
  return { layout: {}, reduceLayout: {}, mm, reduce, splitTarget: 256, groups: 1 };
}

function mockCalibrationDevice({ failPipeline = false, midWins = false } = {}) {
  const device = {
    features: new Set(["timestamp-query"]),
    limits: { maxStorageBufferBindingSize: 1 << 30, maxBufferSize: 1 << 30 },
    compiled: [],
    pipelineLayouts: [],
    pushErrorScope() {},
    async popErrorScope() { return null; },
    createBindGroupLayout() { return {}; },
    createPipelineLayout({ bindGroupLayouts }) {
      const layout = { bindGroupLayouts };
      device.pipelineLayouts.push(layout);
      return layout;
    },
    createShaderModule({ code }) { return { code, async getCompilationInfo() { return { messages: [] }; } }; },
    async createComputePipelineAsync({ compute, label }) {
      if (failPipeline) throw new Error("mock split pipeline compile failure");
      const config = JSON.parse(compute.module.code).config;
      const pipeline = { label, ...config };
      device.compiled.push(pipeline);
      return pipeline;
    },
    createBindGroup() { return {}; },
    createQuerySet({ count }) { return { values: new BigInt64Array(count), destroy() {} }; },
    createBuffer({ size }) {
      return {
        data: new ArrayBuffer(size),
        destroy() {},
        async mapAsync() {},
        getMappedRange() { return this.data; },
        unmap() {},
      };
    },
    queue: {
      writeBuffer() {},
      submit() {},
    },
    createCommandEncoder() {
      return {
        beginComputePass(options = {}) {
          const state = { firstPipeline: null, rows: 0, wroteDispatch: false };
          return {
            setPipeline(pipeline) { state.firstPipeline ||= pipeline; },
            setBindGroup() {},
            dispatchWorkgroups(_x, rows) {
              if (!state.wroteDispatch) {
                state.rows = rows;
                state.wroteDispatch = true;
              }
            },
            end() {
              const writes = options.timestampWrites;
              if (!writes) return;
              const split = state.firstPipeline.splitTarget === 512;
              const durationMs = split && state.rows === 4 && !midWins ? 1.1 : split ? 0.9 : 1;
              const start = BigInt(writes.beginningOfPassWriteIndex) * 10_000_000n;
              const duration = BigInt(Math.round(durationMs * 4 * 1_000_000));
              writes.querySet.values[writes.beginningOfPassWriteIndex] = start;
              writes.querySet.values[writes.endOfPassWriteIndex] = start + duration;
            },
          };
        },
        resolveQuerySet(querySet, first, count, destination, offset) {
          const bytes = new Uint8Array(querySet.values.buffer, first * 8, count * 8);
          new Uint8Array(destination.data, offset, count * 8).set(bytes);
        },
        copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
          new Uint8Array(destination.data, destinationOffset, size).set(new Uint8Array(source.data, sourceOffset, size));
        },
        finish() { return {}; },
      };
    },
  };
  return device;
}

test("FP32 auto selection compares split target 512 per point and protects the shared mid bucket", async (t) => {
  const prior = {
    shaderStage: globalThis.GPUShaderStage,
    bufferUsage: globalThis.GPUBufferUsage,
    mapMode: globalThis.GPUMapMode,
  };
  globalThis.GPUShaderStage = { COMPUTE: 4 };
  globalThis.GPUBufferUsage = { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, QUERY_RESOLVE: 16, MAP_READ: 32 };
  globalThis.GPUMapMode = { READ: 1 };
  t.after(() => {
    if (prior.shaderStage === undefined) delete globalThis.GPUShaderStage;
    else globalThis.GPUShaderStage = prior.shaderStage;
    if (prior.bufferUsage === undefined) delete globalThis.GPUBufferUsage;
    else globalThis.GPUBufferUsage = prior.bufferUsage;
    if (prior.mapMode === undefined) delete globalThis.GPUMapMode;
    else globalThis.GPUMapMode = prior.mapMode;
  });
  const generic = genericPipelines();
  const device = mockCalibrationDevice();
  const weights = { tensors: new Map([
    ["wi", { info: { name: "enc.0.wi", dtype: "q8", shape: [1536, 4096] }, buf: {}, sbuf: {} }],
    ["wqkv", { info: { name: "enc.0.wqkv", dtype: "q8", shape: [1024, 2624] }, buf: {}, sbuf: {} }],
  ]) };
  const wgsl = (kernel, config) => JSON.stringify({ kernel, config });
  const result = await calibrateMatmul(device, wgsl, weights, { pipelines: { generic } });

  assert.equal(result.pipelines.generic, generic);
  assert.equal(result.pipelines.wide, null);
  assert.equal(result.pipelines.split.splitTarget, 512);
  assert.equal(result.pipelines.split.layout, generic.layout);
  assert.equal(result.pipelines.split.reduceLayout, generic.reduceLayout);
  assert.deepEqual(device.pipelineLayouts.map(({ bindGroupLayouts }) => bindGroupLayouts), [[generic.layout], [generic.reduceLayout]]);
  assert.deepEqual(device.compiled.map(({ label, f16, groups, splitTarget, rows }) => ({ label, f16, groups, splitTarget, rows })), [
    { label: "matmul_r3", f16: false, groups: 1, splitTarget: 512, rows: 3 },
    { label: "reduce_r3", f16: false, groups: 1, splitTarget: 512, rows: 3 },
    { label: "matmul_r4", f16: false, groups: 1, splitTarget: 512, rows: 4 },
    { label: "reduce_r4", f16: false, groups: 1, splitTarget: 512, rows: 4 },
  ]);

  const points = result.diagnostics.points[0].points;
  assert.deepEqual(points.map(({ tokens, genericSplits, splitSplits }) => [tokens, genericSplits, splitSplits]), [
    [128, 6, 8], [200, 3, 6], [512, 1, 3],
  ]);
  assert.equal(points[0].kernel, "generic");
  assert.equal(points[0].measuredKernel, "split");
  assert.equal(points[0].bucketReason, "mixed-split-results");
  assert.equal(points[1].kernel, "generic");
  assert.equal(result.selection.get(matmulTuningKey(1536, 4096, 128)), "generic");
  assert.equal(result.selection.get(matmulTuningKey(1536, 4096, 200)), "generic");
  assert.equal(result.selection.get(matmulTuningKey(1536, 4096, 512)), "split");
  const laya = result.diagnostics.points[1].points;
  assert.deepEqual(laya.map(({ tokens, reason }) => [tokens, reason]), [
    [128, "same-split-count"], [200, "stable-generic-win"], [512, "stable-wide-win"],
  ]);
  assert.equal(laya[0].genericSplits, laya[0].splitSplits);
  assert.equal(result.selection.get(matmulTuningKey(1024, 2624, 200)), "generic");
  assert.equal(result.selection.get(matmulTuningKey(1024, 2624, 512)), "split");

  const winning = await calibrateMatmul(mockCalibrationDevice({ midWins: true }), wgsl, weights, { pipelines: { generic: genericPipelines() } });
  const selected = selectedMatmulKernel(winning.selection, 1536, 4096, 140);
  assert.equal(selected, "split");
  assert.equal(rowsPerThread(140, winning.pipelines[selected]), 3);
  const pass = { pipelines: [], setPipeline(pipeline) { this.pipelines.push(pipeline); }, setBindGroup() {}, dispatchWorkgroups() {} };
  encodeMatmul(pass, winning.pipelines[selected], { N: 1536, K: 4096, group: {}, reduce: {} }, 140);
  assert.equal(pass.pipelines[0], winning.pipelines.split.mm[3]);
  assert.ok(pass.pipelines.includes(winning.pipelines.split.reduce[3]));
});

test("FP32 auto selection keeps generic after split pipeline compilation failure", async (t) => {
  const previous = globalThis.GPUShaderStage;
  globalThis.GPUShaderStage = { COMPUTE: 4 };
  t.after(() => {
    if (previous === undefined) delete globalThis.GPUShaderStage;
    else globalThis.GPUShaderStage = previous;
  });
  const generic = genericPipelines();
  const weights = { tensors: new Map([["wi", {
    info: { name: "enc.0.wi", dtype: "q8", shape: [1536, 4096] }, buf: {}, sbuf: {},
  }]]) };
  const result = await calibrateMatmul(mockCalibrationDevice({ failPipeline: true }), null, weights, { pipelines: { generic } });
  assert.equal(result.pipelines.generic, generic);
  assert.equal(result.pipelines.wide, null);
  assert.equal(result.pipelines.split, null);
  assert.equal(result.diagnostics.reason, "split-pipeline-failed");
  assert.equal(result.selection.get(matmulTuningKey(1536, 4096, 128)), "generic");
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
