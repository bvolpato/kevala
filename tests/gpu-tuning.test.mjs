import assert from "node:assert/strict";
import test from "node:test";
import { encodeMatmul, matmulPipelines, mmSplits, rowsPerThread } from "../js/src/gpu.js";
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

test("BM56 keeps the R3 preference and the virtual R4 dispatch and reducer", () => {
  const pipes = { f16: false, groups: 1, splitTarget: 256, mm: [null, "r1", "r2", "r3", "r4"], mm56: "r56", reduce: [null, "reduce1", "reduce2", "reduce3", "reduce4"] };
  const op = { N: 2048, K: 6144, group: {}, reduce: {} };
  const run = (T, config = pipes) => {
    const pipelines = [], grids = [], groups = [];
    encodeMatmul({
      setPipeline(p) { pipelines.push(p); },
      setBindGroup(i, group) { groups.push([i, group]); },
      dispatchWorkgroups(...grid) { grids.push(grid); },
    }, config, op, T);
    return { pipelines, grids, groups };
  };
  for (const T of [49, 53, 55, 56, 97, 99, 105, 111, 112, 145, 168, 193, 198, 224, 392]) {
    const baseline = run(T, { ...pipes, mm56: null });
    const candidate = run(T);
    assert.equal(candidate.pipelines[0], "r56", `T=${T}`);
    assert.deepEqual(candidate.pipelines.slice(1), baseline.pipelines.slice(1));
    assert.deepEqual(candidate.grids, baseline.grids);
    assert.deepEqual(candidate.groups, baseline.groups);
  }
  for (const T of [65, 80, 96, 129, 140, 144]) assert.equal(run(T).pipelines[0], "r3");
  for (const T of [57, 64, 113, 128, 169, 192, 225, 393, 512, 601]) assert.equal(run(T).pipelines[0], "r4");
  for (const [T, p] of [[16, "r1"], [31, "r2"], [47, "r3"]]) assert.equal(run(T).pipelines[0], p);
  for (const config of [{ ...pipes, f16: true }, { ...pipes, groups: 2 }, { ...pipes, mm56: undefined }]) {
    assert.equal(run(105, config).pipelines[0], "r4");
  }
  assert.deepEqual(run(105).grids[0], [32, 2, 4]);
  assert.deepEqual(run(198).grids[0], [32, 4, 2]);
  assert.deepEqual(run(198).pipelines, ["r56", "reduce4"]);
});

test("BM56 is built only for generic FP32 and shares its matrix layout", async (t) => {
  const previous = globalThis.GPUShaderStage;
  globalThis.GPUShaderStage = { COMPUTE: 4 };
  t.after(() => {
    if (previous === undefined) delete globalThis.GPUShaderStage;
    else globalThis.GPUShaderStage = previous;
  });
  for (const [features, kernel, expected] of [[[], "matmul", true], [["shader-f16"], "matmul", false], [[], "matmul_gate_up", false], [["shader-f16"], "matmul_wide", false]]) {
    const requests = [];
    const device = {
      features: new Set(features),
      createBindGroupLayout: (layout) => layout,
      createPipelineLayout: (layout) => layout,
      createShaderModule: (module) => module,
      createComputePipelineAsync: async (pipeline) => pipeline,
    };
    const pipes = await matmulPipelines(device, (name, spec) => { requests.push({ name, spec }); return ""; }, kernel);
    assert.equal(Boolean(pipes.mm56), expected);
    assert.equal(requests.length, expected ? 9 : 8);
    assert.deepEqual(requests.filter(({ spec }) => spec.row56), expected ? [{ name: "matmul", spec: { f16: false, groups: 1, splitTarget: 256, rows: 4, row56: true } }] : []);
    if (expected) assert.equal(pipes.mm56.layout, pipes.mm[4].layout);
    assert.deepEqual(requests.filter(({ name }) => name === "reduce").map(({ spec }) => spec.rows), [1, 2, 3, 4]);
    assert.ok(requests.filter(({ name }) => name === "reduce").every(({ spec }) => !spec.row56));
  }
});

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
