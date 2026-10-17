import test from "node:test";
import assert from "node:assert/strict";
import { calibrateVisionQkv, fuseVisionQkvOps, selectVisionQkv, visionQkvPlan } from "../js/src/gpu-gemma4-vision-qkv.js";
import { GpuGemma4Vision } from "../js/src/gpu-gemma4-vision.js";

const pipes = { weights: "bf16", f16: false, groups: 1, splitTarget: 256, layout: {}, reduceLayout: {}, mm56: {} };
const capacity = 12 * 1024 * 1024;
const guard = () => ({ ok: true, bitMismatches: 0, nonFinite: 0, tailMismatches: 0 });

test("QKV planning retains the actual Fast split and three aligned bounded planes", () => {
  const plan = visionQkvPlan(108, 768, 768, pipes, capacity);
  assert.deepEqual(plan, { rows: 108, width: 768, inputWidth: 768, splits: 6, partBytes: 1990656, offsets: [0, 1990656, 3981312], requiredBytes: 5971968, work: [12, 2, 6] });
  assert.ok(plan.offsets.every((offset) => offset % 256 === 0));
  assert.deepEqual(visionQkvPlan(108, 768, 768, pipes, plan.requiredBytes), plan);
  for (const [rows, N, K, config, bytes, alignment] of [
    [108, 768, 768, pipes, plan.requiredBytes - 1, 256], [315, 768, 768, pipes, capacity, 256],
    [108, 769, 768, pipes, capacity, 256], [108, 768, 769, pipes, capacity, 256],
    ...[{ weights: "f32" }, { f16: true }, { groups: 2 }, { splitTarget: 128 }].map((change) => [108, 768, 768, { ...pipes, ...change }, capacity, 256]),
    [108, 768, 768, pipes, capacity, 257],
  ]) assert.equal(visionQkvPlan(rows, N, K, config, bytes, alignment), null);
});

test("QKV selection requires both strict guards and every finite paired five-percent win", () => {
  const point = { before: guard(), after: guard(), baseline: [0.1, 0.101, 0.102], candidate: [0.094, 0.095, 0.096] };
  assert.equal(selectVisionQkv(point).selected, true);
  assert.equal(selectVisionQkv({ ...point, candidate: [0.094, 0.1, 0.096] }).selected, false);
  for (const field of ["bitMismatches", "nonFinite", "tailMismatches"]) {
    const after = { ...guard(), [field]: 1 };
    assert.equal(selectVisionQkv({ ...point, after }).reason, "numerical-guard-failed");
  }
  assert.equal(selectVisionQkv({ ...point, before: null }).selected, false);
  for (const candidate of [[0.09, NaN, 0.09], [0.09, 0, 0.09], [0.09, 0.09]]) assert.equal(selectVisionQkv({ ...point, candidate }).reason, "insufficient-timing-samples");
});

function graphFixture() {
  const kernel = { getBindGroupLayout() { return {}; } };
  const part = { size: capacity }, input = { size: 108 * 768 * 4 }, globals = { size: 16 };
  const ops = [];
  for (const [i, name] of ["q", "k", "v"].entries()) {
    const weights = { size: 768 * 768 * 2 }, normWeight = { size: 256 }, rope = { size: 108 * 64 * 4 }, output = { size: 108 * 768 * 4 }, params = { size: 16 };
    ops.push({ label: `vision.mm.${name}`, tokens: 108, partialOnly: true, pipes, op: { N: 768, K: 768 }, qkvMatrix: { dtype: "bf16", mode: 0, bias: 0, buffers: [globals, params, input, weights, {}, {}, {}, part] } });
    ops.push({ label: `vision.reduce_norm_rope.${name}`, p: kernel, group: {}, groups: [108 * 12], qkvNorm: { params: {}, values: [108, 12, 64, Number(i < 2), { f: 1e-6 }, { f: -2 }, { f: 2 }, Number(i === 2), 6, 0, 0, 0], buffers: [part, normWeight, rope, output] } });
  }
  const groups = [], device = { limits: { minStorageBufferOffsetAlignment: 256, maxStorageBufferBindingSize: capacity }, createBindGroup(value) { groups.push(value); return value; } };
  const context = { selected: true, pipeline: kernel, layout: {} };
  return { ops, device, context, groups, part };
}

test("QKV graph fusion preserves the original norm parameters, learned weights, RoPE, and outputs", () => {
  const { ops, device, context, groups, part } = graphFixture(), originalNorms = [ops[1], ops[3], ops[5]];
  assert.equal(fuseVisionQkvOps(device, context, ops), 1);
  assert.deepEqual(ops.map((op) => op.label), ["vision.mm.qkv", "vision.reduce_norm_rope.q", "vision.reduce_norm_rope.k", "vision.reduce_norm_rope.v"]);
  assert.equal(ops[0].qkv, true);
  assert.equal(groups.length, 4);
  assert.deepEqual(ops[0].groups, [12, 2, 6]);
  assert.deepEqual(groups[0].entries[6].resource, { buffer: part, offset: 0, size: 5971968 });
  for (const [i, op] of ops.slice(1).entries()) {
    assert.strictEqual(op.p, originalNorms[i].p);
    assert.strictEqual(op.groups, originalNorms[i].groups);
    assert.deepEqual(op.group.entries[1].resource, { buffer: part, offset: i * 1990656, size: 1990656 });
    assert.deepEqual(op.group.entries.slice(2).map((entry) => entry.resource.buffer), originalNorms[i].qkvNorm.buffers.slice(1));
    assert.strictEqual(op.group.entries[0].resource.buffer, originalNorms[i].qkvNorm.params);
  }
});

test("QKV graph fusion rejects unsupported producers, views, reducer contracts, and adjacency", () => {
  const changes = [
    (f) => { f.context.selected = false; }, (f) => { f.ops[0].tokens = 315; },
    (f) => { f.ops[2].qkvMatrix.dtype = "f32"; }, (f) => { f.ops[4].qkvMatrix.bias = 1; },
    (f) => { f.ops[2].partialOnly = false; }, (f) => { f.ops[2].qkvMatrix.buffers[2] = { size: 108 * 768 * 4 }; },
    (f) => { f.ops[3].qkvNorm.values[3] = 0; }, (f) => { f.ops[5].qkvNorm.values[7] = 0; },
    (f) => { f.ops[1].qkvNorm.buffers[0] = { size: capacity }; },
    (f) => { f.part.size = 5971967; }, (f) => { f.ops.splice(2, 0, { label: "vision.clip" }); },
  ];
  for (const change of changes) {
    const f = graphFixture(); change(f);
    const original = f.ops.slice();
    assert.equal(fuseVisionQkvOps(f.device, f.context, f.ops), 0);
    assert.deepEqual(f.ops, original);
    assert.equal(f.groups.length, 0);
  }
});

test("vision cache switches QKV graph selection off for toggles, profiling, and other image grids", () => {
  const savedUsage = globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage = { STORAGE: 1, UNIFORM: 2, COPY_DST: 4, COPY_SRC: 8, MAP_READ: 16 };
  const kernel = { getBindGroupLayout() { return {}; } }, device = { limits: { maxBufferSize: 256 * 1024 * 1024, maxStorageBufferBindingSize: 256 * 1024 * 1024, minStorageBufferOffsetAlignment: 256 },
    queue: { writeBuffer() {} }, createBuffer({ size }) { return { size, destroy() {} }; }, createBindGroup(value) { return value; } };
  const model = Object.create(GpuGemma4Vision.prototype);
  Object.assign(model, { device, owned: [], gpuKernel: "auto", fuseQkv: true, qkv: { selected: true, pipeline: kernel, layout: {} },
    cfg: { hidden: 768, intermediate: 3072, layers: 1, heads: 12, kvHeads: 12, headDim: 64, patch: 16, pool: 3, positionSize: 64, textHidden: 1536, eps: 1e-6 },
    fuseInputClips: true, fuseVisionOps: true, fuseReduceNorm: true, fuseResidualNorm: false, fuseReduceGelu: true, fusedAttention: true,
    p: Object.fromEntries(["position", "clip", "norm", "norm_rope", "reduce_norm_rope", "attention_fused", "gelu", "reduce_gelu", "pool_norm"].map((name) => [name, kernel])),
    mm: new Map([["bf16", pipes]]), clips: new Map(), weights: { get(name) {
      const shape = /(?:gate|up)_proj\.linear$/.test(name) ? [3072, 768] : /down_proj\.linear$/.test(name) ? [768, 3072] : /embedding_projection$/.test(name) ? [1536, 768] : [768, 768];
      return { buf: { name }, info: { shape, dtype: "bf16" } };
    } },
  });
  for (const name of ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o", "mlp.gate", "mlp.up", "mlp.down"]) model.clips.set(`vision_tower.encoder.layers.0.${name}_proj`, [-1, 1, -2, 2]);
  try {
    model.ensure(108, 12); const selectedOps = model.ops;
    assert.equal(model.fusionStats.qkvPasses, 1); assert.equal(model.fusionStats.qkvRemovedPasses, 2);
    model.ensure(108, 12); assert.strictEqual(model.ops, selectedOps);
    const fusedPasses = model.fusionStats.removedPasses;
    model.fuseQkv = false; model.ensure(108, 12);
    assert.notStrictEqual(model.ops, selectedOps); assert.equal(model.fusionStats.qkvPasses, 0); assert.equal(model.fusionStats.removedPasses, fusedPasses - 2);
    model.fuseQkv = true; model.ensure(108, 12); assert.equal(model.fusionStats.qkvPasses, 1);
    model.profiler = {}; model.ensure(108, 12); assert.equal(model.fusionStats.qkvPasses, 0);
    model.profiler = null; model.ensure(108, 12); assert.equal(model.fusionStats.qkvPasses, 1);
    model.gpuKernel = "generic"; model.ensure(108, 12); assert.equal(model.fusionStats.qkvPasses, 0);
    model.gpuKernel = "auto"; model.ensure(315, 35); assert.equal(model.fusionStats.qkvPasses, 0);
    model.fuseReduceNorm = false; model.ensure(108, 12); assert.equal(model.fusionStats.qkvPasses, 0);
  } finally { model.reset(); if (savedUsage === undefined) delete globalThis.GPUBufferUsage; else globalThis.GPUBufferUsage = savedUsage; }
});

test("QKV calibration fails closed before allocations when disabled or unsupported", async () => {
  const device = { features: new Set(), limits: { maxComputeWorkgroupStorageSize: 20480, maxStorageBuffersPerShaderStage: 8 } };
  const options = { cfg: {}, weights: {}, clips: {}, matmul: pipes, norm: {} };
  assert.equal((await calibrateVisionQkv({ device }, () => { throw new Error("must not compile"); }, { ...options, enabled: false })).diagnostics.reason, "disabled");
  const noTimestamps = await calibrateVisionQkv({ device }, () => { throw new Error("must not compile"); }, options);
  assert.equal(noTimestamps.selected, false); assert.equal(noTimestamps.diagnostics.reason, "timestamp-query-unavailable");
  assert.deepEqual(structuredClone(noTimestamps.diagnostics), noTimestamps.diagnostics);
  device.features.add("timestamp-query"); device.limits.maxComputeWorkgroupStorageSize = 16384;
  assert.equal((await calibrateVisionQkv({ device }, () => "", options)).diagnostics.reason, "device-limits");
});

test("QKV calibration releases unused pipelines on optional failure and propagates device loss", async () => {
  const savedStage = globalThis.GPUShaderStage;
  globalThis.GPUShaderStage = { COMPUTE: 4 };
  try {
    for (const loss of [null, "runtime", "message"]) {
      const scopes = [], compiled = [], runtime = {};
      runtime.device = {
        features: new Set(["timestamp-query"]),
        limits: { maxComputeWorkgroupStorageSize: 20480, maxStorageBuffersPerShaderStage: 8, minStorageBufferOffsetAlignment: 256 },
        pushErrorScope(kind) { scopes.push(kind); }, async popErrorScope() { scopes.pop(); return null; },
        createBindGroupLayout(value) { return value; }, createPipelineLayout(value) { return value; },
        createShaderModule(value) { return value; },
        async createComputePipelineAsync({ label }) {
          if (label === "gemma4_vision_qkv") { const result = { label }; compiled.push(result); return result; }
          if (loss === "runtime") runtime.lost = { message: "adapter removed" };
          throw new Error(loss === "message" ? "device lost while compiling validator" : "validator unavailable");
        },
      };
      const options = {
        cfg: { hidden: 768, heads: 12, kvHeads: 12, headDim: 64, theta: 10000 },
        matmul: pipes, norm: {},
        weights: { get(name) { return { buf: {}, info: { dtype: name.endsWith("_norm") ? "f32" : "bf16", shape: name.endsWith("_norm") ? [64] : [768, 768] } }; } },
        clips: { get() { return [-1, 1, -2, 2]; } },
      };
      const result = calibrateVisionQkv(runtime, () => "", options);
      if (loss) await assert.rejects(result, /device lost/i);
      else {
        const context = await result;
        assert.equal(context.selected, false); assert.equal(context.pipeline, null); assert.equal(context.layout, null);
        assert.equal(context.diagnostics.reason, "calibration-failed");
        assert.equal(context.diagnostics.available, false);
        assert.deepEqual(structuredClone(context.diagnostics), context.diagnostics);
      }
      assert.equal(compiled.length, 1);
      assert.deepEqual(scopes, []);
    }
  } finally {
    if (savedStage === undefined) delete globalThis.GPUShaderStage; else globalThis.GPUShaderStage = savedStage;
  }
});
