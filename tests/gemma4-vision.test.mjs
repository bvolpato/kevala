import assert from "node:assert/strict";
import test from "node:test";
import { gemma4VisionConfig, GpuGemma4Vision, sharedInputClipBounds, visionImagePlan, visionMlpReductionPlan } from "../js/src/gpu-gemma4-vision.js";
import { clearGemma4VisionImageData, consumeGemma4VisionImage, disposeGemma4VisionImage, prepareGemma4GpuVisionImage, prepareGemma4Image, prepareGemma4VisionImage } from "../js/src/gemma4-image.js";
import gemma4 from "../js/src/archs/gemma4.js";

const vision = {
  hidden_size: 64, intermediate_size: 128, num_hidden_layers: 2, num_attention_heads: 2,
  num_key_value_heads: 2, head_dim: 32, patch_size: 8, pooling_kernel_size: 3,
  position_embedding_size: 12, rms_norm_eps: 1e-6, hidden_activation: "gelu_pytorch_tanh",
  rope_parameters: { rope_type: "default", rope_theta: 100 }, use_clipped_linears: true,
};
const cfg = gemma4VisionConfig({ config: { vision_config: vision, text_hidden_size: 96 } });

test("input clip fusion requires identical real bounds and falls back on mismatch", () => {
  assert.deepEqual(sharedInputClipBounds([-2, 3, -1, 1], [-2, 3, -4, 4], [-2, 3, -Infinity, Infinity]), [-2, 3]);
  assert.equal(sharedInputClipBounds([-2, 3], [-2, 4], [-2, 3]), null);
  assert.equal(sharedInputClipBounds([-0, 3], [0, 3]), null);
  assert.equal(sharedInputClipBounds([-Infinity, Infinity], [-Infinity, Infinity]), null);
  assert.equal(sharedInputClipBounds([-2, 3], null), null);
});

test("vision MLP fusion keeps aligned partial planes inside the existing scratch", () => {
  const pipes = { f16: false, groups: 1, splitTarget: 256 };
  const capacity = 12 * 1024 * 1024;
  const plan = visionMlpReductionPlan(108, 3072, 768, pipes, capacity);
  assert.deepEqual(plan, { splits: 3, partBytes: 3981312, upOffset: 3981312, requiredBytes: 7962624 });
  assert.equal(visionMlpReductionPlan(108, 3072, 768, pipes, plan.requiredBytes - 1), null);
  assert.deepEqual(visionMlpReductionPlan(108, 3072, 768, pipes, plan.requiredBytes), plan);
  const padded = visionMlpReductionPlan(1, 4, 768, pipes, capacity);
  assert.equal(padded.upOffset % 256, 0);
  assert.ok(padded.upOffset >= padded.partBytes);
  for (const rows of [315, 567]) assert.equal(visionMlpReductionPlan(rows, 3072, 768, pipes, capacity), null);
  assert.equal(visionMlpReductionPlan(108, 3072, 768, { ...pipes, f16: true }, capacity), null);
  assert.equal(visionMlpReductionPlan(108, 3073, 768, pipes, capacity), null);
  assert.equal(visionMlpReductionPlan(108, 3072, 769, pipes, capacity), null);
});

test("vision MLP graph uses disjoint partial views and restores separate reducers on fallback", () => {
  const savedUsage = globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage = { STORAGE: 1, UNIFORM: 2, COPY_DST: 4, COPY_SRC: 8, MAP_READ: 16 };
  const device = {
    limits: { maxBufferSize: 256 * 1024 * 1024, maxStorageBufferBindingSize: 256 * 1024 * 1024, minStorageBufferOffsetAlignment: 256 },
    queue: { writeBuffer() {} },
    createBuffer({ size, label }) { return { size, label, destroy() {} }; },
    createBindGroup(value) { return value; },
  };
  const kernel = { getBindGroupLayout() { return {}; } };
  const model = Object.create(GpuGemma4Vision.prototype);
  Object.assign(model, { device, owned: [], cfg: { hidden: 768, intermediate: 3072, layers: 2, heads: 12, kvHeads: 12, headDim: 64, patch: 16, pool: 3, positionSize: 64, textHidden: 1536, eps: 1e-6 },
    fuseInputClips: true, fuseVisionOps: true, fuseResidualNorm: false, fuseReduceNorm: false, fuseReduceGelu: true, fusedAttention: true,
    p: Object.fromEntries(["position", "clip", "norm", "norm_rope", "residual_norm", "reduce_residual", "attention_fused", "gelu", "reduce_gelu", "pool_norm"].map((name) => [name, kernel])),
    mm: new Map([["bf16", { f16: false, groups: 1, splitTarget: 256, layout: {}, reduceLayout: {} }]]),
    weights: { get(name) {
      const shape = /(?:gate|up)_proj\.linear$/.test(name) ? [3072, 768] : /down_proj\.linear$/.test(name) ? [768, 3072] : /embedding_projection$/.test(name) ? [1536, 768] : [768, 768];
      return { buf: { name }, info: { shape, dtype: "bf16" } };
    } }, clips: new Map(),
  });
  for (let layer = 0; layer < 2; layer++) for (const name of ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o", "mlp.gate", "mlp.up", "mlp.down"]) model.clips.set(`vision_tower.encoder.layers.${layer}.${name}_proj`, [-1, 1, -2, 2]);
  try {
    model.ensure(108, 12);
    assert.equal(model.fusionStats.reduceGeluPasses, 4);
    const projections = model.ops.filter(({ label }) => ["vision.mm.gate", "vision.mm.up"].includes(label));
    assert.equal(projections.length, 4);
    for (let i = 0; i < projections.length; i += 2) {
      assert.equal(projections[i].partialOnly, true);
      assert.equal(projections[i + 1].partialOnly, true);
      const gate = projections[i].op.group.entries[7].resource, up = projections[i + 1].op.group.entries[7].resource;
      assert.equal(gate.buffer, up.buffer);
      assert.equal(gate.offset, 0);
      assert.ok(up.offset >= gate.size);
      assert.ok(up.offset + up.size <= up.buffer.size);
      const fused = model.ops.filter(({ label }) => label === "vision.reduce_gelu")[i / 2];
      assert.deepEqual(fused.group.entries.slice(1, 3).map(({ resource }) => resource), [gate, up]);
    }
    const firstOps = model.ops;
    model.ensure(108, 12);
    assert.equal(model.ops, firstOps);
    model.fuseReduceGelu = false;
    model.ensure(108, 12);
    assert.notEqual(model.ops, firstOps);
    assert.equal(model.fusionStats.reduceGeluPasses, 0);
    assert.ok(model.ops.filter(({ label }) => ["vision.mm.gate", "vision.mm.up"].includes(label)).every((op) => !op.partialOnly));
    assert.equal(model.ops.filter(({ label }) => label === "vision.gelu").length, 2);
    model.fuseReduceGelu = true;
    model.ensure(315, 35);
    assert.equal(model.fusionStats.reduceGeluPasses, 0);
    assert.equal(model.ops.filter(({ label }) => label === "vision.gelu").length, 2);
    model.fuseResidualNorm = true;
    model.fuseReduceResidual = true;
    model.ensure(108, 12);
    assert.equal(model.fusionStats.residualNormPasses, 3);
    assert.equal(model.fusionStats.reduceResidualPasses, 3);
    assert.deepEqual(model.ops.filter(({ label }) => ["vision.mm.o", "vision.mm.down"].includes(label)).map((op) => Boolean(op.partialOnly)), [true, true, true, false]);
    assert.equal(model.ops.filter(({ label }) => label?.endsWith(".reduce.next_norm")).length, 3);
    model.fuseReduceResidual = false;
    model.ensure(108, 12);
    assert.equal(model.fusionStats.reduceResidualPasses, 0);
    assert.equal(model.fusionStats.residualNormPasses, 3);
    assert.ok(model.ops.filter(({ label }) => ["vision.mm.o", "vision.mm.down"].includes(label)).every((op) => !op.partialOnly));
  } finally {
    model.reset();
    if (savedUsage === undefined) delete globalThis.GPUBufferUsage;
    else globalThis.GPUBufferUsage = savedUsage;
  }
});

function image() {
  const positions = new Int32Array(36 * 2).fill(-1);
  for (let y = 0; y < 3; y++) for (let x = 0; x < 6; x++) positions.set([x, y], (y * 6 + x) * 2);
  return { positions, patches: new Float32Array(36 * 192).fill(0.25), softTokens: 2 };
}

test("vision processor plan preserves spatial pools when padded patch rows are reordered", () => {
  const input = image();
  const ordered = visionImagePlan(input, cfg);
  assert.equal(ordered.rows, 18);
  assert.equal(visionImagePlan(input, cfg, false).rows, 36);
  assert.equal(ordered.softRows, 2);
  assert.deepEqual([...ordered.pools], [0, 1, 2, 6, 7, 8, 12, 13, 14, 3, 4, 5, 9, 10, 11, 15, 16, 17]);
  assert.ok(ordered.pixels.every((value) => value === -0.5));
  const reversed = { ...input, positions: Int32Array.from({ length: 72 }, (_, i) => input.positions[(35 - Math.floor(i / 2)) * 2 + i % 2]) };
  const plan = visionImagePlan(reversed, cfg);
  const coords = (source, indices) => [...indices].map((row) => `${source[row * 2]}:${source[row * 2 + 1]}`).sort();
  for (let pool = 0; pool < 2; pool++) assert.deepEqual(coords(plan.positions, plan.pools.subarray(pool * 9, pool * 9 + 9)), coords(ordered.positions, ordered.pools.subarray(pool * 9, pool * 9 + 9)));
  const axis = 32 + 2;
  assert.ok(Math.abs(ordered.rope[axis] - Math.cos(Math.fround(1 / Math.fround(100 ** (1 / 8))))) < 1e-7);
});

test("compacting image patches preserves each position's pixels and excludes padded rows", () => {
  const input = image();
  const patchWidth = 192;
  for (let row = 0; row < 36; row++) input.patches.fill(row / 36, row * patchWidth, (row + 1) * patchWidth);
  const reversed = {
    ...input,
    positions: Int32Array.from({ length: 72 }, (_, i) => input.positions[(35 - Math.floor(i / 2)) * 2 + i % 2]),
    patches: Float32Array.from({ length: input.patches.length }, (_, i) => input.patches[(35 - Math.floor(i / patchWidth)) * patchWidth + i % patchWidth]),
  };
  const plan = visionImagePlan(reversed, cfg);
  assert.equal(plan.rows, 18);
  assert.ok(plan.positions.every((value) => value >= 0));
  for (let row = 0; row < plan.rows; row++) {
    const source = plan.positions[row * 2 + 1] * 6 + plan.positions[row * 2];
    assert.equal(plan.pixels[row * patchWidth], Math.fround(2 * (input.patches[source * patchWidth] - 0.5)));
  }
});

test("vision rejects malformed positions, pixel values and soft-token counts before GPU work", () => {
  let input = image();
  input.positions.set([0, 0], 2);
  assert.throws(() => visionImagePlan(input, cfg), /duplicate/);
  input = image();
  input.positions.set([-1, 0], 0);
  assert.throws(() => visionImagePlan(input, cfg), /outside/);
  input = image();
  input.positions.set([-1, -1], 0);
  assert.throws(() => visionImagePlan(input, cfg), /complete rectangular/);
  input = image();
  assert.throws(() => visionImagePlan({ ...input, softTokens: 4 }, cfg), /reports 4 soft tokens/);
  input.patches[0] = NaN;
  assert.throws(() => visionImagePlan(input, cfg), /finite values/);
  assert.throws(() => visionImagePlan({ ...image(), positions: new Uint32Array(72) }, cfg), /signed Int32/);
});

test("repeated image geometry uses fresh pixels and still rejects changed or invalid input", () => {
  const input = image();
  const first = visionImagePlan(input, cfg);
  first.rope.fill(0);
  first.positions.fill(-1);
  first.pools.fill(99);
  input.patches.fill(0.75);
  const next = visionImagePlan(input, cfg);
  assert.ok(next.pixels.every((value) => value === 0.5));
  assert.equal(next.rope[0], 1);
  assert.equal(next.positions[0], 0);
  assert.equal(next.pools[0], 0);
  input.patches[0] = Infinity;
  assert.throws(() => visionImagePlan(input, cfg), /finite values/);
  input.patches[0] = 0.5;
  assert.throws(() => visionImagePlan({ ...input, softTokens: 3 }, cfg), /reports 3 soft tokens/);
  input.positions.set([0, 0], 2);
  assert.throws(() => visionImagePlan(input, cfg), /duplicate/);
});

const processorCfg = { ...cfg, patch: 16, positionSize: 64 };
const frame = (offset = 0) => ({ width: 32, height: 24, data: Uint8Array.from({ length: 32 * 24 * 4 }, (_, i) => (i * 37 + offset) & 255) });

test("private prepared plans preserve full pixels and geometry across changing frames", () => {
  for (const offset of [0, 71, 0]) {
    const input = frame(offset);
    const reference = visionImagePlan(prepareGemma4Image(input, 16), processorCfg);
    const handle = prepareGemma4VisionImage(input, 16);
    const plan = visionImagePlan(handle, processorCfg);
    assert.deepEqual(plan, reference);
    assert.throws(() => visionImagePlan(handle, processorCfg), /already been consumed or disposed/);
    plan.pixels.fill(0);
    reference.pixels.fill(0);
  }
});

test("private prepared metadata cannot bypass validation or survive an invalid consume", () => {
  const handle = prepareGemma4VisionImage(frame(), 16);
  assert.throws(() => { handle.width = 1; }, TypeError);
  assert.throws(() => visionImagePlan({ ...handle }, processorCfg), /expects Float32 patches/);
  assert.throws(() => visionImagePlan({ ...handle, patches: Float32Array.of(NaN), positions: Int32Array.of(0, 0) }, processorCfg), /inconsistent dimensions/);
  const plan = visionImagePlan(handle, processorCfg);
  plan.pixels.fill(0);
  for (const [config, compact, error] of [[{ ...processorCfg, patch: 8 }, true, /matching compact/], [processorCfg, false, /matching compact/], [{ ...processorCfg, positionSize: 2 }, true, /outside the learned/]]) {
    const pending = prepareGemma4VisionImage(frame(), 16);
    assert.throws(() => visionImagePlan(pending, config, compact), error);
    assert.throws(() => consumeGemma4VisionImage(pending, processorCfg), /already been consumed or disposed/);
  }
  const publicImage = prepareGemma4Image(frame(), 16);
  assert.equal(visionImagePlan(publicImage, processorCfg, false).rows, 144);
  publicImage.patches[0] = NaN;
  assert.throws(() => visionImagePlan(publicImage, processorCfg), /finite values/);
});

test("public vision plans materialize private raw handles as ordinary CPU plans", () => {
  const input = frame();
  const reference = visionImagePlan(prepareGemma4Image(input, 16), processorCfg);
  const handle = prepareGemma4GpuVisionImage(input, 16);
  assert.deepEqual(visionImagePlan(handle, processorCfg), reference);
  assert.throws(() => visionImagePlan(handle, processorCfg), /already been consumed or disposed/);
});

test("optional GPU setup failure falls back before upload and clears both owned representations", async () => {
  const failure = new Error("stop after CPU fallback");
  const reference = visionImagePlan(prepareGemma4Image(frame(), 16), processorCfg);
  let raw, uploaded;
  const device = { pushErrorScope() {}, async popErrorScope() { return null; }, queue: { writeBuffer(_buffer, _offset, values) { uploaded = values; assert.deepEqual(values, reference.pixels); throw failure; } } };
  const model = Object.create(GpuGemma4Vision.prototype);
  Object.assign(model, { cfg: processorCfg, runtime: { device }, device, prepareGpuPixels: true, imageGpu: { enabled: true, async prepare(value) { raw = value.rgba; return null; } }, ensure() {}, reset() {} });
  const handle = prepareGemma4GpuVisionImage(frame(), 16);
  await assert.rejects(model.forward(handle), (error) => error === failure);
  assert.ok(raw.every((value) => value === 0));
  assert.ok(uploaded.every((value) => value === 0));
  assert.equal(model.lastGpuPixelDispatches, 0);
  assert.equal(model.running, false);
});

test("GPU preparation errors dispose raw frames and profiling uses CPU pixels", async () => {
  const failure = new Error("device lost during setup");
  let raw, uploads = 0;
  const device = { pushErrorScope() {}, async popErrorScope() { return null; }, queue: { writeBuffer() { uploads++; throw failure; } } };
  const model = Object.create(GpuGemma4Vision.prototype);
  Object.assign(model, { cfg: processorCfg, runtime: { device }, device, prepareGpuPixels: true, imageGpu: { enabled: true, async prepare(value) { raw = value.rgba; throw failure; } }, ensure() {}, reset() {} });
  await assert.rejects(model.forward(prepareGemma4GpuVisionImage(frame(), 16)), (error) => error === failure);
  assert.ok(raw.every((value) => value === 0));
  assert.equal(uploads, 0);
  model.profiler = {};
  await assert.rejects(model.forward(prepareGemma4GpuVisionImage(frame(), 16)), (error) => error === failure);
  assert.equal(uploads, 1);
  assert.equal(model.gpuPixelDiagnostics.reason, "profiler");
});

test("architecture borrows raw request RGBA without mutating the caller's frame", async () => {
  const input = frame(), requestImage = { ...input, data: new Uint8Array(input.data) }, failure = new Error("stop before decoder");
  let owned;
  const engine = { imageTokens: 70, visionHeader: { config: { vision_config: {} } }, vision: {
    prepareCompactPixels: true, prepareGpuPixels: true, compactPatches: true, imageGpu: { enabled: true }, cfg: { textHidden: 96 },
    async forward(image) {
      owned = consumeGemma4VisionImage(image, processorCfg);
      assert.strictEqual(owned.raw.rgba.buffer, requestImage.data.buffer);
      assert.deepEqual(owned.raw.rgba, input.data);
      assert.equal(owned.raw.borrowedInput, true);
      throw failure;
    },
  } };
  try {
    await assert.rejects(gemma4.run(engine, [{ state: "camera", questions: [], imageTokens: 16, parts: [{ type: "image", data: requestImage }] }]), (error) => error === failure);
  } finally { clearGemma4VisionImageData(owned); }
  assert.deepEqual(requestImage.data, input.data); // requestQueue, not the image helper, owns this wipe
  requestImage.data.fill(0);
  assert.ok(requestImage.data.every((value) => value === 0));
  assert.ok(input.data.some((value) => value !== 0));
});

test("failed or overlapping vision work disposes owned frame pixels", async () => {
  const failure = new Error("upload failed");
  let uploaded;
  const device = {
    pushErrorScope() {}, async popErrorScope() { return null; },
    queue: { writeBuffer(buffer, offset, pixels) { uploaded = pixels; throw failure; } },
  };
  const model = Object.create(GpuGemma4Vision.prototype);
  Object.assign(model, { cfg: processorCfg, runtime: { device }, device, ensure() {}, reset() {} });
  const handle = prepareGemma4VisionImage(frame(), 16);
  await assert.rejects(model.forward(handle), (error) => error === failure);
  assert.ok(uploaded.length > 0);
  assert.ok(uploaded.every((value) => value === 0));
  assert.equal(model.running, false);
  assert.throws(() => visionImagePlan(handle, processorCfg), /already been consumed or disposed/);
  model.running = true;
  const pending = prepareGemma4VisionImage(frame(), 16);
  await assert.rejects(model.forward(pending), /already running an image/);
  assert.throws(() => consumeGemma4VisionImage(pending, processorCfg), /already been consumed or disposed/);
  const publicImage = prepareGemma4Image(frame(), 16);
  const firstPixel = publicImage.patches[0];
  disposeGemma4VisionImage(publicImage);
  assert.equal(publicImage.patches[0], firstPixel);
});

test("architecture selects compact preparation only when enabled and cleans rejected frames", async () => {
  for (const [enabled, compact, owned] of [[true, true, true], [false, true, false], [true, false, false]]) {
    let prepared;
    const input = frame(), requestImage = { ...input, data: new Uint8Array(input.data) };
    const failure = new DOMException("frame aborted", "AbortError");
    const engine = {
      imageTokens: 70, visionHeader: { config: { vision_config: {} } },
      vision: { prepareCompactPixels: enabled, compactPatches: compact, cfg: { textHidden: 96 }, async forward(image) {
        prepared = image;
        assert.equal(image.patches instanceof Float32Array, !owned);
        assert.equal(image.paddedRows ?? image.positions.length / 2, 144);
        assert.equal(image.softTokens, 12);
        assert.deepEqual(requestImage.data, input.data);
        throw failure;
      } },
    };
    await assert.rejects(gemma4.run(engine, [{ state: "camera", questions: [], imageTokens: 16, parts: [{ type: "image", data: requestImage }] }]), (error) => error === failure);
    if (owned) assert.throws(() => consumeGemma4VisionImage(prepared, processorCfg), /already been consumed or disposed/);
    else assert.ok(prepared.patches.every((value) => value === 0));
    assert.deepEqual(requestImage.data, input.data);
    requestImage.data.fill(0); // Simulate the worker request release hook.
    assert.ok(requestImage.data.every((value) => value === 0));
    assert.ok(input.data.some((value) => value !== 0));
  }
});

test("vision config rejects unsupported attention and projection layouts", () => {
  const config = (change) => ({ config: { text_hidden_size: 96, vision_config: { ...vision, ...change } } });
  assert.throws(() => gemma4VisionConfig(config({ head_dim: 20 })), /matrix or head dimensions/);
  assert.throws(() => gemma4VisionConfig(config({ num_key_value_heads: 3 })), /matrix or head dimensions/);
  assert.throws(() => gemma4VisionConfig(config({ attention_bias: true })), /unbiased projections/);
  assert.throws(() => gemma4VisionConfig(config({ rope_parameters: { rope_type: "linear", rope_theta: 100 } })), /axial RoPE/);
});
