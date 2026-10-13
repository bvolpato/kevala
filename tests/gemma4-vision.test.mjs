import assert from "node:assert/strict";
import test from "node:test";
import { gemma4VisionConfig, visionImagePlan } from "../js/src/gpu-gemma4-vision.js";

const vision = {
  hidden_size: 64, intermediate_size: 128, num_hidden_layers: 2, num_attention_heads: 2,
  num_key_value_heads: 2, head_dim: 32, patch_size: 8, pooling_kernel_size: 3,
  position_embedding_size: 12, rms_norm_eps: 1e-6, hidden_activation: "gelu_pytorch_tanh",
  rope_parameters: { rope_type: "default", rope_theta: 100 }, use_clipped_linears: true,
};
const cfg = gemma4VisionConfig({ config: { vision_config: vision, text_hidden_size: 96 } });

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

test("vision config rejects unsupported attention and projection layouts", () => {
  const config = (change) => ({ config: { text_hidden_size: 96, vision_config: { ...vision, ...change } } });
  assert.throws(() => gemma4VisionConfig(config({ head_dim: 20 })), /matrix or head dimensions/);
  assert.throws(() => gemma4VisionConfig(config({ num_key_value_heads: 3 })), /matrix or head dimensions/);
  assert.throws(() => gemma4VisionConfig(config({ attention_bias: true })), /unbiased projections/);
  assert.throws(() => gemma4VisionConfig(config({ rope_parameters: { rope_type: "linear", rope_theta: 100 } })), /axial RoPE/);
});
