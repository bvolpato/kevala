import assert from "node:assert/strict";
import test from "node:test";
import { gemma4Config } from "../js/src/gpu-gemma4.js";

const base = {
  hidden_size: 32,
  intermediate_size: 32,
  num_hidden_layers: 4,
  num_attention_heads: 1,
  num_key_value_heads: 1,
  layer_types: ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
  head_dim: 32,
  global_head_dim: 32,
  num_kv_shared_layers: 2,
  sliding_window: 4,
  max_position_embeddings: 128,
  vocab_size: 64,
  vocab_size_per_layer_input: 64,
  hidden_size_per_layer_input: 32,
  rms_norm_eps: 1e-6,
};

test("Gemma GPU config maps every pack field the trunk reads, flat or nested in text_config", () => {
  const expected = {
    hidden: 32, layers: 4, intermediate: 32, heads: 1, kvHeads: 1, localHeadDim: 32, globalHeadDim: 32,
    layerTypes: base.layer_types, full: [false, true, false, true], sharedStart: 2, kvSharedLayers: 2,
    window: 4, maxPosition: 128, vocab: 64, pleDim: 32, pleVocab: 64, eps: 1e-6,
    // sqrt(32) rounded to bfloat16, as the upstream embedding scale buffer holds it
    embeddingScale: 5.65625, pleEmbeddingScale: 5.65625, pleInputScale: Math.SQRT1_2, pleProjectionScale: 1 / Math.sqrt(32),
    useDoubleWideMlp: false, ropeLocalTheta: 10000, ropeGlobalTheta: 1000000, ropeGlobalFraction: 0.25, causal: 1,
  };
  assert.deepEqual(gemma4Config({ config: base }), expected);
  assert.deepEqual(gemma4Config({ config: { arch: "gemma4", text_config: base } }), expected);

  // values the converter records, and per-type rotary settings, replace the defaults
  const cfg = gemma4Config({ config: {
    ...base, use_double_wide_mlp: true, embedding_scale: 5.5, ple_embedding_scale: 5.75,
    rope_parameters: { sliding_attention: { rope_theta: 5000 }, full_attention: { rope_theta: 2e6, partial_rotary_factor: 0.5 } },
  } });
  assert.deepEqual(
    [cfg.useDoubleWideMlp, cfg.embeddingScale, cfg.pleEmbeddingScale, cfg.ropeLocalTheta, cfg.ropeGlobalTheta, cfg.ropeGlobalFraction],
    [true, 5.5, 5.75, 5000, 2e6, 0.5],
  );
});

test("Gemma GPU config rejects the dimensions and layer names the coordinator rejects", () => {
  const rejects = (patch, message) => assert.throws(() => gemma4Config({ config: { ...base, ...patch } }), message);
  rejects({ layer_types: ["sliding_attention", "global_attention", "sliding_attention", "full_attention"] }, /invalid layer_types/);
  rejects({ num_hidden_layers: 0 }, /invalid model dimensions/);
  rejects({ hidden_size: "32px" }, /invalid model dimensions/);
  rejects({ num_attention_heads: 3, num_key_value_heads: 2 }, /divisible by KV heads/);
  rejects({ head_dim: 31 }, /even local head dimensions/);
  rejects({ global_head_dim: 1024 }, /global dimensions up to 512/);
  rejects({ num_kv_shared_layers: 4 }, /invalid KV sharing count/);
  rejects({ hidden_size: 30 }, /divisible by four/);
  rejects({ sliding_window: 0 }, /invalid sliding window/);
  rejects({ rope_parameters: { full_attention: { partial_rotary_factor: 1.5 } } }, /unsupported global proportional RoPE/);
  rejects({ attention_bias: "yes" }, /attention_bias must be boolean/);
});

test("Gemma GPU config requires a full final layer and a source for every shared type", () => {
  assert.throws(() => gemma4Config({ config: { ...base, layer_types: undefined } }), /requires layer_types/);
  assert.throws(() => gemma4Config({ config: { ...base, layer_types: ["sliding_attention", "full_attention", "sliding_attention", "sliding_attention"] } }), /final decoder layer/);
  assert.throws(() => gemma4Config({ config: { ...base, layer_types: ["sliding_attention", "sliding_attention", "full_attention", "full_attention"] } }), /no non-shared KV source for full_attention/);
});

test("Gemma GPU config rejects bidirectional direct scoring", () => {
  assert.throws(() => gemma4Config({ config: { ...base, use_bidirectional_attention: "all" } }), /does not support use_bidirectional_attention/);
});

test("Gemma GPU config rejects native-incompatible activation and attention variants", () => {
  assert.throws(() => gemma4Config({ config: { ...base, hidden_activation: "silu" } }), /hidden_activation/);
  assert.throws(() => gemma4Config({ config: { ...base, attention_bias: true } }), /attention_bias/);
  assert.throws(() => gemma4Config({ config: { ...base, attention_k_eq_v: true } }), /attention_k_eq_v/);
  assert.throws(() => gemma4Config({ config: { ...base, enable_moe_block: true } }), /enable_moe_block/);
  assert.throws(() => gemma4Config({ config: { ...base, rope_parameters: { sliding_attention: { rope_type: "linear" }, full_attention: { rope_type: "proportional" } } } }), /rope_type/);
});

test("Gemma GPU config accepts a valid zero-sharing fallback", () => {
  const cfg = gemma4Config({ config: { ...base, num_kv_shared_layers: 0 } });
  assert.equal(cfg.sharedStart, cfg.layers);
  assert.equal(cfg.kvSharedLayers, 0);
});
