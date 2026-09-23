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

test("Gemma GPU config canonicalizes the global attention alias", () => {
  const cfg = gemma4Config({ config: { ...base, layer_types: ["sliding_attention", "global_attention", "sliding_attention", "global_attention"] } });
  assert.deepEqual(cfg.layerTypes, ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]);
  assert.deepEqual(cfg.full, [false, true, false, true]);
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
