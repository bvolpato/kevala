import test from "node:test";
import assert from "node:assert/strict";
import { calibrateGemma4Attention, gemma4AttentionScope, gemma4AttentionScratchPlan, gemma4LongAttentionFamilies, selectGemma4AttentionScopes } from "../js/src/gpu-gemma4-attention.js";

const point = (kind, options = {}) => ({ kind, window: 512, numericalCheck: { ok: true }, afterTimingCheck: { ok: true }, baselineMs: [1, 1.01, 0.99], candidateMs: [0.9, 0.91, 0.89], ...options });

test("long subgroup attention stays inside its calibrated causal unbatched range", () => {
  assert.equal(gemma4AttentionScope(2, { batched: true }), "tail");
  for (const tokens of [16, 128, 256]) assert.equal(gemma4AttentionScope(tokens, { batched: true }), "prefill");
  for (const tokens of [257, 512, 602, 640]) assert.equal(gemma4AttentionScope(tokens), "long");
  for (const tokens of [0, 1, 3, 15, 641, 1024, NaN, 602.5]) assert.equal(gemma4AttentionScope(tokens), null);
  for (const options of [{ batched: true }, { causal: false }, { causal: 0 }, { prefixTokens: 89 }, { prefixTokens: NaN }]) assert.equal(gemma4AttentionScope(602, options), null);
});

test("long attention needs exact stable wins at all calibrated lengths with one matching window", () => {
  const records = [point("prefix"), point("suffix"), point("tail"), point("long-257"), point("long-512"), point("long-602")];
  assert.deepEqual([...selectGemma4AttentionScopes(256, records)], ["256:prefill", "256:tail", "256:long:512"]);
  assert.deepEqual([...selectGemma4AttentionScopes(512, records.map((item) => ({ ...item, window: 0 })))], ["512:prefill", "512:tail", "512:long:0"]);
  assert.deepEqual([...selectGemma4AttentionScopes(512, records.slice(0, -1))], ["512:prefill", "512:tail"]);
  for (const candidateMs of [[0.96, 0.97, 0.95], [0.9, 1.1, 0.9], [0.9, NaN, 0.9], [0.9, 0.9]]) {
    assert.equal(selectGemma4AttentionScopes(256, [point("long-257"), point("long-512"), point("long-602", { candidateMs })]).size, 0);
  }
  assert.equal(selectGemma4AttentionScopes(256, records.filter((item) => item.kind !== "long-257")).has("256:long:512"), false);
  assert.equal(selectGemma4AttentionScopes(256, [...records, point("long-257", { candidateMs: [1.2, 1.2, 1.2] })]).has("256:long:512"), false);
  assert.equal(selectGemma4AttentionScopes(256, [...records, point("long-602", { window: 0 })]).has("256:long:512"), false);
  assert.equal(selectGemma4AttentionScopes(256, [...records, point("bad", { numericalCheck: { ok: false } })]).size, 0);
  assert.equal(selectGemma4AttentionScopes(256, [...records, point("bad", { afterTimingCheck: { ok: false } })]).size, 0);
  assert.equal(selectGemma4AttentionScopes(256, [point("long-257"), point("long-512"), point("long-602", { afterTimingCheck: undefined })]).size, 0);
  assert.equal(selectGemma4AttentionScopes(128, records).size, 0);
});

test("long attention scratch is bounded for E2B and E4B dimensions", () => {
  for (const tokens of [257, 512, 602, 640]) for (const headDim of [256, 512]) for (const kvHeads of [1, 2]) {
    const plan = gemma4AttentionScratchPlan({ tokens, total: tokens, heads: 8, kvHeads, headDim });
    assert.equal(plan.withinBudget, true);
    assert.equal(plan.limitBytes, 64 * 1024 * 1024);
    assert.ok(plan.scratchBytes < plan.limitBytes);
    assert.equal(plan.activeFloats, tokens * 8 * headDim);
    assert.equal(plan.readBytes, plan.outputBytes);
  }
  const legacy = gemma4AttentionScratchPlan({ tokens: 97, total: 186, heads: 8, kvHeads: 1, headDim: 512 });
  assert.equal(legacy.limitBytes, 32 * 1024 * 1024);
  assert.equal(legacy.withinBudget, true);
  assert.equal(gemma4AttentionScratchPlan({ tokens: 602, total: 602, heads: 64, kvHeads: 8, headDim: 512 }).withinBudget, false);
  assert.throws(() => gemma4AttentionScratchPlan({ tokens: 602, total: 602, heads: 8, kvHeads: 3, headDim: 512 }), /dimensions/);
});

test("long calibration keeps local and global windows distinct even at equal head dimensions", () => {
  assert.deepEqual(gemma4LongAttentionFamilies({ localHeadDim: 256, globalHeadDim: 256, window: 512 }), [{ headDim: 256, window: 512 }, { headDim: 256, window: 0 }]);
  assert.deepEqual(gemma4LongAttentionFamilies({ localHeadDim: 256, globalHeadDim: 512, window: 512 }), [{ headDim: 256, window: 512 }, { headDim: 512, window: 0 }]);
  assert.deepEqual(gemma4LongAttentionFamilies({ localHeadDim: 256, globalHeadDim: 256, window: 0 }), [{ headDim: 256, window: 0 }]);
  assert.deepEqual(gemma4LongAttentionFamilies({ localHeadDim: 128, globalHeadDim: 128, window: 512 }), []);
});

test("attention calibration stops before allocating on unsupported or disabled devices", async () => {
  const baseline = { getBindGroupLayout() {} };
  const unsupported = await calibrateGemma4Attention({ device: {}, adapter: { info: {} }, subgroup4: false }, () => "", {}, baseline);
  assert.equal(unsupported.diagnostics.reason, "requires-fixed-32-subgroups");
  assert.equal(unsupported.selection.size, 0);
  const disabled = await calibrateGemma4Attention({}, null, {}, baseline, { enabled: false });
  assert.equal(disabled.diagnostics.reason, "disabled");
  assert.equal(disabled.selection.size, 0);
});
