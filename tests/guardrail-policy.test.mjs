import assert from "node:assert/strict";
import { test } from "node:test";
import { assess, DEFAULT_THRESHOLDS } from "../app/views/guardrail-policy.js";

const scores = (override, educational, overrideAct = 1, contextAct = 1) => ({
  instruction_override: { noul: override, action: { act_probability: overrideAct } },
  educational_context: { noul: educational, action: { act_probability: contextAct } },
});

test("an ordinary request is allowed even when analysis context is uncertain", () => {
  assert.equal(assess(scores(0.20, 0.66)).verdict, "allow");
});

test("clear overrides block, while a quoted attack for analysis goes to review", () => {
  assert.equal(assess(scores(0.91, 0.23)).verdict, "block");
  assert.equal(assess(scores(0.96, 0.89)).verdict, "review");
});

test("uncertain scores and a low act probability require review", () => {
  assert.equal(assess(scores(0.55, 0.20)).verdict, "review");
  assert.equal(assess(scores(0.10, 0.20, 0.3)).verdict, "review");
  assert.equal(assess(scores(0.91, 0.20, 1, 0.3)).verdict, "review");
});

test("changing the block threshold can move the same score into review", () => {
  assert.equal(assess(scores(0.84, 0.28)).verdict, "block");
  assert.equal(assess(scores(0.84, 0.28), { ...DEFAULT_THRESHOLDS, block: 0.9 }).verdict, "review");
});
