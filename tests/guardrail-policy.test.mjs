import assert from "node:assert/strict";
import { test } from "node:test";
import { assess, DEFAULT_THRESHOLDS } from "../app/views/guardrail-policy.js";

const scores = (override, educational, direct = 0, overrideAct = 1, contextAct = 1, intentAct = 1) => ({
  instruction_override: { noul: override, action: { act_probability: overrideAct } },
  educational_context: { noul: educational, action: { act_probability: contextAct } },
  request_intent: { probabilities: { direct }, action: { act_probability: intentAct } },
});

test("an ordinary request is allowed even when analysis context is uncertain", () => {
  assert.equal(assess(scores(0.20, 0.66)).verdict, "allow");
});

test("clear overrides block, while a quoted attack for analysis goes to review", () => {
  assert.equal(assess(scores(0.91, 0.23, 0.90)).verdict, "block");
  assert.equal(assess(scores(0.96, 0.89, 0.46)).verdict, "review");
});

test("Laya's DAN sample blocks despite an inflated analysis score", () => {
  assert.equal(assess(scores(1, 0.9377, 0.7312)).verdict, "block");
  assert.equal(assess(scores(0.9627, 0.8452, 0.2324)).verdict, "review");
  assert.equal(assess(scores(0.1456, 0.1241, 0.573)).verdict, "allow");
  assert.equal(assess(scores(0.95, 0.69, 0.69)).verdict, "block");
});

test("uncertain scores and a low act probability require review", () => {
  assert.equal(assess(scores(0.55, 0.20)).verdict, "review");
  assert.equal(assess(scores(0.10, 0.20, 0, 0.3)).verdict, "review");
  assert.equal(assess(scores(0.91, 0.20, 0, 1, 0.3)).verdict, "review");
  assert.equal(assess(scores(0.91, 0.90, 0.80, 1, 1, 0.3)).verdict, "review");
});

test("changing the block threshold can move the same score into review", () => {
  assert.equal(assess(scores(0.84, 0.28)).verdict, "block");
  assert.equal(assess(scores(0.84, 0.28), { ...DEFAULT_THRESHOLDS, block: 0.9 }).verdict, "review");
  assert.equal(assess(scores(1, 0.94, 0.73), { ...DEFAULT_THRESHOLDS, direct: 0.8 }).verdict, "review");
});
