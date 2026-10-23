import assert from "node:assert/strict";
import test from "node:test";
import { assessLayaParity } from "../dev/laya-parity-metrics.js";

function noulFixture(logits) {
  return {
    temperature: [1, 1, 1],
    temperature_by_options: { "noul:2": 1 },
    cases: [{
      id: "sample",
      state: "sample",
      questions: { q: { type: "noul", instructions: "sample?", criteria: { false: "no", true: "yes" } } },
      expect: { q: { logits, options: ["false: no", "true: yes"] } },
    }],
  };
}

function answers(noul) {
  return [{ caseId: "sample", answers: { q: { type: "noul", noul } } }];
}

test("Laya Q8 golden error above 0.024 passes when WASM and WebGPU agree", () => {
  const golden = noulFixture([2.557920217514038, 0.7013065814971924]);
  golden.temperature_by_options["noul:2"] = 1.983399510383606;
  const result = assessLayaParity(golden, answers(0.3057), answers(0.3057));
  assert.equal(result.ok, true);
  assert.equal(result.maxDp.wasmWebgpu, 0);
  assert.ok(Math.abs(result.maxDp.wasmGolden - 0.024006444195624588) < 1e-12);
  assert.equal(result.argmaxAgreement.wasmGolden.matched, 1);
  assert.equal(result.argmaxAgreement.webgpuGolden.matched, 1);
});

test("Laya golden softmax retains the original calibrated temperature clamp", () => {
  const golden = noulFixture([1, 0]);
  golden.temperature_by_options["noul:2"] = 0.1;
  const result = assessLayaParity(golden, answers(0.1192), answers(0.1192));
  assert.equal(result.ok, true);
  assert.ok(result.maxDp.wasmGolden < 0.0001);
});

test("Laya parity enforces the 5e-4 full-vector WASM/WebGPU tolerance", () => {
  const result = assessLayaParity(noulFixture([Math.log(9), 0]), answers(0.1), answers(0.1006));
  assert.ok(Math.abs(result.maxDp.wasmWebgpu - 0.0006) < 1e-12);
  assert.equal(result.maxDp.wasmGolden < 0.03, true);
  assert.equal(result.maxDp.webgpuGolden < 0.03, true);
  assert.equal(result.ok, false);
  assert.match(result.failures[0], /exceeds 0\.0005/);
});

test("Laya parity rejects an argmax change even when rounded vectors are within tolerance", () => {
  const result = assessLayaParity(noulFixture([0, 0]), answers(0.5), answers(0.5001));
  assert.ok(Math.abs(result.maxDp.wasmWebgpu - 0.0001) < 1e-12);
  assert.equal(result.argmaxAgreement.wasmWebgpu.matched, 0);
  assert.equal(result.ok, false);
  assert.match(result.failures.join(" "), /argmax agreement 0\/1/);
});

test("Laya parity checks Q8 golden accuracy independently of backend agreement", () => {
  const result = assessLayaParity(noulFixture([0, 0]), answers(0.531), answers(0.531));
  assert.equal(result.maxDp.wasmWebgpu, 0);
  assert.ok(Math.abs(result.maxDp.wasmGolden - 0.031) < 1e-12);
  assert.ok(Math.abs(result.maxDp.webgpuGolden - 0.031) < 1e-12);
  assert.equal(result.ok, false);
  assert.match(result.failures.join(" "), /Q8 golden maxDp .* exceeds 0\.03/);
});

test("Laya parity rejects non-finite probabilities and incomplete question answers", () => {
  assert.throws(() => assessLayaParity(noulFixture([0, 0]), answers(Number.NaN), answers(0.5)), /non-finite/);
  assert.throws(() => assessLayaParity(noulFixture([0, 0]), [], answers(0.5)), /WASM returned 0 cases for 1 fixtures/);
  assert.throws(() => assessLayaParity(noulFixture([0, 0]), [{ caseId: "sample", answers: {} }], answers(0.5)), /WASM answers do not match/);
});

test("Laya parity validates each choice option key and complete vector length", () => {
  const golden = {
    temperature: [1, 1, 1],
    temperature_by_options: { "choice:2": 1 },
    cases: [{
      id: "choice",
      state: "sample",
      questions: { q: { type: "choice", instructions: "choose", criteria: { allow: "yes", block: "no" } } },
      expect: { q: { logits: [1, 0], options: ["allow: yes", "block: no"] } },
    }],
  };
  const incomplete = [{ caseId: "choice", answers: { q: { type: "choice", choice: "allow", probabilities: { allow: 0.7 } } } }];
  assert.throws(() => assessLayaParity(golden, incomplete, incomplete), /probability keys .* do not match options/);
});

test("Laya choice-list criteria must match golden option IDs and order", () => {
  const golden = {
    temperature: [1, 1, 1],
    temperature_by_options: { "choice:2": 1 },
    cases: [{
      id: "choice-list",
      state: "sample",
      questions: { q: { type: "choice", instructions: "choose", criteria: ["left", "right"] } },
      expect: { q: { logits: [1, 0], options: ["allow", "block"] } },
    }],
  };
  const answers = [{ caseId: "choice-list", answers: { q: {
    type: "choice", choice: "allow", probabilities: { allow: 0.7311, block: 0.2689 },
  } } }];
  assert.throws(() => assessLayaParity(golden, answers, answers), /options do not match criteria keys and order/);
});

test("Laya choice-list criteria preserve first occurrence order after deduplication", () => {
  const golden = {
    temperature: [1, 1, 1],
    temperature_by_options: { "choice:2": 1 },
    cases: [{
      id: "choice-list",
      state: "sample",
      questions: { q: { type: "choice", instructions: "choose", criteria: ["billing", "billing", "technical"] } },
      expect: { q: { logits: [1, 0], options: ["billing", "technical"] } },
    }],
  };
  const answers = [{ caseId: "choice-list", answers: { q: {
    type: "choice", choice: "billing", probabilities: { billing: 0.7311, technical: 0.2689 },
  } } }];
  assert.equal(assessLayaParity(golden, answers, answers).ok, true);
});
