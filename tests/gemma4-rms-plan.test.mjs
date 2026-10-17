import test from "node:test";
import assert from "node:assert/strict";
import { gemma4RmsScaleNextPlan } from "../js/src/gpu-gemma4-rms-plan.js";
import { GpuGemma4 } from "../js/src/gpu-gemma4.js";

function fixture() {
  const globals = {}, x = { size: 256 }, h = { size: 256 };
  const candidate = { nextName: "l.1.attn_norm", width: 16, eps: 1e-6, globals, residual: x, output: h, fusion: { pipeline: {}, group: {}, dispatch() {} } };
  const scale = { label: "scale" }, post = { rmsScaleNext: candidate, rmsPartner: scale };
  const next = { label: "rms.attn", rmsInput: { name: candidate.nextName, width: 16, eps: 1e-6, mode: 1, globals, input: x, output: h } };
  const ple = [{ label: "mm.ple.proj" }, { label: "ple.embed.host" }, { label: "ple.combine" }, { kind: "copy-ple-vector" }];
  return { post, scale, next, ple, candidate, x, h };
}

test("RMS planner crosses only the complete independent PLE prologue", () => {
  const { post, scale, next, ple, candidate } = fixture();
  assert.deepEqual(gemma4RmsScaleNextPlan([post, scale, ...ple, next]).get(post), { fusion: candidate.fusion, next });
  assert.equal(gemma4RmsScaleNextPlan([post, scale, ...ple.slice(0, 3), next]).size, 1);
  assert.equal(gemma4RmsScaleNextPlan([post, scale, next]).size, 1);
  assert.equal(gemma4RmsScaleNextPlan([post, scale, ...ple.slice(0, 2), next]).size, 0);
  for (const kind of ["GATHER", "copy-input", "copy-images", "copy-kv", "copy-kv-prefix", "copy-kv-join"]) {
    assert.equal(gemma4RmsScaleNextPlan([post, scale, { kind }, next]).size, 0, kind);
  }
});

test("RMS planner preserves graph, globals, norm and row-view boundaries", () => {
  const { post, scale, next, candidate, x, h } = fixture();
  assert.equal(gemma4RmsScaleNextPlan([post, scale]).size, 0);
  for (const patch of [{ globals: {} }, { mode: 2 }, { width: 32 }, { eps: 1e-5 }, { name: "l.2.attn_norm" },
    { input: { size: 256 } }, { output: { size: 256 } }, { input: { buffer: x, offset: 64, size: 192 } }]) {
    assert.equal(gemma4RmsScaleNextPlan([post, scale, { ...next, rmsInput: { ...next.rmsInput, ...patch } }]).size, 0);
  }
  candidate.residual = { buffer: x, offset: 64, size: 192 };
  candidate.output = { buffer: h, offset: 64, size: 192 };
  next.rmsInput.input = { buffer: x, offset: 64, size: 192 };
  next.rmsInput.output = { buffer: h, offset: 64, size: 192 };
  assert.equal(gemma4RmsScaleNextPlan([post, scale, next]).size, 1);
  next.label = "rms.final";
  candidate.nextName = next.rmsInput.name = "norm";
  assert.equal(gemma4RmsScaleNextPlan([post, scale, next]).size, 1);
});

test("RMS planner fails closed when identity metadata is absent or malformed", () => {
  for (const key of ["globals", "nextName", "width", "eps"]) {
    const { post, scale, next, candidate } = fixture();
    delete candidate[key];
    delete next.rmsInput[{ nextName: "name" }[key] || key];
    assert.equal(gemma4RmsScaleNextPlan([post, scale, next]).size, 0, key);
  }
  for (const patch of [{ width: 0 }, { width: 1.5 }, { eps: NaN }, { eps: 0 }, { globals: null }]) {
    const { post, scale, next, candidate } = fixture();
    Object.assign(candidate, patch);
    Object.assign(next.rmsInput, patch);
    assert.equal(gemma4RmsScaleNextPlan([post, scale, next]).size, 0);
  }
  for (const fusion of [undefined, {}, { pipeline: {}, group: {} }]) {
    const { post, scale, next, candidate } = fixture();
    candidate.fusion = fusion;
    assert.equal(gemma4RmsScaleNextPlan([post, scale, next]).size, 0);
  }
});

test("RMS encoder skips only the paired norm and preserves independent PLE work and fallback", () => {
  const { post, scale, next, ple, candidate } = fixture();
  const operation = (op, pipeline) => Object.assign(op, { kind: "RMS", pipeline, group: {}, dispatch() {} });
  operation(post, "post"); operation(scale, "scalar"); operation(next, "next");
  post.rmsFusion = operation({}, "scale-fusion");
  operation(candidate.fusion, "scale-next-fusion");
  const prologue = ple.slice(0, 3).map((op, i) => operation(op, `ple${i}`));
  const consumer = operation({}, "consumer");
  const ops = [post, scale, ...prologue, next, consumer];
  ops.rmsNextLinks = gemma4RmsScaleNextPlan(ops);
  const calls = [], pass = { setPipeline(value) { calls.push(value); }, setBindGroup() {}, end() {} };
  const encoder = { beginComputePass() { return pass; } };
  const value = { fuseRms: true, fuseRmsNext: true, rmsValidation: { enabled: true, nextEnabled: true }, lastRmsNextDispatches: 0 };
  GpuGemma4.prototype.encode.call(value, encoder, ops, 28, null);
  assert.deepEqual(calls, ["scale-next-fusion", "ple0", "ple1", "ple2", "consumer"]);
  assert.equal(value.lastRmsNextDispatches, 1);
  for (const patch of [{ fuseRmsNext: false }, { rmsValidation: { enabled: true, nextEnabled: false } }]) {
    calls.length = 0;
    GpuGemma4.prototype.encode.call({ ...value, ...patch }, encoder, ops, 28, null);
    assert.deepEqual(calls, ["scale-fusion", "ple0", "ple1", "ple2", "next", "consumer"]);
  }
  calls.length = 0;
  GpuGemma4.prototype.encode.call({ ...value, fuseRms: false }, encoder, ops, 28, null);
  assert.deepEqual(calls, ["post", "scalar", "ple0", "ple1", "ple2", "next", "consumer"]);
});

test("an RMS scalar pair split across command chunks applies the scalar once", () => {
  const calls = [], pass = { setPipeline(value) { calls.push(value); }, setBindGroup() {}, end() {} };
  const encoder = { beginComputePass() { return pass; } };
  let residual = 3;
  const scalar = { kind: "RESIDUAL", label: "scale", pipeline: "scalar", group: {}, dispatch() { residual *= 2; } };
  const post = { kind: "RMS", label: "rms.ple", pipeline: "post", group: {}, rmsPartner: scalar,
    dispatch() { residual += 1; }, rmsFusion: { pipeline: "scale-fusion", group: {}, dispatch() { residual = (residual + 1) * 2; } } };
  const value = { fuseRms: true, fuseRmsNext: false, rmsValidation: { enabled: true } };
  GpuGemma4.prototype.encode.call(value, encoder, [post], 300, null);
  GpuGemma4.prototype.encode.call(value, encoder, [scalar], 300, null);
  assert.equal(residual, 8);
  assert.deepEqual(calls, ["post", "scalar"]);
});
