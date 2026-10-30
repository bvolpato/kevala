import assert from "node:assert/strict";
import test from "node:test";
import { GpuKev } from "../js/src/gpu-kev.js";

test("Qwen selects projection pipelines by dtype and limits calibrated matvec to one row", () => {
  const gpu = Object.create(GpuKev.prototype);
  gpu.gpuKernel = "auto";
  gpu.precisionTuning = new Map();
  gpu.precisionMatvec = new Map();
  const ops = [];
  for (const dtype of ["q8", "bf16", "f32"]) {
    const generic = { mm: [null, `${dtype}.generic`], reduce: [null, `${dtype}.reduce`], splitTarget: 256, groups: 1 };
    const wide = { ...generic, mm: [null, `${dtype}.wide`] };
    gpu.precisionTuning.set(dtype, { pipelines: { generic, wide }, selection: new Map([["64:32:r1", "wide"]]) });
    gpu.precisionMatvec.set(dtype, { pipelines: { pipeline: `${dtype}.matvec`, columns: 4 }, selection: new Set(["64:32"]) });
    ops.push({ k: "mm", dtype, N: 64, K: 32, group: {}, reduce: {}, pipes: generic });
  }
  const encode = (tokens) => {
    const pipelines = [];
    const pass = { setPipeline: (value) => pipelines.push(value), setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
    gpu.encode({ beginComputePass: () => pass }, ops, tokens, 1);
    return pipelines;
  };
  assert.deepEqual(encode(1), ["q8.matvec", "bf16.matvec", "f32.matvec"]);
  assert.deepEqual(encode(2), ["q8.wide", "bf16.wide", "f32.wide"]);
  gpu.precisionMatvec.get("bf16").selection.clear();
  assert.deepEqual(encode(1), ["q8.matvec", "bf16.wide", "f32.matvec"]);
  gpu.gpuKernel = "generic";
  assert.deepEqual(encode(1), ["q8.generic", "bf16.generic", "f32.generic"]);
  gpu.gpuKernel = "wide";
  assert.deepEqual(encode(1), ["q8.wide", "bf16.wide", "f32.wide"]);
});
