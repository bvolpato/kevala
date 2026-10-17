import assert from "node:assert/strict";
import test from "node:test";

import { calibrateMatvec, compareMatvecOutputs, mapBeforeDeadline, matvecTimingMode, selectMatvecKernel } from "../js/src/gpu-matvec.js";

test("a stalled calibration map is cancelled at its deadline", { timeout: 200 }, async (t) => {
  const previous = globalThis.GPUMapMode;
  globalThis.GPUMapMode = { READ: 1 };
  t.after(() => previous === undefined ? delete globalThis.GPUMapMode : globalThis.GPUMapMode = previous);
  const keepAlive = setTimeout(() => {}, 1000);
  t.after(() => clearTimeout(keepAlive));
  let cancel;
  let unmaps = 0;
  const read = {
    mapAsync: () => new Promise((_, reject) => { cancel = reject; }),
    unmap() { unmaps++; cancel(new Error("map cancelled")); },
  };
  await assert.rejects(mapBeforeDeadline(read, 16, performance.now() + 5), (error) => error.code === "MATVEC_DEADLINE");
  assert.equal(unmaps, 1);
});

test("matvec selection requires a parity pass and three stable paired wins", () => {
  const stable = selectMatvecKernel({
    baseline: [10, 10.2, 9.8],
    candidate: [9.2, 9.3, 9.1],
    numericalCheck: { ok: true, maxAbs: 0.0001 },
  });
  assert.equal(stable.kernel, "matvec");
  assert.equal(stable.selected, true);
  assert.equal(stable.reason, "stable-matvec-win");

  const insufficient = selectMatvecKernel({ baseline: [10, 10], candidate: [8, 8], numericalCheck: { ok: true } });
  assert.equal(insufficient.selected, false);
  assert.equal(insufficient.reason, "insufficient-samples");

  const noisy = selectMatvecKernel({
    baseline: [10, 10, 10],
    candidate: [9, 10.1, 9],
    numericalCheck: { ok: true },
  });
  assert.equal(noisy.selected, false);
  assert.equal(noisy.reason, "noisy");

  const parityFailure = selectMatvecKernel({
    baseline: [10, 10, 10], candidate: [8, 8, 8], numericalCheck: { ok: false, maxAbs: 1 },
  });
  assert.equal(parityFailure.selected, false);
  assert.equal(parityFailure.reason, "numerical-guard-failed");
});

test("matvec output guard rejects non-finite and out-of-tolerance output", () => {
  assert.equal(compareMatvecOutputs([1, -2], [1.0001, -2.0001]).ok, true);
  const nonFinite = compareMatvecOutputs([1, 2], [1, Number.NaN]);
  assert.equal(nonFinite.ok, false);
  assert.equal(nonFinite.nonFinite, 1);
  const mismatch = compareMatvecOutputs([1], [1.01]);
  assert.equal(mismatch.ok, false);
  assert.equal(mismatch.violations, 1);
  assert.equal(compareMatvecOutputs([1, -2], [1, -2], { f16: true }).ok, true);
  assert.equal(compareMatvecOutputs([1], [1 + 2 ** -23], { f16: true }).ok, false);
});

test("matvec timing uses wall-clock pairs without timestamps only when clock resolution is usable", () => {
  assert.deepEqual(matvecTimingMode(new Set(["timestamp-query"]), null), {
    method: "paired-timestamp-query", repetitions: 4,
  });
  assert.deepEqual(matvecTimingMode(new Set(), 0.5), {
    method: "paired-wall-clock", repetitions: 64,
  });
  assert.deepEqual(matvecTimingMode(new Set(), null), {
    method: "baseline", repetitions: 0, reason: "no-wall-clock-resolution",
  });
  assert.deepEqual(matvecTimingMode(new Set(), 3), {
    method: "baseline", repetitions: 0, reason: "no-wall-clock-resolution",
  });
});

test("matvec tuner attempts no-timestamp wall-clock calibration and honors explicit kernel selection", async () => {
  const buffer = {};
  const weights = { tensors: new Map([["gate", {
    info: { name: "L.0.gate", dtype: "q8", shape: [64, 32] }, buf: buffer, sbuf: buffer,
  }]]) };
  const tuning = { pipelines: { generic: { layout: {} } }, selection: new Map() };
  const noTimestampDevice = {
    features: new Set(),
    createPipelineLayout: () => ({}),
    createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
    createComputePipelineAsync: async () => { throw new Error("expected pipeline failure"); },
  };
  const noTimestamps = await calibrateMatvec(noTimestampDevice, () => "unused", weights, tuning);
  assert.equal(noTimestamps.pipelines, null);
  assert.equal(noTimestamps.selection.size, 0);
  assert.equal(noTimestamps.diagnostics.method, matvecTimingMode(noTimestampDevice.features, noTimestamps.diagnostics.clockResolutionMs).method);
  assert.equal(noTimestamps.diagnostics.repetitions, noTimestamps.diagnostics.method === "paired-wall-clock" ? 64 : 0);
  assert.equal(noTimestamps.diagnostics.reason, noTimestamps.diagnostics.method === "paired-wall-clock" ? "pipeline-failed" : "no-wall-clock-resolution");

  const override = await calibrateMatvec({ features: new Set(["timestamp-query"]) }, () => "unused", weights, tuning, { kernel: "generic" });
  assert.equal(override.pipelines, null);
  assert.equal(override.selection.size, 0);
  assert.equal(override.diagnostics.reason, "kernel-override");
});
