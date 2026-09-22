import assert from "node:assert/strict";
import test from "node:test";

// The session module reads the page URL when the shared UI helpers are imported.
globalThis.location = { search: "" };
const { cpuReason } = await import("../app/ui.js");

test("Firefox memory failures recommend freeing GPU memory, not enabling WebGPU", (t) => {
  t.mock.getter(globalThis, "navigator", () => ({ gpu: {}, userAgent: "Firefox/152.0 Linux" }));
  const reason = cpuReason({ backend: "wasm-simd", gpuUnavailable: "Not enough memory left." });
  assert.match(reason, /Close other GPU-heavy tabs or applications/);
  assert.doesNotMatch(reason, /about:config|ignore-blocklist/);
});

test("only unavailable WebGPU gets Firefox enablement guidance", (t) => {
  t.mock.getter(globalThis, "navigator", () => ({ userAgent: "Firefox/152.0 Linux" }));
  assert.match(cpuReason({ backend: "wasm-simd", gpuUnavailable: "this browser has no WebGPU in a worker (navigator.gpu is missing)" }), /about:support/);
  assert.equal(cpuReason({ backend: "webgpu", gpuUnavailable: null }), "");
  assert.equal(cpuReason({ backend: "wasm-simd", gpuUnavailable: null }), "");
});
