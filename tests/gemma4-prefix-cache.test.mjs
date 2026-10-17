import assert from "node:assert/strict";
import test from "node:test";
import { Gemma4TextPrefixCache, prefixRowView, textPrefixCachePlan, TEXT_PREFIX_CACHE_LIMIT } from "../js/src/gpu-gemma4-prefix-cache.js";
import { GpuGemma4 } from "../js/src/gpu-gemma4.js";

function fixture(t) {
  const previous = globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage = { COPY_SRC: 1, COPY_DST: 2 };
  t.after(() => previous === undefined ? delete globalThis.GPUBufferUsage : globalThis.GPUBufferUsage = previous);
  const allocated = [];
  const device = {
    limits: { maxBufferSize: 1 << 26, maxStorageBufferBindingSize: 1 << 26, minStorageBufferOffsetAlignment: 256 },
    createBuffer({ size }) {
      const buffer = { size, destroyed: 0, destroy() { this.destroyed++; } };
      allocated.push(buffer);
      return buffer;
    },
  };
  const cfg = { layers: 3, sharedStart: 2, kvSharedLayers: 1, causal: 1, hidden: 1536, intermediate: 6144, pleDim: 256, heads: 8, kvHeads: 1, localHeadDim: 256, globalHeadDim: 512, full: [false, true, true] };
  const sequences = [1, 2].map((suffix) => ({ ids: new Uint32Array(89 + suffix).fill(10), imageSpans: [{ start: 38, len: 35 }] }));
  const plan = (options = {}, inputs = sequences) => textPrefixCachePlan({ device, cfg, policy: "f16-r3", ...options }, inputs, 89);
  const cache = new Gemma4TextPrefixCache(device);
  t.after(() => cache.dispose());
  const capture = (lease) => { for (const layer of lease.kv.keys()) assert.equal(cache.recordCapture(lease, layer), true); };
  return { device, cfg, sequences, allocated, plan, cache, capture };
}

test("partial captures and failed requests cannot become cache hits", (t) => {
  const { cache, plan, capture, allocated } = fixture(t);
  const partial = cache.begin(plan());
  assert.equal(cache.recordCapture(partial, 0), true);
  assert.equal(cache.finish(partial, true), false);
  assert.ok(allocated.every((buffer) => buffer.destroyed === 1));
  const failed = cache.begin(plan());
  assert.equal(failed.hit, false);
  capture(failed);
  assert.equal(cache.finish(failed, false), false);
  assert.ok(allocated.every((buffer) => buffer.destroyed === 1));
  const complete = cache.begin(plan());
  assert.equal(complete.hit, false);
  capture(complete);
  assert.equal(cache.finish(complete, true), true);
  const warm = cache.begin(plan());
  assert.equal(warm.hit, true);
  assert.equal(warm.kv, complete.kv);
  capture(warm);
  assert.equal(cache.finish(warm, true), true);
});

test("invalidating an active lease retains its buffers until the request finishes", (t) => {
  const { cache, plan, capture, allocated } = fixture(t);
  const cold = cache.begin(plan());
  capture(cold);
  assert.equal(cache.finish(cold, true), true);
  const active = cache.begin(plan());
  cache.invalidate();
  assert.ok(allocated.every((buffer) => buffer.destroyed === 0));
  assert.equal(cache.recordCapture(active, 0), false);
  assert.equal(cache.finish(active, true), false);
  assert.ok(allocated.every((buffer) => buffer.destroyed === 1));
  assert.equal(cache.finish(active, false), false);
  assert.ok(allocated.every((buffer) => buffer.destroyed === 1));
  const next = cache.begin(plan());
  assert.equal(next.hit, false);
  cache.finish(next, false);
});

test("changed static tokens, numeric policy, or device cannot reuse retained K/V", (t) => {
  const { cache, plan, capture, sequences, device, allocated } = fixture(t);
  const changed = sequences.map((sequence) => ({ ...sequence, ids: sequence.ids.slice() }));
  changed.forEach((sequence) => { sequence.ids[0] = 11; });
  for (const candidate of [plan(), plan({}, changed), plan({ policy: "f32-r3" }, changed)]) {
    const previous = allocated.slice();
    const lease = cache.begin(candidate);
    assert.equal(lease.hit, false);
    assert.ok(previous.every((buffer) => buffer.destroyed === 1));
    capture(lease);
    assert.equal(cache.finish(lease, true), true);
  }
  assert.equal(cache.begin(plan({ device: { ...device } })), null);
  assert.ok(allocated.every((buffer) => buffer.destroyed === 1));
});

test("independent calibration policies reuse tuning keys while numeric policies recapture K/V", (t) => {
  const { plan, cache, capture, cfg, sequences, allocated } = fixture(t);
  const original = plan();
  assert.equal(original.calibrationKey, original.key);
  const off = plan({ policy: "ple-off", calibrationPolicy: "same-matrix-policy" });
  const on = plan({ policy: "ple-on", calibrationPolicy: "same-matrix-policy" });
  assert.notEqual(off.key, on.key);
  assert.equal(off.calibrationKey, on.calibrationKey);
  assert.equal(Object.isFrozen(on), true);
  const coldOff = cache.begin(off); capture(coldOff); assert.equal(cache.finish(coldOff, true), true);
  const previous = allocated.slice(), coldOn = cache.begin(on);
  assert.equal(coldOn.hit, false);
  assert.ok(previous.every((buffer) => buffer.destroyed === 1));
  capture(coldOn); assert.equal(cache.finish(coldOn, true), true);
  assert.notEqual(plan({ policy: "ple-on", calibrationPolicy: "changed-rms-policy" }).calibrationKey, on.calibrationKey);
  assert.notEqual(plan({ cfg: { ...cfg, window: 128 }, policy: "ple-on", calibrationPolicy: "same-matrix-policy" }).calibrationKey, on.calibrationKey);
  const changed = sequences.map((sequence) => ({ ...sequence, ids: sequence.ids.slice() }));
  changed.forEach((sequence) => { sequence.ids[0]++; });
  assert.notEqual(plan({ policy: "ple-on", calibrationPolicy: "same-matrix-policy" }, changed).calibrationKey, on.calibrationKey);
  for (const calibrationPolicy of [null, "", 1]) assert.equal(plan({ calibrationPolicy }), null);
});

test("cache plans and row bindings reject excess memory and unaligned or undersized views", (t) => {
  const { plan, cfg, device, allocated } = fixture(t);
  const valid = plan();
  assert.equal(plan({ maxBytes: valid.bytes - 1 }), null);
  assert.equal(plan({ maxBytes: TEXT_PREFIX_CACHE_LIMIT + 1 }), null);
  assert.equal(plan({ cfg: { ...cfg, pleDim: 252 } }), null);
  assert.equal(plan({ cfg: { ...cfg, causal: 0 } }), null);
  assert.equal(plan({ device: { ...device, limits: { ...device.limits, maxBufferSize: 1024 } } }), null);
  const bounded = new Gemma4TextPrefixCache(device, { maxBytes: valid.bytes - 1 });
  assert.equal(bounded.begin(valid), null);
  assert.equal(allocated.length, 0);
  assert.throws(() => new Gemma4TextPrefixCache(device, { maxBytes: TEXT_PREFIX_CACHE_LIMIT + 1 }), RangeError);
  const buffer = { size: 89 * cfg.hidden * 4 };
  assert.deepEqual(prefixRowView(valid, buffer, cfg.hidden), { buffer, offset: 38 * cfg.hidden * 4, size: 51 * cfg.hidden * 4 });
  assert.equal(prefixRowView(valid, { size: buffer.size - 1 }, cfg.hidden), null);
  assert.equal(prefixRowView(valid, buffer, 252), null);
});

test("late GPU validation errors invalidate cached PLE inputs after successful readback", async () => {
  for (const shared of [false, true]) {
    let pops = 0, pleClears = 0;
    const device = { pushErrorScope() {}, async popErrorScope() { return ++pops === 3 ? { message: "injected delayed validation failure" } : null; } };
    const value = {
      runtime: { device }, prefixCaptureOps: [], cacheTextPrefix: true,
      async forwardSharedRows() { return new Float32Array([1]); },
      async forwardRows() { return new Float32Array([1]); },
      clearTextPrefixCache() {}, clearPleVectorCache() { pleClears++; },
    };
    const rows = {};
    const sequences = [10, 11].map((suffix) => ({ ids: new Uint32Array([1, 258880, 2, suffix]), imageSpans: [{ start: 1, len: 1, rows }] }));
    const run = shared ? GpuGemma4.prototype.forwardSharedPrefix.call(value, sequences) : GpuGemma4.prototype.forward.call(value, sequences[0].ids);
    await assert.rejects(run, /injected delayed validation failure/);
    assert.equal(pleClears, 1);
  }
});
