import assert from "node:assert/strict";
import test from "node:test";
import { calibrateGemma4Bn32, compareGemma4Bn32Outputs } from "../js/src/gpu-gemma4-bn32.js";
import { mmSplits } from "../js/src/gpu.js";

const CONFIG = { kernel: "generic", rows: 2, row56: false, splitTokens: 64 };
const PLAN = { logicalRows: 66, staticRows: 38, activeRows: 28, layerWidths: Array(15).fill(256) };
const BASELINE = { f16: false, groups: 1, splitTarget: 256, layout: {}, reduceLayout: {}, mm: { 2: { kind: "baseline" } }, reduce: { 2: { kind: "reduce" } } };
const weightsFor = (entries = [["l.0.gate", 17, 1056]]) => ({ tensors: new Map(entries.map(([name, N, K]) => [name, { info: { dtype: "q8", block: 32, shape: [N, K] }, buf: {}, sbuf: {} }])) });
const optionsFor = (weights) => ({ plan: PLAN, sourcePipelines: { generic: BASELINE }, retile: { selection: new Map([...weights.tensors.values()].map(({ info }) => [info.shape.join(":"), CONFIG])) } });

function gpu(t, options = {}) {
  for (const [name, value] of Object.entries({ GPUBufferUsage: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, MAP_READ: 16, QUERY_RESOLVE: 32 }, GPUMapMode: { READ: 1 }, GPUShaderStage: { COMPUTE: 1 } })) {
    const previous = globalThis[name]; globalThis[name] = value;
    t.after(() => previous === undefined ? delete globalThis[name] : globalThis[name] = previous);
  }
  const allocated = [], timed = new Set(), dispatches = [];
  let lastN, errorFired = false;
  const device = {
    features: new Set(["timestamp-query"]), limits: { maxBufferSize: 1 << 27, maxStorageBufferBindingSize: 1 << 27 },
    createBuffer({ size, label }) { const value = { data: new ArrayBuffer(size), size, label, destroyed: 0, destroy() { this.destroyed++; }, async mapAsync() {}, getMappedRange(offset, length) { return this.data.slice(offset, offset + length); }, unmap() {} }; allocated.push(value); return value; },
    createBindGroupLayout: (spec) => spec, createPipelineLayout: (spec) => spec,
    createShaderModule: (spec) => ({ ...spec, async getCompilationInfo() { return { messages: [] }; } }),
    async createComputePipelineAsync({ compute }) { return { kind: compute.module.code }; },
    createBindGroup: ({ entries }) => entries.map(({ resource }) => resource.buffer),
    createQuerySet: () => ({ stamps: new BigInt64Array(12), destroy() {} }),
    pushErrorScope() {}, async popErrorScope() {
      if (options.expireN !== undefined && lastN === options.expireN) options.expire?.();
      if (options.gpuErrorN !== undefined && lastN === options.gpuErrorN && !errorFired) { errorFired = true; return new Error("synthetic GPU validation failure"); }
      return null;
    },
    queue: {
      writeBuffer(buffer, offset, values) { new Uint8Array(buffer.data).set(new Uint8Array(values instanceof ArrayBuffer ? values : values.buffer, values.byteOffset || 0, values.byteLength), offset); },
      submit(encoders) { for (const encoder of encoders) for (const command of encoder) command(); },
    },
    createCommandEncoder() {
      const commands = [];
      return {
        beginComputePass(spec = {}) {
          const work = []; let activePipeline, group, N, candidate = false;
          return {
            setPipeline(value) { activePipeline = value; }, setBindGroup(_index, value) { group = value; },
            dispatchWorkgroups(...counts) {
              const pipe = activePipeline, buffers = group;
              if (pipe.kind === "reduce") return;
              candidate ||= pipe.kind === "candidate";
              work.push(() => {
                const [T, splitT] = new Uint32Array(buffers[0].data), [cols, K] = new Uint32Array(buffers[1].data); N = cols; lastN = cols;
                dispatches.push({ kind: pipe.kind, N, counts, globals: [...new Uint32Array(buffers[0].data)] });
                const output = new Float32Array(buffers[6].data), partial = new Float32Array(buffers[7].data), splits = mmSplits(splitT, cols, K, 256, 32, 64);
                for (let i = 0; i < T * cols; i++) output[i] = i === 0 ? -0 : i / 128;
                if (splits > 1) for (let i = 0; i < splits * T * cols; i++) partial[i] = i === 0 ? -0 : i / 256;
                if (pipe.kind === "candidate" && cols === options.corruptN && timed.has(cols)) {
                  if (options.corruptTail) output[T * cols] = 0;
                  else if (splits > 1) partial[0] = 0;
                  else output[0] = 0;
                }
              });
            },
            end() { commands.push(() => { for (const run of work) run(); if (spec.timestampWrites) { const s = spec.timestampWrites; s.querySet.stamps[s.beginningOfPassWriteIndex] = 1000n; s.querySet.stamps[s.endOfPassWriteIndex] = candidate ? 3_201_000n : 4_001_000n; timed.add(N); } }); },
          };
        },
        copyBufferToBuffer(src, start, dst, offset, size) { commands.push(() => new Uint8Array(dst.data).set(new Uint8Array(src.data, start, size), offset)); },
        resolveQuerySet(query, start, count, target, offset) { commands.push(() => new BigInt64Array(target.data, offset, count).set(query.stamps.subarray(start, start + count))); },
        finish: () => commands,
      };
    },
  };
  return { device, allocated, dispatches };
}

test("BN32 comparison preserves signed zeros and rejects low-bit or nonfinite changes", () => {
  assert.equal(compareGemma4Bn32Outputs(Float32Array.of(-0, 2), Float32Array.of(-0, 2)).ok, true);
  assert.equal(compareGemma4Bn32Outputs(Float32Array.of(-0), Float32Array.of(0)).ok, false);
  assert.equal(compareGemma4Bn32Outputs(Float32Array.of(1), Float32Array.of(1 + 2 ** -23)).ok, false);
  assert.equal(compareGemma4Bn32Outputs(Float32Array.of(NaN), Float32Array.of(NaN)).ok, false);
  assert.equal(compareGemma4Bn32Outputs(Float32Array.of(1), Float32Array.of(1, 2)).ok, false);
});

test("BN32 calibration skips F16, unsupported retile policies, and suffix plans without compiling", async () => {
  const weights = weightsFor(), base = optionsFor(weights), source = () => { throw new Error("unexpected compilation"); };
  for (const override of [{ enabled: false }, { plan: { ...PLAN, activeRows: 97 } }, { sourcePipelines: { generic: { ...BASELINE, f16: true } } },
    { retile: { selection: new Map([["17:1056", { ...CONFIG, rows: 4 }]]) } }, { retile: { selection: new Map([["17:1056", { ...CONFIG, kernel: "wide" }]]) } },
    { retile: { selection: new Map([["17:1056", { ...CONFIG, row56: true }]]) } }]) {
    const result = await calibrateGemma4Bn32({ features: new Set(["timestamp-query"]) }, source, weights, { ...base, ...override });
    assert.equal(result.selection.size, 0); assert.equal(result.pipeline, null);
  }
  assert.equal((await calibrateGemma4Bn32({ features: new Set() }, source, weights, base)).diagnostics.reason, "no-timestamp-query");
});

test("oversized scratch is rejected before allocating calibration buffers", async (t) => {
  const { device, allocated } = gpu(t), weights = weightsFor([["l.0.gate", 1 << 20, 1056]]);
  const result = await calibrateGemma4Bn32(device, () => "candidate", weights, optionsFor(weights));
  assert.equal(result.selection.size, 0);
  assert.equal(result.diagnostics.points[0].reason, "scratch-limit");
  assert.equal(allocated.length, 0);
});

test("BN32 selects stable exact shapes, preserves globals/splits, prioritizes MLP, and releases scratch", async (t) => {
  const { device, allocated, dispatches } = gpu(t);
  const weights = weightsFor([["l.0.down", 33, 1056], ["l.0.gate", 17, 1056], ["l.15.gate", 19, 1056]]), options = optionsFor(weights);
  const result = await calibrateGemma4Bn32(device, () => "candidate", weights, options);
  assert.deepEqual([...result.selection], ["17:1056", "33:1056"]);
  assert.deepEqual(result.diagnostics.points.map(({ name }) => name), ["l.0.gate", "l.0.down"]);
  assert.ok(result.diagnostics.points.every((point) => point.numericalCheck.ok && point.afterTimingCheck.ok && point.scratchBytes <= 32 * 1024 * 1024));
  assert.ok(dispatches.every(({ globals }) => globals.join() === "28,64,66,0"));
  assert.deepEqual(dispatches.find(({ kind, N }) => kind === "baseline" && N === 33).counts, [1, 1, 8]);
  assert.deepEqual(dispatches.find(({ kind, N }) => kind === "candidate" && N === 33).counts, [2, 1, 8]);
  assert.ok(allocated.every(({ destroyed }) => destroyed === 1));
  assert.deepEqual(structuredClone(result.diagnostics), result.diagnostics);
});

test("post-timing partial, tail, or GPU failures clear previously selected shapes", async (t) => {
  for (const failure of [{ corruptN: 33 }, { corruptN: 33, corruptTail: true }, { gpuErrorN: 33 }]) {
    const { device, allocated } = gpu(t, failure), weights = weightsFor([["l.0.gate", 17, 1056], ["l.0.down", 33, 1056]]);
    const result = await calibrateGemma4Bn32(device, () => "candidate", weights, optionsFor(weights));
    assert.equal(result.selection.size, 0);
    assert.ok(result.diagnostics.points.every((point) => !point.selected));
    assert.ok(allocated.every(({ destroyed }) => destroyed === 1));
    assert.match(result.diagnostics.reason, /guard-failed|calibration-failed/);
  }
});

test("a deadline preserves completed checks and the point cap excludes later shapes", async (t) => {
  const previous = globalThis.performance;
  let clock = 0;
  Object.defineProperty(globalThis, "performance", { configurable: true, value: { now: () => clock } });
  t.after(() => Object.defineProperty(globalThis, "performance", { configurable: true, value: previous }));
  const { device } = gpu(t, { expireN: 17, expire() { clock = 1001; } });
  const weights = weightsFor([["l.0.gate", 17, 1056], ["l.0.down", 33, 1056]]);
  const result = await calibrateGemma4Bn32(device, () => "candidate", weights, optionsFor(weights));
  assert.deepEqual([...result.selection], ["17:1056"]); assert.equal(result.diagnostics.truncated, true); assert.equal(result.diagnostics.reason, "deadline");
  clock = 0;
  const capped = gpu(t), many = weightsFor(Array.from({ length: 10 }, (_, i) => [`l.${i}.gate`, 17 + i, 32]));
  assert.equal((await calibrateGemma4Bn32(capped.device, () => "candidate", many, optionsFor(many))).diagnostics.points.length, 8);
});
