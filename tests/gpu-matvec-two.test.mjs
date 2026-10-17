import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { calibrateMatvecTwo, compareMatvecTwoOutputs, encodeMatvecTwo, selectMatvecTwoKernel } from "../js/src/gpu-matvec-two.js";

const previous = Object.fromEntries(["GPUBufferUsage", "GPUMapMode", "GPUShaderStage"].map((key) => [key, globalThis[key]]));
before(() => {
  globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2, STORAGE: 4, COPY_SRC: 8, MAP_READ: 16, QUERY_RESOLVE: 32 };
  globalThis.GPUMapMode = { READ: 1 };
  globalThis.GPUShaderStage = { COMPUTE: 1 };
});
after(() => { for (const [key, value] of Object.entries(previous)) if (value === undefined) delete globalThis[key]; else globalThis[key] = value; });

test("two-row parity compares all F32 bits and rejects poison, signed zero and length changes", () => {
  assert.equal(compareMatvecTwoOutputs([0, -0, 1, -2], [0, -0, 1, -2]).ok, true);
  const zero = compareMatvecTwoOutputs([0], [-0]);
  assert.equal(zero.ok, false); assert.equal(zero.bitDifferences, 1); assert.equal(zero.maxAbs, 0);
  assert.equal(compareMatvecTwoOutputs([NaN], [NaN]).ok, false);
  assert.equal(compareMatvecTwoOutputs([Infinity], [Infinity]).ok, false);
  assert.equal(compareMatvecTwoOutputs([1], [1 + 2 ** -23]).ok, false);
  assert.equal(compareMatvecTwoOutputs([1, 2], [1]).lengthMismatch, true);
});

test("two-row selection needs exact parity and three stable timestamp pairs", () => {
  const numericalCheck = { ok: true };
  assert.equal(selectMatvecTwoKernel({ baseline: [10, 10, 10], candidate: [9, 9, 9], numericalCheck }).selected, true);
  assert.equal(selectMatvecTwoKernel({ baseline: [10, 10, 10], candidate: [9.6, 9.6, 9.6], numericalCheck }).selected, false);
  assert.equal(selectMatvecTwoKernel({ baseline: [10, 10, 10], candidate: [9, 10.1, 9], numericalCheck }).selected, false);
  assert.equal(selectMatvecTwoKernel({ baseline: [10, 10], candidate: [8, 8], numericalCheck }).reason, "invalid-samples");
  assert.equal(selectMatvecTwoKernel({ baseline: [10, 10, 10], candidate: [0, 8, 8], numericalCheck }).reason, "invalid-samples");
  assert.equal(selectMatvecTwoKernel({ baseline: [10, 10, 10], candidate: [8, 8, 8], numericalCheck: { ok: false } }).selected, false);
});

test("two-row dispatch reuses the op group and rejects other row counts before encoding", () => {
  const calls = [], op = { N: 65, group: {} }, pipes = { pipeline: {}, columns: 4 };
  const pass = { setPipeline: (...args) => calls.push(["pipeline", ...args]), setBindGroup: (...args) => calls.push(["group", ...args]), dispatchWorkgroups: (...args) => calls.push(["dispatch", ...args]) };
  assert.throws(() => encodeMatvecTwo(pass, pipes, op, 1), /requires T=2/); assert.equal(calls.length, 0);
  encodeMatvecTwo(pass, pipes, op, 2);
  assert.deepEqual(calls, [["pipeline", pipes.pipeline], ["group", 0, op.group], ["dispatch", 17, 1]]);
  calls.length = 0; encodeMatvecTwo(pass, { ...pipes, columns: 16 }, op, 2);
  assert.deepEqual(calls.at(-1), ["dispatch", 5, 1]);
});

test("subgroup dispatch refines only its selected F32 shape and supports disabling the refinement", () => {
  const base = {}, refined = {}, op = { N: 65, K: 32, group: {} };
  const pipes = { pipeline: base, f16: false, columns: 4, subgroup: { pipeline: refined, enabled: true, selection: new Set(["65:32"]) } };
  let chosen;
  const pass = { setPipeline(value) { chosen = value; }, setBindGroup() {}, dispatchWorkgroups() {} };
  encodeMatvecTwo(pass, pipes, op, 2); assert.equal(chosen, refined);
  pipes.subgroup.enabled = false; encodeMatvecTwo(pass, pipes, op, 2); assert.equal(chosen, base);
  pipes.subgroup.enabled = true;
  for (const change of [{ K: 64 }, { sourcePolicy: true }]) { encodeMatvecTwo(pass, pipes, { ...op, ...change }, 2); assert.equal(chosen, base); }
  encodeMatvecTwo(pass, { ...pipes, f16: true }, op, 2); assert.equal(chosen, base);
  assert.throws(() => encodeMatvecTwo(pass, pipes, op, 5), /requires T=2/);
});

// This device records the calibration protocol and lifetimes. It does not emulate GPU arithmetic.
function mockDevice(options = {}) {
  const buffers = [], querySets = [], sources = [], inputs = [];
  const device = {
    features: new Set(["timestamp-query", ...(options.f16 ? ["shader-f16"] : []), ...(options.subgroups ? ["subgroups"] : [])]),
    limits: { maxStorageBufferBindingSize: 64 * 1024 * 1024, maxBufferSize: 64 * 1024 * 1024, maxComputeWorkgroupsPerDimension: 65535 },
    maps: 0, scopeFailure: null,
    pushErrorScope() {}, async popErrorScope() { const error = device.scopeFailure; device.scopeFailure = null; return error; },
    createBuffer({ size }) {
      const buffer = { data: new ArrayBuffer(size), destroyed: false, mapped: false,
        async mapAsync() { device.maps++; options.onMap?.(device); if (options.mapFailure === device.maps) throw new Error("map failure"); buffer.mapped = true; },
        getMappedRange(offset = 0, bytes = size) { assert.equal(buffer.mapped, true); return buffer.data.slice(offset, offset + bytes); },
        unmap() { buffer.mapped = false; }, destroy() { buffer.destroyed = true; buffer.mapped = false; },
      };
      buffers.push(buffer); return buffer;
    },
    createBindGroupLayout: (descriptor) => descriptor,
    createPipelineLayout: (descriptor) => descriptor,
    createBindGroup: ({ layout, entries }) => ({ layout, buffers: entries.map((entry) => entry.resource.buffer) }),
    createShaderModule({ code }) { sources.push(code); return { getCompilationInfo: async () => ({ messages: [] }) }; },
    async createComputePipelineAsync({ label }) { if (options.pipelineFailure || options.subgroupPipelineFailure && label.endsWith("_subgroup")) throw new Error("pipeline failure"); return { label }; },
    createQuerySet({ count }) { const query = { values: Array(count).fill(0n), destroyed: false, destroy() { query.destroyed = true; } }; querySets.push(query); return query; },
    createCommandEncoder() {
      const commands = [];
      return {
        beginComputePass(descriptor = {}) {
          const dispatches = []; let pipe, group;
          return {
            setPipeline(value) { pipe = value; }, setBindGroup(_index, value) { group = value; },
            dispatchWorkgroups(x, y, z) { dispatches.push({ pipe, group, x, y, z }); },
            end() {
              commands.push(() => {
                for (const { pipe, group, x, y } of dispatches) {
                  const [globals, params] = group.buffers, N = new Uint32Array(params.data)[0], T = new Uint32Array(globals.data)[0];
                  assert.equal(T, 2);
                  const candidate = pipe.label.startsWith("matvec_two") || pipe.label.startsWith("matvec_gelu_two");
                  const subgroup = pipe.label.endsWith("_subgroup");
                  if (candidate) { assert.equal(y, 1); assert.equal(x, Math.ceil(N / (options.f16 ? 16 : 4))); }
                  const plain = pipe.label === "baseline-plain" || pipe.label.startsWith("matvec_two");
                  const separateGelu = pipe.label === "baseline-separate-gelu";
                  const out = group.buffers[plain ? 6 : separateGelu ? 4 : 7], values = new Float32Array(out.data);
                  if (!separateGelu) inputs.push(new Float32Array(group.buffers[2].data).slice());
                  for (let i = 0; i < T * N; i++) values[i] = i === 0 ? -0 : 0.011 * ((i % 19) - 9);
                  if (candidate && options.corrupt?.({ N, kind: plain ? "plain" : "gelu" })) new Uint32Array(out.data)[0] = 0;
                  if (candidate && options.tailFailure) values[T * N] = 0;
                  if (subgroup && options.subgroupCorrupt?.({ N, kind: plain ? "plain" : "gelu" })) new Uint32Array(out.data)[0] = 0;
                  if (subgroup && options.subgroupTailFailure) values[T * N] = 0;
                }
                const writes = descriptor.timestampWrites;
                if (writes) {
                  const { pipe, group } = dispatches[0], N = new Uint32Array(group.buffers[1].data)[0];
                  const candidate = pipe.label.startsWith("matvec_"), subgroup = pipe.label.endsWith("_subgroup");
                  const duration = subgroup ? options.subgroupTime || 5000 : candidate ? options.twoTime?.(N) || 7000 : 10000;
                  writes.querySet.values[writes.beginningOfPassWriteIndex] = 0n;
                  writes.querySet.values[writes.endOfPassWriteIndex] = BigInt(duration * dispatches.length);
                }
              });
            },
          };
        },
        resolveQuerySet(query, first, count, output, offset) { commands.push(() => new BigInt64Array(output.data, offset, count).set(query.values.slice(first, first + count))); },
        copyBufferToBuffer(src, srcOffset, dst, dstOffset, bytes) { commands.push(() => new Uint8Array(dst.data, dstOffset, bytes).set(new Uint8Array(src.data, srcOffset, bytes))); },
        finish() { return commands; },
      };
    },
    queue: {
      writeBuffer(buffer, offset, value) { const bytes = value instanceof ArrayBuffer ? new Uint8Array(value) : new Uint8Array(value.buffer, value.byteOffset, value.byteLength); new Uint8Array(buffer.data, offset, bytes.length).set(bytes); },
      submit(commands) { for (const command of commands) for (const run of command) run(); },
    },
  };
  return { device, buffers, querySets, sources, inputs };
}

function setup({ f16 = false, pairSelected = true, block = 32 } = {}) {
  const tensor = (name, N, K) => ({ info: { name, dtype: "q8", block, shape: [N, K] }, buf: {}, sbuf: {} });
  const tensors = new Map([
    ["l.0.gate", tensor("l.0.gate", 64, 32)], ["l.0.up", tensor("l.0.up", 64, 32)],
    ["l.15.gate", tensor("l.15.gate", 128, 32)], ["l.15.up", tensor("l.15.up", 128, 32)],
    ["l.15.down", tensor("l.15.down", 32, 128)],
  ]);
  const matvec = { pipelines: { pipeline: { label: "baseline-plain" }, layout: {}, f16, columns: f16 ? 16 : 4, weights: "q8" }, selection: new Set(["64:32", "128:32", "32:128"]) };
  const matvecGelu = { pipelines: { pipeline: { label: "baseline-paired-gelu" }, layout: {}, f16, columns: f16 ? 16 : 4 }, selection: new Set(pairSelected ? ["128:32"] : []) };
  const gelu = { label: "baseline-separate-gelu", getBindGroupLayout: () => ({}) };
  return { weights: { tensors }, options: { matvec, matvecGelu, gelu, sharedStart: 15 }, wgsl: (kernel, spec) => ({ kernel, spec }) };
}

test("unsupported and disabled calibrations allocate nothing and keep empty selections", async () => {
  for (const reason of ["disabled", "no-timestamp-query", "missing-baseline-pipeline", "no-selected-q8-shapes"]) {
    const mock = mockDevice(), data = setup({ block: reason === "no-selected-q8-shapes" ? 16 : 32 });
    if (reason === "no-timestamp-query") mock.device.features.clear();
    if (reason === "missing-baseline-pipeline") data.options.matvec = null;
    if (reason === "disabled") data.options.enabled = false;
    const result = await calibrateMatvecTwo(mock.device, data.wgsl, data.weights, data.options);
    assert.equal(result.diagnostics.reason, reason); assert.equal(result.selection.plain.size, 0); assert.equal(result.selection.gelu.size, 0);
    assert.equal(mock.buffers.length, 0); assert.equal(mock.sources.length, 0);
  }
});

test("calibration prioritizes the shared-tail pair and compares each effective baseline", async () => {
  const mock = mockDevice({ f16: true }), data = setup({ f16: true });
  const result = await calibrateMatvecTwo(mock.device, data.wgsl, data.weights, data.options);
  assert.equal(result.diagnostics.reason, "calibrated");
  assert.equal(result.diagnostics.points[0].name, "l.15.gate"); assert.equal(result.diagnostics.points[0].kind, "gelu");
  assert.equal(result.diagnostics.points[1].name, "l.15.down");
  assert.equal(result.diagnostics.points.find((point) => point.kind === "gelu" && point.key === "128:32").baselineKernel, "matvec-gelu");
  assert.equal(result.diagnostics.points.find((point) => point.kind === "gelu" && point.key === "64:32").baselineKernel, "matvec+matvec+gelu");
  assert.deepEqual([...result.selection.gelu], ["128:32", "64:32"]); assert.equal(result.selection.plain.size, 3);
  assert.equal(result.pipelines.plain.layout, data.options.matvec.pipelines.layout);
  assert.equal(result.pipelines.gelu.layout, data.options.matvecGelu.pipelines.layout);
  assert.equal(result.pipelines.plain.columns, 16); assert.equal(result.pipelines.gelu.columns, 16);
  for (const source of mock.sources) assert.deepEqual(source.spec, { weights: "q8", f16: true, rows: 1, groups: 1, splitTarget: 256 });
  assert(mock.inputs.some((input) => input[0] !== input[(input.length - 4) / 2]));
  assert(mock.buffers.every((buffer) => buffer.destroyed && !buffer.mapped)); assert(mock.querySets.every((query) => query.destroyed));
  assert.deepEqual(structuredClone(result.diagnostics), result.diagnostics);
});

test("a later signed-zero mismatch clears both earlier selection kinds", async () => {
  const mock = mockDevice({ corrupt: ({ kind, N }) => kind === "plain" && N === 32 }), data = setup();
  const result = await calibrateMatvecTwo(mock.device, data.wgsl, data.weights, data.options);
  assert.equal(result.diagnostics.reason, "numerical-guard-failed"); assert.equal(result.selection.plain.size, 0); assert.equal(result.selection.gelu.size, 0);
  assert.equal(result.diagnostics.points[0].selected, false); assert.equal(result.diagnostics.points[0].reason, "aborted-after-calibration-failure");
  assert.equal(result.diagnostics.points.at(-1).numericalCheck.bitDifferences, 1);
  assert(mock.buffers.every((buffer) => buffer.destroyed && !buffer.mapped)); assert(mock.querySets.every((query) => query.destroyed));
});

test("tail overwrites and map failures fail closed and release all scratch", async () => {
  for (const options of [{ tailFailure: true }, { mapFailure: 4 }]) {
    const mock = mockDevice(options), data = setup();
    const result = await calibrateMatvecTwo(mock.device, data.wgsl, data.weights, data.options);
    assert.equal(result.selection.plain.size, 0); assert.equal(result.selection.gelu.size, 0);
    assert.equal(result.diagnostics.reason, options.tailFailure ? "numerical-guard-failed" : "calibration-failed");
    assert(mock.buffers.every((buffer) => buffer.destroyed && !buffer.mapped)); assert(mock.querySets.every((query) => query.destroyed));
  }
});

test("only deadlines preserve earlier validated points; GPU errors take precedence", async (t) => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "performance"); let clock = 0;
  Object.defineProperty(globalThis, "performance", { configurable: true, value: { now: () => clock } });
  t.after(() => Object.defineProperty(globalThis, "performance", descriptor));
  for (const gpuFailure of [false, true]) {
    clock = 0;
    const mock = mockDevice({ onMap(device) { if (device.maps === 4) { clock = 2000; if (gpuFailure) device.scopeFailure = new Error("validation failure"); } } }), data = setup();
    const result = await calibrateMatvecTwo(mock.device, data.wgsl, data.weights, data.options);
    assert.equal(result.diagnostics.reason, gpuFailure ? "calibration-failed" : "deadline");
    assert.equal(result.selection.gelu.size, gpuFailure ? 0 : 1); assert.equal(result.selection.plain.size, 0);
    assert(mock.buffers.every((buffer) => buffer.destroyed && !buffer.mapped)); assert(mock.querySets.every((query) => query.destroyed));
  }
});

test("the scratch cap skips oversized shapes before allocating calibration buffers", async () => {
  const mock = mockDevice(), data = setup();
  data.weights.tensors = new Map([["l.15.down", { info: { dtype: "q8", block: 32, shape: [2 ** 23, 32] }, buf: {}, sbuf: {} }]]);
  data.options.matvec.selection = new Set([`${2 ** 23}:32`]);
  const result = await calibrateMatvecTwo(mock.device, data.wgsl, data.weights, data.options);
  assert.equal(result.diagnostics.points[0].reason, "scratch-limit"); assert.equal(mock.buffers.length, 0);
  assert.equal(result.selection.plain.size, 0); assert.equal(result.selection.gelu.size, 0);
});

test("fixed32 refinement samples only selected old two-row shapes and reuses their layouts", async () => {
  const mock = mockDevice({ subgroups: true, twoTime: (N) => N === 32 ? 10000 : 7000 }), data = setup();
  const result = await calibrateMatvecTwo(mock.device, data.wgsl, data.weights, { ...data.options, subgroup32: true });
  assert.equal(result.diagnostics.reason, "calibrated"); assert.equal(result.subgroups.diagnostics.reason, "calibrated");
  assert.equal(result.selection.plain.has("32:128"), false); assert.equal(result.subgroups.selection.plain.has("32:128"), false);
  assert.deepEqual([...result.subgroups.selection.gelu], ["128:32", "64:32"]);
  assert(result.subgroups.diagnostics.points.every((point) => result.selection[point.kind].has(point.key)));
  assert(result.subgroups.diagnostics.points.every((point) => point.baselineKernel === (point.kind === "plain" ? "matvec-two" : "matvec-gelu-two")));
  for (const kind of ["plain", "gelu"]) {
    assert.equal(result.subgroups.pipelines[kind].layout, result.pipelines[kind].layout);
    assert.equal(result.pipelines[kind].subgroup.pipeline, result.subgroups.pipelines[kind].pipeline);
    assert.equal(result.pipelines[kind].subgroup.selection, result.subgroups.selection[kind]);
  }
  assert.equal(mock.sources.filter((source) => source.spec.subgroups).length, 2);
  assert.deepEqual(structuredClone(result.diagnostics), result.diagnostics);
  assert(mock.buffers.every((buffer) => buffer.destroyed && !buffer.mapped)); assert(mock.querySets.every((query) => query.destroyed));
});

test("F16 and adapters without verified fixed32 support keep their old two-row kernels", async () => {
  for (const [f16, feature, fixed32] of [[true, true, true], [false, true, false], [false, false, true]]) {
    const mock = mockDevice({ f16, subgroups: feature }), data = setup({ f16 });
    const result = await calibrateMatvecTwo(mock.device, data.wgsl, data.weights, { ...data.options, subgroup32: fixed32 });
    assert.equal(result.selection.plain.size, 3); assert.equal(result.selection.gelu.size, 2);
    assert.equal(result.subgroups.selection.plain.size, 0); assert.equal(result.subgroups.selection.gelu.size, 0);
    assert.equal(mock.sources.some((source) => source.spec.subgroups), false);
  }
});

test("refinement compile, numerical, tail and small-win failures retain every old two-row win", async () => {
  for (const options of [{ subgroupPipelineFailure: true }, { subgroupCorrupt: ({ N }) => N === 64 }, { subgroupTailFailure: true }, { subgroupTime: 6800 }]) {
    const mock = mockDevice({ subgroups: true, ...options }), data = setup();
    const result = await calibrateMatvecTwo(mock.device, data.wgsl, data.weights, { ...data.options, subgroup32: true });
    assert.equal(result.diagnostics.reason, "calibrated");
    assert.equal(result.selection.plain.size, 3); assert.equal(result.selection.gelu.size, 2);
    assert.equal(result.subgroups.selection.plain.size, 0); assert.equal(result.subgroups.selection.gelu.size, 0);
    assert.equal(result.subgroups.diagnostics.reason, options.subgroupPipelineFailure ? "pipeline-failed" : options.subgroupTime ? "no-stable-win" : "numerical-guard-failed");
    assert(mock.buffers.every((buffer) => buffer.destroyed && !buffer.mapped)); assert(mock.querySets.every((query) => query.destroyed));
    assert.deepEqual(structuredClone(result.diagnostics), result.diagnostics);
  }
});

test("a refinement deadline preserves old wins and only earlier verified subgroup points", async (t) => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "performance"); let clock = 0;
  Object.defineProperty(globalThis, "performance", { configurable: true, value: { now: () => clock } });
  t.after(() => Object.defineProperty(globalThis, "performance", descriptor));
  for (const gpuFailure of [false, true]) {
    clock = 0;
    const mock = mockDevice({ subgroups: true, onMap(device) { if (device.maps === 19) { clock = 2000; if (gpuFailure) device.scopeFailure = new Error("validation failure"); } } }), data = setup();
    const result = await calibrateMatvecTwo(mock.device, data.wgsl, data.weights, { ...data.options, subgroup32: true });
    assert.equal(result.selection.plain.size, 3); assert.equal(result.selection.gelu.size, 2);
    assert.equal(result.subgroups.diagnostics.reason, gpuFailure ? "calibration-failed" : "deadline");
    assert.equal(result.subgroups.selection.gelu.size, gpuFailure ? 0 : 1); assert.equal(result.subgroups.selection.plain.size, 0);
    assert(mock.buffers.every((buffer) => buffer.destroyed && !buffer.mapped)); assert(mock.querySets.every((query) => query.destroyed));
  }
});
