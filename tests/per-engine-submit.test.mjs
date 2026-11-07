import assert from "node:assert/strict";
import test from "node:test";
import { gpuLayouts } from "../js/src/pack-layout.js";
import { SUBMIT } from "../js/src/gpu.js";
import { GpuTrunk } from "../js/src/gpu-laya.js";
import { GpuKev } from "../js/src/gpu-kev.js";
import { GpuGemma4 } from "../js/src/gpu-gemma4.js";

function layout() {
  return gpuLayouts({ config: { arch: "test" }, model: {}, tokenizer: { offset: 0, size: 0 }, tensors: [] }, 16)[1];
}

function device() {
  const events = [];
  let encoderId = 0;
  const createCommandEncoder = () => {
    const id = ++encoderId;
    return {
      beginComputePass() { return { setBindGroup() {}, end() {} }; },
      copyBufferToBuffer() {},
      finish() { return { id }; },
    };
  };
  return {
    events,
    device: {
      features: new Set(),
      limits: { maxBufferSize: 1 << 20, maxStorageBufferBindingSize: 1 << 20 },
      lost: Promise.resolve(),
      createCommandEncoder,
      pushErrorScope() {},
      async popErrorScope() { return null; },
      queue: {
        writeBuffer() {},
        submit() { events.push("submit"); },
        async onSubmittedWorkDone() { events.push("wait"); },
      },
    },
  };
}

function readback() {
  return {
    async mapAsync() {},
    getMappedRange(_offset, size) { return new ArrayBuffer(size); },
    unmap() {},
  };
}

function layaForward(trunk) {
  Object.assign(trunk, {
    cfg: { hidden: 1, heads: 1 },
    ensure() {},
    globals: {}, blocks: {}, x: {}, tok: {}, rowsBuf: {}, gathered: {}, readback: readback(),
    attnQueries: 16,
    mm: {},
    ops: Array.from({ length: 6 }, () => ({ kind: "noop" })),
  });
  return () => trunk.forward(new Float32Array(257), [{ start: 0, len: 257, qtype: 0 }], [256]);
}

function kevForward(trunk) {
  Object.assign(trunk, {
    cfg: { hidden: 1 }, dims: { kv: 1 }, cap: { S: 1 },
    ensureCarries() {}, ensure() {},
    plan: () => ({ segs: [], carries: [], copies: [] }),
    encode() {},
    finalRowsEnabled: false,
    ops: [[], Array.from({ length: 6 }, () => ({ k: "noop" }))],
    g: {}, g2: {}, x: {}, x2: {}, tok: {}, tok2: {}, segs: {}, segs2: {}, rows: {}, gathered: {}, readback: readback(),
    carry: [],
  });
  const batch = { T1: 0, T2: 257, states: [], branches: [[0, 257, 0]], rows: [256] };
  return () => trunk.forwardPass(new Float32Array(), new Float32Array(257), batch, new Uint32Array());
}

function gemmaForward(trunk) {
  Object.assign(trunk, {
    cfg: { hidden: 1, vocab: 4, pleVocab: 4, hostPle: false },
    initialized: true,
    clearPleVectorCache() {}, clearTextPrefixCache() {},
    ensureRope: async () => {}, ensure() {}, ensureTail() {},
    preparePleVectorCache() {}, finishPleVectorCache() {}, encode() {},
    tailOps: null, prefixOps: null,
    ops: Array.from({ length: 6 }, () => ({ k: "noop" })),
    globals: {}, ids: {}, pleIds: {}, pos: {}, rows: {}, gathered: {}, readback: readback(),
  });
  return () => trunk.forwardRows(new Uint32Array(257).fill(1), null, []);
}

const fixtures = [
  ["Laya", GpuTrunk, { hidden: 1, heads: 1, layers: 0, intermediate: 1, global_every: 1, window: 1, rope_global: 10000, rope_local: 10000, norm_eps: 1e-5, head_layers: 0, head_ff: 1, head_norm_eps: 1e-5 }, layaForward],
  ["Kev", GpuKev, { hidden: 1, heads: 1, kv_heads: 1, lin_key_heads: 1, lin_heads: 1, layers: 0, full: [] }, kevForward],
  ["Gemma 4", GpuGemma4, { hidden: 1, layers: 0, hostPle: false, hostWord: false }, gemmaForward],
];

test("interleaved text trunks keep their own chunk submission modes", async (t) => {
  const previousUsage = globalThis.GPUBufferUsage;
  const previousMapMode = globalThis.GPUMapMode;
  const previousSubmitMode = SUBMIT.mode;
  globalThis.GPUBufferUsage = { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, MAP_READ: 16 };
  globalThis.GPUMapMode = { READ: 1 };
  t.after(() => {
    if (previousUsage === undefined) delete globalThis.GPUBufferUsage;
    else globalThis.GPUBufferUsage = previousUsage;
    if (previousMapMode === undefined) delete globalThis.GPUMapMode;
    else globalThis.GPUMapMode = previousMapMode;
    SUBMIT.mode = previousSubmitMode;
  });

  const splitEvents = Array(6).fill("submit");
  const awaitEvents = ["submit", "wait", "submit", "wait", "submit", "wait", "submit", "wait", "submit", "wait", "submit"];
  const expected = { none: ["submit"], split: splitEvents, await: awaitEvents };
  for (const [name, Trunk, cfg, forward] of fixtures) {
    const engines = ["none", "split", "await"].map((mode) => {
      SUBMIT.mode = mode;
      const runtime = device();
      const trunk = new Trunk({ device: runtime.device, submitMode: mode }, layout(), cfg);
      return { mode, runtime, run: forward(trunk) };
    });

    await Promise.all(engines.map(({ run }) => run()));
    for (const { mode, runtime } of engines) {
      assert.deepEqual(runtime.events, expected[mode], `${name} ${mode}-mode trunk should retain its submission policy`);
    }
  }
});
