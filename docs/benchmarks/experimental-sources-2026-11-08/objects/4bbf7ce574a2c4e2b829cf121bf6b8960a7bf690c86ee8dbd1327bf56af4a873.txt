import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { runInNewContext } from "node:vm";
import { packingFixture, packingGateWGSL } from "./integer-matrix-gate-helper.js";

const MATRIX = "chromium-experimental-subgroup-matrix";
const html = readFileSync(new URL("./integer-matrix-gate.html", import.meta.url), "utf8");
const scripts = [...html.matchAll(/<script type="module">([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length, 1);
const importLine = 'import { packingFixture, packingGateWGSL } from "./integer-matrix-gate-helper.js";';
assert.equal(scripts[0][1].split(importLine).length, 2);
const script = scripts[0][1].replace(importLine, "");

async function capture({ start = true, missingConfig = false, compileError = false, deviceError = false, mapError = false, mutate = () => {} } = {}) {
  const calls = { adapters: 0, devices: 0, modules: 0, pipelines: 0, buffers: [], dispatches: [], submits: 0, destroyedBuffers: 0, destroyedDevice: 0, scopes: [] };
  const limits = { maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupStorageSize: 16384, maxStorageBufferBindingSize: 2 ** 27, maxBufferSize: 2 ** 28 };
  const device = {
    features: new Set(["subgroups", MATRIX]), limits,
    pushErrorScope(kind) { calls.scopes.push(kind); },
    async popErrorScope() { assert(calls.scopes.length > 0); calls.scopes.pop(); return null; },
    createShaderModule({ code }) {
      calls.modules++; assert.equal(code, packingGateWGSL, "only the frozen one-operation shader may be compiled");
      return { async getCompilationInfo() { return { messages: compileError ? [{ type: "error", message: "mock i8 compiler rejection", lineNum: 1, linePos: 1, offset: 0, length: 1 }] : [] }; } };
    },
    async createComputePipelineAsync() { calls.pipelines++; return { getBindGroupLayout() { return {}; } }; },
    createBuffer({ label, size }) {
      const buffer = { label, data: new ArrayBuffer(size), destroy() { calls.destroyedBuffers++; },
        async mapAsync() { if (mapError) throw new Error("mock readback mapping rejected"); },
        getMappedRange() { return this.data; }, unmap() {} };
      calls.buffers.push(buffer); return buffer;
    },
    createBindGroup({ entries }) { return entries.map((entry) => entry.resource.buffer); },
    createCommandEncoder() {
      let buffers;
      return {
        beginComputePass() { return {
          setPipeline() {}, setBindGroup(_index, group) { buffers = group; },
          dispatchWorkgroups(...size) {
            calls.dispatches.push(size); assert.deepEqual(size, [1, 1, 1]);
            // Test-only mock readback. Every numerical test corrupts it deliberately.
            const words = new Int32Array(buffers[2].data);
            words.set(packingFixture().expected, 64); mutate(words);
          }, end() {},
        }; },
        copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
          new Uint8Array(destination.data, destinationOffset, size).set(new Uint8Array(source.data, sourceOffset, size));
        }, finish() { return {}; },
      };
    },
    queue: {
      writeBuffer(buffer, offset, values) { new Uint8Array(buffer.data, offset, values.byteLength).set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength)); },
      submit() { calls.submits++; },
    },
    destroy() { calls.destroyedDevice++; },
  };
  const adapter = { features: new Set(["subgroups", MATRIX, "shader-f16"]), limits,
    info: { vendor: "nvidia", architecture: "blackwell", device: "mock", description: "Mock NVIDIA", isFallbackAdapter: false, subgroupMinSize: 32, subgroupMaxSize: 32,
      subgroupMatrixConfigs: missingConfig ? [] : [{ componentType: "i8", resultComponentType: "i32", M: 16, N: 16, K: 32 }] },
    async requestDevice(options) {
      calls.devices++;
      assert.deepEqual(JSON.parse(JSON.stringify(options)), { requiredFeatures: ["subgroups", MATRIX] });
      if (deviceError) throw new Error("mock device rejected");
      return device;
    },
  };
  const window = {}, navigator = { gpu: { async requestAdapter() { calls.adapters++; return adapter; } } };
  await runInNewContext(`(async () => {\n${script}\n})()`, { window, navigator, packingFixture, packingGateWGSL, Int8Array, Int32Array,
    URL, location: { href: "http://127.0.0.1/integer-matrix-gate" + (start ? "?run=1" : "") },
    GPUBufferUsage: { STORAGE: 128, COPY_DST: 8, COPY_SRC: 4, MAP_READ: 1 }, GPUMapMode: { READ: 1 } }, { timeout: 1000 });
  return { result: JSON.parse(JSON.stringify(window.integerMatrixGate)), calls };
}

test("default navigation stays idle without requesting an adapter or device", async () => {
  const { result, calls } = await capture({ start: false });
  assert.equal(result.status, "idle"); assert.equal(result.done, false);
  assert.equal(calls.adapters, 0); assert.equal(calls.devices, 0);
  assert.equal(calls.modules, 0); assert.equal(calls.dispatches.length, 0);
});

test("classic bootstrap publishes a terminal module-load error before module execution", () => {
  const classic = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  const listeners = {}, window = { addEventListener(name, listener) { listeners[name] = listener; } };
  runInNewContext(classic, { window, ErrorEvent: class ErrorEvent {} }, { timeout: 1000 });
  listeners.error({ target: { tagName: "SCRIPT", type: "module", getAttribute() { return "missing-helper.js"; } } });
  assert.equal(window.integerMatrixGate.status, "error");
  assert.equal(window.integerMatrixGate.done, true);
  assert.equal(window.integerMatrixGate.phase, "module-load");
  assert.match(window.integerMatrixGate.error, /missing-helper\.js/);
});

test("mock readback mismatch at the final element is retained after comparing all256 outputs", async () => {
  const { result, calls } = await capture({ mutate(words) { words[319] ^= 1; } });
  assert.equal(result.status, "complete");
  assert.equal(result.gate.status, "numerical-failed");
  assert.equal(result.gate.correctnessPassed, false);
  assert.equal(result.gate.compared, 256);
  assert.equal(result.gate.different, 1);
  assert.equal(result.gate.mismatches[0].index, 255);
  assert.equal(result.gate.outputs.length, 256);
  assert.equal(result.gate.reference.positiveOutputCount, 106);
  assert.deepEqual(result.gate.reference.activeInputSigns, { left: { positive: 219, negative: 219, zero: 74 }, right: { positive: 219, negative: 220, zero: 73 } });
  assert.equal(calls.modules, 1); assert.equal(calls.pipelines, 1);
  assert.deepEqual(calls.dispatches, [[1, 1, 1]]);
  assert.equal(calls.destroyedBuffers, 4); assert.equal(calls.destroyedDevice, 1);
});

test("mock untouched poison cannot pass or masquerade as positive reference outputs", async () => {
  const { result } = await capture({ mutate(words) { words.fill(0x6badc0de, 64, 320); } });
  assert.equal(result.gate.status, "numerical-failed");
  assert.equal(result.gate.different, 256);
  assert.equal(result.gate.positiveOutputCount, 256);
  assert.equal(result.gate.reference.positiveOutputCount, 106);
});

test("mock first prefix and last tail corruption is retained despite exact payload", async () => {
  const { result } = await capture({ mutate(words) { words[0] ^= 1; words[383] ^= 1; } });
  assert.equal(result.gate.status, "numerical-failed");
  assert.equal(result.gate.different, 0);
  assert.equal(result.gate.canaries.compared, 128);
  assert.equal(result.gate.canaries.changed, 2);
  assert.equal(result.gate.canaries.prefix.length, 64);
  assert.equal(result.gate.canaries.tail.length, 64);
});

test("mock module rejection remains compile-failed with no allocations or dispatch", async () => {
  const { result, calls } = await capture({ compileError: true });
  assert.equal(result.gate.status, "compile-failed");
  assert.equal(result.gate.compilation.diagnostics.messages[0].message, "mock i8 compiler rejection");
  assert.equal(calls.buffers.length, 0); assert.equal(calls.dispatches.length, 0);
  assert.equal(result.gate.outputs, undefined);
});

test("mock absent configuration is skipped separately from compilation and numerical failure", async () => {
  const { result, calls } = await capture({ missingConfig: true });
  assert.equal(result.gate.status, "skipped");
  assert.equal(result.gate.reason, "missing-advertised-i8-i32-M16-N16-K32-config");
  assert.equal(calls.modules, 0); assert.equal(calls.dispatches.length, 0);
});

test("mock device rejection is retained without synthesized output", async () => {
  const { result, calls } = await capture({ deviceError: true });
  assert.equal(result.gate.status, "device-rejected");
  assert.equal(result.device.error.message, "mock device rejected");
  assert.equal(calls.modules, 0); assert.equal(result.gate.outputs, undefined);
});

test("mock mapping rejection is execution-failed and closes all buffer objects", async () => {
  const { result, calls } = await capture({ mapError: true });
  assert.equal(result.gate.status, "execution-failed");
  assert.equal(result.gate.executionError.message, "mock readback mapping rejected");
  assert.equal(result.gate.outputs, undefined);
  assert.equal(calls.destroyedBuffers, 4); assert.equal(calls.destroyedDevice, 1);
  assert.deepEqual(calls.scopes, []);
});
