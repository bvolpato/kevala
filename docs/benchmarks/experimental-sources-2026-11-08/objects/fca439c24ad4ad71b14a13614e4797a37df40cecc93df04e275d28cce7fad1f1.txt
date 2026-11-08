import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { runInNewContext } from "node:vm";

const MATRIX = "chromium-experimental-subgroup-matrix";
const html = readFileSync(new URL("./subgroup-matrix-probe.html", import.meta.url), "utf8");
const scripts = [...html.matchAll(/<script type="module">([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length, 1);
const f32 = { componentType: "f32", resultComponentType: "f32", M: 16, N: 8, K: 32 };

async function capture({ configs = [f32], rejection = null, messages = [], pipelineError = null, advertised = ["subgroups", MATRIX, "shader-f16"], subgroupMaxSize = 32 } = {}) {
  const calls = { deviceRequests: [], shaderModules: [], pipelines: [], destroyed: 0, scopes: [] };
  const limits = { maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupSizeY: 256,
    maxComputeWorkgroupSizeZ: 64, maxComputeWorkgroupStorageSize: 32768, maxStorageBufferBindingSize: 2 ** 27, maxBufferSize: 2 ** 28 };
  const device = {
    features: new Set(), limits,
    pushErrorScope(kind) { calls.scopes.push(kind); },
    async popErrorScope() { assert.equal(calls.scopes.pop(), "validation"); return null; },
    createShaderModule(descriptor) {
      calls.shaderModules.push(JSON.parse(JSON.stringify(descriptor)));
      return { async getCompilationInfo() { return { messages }; } };
    },
    async createComputePipelineAsync(descriptor) {
      calls.pipelines.push({ layout: descriptor.layout, entryPoint: descriptor.compute.entryPoint });
      if (pipelineError) throw Object.assign(new Error(pipelineError), { name: "GPUPipelineError" });
      return {};
    },
    createBuffer() { assert.fail("No GPU buffers are permitted"); },
    createBindGroup() { assert.fail("No bind groups are permitted"); },
    createCommandEncoder() { assert.fail("No command encoding or dispatch is permitted"); },
    destroy() { calls.destroyed++; },
  };
  const adapter = {
    info: { vendor: "nvidia", architecture: "blackwell", device: "mock", description: "Mock NVIDIA GPU", isFallbackAdapter: false,
      subgroupMinSize: 32, subgroupMaxSize, subgroupMatrixConfigs: configs },
    features: new Set(advertised), limits,
    async requestDevice(options) {
      calls.deviceRequests.push(JSON.parse(JSON.stringify(options)));
      if (rejection) throw Object.assign(new Error(rejection), { name: "OperationError" });
      device.features = new Set(options.requiredFeatures);
      return device;
    },
  };
  const window = {}, navigator = { gpu: { async requestAdapter() { return adapter; } } };
  await runInNewContext(`(async () => {\n${scripts[0][1]}\n})()`, { window, navigator }, { timeout: 1000 });
  return { result: JSON.parse(JSON.stringify(window.subgroupMatrixProbe)), calls };
}

test("selects the first exact f32/f32 configuration and compiles its dimensions without shader-f16", async () => {
  const other = [{ ...f32, componentType: "f16" }, { ...f32, resultComponentType: "i32" }];
  const { result, calls } = await capture({ configs: [...other, f32, { ...f32, M: 64 }] });
  assert.equal(result.status, "complete");
  assert.deepEqual(result.adapter.subgroupMatrixConfigs.entries, [...other, f32, { ...f32, M: 64 }]);
  assert.deepEqual(calls.deviceRequests, [{ requiredFeatures: ["subgroups", MATRIX] }]);
  assert.deepEqual(result.compilation.selectedConfig, f32);
  assert.deepEqual(result.compilation.workgroupSize, [32, 1, 1]);
  assert.equal(result.compilation.status, "pipeline-created");
  assert.equal(calls.shaderModules.length, 1);
  assert.equal(calls.shaderModules[0].code, `enable chromium_experimental_subgroup_matrix;
enable subgroups;
const M = 16;
const N = 8;
const K = 32;
alias LeftType = subgroup_matrix_left<f32, K, M>;
alias RightType = subgroup_matrix_right<f32, N, K>;
@group(0) @binding(0) var<storage, read> lhs_input: array<f32, M * K>;
@group(0) @binding(1) var<storage, read> rhs_input: array<f32, K * N>;
@group(0) @binding(2) var<storage, read_write> output: array<f32, M * N>;
@compute @workgroup_size(32)
fn main() {
  let lhs = subgroupMatrixLoad<LeftType, row_major>(&lhs_input, 0, K);
  let rhs = subgroupMatrixLoad<RightType, row_major>(&rhs_input, 0, N);
  let product = subgroupMatrixMultiply<f32>(lhs, rhs);
  subgroupMatrixStore<row_major>(&output, 0, product, N);
}
`);
  assert.equal(result.compilation.wgsl, calls.shaderModules[0].code);
  assert.deepEqual(calls.pipelines, [{ layout: "auto", entryPoint: "main" }]);
  assert.equal(calls.destroyed, 1);
  assert.deepEqual(calls.scopes, []);
});

test("retains all non-f32/f32 configs and skips compilation when none is usable", async () => {
  const configs = [{ ...f32, componentType: "f16" }, { ...f32, resultComponentType: "i32" }];
  const { result, calls } = await capture({ configs });
  assert.equal(result.status, "complete");
  assert.deepEqual(result.adapter.subgroupMatrixConfigs.entries, configs);
  assert.equal(result.compilation.status, "skipped");
  assert.equal(result.compilation.reason, "no-advertised-f32-f32-configuration");
  assert.equal(calls.shaderModules.length, 0);
  assert.equal(calls.pipelines.length, 0);
  assert.equal(calls.destroyed, 1);
});

test("retains shader diagnostics and does not create a pipeline after a module error", async () => {
  const message = { type: "error", message: "mock unsupported matrix dimensions", lineNum: 7, linePos: 1, offset: 120, length: 6 };
  const { result, calls } = await capture({ messages: [message] });
  assert.equal(result.status, "complete");
  assert.equal(result.compilation.status, "failed");
  assert.equal(result.compilation.stage, "shader-module");
  assert.deepEqual(result.compilation.diagnostics.messages, [message]);
  assert.equal(calls.pipelines.length, 0);
  assert.equal(calls.destroyed, 1);
  assert.deepEqual(calls.scopes, []);
});

test("retains a pipeline rejection as a compilation outcome and destroys the device", async () => {
  const { result, calls } = await capture({ pipelineError: "mock pipeline rejected" });
  assert.equal(result.status, "complete");
  assert.equal(result.compilation.status, "failed");
  assert.deepEqual(result.compilation.error, { name: "GPUPipelineError", message: "mock pipeline rejected" });
  assert.equal(calls.pipelines.length, 1);
  assert.equal(calls.destroyed, 1);
  assert.deepEqual(calls.scopes, []);
});

test("retains a device rejection as a capability outcome without compiling", async () => {
  const { result, calls } = await capture({ rejection: "mock matrix device rejected" });
  assert.equal(result.status, "complete");
  assert.equal(result.device.status, "request-rejected");
  assert.deepEqual(result.device.error, { name: "OperationError", message: "mock matrix device rejected" });
  assert.equal(result.compilation.reason, "device-request-rejected");
  assert.equal(calls.shaderModules.length, 0);
  assert.equal(calls.pipelines.length, 0);
  assert.equal(calls.destroyed, 0);
});

test("skips a subgroup width that cannot fit device workgroup limits", async () => {
  const { result, calls } = await capture({ subgroupMaxSize: 512 });
  assert.equal(result.status, "complete");
  assert.equal(result.compilation.reason, "subgroup-max-size-does-not-fit-device-workgroup-limits");
  assert.equal(calls.shaderModules.length, 0);
  assert.equal(calls.destroyed, 1);
});
