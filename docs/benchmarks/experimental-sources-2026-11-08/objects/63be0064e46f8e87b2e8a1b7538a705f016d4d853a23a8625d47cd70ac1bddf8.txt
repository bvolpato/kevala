import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { runInNewContext } from "node:vm";

const html = readFileSync(new URL("./f16-feature-probe.html", import.meta.url), "utf8");
const scripts = [...html.matchAll(/<script type="module">([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length, 1, "execute the page's one actual inline module script");

async function capture(preferred, fallback) {
  const calls = [], deviceRequests = [];
  let destroyed = 0;
  const features = new Set(["timestamp-query", "shader-f16", "subgroups"]);
  const adapter = {
    info: { vendor: "nvidia", architecture: "blackwell", device: "mock", description: "Mock NVIDIA GPU", isFallbackAdapter: false },
    features, limits: { maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 2 ** 31, maxComputeWorkgroupStorageSize: 65536 },
    async requestDevice(options) {
      deviceRequests.push(JSON.parse(JSON.stringify(options)));
      return { features, destroy() { destroyed++; } };
    },
  };
  const window = {};
  const navigator = { gpu: { async requestAdapter(options) {
    calls.push(options === undefined ? null : { powerPreference: options.powerPreference });
    assert(calls.length <= 2, "adapter acquisition must remain bounded to two calls");
    return (calls.length === 1 ? preferred : fallback) ? adapter : null;
  } } };
  await runInNewContext(`(async () => {\n${scripts[0][1]}\n})()`, { window, navigator }, { timeout: 1000 });
  return { result: JSON.parse(JSON.stringify(window.f16FeatureProbe)), calls, deviceRequests, destroyed };
}

test("preferred null falls back to default and records the created device features", async () => {
  const { result, calls, deviceRequests, destroyed } = await capture(false, true);
  assert.equal(result.status, "complete");
  assert.deepEqual(calls, [{ powerPreference: "high-performance" }, null]);
  assert.deepEqual(result.adapterRequests, { highPerformanceReturnedNull: true, defaultFallbackRequested: true, defaultFallbackReturnedAdapter: true, defaultFallbackReturnedNull: false });
  assert.deepEqual(result.device.features, ["shader-f16", "subgroups", "timestamp-query"]);
  assert.deepEqual(deviceRequests, [{ requiredFeatures: ["timestamp-query", "shader-f16", "subgroups"],
    requiredLimits: { maxStorageBufferBindingSize: 1 << 30, maxBufferSize: 1 << 30, maxComputeWorkgroupStorageSize: 32768 } }]);
  assert.equal(result.device.destroyed, true);
  assert.equal(destroyed, 1);
});

test("preferred success does not request a default adapter", async () => {
  const { result, calls, destroyed } = await capture(true, true);
  assert.equal(result.status, "complete");
  assert.deepEqual(calls, [{ powerPreference: "high-performance" }]);
  assert.deepEqual(result.adapterRequests, { highPerformanceReturnedNull: false, defaultFallbackRequested: false, defaultFallbackReturnedAdapter: null, defaultFallbackReturnedNull: null });
  assert.equal(destroyed, 1);
});

test("both adapter requests returning null retains a terminal failure", async () => {
  const { result, calls, deviceRequests, destroyed } = await capture(false, false);
  assert.equal(result.status, "error");
  assert.equal(result.done, true);
  assert.deepEqual(calls, [{ powerPreference: "high-performance" }, null]);
  assert.deepEqual(result.adapterRequests, { highPerformanceReturnedNull: true, defaultFallbackRequested: true, defaultFallbackReturnedAdapter: false, defaultFallbackReturnedNull: true });
  assert.match(result.error, /No WebGPU adapter/);
  assert.deepEqual(deviceRequests, []);
  assert.equal(destroyed, 0);
});
