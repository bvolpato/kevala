import { pipeline, requestDevice } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const logNode = document.getElementById("log");
window.gpuBench = { status: "running", backend: "webgpu" };

async function run() {
  const query = new URLSearchParams(location.search);
  const kernels = query.get("kernels")?.split(",");
  if (!kernels?.length || kernels.some((name) => !/^[a-z][a-z0-9_]*$/.test(name))) throw new Error("kernels must list the registered kernel names");
  const baseline = query.get("baseline") === "1";
  const runtime = await requestDevice({ baseline });
  const device = runtime.device;
  try {
    const source = await kernelSource();
    const specifications = [];
    for (const f16 of [false, true]) for (const subgroups of [false, true]) for (const rows of [1, 2, 3, 4]) for (const groups of [1, 2]) specifications.push({ f16, subgroups, rows, groups });
    specifications.push({ row56: true }, { n: 3072, k: 1024 });
    for (const hidden of [2048, 2560, 4096]) specifications.push({ kev: hidden === 2048 ? { hidden } : { hidden, heads: 16, kv_heads: 4, lin_heads: 32 } });
    const compiled = [];
    const skipped = [];
    const failures = [];
    const languageFeatures = navigator.gpu.wgslLanguageFeatures;
    const started = performance.now();
    for (const kernel of kernels) {
      const seen = new Set();
      for (const spec of specifications) {
        if (kernel === "matmul_wide" && spec.groups === 2) continue;
        const code = source(kernel, spec);
        if (seen.has(code)) continue;
        seen.add(code);
        const required = [...code.matchAll(/\b(?:enable|requires)\s+([a-z0-9_]+)\s*;/g)].map((match) => match[1]);
        const unavailable = required.filter((name) => name === "f16" ? !device.features.has("shader-f16") : name === "subgroups" ? !runtime.subgroup4 : !languageFeatures?.has(name));
        if (unavailable.length) {
          skipped.push({ kernel, spec, unavailable });
          continue;
        }
        try {
          await pipeline(device, code, `${kernel} ${JSON.stringify(spec)}`);
          compiled.push({ kernel, spec });
        } catch (error) {
          const message = String(error?.message || error);
          const storage = message.match(/total use of workgroup storage \((\d+) bytes\) is larger than the maximum allowed \((\d+) bytes\)/);
          if (storage && Number(storage[1]) > device.limits.maxComputeWorkgroupStorageSize && Number(storage[2]) === device.limits.maxComputeWorkgroupStorageSize) {
            skipped.push({ kernel, spec, unavailable: ["workgroup storage"], requiredBytes: Number(storage[1]), availableBytes: Number(storage[2]) });
          } else {
            failures.push({ kernel, spec, error: message });
          }
        }
      }
      logNode.textContent = `${compiled.length} pipelines compiled; ${skipped.length} unsupported; ${failures.length} failures`;
    }
    window.gpuBench = {
      status: failures.length ? "error" : "done", done: true, backend: "webgpu",
      method: "GPU pipeline compilation wall clock", metricMs: (performance.now() - started) / Math.max(1, compiled.length),
      adapter: { name: runtime.name, vendor: runtime.adapter.info?.vendor ?? null, isFallbackAdapter: false },
      deviceFeatures: [...device.features], languageFeatures: [...(languageFeatures || [])],
      capabilities: { timestampQuery: device.features.has("timestamp-query"), subgroup4: runtime.subgroup4, subgroup8: runtime.subgroup4 && runtime.adapter.info.subgroupMinSize >= 8, subgroup16: runtime.subgroup4 && runtime.adapter.info.subgroupMinSize >= 16, subgroup32: runtime.subgroup32 },
      limits: { maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize },
      correctness: { ok: !failures.length, registeredKernels: kernels.length, compiled, skipped, failures },
      ...(failures.length ? { error: failures.map((failure) => `${failure.kernel}: ${failure.error}`).join("; ") } : {}),
    };
  } finally {
    device.destroy();
  }
}

run().catch((error) => {
  window.gpuBench = { status: "error", done: true, backend: "webgpu", error: String(error?.message || error) };
  logNode.textContent = String(error?.stack || error);
});
