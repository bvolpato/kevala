const log = document.getElementById("log");
const params = new URLSearchParams(location.search);
window.gpuBench = { status: "running", done: false, backend: "webgpu", method: "capability-only" };

function finiteOrNull(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function matrixConfig(config) {
  return {
    componentType: config.componentType,
    resultComponentType: config.resultComponentType,
    M: finiteOrNull(config.M),
    N: finiteOrNull(config.N),
    K: finiteOrNull(config.K),
  };
}

async function main() {
  if (params.get("capabilities") !== "1") throw new Error("Open this page with ?capabilities=1");
  if (!navigator.gpu) throw new Error("WebGPU is unavailable in this browser");

  let adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  let adapterSelection = "high-performance";
  if (!adapter) {
    adapter = await navigator.gpu.requestAdapter();
    adapterSelection = "default-after-high-performance-unavailable";
  }
  if (!adapter) throw new Error("WebGPU returned no adapter for high-performance or default selection");

  const info = adapter.info ?? {};
  const supportedFeatures = [...(adapter.features ?? [])].map(String).sort();
  const matrixConfigs = info.subgroupMatrixConfigs;
  let wgslLanguageFeatures = null;
  try {
    if (navigator.gpu.wgslLanguageFeatures) {
      wgslLanguageFeatures = [...navigator.gpu.wgslLanguageFeatures].map(String).sort();
    }
  } catch {
    // Older browser builds may not expose WGSL language feature enumeration.
  }

  const isFallbackAdapter = Boolean(info.isFallbackAdapter ?? adapter.isFallbackAdapter);
  const result = {
    status: "done",
    done: true,
    backend: "webgpu",
    method: "capability-only",
    metricMs: 1,
    metricNote: "Contract sentinel only; this capability query does not measure performance.",
    adapter: {
      name: info.device || info.description || null,
      vendor: info.vendor || null,
      architecture: info.architecture || null,
      device: info.device || null,
      description: info.description || null,
      backend: info.backend || null,
      driver: info.driver || null,
      type: info.type || null,
      isFallbackAdapter,
      selection: adapterSelection,
      highPerformanceAdapterUnavailable: adapterSelection !== "high-performance",
    },
    capabilities: {
      supportedFeatures,
      shaderF16: supportedFeatures.includes("shader-f16"),
      subgroupMatrix: supportedFeatures.includes("chromium-experimental-subgroup-matrix"),
      subgroupMatrixConfigs: Array.isArray(matrixConfigs) ? Array.from(matrixConfigs, matrixConfig) : null,
      subgroupMatrixConfigQueryAvailable: Array.isArray(matrixConfigs),
      subgroupMinSize: finiteOrNull(info.subgroupMinSize ?? adapter.limits?.minSubgroupSize),
      subgroupMaxSize: finiteOrNull(info.subgroupMaxSize ?? adapter.limits?.maxSubgroupSize),
      wgslLanguageFeatures,
    },
    correctness: {
      ok: true,
      method: "capability-query-only",
      numericKernelTest: "not-run",
      gpuDispatches: 0,
    },
  };
  window.gpuBench = result;
  log.textContent = JSON.stringify(result, null, 2);
}

main().catch((error) => {
  const message = String(error?.message || error);
  window.gpuBench = {
    status: "error",
    done: true,
    backend: "webgpu",
    method: "capability-only",
    error: message,
    correctness: { ok: false, method: "capability-query-only", numericKernelTest: "not-run", gpuDispatches: 0 },
  };
  log.textContent = String(error?.stack || message);
});
