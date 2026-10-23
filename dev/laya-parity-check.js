import { Kevala } from "../js/src/index.js";
import { LAYA_Q8_GOLDEN_MAX_DP, LAYA_WASM_GPU_MAX_DP, assessLayaParity } from "./laya-parity-metrics.js";

const PACK_URL = new URL("../tmp/laya-q8.kevala", import.meta.url).href;
const logNode = document.getElementById("log");
const log = (message) => { logNode.textContent += `${message}\n`; };
const label = "laya-q8-backend-parity";

window.gpuBench = { status: "running", backend: "webgpu", check: label };

function modelOptions(backend) {
  return {
    backend,
    cache: false,
    flavor: "simd",
    from: "pack",
    model: PACK_URL,
    precision: "q8",
    threads: 1,
  };
}

function validateModel(model, backend) {
  const info = model?.info;
  const expected = backend === "wasm" ? "wasm-simd" : "webgpu";
  if (!info || info.arch !== "laya" || info.backend !== expected || info.flavor !== "simd" || info.threads !== 1) {
    throw new Error(`${backend} loaded unexpected model metadata: ${JSON.stringify(info)}`);
  }
  if (!info.pack || !Number.isSafeInteger(info.pack.bytes) || info.pack.bytes <= 0) {
    throw new Error(`${backend} did not report a loaded Laya pack size`);
  }
  if (backend === "webgpu" && (typeof info.gpu !== "string" || !info.gpu)) throw new Error("WebGPU did not report the selected adapter");
  if (backend === "webgpu" && !["high-performance", "low-power"].includes(info.gpuPowerPreference)) throw new Error("WebGPU did not report the selected power preference");
  // precision:q8 makes the loader check the pack header before either backend accepts the weights.
  return { backend: info.backend, flavor: info.flavor, threads: info.threads, arch: info.arch, precision: "q8", packBytes: info.pack.bytes, gpu: info.gpu || null, gpuPowerPreference: info.gpuPowerPreference || null };
}

async function collect(model, golden, backend) {
  const rows = [];
  for (const fixture of golden.cases) {
    const response = await model.decide(fixture.state, fixture.questions);
    const forwardMs = response?.timing?.forward;
    if (backend === "webgpu" && (!Number.isFinite(forwardMs) || forwardMs <= 0)) {
      throw new Error(`WebGPU ${fixture.id} did not report a positive forward time`);
    }
    rows.push({ caseId: fixture.id, answers: response?.answers, forwardMs: Number.isFinite(forwardMs) ? forwardMs : null });
  }
  return rows;
}

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  if (!sorted.length || sorted.some((value) => !Number.isFinite(value) || value <= 0)) throw new Error("WebGPU forward timings must be finite positive values");
  return sorted[Math.floor(sorted.length / 2)];
}

async function run() {
  let wasmModel = null;
  let gpuModel = null;
  try {
    const response = await fetch("../tests/fixtures/golden.json");
    if (!response.ok) throw new Error(`could not load Laya golden fixtures: HTTP ${response.status}`);
    const golden = await response.json();

    log("Loading the local Q8 pack on single-threaded SIMD WASM.");
    wasmModel = await Kevala.load(modelOptions("wasm"));
    const wasmInfo = validateModel(wasmModel, "wasm");
    const wasmRows = await collect(wasmModel, golden, "wasm");
    wasmModel.dispose();
    wasmModel = null;

    log("WASM model disposed. Loading the same Q8 pack on WebGPU.");
    gpuModel = await Kevala.load(modelOptions("webgpu"));
    const gpuInfo = validateModel(gpuModel, "webgpu");
    if (gpuInfo.packBytes !== wasmInfo.packBytes) throw new Error("WASM and WebGPU loaded different pack sizes");
    const adapter = { name: gpuInfo.gpu, powerPreference: gpuInfo.gpuPowerPreference };
    const gpuRows = await collect(gpuModel, golden, "webgpu");
    const correctness = assessLayaParity(golden, wasmRows, gpuRows);
    const metricMs = median(gpuRows.map((row) => row.forwardMs));
    const result = {
      status: "done",
      done: true,
      check: label,
      backend: "webgpu",
      gpu: gpuInfo.gpu,
      adapter,
      model: "laya",
      precision: "q8",
      pack: { url: PACK_URL, bytes: gpuInfo.packBytes },
      wasm: wasmInfo,
      webgpu: gpuInfo,
      metricMs,
      metric: "median measured WebGPU API forward time over all golden fixtures",
      correctness,
    };
    window.gpuBench = result;
    log(JSON.stringify(result));
  } catch (error) {
    const message = String(error?.message || error);
    window.gpuBench = {
      status: "error",
      done: true,
      check: label,
      backend: "webgpu",
      model: "laya",
      precision: "q8",
      correctness: {
        ok: false,
        tolerances: { wasmWebgpuMaxDp: LAYA_WASM_GPU_MAX_DP, q8GoldenMaxDp: LAYA_Q8_GOLDEN_MAX_DP },
      },
      error: message,
    };
    log(`ERROR ${message}`);
  } finally {
    wasmModel?.dispose();
    gpuModel?.dispose();
  }
}

run();
