import { Kevala } from "../js/src/index.js";

const button = document.getElementById("run");
const log = (value) => { document.getElementById("log").textContent += `${value}\n`; };
const questions = {
  person_present: { type: "noul", instructions: "Does the frame visibly contain at least one person?" },
  smiling: { type: "noul", instructions: "Does a visible person have a smiling or visibly happy facial expression? Judge only the visible expression, not inner emotion." },
  dominant_color: { type: "choice", instructions: "Which color covers most of the image?", criteria: { red: "red", blue: "blue" } },
};
const noulKeys = ["person_present", "smiling"];

function frame(color) {
  const canvas = new OffscreenCanvas(320, 240);
  const c = canvas.getContext("2d");
  c.fillStyle = color === "red" ? "#ff0000" : "#0000ff";
  c.fillRect(0, 0, 320, 240);
  const pixels = c.getImageData(0, 0, 320, 240);
  canvas.width = canvas.height = 1;
  return pixels;
}

async function main() {
  if (button.disabled) return;
  button.disabled = true;
  window.gpuBench = { status: "running", backend: "webgpu", phase: "load" };
  let model;
  let lastPhase;
  const started = performance.now();
  try {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    const adapterInfo = adapter?.info || {};
    const adapterDetails = {
      vendor: adapterInfo.vendor || null, architecture: adapterInfo.architecture || null,
      device: adapterInfo.device || null, description: adapterInfo.description || null,
      isFallbackAdapter: Boolean(adapterInfo.isFallbackAdapter || adapter?.isFallbackAdapter),
      supportedFeatures: adapter ? [...adapter.features].sort() : [],
      limits: adapter ? {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      } : null,
    };
    window.gpuBench.adapter = adapterDetails;
    const response = await fetch("/.kevala/models.json");
    if (!response.ok) throw new Error(`local pack inventory HTTP ${response.status}`);
    const inventory = (await response.json()).models["gemma-4-e2b"];
    if (!inventory?.q8 || !inventory.vision) throw new Error("the local Gemma E2B decoder and BF16 vision companion are required");
    model = await Kevala.load({
      model: new URL(inventory.q8.url, location.href).href,
      vision: new URL(inventory.vision.url, location.href).href,
      backend: "webgpu", cache: false, gpuKernel: "generic", imageTokens: 70,
      signal: AbortSignal.timeout(180000),
      onProgress: (progress) => {
        window.gpuBench.phase = progress.phase;
        if (progress.phase !== lastPhase) { log(`${progress.phase}: ${progress.message || ""}`); lastPhase = progress.phase; }
      },
    });
    const loadMs = performance.now() - started;
    Object.assign(window.gpuBench, {
      loadMs, info: model.info, memory: model.info.memory, pack: model.info.pack, vision: model.info.vision,
      adapter: { ...adapterDetails, name: model.info.gpu },
    });
    if (model.info.backend !== "webgpu" || !model.info.modalities.includes("image")) throw new Error("the loaded runtime does not expose WebGPU image support");
    if (model.info.memory?.hostWeights !== 2642411520 || model.info.memory.gpuWeights < 2800000000 || model.info.memory.gpuWeights > 3000000000) throw new Error(`unexpected image memory placement: ${JSON.stringify(model.info.memory)}`);
    log(`loaded in ${(loadMs / 1000).toFixed(2)}s; ${model.info.gpu}`);
    log(`memory: ${JSON.stringify(model.info.memory)}`);
    window.gpuBench.phase = "decide";
    const cases = [];
    window.gpuBench.correctness = { ok: false, modalities: model.info.modalities, imageTokens: model.info.imageTokens, maxInFlight: 1, cases };
    for (const color of ["red", "blue"]) {
      const pixels = frame(color);
      const t = performance.now();
      try {
        const result = await model.decide("Classify this camera frame.", questions, { parts: [{ type: "image", data: pixels }] });
        const scores = Object.fromEntries(noulKeys.map((key) => [key, result.answers?.[key]?.noul]));
        const probabilities = result.raw_probabilities;
        const dominantColor = result.answers?.dominant_color?.choice;
        const row = { id: color, expectedColor: color, dominantColor, answers: result.answers, scores, probabilities, elapsedMs: performance.now() - t, timing: result.timing };
        cases.push(row);
        log(JSON.stringify(row));
        for (const [key, score] of Object.entries(scores)) if (!Number.isFinite(score) || score < 0 || score > 1) throw new Error(`invalid ${key} score ${score}`);
        for (const key of Object.keys(questions)) {
          const values = probabilities?.[key];
          if (!Array.isArray(values) || values.length !== 2 || values.some((value) => !Number.isFinite(value) || value < 0 || value > 1) || Math.abs(values[0] + values[1] - 1) > 1e-4) throw new Error(`invalid ${key} raw probabilities`);
        }
        if (!Number.isFinite(result.timing?.total) || result.timing.total <= 0 || !Number.isSafeInteger(result.timing.tokens) || result.timing.tokens <= 0) throw new Error("missing image inference timing or token count");
        if (!["red", "blue"].includes(dominantColor)) throw new Error(`invalid dominant color answer ${dominantColor}`);
      } finally { pixels.data.fill(0); }
    }
    const maxScoreDelta = Math.max(...noulKeys.map((key) => Math.abs(cases[0].scores[key] - cases[1].scores[key])));
    const maxProbabilityDelta = Math.max(...Object.keys(questions).flatMap((key) => cases[0].probabilities[key].map((value, i) => Math.abs(value - cases[1].probabilities[key][i]))));
    const colorProbabilityDelta = Math.abs(cases[0].probabilities.dominant_color[0] - cases[1].probabilities.dominant_color[0]);
    const colorCorrect = cases.every((row) => row.dominantColor === row.expectedColor);
    Object.assign(window.gpuBench.correctness, { frames: cases.length, maxScoreDelta, maxProbabilityDelta, colorProbabilityDelta, colorCorrect, changed: maxProbabilityDelta > 0 });
    if (!maxProbabilityDelta) throw new Error("different image frames produced identical option probabilities; full image inference is unverified");
    if (!colorCorrect) throw new Error("the red/blue diagnostic did not identify both generated colors; inspect retained raw probabilities");
    let demoSample;
    let profiledSample;
    if (new URLSearchParams(location.search).get("profile") === "1") {
      const pixels = frame("red");
      try {
        const demoStarted = performance.now();
        const demoResult = await model.decide("Classify this camera frame.", Object.fromEntries(noulKeys.map((key) => [key, questions[key]])), { parts: [{ type: "image", data: pixels }] });
        demoSample = { elapsedMs: performance.now() - demoStarted, timing: demoResult.timing, answers: demoResult.answers };
        window.gpuBench.demoSample = demoSample;
        if (!(await model.profile(true))) throw new Error("GPU profiling is unavailable");
        const profileStarted = performance.now();
        const profileResult = await model.decide("Classify this camera frame.", questions, { parts: [{ type: "image", data: pixels }] });
        profiledSample = { elapsedMs: performance.now() - profileStarted, timing: profileResult.timing, answers: profileResult.answers };
        window.gpuBench.profiledSample = profiledSample;
        if (!Object.keys(profileResult.timing?.gpu || {}).some((key) => key.startsWith("vision."))) throw new Error("image kernels are missing from the public API profile");
        await model.profile(false);
      } finally { pixels.data.fill(0); }
    }
    window.gpuBench = {
      status: "done", done: true, backend: "webgpu", phase: "disposed",
      adapter: { ...adapterDetails, name: model.info.gpu }, info: model.info,
      metricMs: cases.reduce((sum, row) => sum + row.elapsedMs, 0) / cases.length,
      loadMs, pack: model.info.pack, vision: model.info.vision, memory: model.info.memory,
      ...(demoSample ? { demoSample, profiledSample } : {}),
      correctness: { ok: true, modalities: model.info.modalities, imageTokens: model.info.imageTokens, frames: 2, maxInFlight: 1, maxScoreDelta, maxProbabilityDelta, colorProbabilityDelta, colorCorrect, changed: true, cases },
    };
  } catch (error) {
    window.gpuBench = { ...window.gpuBench, status: "error", done: true, error: String(error?.message || error) };
    log(error.stack || error);
  } finally { model?.dispose(); button.disabled = false; }
}

window.gpuBench = { status: "idle", backend: "webgpu" };
button.addEventListener("click", main);
if (new URLSearchParams(location.search).get("autorun") === "1") main();
