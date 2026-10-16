import { Kevala, MODELS } from "../js/src/index.js";

const WIDTH = 320;
const HEIGHT = 240;
const STATE = "Classify this camera frame.";
const QUESTIONS = {
  person_present: { type: "noul", instructions: "Does the frame visibly contain at least one person?" },
  smiling: { type: "noul", instructions: "Does a visible person have a smiling or visibly happy facial expression? Judge only the visible expression, not inner emotion." },
};
const KEYS = Object.keys(QUESTIONS);
const SAMPLE_COUNT = 12;
const WARMUPS = 2;
const BLOCK_SIZE = 3;
const DEFAULT_BUDGETS = [70, 40, 16];

const button = document.getElementById("run");
const log = (value) => { document.getElementById("log").textContent += `${value}\n`; };

function budgetsFromUrl(params) {
  const value = params.get("budgets");
  if (!value) return DEFAULT_BUDGETS;
  const budgets = value.split(",").map((item) => Number(item.trim()));
  if (!budgets.length || budgets.some((item) => !Number.isSafeInteger(item) || item < 1 || item > 70) || !budgets.includes(70) || !budgets.some((item) => item < 70) || new Set(budgets).size !== budgets.length) {
    throw new Error("budgets must be unique integers from 1 to 70, include 70, and include a lower budget");
  }
  return budgets;
}

function quantile(values, q) {
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * q;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

function validateResult(result) {
  const scores = {};
  for (const key of KEYS) {
    const score = result.answers?.[key]?.noul;
    if (!Number.isFinite(score) || score < 0 || score > 1) throw new Error(`invalid ${key} score ${score}`);
    scores[key] = score;
    const probabilities = result.raw_probabilities?.[key];
    if (!Array.isArray(probabilities) || probabilities.length !== 2 || probabilities.some((value) => !Number.isFinite(value) || value < 0 || value > 1) || Math.abs(probabilities[0] + probabilities[1] - 1) > 1e-4) {
      throw new Error(`invalid ${key} raw probabilities`);
    }
  }
  const actualSoftTokens = result.timing?.image?.softTokens;
  if (!Number.isSafeInteger(actualSoftTokens) || actualSoftTokens <= 0) throw new Error("image timing did not report the actual soft-token count");
  if (!Number.isFinite(result.timing?.total) || result.timing.total <= 0) throw new Error("missing completed inference timing");
  return { scores, actualSoftTokens };
}

function makeSyntheticFrames(count = SAMPLE_COUNT) {
  const canvas = document.createElement("canvas");
  canvas.width = WIDTH;
  canvas.height = HEIGHT;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("a 2D canvas is required to prepare benchmark frames");
  const frames = [];
  for (let index = 0; index < count; index++) {
    const phase = index / count;
    const background = context.createLinearGradient(0, 0, WIDTH, HEIGHT);
    background.addColorStop(0, `hsl(${(index * 31) % 360} 42% 27%)`);
    background.addColorStop(1, `hsl(${(index * 31 + 105) % 360} 38% 66%)`);
    context.fillStyle = background;
    context.fillRect(0, 0, WIDTH, HEIGHT);
    context.globalAlpha = 0.25;
    context.strokeStyle = "#ffffff";
    context.lineWidth = 2;
    for (let line = 0; line < 9; line++) {
      const x = ((line * 47 + index * 13) % WIDTH);
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo((x + 48 + index * 3) % WIDTH, HEIGHT);
      context.stroke();
    }
    context.globalAlpha = 1;
    const centerX = WIDTH * (0.47 + 0.07 * Math.sin(phase * Math.PI * 2));
    const centerY = 96 + 8 * Math.cos(phase * Math.PI * 2);
    context.fillStyle = "#423b38";
    context.beginPath();
    context.ellipse(centerX, centerY + 95, 65, 48, 0, Math.PI, Math.PI * 2);
    context.fill();
    context.fillStyle = ["#d7a17b", "#b77a58", "#edc19d"][index % 3];
    context.beginPath();
    context.ellipse(centerX, centerY, 39, 48, Math.sin(phase * 2) * 0.12, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = "#282522";
    for (const eyeOffset of [-13, 13]) {
      context.beginPath();
      context.ellipse(centerX + eyeOffset, centerY - 6, 3, 2 + (index % 2), 0, 0, Math.PI * 2);
      context.fill();
    }
    context.strokeStyle = "#6f2825";
    context.lineWidth = 3;
    context.beginPath();
    const mouthY = centerY + 17;
    if (index % 3 === 0) context.ellipse(centerX, mouthY, 8, 5, 0, 0, Math.PI);
    else if (index % 3 === 1) { context.moveTo(centerX - 9, mouthY); context.lineTo(centerX + 9, mouthY); }
    else context.ellipse(centerX, mouthY - 4, 5, 7, 0, 0, Math.PI * 2);
    context.stroke();
    const pixels = context.getImageData(0, 0, WIDTH, HEIGHT);
    frames.push({ id: `synthetic-${index}`, pixels: new Uint8ClampedArray(pixels.data) });
    context.clearRect(0, 0, WIDTH, HEIGHT);
  }
  canvas.width = canvas.height = 1;
  return frames;
}

function validateManifestRow(row, index) {
  if (!row || typeof row !== "object" || typeof row.url !== "string" || !row.url.trim()) throw new Error(`image manifest row ${index} needs a URL`);
  for (const key of ["expectedPerson", "expectedSmile"]) {
    if (row[key] != null && typeof row[key] !== "boolean") throw new Error(`image manifest row ${index} ${key} must be true or false`);
  }
  return { id: row.id == null ? `image-${index}` : String(row.id), url: row.url, ...(row.expectedPerson == null ? {} : { expectedPerson: row.expectedPerson }), ...(row.expectedSmile == null ? {} : { expectedSmile: row.expectedSmile }) };
}

async function loadManifestFrames(manifestUrl) {
  const response = await fetch(manifestUrl);
  if (!response.ok) throw new Error(`image manifest HTTP ${response.status}`);
  const body = await response.json();
  const rows = Array.isArray(body) ? body : body?.images;
  if (!Array.isArray(rows) || !rows.length || rows.length > SAMPLE_COUNT) throw new Error(`image manifest must contain 1 to ${SAMPLE_COUNT} rows`);
  const frames = [];
  const canvas = document.createElement("canvas");
  canvas.width = WIDTH;
  canvas.height = HEIGHT;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("a 2D canvas is required to prepare manifest frames");
  try {
    for (let index = 0; index < rows.length; index++) {
      const row = validateManifestRow(rows[index], index);
      const imageUrl = new URL(row.url, manifestUrl);
      const imageResponse = await fetch(imageUrl);
      if (!imageResponse.ok) throw new Error(`image ${row.id} HTTP ${imageResponse.status}`);
      const imageBytes = Number(imageResponse.headers.get("content-length"));
      if (imageBytes > 32 * 1024 * 1024) { await imageResponse.body?.cancel(); throw new Error(`image ${row.id} exceeds 32 MiB`); }
      const bitmap = await createImageBitmap(await imageResponse.blob());
      try {
        const targetAspect = WIDTH / HEIGHT;
        const sourceAspect = bitmap.width / bitmap.height;
        let cropX = 0;
        let cropY = 0;
        let cropWidth = bitmap.width;
        let cropHeight = bitmap.height;
        if (sourceAspect > targetAspect) {
          cropWidth = bitmap.height * targetAspect;
          cropX = (bitmap.width - cropWidth) / 2;
        } else {
          cropHeight = bitmap.width / targetAspect;
          cropY = (bitmap.height - cropHeight) / 2;
        }
        context.drawImage(bitmap, cropX, cropY, cropWidth, cropHeight, 0, 0, WIDTH, HEIGHT);
        frames.push({ ...row, pixels: new Uint8ClampedArray(context.getImageData(0, 0, WIDTH, HEIGHT).data) });
        context.clearRect(0, 0, WIDTH, HEIGHT);
      } finally { bitmap.close(); }
    }
  } finally {
    canvas.width = canvas.height = 1;
  }
  return frames;
}

function imageData(frame) {
  return new ImageData(new Uint8ClampedArray(frame.pixels), WIDTH, HEIGHT);
}

function summarizeBudgets(samples, budgets) {
  return Object.fromEntries(budgets.map((budget) => {
    const rows = samples.filter((row) => row.budget === budget);
    const times = rows.map((row) => row.elapsedMs);
    const totalMs = times.reduce((sum, value) => sum + value, 0);
    return [budget, {
      requestedImageTokens: budget,
      actualSoftTokens: [...new Set(rows.map((row) => row.actualSoftTokens))],
      samples: rows.length,
      p50Ms: quantile(times, 0.5),
      p95Ms: quantile(times, 0.95),
      meanMs: totalMs / rows.length,
      completedFps: rows.length * 1000 / totalMs,
      scoreMeans: Object.fromEntries(KEYS.map((key) => [key, rows.reduce((sum, row) => sum + row.scores[key], 0) / rows.length])),
    }];
  }));
}

function compareWithDetail(samples, budgets) {
  const detail = new Map(samples.filter((row) => row.budget === 70).map((row) => [row.sampleIndex, row]));
  return Object.fromEntries(budgets.filter((budget) => budget !== 70).map((budget) => {
    const rows = samples.filter((row) => row.budget === budget);
    const signChanges = Object.fromEntries(KEYS.map((key) => [key, 0]));
    let maxProbabilityDelta = 0;
    let comparedSamples = 0;
    for (const row of rows) {
      const baseline = detail.get(row.sampleIndex);
      if (!baseline) continue;
      comparedSamples++;
      for (const key of KEYS) {
        if ((row.scores[key] >= 0.5) !== (baseline.scores[key] >= 0.5)) signChanges[key]++;
        for (let index = 0; index < 2; index++) maxProbabilityDelta = Math.max(maxProbabilityDelta, Math.abs(row.rawProbabilities[key][index] - baseline.rawProbabilities[key][index]));
      }
    }
    return [budget, { comparedSamples, decisionSignChanges: signChanges, maxProbabilityDelta }];
  }));
}

function labelChecks(samples) {
  const checks = [];
  for (const row of samples) {
    for (const [key, label] of [["person_present", row.expectedPerson], ["smiling", row.expectedSmile]]) {
      if (label == null) continue;
      checks.push({ id: row.id, sampleIndex: row.sampleIndex, budget: row.budget, key, expected: label, predicted: row.scores[key] >= 0.5, correct: (row.scores[key] >= 0.5) === label, score: row.scores[key] });
    }
  }
  return { count: checks.length, correct: checks.filter((item) => item.correct).length, accuracy: checks.length ? checks.filter((item) => item.correct).length / checks.length : null, checks };
}

async function main() {
  if (button.disabled) return;
  button.disabled = true;
  document.getElementById("download").disabled = true;
  const params = new URLSearchParams(location.search);
  const gpuKernel = params.get("kernel") || "auto";
  const visionAttention = params.get("attention") || "auto";
  let budgets;
  let model;
  let lastPhase;
  const runStarted = performance.now();
  let frames = [];
  let allocationFrames = [];
  window.gpuBench = { status: "running", backend: "webgpu", phase: "validate" };
  try {
    budgets = budgetsFromUrl(params);
    const adapter = (await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" })) || (await navigator.gpu?.requestAdapter());
    if (!adapter) throw new Error("WebGPU adapter is unavailable");
    const info = adapter.info || {};
    const adapterDetails = {
      vendor: info.vendor || null, architecture: info.architecture || null,
      device: info.device || null, description: info.description || null,
      isFallbackAdapter: Boolean(info.isFallbackAdapter || adapter.isFallbackAdapter),
      supportedFeatures: [...adapter.features].sort(),
      limits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      },
    };
    window.gpuBench.adapter = adapterDetails;
    const source = params.get("images");
    const manifestUrl = source ? new URL(source, location.href).href : null;
    window.gpuBench.phase = "frames";
    frames = manifestUrl ? await loadManifestFrames(manifestUrl) : makeSyntheticFrames();
    allocationFrames = manifestUrl ? makeSyntheticFrames(3) : frames.slice(0, 3);
    if (!frames.length) throw new Error("no benchmark frames were prepared");
    const hosted = params.get("hosted") === "1";
    let modelSource = "gemma-4-e2b";
    let visionSource = MODELS[modelSource].vision.hosted;
    if (!hosted) {
      const response = await fetch("/.kevala/models.json");
      if (!response.ok) throw new Error(`local pack inventory HTTP ${response.status}; use ?hosted=1 for the published packs`);
      const inventory = (await response.json()).models?.["gemma-4-e2b"];
      if (!inventory?.q8 || !inventory.vision) throw new Error("the local Gemma E2B decoder and BF16 vision companion are required");
      modelSource = new URL(inventory.q8.url, location.href).href;
      visionSource = new URL(inventory.vision.url, location.href).href;
    }

    window.gpuBench.phase = "load";
    const loadStarted = performance.now();
    model = await Kevala.load({
      model: modelSource, vision: visionSource,
      backend: "webgpu", cache: hosted, gpuKernel, visionAttention, imageTokens: 70, profile: false,
      signal: AbortSignal.timeout(hosted ? 30 * 60 * 1000 : 180000),
      onProgress: (progress) => {
        window.gpuBench.phase = progress.phase;
        if (progress.phase !== lastPhase) { log(`${progress.phase}: ${progress.message || ""}`); lastPhase = progress.phase; }
      },
    });
    const loadMs = performance.now() - loadStarted;
    const offGpu = model.info.memory?.hostWeights + (model.info.memory.opfsWeights || 0);
    if (model.info.backend !== "webgpu" || !model.info.modalities.includes("image")) throw new Error("the loaded runtime does not expose WebGPU image support");
    if (model.info.imageTokens !== 70) throw new Error(`expected a 70-token model capacity, got ${model.info.imageTokens}`);
    if ((offGpu !== 2642411520 && offGpu !== 3095396352) || model.info.memory.gpuWeights < 2400000000 || model.info.memory.gpuWeights > 3000000000) throw new Error(`unexpected image memory placement: ${JSON.stringify(model.info.memory)}`);
    log(`loaded in ${(loadMs / 1000).toFixed(2)}s; ${model.info.gpu}`);
    log(`memory: ${JSON.stringify(model.info.memory)}`);

    let inFlight = 0;
    let maxInFlight = 0;
    const warmups = [];
    const samples = [];
    const allocationSequence = [];
    const expectedFrame = (sampleIndex) => frames[sampleIndex % frames.length];
    const infer = async (frame, budget, sampleIndex, phase) => {
      const pixels = imageData(frame);
      const started = performance.now();
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      try {
        const result = await model.decide(STATE, QUESTIONS, {
          parts: [{ type: "image", data: pixels }], imageTokens: budget,
        });
        const elapsedMs = performance.now() - started;
        const { scores, actualSoftTokens } = validateResult(result);
        if (actualSoftTokens > budget) throw new Error(`budget ${budget} produced ${actualSoftTokens} soft tokens`);
        const row = {
          phase, id: frame.id, sampleIndex, frameIndex: sampleIndex % frames.length,
          budget, requestedImageTokens: budget, actualSoftTokens, elapsedMs,
          completedFps: 1000 / elapsedMs, scores,
          rawProbabilities: result.raw_probabilities,
          timing: result.timing,
          ...(frame.expectedPerson == null ? {} : { expectedPerson: frame.expectedPerson }),
          ...(frame.expectedSmile == null ? {} : { expectedSmile: frame.expectedSmile }),
        };
        if (phase === "sample") {
          samples.push(row);
          log(`${budget} tokens, frame ${sampleIndex + 1}: ${actualSoftTokens} actual tokens, ${elapsedMs.toFixed(1)} ms, ${JSON.stringify(scores)}`);
        } else if (phase === "warmup") warmups.push(row);
        return row;
      } finally {
        inFlight--;
        pixels.data.fill(0);
      }
    };

    window.gpuBench = {
      ...window.gpuBench, phase: "warmup", loadMs, adapter: { ...adapterDetails, name: model.info.gpu },
      info: model.info, budgets, warmups, samples, source: manifestUrl ? "manifest" : "synthetic",
    };
    for (const budget of budgets) {
      for (let index = 0; index < WARMUPS; index++) await infer(expectedFrame(index), budget, index, "warmup");
      window.gpuBench.warmups = warmups;
    }

    window.gpuBench.phase = "benchmark";
    const blocks = SAMPLE_COUNT / BLOCK_SIZE;
    if (!Number.isInteger(blocks)) throw new Error("sample count must divide evenly into benchmark blocks");
    for (let block = 0; block < blocks; block++) {
      const offset = block % budgets.length;
      const order = [...budgets.slice(offset), ...budgets.slice(0, offset)];
      for (const budget of order) {
        for (let position = 0; position < BLOCK_SIZE; position++) {
          const sampleIndex = block * BLOCK_SIZE + position;
          await infer(expectedFrame(sampleIndex), budget, sampleIndex, "sample");
          window.gpuBench.samples = samples;
        }
      }
    }

    window.gpuBench.phase = "shape-reallocation";
    const smallestBudget = Math.min(...budgets);
    for (const [index, budget] of [smallestBudget, 70, smallestBudget].entries()) {
      allocationSequence.push(await infer(allocationFrames[index], budget, index, "shapeSequence"));
    }
    const allocationTokens = allocationSequence.map((row) => row.actualSoftTokens);
    const allocationPixelsChanged = new Set(allocationFrames.map((frame) => {
      let hash = 2166136261;
      for (let index = 0; index < frame.pixels.length; index += 97) hash = Math.imul(hash ^ frame.pixels[index], 16777619);
      return hash >>> 0;
    })).size === allocationFrames.length;
    const lowHighLowReallocation = allocationPixelsChanged && allocationTokens[0] === allocationTokens[2] && allocationTokens[1] > allocationTokens[0] && allocationSequence.length === 3;
    if (!lowHighLowReallocation) throw new Error(`low-high-low image shape reallocation failed: tokens=${allocationTokens.join(",")}, changedPixels=${allocationPixelsChanged}`);

    const summary = summarizeBudgets(samples, budgets);
    const comparison = compareWithDetail(samples, budgets);
    const labels = labelChecks(samples);
    window.gpuBench.phase = "mixed-budget-batch";
    const batchBudgets = [smallestBudget, 70, budgets.find((budget) => budget !== 70)];
    const batchPixels = batchBudgets.map((_, index) => imageData(expectedFrame(index)));
    let mixedBudgetBatch;
    try {
      const results = await model.decideMany(batchBudgets.map((budget, index) => ({
        state: STATE, questions: QUESTIONS, imageTokens: budget,
        parts: [{ type: "image", data: batchPixels[index] }],
      })));
      let maxProbabilityDelta = 0;
      results.forEach((result, index) => {
        validateResult(result);
        const control = samples.find((row) => row.budget === batchBudgets[index] && row.sampleIndex === index);
        for (const key of KEYS) for (let option = 0; option < 2; option++) {
          maxProbabilityDelta = Math.max(maxProbabilityDelta, Math.abs(result.raw_probabilities[key][option] - control.rawProbabilities[key][option]));
        }
      });
      if (results.length !== batchBudgets.length || maxProbabilityDelta > 5e-4) throw new Error(`mixed-budget batch differs from independent decisions: ${maxProbabilityDelta}`);
      mixedBudgetBatch = { budgets: batchBudgets, maxProbabilityDelta, timing: results[0].timing };
    } finally { for (const pixels of batchPixels) pixels.data.fill(0); }
    let profiledSample;
    if (params.get("profile") === "1") {
      const profileBudget = Number(params.get("profileBudget") || 70);
      if (!budgets.includes(profileBudget)) throw new Error("profileBudget must be one of the benchmark budgets");
      const supported = await model.profile(true);
      if (!supported) throw new Error("GPU timestamp profiling is unavailable");
      try {
        profiledSample = await infer(expectedFrame(0), profileBudget, 0, "profile");
        profiledSample.gpu = profiledSample.timing.gpu || null;
        if (!Object.keys(profiledSample.gpu || {}).some((key) => key.startsWith("vision."))) throw new Error("image kernels are missing from the public API profile");
      } finally { await model.profile(false); }
    }

    if (samples.length !== SAMPLE_COUNT * budgets.length || maxInFlight !== 1 || inFlight !== 0) throw new Error("benchmark did not complete sequential samples");
    const tokenCounts = Object.fromEntries(budgets.map((budget) => [budget, summary[budget].actualSoftTokens]));
    if (tokenCounts[70]?.[0] <= tokenCounts[16]?.[0]) throw new Error("70-token and 16-token budgets did not exercise different image shapes");
    const correctness = {
      ok: true, maxInFlight, samples: samples.length, warmups: warmups.length,
      budgets, actualSoftTokens: tokenCounts, allScoresFinite: true,
      allRawProbabilitiesFinite: true, allSamplesWithinBudget: true,
      shapeChanged: new Set(Object.values(tokenCounts).map((values) => values[0])).size > 1,
      lowHighLowReallocation: { ok: lowHighLowReallocation, requestedBudgets: [smallestBudget, 70, smallestBudget], actualSoftTokens: allocationTokens, changedPixels: allocationPixelsChanged, samples: allocationSequence },
      mixedBudgetBatch,
      labelChecks: labels,
    };
    window.gpuBench = {
      status: "done", done: true, backend: "webgpu", phase: "disposed",
      adapter: { ...adapterDetails, name: model.info.gpu }, info: model.info,
      pack: model.info.pack, vision: model.info.vision, memory: model.info.memory,
      loadMs, elapsedMs: performance.now() - runStarted,
      metricMs: summary[40]?.meanMs ?? summary[Math.min(...budgets)].meanMs,
      gpuKernel, visionAttention, source: manifestUrl ? "manifest" : "synthetic", manifestUrl,
      weights: hosted ? "published" : "local", userAgent: navigator.userAgent,
      frameSize: { width: WIDTH, height: HEIGHT }, questions: QUESTIONS, budgets,
      warmupCountPerBudget: WARMUPS, samplesPerBudget: SAMPLE_COUNT, blockSize: BLOCK_SIZE,
      schedule: "3-sample budget blocks, order rotated once per block",
      summary, comparisonWith70: comparison, labelChecks: labels, lowHighLowReallocation: correctness.lowHighLowReallocation, mixedBudgetBatch,
      warmups, samples, allocationSequence, ...(profiledSample ? { profiledSample } : {}), correctness,
    };
    for (const frame of frames) frame.pixels.fill(0);
    for (const frame of allocationFrames) frame.pixels.fill(0);
    for (const [budget, stats] of Object.entries(summary)) log(`summary ${budget}: ${stats.actualSoftTokens.join(",")} actual tokens, p50 ${stats.p50Ms.toFixed(1)} ms, p95 ${stats.p95Ms.toFixed(1)} ms, ${stats.completedFps.toFixed(2)} completed FPS`);
    log(`comparison with 70: ${JSON.stringify(comparison)}`);
    if (labels.count) log(`labeled image checks: ${labels.correct}/${labels.count} correct; inspect per-image scores and budget changes`);
  } catch (error) {
    window.gpuBench = { ...window.gpuBench, status: "error", done: true, error: String(error?.message || error) };
    log(error.stack || error);
  } finally {
    for (const frame of frames) frame.pixels.fill(0);
    for (const frame of allocationFrames) frame.pixels.fill(0);
    model?.dispose();
    button.disabled = false;
    document.getElementById("download").disabled = false;
  }
}

window.gpuBench = { status: "idle", backend: "webgpu" };
button.addEventListener("click", main);
document.getElementById("download").addEventListener("click", () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(window.gpuBench, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "kevala-camera-benchmark.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
if (new URLSearchParams(location.search).get("autorun") === "1") main();
