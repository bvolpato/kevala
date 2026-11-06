import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { open, stat } from "node:fs/promises";
import { packHeaderLength, packPrecision, packSize, parsePackHeader } from "../../js/src/pack-layout.js";

export const PRECISIONS = ["q8", "bf16", "f32"];
export const SHAPES = ["1 question, ~40 tokens", "1 question, ~128 tokens", "1 question, ~512 tokens", "5 questions, ~60 tokens each", "32 states x 1 question (~45 tokens)"];
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const check = (condition, message) => { if (!condition) throw new Error(message); };

export function classifyProfileFailure(error, { stage = "browser" } = {}) {
  const reason = String(error?.message || error);
  const failed = { status: "failed", reasonCode: "measurement-failed", reason };
  if (stage !== "browser") return failed;
  if (error?.code === "ETIMEDOUT" || error?.code === "PROFILE_DEADLINE" || /page did not expose a completed result before timeout|timed out receiving message from renderer/i.test(reason)) {
    return { status: "blocked", reasonCode: "controller-deadline", reason };
  }
  if (error?.code === "ENOMEM" || /JavaScript heap out of memory|Array buffer allocation failed|WebAssembly\.Memory\(\).*could not allocate memory/i.test(reason)) {
    return { status: "blocked", reasonCode: "host-memory-limit", reason };
  }
  if (error?.name === "GPUOutOfMemoryError" || /GPUOutOfMemoryError|out[\s-]+of[\s-]+GPU[\s-]+memory/i.test(reason) || /\b(?:WebGPU|GPU)\b/i.test(reason) && /out[\s-]+of[\s-]+memory|not enough memory left/i.test(reason)) {
    return { status: "blocked", reasonCode: "gpu-memory-limit", reason };
  }
  if (/\b(?:maxBufferSize|maxStorageBufferBindingSize|maxComputeWorkgroupStorageSize)\b/i.test(reason) && /exceed|larger than|maximum allowed|required.*limit/i.test(reason)) {
    return { status: "blocked", reasonCode: "gpu-device-limit", reason };
  }
  if (/timestamp-query is unavailable|selected a software WebGPU adapter|no available adapters|navigator\.gpu is missing|WebGPU is not supported/i.test(reason)) {
    return { status: "unsupported", reasonCode: "gpu-capability", reason };
  }
  return failed;
}

export function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function metrics(values, blockValues) {
  check(values.length > 0 && values.every((value) => finite(value) && value >= 0), "Invalid timing samples");
  const sorted = [...values].sort((a, b) => a - b);
  return { medianMs: median(values), meanMs: values.reduce((sum, value) => sum + value, 0) / values.length,
    p95Ms: sorted[Math.ceil(sorted.length * 0.95) - 1], samples: values,
    blockMediansMs: blockValues.map(median) };
}

export function profilePlan(catalog, { models = Object.keys(catalog), precisions = ["q8"], browsers = ["firefox"] } = {}) {
  check(models.length > 0 && new Set(models).size === models.length && models.every((id) => Object.hasOwn(catalog, id)), "Models must be unique catalog IDs");
  check(browsers.length > 0 && new Set(browsers).size === browsers.length && browsers.every((browser) => ["firefox", "chromium"].includes(browser)), "Browsers must be firefox or chromium");
  check(precisions === "all" || precisions.length > 0 && new Set(precisions).size === precisions.length && precisions.every((value) => PRECISIONS.includes(value)), "Precisions must be q8, bf16, f32, or all");
  return models.flatMap((id) => {
    const spec = catalog[id];
    const supported = spec.precisions || [spec.defaultPrecision || "q8"];
    return (precisions === "all" ? PRECISIONS.filter((value) => supported.includes(value)) : precisions).flatMap((precision) => browsers.map((browser) => ({
      id, name: spec.label || id, arch: spec.arch, precision, browser,
      status: supported.includes(precision) ? "pending" : "unsupported",
      ...(supported.includes(precision) ? {} : { reason: `Catalog does not support ${precision}` }),
    })));
  });
}

export async function fileHash(path) {
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  return digest.digest("hex");
}

export function responseModelName(header) {
  const arch = header.config?.arch || "laya";
  if (arch === "laya") return "laya-rl-agent";
  if (arch === "kev") {
    if (header.config.readout === "h2o-lightning") {
      check(typeof header.config.h2o_config?.model === "string" && header.config.h2o_config.model.length > 0, "H2O pack has no response model identity");
      return header.config.h2o_config.model;
    }
    if (!["semif", "direct-options"].includes(header.config.readout)) return "kev-latest";
  }
  return header.model.name;
}

export function verifyPackHeader(header, headerBytes, bytes, id, precision, spec) {
  check(header.model?.name === (spec.name || id), `${id}: pack model name differs from catalog`);
  check((header.config?.arch || "laya") === spec.arch, `${id}: pack architecture differs from catalog`);
  check(packPrecision(header) === precision, `${id}: pack precision does not match ${precision}`);
  check(header.model?.source === `https://huggingface.co/${spec.repo}` && header.model?.revision === spec.revision, `${id}: pack source or revision differs from catalog`);
  if (spec.base) check(header.model.base === `https://huggingface.co/${spec.base.repo}` && header.model.base_revision === spec.base.revision, `${id}: pack base revision differs from catalog`);
  const minimum = packSize(header, headerBytes);
  check(minimum === bytes || Math.ceil(minimum / 64) * 64 === bytes, `${id}: pack file length disagrees with its tensor layout`);
  const expected = spec.precisionPackBytes?.[precision] ?? (precision === "q8" ? spec.pack : null);
  if (Number.isSafeInteger(expected) && spec.precisionPackApprox?.[precision] !== true) check(bytes === expected, `${id}: pack byte count differs from catalog`);
  const arch = header.config?.arch || "laya";
  const answerUnit = arch === "gemma4" || arch === "kev" && header.config.readout !== "h2o-lightning" && header.config.response_version !== 1 ? 0.01 : 0.0001;
  return { model: header.model, arch, precision, responseModel: responseModelName(header), answerUnit };
}

export async function inspectPack(path, id, precision, spec) {
  const before = await stat(path);
  check(before.isFile(), `${id}: pack path is not a regular file`);
  const file = await open(path, "r");
  let identity;
  let headerSha256;
  let headerBytes;
  try {
    const prefix = Buffer.alloc(16);
    check((await file.read(prefix, 0, 16, 0)).bytesRead === 16 && prefix.toString("ascii", 0, 4) === "KVLA", `${id}: invalid pack prefix`);
    const length = packHeaderLength(prefix);
    check(length >= 16 && length <= Math.min(before.size, 16 * 1024 * 1024), `${id}: invalid pack header length`);
    const bytes = Buffer.alloc(length);
    check((await file.read(bytes, 0, length, 0)).bytesRead === length, `${id}: truncated pack header`);
    identity = verifyPackHeader(parsePackHeader(bytes), length, before.size, id, precision, spec);
    headerSha256 = createHash("sha256").update(bytes).digest("hex");
    headerBytes = length;
  } finally { await file.close(); }
  const sha256 = await fileHash(path);
  const after = await stat(path);
  check(before.size === after.size && before.mtimeMs === after.mtimeMs && before.ino === after.ino, `${id}: pack changed while hashing`);
  const pinned = precision === "q8" ? spec.packSha256 : spec.precisionPackSha256?.[precision];
  if (pinned) check(sha256 === pinned, `${id}: pack SHA-256 differs from catalog`);
  return { ...identity, bytes: before.size, sha256, headerSha256, headerBytes,
    verification: pinned ? "catalog-pinned SHA-256, header identity, layout, and size" : "local conversion SHA-256, header identity, layout, and size; no catalog-pinned digest",
    catalogHashVerified: Boolean(pinned), file: { size: after.size, mtimeMs: after.mtimeMs, ino: after.ino } };
}

export function adapterKey(adapter) {
  check(adapter && adapter.isFallbackAdapter === false && typeof adapter.name === "string" && !/swiftshader|llvmpipe|software/i.test(adapter.name), "Hardware WebGPU adapter is required");
  check(Array.isArray(adapter.features) && adapter.features.includes("timestamp-query"), "timestamp-query is required for a paired diagnostic profile");
  return JSON.stringify({ name: adapter.name, vendor: adapter.vendor, architecture: adapter.architecture, device: adapter.device,
    description: adapter.description, features: [...adapter.features].sort(), limits: adapter.limits });
}

export function validateProfileBrowser(runner, browser) {
  check(["firefox", "chromium"].includes(browser), `Unsupported profiling browser selector: ${browser}`);
  check(runner?.browserRequested === browser, `Recorded requested browser differs from selector ${browser}`);
  const products = browser === "chromium" ? ["chrome", "chromium"] : ["firefox"];
  check(products.includes(runner?.browser), `Recorded browser product ${runner?.browser} differs from selector ${browser}`);
  return browser;
}

function decisions(output, label) {
  check(output && typeof output.model === "string" && output.answers && Object.keys(output.answers).length > 0, `${label}: missing decision output`);
  const hasRaw = Object.hasOwn(output, "raw_probabilities");
  if (hasRaw) {
    const map = output.raw_probabilities;
    const questions = Object.keys(output.answers);
    check(map && typeof map === "object" && !Array.isArray(map) && Object.keys(map).length === questions.length && questions.every((name) => Object.hasOwn(map, name)), `${label}: invalid or incomplete raw probability map`);
  }
  return Object.fromEntries(Object.entries(output.answers).map(([name, answer]) => {
    check(answer.type === "noul" && finite(answer.noul) && answer.noul >= 0 && answer.noul <= 1, `${label}: invalid noul answer`);
    const raw = output.raw_probabilities?.[name];
    if (hasRaw) check(raw && typeof raw === "object", `${label}: invalid raw probability distribution`);
    if (hasRaw && !Array.isArray(raw)) check(Object.keys(raw).length === 2 && Object.hasOwn(raw, "false") && Object.hasOwn(raw, "true"), `${label}: invalid raw probability option keys`);
    const probabilities = hasRaw ? Array.isArray(raw) ? Array.from(raw) : [raw.false, raw.true] : [1 - answer.noul, answer.noul];
    check(probabilities.length === 2 && probabilities.every((value) => finite(value) && value >= 0 && value <= 1) && Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) <= 0.0005, `${label}: invalid raw probability distribution`);
    return [name, probabilities];
  }));
}

function gpuValues(gpu, label) {
  check(gpu && typeof gpu === "object" && !Array.isArray(gpu), `${label}: invalid GPU timestamp profile`);
  const values = Object.values(gpu);
  const total = values.reduce((a, b) => a + b, 0);
  check(values.length > 0 && values.every((value) => finite(value) && value >= 0) && finite(total) && total > 0, `${label}: invalid GPU timestamp profile`);
  return values;
}

export function validateProfileRun(result, expected) {
  const { id, arch, precision, browser, spec, samples, warmups, profile, shapes, runtime, packBytes, responseModel, cacheMode = "uncached" } = expected;
  check(typeof responseModel === "string" && responseModel.length > 0, `${id}: missing verified response model contract`);
  check(result?.done === true && !result.error && result.backend === "webgpu", `${id}: benchmark did not complete on WebGPU`);
  check(result.model === id && result.arch === arch && result.precision === precision && result.modelInfo?.name === (spec.name || id), `${id}: loaded model identity differs`);
  check(result.modelInfo.revision === spec.revision && result.modelInfo.source === `https://huggingface.co/${spec.repo}`, `${id}: loaded model source or revision differs`);
  check(result.runtime === runtime && result.runs === samples && result.warmups === warmups && result.profile === profile && result.unique === (cacheMode === "uncached"), `${id}: sampling protocol differs`);
  check(result.context === "page" && result.cache?.packEnabled === false && result.pack?.cached === false && result.pack.bytes === packBytes, `${id}: pack cache or execution context differs`);
  check(cacheMode === "uncached" ? result.cache.state?.enabled === false : result.cache.state?.enabled === true, `${id}: state-cache mode differs`);
  if (browser !== undefined) validateProfileBrowser(result.runner, browser);
  const hardware = adapterKey(result.adapter);
  check(Array.isArray(result.cases) && result.cases.length === shapes.length, `${id}: incomplete request cases`);
  for (const [index, item] of result.cases.entries()) {
    check(item.name === SHAPES[shapes[index]], `${id}: request case order differs`);
    check(item.samples?.length === samples && item.wallMs?.length === samples, `${id}: incomplete raw samples`);
    for (const [sampleIndex, sample] of item.samples.entries()) {
      check(finite(sample.wallMs) && sample.wallMs > 0 && sample.wallMs === item.wallMs[sampleIndex] && Number.isSafeInteger(sample.tokens) && sample.tokens > 0 && /^[a-f0-9]{64}$/.test(sample.requestHash), `${id}: invalid timing, token count, or request fingerprint`);
      check(Array.isArray(sample.outputs) && sample.outputs.length === (shapes[index] === 4 ? 32 : 1), `${id}: missing request outputs`);
      sample.outputs.forEach((output, outputIndex) => {
        check(output.model === responseModel, `${id}: response model differs from the verified family response contract (${responseModel})`);
        const expectedQuestions = shapes[index] === 3 ? ["q0", "q1", "q2", "q3", "q4"] : ["q"];
        check(JSON.stringify(Object.keys(output.answers || {}).sort()) === JSON.stringify(expectedQuestions), `${id}: response question keys differ`);
        check(!["kev", "gemma4"].includes(arch) || Object.hasOwn(output, "raw_probabilities"), `${id}: missing raw probability distribution for ${arch}`);
        decisions(output, `${id}.${index}.${sampleIndex}.${outputIndex}`);
      });
      if (profile) gpuValues(sample.gpu, id);
      else check(sample.gpu == null, `${id}: wall latency includes GPU profiling`);
    }
    check(cacheMode !== "uncached" || new Set(item.samples.map((sample) => sample.requestHash)).size === samples, `${id}: uncached case requires distinct request fingerprints`);
  }
  return hardware;
}

export function summarizeProfileRuns(runs, { shapes, probabilityTolerance = 0.0005, answerUnit = 0.0001 } = {}) {
  check(runs.length >= 4 && runs.length % 2 === 0, "At least two complete wall/profile blocks are required");
  const blocks = new Set(runs.map((run) => run.block));
  check([...blocks].every((block) => runs.filter((run) => run.block === block).length === 2 && runs.some((run) => run.block === block && run.profile) && runs.some((run) => run.block === block && !run.profile)), "Incomplete wall/profile block");
  const reference = runs[0].result;
  const hardware = adapterKey(reference.adapter);
  let maxProbabilityDelta = 0;
  let maxPublicNoulDelta = 0;
  const publicNoulTolerance = 2 * probabilityTolerance + answerUnit;
  for (const run of runs) {
    check(run.result.done === true && !run.result.error, "Cannot attribute an incomplete GPU profile");
    check(adapterKey(run.result.adapter) === hardware && run.result.runner?.browserVersion === reference.runner?.browserVersion, "Adapter, device features, or browser changed during profile");
    for (const [index, item] of run.result.cases.entries()) for (const [sampleIndex, sample] of item.samples.entries()) {
      if (run.profile) gpuValues(sample.gpu, "Profile");
      const prior = reference.cases[index].samples[sampleIndex];
      check(sample.tokens === prior.tokens && sample.requestHash === prior.requestHash, "Request or token count changed between blocks or timing modes");
      sample.outputs.forEach((output, outputIndex) => {
        const before = decisions(prior.outputs[outputIndex], "Reference");
        const after = decisions(output, "Candidate");
        check(JSON.stringify(Object.keys(before)) === JSON.stringify(Object.keys(after)), "Decision question keys changed");
        for (const name of Object.keys(before)) after[name].forEach((value, option) => {
          const difference = Math.abs(value - before[name][option]);
          maxProbabilityDelta = Math.max(maxProbabilityDelta, difference);
          check(difference <= probabilityTolerance, `Profile changed an answer probability by ${difference}, exceeding ${probabilityTolerance}`);
        });
        for (const name of Object.keys(before)) {
          const delta = Math.abs(output.answers[name].noul - prior.outputs[outputIndex].answers[name].noul);
          maxPublicNoulDelta = Math.max(maxPublicNoulDelta, delta);
          check(delta <= publicNoulTolerance, `Profile changed a public noul answer by ${delta}, exceeding ${publicNoulTolerance}`);
        }
      });
    }
  }
  const cases = reference.cases.map((item, index) => {
    const wallBlocks = runs.filter((run) => !run.profile).map((run) => run.result.cases[index].samples.map((sample) => sample.wallMs));
    const gpuBlocks = runs.filter((run) => run.profile).map((run) => run.result.cases[index].samples.map((sample) => Object.values(sample.gpu).reduce((a, b) => a + b, 0)));
    const profiled = runs.filter((run) => run.profile).flatMap((run) => run.result.cases[index].samples);
    const labels = [...new Set(profiled.flatMap((sample) => Object.keys(sample.gpu)))].sort();
    const heterogeneous = profiled.some((sample) => labels.some((label) => !Object.hasOwn(sample.gpu, label)));
    const kernels = labels.map((name) => ({ name,
      medianMs: median(profiled.map((sample) => Object.hasOwn(sample.gpu, name) ? sample.gpu[name] : 0)),
      ...(heterogeneous ? { dispatchedSamples: profiled.filter((sample) => Object.hasOwn(sample.gpu, name)).length, totalSamples: profiled.length } : {}) }));
    const total = kernels.reduce((sum, kernel) => sum + kernel.medianMs, 0);
    kernels.forEach((kernel) => { kernel.share = total ? kernel.medianMs / total : 0; });
    kernels.sort((a, b) => b.medianMs - a.medianMs);
    const tokens = item.samples.map((sample) => sample.tokens);
    return { name: item.name, shape: shapes?.[index] ?? index, tokens: item.tokens, tokenCounts: tokens,
      tokenRange: [Math.min(...tokens), Math.max(...tokens)],
      wall: metrics(wallBlocks.flat(), wallBlocks),
      gpu: { ...metrics(gpuBlocks.flat(), gpuBlocks), kernels,
        ...(heterogeneous ? { kernelAttribution: "median per-request GPU timestamp contribution; labels absent from completed samples were not dispatched and contribute zero" } : {}) } };
  });
  return { cases, correctness: { maxProbabilityDelta, probabilityTolerance, maxPublicNoulDelta, publicNoulTolerance,
    scope: "same-pack repeated requests and profiled/unprofiled raw probability and rounded noul agreement; not an accuracy benchmark" } };
}
