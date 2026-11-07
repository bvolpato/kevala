import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { MODELS } from "../js/src/source.js";
import { adapterKey, classifyProfileFailure, inspectPack, metrics, profilePlan, profileTuningSnapshot, responseModelName, SHAPES, splitResidentProfileBlocks, summarizeProfileRuns, validateProfileRun, verifyPackHeader } from "../scripts/lib/model-profile.mjs";

const spec = { arch: "kev", repo: "test/model", revision: "a".repeat(40), precisions: ["q8", "bf16"] };
const expected = { id: "test", arch: "kev", precision: "q8", spec, samples: 3, warmups: 1, shapes: [0], runtime: "http://localhost/js/src/index.js", packBytes: 4288, responseModel: "test" };

function result(profile) {
  const samples = Array.from({ length: 3 }, (_, index) => ({ tokens: 10 + index, wallMs: index + 1,
    requestHash: createHash("sha256").update(String(index)).digest("hex"),
    outputs: [{ model: "test", answers: { q: { type: "noul", noul: 0.7 } }, raw_probabilities: { q: { false: 0.3, true: 0.7 } } }],
    ...(profile ? { gpu: { projection: 0.8 + index * 0.1, norm: 0.1 } } : {}) }));
  return { done: true, model: "test", arch: "kev", precision: "q8", backend: "webgpu",
    modelInfo: { name: "test", source: "https://huggingface.co/test/model", revision: spec.revision },
    runtime: expected.runtime, runs: 3, warmups: 1, profile, unique: true, context: "page",
    cache: { packEnabled: false, state: { enabled: false } }, pack: { cached: false, bytes: 4288 },
    adapter: { name: "hardware", vendor: "vendor", architecture: "arch", device: "device", description: null, isFallbackAdapter: false,
      features: ["timestamp-query"], limits: { maxBufferSize: 1024 } },
    runner: { browserVersion: "152" },
    cases: [{ name: SHAPES[0], tokens: 12, samples, wallMs: samples.map((sample) => sample.wallMs) }] };
}

const runs = () => [0, 1].flatMap((block) => [false, true].map((profile) => ({ block, profile, result: result(profile) })));
const requestStageRuns = () => runs().map((run) => {
  const scale = run.profile ? 1.25 : 1;
  run.result.cases[0].samples.forEach((sample, index) => {
    sample.prepare = (0.2 + index * 0.05) * scale;
    sample.forward = (0.2 + index * 0.05) * scale;
    sample.total = sample.prepare + sample.forward + (0.1 + index * 0.05) * scale;
  });
  return run;
});

test("the observed GPU upload exhaustion is a blocked resource limit, not a failed model", () => {
  const reason = "WebGPU: high-performance: out of GPU memory while uploading model weights: Not enough memory left.; low-power: out of GPU memory while uploading model weights: Not enough memory left.";
  assert.deepEqual(classifyProfileFailure(new Error(reason)), { status: "blocked", reasonCode: "gpu-memory-limit", reason });
});

test("only explicit browser resource and controller errors become unavailable measurements", () => {
  assert.equal(classifyProfileFailure(new Error("GPUOutOfMemoryError: memory exhausted")).reasonCode, "gpu-memory-limit");
  assert.equal(classifyProfileFailure(Object.assign(new Error("Driver allocation exhausted"), { name: "GPUOutOfMemoryError" })).reasonCode, "gpu-memory-limit");
  assert.equal(classifyProfileFailure(new Error("WebGPU: Out of memory creating buffer")).reasonCode, "gpu-memory-limit");
  assert.equal(classifyProfileFailure(Object.assign(new Error("spawnSync uv ETIMEDOUT"), { code: "ETIMEDOUT" })).reasonCode, "controller-deadline");
  assert.equal(classifyProfileFailure(new Error("bench page did not expose a completed result before timeout")).reasonCode, "controller-deadline");
  assert.equal(classifyProfileFailure(Object.assign(new Error("spawn could not allocate process memory"), { code: "ENOMEM" })).reasonCode, "host-memory-limit");
  assert.equal(classifyProfileFailure(new Error("WebGPU buffer size exceeds maxBufferSize")).reasonCode, "gpu-device-limit");
  assert.equal(classifyProfileFailure(new Error("timestamp-query is unavailable for GPU profiling")).status, "unsupported");
  for (const reason of ["invalid GPU timestamp profile", "Profile changed an answer probability by 0.02", "model allocation metadata differs from expected layout", "Device lost: internal error", "Browser exited SIGKILL"]) {
    assert.equal(classifyProfileFailure(new Error(reason)).status, "failed", reason);
  }
  assert.equal(classifyProfileFailure(new Error("out of GPU memory"), { stage: "validation" }).status, "failed");
  assert.equal(classifyProfileFailure(Object.assign(new Error("resource annotation mismatch"), { code: "ENOMEM" }), { stage: "source-verification" }).status, "failed");
});

test("the profiling plan accounts for every catalog model and explicitly requested precision", () => {
  const rows = profilePlan(MODELS, { precisions: ["q8", "bf16"], browsers: ["firefox", "chromium"] });
  assert.deepEqual([...new Set(rows.map((row) => row.id))], Object.keys(MODELS));
  for (const id of Object.keys(MODELS)) {
    const own = rows.filter((row) => row.id === id);
    assert.deepEqual(own.map((row) => `${row.precision}/${row.browser}`), ["q8/firefox", "q8/chromium", "bf16/firefox", "bf16/chromium"]);
    assert.equal(own[0].status, "pending");
  }
  assert.equal(rows.find((row) => row.id === "laya" && row.precision === "bf16").status, "unsupported");
  assert.throws(() => profilePlan(MODELS, { models: ["laya", "laya"] }), /unique/);
  assert.throws(() => profilePlan(MODELS, { models: ["unknown"] }), /catalog/);
});

test("resident lanes retain one load identity and an exact canonical tuning snapshot", () => {
  const source = { z: 1, a: { y: true, x: null } };
  const tuning = profileTuningSnapshot(source);
  source.a.x = false;
  assert.equal(tuning.planJson, '{"a":{"x":null,"y":true},"z":1}');
  assert.equal(tuning.sha256, createHash("sha256").update(tuning.planJson).digest("hex"));
  assert.deepEqual(tuning.value, { a: { x: null, y: true }, z: 1 });
  assert.throws(() => profileTuningSnapshot({ hidden: undefined }), /non-JSON/);
  assert.throws(() => profileTuningSnapshot({ invalid: NaN }), /non-finite/);

  const page = { done: true, protocol: "resident", resident: { sessionId: "session-1", loadId: "load-1", tuning },
    blocks: [0, 1].map((block) => ({ block, lanes: (block ? [true, false] : [false, true]).map((profile) => ({ profile, cases: [{ profile }] })) })) };
  const lanes = splitResidentProfileBlocks(page, 2);
  assert.deepEqual(lanes.map(({ block, profile }) => [block, profile]), [[0, false], [0, true], [1, true], [1, false]]);
  assert.ok(lanes.every(({ result }) => result.resident.sessionId === "session-1" && result.resident.loadId === "load-1"));
  assert.ok(lanes.every(({ result }) => result.resident.tuning.planJson === tuning.planJson && result.resident.tuning.sha256 === tuning.sha256));
  assert.notStrictEqual(lanes[0].result.resident.tuning.value, lanes[1].result.resident.tuning.value);
  assert.deepEqual(lanes[0].result.gpuTuning, JSON.parse(tuning.planJson));
  lanes[0].result.resident.tuning.value.a.x = "mutated copy";
  assert.equal(lanes[1].result.resident.tuning.value.a.x, null);
  assert.throws(() => splitResidentProfileBlocks({ ...page, blocks: page.blocks.slice(0, 1) }, 2), /incomplete blocks/);

  const numericTuning = profileTuningSnapshot({ ratio: 0.12130666177280236 });
  const transportedTuning = { ...numericTuning, value: { ratio: 0.12130666177280237 } };
  const numericPage = { ...page, resident: { ...page.resident, tuning: transportedTuning } };
  const numericLane = splitResidentProfileBlocks(numericPage, 2)[0].result;
  assert.equal(numericLane.resident.tuning.value.ratio, JSON.parse(numericTuning.planJson).ratio);
  assert.equal(numericLane.gpuTuning.ratio, JSON.parse(numericTuning.planJson).ratio);

  const comparisons = runs().map((run) => ({ ...run, result: { ...run.result, resident: structuredClone(lanes[1].result.resident) } }));
  assert.doesNotThrow(() => summarizeProfileRuns(comparisons, { shapes: [0] }));
  comparisons[1].result.resident.loadId = "load-2";
  assert.throws(() => summarizeProfileRuns(comparisons, { shapes: [0] }), /share a session, model load/);
});

test("raw profile validation refuses changed identity, caches, timings, requests and adapters", () => {
  for (const profile of [false, true]) assert.doesNotThrow(() => validateProfileRun(result(profile), { ...expected, profile }));
  const invalid = [
    ["identity", (r) => { r.modelInfo.revision = "b".repeat(40); }],
    ["cache", (r) => { r.cache.state.enabled = true; }],
    ["timing", (r) => { r.cases[0].samples[0].wallMs = NaN; }],
    ["request", (r) => { delete r.cases[0].samples[0].requestHash; }],
    ["output", (r) => { r.cases[0].samples[0].outputs[0].raw_probabilities.q.true = Infinity; }],
    ["software", (r) => { r.adapter.isFallbackAdapter = true; }],
    ["timestamp", (r) => { r.adapter.features = []; }],
    ["GPU profile", (r) => { r.cases[0].samples[0].gpu.projection = -1; }],
  ];
  for (const [label, mutate] of invalid) {
    const r = result(true); mutate(r);
    assert.throws(() => validateProfileRun(r, { ...expected, profile: true }), undefined, label);
  }
  assert.throws(() => validateProfileRun(result(true), { ...expected, profile: false }), /protocol/);
});

test("optional browser validation accepts Selenium product aliases and rejects selector or product mismatches", () => {
  for (const [browser, product] of [["chromium", "chrome"], ["chromium", "chromium"], ["firefox", "firefox"]]) {
    const r = result(false);
    Object.assign(r.runner, { browser: product, browserRequested: browser });
    const contract = { ...expected, browser, profile: false };
    assert.doesNotThrow(() => validateProfileRun(r, contract));
    r.runner.browserRequested = browser === "chromium" ? "firefox" : "chromium";
    assert.throws(() => validateProfileRun(r, contract), /requested browser/);
    r.runner.browserRequested = browser;
    r.runner.browser = browser === "chromium" ? "firefox" : "chrome";
    assert.throws(() => validateProfileRun(r, contract), /browser product/);
    delete r.runner.browser;
    assert.throws(() => validateProfileRun(r, contract), /browser product/);
  }
  assert.doesNotThrow(() => validateProfileRun(result(false), { ...expected, profile: false }));
});

test("provided raw maps cannot omit or replace a question distribution, while omitted Laya maps remain valid", () => {
  const invalidMaps = [{}, { q: null }, { q: 0 }, { q: undefined }, null, 0, []];
  for (const map of invalidMaps) {
    const r = result(false);
    r.cases[0].samples[0].outputs[0].raw_probabilities = map;
    assert.throws(() => validateProfileRun(r, { ...expected, profile: false }), /raw probability/, JSON.stringify(map));
  }
  const r = result(false);
  Object.assign(r, { model: "laya", arch: "laya", modelInfo: { name: "laya", source: `https://huggingface.co/${MODELS.laya.repo}`, revision: MODELS.laya.revision } });
  r.cases[0].samples.forEach((sample) => {
    sample.outputs[0].model = "laya-rl-agent";
    delete sample.outputs[0].raw_probabilities;
  });
  assert.doesNotThrow(() => validateProfileRun(r, { ...expected, id: "laya", arch: "laya", spec: MODELS.laya, profile: false, responseModel: "laya-rl-agent" }));
});

test("Kev and Gemma require raw distributions, while Laya and Clef may omit them", () => {
  for (const arch of ["kev", "gemma4", "laya", "clef"]) {
    const r = result(false);
    r.arch = arch;
    r.cases[0].samples.forEach((sample) => { delete sample.outputs[0].raw_probabilities; });
    const validate = () => validateProfileRun(r, { ...expected, arch, spec: { ...spec, arch }, profile: false });
    if (arch === "kev" || arch === "gemma4") assert.throws(validate, /missing raw probability/, arch);
    else {
      assert.doesNotThrow(validate, arch);
      r.cases[0].samples[0].outputs[0].raw_probabilities = { q: null };
      assert.throws(validate, /raw probability/, `${arch} still validates any supplied raw map`);
    }
  }
});

test("uncached cases require distinct measured requests even when unique is asserted", () => {
  const r = result(false);
  r.cases[0].samples[1].requestHash = r.cases[0].samples[0].requestHash;
  assert.throws(() => validateProfileRun(r, { ...expected, profile: false }), /distinct request/);
  r.unique = false;
  r.cache.state.enabled = true;
  assert.doesNotThrow(() => validateProfileRun(r, { ...expected, profile: false, cacheMode: "cached" }));
});

test("optional request stage timings are complete and consistent across measured samples", () => {
  const partial = result(false);
  partial.cases[0].samples[0].prepare = 0.2;
  assert.throws(() => validateProfileRun(partial, { ...expected, profile: false }), /partial request stage timing fields/);

  const malformed = result(false);
  malformed.cases[0].samples.forEach((sample) => Object.assign(sample, { prepare: 0.2, forward: 0.2, total: 0.5 }));
  malformed.cases[0].samples[1].forward = Infinity;
  assert.throws(() => validateProfileRun(malformed, { ...expected, profile: false }), /invalid request stage timing values/);

  const mixed = requestStageRuns();
  for (const field of ["prepare", "forward", "total"]) delete mixed[0].result.cases[0].samples[1][field];
  assert.throws(() => summarizeProfileRuns(mixed, { shapes: [0] }), /mixed request stage timing coverage/);
});

test("family response aliases preserve Laya and pointer contracts without weakening pack identity", () => {
  assert.equal(responseModelName({ model: { name: "laya" }, config: {} }), "laya-rl-agent");
  assert.equal(responseModelName({ model: { name: "kev-4b" }, config: { arch: "kev" } }), "kev-latest");
  assert.equal(responseModelName({ model: { name: "semif-qwen3.5-4b" }, config: { arch: "kev", readout: "semif" } }), "semif-qwen3.5-4b");
  assert.equal(responseModelName({ model: { name: "h2o-lightning-4b" }, config: { arch: "kev", readout: "h2o-lightning", h2o_config: { model: "h2oai/h2o-lightning-4b" } } }), "h2oai/h2o-lightning-4b");
  const r = result(false);
  Object.assign(r, { model: "laya", arch: "laya", modelInfo: { name: "laya", source: `https://huggingface.co/${MODELS.laya.repo}`, revision: MODELS.laya.revision } });
  r.cases[0].samples.forEach((sample) => { sample.outputs[0].model = "laya-rl-agent"; });
  const contract = { ...expected, id: "laya", arch: "laya", spec: MODELS.laya, profile: false, responseModel: "laya-rl-agent" };
  assert.doesNotThrow(() => validateProfileRun(r, contract));
  r.cases[0].samples[0].outputs[0].model = "kev-latest";
  assert.throws(() => validateProfileRun(r, contract), /family response contract/);
  r.cases[0].samples[0].outputs[0].model = "laya-rl-agent";
  r.modelInfo.name = "another-pack";
  assert.throws(() => validateProfileRun(r, contract), /identity/);
});

test("separate timing modes retain raw samples, kernel shares and repeated block medians", () => {
  const summary = summarizeProfileRuns(runs(), { shapes: [0] });
  const entry = summary.cases[0];
  assert.equal(entry.wall.medianMs, 2);
  assert.equal(entry.wall.p95Ms, 3);
  assert.deepEqual(entry.wall.blockMediansMs, [2, 2]);
  assert.equal(entry.wall.samples.length, 6);
  assert.equal(entry.gpu.samples.length, 6);
  assert.equal(Object.hasOwn(entry, "requestStages"), false);
  assert.equal(entry.gpu.kernels[0].name, "projection");
  assert.equal(Object.hasOwn(entry.gpu, "kernelAttribution"), false);
  assert.deepEqual(Object.keys(entry.gpu.kernels[0]), ["name", "medianMs", "share"]);
  assert.ok(Math.abs(entry.gpu.kernels.reduce((sum, kernel) => sum + kernel.share, 0) - 1) < 1e-12);
  assert.deepEqual(entry.tokenRange, [10, 12]);
  assert.equal(summary.correctness.maxProbabilityDelta, 0);
  assert.deepEqual(metrics(Array.from({ length: 20 }, (_, i) => i + 1), [[1, 2]]).p95Ms, 19);
});

test("request stage summaries retain samplewise residuals, fractions, lanes, and block identity", () => {
  const summary = summarizeProfileRuns(requestStageRuns(), { shapes: [0] });
  const stages = summary.cases[0].requestStages;
  assert.ok(stages.attribution.includes("architecture-specific"));
  assert.deepEqual(stages.wall.blocks.map(({ block }) => block), [0, 1]);
  assert.deepEqual(stages.gpuProfiler.blocks.map(({ block }) => block), [0, 1]);

  const wallSample = stages.wall.blocks[0].samples[0];
  assert.equal(wallSample.sampleIndex, 0);
  assert.equal(wallSample.requestHash, createHash("sha256").update("0").digest("hex"));
  assert.equal(wallSample.tokens, 10);
  assert.equal(wallSample.reportedPrepareMs, 0.2);
  assert.equal(wallSample.reportedForwardMs, 0.2);
  assert.equal(wallSample.reportedTotalMs, 0.5);
  assert.ok(Math.abs(wallSample.postForwardResidualMs - 0.1) < 1e-12);
  assert.equal(wallSample.outerResidualMs, 0.5);
  assert.equal(wallSample.prepareOfRuntimeTotal, 0.4);
  assert.equal(wallSample.forwardOfRuntimeTotal, 0.4);
  assert.ok(Math.abs(wallSample.postForwardResidualOfRuntimeTotal - 0.2) < 1e-12);
  assert.ok(Math.abs(wallSample.postForwardResidualOfWall - 0.1) < 1e-12);
  assert.equal(wallSample.runtimeTotalOfWall, 0.5);
  assert.equal(wallSample.outerResidualOfWall, 0.5);
  assert.equal(stages.gpuProfiler.blocks[0].samples[0].reportedPrepareMs, 0.25);

  const residual = stages.wall.distributions.postForwardResidualMs;
  assert.equal(residual.samples.length, 6);
  assert.ok(Math.abs(residual.medianMs - 0.15) < 1e-12);
  assert.ok(Math.abs(residual.p95Ms - 0.2) < 1e-12);
  assert.deepEqual(residual.blockMediansMs.map((value) => Math.round(value * 100)), [15, 15]);
  const fraction = stages.wall.distributions.prepareOfRuntimeTotal;
  assert.equal(fraction.samples.length, 6);
  assert.equal(fraction.samples[0], 0.4);
  assert.ok(Math.abs(fraction.median - 0.3846153846153846) < 1e-12);
  assert.deepEqual(fraction.blockMedians, [fraction.median, fraction.median]);

  const wallResidualFraction = stages.wall.distributions.postForwardResidualOfWall;
  assert.equal(wallResidualFraction.samples.length, 6);
  assert.ok(Math.abs(wallResidualFraction.samples[0] - 0.1) < 1e-12);
  assert.ok(Math.abs(wallResidualFraction.median - 0.075) < 1e-12);
  assert.ok(Math.abs(wallResidualFraction.p95 - 0.1) < 1e-12);
  assert.deepEqual(wallResidualFraction.blockMedians.map((value) => Math.round(value * 1000)), [75, 75]);
});

test("request stage residuals preserve small signed timer-rounding differences", () => {
  const data = requestStageRuns();
  const sample = data[0].result.cases[0].samples[0];
  Object.assign(sample, { prepare: 0.1, forward: 0.2, total: 0.3 });
  assert.doesNotThrow(() => validateProfileRun(data[0].result, { ...expected, profile: false }));
  const residual = summarizeProfileRuns(data, { shapes: [0] }).cases[0].requestStages.wall.blocks[0].samples[0].postForwardResidualMs;
  assert.ok(residual < 0);
  assert.equal(residual, sample.total - sample.prepare - sample.forward);
});

test("cross-lane raw probability drift fails above tolerance even when each lane is stable", () => {
  const data = runs();
  for (const run of data.filter((entry) => entry.profile)) for (const sample of run.result.cases[0].samples) {
    sample.outputs[0].raw_probabilities.q.false -= 0.0006;
    sample.outputs[0].raw_probabilities.q.true += 0.0006;
  }
  const withinLaneValues = (profile) => data.filter((run) => run.profile === profile).flatMap((run) => run.result.cases[0].samples.map((sample) => sample.outputs[0].raw_probabilities.q.true));
  assert.equal(new Set(withinLaneValues(false)).size, 1);
  assert.equal(new Set(withinLaneValues(true)).size, 1);
  assert.throws(() => summarizeProfileRuns(data, { shapes: [0], probabilityTolerance: 0.0005 }), /exceeding 0.0005/);
});

test("different valid dispatch paths retain zero contributions and dispatch counts", () => {
  const data = runs();
  data[3].result.cases[0].samples.forEach((sample, index) => {
    delete sample.gpu.projection;
    sample.gpu.fused = 1.6 + index * 0.2;
  });
  for (const run of data) validateProfileRun(run.result, { ...expected, profile: run.profile });
  const summary = summarizeProfileRuns(data, { shapes: [0] });
  const gpu = summary.cases[0].gpu;
  assert.equal(gpu.kernelAttribution, "median per-request GPU timestamp contribution; labels absent from completed samples were not dispatched and contribute zero");
  const byName = Object.fromEntries(gpu.kernels.map((kernel) => [kernel.name, kernel]));
  assert.equal(byName.projection.medianMs, 0.4);
  assert.equal(byName.fused.medianMs, 0.8);
  assert.equal(byName.norm.medianMs, 0.1);
  assert.deepEqual(gpu.kernels.map(({ name, dispatchedSamples, totalSamples }) => ({ name, dispatchedSamples, totalSamples })), [
    { name: "fused", dispatchedSamples: 3, totalSamples: 6 },
    { name: "projection", dispatchedSamples: 3, totalSamples: 6 },
    { name: "norm", dispatchedSamples: 6, totalSamples: 6 },
  ]);
  assert.ok(Math.abs(byName.projection.share - 0.4 / 1.3) < 1e-12);
  assert.equal(summary.correctness.maxProbabilityDelta, 0);
});

test("heterogeneous attribution never converts invalid or incomplete GPU timings into zero", () => {
  for (const gpu of [{}, { projection: NaN }, { projection: Infinity }, { projection: -1 }, { projection: 0 }, { a: Number.MAX_VALUE, b: Number.MAX_VALUE }, null, [1]]) {
    const data = runs();
    data[3].result.cases[0].samples[0].gpu = gpu;
    assert.throws(() => validateProfileRun(data[3].result, { ...expected, profile: true }), /GPU timestamp/);
    assert.throws(() => summarizeProfileRuns(data, { shapes: [0] }), /GPU timestamp/);
  }
  const incomplete = runs();
  incomplete[3].result.done = false;
  assert.throws(() => summarizeProfileRuns(incomplete, { shapes: [0] }), /incomplete GPU profile/);
});

test("a profile comparison fails closed when any matched input, output or hardware changes", () => {
  const cases = [
    (r) => { r[2].result.cases[0].samples[0].tokens++; },
    (r) => { r[2].result.cases[0].samples[0].requestHash = "f".repeat(64); },
    (r) => { r[2].result.cases[0].samples[0].outputs[0].raw_probabilities.q = { false: 0.299, true: 0.701 }; },
    (r) => { r[2].result.cases[0].samples[0].outputs[0].answers.q.noul = 0.9; },
    (r) => { r[2].result.runner.browserVersion = "153"; },
    (r) => { r[2].result.adapter.device = "other"; },
    (r) => { r[3].result.cases[0].samples[0].gpu.norm = Infinity; },
  ];
  for (const mutate of cases) { const data = runs(); mutate(data); assert.throws(() => summarizeProfileRuns(data, { shapes: [0] })); }
  assert.throws(() => summarizeProfileRuns(runs().slice(1), { shapes: [0] }), /complete/);
  assert.throws(() => adapterKey({ name: "SwiftShader", isFallbackAdapter: false, features: ["timestamp-query"] }), /Hardware/);
});

function pack() {
  const header = { model: { name: "test", source: "https://huggingface.co/test/model", revision: spec.revision },
    config: { arch: "kev" }, tokenizer: { offset: 4096, size: 8 },
    tensors: [{ name: "L.0.w", dtype: "q8", shape: [1, 32], offset: 4160, size: 32, block: 32, scales_offset: 4224, scales_size: 4 }] };
  const bytes = Buffer.alloc(4288);
  bytes.write("KVLA"); bytes.writeUInt32LE(1, 4);
  const json = Buffer.from(JSON.stringify(header)); bytes.writeUInt32LE(json.length, 8); json.copy(bytes, 16);
  return { header, headerBytes: 16 + json.length, bytes };
}

test("local conversion preflight validates identity and size without claiming a catalog-pinned digest", async () => {
  const directory = await mkdtemp(join(tmpdir(), "kevala-profile-pack-"));
  try {
    const fixture = pack(); const path = join(directory, "test-q8.kevala");
    await writeFile(path, fixture.bytes);
    const local = await inspectPack(path, "test", "q8", spec);
    assert.equal(local.bytes, fixture.bytes.length);
    assert.equal(local.sha256, createHash("sha256").update(fixture.bytes).digest("hex"));
    assert.equal(local.catalogHashVerified, false);
    const pinned = await inspectPack(path, "test", "q8", { ...spec, packSha256: local.sha256 });
    assert.equal(pinned.catalogHashVerified, true);
    await assert.rejects(inspectPack(path, "test", "q8", { ...spec, packSha256: "0".repeat(64) }), /SHA-256/);
    assert.throws(() => verifyPackHeader(fixture.header, fixture.headerBytes, fixture.bytes.length, "test", "bf16", spec), /precision/);
    assert.throws(() => verifyPackHeader({ ...fixture.header, model: { ...fixture.header.model, revision: "wrong" } }, fixture.headerBytes, fixture.bytes.length, "test", "q8", spec), /revision/);
    assert.throws(() => verifyPackHeader(fixture.header, fixture.headerBytes, fixture.bytes.length - 1, "test", "q8", spec), /length/);
  } finally { await rm(directory, { recursive: true }); }
});
