#!/usr/bin/env node
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { createReadStream, createWriteStream } from "node:fs";
import { link, mkdir, mkdtemp, readdir, readFile, rm, stat, statfs, unlink, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";
import { MODELS } from "../js/src/source.js";

const root = fileURLToPath(new URL("../", import.meta.url));
const args = process.argv.slice(2);
if (args.some((arg) => !/^--(base|cdp|baseline|current|output|models|download|local-packs)=.+$/.test(arg))) throw new Error("options require --name=value: base, cdp, baseline, current, output, models, download, local-packs");
const option = (name, fallback) => args.find((arg) => arg.startsWith(`--${name}=`))?.slice(name.length + 3) || fallback;
const base = new URL(option("base", "http://127.0.0.1:8123/"));
const cdp = option("cdp", "http://127.0.0.1:9333");
const baseline = new URL(option("baseline", "tmp/m4-opt/baseline/js/src/index.js"), base);
const current = new URL(option("current", "js/src/index.js"), base);
const output = resolve(root, option("output", "tmp/gpu-comparison"));
const models = option("models", Object.keys(MODELS).join(",")).split(",");
const downloader = option("download", "fetch");
const localPacks = option("local-packs", null);
if (!["fetch", "hf"].includes(downloader)) throw new Error("download must be fetch or hf");
if (models.some((model) => !MODELS[model]) || new Set(models).size !== models.length) throw new Error("unknown or duplicate model");
if (baseline.origin !== base.origin || current.origin !== base.origin) throw new Error("runtimes must be same-origin");
const endpoint = await (await fetch(`${cdp}/json/version`)).json();
const socket = new WebSocket(endpoint.webSocketDebuggerUrl);
await new Promise((accept, reject) => { socket.onopen = accept; socket.onerror = reject; });
const pending = new Map();
let sequence = 0;
socket.onmessage = ({ data }) => {
  const message = JSON.parse(data);
  const request = pending.get(message.id);
  if (!request) return;
  pending.delete(message.id);
  clearTimeout(request.timer);
  if (message.error) request.reject(new Error(`${request.method}: ${message.error.message}`));
  else request.accept(message.result);
};
socket.onclose = () => {
  for (const request of pending.values()) { clearTimeout(request.timer); request.reject(new Error("CDP disconnected")); }
  pending.clear();
};
function send(method, params = {}, sessionId) {
  if (socket.readyState !== WebSocket.OPEN) return Promise.reject(new Error("CDP disconnected"));
  return new Promise((accept, reject) => {
    const id = ++sequence;
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`${method} timed out`)); }, 30000);
    pending.set(id, { method, accept, reject, timer });
    socket.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
  });
}
const sleep = (milliseconds) => new Promise((accept) => setTimeout(accept, milliseconds));
const median = (values) => {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};
async function fingerprint(runtime) {
  const directory = resolve(root, `.${runtime.pathname}`, "..");
  const paths = (await readdir(directory, { recursive: true })).filter((path) => /\.(js|wasm)$/.test(path)).sort();
  const digest = createHash("sha256");
  for (const path of paths) {
    const local = createHash("sha256").update(await readFile(resolve(directory, path))).digest("hex");
    const response = await fetch(new URL(path, runtime));
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    const served = createHash("sha256").update(new Uint8Array(await response.arrayBuffer())).digest("hex");
    if (served !== local) throw new Error(`${path}: served bytes do not match local runtime`);
    digest.update(`${path}:${served}\n`);
  }
  return digest.digest("hex");
}
async function pageResult(url, name) {
  const target = (await send("Target.createTarget", { url: "about:blank", hidden: true, background: true })).targetId;
  try {
    const session = (await send("Target.attachToTarget", { targetId: target, flatten: true })).sessionId;
    await sleep(1000);
    await send("Page.navigate", { url: url.href }, session);
    const deadline = Date.now() + 1800000;
    while (Date.now() < deadline) {
      let evaluated;
      try { evaluated = await send("Runtime.evaluate", { expression: `window.${name} || null`, returnByValue: true }, session); }
      catch (error) {
        if (!/context|navigat|frame/i.test(error.message)) throw error;
        await sleep(250);
        continue;
      }
      const result = evaluated.result?.value;
      if (result?.error || result?.status === "error") throw new Error(`${url.href}: ${result.error || JSON.stringify(result)}`);
      if (result?.done || result?.status === "done") return result;
      await sleep(250);
    }
    throw new Error(`${url.href}: page timed out`);
  } finally {
    await send("Target.closeTarget", { targetId: target }).catch((error) => { console.error(`Target cleanup: ${error.message}`); });
  }
}
function validateBench(result, model, profile) {
  if (result.model !== model || result.backend !== "webgpu" || !result.cases?.length) throw new Error(`${model}: invalid benchmark identity`);
  if (result.runs !== 9 || result.warmups !== 3 || result.unique !== true) throw new Error(`${model}: benchmark sampling protocol differs`);
  if (result.modelInfo?.revision !== MODELS[model].revision || result.arch !== MODELS[model].arch || result.cases.length !== 3) throw new Error(`${model}: loaded model identity or input count differs`);
  for (const item of result.cases) {
    if (item.samples?.length !== 9 || item.wallMs.length !== item.samples.length || item.wallMs.some((value) => !Number.isFinite(value) || value <= 0) || item.samples.some((sample) => !Number.isSafeInteger(sample.tokens) || sample.tokens <= 0)) throw new Error(`${model}: invalid wall samples or token count`);
    if (profile) for (const sample of item.samples) {
      const values = Object.values(sample.gpu || {});
      if (!values.length || values.some((value) => !Number.isFinite(value) || value < 0) || values.reduce((sum, value) => sum + value, 0) <= 0) throw new Error(`${model}: invalid GPU timestamps`);
    }
    else if (item.samples.some((sample) => sample.gpu)) throw new Error(`${model}: unprofiled latency contains GPU instrumentation`);
  }
}

function validateDecisions(result, model) {
  if (result.model !== model || result.backend !== "webgpu" || result.errors?.length || result.quality?.valid !== 108 || result.quality.total !== 108 || result.timingAllRows?.length !== 108 || result.rawProbs?.length !== 108 || result.stateCache?.enabled !== false || result.metadata?.profileDuringMeasurements !== false) throw new Error(`${model}: incomplete regression replay`);
  if (result.modelRevision !== MODELS[model].revision || result.modelArch !== MODELS[model].arch) throw new Error(`${model}: loaded model identity differs from catalog`);
  for (const [index, row] of result.timingAllRows.entries()) {
    const probabilities = result.rawProbs[index];
    const values = Object.values(probabilities.probabilities || {});
    if (row.invalid || row.error || probabilities.caseId !== row.caseId || probabilities.permutation !== row.permutation || !Number.isFinite(row.wallMs) || row.wallMs <= 0 || values.length !== row.options.length || row.options.some((key) => !Object.hasOwn(probabilities.probabilities, key)) || values.some((value) => !Number.isFinite(value) || value < 0 || value > 1) || Math.abs(values.reduce((sum, value) => sum + value, 0) - 1) > 0.0005) throw new Error(`${model}: invalid regression row ${index}`);
  }
}

const comparisons = [];
await mkdir(resolve(root, "tmp"), { recursive: true });
const packFolder = await mkdtemp(resolve(root, "tmp/kevala-gpu-packs-"));
async function downloadPack(model) {
  const spec = MODELS[model];
  const disk = await statfs(packFolder);
  if (disk.bavail * disk.bsize < (localPacks ? 0 : spec.pack) + 4 * 1024 ** 3) throw new Error(`${model}: insufficient disk space for verified local pack`);
  const path = resolve(packFolder, `${model}-q8.kevala`);
  let bytes, sha256;
  if (localPacks) {
    await link(resolve(root, localPacks, `${model}-q8.kevala`), path);
    bytes = (await stat(path)).size;
    sha256 = await packHash(path);
  } else if (downloader === "hf") {
    const hosted = new URL(spec.hosted);
    const match = hosted.pathname.match(/^\/([^/]+\/[^/]+)\/resolve\/([a-f0-9]{40})\/([^/]+)$/);
    if (hosted.hostname !== "huggingface.co" || !match || match[3] !== `${model}-q8.kevala`) throw new Error(`${model}: hf requires a pinned Hugging Face pack URL`);
    await new Promise((accept, reject) => {
      const child = spawn("hf", ["download", match[1], match[3], "--revision", match[2], "--local-dir", packFolder, "--quiet"], { stdio: ["ignore", "ignore", "inherit"], env: { ...process.env, HF_XET_CHUNK_CACHE_SIZE_BYTES: "0" } });
      child.on("error", reject);
      child.on("exit", (code, signal) => code === 0 ? accept() : reject(new Error(`${model}: hf download failed (${signal || code})`)));
    });
    bytes = (await stat(path)).size;
    sha256 = await packHash(path);
  } else {
    const response = await fetch(spec.hosted);
    if (!response.ok || !response.body) throw new Error(`${model}: pack HTTP ${response.status}`);
    const digest = createHash("sha256");
    bytes = 0;
    const hashStream = new Transform({ transform(chunk, encoding, next) { bytes += chunk.length; digest.update(chunk); next(null, chunk); } });
    await pipeline(Readable.fromWeb(response.body), hashStream, createWriteStream(path, { flags: "wx" }));
    sha256 = digest.digest("hex");
  }
  if (bytes !== spec.pack || sha256 !== spec.packSha256) throw new Error(`${model}: pack size or SHA-256 differs from catalog`);
  const url = new URL(`tmp/${packFolder.split("/").at(-1)}/${model}-q8.kevala`, base);
  const served = await fetch(url, { method: "HEAD" });
  if (!served.ok || Number(served.headers.get("content-length")) !== bytes) throw new Error(`${model}: local server does not expose verified pack`);
  console.error(`${model}: ${localPacks ? "linked local pack" : "downloaded"} and verified ${bytes} bytes (${sha256})`);
  return { path, url: url.href, bytes, sha256 };
}

async function packHash(path) {
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  return digest.digest("hex");
}
try {
  await mkdir(output, { recursive: true });
  const fingerprints = { baseline: await fingerprint(baseline), current: await fingerprint(current) };
  for (const model of models) {
    const pack = await downloadPack(model);
    try {
      const runs = [];
      const folder = resolve(output, model);
      await mkdir(folder, { recursive: true });
      for (const [round, label] of ["baseline", "current", "current", "baseline"].entries()) for (const profile of [false, true]) {
        const url = new URL("bench.html", base);
        url.hash = new URLSearchParams({ auto: "", backend: "webgpu", model, packUrl: pack.url, cache: "0", runtime: label === "baseline" ? baseline.href : current.href, profile: profile ? "1" : "0", stateCache: "0", runs: "9", warmups: "3", unique: "1", shapes: "0,1,2" }).toString();
        const result = await pageResult(url, "bench");
        validateBench(result, model, profile);
        const record = { round, label, profile, result };
        runs.push(record);
        await writeFile(resolve(folder, `${round}-${label}-${profile ? "profile" : "wall"}.json`), JSON.stringify(record, null, 2) + "\n");
        console.error(`${model} ${round} ${label} ${profile ? "profile" : "wall"}: ${result.cases.map((item) => item.p50.toFixed(1)).join(" / ")} ms`);
      }
      for (const run of runs) for (const [index, item] of run.result.cases.entries()) {
        const reference = runs[0].result.cases[index];
        if (item.name !== reference.name || item.tokens !== reference.tokens || JSON.stringify(item.samples.map((sample) => sample.tokens)) !== JSON.stringify(reference.samples.map((sample) => sample.tokens))) throw new Error(`${model}: input changed between builds or measurement modes`);
      }
      const cases = [0, 1, 2].map((index) => {
        const metric = (label, profile) => {
          const rows = runs.filter((run) => run.label === label && run.profile === profile).map((run) => run.result.cases[index]);
          const values = rows.flatMap((row) => profile ? row.samples.map((sample) => Object.values(sample.gpu).reduce((sum, value) => sum + value, 0)) : row.wallMs);
          if (rows.some((row) => row.name !== rows[0].name || row.tokens !== rows[0].tokens)) throw new Error(`${model}: input changed between blocks`);
          return { medianMs: median(values), meanMs: values.reduce((sum, value) => sum + value, 0) / values.length, samples: values, blockMediansMs: rows.map((row) => profile ? median(row.samples.map((sample) => Object.values(sample.gpu).reduce((sum, value) => sum + value, 0))) : row.p50) };
        };
        const tokenCounts = runs[0].result.cases[index].samples.map((sample) => sample.tokens);
        return { name: runs[0].result.cases[index].name, tokens: runs[0].result.cases[index].tokens, tokenCounts, tokenRange: [Math.min(...tokenCounts), Math.max(...tokenCounts)], wall: { baseline: metric("baseline", false), current: metric("current", false) }, gpu: { baseline: metric("baseline", true), current: metric("current", true) } };
      });
      const decisions = {};
      for (const [label, runtime] of [["baseline", baseline], ["current", current]]) {
        const url = new URL("dev/decision-bench.html", base);
        url.search = new URLSearchParams({ model, backend: "webgpu", packUrl: pack.url, runtime: runtime.href, dataset: "kevala", permutations: "3", warmups: "5", cache: "0" }).toString();
        const result = await pageResult(url, "decisionBench");
        validateDecisions(result, model);
        decisions[label] = result;
        await writeFile(resolve(folder, `decisions-${label}.json`), JSON.stringify(result, null, 2) + "\n");
      }
      if (JSON.stringify(decisions.baseline.metadata.datasetFileHashes) !== JSON.stringify(decisions.current.metadata.datasetFileHashes)) throw new Error(`${model}: regression dataset changed`);
      let maxProbabilityDifference = 0;
      const changedChoices = [];
      for (const [index, before] of decisions.baseline.timingAllRows.entries()) {
        const after = decisions.current.timingAllRows[index];
        if (before.caseId !== after.caseId || before.permutation !== after.permutation || before.expected !== after.expected || JSON.stringify(before.options) !== JSON.stringify(after.options)) throw new Error(`${model}: unpaired regression rows`);
        for (const key of before.options) maxProbabilityDifference = Math.max(maxProbabilityDifference, Math.abs(before.rawProbabilities[key] - after.rawProbabilities[key]));
        if (before.prediction !== after.prediction) changedChoices.push({ caseId: before.caseId, expected: before.expected, baseline: before.prediction, current: after.prediction, baselineProbabilities: before.rawProbabilities, currentProbabilities: after.rawProbabilities });
      }
      const regression = { rows: 108, baselineCorrect: decisions.baseline.quality.correct, currentCorrect: decisions.current.quality.correct, maxProbabilityDifference, changedChoices };
      console.error(`${model}: ${108 - changedChoices.length}/108 unchanged choices; max probability difference ${maxProbabilityDifference}`);
      if (await fingerprint(baseline) !== fingerprints.baseline || await fingerprint(current) !== fingerprints.current) throw new Error("runtime changed during comparison");
      if (await packHash(pack.path) !== pack.sha256) throw new Error(`${model}: pack changed during comparison`);
      const comparison = { model, pack: { url: pack.url, bytes: pack.bytes, sha256: pack.sha256, source: localPacks ? "verified local pack; private hard link; source preserved" : downloader, verification: "pre- and post-comparison file SHA-256; local server Content-Length; browser persistent download cache disabled" }, fingerprints, cases, regression, runs: runs.map(({ round, label, profile, result }) => ({ round, label, profile, modelInfo: result.modelInfo, gpuTuning: result.gpuTuning, loadSeconds: result.load_s })) };
      comparisons.push(comparison);
      await writeFile(resolve(folder, "comparison.json"), JSON.stringify(comparison, null, 2) + "\n");
    } finally {
      await unlink(pack.path);
    }
  }
  await writeFile(resolve(output, "summary.json"), JSON.stringify({ date: new Date().toISOString(), browser: endpoint.Browser, targetVisibility: "hidden", browserContext: "default; fresh owned hidden target per measurement; closed after every wall/profile block and decision replay; persistent pack cache disabled", baseline: baseline.href, current: current.href, fingerprints, protocol: "ABBA blocks; three warmups and nine unique requests per input size; state cache disabled; wall latency measured separately from instrumented GPU profiles; loading excluded", models: comparisons }, null, 2) + "\n");
} finally {
  await rm(packFolder, { recursive: true, force: true });
  socket.close();
}
