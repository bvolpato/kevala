import { createHash } from "node:crypto";
import { spawn, execFileSync } from "node:child_process";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { cpus, release, totalmem, type } from "node:os";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { fetchRange, MODELS } from "../js/src/source.js";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const base = args.shift() || "http://127.0.0.1:18093/";
const output = resolve(args.shift() || "tmp/decisions");
let passes = 3;
let models = Object.keys(MODELS);
let packSource = "local";
const cdp = process.env.KEVALA_BENCH_CDP || "http://127.0.0.1:9333";
for (const arg of args) {
  if (arg.startsWith("--passes=")) passes = Number(arg.slice(9));
  else if (arg.startsWith("--models=")) models = arg.slice(9).split(",");
  else if (arg.startsWith("--pack=")) packSource = arg.slice(7);
  else throw new Error(`unknown option: ${arg}`);
}
if (!Number.isInteger(passes) || passes < 1 || passes > 10) throw new Error("passes must be an integer from 1 to 10");
if (!models.length || new Set(models).size !== models.length || models.some((model) => !MODELS[model])) throw new Error("models must be distinct catalog IDs");
if (!["local", "hosted"].includes(packSource)) throw new Error("pack must be local or hosted");
const baseUrl = new URL(base.endsWith("/") ? base : `${base}/`);
const cdpUrl = new URL(cdp);
if (![baseUrl, cdpUrl].every((url) => ["127.0.0.1", "localhost", "[::1]"].includes(url.hostname))) throw new Error("run on the benchmark host with local server and CDP URLs so host metadata describes the browser");
await mkdir(dirname(output), { recursive: true });
await mkdir(output);

function runCommand(command, argumentsList) {
  return new Promise((accept, reject) => {
    const child = spawn(command, argumentsList, { cwd: root, stdio: "inherit" });
    child.once("error", reject);
    child.once("exit", (code, signal) => code === 0 ? accept() : reject(new Error(`${command}: exited ${code ?? signal}`)));
  });
}

async function browserInfo() {
  const response = await fetch(`${cdp.replace(/\/$/, "")}/json/version`);
  if (!response.ok) throw new Error(`CDP version: HTTP ${response.status}`);
  const version = await response.json();
  const socket = new WebSocket(version.webSocketDebuggerUrl);
  try {
    await new Promise((accept, reject) => { socket.onopen = accept; socket.onerror = reject; });
    const info = await new Promise((accept, reject) => {
      const timer = setTimeout(() => reject(new Error("SystemInfo.getInfo timed out")), 30000);
      socket.onmessage = ({ data }) => {
        const reply = JSON.parse(data);
        if (reply.id !== 1) return;
        clearTimeout(timer);
        if (reply.error) reject(new Error(JSON.stringify(reply.error)));
        else accept(reply.result);
      };
      socket.send(JSON.stringify({ id: 1, method: "SystemInfo.getInfo" }));
    });
    return { version: version.Browser, gpu: info.gpu };
  } finally { socket.close(); }
}

async function hashUrl(path, size = null) {
  const url = new URL(path, baseUrl);
  let chunks;
  if (size !== null) {
    chunks = fetchRange(url.href, 0, size, path);
  } else {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    chunks = response.body;
  }
  const hash = createHash("sha256");
  let bytes = 0;
  for await (const chunk of chunks) { hash.update(chunk); bytes += chunk.length; }
  return { sha256: hash.digest("hex"), bytes };
}

const sourceFiles = (await readdir(resolve(root, "js/src"), { recursive: true }))
  .filter((path) => /\.(?:js|wasm)$/.test(path)).map((path) => `js/src/${path}`);
sourceFiles.push("dev/decision-bench.js", "benchmarks/decisions/metrics.js", "benchmarks/decisions/manifest.json");
async function sourceFingerprint() {
  const hash = createHash("sha256");
  for (const path of sourceFiles.sort()) {
    const digest = await hashUrl(path);
    const local = createHash("sha256").update(await readFile(resolve(root, path))).digest("hex");
    if (digest.sha256 !== local) throw new Error(`${path}: served build differs from this checkout`);
    hash.update(`${path}:${digest.sha256}\n`);
  }
  return hash.digest("hex");
}

const browser = await browserInfo();
const hardware = { gpu: browser.gpu.devices?.[0]?.deviceString || browser.gpu.auxAttributes?.glRenderer || "WebGPU", cpu: cpus()[0]?.model || "unknown", os: `${type()} ${release()}`, vramGiB: null, driver: browser.gpu.devices?.[0]?.driverVersion || null };
hardware.cpuCores = cpus().length;
hardware.memoryGiB = totalmem() / 2 ** 30;
if (type() === "Darwin") {
  hardware.os = `macOS ${execFileSync("sw_vers", ["-productVersion"], { encoding: "utf8" }).trim()}`;
  const displays = JSON.parse(execFileSync("system_profiler", ["SPDisplaysDataType", "-json"], { encoding: "utf8", timeout: 30000 })).SPDisplaysDataType;
  const gpu = displays?.find((display) => hardware.gpu.includes(display.sppci_model));
  if (gpu?.sppci_cores) hardware.gpuCores = Number(gpu.sppci_cores);
  if (/^Apple M\d+/.test(gpu?.sppci_model ?? "")) hardware.unifiedMemoryGiB = hardware.memoryGiB;
}
const renderer = browser.gpu.auxAttributes?.glRenderer || hardware.gpu;
if (/swiftshader|llvmpipe|software/i.test(renderer)) throw new Error(`software GPU renderer is not a publishable hardware benchmark: ${renderer}`);
if (type() === "Linux") {
  try {
    const devices = execFileSync("nvidia-smi", ["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"], { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim().split("\n");
    const device = devices.map((line) => line.split(",").map((value) => value.trim())).find(([name]) => renderer.includes(name));
    if (device) Object.assign(hardware, { gpu: device[0], vramGiB: Math.round(Number(device[1]) / 1024), driver: device[2] });
  } catch (error) {
    hardware.nvidiaSmiUnavailable = String(error.message);
  }
}
const engine = { revision: execFileSync("git", ["rev-parse", "HEAD"], { cwd: root, encoding: "utf8" }).trim(), dirty: Boolean(execFileSync("git", ["status", "--porcelain"], { cwd: root, encoding: "utf8" }).trim()), servedSourceSha256: await sourceFingerprint() };
const packs = new Map();
for (const model of models) {
  console.error(`Verifying full ${packSource} pack: ${model}`);
  const digest = await hashUrl(packSource === "hosted" ? MODELS[model].hosted : `tmp/${model}-q8.kevala`, packSource === "hosted" ? MODELS[model].pack : null);
  if (digest.sha256 !== MODELS[model].packSha256 || digest.bytes !== MODELS[model].pack) throw new Error(`${model}: served full-pack hash or size does not match the pinned catalog (${digest.sha256}, ${digest.bytes} bytes)`);
  packs.set(model, digest);
}

const directories = [];
for (let pass = 1; pass <= passes; pass++) {
  const directory = resolve(output, `run-${pass}`);
  await mkdir(directory, { recursive: true });
  const resultsPath = relative(root, output).replaceAll("\\", "/");
  const campaign = { schema: "kevala-decision-campaign-v1", date: new Date().toISOString().slice(0, 10), hardware, browser: { name: "chrome", version: browser.version, visibility: "hidden" }, engineRevision: engine.revision, passes: 1, resultsPath, reportPath: `${resultsPath}/report.md`, notes: [engine.dirty ? "**Build caveat:** this build contains uncommitted changes; the served JS/WASM fingerprint is recorded with every result." : "Build and served JS/WASM hashes are recorded with every result."] };
  await writeFile(resolve(directory, "campaign.json"), JSON.stringify(campaign, null, 2) + "\n");
  for (const model of models) {
    const url = new URL("dev/decision-bench.html", baseUrl);
    url.search = new URLSearchParams({ model, backend: "webgpu", pack: packSource, dataset: "all", permutations: "3", warmups: "5" }).toString();
    const file = resolve(directory, `decision-${model}.json`);
    await runCommand("uv", ["run", "scripts/bench-gpu.py", "--cdp", cdp, "--result", "decision", "--timeout", "1800", "--url", url.href, "--output", file]);
    if (await sourceFingerprint() !== engine.servedSourceSha256) throw new Error("served build changed during evaluation");
    const result = JSON.parse(await readFile(file, "utf8"));
    result.environment = { hardware, browserGpu: browser.gpu };
    result.metadata.engine = engine;
    result.postRunVerification = { verifiedUtc: new Date().toISOString(), pack: { ...packs.get(model), method: `SHA-256 of the complete ${packSource} HTTP-served pack before the campaign; matched the pinned catalog digest` } };
    await writeFile(file, JSON.stringify(result, null, 2) + "\n");
    const scores = Object.entries(result.quality.perDataset).map(([dataset, metric]) => `${dataset}: ${(100 * metric.accuracy).toFixed(1)}%`).join(", ");
    console.error(`${model}, pass ${pass}: ${scores}; mean ${result.latency.meanMs.toFixed(1)} ms`);
  }
  campaign.dateEnd = new Date().toISOString().slice(0, 10);
  await writeFile(resolve(directory, "campaign.json"), JSON.stringify(campaign, null, 2) + "\n");
  directories.push(directory);
}
if (models.length === Object.keys(MODELS).length) {
  const reporterArgs = ["scripts/report-decisions.mjs", "--results", directories[0], "--no-references", "--output", resolve(output, "summary.json"), "--markdown", resolve(output, "report.md")];
  for (const directory of directories.slice(1)) reporterArgs.push("--repeat-results", directory);
  for (const model of models.filter((model) => model.startsWith("bruv1-"))) reporterArgs.push("--include-model", model);
  await runCommand(process.execPath, reporterArgs);
} else {
  console.error("Partial model smoke only; not a complete publishable comparison table.");
}
