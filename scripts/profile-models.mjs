#!/usr/bin/env node
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { mkdir, readFile, readdir, realpath, stat, unlink, writeFile } from "node:fs/promises";
import { cpus, freemem, loadavg, platform, release, totalmem } from "node:os";
import { dirname, resolve, relative, sep } from "node:path";
import { MODELS } from "../js/src/source.js";
import { BASE_URL, ROOT, failed, runPage } from "./lib/browser-run.mjs";
import { classifyProfileFailure, fileHash, inspectPack, profilePlan, SHAPES, splitResidentProfileBlocks, summarizeProfileRuns, validateProfileRun } from "./lib/model-profile.mjs";
import { withProfileLock } from "./lib/profile-lock.mjs";

const HELP = `Profile local catalog packs serially, with separate wall and diagnostic GPU blocks.

  pnpm exec node scripts/profile-models.mjs --base=http://127.0.0.1:18171/ --output=tmp/model-profile

Options use --name=value:
  --models=ID,...          Default: every current catalog model
  --precisions=q8|all|...  Default: q8; all includes catalog BF16/F32 variants
  --browser=firefox        Or --browsers=firefox,chromium
  --runtime=URL           Same-origin runtime entry point; default js/src/index.js
  --packs=DIRECTORY       Local packs named <model>-<precision>.kevala; default tmp
  --pack-base=URL         Same-origin served pack directory; default tmp/
  --protocol=resident|isolated  Default: resident; isolated preserves per-lane loads
  --submit=await|split|none  Override the model's runtime-default submission policy
  --samples=9 --warmups=3 --blocks=2 --shapes=0,1,2 --timeout=600 (per lane)
  --binary=PATH           One selected browser's executable
  --chromium-arg=--FLAG    Repeat for isolated Chromium flags
  --max-pack-bytes=N      Explicit preflight memory-policy bound, if needed
  --probability-tolerance=0.0005
  --lock=PATH             Default: canonical ~/.cache/kevala/gpu-profile.lock
  --list                  Print the coverage plan without hashing or starting browsers
  --help                  Print this help

State and pack caches are disabled. Every measured request has a deterministic unique
state. All blocks use the on-page GPU engine; wall timings exclude loading, tuning and
fixture hashing. Instrumented GPU sums are diagnostic and are not application latency.
Every pack is checked against its header and hashed before and after measurement.
Exit status: 0 complete, 1 invalid/failed measurement, 2 incomplete coverage.
`;

const HARNESS_ASSETS = ["bench.html", "dev/adapter-metadata.js"];

function options(args) {
  const values = {};
  const chromiumArgs = [];
  const allowed = new Set(["base", "output", "models", "precisions", "browser", "browsers", "runtime", "packs", "pack-base", "protocol", "submit", "samples", "warmups", "blocks", "shapes", "timeout", "binary", "chromium-arg", "max-pack-bytes", "probability-tolerance", "lock"]);
  for (const arg of args) {
    if (arg === "--help" || arg === "--list") { values[arg.slice(2)] = true; continue; }
    const match = /^--([^=]+)=(.+)$/.exec(arg);
    if (!match || !allowed.has(match[1])) throw new Error(`Unknown option ${arg}; use --help`);
    if (match[1] === "chromium-arg") { if (!match[2].startsWith("--")) throw new Error("chromium-arg must be a --flag"); chromiumArgs.push(match[2]); }
    else { if (Object.hasOwn(values, match[1])) throw new Error(`Duplicate option --${match[1]}`); values[match[1]] = match[2]; }
  }
  const integer = (name, fallback, min, max) => {
    const value = Number(values[name] ?? fallback);
    if (!Number.isSafeInteger(value) || value < min || value > max) throw new Error(`${name} must be an integer from ${min} through ${max}`);
    return value;
  };
  if (values.browser && values.browsers) throw new Error("Use browser or browsers, not both");
  const browsers = (values.browsers || values.browser || process.env.KEVALA_BENCH_BROWSER || "firefox").split(",");
  if (values.binary && browsers.length !== 1) throw new Error("binary requires one selected browser");
  const shapes = (values.shapes || "0,1,2").split(",").map(Number);
  if (!shapes.length || new Set(shapes).size !== shapes.length || shapes.some((value, index) => !Number.isInteger(value) || value < 0 || value >= SHAPES.length || index > 0 && value <= shapes[index - 1])) throw new Error("shapes must be unique ascending indices from 0 through 4");
  const base = new URL(values.base || BASE_URL);
  if (!base.pathname.endsWith("/")) base.pathname += "/";
  const runtime = new URL(values.runtime || "js/src/index.js", base);
  const packBase = new URL(values["pack-base"] || "tmp/", base);
  if (!packBase.pathname.endsWith("/")) packBase.pathname += "/";
  if (runtime.origin !== base.origin || packBase.origin !== base.origin) throw new Error("runtime and pack-base must use the base origin");
  const probabilityTolerance = Number(values["probability-tolerance"] ?? 0.0005);
  if (!Number.isFinite(probabilityTolerance) || probabilityTolerance < 0 || probabilityTolerance > 0.005) throw new Error("probability-tolerance must be between 0 and 0.005");
  const protocol = values.protocol || "resident";
  if (!["resident", "isolated"].includes(protocol)) throw new Error("protocol must be resident or isolated");
  if (values.submit !== undefined && !["await", "split", "none"].includes(values.submit)) throw new Error('submit must be "await", "split", or "none"');
  return { ...values, base, runtime, packBase, browsers, chromiumArgs, shapes, probabilityTolerance,
    protocol,
    models: values.models?.split(",") || Object.keys(MODELS),
    precisions: values.precisions === "all" ? "all" : (values.precisions || "q8").split(","),
    output: resolve(ROOT, values.output || "tmp/model-profile"), packs: resolve(ROOT, values.packs || "tmp"),
    samples: integer("samples", 9, 3, 50), warmups: integer("warmups", 3, 1, 20), blocks: integer("blocks", 2, 2, 10),
    timeout: integer("timeout", 600, 30, 3600), maxPackBytes: integer("max-pack-bytes", Number.MAX_SAFE_INTEGER, 1, Number.MAX_SAFE_INTEGER) };
}

async function telemetry() {
  let memory = null;
  try {
    const raw = await readFile("/proc/meminfo", "utf8");
    const value = (name) => Number(new RegExp(`^${name}:\\s+(\\d+)`, "m").exec(raw)?.[1]) * 1024;
    memory = { availableBytes: value("MemAvailable"), swapFreeBytes: value("SwapFree") };
  } catch { /* Non-Linux hosts still report the portable memory fields. */ }
  const gpu = spawnSync("nvidia-smi", ["--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], { encoding: "utf8", timeout: 10000 });
  return { capturedAt: new Date().toISOString(), loadAverage: loadavg(), freeMemoryBytes: freemem(), memory,
    nvidia: gpu.status === 0 ? { available: true, columns: ["index", "name", "driverVersion", "memoryTotalMiB", "memoryUsedMiB", "utilizationPercent", "temperatureC", "powerW"], rows: gpu.stdout.trim().split("\n").map((line) => line.split(",").map((value) => value.trim())) } : { available: false } };
}

async function fingerprint(runtime, base) {
  if (!runtime.pathname.startsWith(base.pathname)) throw new Error("Runtime must be served beneath the configured base path");
  const directory = dirname(resolve(ROOT, decodeURIComponent(runtime.pathname.slice(base.pathname.length))));
  if (relative(ROOT, directory).startsWith(`..${sep}`)) throw new Error("Runtime path leaves this checkout");
  const files = (await readdir(directory, { recursive: true })).filter((file) => /\.(js|wasm)$/.test(file)).sort();
  if (!files.length || !files.some((file) => file.endsWith(".wasm"))) throw new Error("Runtime fingerprint requires JavaScript and built WASM files");
  const digest = createHash("sha256");
  const assets = [];
  for (const file of files) {
    const bytes = await readFile(resolve(directory, file));
    const localHash = createHash("sha256").update(bytes).digest("hex");
    const response = await fetch(new URL(file.split(sep).join("/"), runtime));
    if (!response.ok) throw new Error(`Runtime ${file}: HTTP ${response.status}`);
    const servedHash = createHash("sha256").update(new Uint8Array(await response.arrayBuffer())).digest("hex");
    if (servedHash !== localHash) throw new Error(`Runtime ${file}: served bytes differ from the local source`);
    digest.update(`${file}:${servedHash}\n`);
    assets.push({ path: file, bytes: bytes.length, sha256: servedHash });
  }
  return { sha256: digest.digest("hex"), assets };
}

async function fingerprintHarness(base) {
  const assets = [];
  for (const path of HARNESS_ASSETS) {
    const local = await readFile(resolve(ROOT, path));
    const response = await fetch(new URL(path, base));
    if (!response.ok) throw new Error(`Harness ${path}: HTTP ${response.status}`);
    const served = Buffer.from(await response.arrayBuffer());
    if (!served.equals(local)) throw new Error(`Harness ${path}: served bytes differ from the local source`);
    assets.push({ path, bytes: local.length, sha256: createHash("sha256").update(local).digest("hex") });
  }
  return { assets, sha256: assets[0].sha256 };
}

const git = (...args) => {
  const result = spawnSync("git", args, { cwd: ROOT, encoding: "utf8" });
  if (result.status !== 0) throw new Error(`git ${args[0]} failed`);
  return result.stdout.trim();
};

async function run(config, plan) {
  await mkdir(config.output, { recursive: true });
  try { await stat(resolve(config.output, "summary.json")); throw new Error("Output already contains a profile; choose a fresh output directory to avoid mixing captures"); }
  catch (error) { if (error.code !== "ENOENT") throw error; }
  const initialFingerprint = await fingerprint(config.runtime, config.base);
  const initialHarness = await fingerprintHarness(config.base);
  const harnessSha256 = initialHarness.sha256;
  const summary = { schemaVersion: 1, status: "running", generatedAt: new Date().toISOString(),
    runtime: { entry: config.runtime.href, commit: git("rev-parse", "HEAD"), tree: git("rev-parse", "HEAD^{tree}"), workingTreeDirty: git("status", "--porcelain=v1").length > 0, ...initialFingerprint, harnessSha256, harnessAssets: initialHarness.assets },
    protocol: { samples: config.samples, warmups: config.warmups, blocks: config.blocks, shapes: config.shapes,
      models: config.models, precisions: config.precisions, browsers: config.browsers,
      cacheMode: "uncached", packCache: false, stateCache: false, uniqueStates: true, context: "on-page WebGPU engine",
      submitMode: config.submit ?? "runtime-default", submitExplicit: config.submit !== undefined,
      profileProtocol: config.protocol,
      modelLoadScope: config.protocol === "resident" ? "one model load per row; exact load-time GPU tuning snapshot is recorded; per-shape caches may evolve during measurement" : "isolated model load for each wall/profile lane (legacy protocol)",
      blockOrder: "wall/profile, then profile/wall, alternating by block",
      wall: "unprofiled awaited request milliseconds; loading, tuning and fixture hashing excluded",
      gpu: "separate instrumented per-kernel GPU timestamps; profiling changes dispatch passes and may change optional paths; not application latency",
      kernelShare: "per-label median divided by the sum of label medians; not a share of wall latency",
      probabilityTolerance: config.probabilityTolerance, accuracyEvaluated: false },
    environment: { platform: platform(), release: release(), cpu: cpus()[0]?.model || null, logicalCpus: cpus().length, totalMemoryBytes: totalmem(), vkDriverFiles: process.env.VK_DRIVER_FILES || null },
    models: plan.map((row) => ({ ...row, cacheMode: "uncached" })) };
  const persist = async () => {
    summary.generatedAt = new Date().toISOString();
    const measuredIds = [...new Set(summary.models.filter((row) => row.status === "measured").map((row) => row.id))];
    summary.coverage = { catalogModels: Object.keys(MODELS).length, requestedModels: config.models.length,
      catalogModelIds: Object.keys(MODELS), requestedModelIds: config.models, measuredModelIds: measuredIds,
      allCatalogModelsMeasured: Object.keys(MODELS).every((id) => measuredIds.includes(id)),
      requestedRows: summary.models.length, measured: summary.models.filter((row) => row.status === "measured").length,
      blocked: summary.models.filter((row) => row.status === "blocked").length,
      unsupported: summary.models.filter((row) => row.status === "unsupported").length,
      failed: summary.models.filter((row) => row.status === "failed").length,
      pending: summary.models.filter((row) => row.status === "pending").length };
    await writeFile(resolve(config.output, "summary.json"), JSON.stringify(summary, null, 2) + "\n");
  };
  await persist();
  for (const row of summary.models) {
    if (row.status !== "pending") continue;
    const folder = resolve(config.output, row.browser, `${row.id}-${row.precision}`);
    await mkdir(folder, { recursive: true });
    const packPath = resolve(config.packs, `${row.id}-${row.precision}.kevala`);
    const url = new URL(`${row.id}-${row.precision}.kevala`, config.packBase);
    row.startedAt = new Date().toISOString();
    row.before = await telemetry();
    let pack;
    let stage = "pack-preflight";
    try {
      try { pack = await inspectPack(packPath, row.id, row.precision, MODELS[row.id]); }
      catch (error) {
        if (error.code === "ENOENT") { row.status = "blocked"; row.reasonCode = "missing-pack"; row.reason = `Local ${row.precision} pack is unavailable; no weights were downloaded`; continue; }
        throw error;
      }
      row.pack = { ...pack, file: undefined };
      if (pack.bytes > config.maxPackBytes) { row.status = "blocked"; row.reasonCode = "pack-size-policy"; row.reason = `Pack needs ${pack.bytes} bytes, above explicit policy ${config.maxPackBytes}`; continue; }
      if (pack.bytes > 8 * 1024 ** 3 && row.before.memory && (row.before.memory.availableBytes < 32 * 1024 ** 3 || row.before.memory.swapFreeBytes < 4 * 1024 ** 3)) {
        row.status = "blocked"; row.reasonCode = "host-memory-headroom"; row.reason = "Large-pack run requires at least 32 GiB available host memory and 4 GiB free swap"; continue;
      }
      const head = await fetch(url, { method: "HEAD" });
      if (!head.ok || Number(head.headers.get("content-length")) !== pack.bytes) throw new Error(`${row.id}: served pack Content-Length differs from the verified file`);
      if (!url.pathname.startsWith(config.base.pathname)) throw new Error(`${row.id}: pack URL leaves the configured base path`);
      const servedPath = resolve(ROOT, decodeURIComponent(url.pathname.slice(config.base.pathname.length)));
      if (await realpath(servedPath) !== await realpath(packPath)) throw new Error(`${row.id}: pack URL does not map to the verified local file`);
      const headerResponse = await fetch(url, { headers: { Range: `bytes=0-${pack.headerBytes - 1}` } });
      if (headerResponse.status !== 206 || Number(headerResponse.headers.get("content-length")) !== pack.headerBytes) throw new Error(`${row.id}: server must support bounded byte ranges`);
      if (createHash("sha256").update(new Uint8Array(await headerResponse.arrayBuffer())).digest("hex") !== pack.headerSha256) throw new Error(`${row.id}: served pack header differs from the verified file`);
      const runs = [];
      if (config.protocol === "resident") {
        const page = new URL("bench.html", config.base);
        page.hash = new URLSearchParams({ auto: "", backend: "webgpu", model: row.id, precision: row.precision, packUrl: url.href,
          runtime: config.runtime.href, profile: "0", protocol: "resident", blocks: String(config.blocks), cache: "0", stateCache: "0", unique: "1", onPage: "1",
          ...(config.submit === undefined ? {} : { submit: config.submit }),
          runs: String(config.samples), warmups: String(config.warmups), shapes: config.shapes.join(",") }).toString();
        const pagePath = resolve(folder, "resident-page.json");
        stage = "browser";
        const child = runPage({ result: "latency", url: page.href, output: pagePath, browser: row.browser,
          binary: config.binary, chromiumArgs: config.chromiumArgs.length ? config.chromiumArgs : undefined, timeoutSeconds: config.timeout * 2 * config.blocks, stdio: "pipe" });
        let pageResult;
        try {
          pageResult = JSON.parse(await readFile(pagePath, "utf8"));
          const raw = relative(config.output, pagePath).split(sep).join("/");
          if (!(row.raw ||= []).includes(raw)) row.raw.push(raw);
        } catch { /* Child startup failures may precede a page result. */ }
        if (failed(child)) throw Object.assign(new Error(pageResult?.error || child.error?.message || child.stderr?.trim() || `Browser exited ${child.status ?? child.signal}`), { code: child.error?.code, fatal: child.error?.code === "ETIMEDOUT" });
        stage = "validation";
        for (const record of splitResidentProfileBlocks(pageResult, config.blocks)) {
          const filename = `${record.block}-${record.profile ? "gpu" : "wall"}.json`;
          const path = resolve(folder, filename);
          await writeFile(path, JSON.stringify(record.result, null, 2) + "\n");
          const raw = relative(config.output, path).split(sep).join("/");
          if (!(row.raw ||= []).includes(raw)) row.raw.push(raw);
          validateProfileRun(record.result, { ...row, spec: MODELS[row.id], samples: config.samples, warmups: config.warmups, profile: record.profile,
            shapes: config.shapes, runtime: config.runtime.href, packBytes: pack.bytes, responseModel: pack.responseModel });
          runs.push(record);
          console.error(`${row.id} ${row.precision} ${row.browser} block ${record.block + 1} ${record.profile ? "GPU diagnostic" : "wall"}: ${record.result.cases.map((item) => item.p50.toFixed(2)).join(" / ")} ms wall`);
        }
        await unlink(pagePath);
        const pageRaw = relative(config.output, pagePath).split(sep).join("/");
        row.raw = row.raw.filter((raw) => raw !== pageRaw);
      } else {
        for (let block = 0; block < config.blocks; block++) for (const profile of block % 2 ? [true, false] : [false, true]) {
          const page = new URL("bench.html", config.base);
          page.hash = new URLSearchParams({ auto: "", backend: "webgpu", model: row.id, precision: row.precision, packUrl: url.href,
            runtime: config.runtime.href, profile: profile ? "1" : "0", protocol: "isolated", cache: "0", stateCache: "0", unique: "1", onPage: "1",
            ...(config.submit === undefined ? {} : { submit: config.submit }),
            runs: String(config.samples), warmups: String(config.warmups), shapes: config.shapes.join(",") }).toString();
          const filename = `${block}-${profile ? "gpu" : "wall"}.json`;
          const path = resolve(folder, filename);
          stage = "browser";
          const child = runPage({ result: profile ? "bench" : "latency", url: page.href, output: path, browser: row.browser,
            binary: config.binary, chromiumArgs: config.chromiumArgs.length ? config.chromiumArgs : undefined, timeoutSeconds: config.timeout, stdio: "pipe" });
          let result;
          try {
            result = JSON.parse(await readFile(path, "utf8"));
            const raw = relative(config.output, path).split(sep).join("/");
            if (!(row.raw ||= []).includes(raw)) row.raw.push(raw);
          } catch { /* Child startup failures may precede a page result. */ }
          if (failed(child)) throw Object.assign(new Error(result?.error || child.error?.message || child.stderr?.trim() || `Browser exited ${child.status ?? child.signal}`), { code: child.error?.code, fatal: child.error?.code === "ETIMEDOUT" });
          stage = "validation";
          validateProfileRun(result, { ...row, spec: MODELS[row.id], samples: config.samples, warmups: config.warmups, profile,
            shapes: config.shapes, runtime: config.runtime.href, packBytes: pack.bytes, responseModel: pack.responseModel });
          runs.push({ block, profile, result });
          console.error(`${row.id} ${row.precision} ${row.browser} block ${block + 1} ${profile ? "GPU diagnostic" : "wall"}: ${result.cases.map((item) => item.p50.toFixed(2)).join(" / ")} ms wall`);
        }
      }
      stage = "validation";
      const measured = summarizeProfileRuns(runs, { shapes: config.shapes, probabilityTolerance: config.probabilityTolerance, answerUnit: pack.answerUnit });
      stage = "source-verification";
      if (await fileHash(packPath) !== pack.sha256) throw new Error(`${row.id}: pack changed during measurement`);
      const after = await stat(packPath);
      if (after.size !== pack.file.size || after.ino !== pack.file.ino || after.mtimeMs !== pack.file.mtimeMs) throw new Error(`${row.id}: pack file identity changed during measurement`);
      if ((await fingerprint(config.runtime, config.base)).sha256 !== initialFingerprint.sha256) throw Object.assign(new Error("Runtime changed during measurement"), { fatal: true });
      const currentHarness = await fingerprintHarness(config.base);
      if (JSON.stringify(currentHarness.assets) !== JSON.stringify(initialHarness.assets)) throw Object.assign(new Error("Benchmark harness changed during measurement"), { fatal: true });
      Object.assign(row, measured, { status: "measured", adapter: runs[0].result.adapter, runner: runs[0].result.runner,
        tuning: runs.map((record) => ({ block: record.block, profile: record.profile, gpuTuning: record.result.gpuTuning })) });
    } catch (error) {
      Object.assign(row, classifyProfileFailure(error, { stage })); console.error(`${row.id} ${row.precision} ${row.browser}: ${row.reason}`);
      if (error.fatal) { summary.aborted = row.reason; break; }
    }
    finally { row.after = await telemetry(); row.completedAt = new Date().toISOString(); await persist(); }
  }
  summary.status = summary.models.some((row) => row.status === "failed") ? "failed" : summary.models.every((row) => row.status === "measured") ? "complete" : "partial";
  await persist();
  console.log(JSON.stringify({ status: summary.status, coverage: summary.coverage, summary: resolve(config.output, "summary.json") }));
  process.exitCode = summary.status === "failed" ? 1 : summary.status === "partial" ? 2 : 0;
}

const config = options(process.argv.slice(2));
if (config.help) console.log(HELP);
else {
  const plan = profilePlan(MODELS, config);
  if (config.list) console.log(JSON.stringify({ base: config.base.href, runtime: config.runtime.href, rows: plan }, null, 2));
  else await withProfileLock(() => run(config, plan), { ...(config.lock ? { path: config.lock } : {}) });
}
