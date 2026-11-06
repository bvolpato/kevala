#!/usr/bin/env node
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdir, readFile, readdir, rename, stat, writeFile } from "node:fs/promises";
import { dirname, extname, isAbsolute, relative as relativePath, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { BASE_URL, ROOT, runPage } from "./lib/browser-run.mjs";
import { withProfileLock } from "./lib/profile-lock.mjs";
import { KERNEL_HARNESSES, harnessKernels, harnessUrl, kernelCoverage, kernelHarnessSkipReason, parseKernelRegistry, validateKernelResult } from "./lib/kernel-profile-catalog.mjs";

const HELP = `Usage: node scripts/profile-kernels.mjs [options]

Run registered-kernel compilation, isolated timestamp fixtures, and focused
numerical guards serially. This does not prove a universally fastest kernel.

  --base URL            Checkout server (default KEVALA_BENCH_URL or ${BASE_URL})
  --output DIRECTORY    Raw runs and summary.json (default tmp/kernel-profile)
  --browser NAME        firefox or chromium (default KEVALA_BENCH_BROWSER/firefox)
  --binary FILE         Browser binary (or KEVALA_BENCH_BINARY)
  --chromium-arg=FLAG   Extra Chromium flag; repeat as needed
  --baseline-wasm URL   Same-origin control WASM; requires exact isolated parity
  --only IDS            Comma-separated harness IDs; compile is always included
  --timeout SECONDS     Per-page limit, 1..3600 (default 300)
  --budget SECONDS      Whole campaign limit, 1..86400 (default 3600)
  --list                Print harnesses, mapped kernels, and unmapped kernels
  --help                Print this message

The canonical cross-worktree GPU profile lock is held for the whole campaign.
Missing capabilities, failed fixtures, and absent dispatched coverage remain
explicit. Compile wall time and guard elapsed time never count as GPU timing.
Exit codes: 0 = complete coverage; 2 = partial coverage; 1 = failed run.
`;

export function profileOptions(args) {
  const { values } = parseArgs({ args, strict: true, allowPositionals: false, options: {
    base: { type: "string" }, output: { type: "string" }, browser: { type: "string" }, binary: { type: "string" },
    "chromium-arg": { type: "string", multiple: true }, "baseline-wasm": { type: "string" }, only: { type: "string" },
    timeout: { type: "string" }, budget: { type: "string" }, list: { type: "boolean" }, help: { type: "boolean" },
  } });
  const integer = (name, fallback, maximum) => {
    const value = values[name] === undefined ? fallback : Number(values[name]);
    if (!Number.isSafeInteger(value) || value < 1 || value > maximum) throw new Error(`${name} must be an integer from 1 through ${maximum}`);
    return value;
  };
  const browser = values.browser || process.env.KEVALA_BENCH_BROWSER || "firefox";
  if (!["firefox", "chromium"].includes(browser)) throw new Error("browser must be firefox or chromium");
  const base = new URL(values.base || BASE_URL);
  if (!["http:", "https:"].includes(base.protocol) || base.username || base.password || base.search || base.hash) throw new Error("base must be an HTTP(S) directory URL without credentials, query, or fragment");
  if (!base.pathname.endsWith("/")) base.pathname += "/";
  const only = values.only?.split(",");
  if (only && (!only.length || new Set(only).size !== only.length || only.some((id) => !KERNEL_HARNESSES.some((item) => item.id === id)))) throw new Error("only contains an unknown or duplicate harness ID");
  const chromiumArgs = values["chromium-arg"];
  if (chromiumArgs?.some((argument) => !argument.startsWith("--") || argument.length < 3)) throw new Error("each chromium-arg must be a --flag");
  return { help: values.help, list: values.list, base: base.href, output: resolve(ROOT, values.output || "tmp/kernel-profile"), browser,
    binary: values.binary, chromiumArgs, baselineWasm: values["baseline-wasm"], only,
    timeout: integer("timeout", 300, 3600), budget: integer("budget", 3600, 86400) };
}

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

async function runtimeIdentity(base, verifyServed = false) {
  const folder = resolve(ROOT, "js/src");
  const paths = (await readdir(folder, { recursive: true })).filter((path) => [".js", ".wasm"].includes(extname(path))).sort();
  if (!paths.some((path) => path.endsWith(".wasm"))) throw new Error("build the WASM flavors before profiling");
  const files = [];
  for (const path of paths) {
    const hash = sha256(await readFile(resolve(folder, path)));
    if (verifyServed) {
      const url = new URL(`js/src/${path.split("\\").join("/")}`, base);
      const response = await fetch(url, { signal: AbortSignal.timeout(30000), cache: "no-store" });
      if (!response.ok || sha256(new Uint8Array(await response.arrayBuffer())) !== hash) throw new Error(`${path}: served runtime differs from this checkout`);
    }
    files.push({ path: `js/src/${path}`, sha256: hash });
  }
  return { fingerprint: sha256(files.map((file) => `${file.path}:${file.sha256}\n`).join("")), files };
}

async function baselineIdentity(baselineWasm, base) {
  if (!baselineWasm) return null;
  const url = new URL(baselineWasm, base);
  if (url.origin !== new URL(base).origin) throw new Error("baseline WASM must be same-origin");
  const response = await fetch(url, { signal: AbortSignal.timeout(30000), cache: "no-store" });
  if (!response.ok) throw new Error(`baseline WASM: HTTP ${response.status}`);
  return { url: url.href, sha256: sha256(new Uint8Array(await response.arrayBuffer())) };
}

export async function harnessIdentity(selected, root = ROOT) {
  root = resolve(root);
  const names = new Set(["scripts/profile-kernels.mjs", "scripts/lib/kernel-profile-catalog.mjs", "dev/wgsl.js", "dev/kernel-profile-fixtures.js"]);
  const optional = new Set();
  for (const harness of selected) {
    const path = harness.path.slice(1).split("?")[0];
    names.add(path);
    const companion = path.replace(/\.html$/, ".js");
    if (companion !== path) { names.add(companion); optional.add(companion); }
  }
  const pending = [...names], files = [], visited = new Set();
  while (pending.length) {
    const path = pending.pop();
    if (visited.has(path)) continue;
    const absolute = resolve(root, path);
    const exists = await stat(absolute).catch((error) => { if (error.code === "ENOENT") return null; throw error; });
    if (!exists?.isFile()) {
      if (optional.has(path)) continue;
      throw new Error(`harness dependency is missing or is not a file: ${path}`);
    }
    visited.add(path);
    const bytes = await readFile(absolute);
    files.push({ path, sha256: sha256(bytes) });
    for (const match of bytes.toString("utf8").matchAll(/\b(?:from\s+|import\s*\(?\s*)["'](\.[^"']+)["']/g)) {
      const imported = resolve(dirname(absolute), match[1].split(/[?#]/)[0]);
      const relative = relativePath(root, imported).split(sep).join("/");
      if (relative === ".." || relative.startsWith("../") || isAbsolute(relative) || ![".js", ".mjs"].includes(extname(imported))) continue;
      optional.delete(relative);
      if (!visited.has(relative)) pending.push(relative);
    }
  }
  files.sort((a, b) => a.path.localeCompare(b.path));
  return { fingerprint: sha256(files.map((file) => `${file.path}:${file.sha256}\n`).join("")), files };
}

async function save(path, data) {
  await writeFile(`${path}.tmp`, `${JSON.stringify(data, null, 2)}\n`);
  await rename(`${path}.tmp`, path);
}

export async function runKernelProfile(options) {
  const registryPath = resolve(ROOT, "crates/kevala/src/gpu.rs");
  const registrySource = await readFile(registryPath, "utf8");
  const registry = parseKernelRegistry(registrySource);
  const selected = KERNEL_HARNESSES.filter((harness) => harness.id === "compile" || !options.only || options.only.includes(harness.id));
  const startedAt = new Date().toISOString();
  const started = Date.now();
  const runs = [];
  const folder = resolve(options.output, "runs");
  await mkdir(folder, { recursive: true });
  // Existing results must never silently become evidence for a new campaign.
  const existing = await readdir(folder);
  if (existing.length) throw new Error("output/runs is not empty; use a new output directory");
  const identity = await runtimeIdentity(options.base, true);
  const baseline = await baselineIdentity(options.baselineWasm, options.base);
  const source = {
    commit: execFileSync("git", ["rev-parse", "HEAD"], { cwd: ROOT, encoding: "utf8" }).trim(),
    workingTreeDirty: Boolean(execFileSync("git", ["status", "--porcelain=v1"], { cwd: ROOT, encoding: "utf8" }).trim()),
    registrySha256: sha256(registrySource), ...identity, harnesses: await harnessIdentity(selected), baseline,
  };
  let capabilities = null;
  let sourceChanged = false;
  let sourceChangeReason = null;
  const writeSummary = async (executionComplete = false) => {
    const coverage = kernelCoverage(registry, runs);
    const failed = sourceChanged || runs.some((run) => run.status === "failed");
    const summary = { schemaVersion: 1, status: failed ? "failed" : executionComplete && coverage.coverageComplete ? "complete" : "partial",
      startedAt, ...(executionComplete ? { completedAt: new Date().toISOString() } : {}), executionComplete,
      sourceChanged, sourceChangeReason, source, browser: options.browser, base: options.base, capabilities,
      protocol: { isolation: "one GPU writer, serial isolated browser processes", timeoutSeconds: options.timeout, budgetSeconds: options.budget,
        isolatedSeed: 20261106, isolatedTokens: [1, 47, 128, 512], isolatedWarmups: 3, isolatedSamples: 7,
        timing: "isolated GPU timestamp batches, or explicitly labeled paired/composite focused fixtures; loading, compilation wall time and guard elapsed time excluded",
        numerical: "independent CPU references and focused registered-kernel guards where available; finite-only smoke is not numerical equivalence",
        optimality: "This inventory measures tested fixtures and candidates. It does not prove a universally fastest kernel." },
      ...coverage, runs: runs.map(({ result, ...run }) => ({ ...run, adapter: result?.adapter || null, runner: result?.runner || null })) };
    await save(resolve(options.output, "summary.json"), summary);
    return summary;
  };
  await writeSummary();
  for (const harness of selected) {
    const remaining = options.budget - (Date.now() - started) / 1000;
    const skipReason = kernelHarnessSkipReason(harness, capabilities);
    const run = { id: harness.id, source: `runs/${harness.id}.json`, url: harnessUrl(harness, options.base, registry, options.baselineWasm), startedAt: new Date().toISOString(), status: "not-run" };
    if (remaining < 1) {
      run.reason = "Whole campaign time budget exhausted before this harness.";
    } else if (skipReason) {
      run.status = "skipped";
      run.reason = skipReason;
    } else {
      const path = resolve(options.output, run.source);
      const child = runPage({ result: harness.kind, url: run.url, output: path, browser: options.browser, binary: options.binary, chromiumArgs: options.chromiumArgs,
        timeoutSeconds: Math.max(1, Math.min(options.timeout, Math.floor(remaining))), stdio: "pipe" });
      try {
        run.result = JSON.parse(await readFile(path, "utf8"));
        if (child.error) throw child.error;
        const outcome = validateKernelResult(harness, run.result, registry);
        if (outcome?.status === "skipped") { run.status = "skipped"; run.reason = outcome.reason; }
        else if (child.status !== 0) throw new Error(`browser runner exited ${child.status ?? child.signal}`);
        else run.status = "passed";
        if (harness.id === "compile" && run.status === "passed") {
          capabilities = run.result.capabilities;
          if (!capabilities || ["timestampQuery", "shaderF16", "subgroup4", "subgroup8", "subgroup16", "subgroup32", "visionQkv"].some((key) => typeof capabilities[key] !== "boolean")) throw new Error("compilation probe did not report capabilities");
          capabilities = { ...capabilities, sourceProjection: ["matmul_gemma_source", "matmul_wide_gemma_source", "matmul_k64_gemma_source"].every((name) => run.result.correctness.compiled.some((item) => item.kernel === name && item.spec.f16 === false && item.spec.rows === 4 && item.spec.groups === 1)) };
        }
      } catch (error) {
        run.status = "failed";
        run.reason = `${error.message}${child.stderr ? `: ${child.stderr.trim().slice(-1000)}` : ""}`;
      }
    }
    run.completedAt = new Date().toISOString();
    runs.push(run);
    if (run.status !== "passed") await save(resolve(options.output, run.source), { status: run.status, reason: run.reason, ...(run.result ? { pageResult: run.result } : {}) });
    console.error(`${run.id}: ${run.status}${run.reason ? ` (${run.reason})` : ""}`);
    await writeSummary();
  }
  try {
    sourceChanged = (await runtimeIdentity(options.base, true)).fingerprint !== source.fingerprint
      || (await harnessIdentity(selected)).fingerprint !== source.harnesses.fingerprint
      || sha256(await readFile(registryPath)) !== source.registrySha256
      || JSON.stringify(await baselineIdentity(options.baselineWasm, options.base)) !== JSON.stringify(baseline);
  } catch (error) { sourceChanged = true; sourceChangeReason = error.message; }
  const summary = await writeSummary(true);
  console.log(JSON.stringify({ status: summary.status, counts: summary.counts, output: resolve(options.output, "summary.json") }));
  return summary;
}

async function main() {
  const options = profileOptions(process.argv.slice(2));
  if (options.help) { console.log(HELP); return; }
  if (options.list) {
    const registry = parseKernelRegistry(await readFile(resolve(ROOT, "crates/kevala/src/gpu.rs"), "utf8"));
    const dispatched = new Set(KERNEL_HARNESSES.filter((item) => item.id !== "compile").flatMap((item) => harnessKernels(item, registry)));
    console.log(JSON.stringify({ registered: registry.length, harnesses: KERNEL_HARNESSES.map((item) => ({ id: item.id, kernels: harnessKernels(item, registry), required: item.required, timing: item.timing })),
      withoutDispatchedFixture: registry.filter((name) => !dispatched.has(name)) }, null, 2));
    return;
  }
  await withProfileLock(async () => {
    const result = await runKernelProfile(options);
    process.exitCode = result.status === "failed" ? 1 : result.coverageComplete ? 0 : 2;
  });
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch((error) => { console.error(error.stack || String(error)); process.exitCode = 1; });
