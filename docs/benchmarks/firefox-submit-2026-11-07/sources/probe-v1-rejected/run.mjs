import { createHash } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import { mkdir, open, readFile, stat, writeFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { freemem, loadavg } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { MODELS } from "../../../js/src/source.js";
import { packHeaderLength, parsePackHeader } from "../../../js/src/pack-layout.js";
import { runPage, failed } from "../../../scripts/lib/browser-run.mjs";
import { withProfileLock } from "../../../scripts/lib/profile-lock.mjs";
import { verifyPackHeader } from "../../../scripts/lib/model-profile.mjs";
import { verifyFixtures } from "./preflight.mjs";
import { validateResult } from "./validate.mjs";

const taskDir = dirname(fileURLToPath(import.meta.url));
const root = resolve(taskDir, "../../../");
const runtimeDir = join(root, "tmp/sota-20261106-kdDTGn/staged-runtime-v3");
const runtimeIdentityPath = join(root, "tmp/sota-20261106-kdDTGn/staged-runtime-v3-identity.json");
const validationManifestPath = join(root, "tmp/sota-20261106-kdDTGn/staged-validation-v4/binaries.json");
const referencePath = join(root, "tmp/kernel-profile-2026-11-06-I6lJc0/firefox-current-models/firefox/laya-q8/0-wall.json");
const summaryPath = join(root, "tmp/kernel-profile-2026-11-06-I6lJc0/firefox-current-models/summary-validated.json");
const packPath = join(root, "tmp/laya-q8.kevala");
const packSha256 = "b9cab75c38f026236b8d736377aa40f889d842c219d37da45cc06c782395555c";
const packBytes = 478886720;
const packHeaderSha256 = "f82cc6ce21f0b3f55000f505db8d7ee08f5f956ce9d1d1206607bf2c6476ecc8";
const packHeaderBytes = 26313;
const referenceSha256 = "2f6316f374500475038c3dab3280223c25b568cb4b061cd676e79eccb37b7375";
const runtimeIdentitySha256 = "c5d3c308a5a0c5fd23f3d60f65093c21a20029adbb2a3672a6bbb4eb397eab4a";
const runtimeFingerprint = "eb46c3119c1befb08f1e63bcd3499583ed6684af00d9d6094caa845ca7754bc6";
const validationManifestSha256 = "5bc792d9610a01b877563c629d7ce504ded110db2b887e03998b571703e6df91";
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");

async function fileSha256(path) {
  const digest = createHash("sha256");
  const file = await open(path, "r");
  try {
    const chunk = Buffer.allocUnsafe(1024 * 1024);
    let position = 0;
    for (;;) {
      const { bytesRead } = await file.read(chunk, 0, chunk.length, position);
      if (!bytesRead) break;
      digest.update(chunk.subarray(0, bytesRead));
      position += bytesRead;
    }
  } finally {
    await file.close();
  }
  return digest.digest("hex");
}

async function runtimeSnapshot() {
  const identityBytes = await readFile(runtimeIdentityPath);
  const identity = JSON.parse(identityBytes);
  if (hash(identityBytes) !== runtimeIdentitySha256 || identity.fingerprint !== runtimeFingerprint
    || identity.name !== "staged-runtime-v3" || identity.sourceCommit !== "be350ec53b774bbfff3733d690cc9689625a02e7") {
    throw new Error("unexpected staged-runtime-v3 identity");
  }
  const assets = [];
  for (const item of identity.assets) {
    const path = join(runtimeDir, item.path);
    const file = await stat(path);
    if (!file.isFile() || file.size !== item.bytes) throw new Error(`runtime asset size changed: ${item.path}`);
    const sha256 = await fileSha256(path);
    if (sha256 !== item.sha256) throw new Error(`runtime asset hash changed: ${item.path}`);
    assets.push({ path: item.path, bytes: file.size, sha256 });
  }
  const validationBytes = await readFile(validationManifestPath);
  const validation = JSON.parse(validationBytes);
  if (hash(validationBytes) !== validationManifestSha256 || validation.artifacts.length !== 3) {
    throw new Error("staged WASM validation manifest changed or is incomplete");
  }
  const identityWasm = new Map(identity.assets
    .filter((item) => /^kevala-(base|simd|relaxed)\.wasm$/.test(item.path))
    .map((item) => [item.path, item.sha256]));
  for (const item of validation.artifacts) {
    if (identityWasm.get(`kevala-${item.flavor}.wasm`) !== item.sha256) throw new Error(`staged ${item.flavor} WASM differs from its validation receipt`);
  }
  return {
    identity: { path: "tmp/sota-20261106-kdDTGn/staged-runtime-v3-identity.json", sha256: hash(identityBytes), fingerprint: identity.fingerprint },
    validation: { path: "tmp/sota-20261106-kdDTGn/staged-validation-v4/binaries.json", sha256: hash(validationBytes) },
    assets,
  };
}

async function packSnapshot() {
  const before = await stat(packPath);
  const spec = MODELS.laya;
  if (!before.isFile() || before.size !== packBytes) throw new Error("Laya Q8 pack size differs from the pinned fixture");
  if (spec?.packSha256 !== packSha256) throw new Error("Laya catalog pack digest differs from the pinned fixture");
  const summaryBytes = await readFile(summaryPath);
  const summary = JSON.parse(summaryBytes);
  const prior = summary.models.find((row) => row.id === "laya" && row.precision === "q8" && row.browser === "firefox");
  if (prior?.pack?.sha256 !== packSha256 || prior.pack.catalogHashVerified !== true) {
    throw new Error("Laya pack digest is not backed by the prior validated capture receipt");
  }
  const file = await open(packPath, "r");
  let identity;
  let headerBytes;
  let headerSha256;
  try {
    const prefix = Buffer.alloc(16);
    if ((await file.read(prefix, 0, prefix.length, 0)).bytesRead !== prefix.length || prefix.toString("ascii", 0, 4) !== "KVLA") {
      throw new Error("Laya pack has an invalid KVLA prefix");
    }
    const length = packHeaderLength(prefix);
    if (length !== packHeaderBytes || length > Math.min(before.size, 16 * 1024 * 1024)) throw new Error("Laya pack header length differs from the pinned fixture");
    const bytes = Buffer.alloc(length);
    if ((await file.read(bytes, 0, length, 0)).bytesRead !== length) throw new Error("Laya pack header is truncated");
    identity = verifyPackHeader(parsePackHeader(bytes), length, before.size, "laya", "q8", spec);
    headerBytes = length;
    headerSha256 = hash(bytes);
  } finally {
    await file.close();
  }
  const after = await stat(packPath);
  if (before.size !== after.size || before.ino !== after.ino || before.mtimeMs !== after.mtimeMs) throw new Error("Laya pack changed while reading its header");
  if (headerSha256 !== packHeaderSha256) throw new Error("Laya pack header differs from the pinned fixture");
  return {
    path: "tmp/laya-q8.kevala",
    bytes: after.size,
    ino: after.ino,
    mtimeMs: after.mtimeMs,
    headerBytes,
    headerSha256,
    expectedFullSha256: packSha256,
    priorVerifiedReceipt: { path: "tmp/kernel-profile-2026-11-06-I6lJc0/firefox-current-models/summary-validated.json", sha256: hash(summaryBytes) },
    identity,
  };
}

async function memorySnapshot() {
  let meminfo = null;
  try {
    const contents = await readFile("/proc/meminfo", "utf8");
    const field = (name) => Number(new RegExp(`^${name}:\\s+(\\d+)`, "m").exec(contents)?.[1]) * 1024;
    meminfo = { availableBytes: field("MemAvailable"), swapFreeBytes: field("SwapFree") };
  } catch {}
  const gpu = spawnSync("nvidia-smi", ["--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], { encoding: "utf8", timeout: 10000 });
  return { freeMemoryBytes: freemem(), loadAverage: loadavg(), meminfo, nvidia: gpu.status === 0 ? gpu.stdout.trim() : null };
}

async function sourceSnapshot() {
  const runtime = await runtimeSnapshot();
  const pack = await packSnapshot();
  const inputs = [];
  for (const path of [
    "js/src/source.js",
    "js/src/pack-layout.js",
    "scripts/lib/model-profile.mjs",
    "scripts/lib/browser-run.mjs",
    "scripts/lib/profile-lock.mjs",
    "scripts/bench-gpu.py",
    "tmp/kernel-profile-2026-11-06-I6lJc0/firefox-current-models/firefox/laya-q8/0-wall.json",
    "tmp/kernel-profile-2026-11-06-I6lJc0/firefox-current-models/summary-validated.json",
  ]) {
    const bytes = await readFile(join(root, path));
    const sha256 = hash(bytes);
    if (path.endsWith("/0-wall.json") && sha256 !== referenceSha256) throw new Error("historical Laya fixture reference SHA-256 changed");
    inputs.push({ path, bytes: bytes.length, sha256 });
  }
  const harness = [];
  for (const path of ["index.html", "firefox-submit.js", "fixtures.mjs", "preflight.mjs", "validate.mjs", "run.mjs"]) {
    const bytes = await readFile(join(taskDir, path));
    harness.push({ path, bytes: bytes.length, sha256: hash(bytes) });
  }
  return { runtime, pack, inputs, harness };
}

async function verifyServedPack(base, pack) {
  const url = new URL("tmp/laya-q8.kevala", base);
  const head = await fetch(url, { method: "HEAD", cache: "no-store" });
  if (!head.ok || Number(head.headers.get("content-length")) !== pack.bytes) throw new Error("served Laya pack length differs from the pinned file");
  const range = await fetch(url, { headers: { Range: `bytes=0-${pack.headerBytes - 1}` }, cache: "no-store" });
  if (range.status !== 206 || Number(range.headers.get("content-length")) !== pack.headerBytes) throw new Error("served Laya pack does not provide the bounded pinned header range");
  const header = Buffer.from(await range.arrayBuffer());
  const headerSha256 = hash(header);
  if (headerSha256 !== pack.headerSha256) throw new Error("served Laya pack header differs from the pinned local header");
  return { url: url.href, headStatus: head.status, contentLength: Number(head.headers.get("content-length")), rangeStatus: range.status, headerBytes: header.length, headerSha256 };
}

async function execute() {
  const fixturePreflight = await verifyFixtures(root);
  console.log(JSON.stringify({ status: "fixture-preflight-passed", fixtures: fixturePreflight }));
  const outputDir = join(taskDir, "results");
  await mkdir(outputDir, { recursive: true });
  const stamp = new Date().toISOString().replaceAll(":", "-");
  const rawPath = join(outputDir, `raw-${stamp}.json`);
  const receiptPath = join(outputDir, `runner-${stamp}.json`);
  const base = new URL(process.env.KEVALA_BENCH_URL || "http://127.0.0.1:18174/");
  if (!base.pathname.endsWith("/")) base.pathname += "/";
  const page = new URL("tmp/sota-20261106-kdDTGn/firefox-submit-v1/index.html", base);
  const memoryBefore = await memorySnapshot();
  const before = await sourceSnapshot();
  let child = null;
  let pageResult = null;
  let validation = null;
  let servedPack = null;
  let runnerError = null;
  try {
    servedPack = await verifyServedPack(base, before.pack);
    const pageResponse = await fetch(page, { cache: "no-store" });
    if (!pageResponse.ok) throw new Error(`Firefox submit probe page: HTTP ${pageResponse.status}`);
    child = runPage({ result: "latency", url: page.href, output: rawPath, backend: "webgpu", browser: "firefox", timeoutSeconds: 900, stdio: "pipe" });
    if (child.stdout) process.stdout.write(child.stdout);
    if (child.stderr) process.stderr.write(child.stderr);
    try { pageResult = JSON.parse(await readFile(rawPath, "utf8")); } catch {}
    if (failed(child)) throw new Error(pageResult?.error || child.error?.message || child.stderr?.trim() || `browser runner exited ${child.status ?? child.signal}`);
    validation = validateResult(pageResult);
  } catch (error) {
    runnerError = String(error?.message || error);
  }
  let after;
  try {
    after = await sourceSnapshot();
  } catch (error) {
    runnerError ||= String(error?.message || error);
  }
  const memoryAfter = await memorySnapshot();
  const sourceStable = Boolean(after && isDeepStrictEqual(before, after));
  if (!sourceStable) runnerError ||= "source, runtime, or pack identity changed during the probe";
  const rawSha256 = pageResult ? hash(await readFile(rawPath)) : null;
  const receipt = {
    schemaVersion: 1,
    status: !runnerError && sourceStable ? "passed" : "failed",
    capturedAt: new Date().toISOString(),
    lock: "scripts/lib/profile-lock.mjs withProfileLock using the canonical ~/.cache/kevala/gpu-profile.lock",
    browserRunner: "scripts/lib/browser-run.mjs runPage -> scripts/bench-gpu.py --result latency",
    page: page.href,
    timeoutSeconds: 900,
    browser: "firefox",
    memory: { before: memoryBefore, after: memoryAfter },
    packVerificationScope: "Reuse the previously verified full-pack SHA-256; before/after compare size, inode, mtime, and a bounded local header SHA-256. The served page verifies HEAD length and byte-range header SHA-256. No full-pack rehash.",
    servedPack,
    sourceIdentity: { before, after, unchanged: sourceStable },
    rawResult: { path: rawPath, sha256: rawSha256, status: pageResult?.status || null, cases: pageResult?.cases?.length || 0 },
    fixturePreflight,
    validation,
    browserProcess: child ? { status: child.status, signal: child.signal, error: child.error?.message || null } : null,
    runnerError,
  };
  await writeFile(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, { flag: "wx" });
  console.log(JSON.stringify({ status: receipt.status, rawPath, rawSha256, receiptPath, runnerError }, null, 2));
  if (receipt.status !== "passed") process.exitCode = 1;
}

await withProfileLock(execute);
