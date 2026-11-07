#!/usr/bin/env node
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { lstat, readFile, realpath, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync, gunzipSync } from "node:zlib";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../../..");
const capture = "/tmp/kevala-no-cache-gpu-carries/tmp/kev9-bf16-short-medium-v1";
const lock = "/home/bruno/.cache/kevala/gpu-profile.lock";
const archivePath = join(here, "evidence.json.gz");
const manifestPath = join(here, "manifest.json");
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const mode = process.argv[2];
const allowIo = process.argv[3] === "--allow-io";
const capturePins = [
  ["capture/summary.json", "summary.json", 335457, "f97535b017116d31133e5ea0386502158148864ff4c7d657ad88ab5a27022855"],
  ["capture/chromium/kev-9b-bf16/0-wall.json", "chromium/kev-9b-bf16/0-wall.json", 139961, "01842ce8dbc5ea6084e25b6f99abb34254c8c51af074f9163e87a098e180bd47"],
  ["capture/chromium/kev-9b-bf16/0-gpu.json", "chromium/kev-9b-bf16/0-gpu.json", 149829, "a579a27b8b20d20d0f2883d9df1fcc5870f4e606e4239e38ec666a40b542ee80"],
  ["capture/chromium/kev-9b-bf16/1-gpu.json", "chromium/kev-9b-bf16/1-gpu.json", 149837, "0c7bfc6b6487da1ae201b7e30669bd40788f0dca8ba0d77bbae240674cc8355b"],
  ["capture/chromium/kev-9b-bf16/1-wall.json", "chromium/kev-9b-bf16/1-wall.json", 140129, "c5e94ef076c8f50c3dd9ed3ebe0a71faee6ddacf2da6f4e22097596cefa7ab7f"],
];
const sourcePins = [
  ["runtime/js/src/gpu-kev.js", "js/src/gpu-kev.js", 45464, "31f9e322832be04b0773e1866be724c49a5d7c9de7d523d5d07fdb8fafd6b614"],
  ["harness/bench.html", "bench.html", 14112, "b77a603f70145f8404e93dc74b2c6b8a3a6b35e50ede0ed10ecd5ed59d540d6c"],
  ["harness/dev/adapter-metadata.js", "dev/adapter-metadata.js", 898, "6aa9f18aef5334eb722749a113a73a02d9e26ec69770ec116a0cf12b98d4c9f8"],
  ["reproduction/scripts/profile-models.mjs", "scripts/profile-models.mjs", 22747, "85b2990345d39fdf0425639843dc3051aea33a0aa01dccf2fc8fef379fe1e7d1"],
  ["publication/docs/kev9-bf16-profile.md", "docs/kev9-bf16-profile.md", 2863, "c9a20a9b791443e57e950d81b0517b0aa4df57d1b361199319b29a1b4375212b"],
  ["publication/docs/benchmarks/kev9-bf16-2026-11-07/README.md", "docs/benchmarks/kev9-bf16-2026-11-07/README.md", 3688, "94ddbb9cbf0569bacb55ae6012e98b5e9d3e914363c9916a926a498a07242db3"],
  ["publication/app/views/how.js", "app/views/how.js", 16468, "dad62d1697a37c2456c3be6e3fcf0fb0d7a634dcde51d799b331270b356dc7d5"],
];
const exclusions = ["Kev-9B BF16 model weight and scale payloads", "WASM binaries and other compiled runtime payloads", "build trees and credentials"];
assert(["--plan", "--archive", "--verify"].includes(mode), "use --plan or --archive/--verify --allow-io");
assert(mode === "--plan" || allowIo, `${mode} requires --allow-io and the owner's I/O window`);

async function lease() {
  assert.equal(await realpath(lock), lock, "canonical profile lock path must resolve to itself");
  const info = await lstat(lock, { bigint: true });
  assert(info.isFile() && !info.isSymbolicLink(), "canonical profile lock must be a regular file");
  const major = info.dev >> 8n & 0xfffn | info.dev >> 32n & 0xfffff000n;
  const minor = info.dev & 0xffn | info.dev >> 12n & 0xffffff00n;
  const matches = (await readFile("/proc/locks", "utf8")).split("\n").filter(line => {
    const fields = line.trim().split(/\s+/), device = fields[5]?.split(":");
    return fields[1] === "FLOCK" && fields[2] === "ADVISORY" && fields[3] === "WRITE" &&
      fields[4] === String(process.pid) && device?.length === 3 &&
      BigInt(`0x${device[0]}`) === major && BigInt(`0x${device[1]}`) === minor &&
      BigInt(device[2]) === info.ino && fields[6] === "0" && fields[7] === "EOF";
  });
  assert.equal(matches.length, 1, "current Node PID must hold the canonical FLOCK ADVISORY WRITE 0..EOF");
  return { pid: process.pid, major: String(major), minor: String(minor), inode: String(info.ino), rawLine: matches[0] };
}

async function pinnedFile(path, location, bytes, digest) {
  const info = await lstat(location);
  assert(info.isFile() && !info.isSymbolicLink(), `not a regular file: ${path}`);
  assert.equal(await realpath(location), resolve(location), `symlink path rejected: ${path}`);
  assert.equal(info.size, bytes, `byte count differs: ${path}`);
  const data = await readFile(location);
  assert.equal(hash(data), digest, `SHA-256 differs: ${path}`);
  return { descriptor: { path, bytes: data.length, sha256: digest }, data };
}

async function publisherFile(includeData) {
  const path = "publication/docs/benchmarks/kev9-bf16-2026-11-07/archive-publication.mjs";
  const location = fileURLToPath(import.meta.url);
  const info = await lstat(location);
  assert(info.isFile() && !info.isSymbolicLink() && info.size < 1024 * 1024, "publisher must be a small regular file");
  assert.equal(await realpath(location), resolve(location), "publisher path must not traverse a symlink");
  const data = await readFile(location);
  return { path, bytes: data.length, sha256: hash(data), ...(includeData ? { encoding: "base64", data: data.toString("base64") } : {}) };
}

function referencesFor(summary) {
  return {
    runtime: { commit: summary.runtime.commit, tree: summary.runtime.tree, fingerprint: summary.runtime.sha256, assetCount: summary.runtime.assets.length, wasm: summary.runtime.assets.filter(asset => asset.path.endsWith(".wasm")) },
    adapter: { browser: "Chromium 156", fallbackStatus: "adapter.info.isFallbackAdapter=false; GPUAdapter.isFallbackAdapter unavailable" },
    pack: { bytes: summary.models[0].pack.bytes, sha256: summary.models[0].pack.sha256, verification: summary.models[0].pack.verification, catalogHashVerified: summary.models[0].pack.catalogHashVerified },
    sourceProfile: "resident uncached WebGPU; wall and diagnostic GPU blocks are separate",
  };
}

async function collect(includeData) {
  const files = [];
  let summaryBytes;
  for (const [path, relative, bytes, digest] of capturePins) {
    const found = await pinnedFile(path, join(capture, relative), bytes, digest);
    files.push({ ...found.descriptor, ...(includeData ? { encoding: "base64", data: found.data.toString("base64") } : {}) });
    if (path === "capture/summary.json") summaryBytes = found.data;
  }
  assert(summaryBytes, "pinned summary is missing");
  for (const [path, relative, bytes, digest] of sourcePins) {
    const found = await pinnedFile(path, resolve(repo, relative), bytes, digest);
    files.push({ ...found.descriptor, ...(includeData ? { encoding: "base64", data: found.data.toString("base64") } : {}) });
  }
  files.push(await publisherFile(includeData));

  const summary = JSON.parse(summaryBytes);
  assert.equal(summary.status, "complete");
  assert.equal(summary.runtime.commit, "2883619ff46856be57760112ae250df272e82b82");
  assert.equal(summary.runtime.tree, "c3069b7aace1bae00273de34b6e262593f831f68");
  assert.equal(summary.runtime.workingTreeDirty, false);
  assert.equal(summary.runtime.sha256, "311d4bc4b2046c5f13918135368b8805065e1ac4f26124b6c4232a77ac95271b");
  assert.equal(summary.runtime.assets.length, 53);
  assert.deepEqual(summary.runtime.assets.find(asset => asset.path === "gpu-kev.js"), { path: "gpu-kev.js", bytes: 45464, sha256: "31f9e322832be04b0773e1866be724c49a5d7c9de7d523d5d07fdb8fafd6b614" });
  assert.equal(summary.runtime.harnessSha256, "b77a603f70145f8404e93dc74b2c6b8a3a6b35e50ede0ed10ecd5ed59d540d6c");
  assert.deepEqual(summary.runtime.harnessAssets, [
    { path: "bench.html", bytes: 14112, sha256: "b77a603f70145f8404e93dc74b2c6b8a3a6b35e50ede0ed10ecd5ed59d540d6c" },
    { path: "dev/adapter-metadata.js", bytes: 898, sha256: "6aa9f18aef5334eb722749a113a73a02d9e26ec69770ec116a0cf12b98d4c9f8" },
  ]);
  assert.deepEqual(summary.models.map(model => model.id), ["kev-9b"]);
  assert.deepEqual(summary.models[0].raw, capturePins.slice(1).map(([, relative]) => relative));
  assert.equal(summary.models[0].pack.bytes, 15902057920);
  assert.equal(summary.models[0].pack.sha256, "32610758de1b2a192c668cc05cc6bbc97808bd74160f8034c1d11ea15070ff4e");
  assert.deepEqual(summary.models[0].cases.map(row => row.tokenRange), [[31, 32], [124, 125]]);
  assert.deepEqual(summary.protocol.models, ["kev-9b"]);
  assert.deepEqual(summary.protocol.precisions, ["bf16"]);
  assert.deepEqual(summary.protocol.browsers, ["chromium"]);
  assert.equal(summary.protocol.cacheMode, "uncached");
  assert.equal(summary.protocol.packCache, false);
  assert.equal(summary.protocol.stateCache, false);
  assert.equal(summary.protocol.profileProtocol, "resident");
  assert.equal(summary.protocol.samples, 9);
  assert.equal(summary.protocol.warmups, 3);
  assert.equal(summary.protocol.blocks, 2);

  const references = referencesFor(summary);
  files.sort((a, b) => a.path.localeCompare(b.path));
  return { files, references, exclusions };
}

async function verify() {
  const before = await lease();
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  assert.equal(manifest.schemaVersion, 1);
  assert.equal(manifest.archive.path, "evidence.json.gz");
  const compressed = await readFile(archivePath);
  assert.equal(compressed.length, manifest.archive.bytes);
  assert.equal(hash(compressed), manifest.archive.sha256);
  const decoded = gunzipSync(compressed);
  assert.equal(decoded.length, manifest.archive.decodedBytes);
  assert.equal(hash(decoded), manifest.archive.decodedSha256);
  const payload = JSON.parse(decoded);
  assert.equal(payload.files.length, manifest.files.length);
  const expected = new Map([...capturePins, ...sourcePins].map(([path, , bytes, sha256]) => [path, { bytes, sha256 }]));
  expected.set("publication/docs/benchmarks/kev9-bf16-2026-11-07/archive-publication.mjs", null);
  const seen = new Set();
  for (const [index, file] of payload.files.entries()) {
    const { data, encoding, ...descriptor } = file;
    assert.equal(encoding, "base64");
    assert.deepEqual(descriptor, manifest.files[index]);
    assert(expected.has(file.path), `unexpected archived path: ${file.path}`);
    assert(!seen.has(file.path), `duplicate archived path: ${file.path}`);
    seen.add(file.path);
    const pin = expected.get(file.path);
    if (pin) assert.deepEqual({ bytes: file.bytes, sha256: file.sha256 }, pin);
    const original = Buffer.from(data, "base64");
    assert.equal(original.length, file.bytes);
    assert.equal(hash(original), file.sha256);
  }
  assert.equal(payload.files.length, expected.size);
  const summaryFile = payload.files.find(file => file.path === "capture/summary.json");
  const summary = JSON.parse(Buffer.from(summaryFile.data, "base64"));
  assert.deepEqual(manifest.references, referencesFor(summary));
  assert.deepEqual(manifest.exclusions, exclusions);
  const after = await lease();
  assert.deepEqual(after, before);
  console.log(JSON.stringify({ verified: true, files: payload.files.length, archiveSha256: hash(compressed), manifestSha256: hash(await readFile(manifestPath)) }));
}

if (mode === "--verify") await verify();
else {
  const before = mode === "--archive" ? await lease() : null;
  const result = await collect(mode === "--archive");
  if (mode === "--plan") {
    console.log(JSON.stringify({ status: "metadata-only; no compression or copying", files: result.files, references: result.references, exclusions: result.exclusions }, null, 2));
  } else {
    for (const path of [archivePath, manifestPath]) {
      try { await lstat(path); throw Error(`refusing existing output: ${path}`); }
      catch (error) { if (error.code !== "ENOENT") throw error; }
    }
    const decoded = Buffer.from(JSON.stringify({ schemaVersion: 1, files: result.files }) + "\n");
    const compressed = gzipSync(decoded, { level: 9, mtime: 0 });
    assert.equal(hash(gunzipSync(compressed)), hash(decoded), "gzip byte roundtrip failed");
    const after = await lease();
    assert.deepEqual(after, before);
    const manifest = {
      schemaVersion: 1,
      recordedAt: new Date().toISOString(),
      archive: { path: "evidence.json.gz", bytes: compressed.length, sha256: hash(compressed), decodedBytes: decoded.length, decodedSha256: hash(decoded) },
      files: result.files.map(({ data, encoding, ...descriptor }) => descriptor),
      references: result.references,
      exclusions: result.exclusions,
      lease: { before, after },
    };
    await writeFile(archivePath, compressed, { flag: "wx" });
    await writeFile(manifestPath, JSON.stringify(manifest, null, 2) + "\n", { flag: "wx" });
    console.log(JSON.stringify({ archived: true, files: result.files.length, archiveSha256: hash(compressed), manifestSha256: hash(Buffer.from(JSON.stringify(manifest, null, 2) + "\n")) }));
  }
}
