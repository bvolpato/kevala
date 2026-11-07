import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { lstat, readFile, realpath, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync, gunzipSync } from "node:zlib";

const here = dirname(fileURLToPath(import.meta.url)), repo = resolve(here, "../../..");
const source = "/home/bruno/githubworkspace/kevala/tmp/sota-20261106-kdDTGn/f32-weight-cache-v1";
const identityPath = "/home/bruno/githubworkspace/kevala/tmp/sota-20261106-kdDTGn/native-2x4-delivery/ci-default-smoke-v1/runtime-identity.json";
const lock = "/home/bruno/.cache/kevala/gpu-profile.lock", archivePath = join(here, "evidence.json.gz"), manifestPath = join(here, "manifest.json");
const pins = { plan: "14b01b9a47b41e743943a5f789717e458bc126411e01e011e0fcaea9fc874e5a", index: "6aaae640f67e9206a0d1c69f0ce4180b02d847fc0736dcc1b5beebec28419b71", inventory: "cda0e3604f53e87256cfafdbf3188545e27e8c325e1eb5afe26262857520bde4", receipt: "8178d96770ba9e4b383f840a344451cbda9d4577115b4ef42114eb33b6f070e7", raw: "6c7d68548a1ab49556d77a1f00a2d4de16405b2cdb8a3cd478d918eb66c08321", draftIndex: "9739121060a1509787fcef24836ca629d9d35bbf3de1a29c8f3f6fd70582b126", identity: "68356dc277b6b622273fa3e37c378a03e71251e66747915a28fb62fa0cd797b7" };
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const mode = process.argv[2];
assert(["--plan", "--archive", "--verify"].includes(mode), "use --plan or --archive/--verify --allow-io");
assert(mode === "--plan" || process.argv[3] === "--allow-io", "payload I/O requires the owner's separate window");
async function lease() {
  assert.equal(await realpath(lock), lock);
  const info = await lstat(lock, { bigint: true }); assert(info.isFile() && !info.isSymbolicLink());
  const major = info.dev >> 8n & 0xfffn | info.dev >> 32n & 0xfffff000n, minor = info.dev & 0xffn | info.dev >> 12n & 0xffffff00n;
  const rows = (await readFile("/proc/locks", "utf8")).split("\n").filter(line => {
    const f = line.trim().split(/\s+/), d = f[5]?.split(":");
    return f[1] === "FLOCK" && f[2] === "ADVISORY" && f[3] === "WRITE" && f[4] === String(process.pid) && d?.length === 3 && BigInt(`0x${d[0]}`) === major && BigInt(`0x${d[1]}`) === minor && d[2] === String(info.ino) && f[6] === "0" && f[7] === "EOF";
  });
  assert.equal(rows.length, 1, "current Node PID must hold the canonical FLOCK ADVISORY WRITE 0..EOF");
  return { pid: process.pid, identity: { major: String(major), minor: String(minor), inode: String(info.ino) }, rawKernelLine: rows[0] };
}
async function json(path, expected) { const bytes = await readFile(path); assert.equal(hash(bytes), expected, path); return JSON.parse(bytes); }
async function collect(payloads) {
  const index = await json(join(source, "source-index.json"), pins.index), draft = await json(join(source, "pre-review-v1/source-index.json"), pins.draftIndex);
  const inventory = await json(join(source, "runs-v1/inventory.json"), pins.inventory), identity = await json(identityPath, pins.identity);
  const plan = await json(join(source, "plan.json"), pins.plan), receipt = await json(join(source, "runs-v1/receipt.json"), pins.receipt);
  assert.equal(index.planSha256, pins.plan); assert.equal(receipt.status, "passed"); assert.equal(receipt.rawBrowser.sha256, pins.raw); assert.equal(receipt.validation.candidateModelOrDefaultPromotion, false);
  assert.deepEqual(index.files.map(x => x.path).sort(), ["prepare.mjs", "probe.html", "probe.js", "run.mjs", "validate.mjs"]);
  assert.deepEqual(draft.files.map(x => x.path).sort(), index.files.map(x => x.path).sort());
  assert.deepEqual(inventory.files.map(x => x.path).sort(), ["admission.json", "command.json", "input.json", "raw-browser.json", "receipt.json", "runner.stderr", "runner.stdout"]);
  const specs = [], add = (path, location, pin = null, bytes = null) => specs.push({ path, location, pin, bytes });
  for (const f of index.files) add(`source/${f.path}`, join(source, f.path), f.sha256, f.bytes);
  add("source/plan.json", join(source, "plan.json"), pins.plan); add("source/source-index.json", join(source, "source-index.json"), pins.index);
  for (const f of inventory.files) add(`results/${f.path}`, join(source, "runs-v1", f.path), f.sha256, f.bytes);
  add("results/inventory.json", join(source, "runs-v1/inventory.json"), pins.inventory);
  for (const f of draft.files) add(`draft/${f.path}`, join(source, "pre-review-v1", f.path), f.sha256, f.bytes);
  add("draft/plan.json", join(source, "pre-review-v1/plan.json"), draft.planSha256); add("draft/source-index.json", join(source, "pre-review-v1/source-index.json"), pins.draftIndex);
  add("identity/runtime-ci71.json", identityPath, pins.identity);
  for (const name of ["README.md", "archive-publication.mjs"]) add(`publication/${name}`, join(here, name));
  add("publication/performance-reference.md", join(repo, "docs/performance-reference.md"));
  const files = [];
  for (const spec of specs.sort((a, b) => a.path.localeCompare(b.path))) {
    const info = await lstat(spec.location); assert(info.isFile() && !info.isSymbolicLink() && info.size < 1024 ** 2, spec.path);
    if (spec.bytes !== null) assert.equal(info.size, spec.bytes, spec.path);
    const data = payloads || !spec.pin ? await readFile(spec.location) : null, sha256 = data ? hash(data) : spec.pin;
    if (spec.pin) assert.equal(sha256, spec.pin, spec.path);
    files.push({ path: spec.path, bytes: info.size, sha256, ...(payloads ? { encoding: "base64", data: data.toString("base64") } : {}) });
  }
  assert.equal(files.length, 26);
  const references = { runtime: { commit: identity.commit, runId: identity.runId, buildAttempt: identity.buildAttempt, fingerprint: identity.fingerprint, assets: identity.assets, provenance: identity.provenance, pagesVerified: false }, browserRunner: { commit: "4e27753e6f385d6fabbb6040fdd82f058a9c48a8", ...plan.benchRunner }, model: { ...plan.pack, selectedTensor: receipt.packBefore.tensor, freshWholePackHashClaimed: false }, draft: { status: "unexecuted first source draft; exact bytes included under draft/", planSha256: draft.planSha256 } };
  return { files, references, exclusions: ["model weight and scale payloads", "activation and output buffers", "compiled runtime binaries", "build trees", "credentials"] };
}
async function verify() {
  const before = await lease(), manifest = JSON.parse(await readFile(manifestPath, "utf8")), compressed = await readFile(archivePath);
  assert.equal(compressed.length, manifest.archive.bytes); assert.equal(hash(compressed), manifest.archive.sha256);
  const bytes = gunzipSync(compressed); assert.equal(bytes.length, manifest.archive.decodedBytes); assert.equal(hash(bytes), manifest.archive.decodedSha256);
  const payload = JSON.parse(bytes); assert.equal(payload.files.length, manifest.files.length); assert.equal(payload.files.length, 26);
  for (const [i, file] of payload.files.entries()) { const { data, encoding, ...descriptor } = file; assert.equal(encoding, "base64"); assert.deepEqual(descriptor, manifest.files[i]); const original = Buffer.from(data, "base64"); assert.equal(original.length, file.bytes); assert.equal(hash(original), file.sha256); }
  for (const [path, expected] of [["results/receipt.json", pins.receipt], ["results/raw-browser.json", pins.raw], ["source/plan.json", pins.plan]]) assert.equal(payload.files.find(x => x.path === path)?.sha256, expected);
  const after = await lease(); assert.deepEqual(after.identity, before.identity); assert.equal(after.pid, before.pid);
  console.log(JSON.stringify({ verified: true, files: payload.files.length, archiveSha256: hash(compressed), manifestSha256: hash(await readFile(manifestPath)) }));
}
if (mode === "--verify") await verify();
else {
  const before = mode === "--archive" ? await lease() : null, result = await collect(mode === "--archive");
  if (mode === "--plan") console.log(JSON.stringify({ status: "metadata-only; no compression or copying", files: result.files, references: result.references, exclusions: result.exclusions }, null, 2));
  else {
    for (const path of [archivePath, manifestPath]) { try { await lstat(path); throw Error(`refusing existing output: ${path}`); } catch (error) { if (error.code !== "ENOENT") throw error; } }
    const decoded = Buffer.from(JSON.stringify({ schemaVersion: 1, files: result.files }) + "\n"), compressed = gzipSync(decoded, { level: 9, mtime: 0 });
    const after = await lease(); assert.deepEqual(after.identity, before.identity); assert.equal(after.pid, before.pid);
    const manifest = { schemaVersion: 1, recordedAt: new Date().toISOString(), verdict: "rejected projection cache screen; no model/default promotion", archive: { path: "evidence.json.gz", bytes: compressed.length, sha256: hash(compressed), decodedBytes: decoded.length, decodedSha256: hash(decoded) }, files: result.files.map(({ data, encoding, ...descriptor }) => descriptor), references: result.references, exclusions: result.exclusions, lease: { before, after } };
    await writeFile(archivePath, compressed, { flag: "wx" }); await writeFile(manifestPath, JSON.stringify(manifest, null, 2) + "\n", { flag: "wx" });
    console.log(JSON.stringify({ archived: true, files: result.files.length, archiveSha256: hash(compressed), manifestSha256: hash(Buffer.from(JSON.stringify(manifest, null, 2) + "\n")) }));
  }
}
