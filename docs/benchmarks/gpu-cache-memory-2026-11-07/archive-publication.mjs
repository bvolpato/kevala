import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { lstat, readFile, realpath, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync, gunzipSync } from "node:zlib";

const here = dirname(fileURLToPath(import.meta.url)), repo = resolve(here, "../../..");
const source = "/home/bruno/githubworkspace/kevala/tmp/sota-20261106-kdDTGn/no-cache-carries";
const ci = "/home/bruno/githubworkspace/kevala/tmp/sota-20261106-kdDTGn/native-auto4-delivery/ci-site-4e27753e6f385d6fabbb6040fdd82f058a9c48a8-run42350139676-v1/js/src";
const lock = "/home/bruno/.cache/kevala/gpu-profile.lock", archivePath = join(here, "evidence.json.gz"), manifestPath = join(here, "manifest.json");
const hash = value => createHash("sha256").update(value).digest("hex");
const args = process.argv.slice(2), mode = args[0];
assert(["--plan", "--archive", "--verify"].includes(mode), "use --plan or --archive/--verify --allow-io");
assert.deepEqual(args, mode === "--plan" ? ["--plan"] : [mode, "--allow-io"], "unknown argument or missing --allow-io");
const maxFile = 64 * 1024 ** 2, maxEnvelope = 192 * 1024 ** 2;
const packets = [
  { id: "control", dir: "diagnostic-control-v1", runs: "runs-control-v1", index: "1f2b5aabb568ceab80d1556659528003a846c7fcfb337cfbec398b4526a37c8a", files: [
    ["receipt.json", 107102, "0a88207c22cfad784d2c4b742082b71be7ae04fea508a1bdce934c8d81c5ca69"],
    ["raw-browser.json", 63634310, "45b0f3e683e8a5e87c2d554a160dc23b88913f884fccd4038363b2400b75b788"],
    ["input.json", 4100753, "18e81950607d1eda78bd8aee95a8b27dd470cc1f9dd1200769f7ff041cacc63c"],
    ["command.json", 1081, "75c816fe9b529af1671579cb79e92af7d770a0bfca0921601f4449251b5497e3"],
  ] },
  { id: "failed-v2", dir: "hardware-v2", runs: "runs-v2", index: "a0a6bb27c1188a9290137532a0de5d5920312b9f397da07e8d36dadae3248d43", files: [
    ["receipt.json", 27922, "86bc04bd8affde2ae8abd2d9d094c6e9c38eb64615ee72acf01ca156095e9780"],
    ["raw-browser.json", 17980192, "77de150e61ef9270c210a0ac158424fd34cd0ff37a4f81bb36335feb7b4d5732"],
    ["input.json", 20764, "1418e55428860b798400317ad82cd540b48338d037cf2e89b1174ddfac63366f"],
    ["command.json", 1045, "d4123ba73767a9f3d2cfde6aeeafd5508c5051d235263ce55a1e36a1badb2cb3"],
  ] },
  { id: "failed-v1", dir: "hardware-v1", runs: "runs-v1", index: "709b99d1372cc49c4fec44ef245287a8b20d6b095d4548e242d4bff209c4e9f6", files: [
    ["receipt.json", 27380, "3ece5ec96030961100719e04f767490e67f00fc5d84a09202c4157a46e492bbf"],
    ["raw-browser.json", 29765, "235ad5d6e80876ee9fddcfcfac75b77bd5c71e0587c65a046a91309733b71106"],
    ["input.json", 20388, "8191224bca824abc9be5e86e70a3b834dbbe5660a5bce80c077332247653112f"],
    ["command.json", 1045, "c0090c5a984cc5a5400589205b0e918e70f4d44f9d402d2d43c53cb1809b3de7"],
  ] },
];
const freshness = [
  ["publication/README.md", "docs/benchmarks/gpu-cache-memory-2026-11-07/README.md"],
  ["publication/archive-publication.mjs", "docs/benchmarks/gpu-cache-memory-2026-11-07/archive-publication.mjs"],
  ["publication/gpu-cache-memory.md", "docs/gpu-cache-memory.md"],
  ["publication/how.js", "app/views/how.js"],
  ["source/candidate-gpu-kev.js", "js/src/gpu-kev.js"],
];
const gpuSources = [
  ["source/original-gpu-kev.js", join(ci, "gpu-kev.js"), 45228, "37519ef8c06c537c5fc736a7badef762062d082a7c3c19e605e462912c935a57"],
  ["source/candidate-gpu-kev.js", join(repo, "js/src/gpu-kev.js"), 45464, "31f9e322832be04b0773e1866be724c49a5d7c9de7d523d5d07fdb8fafd6b614"],
];
const safe = path => { assert(typeof path === "string" && !path.startsWith("/") && path.split(/[\\/]/).every(part => part && part !== "." && part !== ".."), `unsafe archive path: ${path}`); return path; };
async function lease() {
  assert.equal(await realpath(lock), lock); const info = await lstat(lock, { bigint: true }); assert(info.isFile() && !info.isSymbolicLink());
  const major = info.dev >> 8n & 0xfffn | info.dev >> 32n & 0xfffff000n, minor = info.dev & 0xffn | info.dev >> 12n & 0xffffff00n;
  const rows = (await readFile("/proc/locks", "utf8")).split("\n").filter(line => { const f = line.trim().split(/\s+/), d = f[5]?.split(":"); return f[1] === "FLOCK" && f[2] === "ADVISORY" && f[3] === "WRITE" && f[4] === String(process.pid) && d?.length === 3 && BigInt(`0x${d[0]}`) === major && BigInt(`0x${d[1]}`) === minor && d[2] === String(info.ino) && f[6] === "0" && f[7] === "EOF"; });
  assert.equal(rows.length, 1, "current Node PID must own the canonical full-range FLOCK");
  return { pid: process.pid, device: String(info.dev), inode: String(info.ino), rawKernelLine: rows[0] };
}
async function checkedRead(path, pin, limit = maxFile) {
  const before = await lstat(path, { bigint: true }); assert(before.isFile() && !before.isSymbolicLink() && before.size <= BigInt(limit), path);
  const bytes = await readFile(path), after = await lstat(path, { bigint: true });
  for (const key of ["dev", "ino", "size", "mtimeNs", "ctimeNs"]) assert.equal(after[key], before[key], `file changed: ${path}`);
  if (pin) assert.equal(hash(bytes), pin, path);
  return bytes;
}
async function collect(payloads) {
  const specs = [], add = (path, location, pin = null, bytes = null, payload = false) => specs.push({ path: safe(path), location, pin, bytes, payload });
  let receipt;
  for (const packet of packets) {
    const directory = join(source, packet.dir), index = JSON.parse(await checkedRead(join(directory, "source-index.json"), packet.index, 256 * 1024));
    add(`${packet.id}/source/source-index.json`, join(directory, "source-index.json"), packet.index);
    for (const file of index.files) add(`${packet.id}/source/${safe(file.path)}`, join(directory, file.path), file.sha256, file.bytes);
    for (const [name, bytes, pin] of packet.files) add(`${packet.id}/results/${name}`, join(directory, packet.runs, name), pin, bytes, name === "input.json" || name === "raw-browser.json");
    if (packet.id === "control") receipt = JSON.parse(await checkedRead(join(directory, packet.runs, "receipt.json"), packet.files[0][2], 256 * 1024));
  }
  assert.equal(receipt.status, "diagnostic-complete-existing-cross-mode-difference"); assert.equal(receipt.raw.sha256, packets[0].files[1][2]);
  for (const [path, location, bytes, pin] of gpuSources) add(path, location, pin, bytes);
  for (const path of ["scripts/bench-gpu.py", "scripts/lib/model-profile.mjs", "js/src/source.js", "js/src/pack-layout.js"]) { const file = receipt.sourcesBefore.find(item => item.path === `/home/bruno/githubworkspace/kevala/${path}`); assert(file); add(`dependencies/${path}`, join(repo, path), file.sha256, file.bytes); }
  for (const [path, repoPath] of freshness.filter(([path]) => path.startsWith("publication/"))) add(path, join(repo, repoPath));
  assert.equal(specs.length, 40); assert.equal(new Set(specs.map(x => x.path)).size, specs.length);
  const files = [];
  for (const spec of specs.sort((a, b) => a.path.localeCompare(b.path))) {
    const info = await lstat(spec.location); assert(info.isFile() && !info.isSymbolicLink() && info.size <= maxFile, spec.path); if (spec.bytes !== null) assert.equal(info.size, spec.bytes, spec.path);
    const data = payloads || !spec.payload ? await checkedRead(spec.location, spec.pin) : null;
    assert(data || spec.pin, `payload requires a pinned digest: ${spec.path}`);
    files.push({ path: spec.path, bytes: info.size, sha256: data ? hash(data) : spec.pin, ...(payloads ? { encoding: "base64", data: data.toString("base64") } : {}) });
  }
  const runtime = ["original-ci73", "candidate"].map(id => { const marker = `/runtime-control-v1/${id}/`; const assets = receipt.sourcesBefore.filter(x => x.path.includes(marker)).map(x => ({ path: x.path.split(marker)[1], bytes: x.bytes, sha256: x.sha256 })); assert.equal(assets.length, 53); return { id, assets }; });
  return { files, references: { runtime: { commit: "4e27753e6f385d6fabbb6040fdd82f058a9c48a8", run: "42350139676", arms: runtime, compiledPayloadsIncluded: false }, controllerCheckout: receipt.controllerCheckout, model: { id: "kev-0.8b", precision: "q8", bytes: 857259648, sha256: "a121fd39a959423e5f6a07d75a4cb33fdb87c0534aa27b7acf4f5381788097e3", headerSha256: "bc43c176cf887d05f5e1908fa8a68bb65329d324a78c3fe63053a39df42aab38" }, auditedReceiptSha256: packets[0].files[0][2] }, exclusions: ["model weights", "copied compiled runtime binaries and runtime trees", "build trees", "browser profiles and logs"] };
}
function decode(compressed, manifest) {
  assert.equal(compressed.length, manifest.archive.bytes); assert.equal(hash(compressed), manifest.archive.sha256);
  const bytes = gunzipSync(compressed, { maxOutputLength: maxEnvelope }); assert.equal(bytes.length, manifest.archive.decodedBytes); assert.equal(hash(bytes), manifest.archive.decodedSha256);
  const payload = JSON.parse(bytes); assert.equal(payload.files.length, 40); assert.equal(manifest.files.length, 40);
  const decoded = new Map();
  payload.files.forEach((file, i) => { safe(file.path); const { data, encoding, ...descriptor } = file; assert.equal(encoding, "base64"); assert.deepEqual(descriptor, manifest.files[i]); assert(!decoded.has(file.path)); const original = Buffer.from(data, "base64"); assert.equal(original.length, file.bytes); assert.equal(hash(original), file.sha256); decoded.set(file.path, original); });
  for (const packet of packets) {
    const indexPath = `${packet.id}/source/source-index.json`; assert.equal(hash(decoded.get(indexPath)), packet.index); const index = JSON.parse(decoded.get(indexPath));
    for (const file of index.files) { const original = decoded.get(`${packet.id}/source/${file.path}`); assert.equal(original.length, file.bytes); assert.equal(hash(original), file.sha256); }
    for (const [name, size, pin] of packet.files) { const original = decoded.get(`${packet.id}/results/${name}`); assert.equal(original.length, size); assert.equal(hash(original), pin); }
  }
  for (const [path, , size, pin] of gpuSources) { assert.equal(decoded.get(path).length, size); assert.equal(hash(decoded.get(path)), pin); }
  const receipt = JSON.parse(decoded.get("control/results/receipt.json"));
  for (const path of ["scripts/bench-gpu.py", "scripts/lib/model-profile.mjs", "js/src/source.js", "js/src/pack-layout.js"]) { const expected = receipt.sourcesBefore.find(file => file.path === `/home/bruno/githubworkspace/kevala/${path}`), actual = decoded.get(`dependencies/${path}`); assert.equal(actual.length, expected.bytes); assert.equal(hash(actual), expected.sha256); }
  return decoded;
}
async function checkFreshness(files) {
  for (const [path, repoPath] of freshness) assert.equal(hash(await checkedRead(join(repo, repoPath))), files.find(file => file.path === path)?.sha256, `publication is stale: ${repoPath}`);
}
if (mode === "--plan") {
  const result = await collect(false); console.log(JSON.stringify({ status: "metadata and small sources only; payloads unread; no output", files: result.files, references: result.references, exclusions: result.exclusions }, null, 2));
} else {
  const before = await lease();
  if (mode === "--verify") {
    const manifestBytes = await checkedRead(manifestPath), manifest = JSON.parse(manifestBytes), compressed = await checkedRead(archivePath, null, maxEnvelope);
    decode(compressed, manifest); await checkFreshness(manifest.files); const after = await lease(); assert.equal(after.pid, before.pid); assert.equal(after.inode, before.inode); assert.equal(after.device, before.device);
    console.log(JSON.stringify({ verified: true, files: manifest.files.length, archiveSha256: hash(compressed), manifestSha256: hash(manifestBytes), numericalChecksRerun: false }));
  } else {
    for (const path of [archivePath, manifestPath]) { try { await lstat(path); throw Error(`refusing existing output: ${path}`); } catch (error) { if (error.code !== "ENOENT") throw error; } }
    const result = await collect(true), bytes = Buffer.from(JSON.stringify({ schemaVersion: 1, files: result.files }) + "\n"); assert(bytes.length <= maxEnvelope);
    const compressed = gzipSync(bytes, { level: 9 }), files = result.files.map(({ data, encoding, ...descriptor }) => descriptor);
    const manifest = { schemaVersion: 1, recordedAt: new Date().toISOString(), verdict: "scoped uncached carry allocation improvement; exact source parity; legacy cross-mode failure retained", archive: { path: "evidence.json.gz", bytes: compressed.length, sha256: hash(compressed), decodedBytes: bytes.length, decodedSha256: hash(bytes) }, files, references: result.references, exclusions: result.exclusions, freshness };
    decode(compressed, manifest); await checkFreshness(files); const after = await lease(); assert.equal(after.pid, before.pid); assert.equal(after.inode, before.inode); assert.equal(after.device, before.device); manifest.lease = { before, after };
    await writeFile(archivePath, compressed, { flag: "wx" }); const manifestBytes = Buffer.from(JSON.stringify(manifest, null, 2) + "\n"); await writeFile(manifestPath, manifestBytes, { flag: "wx" });
    console.log(JSON.stringify({ archived: true, files: files.length, archiveSha256: hash(compressed), manifestSha256: hash(manifestBytes) }));
  }
}
