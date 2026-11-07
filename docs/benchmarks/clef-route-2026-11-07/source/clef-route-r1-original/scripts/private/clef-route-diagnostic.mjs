import assert from "node:assert/strict";
import { createHash, randomUUID } from "node:crypto";
import { constants as fsConstants, createReadStream } from "node:fs";
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { access, link, lstat, mkdir, open, readFile, readlink, realpath, readdir, rename, stat, unlink } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { createKevalaServer } from "../serve.mjs";
import { q8GlobalInventoryDue, q8KnownChildIdentitySample, q8KnownChildSample } from "./q8-monitor-cadence.mjs";
import { MODELS } from "../../js/src/source.js";

const ROOT = resolve(fileURLToPath(new URL("../../", import.meta.url)));
const PRIMARY = "/home/bruno/githubworkspace/kevala";
const CAMPAIGN = "/home/bruno/githubworkspace/kevala/tmp/sota-20261106-kdDTGn/campaign/clef-affinity-startup-v1-private-session-bus-r2";
const SESSION_BUS_HELPER = join(CAMPAIGN, "session_bus.py");
const SESSION_BUS_EVIDENCE = join(CAMPAIGN, "independent-audit-v1.json");
const PROFILE_LOCK = "/home/bruno/.cache/kevala/gpu-profile.lock";
const WASM_ARTIFACT_ROOT = join(PRIMARY, "js/src");
const CHROMIUM = {
  path: "/home/bruno/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome",
  version: "156.0.8078.4",
  sha256: "377a9fc07b495c3b8d0ab8ec1fdfd153641759f9756fea34d8dfce4de82f5025",
};
const LEASE_DIR = resolve(CAMPAIGN, "../lease-fdinfo-safety-v1/async-bind-r1");
const LEASE_HELPERS = [
  { path: join(LEASE_DIR, "fd-lease.mjs"), sha256: "cf3ad8822a1386ec435723bb478b6a2ef1d753a0b9896c23e1b27d6e6d4b2915" },
  { path: join(LEASE_DIR, "lease-parse.mjs"), sha256: "6efc5046dc7b769f4c59ea256661581087b6e3ee1a366898b7852192697b5a31" },
];
const FLAVORS = {
  base: { file: "kevala-base.wasm", sha256: "793351ff820ca6818eae749c09c829f974fe718d805447bd2141ab99b4c1e831", bytes: 1375771 },
  simd: { file: "kevala-simd.wasm", sha256: "f463d88be692c2da5a609cade3216ff94cb05732ab560d0df72517afe9db67f5", bytes: 1382096 },
  relaxed: { file: "kevala-relaxed.wasm", sha256: "55aa0e56bce7a8f808d27c8db50767fb24102eb85d8d45bb369451c934836c0a", bytes: 1383425 },
};
const TRACKED_RUNTIME_JS_PATHS = [
  "js/src/archs/clef.js", "js/src/archs/gemma4.js", "js/src/archs/index.js", "js/src/archs/kev.js", "js/src/archs/laya.js",
  "js/src/cpu-calibrate.js", "js/src/cpu-policy.js", "js/src/cpu-tune.js", "js/src/engine-worker.js",
  "js/src/gemma4-image.js", "js/src/gemma4-ple-store.js", "js/src/gemma4-vision-pack.js",
  "js/src/gpu-clef-head.js", "js/src/gpu-gemma4-attention.js", "js/src/gpu-gemma4-batch-mlp.js", "js/src/gpu-gemma4-batch-single.js",
  "js/src/gpu-gemma4-bn32.js", "js/src/gpu-gemma4-grid-mlp.js", "js/src/gpu-gemma4-image.js", "js/src/gpu-gemma4-k64.js",
  "js/src/gpu-gemma4-mlp.js", "js/src/gpu-gemma4-ple.js", "js/src/gpu-gemma4-prefix-cache.js", "js/src/gpu-gemma4-prefix-gelu.js",
  "js/src/gpu-gemma4-prefix-retile.js", "js/src/gpu-gemma4-rms-plan.js", "js/src/gpu-gemma4-rms.js", "js/src/gpu-gemma4-vision-qkv.js",
  "js/src/gpu-gemma4-vision.js", "js/src/gpu-gemma4.js", "js/src/gpu-kev.js", "js/src/gpu-laya.js",
  "js/src/gpu-matvec-gelu.js", "js/src/gpu-matvec-two.js", "js/src/gpu-matvec.js", "js/src/gpu-recur-tuning.js",
  "js/src/gpu-tuning.js", "js/src/gpu.js", "js/src/image-header.js", "js/src/images.js", "js/src/index.js",
  "js/src/node.js", "js/src/pack-layout.js", "js/src/presets.js", "js/src/request-queue.js", "js/src/shard-client.js",
  "js/src/shard-worker.js", "js/src/source.js", "js/src/stats.js", "js/src/wasm.js",
];
const SOURCE_PATHS = [...new Set([
  "bench.html", "bench-clef-route.html", "bench-clef-route-smoke.html", "dev/adapter-metadata.js", ...TRACKED_RUNTIME_JS_PATHS,
  "scripts/private/clef-route-browser.py", "scripts/private/clef-route-diagnostic.mjs", "scripts/private/q8-monitor-cadence.mjs",
  "scripts/serve.mjs", "scripts/lib/local-packs.mjs", "scripts/lib/static-path.mjs",
  "app/model-policy.js",
])];
const PACK_PATH = join(PRIMARY, "tmp/clef-flash-q8.kevala");
const PACK_LINK = join(ROOT, "tmp/clef-flash-q8.kevala");
const OUTPUT_ROOT = join(ROOT, "tmp");
const DIAGNOSTIC_LOG_BUDGET = 4 * 1024 * 1024;
const CHILD_DEADLINE_MS = 930000;
const CHILD_TERM_GRACE_MS = 1000;
const CHILD_CLEANUP_MAX_MS = 6000;
const ACTIVE_SCOPE_AND_KNOWN_PID_INTERVAL_MS = 1000;
const ACTIVE_FULL_GROUP_DISCOVERY_INTERVAL_MS = 5000;
const CLEANUP_FULL_GROUP_DISCOVERY_MIN_DELAY_MS = 100;
const hex = (bytes) => createHash("sha256").update(bytes).digest("hex");
function sourceIdentity() {
  return Object.fromEntries(SOURCE_PATHS.map((path) => {
    const bytes = readFileSync(join(ROOT, path));
    return [path, { bytes: bytes.byteLength, sha256: hex(bytes) }];
  }));
}
const PLAN = {
  schemaVersion: 1,
  purpose: "One resident Clef Flash Q8 request-profile capture that records post-load and per-request head route diagnostics plus optional call/wave timing attribution.",
  baseCommit: "5becbb0d35208ba6b0a2ff04c0535a52a8963d5a",
  productionPromotion: false,
  model: { id: "clef-flash", precision: "q8", catalogBytes: MODELS["clef-flash"].precisionPackBytes.q8, packPath: PACK_PATH,
    serving: "Pass the stat-verified existing file as a same-origin packUrl. The catalog marks its Q8 size approximate, so local-packs intentionally does not advertise it through pack=local discovery." },
  request: { backend: "webgpu", onPage: true, cache: false, stateCache: false, profile: false, protocol: "isolated", runs: 9, warmups: 3, unique: true, shapeIndices: [0, 1, 2], requestedHead: "auto" },
  routeSelection: "The URL omits flavor, GPU kernel, and clefHead overrides. Runtime defaults select WebGPU, WASM flavor, and clefHead auto; record the actual post-load and after-finish snapshots without forcing either GPU or CPU head.",
    trace: {
    routeSnapshots: "The private page clone preserves bench.html's after-load snapshot and per-measured-request finish snapshots, including request index, requested, selected, validated, reason, flavor, coordTile, and headDiagnosticsTile.",
    callTiming: "For each measured request, performance.now wraps private ClefHead.call and aggregates count, total, maximum, and failures by Rust export name. This includes JavaScript wrapper work and synchronous Rust export work.",
    waveTiming: "For each measured request, performance.now wraps async ClefHead.wave and aggregates count, total, maximum, and failures by phase. This asynchronous wall includes encoding, copies/readback, waits, and scheduling.",
    cpuFallbackTiming: "For each measured request, performance.now wraps ClefHead.cpu and records count, total, maximum, and failures for the whole-head CPU fallback. It includes its nested Rust export call.",
    overlap: "Rust calls such as status can occur inside wave; CPU fallback wraps its nested Rust export. Call, wave, and CPU fallback durations overlap and must not be summed as disjoint latency components.",
    limit: "No arguments, tensors, pointers, or model data are retained. Call timings cover wrapper plus synchronous Rust export work; wave timings cover async wall including encoding, copies/readback, waits, and scheduling. These sets overlap, do not form a partition, and omit surrounding head.finish and Clef architecture copies. Each measured request has a bounded 64-event accumulator with droppedEvents required to be zero. Instrumented wall samples include wrapper overhead and are diagnostic attribution, not a clean latency comparison.",
  },
  staticAssets: {
    entry: "bench-clef-route.html",
    smokeEntry: "bench-clef-route-smoke.html",
    basePage: "bench.html",
    runtimeEntry: "js/src/index.js",
    workerEntry: "js/src/engine-worker.js",
    trackedRuntimeModules: TRACKED_RUNTIME_JS_PATHS,
    requiredDevFiles: ["dev/adapter-metadata.js"],
    wasmCandidates: Object.fromEntries(Object.entries(FLAVORS).map(([flavor, value]) => [flavor,
      { source: join(WASM_ARTIFACT_ROOT, value.file), servedPath: join(ROOT, "js/src", value.file), bytes: value.bytes, sha256: value.sha256 }])),
    bootstrap: "The private profile and smoke pages use the same classic terminal-error bootstrap as current main bench.html. The smoke entry imports adapter metadata, index.js, and engine-worker.js without calling Kevala.load; it rejects wasm/kevala resource requests and creates no GPU device. The profile page imports index.js before its single Kevala.load call. All pages and the full tracked js/src JavaScript inventory are source-pinned.",
    sourcePaths: SOURCE_PATHS,
    identity: "The metadata-only plan records SHA-256 and byte size for every source path. The Q8 pack is checked by catalog size and filesystem identity without hashing its bytes.",
    wasm: "Runtime-default WASM flavor is not forced. The three CI80/83 WASM artifacts are SHA-256 checked only at execute time, then hardlinked into the sparse worktree for same-origin serving; the page records the actual selected flavor. Metadata-only --plan does not read them.",
  },
  launch: {
    entrypoint: ["env", "-u", "VK_DRIVER_FILES", "-u", "KEVALA_PROFILE_LOCK_HELD", "-u", "KEVALA_BENCH_CDP", "KEVALA_CLEF_ROUTE_DIAGNOSTIC_RUN_AUTHORIZED=1", "systemd-run", "--user", "--scope", "--collect", "--unit=<unique-scope>.scope", "--property=MemoryHigh=34359738368", "--property=MemoryMax=42949672960", "--property=MemorySwapMax=4294967296", "timeout", "--signal=TERM", "--kill-after=15s", "960s", "flock", "--exclusive", "--nonblock", "--conflict-exit-code", "75", "--no-fork", PROFILE_LOCK, process.execPath, resolve(fileURLToPath(import.meta.url)), "--execute", "--scope-unit", "<unique-scope>.scope", "--output", "<absolute-worktree-tmp-capture>", "--binary", CHROMIUM.path],
    lease: "The outer flock --no-fork execs this Node process with the canonical descriptor inherited. This controller verifies its own PID and exact descriptor with the pinned fdinfo guard; no marker or inner lock is accepted.",
    leaseHelpers: LEASE_HELPERS,
    timeoutOwner: "systemd-run owns the fresh scope; timeout bounds Node; Node retains the inherited lock through child cleanup, server closure, link cleanup, and final lease proof",
    chromium: CHROMIUM,
    chromiumArgs: ["--enable-gpu", "--enable-unsafe-webgpu", "--use-angle=vulkan", "--enable-features=Vulkan", "--disable-vulkan-surface"],
    cpu: "normal unpinned scheduling; do not set CPU affinity or CPU limits",
  },
  containment: {
    browser: "fresh Chromium 156 session, no existing attachment",
    sessionBus: "reviewed absent-endpoint helper passed only through ChromeService.env",
    sessionBusHelper: { path: SESSION_BUS_HELPER, sha256: "bac20966653dfaf8cbd7df636c03d02e77ee80b4ac1e9ac9931a6a30bb40c716" },
    priorEndpointCleanupEvidence: { path: SESSION_BUS_EVIDENCE, sha256: "d67d3da3f6e92f5b5e185bfb37424606b5b9707fd8f65f3812eacd1e263b04cf" },
    diagnostics: { directory: "unique capture-local diagnostics directory", files: ["chromedriver.log", "chrome-gpu.log"], budgetBytes: 32 * 1024 * 1024, sampleIntervalMs: 250, levels: { chromeDriver: "WARNING", chromeGpu: "WARNING" }, overshoot: "stop browser on the next size sample and retain both logs plus overshoot metadata" },
    monitor: {
      scopeAndKnownPidIntervalMs: ACTIVE_SCOPE_AND_KNOWN_PID_INTERVAL_MS,
      fullProcessGroupDiscoveryIntervalMs: ACTIVE_FULL_GROUP_DISCOVERY_INTERVAL_MS,
      firstFullDiscovery: "immediately after owned browser-runner identity admission",
      fastSample: "enumerate exact scope members, then individually check previously observed child PID/startTicks/cgroup identities; stop and retain expected/observed cgroup evidence for a same-identity PID outside the scope",
      fullDiscovery: "enumerate /proc and discover processes in the owned browser-runner process group/session",
      cadence: "1,000 ms active scope/known-PID target and 5,000 ms full-discovery target; record attempt/completion counts, start gaps, and durations",
      cleanup: "full process discovery and previously observed stable PID/startTicks identities remain in cleanup accounting across cgroups; signals still require existing exact scope or owned-session gates; unresolved known survivors keep cleanup failing and the lease held",
      assurance: "periodic sampled observation only; the 5-second process-group discovery interval is not continuous containment",
    },
    memory: "32 GiB high, 40 GiB max, 4 GiB swap",
    cpu: "normal unpinned scheduling",
    cgroups: "sample process and thread IDs, start ticks, and exact cgroups at create/navigation/result/quit bookends; after cleanup, only initial Node and timeout members remain",
    assurance: "sampled only, not continuous containment or CPU exclusivity",
    session: "one fresh desktop-style Chromium session for this diagnostic capture",
  },
};

function planOnly() {
  process.stdout.write(`${JSON.stringify({ ...PLAN, sourceIdentity: sourceIdentity() }, null, 2)}\n`);
  process.stdout.write("No build, browser, model, or GPU work is started by default.\n");
}

async function verifyRuntimeFiles() {
  for (const [flavor, expected] of Object.entries(FLAVORS)) {
    const path = join(WASM_ARTIFACT_ROOT, expected.file), bytes = await readFile(path);
    assert.equal(bytes.byteLength, expected.bytes, `pinned ${flavor} WASM size changed`);
    assert.equal(hex(bytes), expected.sha256, `pinned ${flavor} WASM hash changed`);
  }
  const helperBytes = await readFile(SESSION_BUS_HELPER);
  assert.equal(hex(helperBytes), PLAN.containment.sessionBusHelper.sha256, "reviewed private session-bus helper changed");
  assert.equal(hex(await readFile(SESSION_BUS_EVIDENCE)), PLAN.containment.priorEndpointCleanupEvidence.sha256, "reviewed session-bus endpoint/cleanup evidence changed");
}

async function verifyExternalPins() {
  for (const helper of [ ...LEASE_HELPERS,
    { path: SESSION_BUS_HELPER, sha256: PLAN.containment.sessionBusHelper.sha256 },
    { path: SESSION_BUS_EVIDENCE, sha256: PLAN.containment.priorEndpointCleanupEvidence.sha256 },
  ]) assert.equal(hex(await readFile(helper.path)), helper.sha256, `reviewed external source changed: ${helper.path}`);
}

async function chromiumStat(path) {
  const actualPath = await realpath(path);
  assert.equal(actualPath, CHROMIUM.path, "Chromium executable resolved to an unexpected path");
  const info = await stat(actualPath, { bigint: true });
  assert(info.isFile(), "pinned Chromium executable is not a regular file");
  await access(actualPath, fsConstants.X_OK);
  return { path: actualPath, dev: String(info.dev), ino: String(info.ino), bytes: String(info.size), mtimeNs: String(info.mtimeNs), ctimeNs: String(info.ctimeNs) };
}

async function verifyChromiumBinary(path) {
  assert.equal(resolve(path), CHROMIUM.path, "use the explicitly pinned Chromium 156 executable");
  const identity = await chromiumStat(path);
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  const sha256 = digest.digest("hex");
  assert.equal(sha256, CHROMIUM.sha256, "pinned Chromium 156 executable hash changed");
  assert.deepEqual(await chromiumStat(path), identity, "pinned Chromium executable changed while hashing");
  return { ...identity, version: CHROMIUM.version, sha256 };
}

function benchmarkUrl(base) {
  const query = new URLSearchParams({ auto: "", backend: "webgpu", pack: "local", packUrl: `${base}/tmp/clef-flash-q8.kevala`, model: "clef-flash", precision: "q8", onPage: "1", cache: "0", stateCache: "0", profile: "0", protocol: "isolated", runs: "9", warmups: "3", unique: "1", shapes: "0,1,2" });
  return `${base}/bench-clef-route.html#${query}`;
}

function argumentsForRun() {
  const { values, positionals } = parseArgs({
    options: { execute: { type: "boolean" }, output: { type: "string" }, "scope-unit": { type: "string" }, binary: { type: "string" } },
    strict: true,
    allowPositionals: false,
  });
  assert.equal(positionals.length, 0);
  assert.equal(values.execute, true, "use --execute under the external systemd-run/timeout/flock envelope");
  assert(values.output && values["scope-unit"], "--execute requires --output and --scope-unit");
  return values;
}

function parseStat(pid, raw) {
  const closeParen = raw.lastIndexOf(")");
  assert(closeParen > 0, `malformed /proc/${pid}/stat`);
  const fields = raw.slice(closeParen + 1).trim().split(/\s+/);
  assert(fields.length > 19, `short /proc/${pid}/stat`);
  assert(/^\d+$/.test(fields[19]), `invalid start tick for PID ${pid}`);
  return { pid, state: fields[0], ppid: Number(fields[1]), pgrp: Number(fields[2]), session: Number(fields[3]), startTicks: fields[19] };
}

async function processInfo(pid) {
  try {
    const raw = await readFile(`/proc/${pid}/stat`, "utf8");
    const cgroups = await readFile(`/proc/${pid}/cgroup`, "utf8");
    const unified = cgroups.split("\n").find((line) => line.startsWith("0::"));
    assert(unified, `PID ${pid} has no unified cgroup identity`);
    return { ...parseStat(pid, raw), cgroup: unified };
  } catch (error) {
    if (error.code === "ENOENT" || error.code === "ESRCH") return null;
    throw error;
  }
}

async function readScopeMembers(scope) {
  const text = await readFile(join(scope.directory, "cgroup.procs"), "utf8");
  const pids = text.trim() ? text.trim().split(/\s+/).map(Number) : [];
  assert(pids.length <= 256, "controller scope process inventory exceeded its fixed bound");
  const rows = [];
  for (const pid of [...new Set(pids)].sort((a, b) => a - b)) {
    const row = await processInfo(pid);
    if (!row) continue;
    assert.equal(row.cgroup, scope.cgroup, `scope member PID ${pid} escaped its exact cgroup`);
    rows.push(row);
  }
  return rows;
}

async function scopeAdmission(unit, outputPath) {
  assert(/^kevala-clef-route-[a-zA-Z0-9-]+\.scope$/.test(unit), "a unique Clef route systemd scope is required");
  const cgroups = await readFile("/proc/self/cgroup", "utf8");
  const match = /^0::(.+)$/m.exec(cgroups);
  assert(match, "unified cgroup metadata is unavailable");
  const cgroup = `0::${match[1]}`;
  assert(match[1].endsWith(`/${unit}`), "Node controller is outside the supplied scope unit");
  const directory = resolve("/sys/fs/cgroup", match[1].replace(/^\/+/, ""));
  const expected = { "memory.high": "34359738368", "memory.max": "42949672960", "memory.swap.max": "4294967296" };
  for (const [name, value] of Object.entries(expected)) assert.equal((await readFile(join(directory, name), "utf8")).trim(), value, `scope ${name} differs from the fixed 32/40/4 GiB protocol`);
  const owner = await processInfo(process.pid);
  assert(owner && owner.cgroup === cgroup);
  const timeoutParent = await processInfo(process.ppid);
  assert(timeoutParent && timeoutParent.cgroup === cgroup, "timeout parent is outside the admitted scope");
  assert((await readlink(`/proc/${process.ppid}/exe`)).endsWith("/timeout"), "Node must be the child of the external timeout command");
  const parentArgs = (await readFile(`/proc/${process.ppid}/cmdline`, "utf8")).split("\0").filter(Boolean);
  assert.equal(parentArgs[1], "--signal=TERM");
  assert.equal(parentArgs[2], "--kill-after=15s");
  assert.equal(parentArgs[3], "960s");
  const flockIndex = parentArgs.findIndex((arg) => arg === "flock" || arg.endsWith("/flock"));
  assert(flockIndex >= 0, "external timeout must invoke flock directly");
  const flockArgs = parentArgs.slice(flockIndex);
  for (const flag of ["--exclusive", "--nonblock", "--no-fork", PROFILE_LOCK, process.execPath, resolve(process.argv[1]), "--execute", "--scope-unit", unit, "--output", outputPath]) {
    assert(flockArgs.includes(flag), `external lock envelope is missing ${flag}`);
  }
  const scope = { unit, cgroup, directory, expectedLimits: expected, owner, timeoutParent: { ...timeoutParent, args: parentArgs } };
  const members = await readScopeMembers(scope);
  const expectedPids = [process.pid, process.ppid].sort((a, b) => a - b);
  assert.deepEqual(members.map((row) => row.pid), expectedPids, "scope must initially contain only Node and its timeout parent");
  scope.initialMembers = members;
  return scope;
}

async function syncDirectory(path) {
  const handle = await open(path, "r");
  try { await handle.sync(); } finally { await handle.close(); }
}

async function durableWrite(path, value) {
  const temporary = `${path}.${process.pid}.tmp`;
  const handle = await open(temporary, "wx", 0o600);
  try { await handle.writeFile(JSON.stringify(value, null, 2) + "\n"); await handle.sync(); }
  finally { await handle.close(); }
  await rename(temporary, path);
  await syncDirectory(dirname(path));
}

async function createLeaseGuard(output) {
  for (const helper of LEASE_HELPERS) assert.equal(hex(await readFile(helper.path)), helper.sha256, `reviewed lease guard changed: ${helper.path}`);
  const { createLeaseGuard: createGuard } = await import(pathToFileURL(LEASE_HELPERS[0].path).href);
  return createGuard({ canonicalPath: PROFILE_LOCK, recordEvidence: async (evidence) => {
    const name = `lease-${String(evidence.sequence).padStart(3, "0")}-${evidence.label}.json`;
    const path = join(output, name), bytes = JSON.stringify(evidence, null, 2) + "\n", handle = await open(path, "wx", 0o600);
    try { await handle.writeFile(bytes); await handle.sync(); } finally { await handle.close(); }
    await syncDirectory(output);
    return { path, sha256: hex(Buffer.from(bytes)) };
  } });
}

function sameIdentity(a, b) { return a.pid === b.pid && a.startTicks === b.startTicks && a.cgroup === b.cgroup; }

async function allProcessRows() {
  const pids = await readdir("/proc");
  assert(pids.length <= 32768, "process inventory exceeded its bound");
  const rows = [];
  for (const name of pids) {
    if (!/^\d+$/.test(name)) continue;
    const row = await processInfo(Number(name));
    if (row) rows.push(row);
  }
  return rows;
}

async function observeOwnedChildren(scope, initialMembers, observed, childRoot, { discoverGroup = true, checkKnown = true, enforceKnownCgroup = true } = {}) {
  const scopeRows = await readScopeMembers(scope);
  const owned = new Map();
  for (const row of scopeRows) {
    if (initialMembers.some((initial) => sameIdentity(initial, row))) continue;
    const key = `${row.pid}:${row.startTicks}`;
    if (!observed.has(key)) assert(observed.size < 1024, "owned process identity inventory exceeded its fixed bound");
    observed.set(key, row);
    owned.set(key, row);
  }
  if (checkKnown) {
    for (const [key, known] of observed) {
      if (owned.has(key)) continue;
      const current = await processInfo(known.pid);
      const checked = enforceKnownCgroup
        ? q8KnownChildSample(known, current, scope.cgroup)
        : q8KnownChildIdentitySample(known, current);
      if (!checked) continue;
      observed.set(key, checked);
      owned.set(key, checked);
    }
  }
  if (childRoot && discoverGroup) {
    const groupRows = (await allProcessRows()).filter((row) => row.pgrp === childRoot.pgrp && row.session === childRoot.session);
    for (const row of groupRows) {
      const key = `${row.pid}:${row.startTicks}`;
      if (!observed.has(key)) assert(observed.size < 1024, "owned process identity inventory exceeded its fixed bound");
      observed.set(key, row);
      owned.set(key, row);
    }
  }
  return { scopeRows, ownedRows: [...owned.values()], active: [...owned.values()].filter((row) => row.state !== "Z") };
}

async function cleanupOwnedChildren(scope, state, record) {
  if (!scope) return { checked: false, initialMembers: [], ownedChildren: [], remaining: [] };
  const childPid = state.child?.pid || null;
  const startedAt = Date.now(), termDeadline = startedAt + CHILD_TERM_GRACE_MS, deadline = startedAt + CHILD_CLEANUP_MAX_MS;
  const signals = [];
  const signalKeys = new Set();
  const retainSignal = (entry) => {
    const key = `${entry.signal}:${entry.pgid || entry.pid}:${entry.startTicks || entry.members?.[0]?.startTicks || ""}`;
    if (signalKeys.has(key)) return;
    assert(signals.length < 2048, "cleanup signal record exceeded its fixed bound");
    signalKeys.add(key);
    signals.push(entry);
  };
  let snapshot = null;
  while (Date.now() < deadline) {
    snapshot = await observeOwnedChildren(scope, scope.initialMembers, state.observedChildren, state.childRoot,
      { discoverGroup: true, checkKnown: true, enforceKnownCgroup: false });
    record.cleanupSamples = (record.cleanupSamples || 0) + 1;
    if (!snapshot.active.some((row) => row.state !== "Z")) break;
    const signal = Date.now() < termDeadline ? "SIGTERM" : "SIGKILL";
    const processRows = await allProcessRows();
    const groups = new Map();
    for (const row of snapshot.active) (groups.get(row.pgrp) || groups.set(row.pgrp, []).get(row.pgrp)).push(row);
    for (const [pgid, groupRows] of groups) {
      const globalGroup = processRows.filter((row) => row.pgrp === pgid);
      const groupIsOwned = globalGroup.length > 0 && globalGroup.every((row) => row.cgroup === scope.cgroup &&
        groupRows.some((owned) => sameIdentity(owned, row)));
      if (groupIsOwned) {
        try { process.kill(-pgid, signal); retainSignal({ pgid, signal, members: globalGroup.map(({ pid, startTicks }) => ({ pid, startTicks })) }); }
        catch (error) { if (error.code !== "ESRCH") throw error; }
      } else {
        for (const row of groupRows) {
          const current = await processInfo(row.pid);
          const inScope = current?.cgroup === scope.cgroup;
          const inOwnedSession = state.childRoot && current?.pgrp === state.childRoot.pgrp && current?.session === state.childRoot.session;
          if (!current || !sameIdentity(current, row) || (!inScope && !inOwnedSession)) continue;
          try { process.kill(row.pid, signal); retainSignal({ pid: row.pid, startTicks: row.startTicks, signal }); }
          catch (error) { if (error.code !== "ESRCH") throw error; }
        }
      }
    }
    await new Promise((resolveDelay) => setTimeout(resolveDelay, CLEANUP_FULL_GROUP_DISCOVERY_MIN_DELAY_MS));
  }
  snapshot = await observeOwnedChildren(scope, scope.initialMembers, state.observedChildren, state.childRoot,
    { discoverGroup: true, checkKnown: true, enforceKnownCgroup: false });
  if (state.childClose && !state.childClosed && Date.now() < deadline) {
    let closeTimer;
    const closed = await Promise.race([
      state.childClose.then((result) => ({ complete: true, result })),
      new Promise((resolveClose) => { closeTimer = setTimeout(() => resolveClose({ complete: false }), Math.max(0, deadline - Date.now())); }),
    ]);
    if (closeTimer) clearTimeout(closeTimer);
    if (!closed.complete) throw new Error("owned browser runner stdio did not close before cleanup deadline");
  }
  if (state.childClose && !state.childClosed) throw new Error("owned browser runner stdio did not close before cleanup deadline");
  if (state.childCloseResult) record.runnerExit = state.childCloseResult;
  await state.logChain?.catch((error) => { state.logError ||= String(error); });
  if (state.logFile) {
    try { await state.logFile.sync(); } catch (error) { state.logError ||= String(error); }
    try { await state.logFile.close(); } catch (error) { state.logError ||= String(error); }
    state.logFile = null;
    if (state.logPath) {
      try { record.runnerLog.sha256 = hex(await readFile(state.logPath)); }
      catch (error) { state.logError ||= String(error); }
    }
  }
  if (record.runnerLog) record.runnerLog.overshootBytes = Math.max(0, record.runnerLog.observedBytes - record.runnerLog.budgetBytes);
  const after = await readScopeMembers(scope);
  const remaining = after.filter((row) => !scope.initialMembers.some((initial) => sameIdentity(initial, row)) && row.state !== "Z");
  const remainingGroup = snapshot.active.filter((row) => row.state !== "Z");
  record.cleanup = { checked: true, childProcessGroup: childPid, scopeMembershipOwner: "fresh isolated scope; process identity checked by PID/startTicks/cgroup, plus detached child process group/session",
    observedOwnedChildren: [...state.observedChildren.values()], signals, remainingLiveScopeChildren: remaining, remainingLiveProcessGroup: remainingGroup,
    runnerStdioClosed: !state.childClose || state.childClosed === true, runnerLogError: state.logError || state.streamError || null };
  assert.equal(remaining.length, 0, "owned scope still contains a live child process");
  assert.equal(remainingGroup.length, 0, "owned browser process group still contains a live process");
  return record.cleanup;
}

async function ensurePackLink(record, ownedLinks) {
  const spec = MODELS["clef-flash"];
  const expectedBytes = spec.precisionPackBytes?.q8;
  assert(Number.isSafeInteger(expectedBytes) && expectedBytes > 0, "Clef Flash Q8 catalog size is unavailable");
  const source = await realpath(PACK_PATH), before = await stat(source, { bigint: true });
  assert(before.isFile() && Number(before.size) === expectedBytes, "local Clef Flash Q8 pack size differs from the catalog");
  const existing = await lstat(PACK_LINK).catch((error) => { if (error.code === "ENOENT") return null; throw error; });
  assert.equal(existing, null, `refusing to replace existing local pack path ${PACK_LINK}`);
  await link(source, PACK_LINK);
  const owned = { path: PACK_LINK, identity: { dev: String(before.dev), ino: String(before.ino) } };
  ownedLinks.push(owned);
  const linked = await lstat(PACK_LINK, { bigint: true });
  const afterLink = await stat(source, { bigint: true });
  assert(linked.isFile() && String(linked.dev) === String(afterLink.dev) && String(linked.ino) === String(afterLink.ino) &&
    String(afterLink.dev) === String(before.dev) && String(afterLink.ino) === String(before.ino) && String(afterLink.size) === String(before.size) &&
    String(afterLink.mtimeNs) === String(before.mtimeNs), "temporary Clef Flash Q8 pack hardlink identity differs from its source");
  const sourceIdentity = { path: source, dev: String(afterLink.dev), ino: String(afterLink.ino), size: String(afterLink.size),
    mtimeNs: String(afterLink.mtimeNs), ctimeNs: String(afterLink.ctimeNs), nlink: String(afterLink.nlink) };
  owned.identity = sourceIdentity;
  record.pack = sourceIdentity;
  return sourceIdentity;
}

async function ensureWasmLinks(record, ownedLinks) {
  record.wasmArtifacts = {};
  for (const [flavor, expected] of Object.entries(FLAVORS)) {
    const source = join(WASM_ARTIFACT_ROOT, expected.file), servedPath = join(ROOT, "js/src", expected.file);
    const sourcePath = await realpath(source), before = await lstat(sourcePath, { bigint: true });
    assert.equal(sourcePath, source, `pinned ${flavor} WASM source resolved through a link`);
    assert(before.isFile() && Number(before.size) === expected.bytes, `pinned ${flavor} WASM file metadata changed`);
    const existing = await lstat(servedPath).catch((error) => { if (error.code === "ENOENT") return null; throw error; });
    assert.equal(existing, null, `refusing to replace existing WASM path ${servedPath}`);
    await link(sourcePath, servedPath);
    // Record the created inode immediately so every later failure remains cleanable.
    const identity = { source: sourcePath, servedPath, dev: String(before.dev), ino: String(before.ino), bytes: String(before.size),
      mtimeNs: String(before.mtimeNs), ctimeNs: String(before.ctimeNs), sourceNlinkBefore: String(before.nlink),
      sourceNlinkAfterLink: null, sha256: expected.sha256 };
    record.wasmArtifacts[flavor] = identity;
    ownedLinks.push({ path: servedPath, identity, kind: "wasm" });
    const linked = await lstat(servedPath, { bigint: true }), after = await stat(sourcePath, { bigint: true });
    assert(linked.isFile() && String(linked.dev) === String(after.dev) && String(linked.ino) === String(after.ino) &&
      String(after.dev) === String(before.dev) && String(after.ino) === String(before.ino) && String(after.size) === String(before.size) &&
      String(after.mtimeNs) === String(before.mtimeNs), `temporary ${flavor} WASM hardlink identity differs from its source`);
    Object.assign(identity, { dev: String(after.dev), ino: String(after.ino), bytes: String(after.size),
      mtimeNs: String(after.mtimeNs), ctimeNs: String(after.ctimeNs), sourceNlinkAfterLink: String(after.nlink) });
  }
}

async function verifyWasmLinksRemoved(record) {
  for (const identity of Object.values(record.wasmArtifacts || {})) {
    const served = await lstat(identity.servedPath).catch((error) => { if (error.code === "ENOENT") return null; throw error; });
    assert.equal(served, null, `temporary WASM hardlink remained after cleanup: ${identity.servedPath}`);
    const source = await stat(identity.source, { bigint: true });
    assert.equal(String(source.dev), identity.dev, "pinned WASM source device changed");
    assert.equal(String(source.ino), identity.ino, "pinned WASM source inode changed");
    assert.equal(String(source.size), identity.bytes, "pinned WASM source size changed");
    assert.equal(String(source.mtimeNs), identity.mtimeNs, "pinned WASM source mtime changed");
    assert.equal(String(source.nlink), identity.sourceNlinkBefore, "pinned WASM source link count did not return after cleanup");
  }
}

async function removeOwnedLinks(ownedLinks) {
  const removed = [];
  for (const entry of [...ownedLinks].reverse()) {
    const current = await lstat(entry.path, { bigint: true }).catch((error) => { if (error.code === "ENOENT") return null; throw error; });
    if (!current) continue;
    assert(current.isFile() && String(current.dev) === entry.identity.dev && String(current.ino) === entry.identity.ino,
      `refusing to unlink changed owned path ${entry.path}`);
    await unlink(entry.path);
    removed.push(entry.path);
  }
  return removed;
}

async function readRunnerResult(output) {
  const result = JSON.parse(await readFile(join(output, "result.json"), "utf8"));
  if (typeof result.error === "string" && result.error) {
    const error = new Error(`Clef route page reported ${result.status || "error"}: ${result.error}`);
    const containment = result.runner?.containment;
    const snapshotFailure = containment?.snapshotFailure || containment?.scope?.snapshotFailures?.at(-1) || null;
    error.pageFailure = { status: result.status || null, phase: result.phase || null, error: result.error,
      timeoutDiagnostics: result.timeoutDiagnostics || null,
      pageFailureDiagnostics: result.pageFailureDiagnostics || null,
      snapshotFailure, snapshotFailureReference: containment?.snapshotFailureReference || null };
    throw error;
  }
  assert.equal(result.done, true, "Clef route page did not complete");
  assert.equal(result.arch, "clef");
  assert.equal(result.model, "clef-flash");
  assert.equal(result.backend, "webgpu");
  assert.equal(result.profile, false, "GPU profiler must remain disabled for the wall lane");
  assert.equal(result.runs, 9);
  assert.equal(result.warmups, 3);
  assert.equal(result.unique, true);
  assert.deepEqual(result.cache, { packEnabled: false, stateCache: false });
  const routeSnapshot = (row, label) => {
    assert(row && typeof row === "object", `${label} head route snapshot is missing`);
    assert.equal(row.requested, "auto", `${label} head request mode was overridden`);
    assert(["cpu", "gpu"].includes(row.selected), `${label} selected head route is missing`);
    assert.equal(typeof row.validated, "boolean", `${label} validation state is missing`);
    assert(row.reason === null || typeof row.reason === "string", `${label} head route reason is invalid`);
    assert(row.flavor === null || typeof row.flavor === "string", `${label} runtime flavor is invalid`);
    assert(Number.isSafeInteger(row.coordTile) || row.coordTile === null, `${label} coordinator tile is invalid`);
    assert(Number.isSafeInteger(row.headDiagnosticsTile) || row.headDiagnosticsTile === null, `${label} head diagnostics tile is invalid`);
  };
  routeSnapshot(result.clefHeadAfterWarmup, "post-load");
  assert.equal(result.clefHeadInstrumentation,
    "per-request performance.now around private ClefHead.call Rust exports, async ClefHead.wave, and ClefHead.cpu fallback; includes wrapper overhead and is diagnostic, not a clean latency lane");
  assert.equal(result.runner?.browserVersion, CHROMIUM.version, `expected Chromium ${CHROMIUM.version}, got ${result.runner?.browserVersion}`);
  assert.equal(typeof result.selectedFlavor, "string", "runtime-selected WASM flavor was not reported");
  assert.equal(result.flavor, result.selectedFlavor, "top-level runtime flavor differs from the selected flavor");
  assert.equal(result.cpuTiles, null, "WebGPU run must not report CPU-tile tuning");
  assert.equal(result.cases?.length, 3, "expected exactly the three requested Clef shapes");
  const expectedNames = ["1 question, ~40 tokens", "1 question, ~128 tokens", "1 question, ~512 tokens"];
  for (const [caseIndex, sampleCase] of result.cases.entries()) {
    assert.equal(sampleCase.name, expectedNames[caseIndex], `unexpected Clef shape at index ${caseIndex}`);
    assert.equal(sampleCase.samples?.length, 9, `Clef case ${caseIndex} did not retain 9 measured samples`);
    for (const [sampleIndex, sample] of sampleCase.samples.entries()) {
      assert(Number.isFinite(sample.wallMs) && sample.wallMs > 0, `Clef case ${caseIndex} sample ${sampleIndex} has invalid wall time`);
      assert.match(sample.requestHash, /^[0-9a-f]{64}$/, `Clef case ${caseIndex} sample ${sampleIndex} request identity is invalid`);
      assert.equal(sample.clefHeadRequests?.length, 1, `Clef case ${caseIndex} sample ${sampleIndex} lacks one associated head trace`);
      assert.equal(sample.clefHeadRequests[0].requestIndex, 0, `Clef case ${caseIndex} sample ${sampleIndex} trace order changed`);
      routeSnapshot(sample.clefHeadRequests[0], `case ${caseIndex} sample ${sampleIndex}`);
      const attribution = sample.clefHeadTiming;
      assert(attribution && attribution.eventLimit === 64 && Number.isInteger(attribution.eventCount) && attribution.eventCount >= 1 && attribution.eventCount <= 64,
        `Clef case ${caseIndex} sample ${sampleIndex} has incomplete head timing events`);
      assert.equal(attribution.overflow, false, `Clef case ${caseIndex} sample ${sampleIndex} exceeded the head event bound`);
      assert.equal(attribution.droppedEvents, 0, `Clef case ${caseIndex} sample ${sampleIndex} dropped head timing events`);
      let eventCount = 0;
      for (const groupName of ["rustCalls", "gpuWaves", "cpuFallbacks"]) {
        const group = attribution[groupName];
        assert(group && typeof group === "object", `Clef case ${caseIndex} sample ${sampleIndex} is missing ${groupName}`);
        for (const [name, entry] of Object.entries(group)) {
          assert(name.length > 0 && Number.isInteger(entry.count) && entry.count > 0 && Number.isFinite(entry.totalMs) &&
            Number.isFinite(entry.maxMs) && Number.isInteger(entry.errors), `invalid ${groupName}.${name} attribution`);
          eventCount += entry.count;
        }
      }
      assert.equal(eventCount, attribution.eventCount, `Clef case ${caseIndex} sample ${sampleIndex} has inconsistent timing-event counts`);
      const route = sample.clefHeadRequests[0].selected;
      if (route === "gpu") {
        for (const exportName of ["kevala_clef_head_begin", "kevala_clef_head_memory_ready", "kevala_clef_head_finish"])
          assert.equal(attribution.rustCalls[exportName]?.count, 1, `GPU route did not time exactly one ${exportName}`);
        assert.equal(attribution.gpuWaves.memory?.count, 1, "GPU route did not time one memory wave");
        assert.equal(attribution.gpuWaves.kv?.count, 1, "GPU route did not time one KV wave");
        assert.equal(attribution.cpuFallbacks.cpu?.count || 0, 0, "GPU route unexpectedly entered CPU fallback");
      } else {
        assert.equal(attribution.cpuFallbacks.cpu?.count, 1, "CPU route did not time its head.cpu fallback");
        assert(attribution.rustCalls.kevala_clef_finish?.count >= 1, "CPU route did not time the whole-head Rust export");
      }
    }
  }
  assert.equal(result.runner?.containment?.error, undefined);
  assert.equal(result.runner?.containment?.sessionBusError, undefined);
  assert.equal(result.runner?.containment?.newAppScopes?.length, 0);
  assert.deepEqual(result.runner?.containment?.snapshots?.map((row) => row.stage),
    ["before-browser-create", "after-browser-create", "before-navigation", "after-benchmark-result", "after-browser-quit"]);
  const bus = result.runner?.containment?.sessionBus;
  assert.equal(bus?.cleanup?.removed, true);
  assert(bus?.checks?.length > 0 && bus.checks.every((row) => row.status === "passed"), "private session bus checks did not all pass");
  return result;
}

async function signalOwnedChildGroup(child, signal, record, state) {
  if (!child?.pid || !state.scope) return;
  const group = (await allProcessRows()).filter((row) => row.pgrp === child.pid);
  if (!group.length) return;
  const leader = group.find((row) => row.pid === child.pid);
  const leaderMatches = Boolean(state.childRoot && leader && sameIdentity(leader, state.childRoot) && leader.session === state.childRoot.session);
  const safeGroup = leaderMatches && group.every((row) => row.cgroup === state.scope.cgroup && row.session === state.childRoot.session);
  if (safeGroup) {
    try { process.kill(-child.pid, signal); record.groupSignals = [...(record.groupSignals || []), { pgid: child.pid, signal, members: group.map(({ pid, startTicks }) => ({ pid, startTicks })) }]; }
    catch (error) { if (error.code !== "ESRCH") record.stopSignalError = String(error); }
    return;
  }
  record.groupSignalRefused = { pgid: child.pid, reason: "exact child leader/session identity is absent or the group crosses its admitted cgroup" };
  const known = new Map([...(state.observedChildren?.values() || []), ...(state.childRoot ? [state.childRoot] : [])].map((row) => [`${row.pid}:${row.startTicks}`, row]));
  for (const row of group) {
    const identityKnown = known.has(`${row.pid}:${row.startTicks}`);
    const inScope = row.cgroup === state.scope.cgroup;
    const inOwnedSession = Boolean(state.childRoot && row.session === state.childRoot.session && row.pgrp === state.childRoot.pgrp);
    if ((!inScope && !inOwnedSession) || !identityKnown) continue;
    try { process.kill(row.pid, signal); record.groupSignals = [...(record.groupSignals || []), { pid: row.pid, startTicks: row.startTicks, signal }]; }
    catch (error) { if (error.code !== "ESRCH") record.stopSignalError = String(error); }
  }
}

function requestStop(reason, child, record, state) {
  if (state.stopReason) return;
  state.stopReason = reason;
  record.stopReason = reason;
  if (child?.pid) {
    void signalOwnedChildGroup(child, "SIGTERM", record, state).catch((error) => { record.stopSignalError ||= String(error); });
    if (!state.killTimer) state.killTimer = setTimeout(() => {
      void signalOwnedChildGroup(child, "SIGKILL", record, state).catch((error) => { record.stopSignalError ||= String(error); });
    }, CHILD_TERM_GRACE_MS);
  }
}

async function runBrowserChild({ childArgs, output, diagnosticsDir, url, scope, record, state }) {
  const logPath = join(output, "runner.log"), logFile = await open(logPath, "wx", 0o600);
  const child = spawn("uv", childArgs, { cwd: ROOT, env: process.env, detached: true, stdio: ["ignore", "pipe", "pipe"] });
  state.child = child;
  state.logFile = logFile;
  state.logPath = logPath;
  state.logChain = Promise.resolve();
  record.runnerLog = { path: logPath, budgetBytes: DIAGNOSTIC_LOG_BUDGET, observedBytes: 0, retainedBytes: 0, sha256: null };
  const retain = (chunk) => {
    const bytes = Buffer.from(chunk);
    record.runnerLog.observedBytes += bytes.length;
    const remaining = DIAGNOSTIC_LOG_BUDGET - record.runnerLog.retainedBytes;
    if (remaining > 0) {
      const saved = bytes.subarray(0, remaining);
      record.runnerLog.retainedBytes += saved.length;
      state.logChain = state.logChain.then(() => logFile.writeFile(saved)).catch((error) => {
        state.logError ||= String(error);
        requestStop("runner-log-write-error", child, record, state);
      });
    }
    if (record.runnerLog.observedBytes > DIAGNOSTIC_LOG_BUDGET) {
      record.runnerLog.budgetExceeded = true;
      requestStop("runner-log-budget-exceeded", child, record, state);
    }
  };
  child.stdout.on("data", retain); child.stderr.on("data", retain);
  child.stdout.on("error", (error) => { state.streamError ||= String(error); requestStop("runner-stdout-error", child, record, state); });
  child.stderr.on("error", (error) => { state.streamError ||= String(error); requestStop("runner-stderr-error", child, record, state); });
  const spawned = new Promise((resolveSpawn, rejectSpawn) => {
    child.once("spawn", resolveSpawn);
    child.once("error", rejectSpawn);
  });
  state.childClosed = false;
  state.childClose = new Promise((resolveClose) => {
    child.once("close", (code, signal) => {
      state.childClosed = true;
      state.childCloseResult = { code, signal };
      resolveClose(state.childCloseResult);
    });
  });
  const deadline = setTimeout(() => requestStop("runner-deadline", child, record, state), CHILD_DEADLINE_MS);
  let monitor;
  try {
    await spawned;
    const root = await processInfo(child.pid);
    assert(root && root.cgroup === scope.cgroup && root.pgrp === child.pid && root.ppid === process.pid,
      "browser runner did not start as the expected owned process group in the controller scope");
    state.childRoot = root;
    state.observedChildren = new Map([[`${root.pid}:${root.startTicks}`, root]]);
    record.childRoot = root;
    record.childStart = Date.now();
    record.url = url;
    record.diagnosticsDirectory = diagnosticsDir;
    const monitorStartedAt = performance.now();
    let lastFullDiscoveryAt = null;
    let lastSampleAttemptAt = null;
    let lastFullDiscoveryAttemptAt = null;
    const monitorSample = async () => {
      if (state.monitorBusy || !child.pid) return;
      state.monitorBusy = true;
      const startedAt = performance.now();
      const globalDiscoveryDue = q8GlobalInventoryDue(startedAt, lastFullDiscoveryAt, ACTIVE_FULL_GROUP_DISCOVERY_INTERVAL_MS);
      const stats = record.monitor;
      stats.scopeAndKnownPidSampleAttempts++;
      stats.lastSampleStartElapsedMs = Math.round(startedAt - monitorStartedAt);
      if (lastSampleAttemptAt !== null) {
        stats.lastSampleGapMs = Math.round(startedAt - lastSampleAttemptAt);
        stats.maxSampleGapMs = Math.max(stats.maxSampleGapMs, stats.lastSampleGapMs);
      }
      lastSampleAttemptAt = startedAt;
      if (globalDiscoveryDue) {
        stats.fullGroupDiscoveryAttempts++;
        if (lastFullDiscoveryAttemptAt !== null) {
          stats.lastFullGroupDiscoveryGapMs = Math.round(startedAt - lastFullDiscoveryAttemptAt);
          stats.maxFullGroupDiscoveryGapMs = Math.max(stats.maxFullGroupDiscoveryGapMs, stats.lastFullGroupDiscoveryGapMs);
        }
        lastFullDiscoveryAttemptAt = startedAt;
      }
      try {
        await observeOwnedChildren(scope, scope.initialMembers, state.observedChildren, state.childRoot,
          { discoverGroup: globalDiscoveryDue, checkKnown: true });
        const completedAt = performance.now();
        stats.scopeAndKnownPidSamplesCompleted++;
        stats.lastSampleElapsedMs = Math.round(completedAt - monitorStartedAt);
        stats.lastSampleDurationMs = Math.round(completedAt - startedAt);
        stats.maxSampleDurationMs = Math.max(stats.maxSampleDurationMs, stats.lastSampleDurationMs);
        if (globalDiscoveryDue) {
          stats.fullGroupDiscoveriesCompleted++;
          stats.lastFullGroupDiscoveryElapsedMs = stats.lastSampleElapsedMs;
          lastFullDiscoveryAt = completedAt;
        }
      } catch (error) {
        stats.sampleFailures++;
        record.scopeMonitorError = String(error);
        if (error.scopeMonitorEvidence) record.scopeMonitorEvidence = error.scopeMonitorEvidence;
        requestStop("scope-monitor-error", child, record, state);
      } finally { state.monitorBusy = false; }
    };
    await monitorSample();
    if (!state.stopReason) monitor = setInterval(() => { void monitorSample(); }, ACTIVE_SCOPE_AND_KNOWN_PID_INTERVAL_MS);
    const exit = await state.childClose;
    clearTimeout(deadline);
    clearInterval(monitor);
    while (state.monitorBusy) await new Promise((resolveDelay) => setTimeout(resolveDelay, 25));
    await state.logChain;
    try { await logFile.sync(); } catch (error) { state.logError ||= String(error); }
    try { await logFile.close(); } catch (error) { state.logError ||= String(error); }
    state.logFile = null;
    try { record.runnerLog.sha256 = hex(await readFile(logPath)); } catch (error) { state.logError ||= String(error); }
    record.runnerExit = exit;
    const pageResult = await readRunnerResult(output).catch((error) => ({ status: "invalid-or-missing", error: String(error), pageFailure: error.pageFailure || null }));
    record.result = pageResult.status === "invalid-or-missing" ? pageResult : {
      status: pageResult.done === true ? "complete" : pageResult.status || "incomplete",
      arch: pageResult.arch,
      model: pageResult.model,
      backend: pageResult.backend,
      flavor: pageResult.flavor,
      afterWarmup: pageResult.clefHeadAfterWarmup,
      cases: pageResult.cases?.map((sampleCase) => ({ name: sampleCase.name, sampleCount: sampleCase.samples?.length,
        routes: sampleCase.samples?.reduce((counts, sample) => {
          const selected = sample.clefHeadRequests?.[0]?.selected || "missing";
          counts[selected] = (counts[selected] || 0) + 1;
          return counts;
        }, {}),
      })),
      containmentStages: pageResult.runner?.containment?.snapshots?.map((row) => row.stage),
    };
    record.runnerLog.overshootBytes = Math.max(0, record.runnerLog.observedBytes - record.runnerLog.budgetBytes);
    if (exit.code !== 0 || record.runnerLog.budgetExceeded || record.stopReason || record.result.status === "invalid-or-missing" || state.logError || state.streamError) {
      const pageCause = record.result.pageFailure?.error ? ` page=${record.result.pageFailure.status || "error"}: ${record.result.pageFailure.error}` : "";
      const logCause = state.logError || state.streamError ? ` log=${state.logError || state.streamError}` : "";
      throw new Error(`browser runner failed: exit=${exit.code} signal=${exit.signal} stop=${record.stopReason || "none"}${pageCause}${logCause}${exit.error ? ` error=${exit.error}` : ""}`);
    }
    return pageResult;
  } catch (error) {
    clearTimeout(deadline);
    clearInterval(monitor);
    requestStop("runner-error", child, record, state);
    while (state.monitorBusy) await new Promise((resolveDelay) => setTimeout(resolveDelay, 25));
    throw error;
  }
}

async function execute(values) {
  assert.equal(process.env.KEVALA_CLEF_ROUTE_DIAGNOSTIC_RUN_AUTHORIZED, "1", "the source-reviewed Clef route diagnostic grant is required");
  assert(!process.env.KEVALA_BENCH_CDP, "an existing Chrome attachment is forbidden");
  assert(!Object.hasOwn(process.env, "VK_DRIVER_FILES"), "VK_DRIVER_FILES must be unset for the recorded host driver");
  assert(!Object.hasOwn(process.env, "KEVALA_PROFILE_LOCK_HELD"), "a profile-lock environment marker is not accepted");
  const browser = values.binary;
  assert(browser, `pass --binary ${CHROMIUM.path}; automatic browser selection is forbidden`);
  assert.equal(resolve(browser), CHROMIUM.path, "the execution packet requires the pinned Chromium 156 path");
  const output = resolve(values.output), outputRoot = resolve(OUTPUT_ROOT);
  assert.equal(dirname(output), outputRoot, "capture output must be a unique direct child of this worktree's tmp directory");
  await mkdir(outputRoot, { recursive: true });
  const rootInfo = await lstat(outputRoot);
  assert(rootInfo.isDirectory() && !rootInfo.isSymbolicLink() && rootInfo.uid === process.getuid(), "capture tmp directory must be owned and not a symlink");
  await mkdir(output, { recursive: false, mode: 0o700 });
  const diagnosticsDir = join(output, "diagnostics");
  await mkdir(diagnosticsDir, { recursive: false, mode: 0o700 });
  const record = { status: "running", sourceIdentity: null, scope: null, lease: {}, sessionBus: null, server: null, pack: null, runnerExit: null,
    model: PLAN.model, request: PLAN.request, trace: PLAN.trace,
    browser: "Chromium 156", cpuPolicy: "normal unpinned scheduling", productionPromotion: false,
    monitor: { scopeAndKnownPidIntervalMs: ACTIVE_SCOPE_AND_KNOWN_PID_INTERVAL_MS, fullGroupDiscoveryIntervalMs: ACTIVE_FULL_GROUP_DISCOVERY_INTERVAL_MS,
      firstFullGroupDiscovery: "immediately after child identity admission", scopeAndKnownPidSampleAttempts: 0, scopeAndKnownPidSamplesCompleted: 0,
      fullGroupDiscoveryAttempts: 0, fullGroupDiscoveriesCompleted: 0, sampleFailures: 0, lastSampleElapsedMs: null,
      lastSampleStartElapsedMs: null, lastSampleGapMs: null, maxSampleGapMs: 0, lastSampleDurationMs: null, maxSampleDurationMs: 0,
      lastFullGroupDiscoveryElapsedMs: null, lastFullGroupDiscoveryGapMs: null, maxFullGroupDiscoveryGapMs: 0,
      cleanupFullGroupScanMinDelayMs: CLEANUP_FULL_GROUP_DISCOVERY_MIN_DELAY_MS, assurance: "periodic sampled active monitoring only; cleanup retains full process-group scans and exact identity/signal proofs" } };
  const ownedLinks = [], abort = new AbortController(), state = { stopReason: null, killTimer: null, child: null, childRoot: null, childClosed: false, observedChildren: new Map(), monitorBusy: false };
  let server, leaseGuard, scope, sourceBefore, failure = null, outputCreated = true, serverClosed = false, cleanupError = null, scopeAdmissionAttempted = false;
  const stop = (signal) => {
    record.status = "failed";
    record.stopSignal = signal;
    record.error ||= `Node controller received ${signal}`;
    abort.abort(new Error(`Node controller received ${signal}`));
    requestStop(signal, state.child, record, state);
  };
  const stopTerm = () => stop("SIGTERM"), stopInt = () => stop("SIGINT");
  process.on("SIGTERM", stopTerm); process.on("SIGINT", stopInt);
  const save = () => durableWrite(join(output, "controller.json"), record);
  try {
    leaseGuard = await createLeaseGuard(output);
    record.lease.bind = await leaseGuard.bind("clef-route-controller-bind-before-inputs");
    scopeAdmissionAttempted = true;
    record.scope = await scopeAdmission(values["scope-unit"], output);
    scope = record.scope;
    state.scope = scope;
    record.lease.beforeInputs = await leaseGuard.check("before-source-and-runtime-inputs");
    sourceBefore = sourceIdentity();
    record.sourceIdentity = sourceBefore;
    await verifyExternalPins();
    record.browserExecutable = await verifyChromiumBinary(browser);
    await save();
    abort.signal.throwIfAborted();
    record.lease.beforeBrowserAssets = await leaseGuard.check("before-browser-load-of-runtime-wasm-and-model-pack");
    await verifyRuntimeFiles();
    abort.signal.throwIfAborted();
    await ensureWasmLinks(record, ownedLinks);
    abort.signal.throwIfAborted();
    record.lease.beforePackLink = await leaseGuard.check("before-pack-identity-and-link");
    const pack = await ensurePackLink(record, ownedLinks);
    abort.signal.throwIfAborted();
    const modelCatalog = { "clef-flash": MODELS["clef-flash"] };
    server = createKevalaServer({ root: ROOT, isolate: true, models: modelCatalog });
    await new Promise((resolveListen, rejectListen) => { server.once("error", rejectListen); server.listen(0, "127.0.0.1", resolveListen); });
    const address = server.address();
    assert(address && typeof address === "object" && address.port > 0, "owned static server did not bind an ephemeral port");
    const base = `http://127.0.0.1:${address.port}`;
    const url = benchmarkUrl(base);
    record.server = { host: "127.0.0.1", port: address.port, isolatedHeaders: true, root: ROOT, pack, owned: true };
    const diagnosticsDirectory = diagnosticsDir;
    const childArgs = ["run", "--offline", "scripts/private/clef-route-browser.py", "--result", "clef-route", "--backend", "webgpu", "--browser", "chromium", "--binary", CHROMIUM.path,
      "--timeout", "900", "--url", url, "--output", join(output, "result.json"), "--q8-swiglu-session-bus-helper", SESSION_BUS_HELPER,
      "--q8-swiglu-scope-unit", values["scope-unit"], "--chrome-diagnostics-dir", diagnosticsDirectory,
      "--chromium-arg=--enable-gpu", "--chromium-arg=--enable-unsafe-webgpu", "--chromium-arg=--use-angle=vulkan",
      "--chromium-arg=--enable-features=Vulkan", "--chromium-arg=--disable-vulkan-surface"];
    record.browserCommand = { executable: "uv", args: childArgs, output: join(output, "result.json"), diagnosticsDirectory,
      diagnosticsBudgetBytes: 32 * 1024 * 1024, runnerLogBudgetBytes: DIAGNOSTIC_LOG_BUDGET };
    record.lease.beforeBrowser = await leaseGuard.check("before-browser-controller");
    await save();
    abort.signal.throwIfAborted();
    await runBrowserChild({ childArgs, output, diagnosticsDir, url, scope, record, state });
    record.lease.afterBrowser = await leaseGuard.check("after-browser-controller");
    record.scopeAfterBrowser = await readScopeMembers(scope);
    record.lease.beforePostRunIdentityCheck = await leaseGuard.check("before-post-run-source-and-runtime-identity-check");
    assert.deepEqual(sourceIdentity(), sourceBefore, "reviewed source files changed during the capture");
    await verifyRuntimeFiles();
    await verifyExternalPins();
    record.packAfter = await stat(pack.path, { bigint: true }).then((value) => ({ dev: String(value.dev), ino: String(value.ino), size: String(value.size), mtimeNs: String(value.mtimeNs), ctimeNs: String(value.ctimeNs), nlink: String(value.nlink) }));
    assert.deepEqual(record.packAfter, { dev: pack.dev, ino: pack.ino, size: pack.size, mtimeNs: pack.mtimeNs, ctimeNs: pack.ctimeNs, nlink: pack.nlink }, "model pack identity changed during the capture");
    record.browserExecutableAfter = await chromiumStat(browser);
    assert.deepEqual(record.browserExecutableAfter, { path: record.browserExecutable.path, dev: record.browserExecutable.dev, ino: record.browserExecutable.ino,
      bytes: record.browserExecutable.bytes, mtimeNs: record.browserExecutable.mtimeNs, ctimeNs: record.browserExecutable.ctimeNs }, "pinned Chromium executable identity changed during the capture");
    record.status = "complete";
  } catch (error) {
    failure = error;
    record.status = "failed";
    record.error = String(error?.message || error);
  } finally {
    if (scope) {
      while (true) {
        try { await cleanupOwnedChildren(scope, state, record); cleanupError = null; break; }
        catch (error) {
          cleanupError = error;
          record.status = "failed";
          record.cleanupError = String(error?.message || error);
          record.leaseRetainedDuringCleanupRetries = true;
          record.cleanupRetryErrors ||= [];
          if (record.cleanupRetryErrors.length < 32) record.cleanupRetryErrors.push(record.cleanupError);
          try { await save(); } catch (saveError) { record.cleanupEvidenceSaveError = String(saveError?.message || saveError); }
          await new Promise((resolveDelay) => setTimeout(resolveDelay, 250));
        }
      }
    } else {
      record.cleanup = { checked: false, reason: scopeAdmissionAttempted ? "scope admission failed before ownership could be established" : "scope admission was not attempted" };
      if (scopeAdmissionAttempted) {
        cleanupError = new Error(record.cleanup.reason);
        record.leaseHeldForOuterTermination = true;
      }
    }
    if (cleanupError) {
      record.status = "failed";
      record.leaseHeldForOuterTermination = true;
      try { await save(); } catch (error) { record.cleanupEvidenceSaveError = String(error?.message || error); }
      await new Promise(() => { setInterval(() => {}, 60000); });
    }
    if (state.killTimer) clearTimeout(state.killTimer);
    if (state.logError || state.streamError) record.runnerLogCleanupError = state.logError || state.streamError;
    if (server) {
      try {
        if (server.listening) {
          server.closeAllConnections?.();
          await new Promise((resolveClose, rejectClose) => server.close((error) => error ? rejectClose(error) : resolveClose()));
        }
        serverClosed = true;
        record.serverClosed = true;
      } catch (error) { cleanupError ||= error; record.serverCloseError = String(error?.message || error); }
    }
    try {
      record.removedOwnedLinks = await removeOwnedLinks(ownedLinks);
      await verifyWasmLinksRemoved(record);
      if (record.pack) {
        const current = await stat(record.pack.path, { bigint: true });
        record.packAfterLinkCleanup = { dev: String(current.dev), ino: String(current.ino), size: String(current.size), mtimeNs: String(current.mtimeNs), nlink: String(current.nlink) };
        assert.deepEqual(record.packAfterLinkCleanup, { dev: record.pack.dev, ino: record.pack.ino, size: record.pack.size,
          mtimeNs: record.pack.mtimeNs, nlink: String(BigInt(record.pack.nlink) - 1n) }, "source pack identity changed during owned hardlink cleanup");
      }
    }
    catch (error) { cleanupError ||= error; record.linkCleanupError = String(error?.message || error); }
    if (scope) {
      try {
        record.scopeAfterCleanup = await readScopeMembers(scope);
        const actual = record.scopeAfterCleanup.filter((row) => row.state !== "Z").map((row) => ({ pid: row.pid, startTicks: row.startTicks, cgroup: row.cgroup })).sort((a, b) => a.pid - b.pid);
        const expected = scope.initialMembers.map(({ pid, startTicks, cgroup }) => ({ pid, startTicks, cgroup })).sort((a, b) => a.pid - b.pid);
        assert.deepEqual(actual, expected, "controller scope did not return to its exact initial Node/timeout members");
      } catch (error) { cleanupError ||= error; record.scopeCleanupError = String(error?.message || error); }
    }
    if (cleanupError) {
      record.status = "failed";
      record.cleanupError ||= String(cleanupError?.message || cleanupError);
      record.leaseHeldForOuterTermination = true;
      try { await save(); } catch (error) { record.cleanupEvidenceSaveError = String(error?.message || error); }
      await new Promise(() => { setInterval(() => {}, 60000); });
    }
    if (leaseGuard) {
      try { await verifyExternalPins(); record.lease.afterOwnedCleanup = await leaseGuard.check("after-owned-process-server-and-link-cleanup-and-scope-proof"); }
      catch (error) { cleanupError ||= error; record.finalLeaseError = String(error?.message || error); }
    }
    record.serverClosed = serverClosed;
    if (outputCreated) {
      try { await save(); } catch (error) { cleanupError ||= error; }
    }
    process.off("SIGTERM", stopTerm); process.off("SIGINT", stopInt);
  }
  if (cleanupError) throw cleanupError;
  if (failure) throw failure;
  if (state.stopReason) throw new Error(`Clef route controller stopped: ${state.stopReason}`);
  process.stdout.write(`Saved private Clef route diagnostic capture to ${join(output, "result.json")}\n`);
}

async function main() {
  if (process.argv.length === 2 || process.argv[2] === "--plan") { planOnly(); return; }
  const values = argumentsForRun();
  await execute(values);
}

main().catch((error) => { console.error(error?.message || error); process.exitCode = 1; });
