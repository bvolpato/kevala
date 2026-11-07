import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, mkdir, open, readFile, readdir, readlink, realpath, rename, stat } from "node:fs/promises";
import { createServer } from "node:http";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const SELF = fileURLToPath(import.meta.url);
const NODE = "/home/linuxbrew/.linuxbrew/Cellar/node@22/22.22.2_2/bin/node";
const LOCK = "/home/bruno/.cache/kevala/gpu-profile.lock";
const CAMPAIGN = "/home/bruno/githubworkspace/kevala/tmp/sota-20261106-kdDTGn";
const LEASE_DIRECTORY = join(CAMPAIGN, "campaign/lease-fdinfo-safety-v1/async-bind-r1");
const PINS = [
  { path: join(LEASE_DIRECTORY, "fd-lease.mjs"), sha256: "cf3ad8822a1386ec435723bb478b6a2ef1d753a0b9896c23e1b27d6e6d4b2915" },
  { path: join(LEASE_DIRECTORY, "lease-parse.mjs"), sha256: "6efc5046dc7b769f4c59ea256661581087b6e3ee1a366898b7852192697b5a31" },
  { path: join(CAMPAIGN, "shared-browser-boundary-v4/bench-gpu.py"), sha256: "eee15de6ed7e26d4329d9bdd4cb0653dcf62cad5b45d44f01cce5e110a599ff0" },
  { path: join(CAMPAIGN, "campaign/clef-affinity-startup-v1-private-session-bus-r2/session_bus.py"), sha256: "bac20966653dfaf8cbd7df636c03d02e77ee80b4ac1e9ac9931a6a30bb40c716" },
];
const CHROME = { path: "/home/bruno/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome", bytes: "301186376",
  version: "156.0.8078.4", sha256: "377a9fc07b495c3b8d0ab8ec1fdfd153641759f9756fea34d8dfce4de82f5025" };
const SOURCES = ["scripts/private/f16-feature-probe.mjs", "scripts/private/f16-feature-probe.py", "scripts/private/f16-feature-probe.html", "scripts/private/f16-feature-probe.test.mjs"];
const FLAGS = ["--enable-gpu", "--enable-unsafe-webgpu", "--use-angle=vulkan", "--enable-features=Vulkan", "--disable-vulkan-surface"];
const TOGGLE = "--enable-dawn-features=vulkan_enable_f16_on_nvidia";
const LIMITS = { "memory.high": "34359738368", "memory.max": "42949672960", "memory.swap.max": "4294967296" };
const PLAN = {
  purpose: "Two fresh Chrome processes capture adapter.features and created device.features only.",
  baseCommit: "81a27608f24e0b2252318515e272387c48dec7d8",
  capabilityOnly: true, benchmarkResult: false, correctnessResult: false, productionPromotion: false,
  lanes: [{ name: "baseline", chromiumArgs: FLAGS }, { name: "f16-toggle", chromiumArgs: [...FLAGS, TOGGLE] }],
  browser: CHROME, externalSourcePins: PINS,
  page: { adapter: { powerPreference: "high-performance", defaultFallback: "one requestAdapter() call only if the preferred request returns no adapter; record both outcomes" }, requiredIdentity: "NVIDIA Blackwell; observed nonfallback flag=false",
    optionalFeatures: ["timestamp-query", "shader-f16", "subgroups"], limits: "same requested limits as js/src/gpu.js",
    deviceLifecycle: "requestDevice then device.features then destroy; no shaders, pipelines, buffers, dispatch, runtime imports, model or WASM access" },
  containment: { lease: "external exclusive nonblocking flock --no-fork execs actual Node; pinned own-PID/fdinfo guard binds inherited canonical FD",
    profileLock: LOCK, memory: "32 GiB high, 40 GiB max, 4 GiB swap", cpu: "normal inherited scheduling; no affinity or priority mutation",
    timeout: "external timeout --signal=TERM --kill-after=15s 240s; per-page 60s; each UV child 90s",
    sessionBus: "pinned absent-endpoint helper passed only through ChromeService.env",
    monitor: "1s scope/known-PID samples; 5s owned process-group/session discovery; sampled assurance only",
    cleanup: "retain lease through child close/stdio close, server close, exact PID/start/cgroup survivor checks, final own-FD proof; unresolved survivors hold lease for outer timeout",
    network: "UV and Selenium Manager offline; Node serves one in-memory page on an ephemeral loopback port" },
  outputBounds: { pageBytes: 32768, laneJsonBytes: 16777216, controllerJsonBytes: 4194304, runnerLogBytes: 1048576, chromeLogsPerLaneBytes: 33554432 },
  requiredEnvironmentUnsets: ["VK_DRIVER_FILES", "KEVALA_PROFILE_LOCK_HELD", "KEVALA_BENCH_CDP"],
};
const hex = (value) => createHash("sha256").update(value).digest("hex");
const delay = (ms) => new Promise((done) => setTimeout(done, ms));
const key = (row) => `${row.pid}:${row.startTicks}`;
const same = (left, right) => left && right && key(left) === key(right);

async function bounded(path, limit) {
  const handle = await open(path, "r");
  try {
    const buffer = Buffer.alloc(limit + 1);
    let size = 0;
    while (size < buffer.length) {
      const { bytesRead } = await handle.read(buffer, size, buffer.length - size, null);
      if (!bytesRead) break;
      size += bytesRead;
    }
    assert(size <= limit, `read bound exceeded: ${path}`);
    return buffer.subarray(0, size);
  } finally { await handle.close(); }
}

async function writeJson(path, value, maximum = 4 * 1024**2) {
  const data = Buffer.from(JSON.stringify(value, null, 2) + "\n");
  assert(data.length <= maximum, `JSON output bound exceeded: ${path}`);
  const temporary = `${path}.${process.pid}.tmp`;
  const handle = await open(temporary, "wx", 0o600);
  try { await handle.writeFile(data); await handle.sync(); } finally { await handle.close(); }
  await rename(temporary, path);
  const directory = await open(dirname(path), "r");
  try { await directory.sync(); } finally { await directory.close(); }
  return { path, bytes: data.length, sha256: hex(data) };
}

async function sourceIdentity() {
  const rows = {};
  for (const path of SOURCES) {
    const bytes = await bounded(join(ROOT, path), 128 * 1024);
    rows[path] = { bytes: bytes.length, sha256: hex(bytes) };
  }
  return rows;
}

async function checkPins() {
  for (const pin of PINS) assert.equal(hex(await bounded(pin.path, 1024**2)), pin.sha256, `external source changed: ${pin.path}`);
}

async function processInfo(pid) {
  try {
    const raw = (await bounded(`/proc/${pid}/stat`, 16384)).toString();
    const fields = raw.slice(raw.lastIndexOf(")") + 1).trim().split(/\s+/);
    const cgroup = (await bounded(`/proc/${pid}/cgroup`, 16384)).toString().split("\n").find((line) => line.startsWith("0::"));
    assert(/^\d+$/.test(fields[19]) && cgroup, `invalid process metadata: ${pid}`);
    return { pid, state: fields[0], ppid: Number(fields[1]), pgrp: Number(fields[2]), session: Number(fields[3]),
      startTicks: fields[19], nice: Number(fields[16]), schedulingPolicy: Number(fields[38]), cgroup };
  } catch (error) {
    if (error.code === "ENOENT" || error.code === "ESRCH") return null;
    throw error;
  }
}

async function scopeMembers(scope) {
  const ids = (await bounded(join(scope.directory, "cgroup.procs"), 16384)).toString().trim().split(/\s+/).filter(Boolean).map(Number);
  assert(ids.length <= 256, "scope process count exceeded its bound");
  const rows = [];
  for (const pid of [...new Set(ids)].sort((a, b) => a - b)) {
    const row = await processInfo(pid);
    if (row) { assert.equal(row.cgroup, scope.cgroup, `scope member escaped: ${pid}`); rows.push(row); }
  }
  return rows;
}

async function processInventory() {
  const names = await readdir("/proc");
  assert(names.length <= 32768, "process inventory exceeded its bound");
  const rows = [];
  for (const name of names) if (/^\d+$/.test(name)) { const row = await processInfo(Number(name)); if (row) rows.push(row); }
  return rows;
}

async function admitScope(unit, output) {
  assert(/^kevala-f16-feature-[a-zA-Z0-9-]+\.scope$/.test(unit), "use a unique feature-probe scope");
  const owner = await processInfo(process.pid), parent = await processInfo(process.ppid);
  assert(owner && parent && owner.cgroup === parent.cgroup && owner.cgroup.endsWith(`/${unit}`), "Node/timeout scope identity differs");
  assert.equal(owner.nice, 0, "Node must retain normal priority");
  assert.equal(owner.schedulingPolicy, 0, "Node must retain SCHED_OTHER");
  assert.equal(process.execPath, NODE, "use the pinned absolute Node executable");
  assert((await readlink(`/proc/${parent.pid}/exe`)).endsWith("/timeout"), "Node must be a direct timeout child");
  const command = (await bounded(`/proc/${parent.pid}/cmdline`, 32768)).toString().split("\0").filter(Boolean);
  assert.deepEqual(command.slice(1), ["--signal=TERM", "--kill-after=15s", "240s", "flock", "--exclusive", "--nonblock", "--conflict-exit-code", "75", "--no-fork", LOCK,
    NODE, SELF, "--execute", "--scope-unit", unit, "--output", output], "external timeout/flock command differs from the reviewed envelope");
  const scope = { unit, owner, timeoutParent: parent, cgroup: owner.cgroup,
    directory: resolve("/sys/fs/cgroup", owner.cgroup.slice(3).replace(/^\/+/, "")), limits: LIMITS };
  for (const [file, value] of Object.entries(LIMITS)) assert.equal((await bounded(join(scope.directory, file), 128)).toString().trim(), value, `scope ${file} differs`);
  scope.initialMembers = await scopeMembers(scope);
  assert.deepEqual(scope.initialMembers.map((row) => row.pid).sort((a, b) => a - b), [owner.pid, parent.pid].sort((a, b) => a - b), "scope must initially contain only Node and timeout");
  return scope;
}

async function chromeIdentity() {
  assert.equal(await realpath(CHROME.path), CHROME.path, "Chrome path changed");
  const value = await stat(CHROME.path, { bigint: true });
  assert(value.isFile() && String(value.size) === CHROME.bytes, "Chrome file type or byte size changed");
  return { path: CHROME.path, bytes: String(value.size), dev: String(value.dev), ino: String(value.ino), mtimeNs: String(value.mtimeNs), ctimeNs: String(value.ctimeNs) };
}

async function verifyChrome() {
  const before = await chromeIdentity(), digest = createHash("sha256");
  for await (const chunk of createReadStream(CHROME.path)) digest.update(chunk);
  assert.equal(digest.digest("hex"), CHROME.sha256, "Chrome SHA-256 changed");
  assert.deepEqual(await chromeIdentity(), before, "Chrome changed while hashing");
  return { ...before, sha256: CHROME.sha256, version: CHROME.version };
}

function remember(state, row) {
  if (state.scope.initialMembers.some((initial) => same(initial, row))) return;
  if (!state.known.has(key(row))) assert(state.known.size < 1024, "known child identity bound exceeded");
  state.known.set(key(row), state.known.get(key(row)) || row);
}

async function observe(state, discover = false, enforce = true) {
  const live = new Map();
  for (const row of await scopeMembers(state.scope)) { remember(state, row); if (state.known.has(key(row))) live.set(key(row), row); }
  if (discover) {
    for (const row of await processInventory()) if (state.roots.some((root) => row.pgrp === root.pgrp && row.session === root.session)) {
      remember(state, row); live.set(key(row), row);
    }
  }
  for (const original of state.known.values()) {
    const row = await processInfo(original.pid);
    if (!same(original, row)) continue;
    live.set(key(row), row);
    if (enforce && row.cgroup !== state.scope.cgroup) {
      const error = new Error(`known child PID ${row.pid} escaped its exact cgroup`);
      error.identityEvidence = { expected: original, observed: row };
      throw error;
    }
  }
  return [...live.values()];
}

async function signalOwned(state, rows, signal, record) {
  for (const row of rows) {
    const current = await processInfo(row.pid);
    const inSession = state.roots.some((root) => current?.pgrp === root.pgrp && current?.session === root.session);
    if (!same(row, current) || (!inSession && current.cgroup !== state.scope.cgroup)) continue;
    try {
      process.kill(current.pid, signal);
      if (!record.signals.some((entry) => entry.pid === current.pid && entry.startTicks === current.startTicks && entry.signal === signal)) {
        assert(record.signals.length < 2048, "cleanup signal evidence exceeded its bound");
        record.signals.push({ pid: current.pid, startTicks: current.startTicks, cgroup: current.cgroup, signal });
      }
    }
    catch (error) { if (error.code !== "ESRCH") throw error; }
  }
}

async function cleanup(state, record) {
  const started = Date.now();
  let remaining;
  record.signals ||= [];
  do {
    remaining = await observe(state, true, false);
    if (!remaining.length && (!state.child || state.child.closed)) break;
    await signalOwned(state, remaining.filter((row) => row.state !== "Z"), Date.now() - started < 1000 ? "SIGTERM" : "SIGKILL", record);
    await delay(100);
  } while (Date.now() - started < 6000);
  remaining = await observe(state, true, false);
  record.cleanup = { knownOwnedProcesses: [...state.known.values()], remaining, childStdioClosed: !state.child || state.child.closed, checked: true };
  assert.equal(remaining.length, 0, "owned child PID/start identity survived cleanup");
  assert(!state.child || state.child.closed, "owned UV/Python child stdio survived cleanup");
}

async function runLane(lane, state, record, output, url, lease, save) {
  await lease.check(`before-${lane}-browser`);
  const laneDirectory = join(output, lane);
  await mkdir(laneDirectory, { mode: 0o700 });
  const diagnostics = join(laneDirectory, "diagnostics");
  await mkdir(diagnostics, { mode: 0o700 });
  const log = await open(join(laneDirectory, "runner.log"), "wx", 0o600);
  const args = ["run", "--offline", "scripts/private/f16-feature-probe.py", "--url", url, "--output", join(laneDirectory, "result.json"),
    "--chrome-diagnostics-dir", diagnostics, "--scope-unit", state.scope.unit, "--node-owner-pid", String(process.pid), "--node-owner-start-ticks", state.scope.owner.startTicks, "--lane", lane];
  const child = spawn("uv", args, { cwd: ROOT, env: { ...process.env, SE_OFFLINE: "true", SE_AVOID_STATS: "true" }, detached: true, stdio: ["ignore", "pipe", "pipe"] });
  const laneRecord = { lane, command: { executable: "uv", args }, logBytesObserved: 0, logBytesRetained: 0, monitorSamples: 0, fullDiscoveries: 0 };
  record.lanes.push(laneRecord);
  const childState = state.child = { process: child, closed: false, exit: null, logError: null, writes: Promise.resolve() };
  const close = new Promise((done) => child.once("close", (code, signal) => { childState.closed = true; childState.exit = { code, signal }; done(); }));
  const stop = (reason) => { state.stopReason ||= reason; };
  const retain = (chunk) => {
    laneRecord.logBytesObserved += chunk.length;
    const available = 1024**2 - laneRecord.logBytesRetained;
    if (available > 0) {
      const bytes = Buffer.from(chunk.subarray(0, available));
      laneRecord.logBytesRetained += bytes.length;
      childState.writes = childState.writes.then(() => log.writeFile(bytes)).catch((error) => { childState.logError = String(error); stop("runner-log-write"); });
    }
    if (laneRecord.logBytesObserved > 1024**2) stop("runner-log-budget");
  };
  child.stdout.on("data", retain); child.stderr.on("data", retain);
  child.stdout.on("error", (error) => { childState.logError = String(error); stop("runner-stdout"); });
  child.stderr.on("error", (error) => { childState.logError = String(error); stop("runner-stderr"); });
  try {
    await new Promise((done, reject) => { child.once("spawn", done); child.once("error", reject); });
    const root = await processInfo(child.pid);
    laneRecord.spawnedRootObserved = root;
    assert(root && root.ppid === process.pid && root.pgrp === child.pid && root.session === child.pid, "UV child parent/group/session admission failed");
    state.roots.push(root); remember(state, root); laneRecord.childRoot = root;
    assert.equal(root.cgroup, state.scope.cgroup, "UV child exact cgroup admission failed");
    const started = Date.now();
    let lastDiscovery = 0;
    while (!childState.closed) {
      if (Date.now() - started > 90000) stop("runner-deadline");
      const full = Date.now() - lastDiscovery >= 5000;
      try { await observe(state, full); laneRecord.monitorSamples++; if (full) { laneRecord.fullDiscoveries++; lastDiscovery = Date.now(); } }
      catch (error) { laneRecord.monitorError = String(error); laneRecord.monitorIdentityEvidence = error.identityEvidence || null; stop("scope-monitor"); }
      if (state.stopReason) throw new Error(state.stopReason);
      await Promise.race([close, delay(1000)]);
    }
    laneRecord.exit = childState.exit;
    assert.equal(childState.exit.code, 0, `${lane} feature runner failed`);
    const raw = await bounded(join(laneDirectory, "result.json"), 16 * 1024**2);
    const result = JSON.parse(raw.toString());
    assert.equal(result.status, "complete", `${lane} metadata capture did not complete`);
    assert.equal(result.scope, "capability-metadata-only");
    assert.equal(result.lane, lane);
    assert.equal(result.runner.browserVersion, CHROME.version);
    assert.deepEqual(result.runner.chromiumArgs, lane === "baseline" ? FLAGS : [...FLAGS, TOGGLE]);
    assert.deepEqual(result.page.identity, { nvidia: true, blackwell: true, nonfallback: true });
    assert.equal(result.page.device.destroyed, true);
    laneRecord.result = { path: join(laneDirectory, "result.json"), bytes: raw.length, sha256: hex(raw) };
    laneRecord.features = { adapter: result.page.adapter.features, device: result.page.device.features };
  } finally {
    try { await cleanup(state, laneRecord); }
    finally {
      if (childState.closed) {
        await childState.writes;
        try { await log.sync(); } finally { await log.close(); }
        childState.logClosed = true;
      } else { childState.openLog = log; }
      await save();
    }
  }
  assert(!childState.logError, childState.logError);
  await lease.check(`after-${lane}-children-and-stdio-closed`);
}

async function execute(values) {
  assert.equal(process.env.KEVALA_F16_FEATURE_PROBE_AUTHORIZED, "1", "a source-reviewed feature-probe execution grant is required");
  for (const name of PLAN.requiredEnvironmentUnsets) assert(!Object.hasOwn(process.env, name), `${name} must be unset`);
  const output = resolve(values.output), outputRoot = join(ROOT, "tmp");
  assert.equal(dirname(output), outputRoot.replace(/\/$/, ""), "output must be a new direct child of the worktree tmp directory");
  await mkdir(outputRoot, { recursive: true });
  const info = await lstat(outputRoot);
  assert(info.isDirectory() && !info.isSymbolicLink() && info.uid === process.getuid(), "output parent must be owned and nonsymlink");
  await mkdir(output, { mode: 0o700 });
  const record = { status: "running", capabilityOnly: true, productionPromotion: false, lanes: [], lease: {}, plan: PLAN };
  const save = () => writeJson(join(output, "controller.json"), record);
  const state = { scope: null, roots: [], known: new Map(), child: null, stopReason: null };
  const stopped = (signal) => { state.stopReason ||= `controller-${signal}`; record.stopSignal = signal; };
  const term = () => stopped("SIGTERM"), interrupt = () => stopped("SIGINT");
  process.on("SIGTERM", term); process.on("SIGINT", interrupt);
  let lease, server, sourceBefore, failure, cleanupFailure;
  try {
    await checkPins();
    const { createLeaseGuard } = await import(pathToFileURL(PINS[0].path).href);
    lease = createLeaseGuard({ canonicalPath: LOCK, recordEvidence: (evidence) => writeJson(join(output, `lease-${String(evidence.sequence).padStart(3, "0")}-${evidence.label}.json`), evidence) });
    record.lease.bind = await lease.bind("feature-controller-bind-inherited-fd");
    state.scope = record.scope = await admitScope(values["scope-unit"], output);
    sourceBefore = record.sourceIdentity = await sourceIdentity();
    record.lease.beforeBrowserBinaryRead = await lease.check("before-browser-binary-identity-read");
    record.browserExecutable = await verifyChrome();
    const page = await bounded(join(ROOT, SOURCES[2]), 32768);
    record.server = { ownedByNodePid: process.pid, host: "127.0.0.1", pageSha256: hex(page), requests: [], closed: false };
    server = createServer((request, response) => {
      if (record.server.requests.length < 64) record.server.requests.push({ method: String(request.method).slice(0, 16), path: String(request.url).slice(0, 256) });
      else state.stopReason ||= "server-request-count-bound";
      response.setHeader("Cache-Control", "no-store");
      response.setHeader("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; connect-src 'none'; object-src 'none'");
      if (request.method === "GET" && request.url === "/feature-probe") { response.writeHead(200, { "Content-Type": "text/html; charset=utf-8" }); response.end(page); }
      else { response.writeHead(404); response.end(); }
    });
    server.requestTimeout = 5000; server.headersTimeout = 5000; server.keepAliveTimeout = 1000;
    await new Promise((done, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", done); });
    const address = server.address();
    assert(address && typeof address === "object" && address.port > 0);
    record.server.port = address.port;
    const url = `http://127.0.0.1:${address.port}/feature-probe`;
    await save();
    for (const lane of ["baseline", "f16-toggle"]) { assert(!state.stopReason, state.stopReason); await runLane(lane, state, record, output, url, lease, save); }
    assert.deepEqual(await sourceIdentity(), sourceBefore, "feature probe source changed during capture");
    await checkPins();
    const after = await chromeIdentity();
    const { sha256: _sha256, version: _version, ...before } = record.browserExecutable;
    assert.deepEqual(after, before, "Chrome executable metadata changed during capture");
    record.status = "complete";
  } catch (error) { failure = error; record.status = "failed"; record.error = String(error?.message || error).slice(0, 4096); }
  finally {
    if (state.scope) {
      try { await cleanup(state, record); } catch (error) { cleanupFailure = error; record.cleanupError = String(error); }
    }
    if (state.child?.closed && state.child.openLog) {
      try { await state.child.writes; await state.child.openLog.sync(); await state.child.openLog.close(); state.child.logClosed = true; }
      catch (error) { cleanupFailure ||= error; record.runnerLogCleanupError = String(error); }
    }
    if (server) {
      try { server.closeAllConnections(); await new Promise((done, reject) => server.close((error) => error ? reject(error) : done())); record.server.closed = true; }
      catch (error) { cleanupFailure ||= error; record.serverCleanupError = String(error); }
    }
    if (state.scope) {
      try {
        const final = await scopeMembers(state.scope);
        record.scopeAfterCleanup = final;
        assert.deepEqual(final.map(key).sort(), state.scope.initialMembers.map(key).sort(), "scope did not return to exact initial Node/timeout identities");
        assert(!state.child || (state.child.closed && state.child.logClosed), "UV/Python stdio or runner log remains open");
        assert.equal((await observe(state, true, false)).length, 0, "known owned process survived final cleanup");
      } catch (error) { cleanupFailure ||= error; record.finalScopeError = String(error); }
    }
    if (lease && !cleanupFailure) {
      try { record.lease.final = await lease.check("after-owned-processes-stdio-and-server-closed"); }
      catch (error) { cleanupFailure = error; record.finalLeaseError = String(error); }
    }
    if (cleanupFailure) {
      record.status = "failed"; record.leaseHeldForOuterTermination = true;
      try { await save(); } catch (error) { console.error(`Failed to save cleanup evidence: ${String(error).slice(0, 4096)}`); }
      await new Promise(() => { setInterval(() => {}, 1000); });
    }
    await save();
    process.off("SIGTERM", term); process.off("SIGINT", interrupt);
  }
  if (failure) throw failure;
  if (state.stopReason) throw new Error(state.stopReason);
  process.stdout.write(`Saved capability metadata only to ${output}\n`);
}

async function main() {
  if (process.argv.length === 2 || (process.argv.length === 3 && process.argv[2] === "--plan")) {
    process.stdout.write(JSON.stringify({ ...PLAN, sourceIdentity: await sourceIdentity() }, null, 2) + "\n");
    return;
  }
  const { values } = parseArgs({ options: { execute: { type: "boolean" }, "scope-unit": { type: "string" }, output: { type: "string" } }, strict: true, allowPositionals: false });
  assert(values.execute && values["scope-unit"] && values.output, "use --execute --scope-unit <unique.scope> --output <new-absolute-directory>");
  await execute(values);
}

main().catch((error) => { console.error(error?.message || error); process.exitCode = 1; });
