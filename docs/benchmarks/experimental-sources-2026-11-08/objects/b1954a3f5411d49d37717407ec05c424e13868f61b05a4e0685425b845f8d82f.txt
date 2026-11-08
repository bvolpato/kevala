#!/usr/bin/env node
// Prospective runner only. Run as the FLOCK owner inside the memory scope created by root.
import { createHash, randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import { lstat, mkdir, open, readFile, readlink, realpath, stat, symlink, unlink, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { createKevalaServer } from "../serve.mjs";
import { MODELS } from "../../js/src/source.js";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const LOCK = "/home/bruno/.cache/kevala/gpu-profile.lock";
const PRIMARY = "/home/bruno/githubworkspace/kevala";
const FROZEN = {
  "js/src/gpu-laya.js": "f7811cb50cbdd37b31401076b94ff5fe1e89abd7f4390dd0b51573db5a4f01b5",
  "js/src/gpu-laya-geglu-epilogue.js": "72c268d37d7fbe4aa6f6968ef7e81c7da1dfc663f509a6ff00cfe63e2b2fb738",
  "scripts/private/laya-geglu-protocol.mjs": "e65f90b92d24fbdee78da668d43d3a00d107664cbdd2b781832bef70d89c6496",
  "scripts/private/laya-geglu-protocol.json": "5c6dda26af52e5af14c3cd62c70f4b82d019649e59d443c607a1cac13a81acef",
};
const check = (ok, message) => { if (!ok) throw new Error(message); };
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const identity = (s) => ({ device: String(s.dev), inode: String(s.ino), bytes: String(s.size), mtimeNs: String(s.mtimeNs) });
const delay = (ms) => new Promise((resolveDelay) => setTimeout(resolveDelay, ms));

async function processInfo(pid) {
  try {
    const raw = await readFile(`/proc/${pid}/stat`, "utf8"), fields = raw.slice(raw.lastIndexOf(")") + 1).trim().split(/\s+/);
    return { pid, state: fields[0], ppid: Number(fields[1]), pgid: Number(fields[2]), session: Number(fields[3]), startTicks: fields[19],
      cgroup: (await readFile(`/proc/${pid}/cgroup`, "utf8")).trim() };
  } catch (error) { if (["ENOENT", "ESRCH"].includes(error.code)) return null; throw error; }
}

async function scopeMembers(scope) {
  const members = (await readFile(resolve(scope.directory, "cgroup.procs"), "utf8")).trim().split(/\s+/).filter(Boolean).map(Number);
  check(members.length <= 256 && members.every((pid) => Number.isSafeInteger(pid) && pid > 0), "Owned scope process bound exceeded");
  return members;
}

async function containment() {
  const cgroupText = await readFile("/proc/self/cgroup", "utf8"), match = cgroupText.match(/^0::(.+)$/m);
  check(match, "Unified cgroup metadata is unavailable");
  const cgroup = resolve("/sys/fs/cgroup", `.${match[1]}`);
  const limits = Object.fromEntries(await Promise.all(["memory.high", "memory.max", "memory.swap.max"].map(async (name) => [name, (await readFile(resolve(cgroup, name), "utf8")).trim()])));
  check(limits["memory.high"] === "34359738368" && limits["memory.max"] === "42949672960" && limits["memory.swap.max"] === "4294967296", "Runner is outside the required 32G/40G/4G memory scope");
  const meminfo = await readFile("/proc/meminfo", "utf8");
  const kib = (name) => Number(meminfo.match(new RegExp(`^${name}:\\s+(\\d+) kB$`, "m"))?.[1]);
  check(kib("MemAvailable") >= 32 * 1024 ** 2 && kib("SwapFree") >= 4 * 1024 ** 2, "Insufficient host memory or swap headroom");
  const affinity = (await readFile("/proc/self/status", "utf8")).match(/^Cpus_allowed_list:\s+(.+)$/m)?.[1];
  return { cgroup: match[1], limits, cpuAllowedList: affinity, cpuPolicy: "normal inherited scheduling; no taskset or CPU7 pinning",
    memAvailableKiB: kib("MemAvailable"), swapFreeKiB: kib("SwapFree"), sampledOnly: true };
}

async function verifiedPack(path, receipt, signal) {
  const file = await open(path, "r");
  try {
    const before = identity(await file.stat({ bigint: true }));
    check(before.bytes === receipt.identity.size, "Local pack size differs from the header receipt");
    const header = Buffer.alloc(receipt.range?.endExclusive || receipt.hashedRange.endExclusive);
    check((await file.read(header, 0, header.length, 0)).bytesRead === header.length && hash(header) === receipt.sha256, "Local pack header differs from the frozen receipt");
    const digest = createHash("sha256");
    for await (const chunk of file.createReadStream({ start: 0, autoClose: false, highWaterMark: 1024 * 1024 })) { signal.throwIfAborted(); digest.update(chunk); }
    const sha256 = digest.digest("hex");
    check(sha256 === MODELS.laya.packSha256, "Actual full pack hash differs from the catalog pin");
    check(JSON.stringify(identity(await file.stat({ bigint: true }))) === JSON.stringify(before), "Pack changed while hashing");
    const parsed = JSON.parse(header.toString("utf8", 16));
    return { path: await realpath(path), identity: before, bytes: Number(before.bytes), sha256,
      headerBytes: header.length, headerSha256: hash(header), headerJsonSha256: hash(JSON.stringify(parsed)) };
  } finally { await file.close(); }
}

async function verifiedBrowserExecutable(path, expected, signal) {
  check(path === expected.path && await realpath(path) === expected.path, "Explicit browser path differs from the reviewed Chrome executable");
  const file = await open(path, "r");
  try {
    const info = await file.stat({ bigint: true }), before = identity(info);
    check(info.isFile() && (info.mode & 0o111n) !== 0n && before.bytes === String(expected.bytes), "Chrome executable type, mode, or size differs");
    const digest = createHash("sha256");
    for await (const chunk of file.createReadStream({ start: 0, autoClose: false, highWaterMark: 1024 * 1024 })) { signal.throwIfAborted(); digest.update(chunk); }
    const sha256 = digest.digest("hex");
    check(sha256 === expected.sha256, "Chrome executable full hash differs from the reviewed pin");
    check(JSON.stringify(identity(await file.stat({ bigint: true }))) === JSON.stringify(before), "Chrome executable changed while hashing");
    return { path, identity: before, mode: (info.mode & 0o777n).toString(8), uid: String(info.uid), gid: String(info.gid), sha256, expectedVersion: expected.version };
  } finally { await file.close(); }
}

async function run() {
  const { values } = parseArgs({ options: { output: { type: "string" },
    pack: { type: "string", default: `${PRIMARY}/tmp/laya-q8.kevala` }, "wasm-source": { type: "string", default: `${PRIMARY}/js/src` },
    binary: { type: "string" }, "scope-unit": { type: "string" }, "packet-sha256": { type: "string" }, execute: { type: "boolean" } }, strict: true });
  check(values.execute === true && values.output && values["scope-unit"] && /^[0-9a-f]{64}$/.test(values["packet-sha256"] || ""), "Provide --execute, --output, --scope-unit, and the reviewed --packet-sha256");
  const packetBytes = await readFile(new URL("./laya-geglu-controller-packet.json", import.meta.url));
  check(hash(packetBytes) === values["packet-sha256"], "Controller execution packet differs from the reviewed hash");
  const packet = JSON.parse(packetBytes);
  check(values.binary && values.binary === packet.browserExecutable?.path, "Provide the packet's explicit --binary Chrome 156 path");
  const output = resolve(values.output), wrapper = await realpath(packet.sharedWrapper.path);
  const record = { status: "running", execution: "private-laya-geglu-v2", packetSha256: values["packet-sha256"], continuousContainmentClaim: false, idleHostClaim: false };
  const ownedLinks = [], tag = randomUUID(), temporaryManifest = resolve(ROOT, `tmp/laya-geglu-manifest-${tag}.json`);
  let server, child, leaseGuard, admittedScope, childExit, childClose, wrapperLog, wrapperLogPath, timeout, killTimer, cleanupPromise;
  let outputCreated = false, manifestCreated = false;
  const knownChildren = new Map(), sockets = new Set();
  const abort = new AbortController();
  let failWrapperLog;
  const wrapperLogFailure = new Promise((resolveFailure) => { failWrapperLog = resolveFailure; });
  const stop = (signal) => {
    record.status = "failed"; record.stopSignal = signal;
    abort.abort(new Error(`Owned Node controller received ${signal}`));
    if (child?.pid) void signalChildren("SIGTERM").catch((error) => { record.signalCleanupError = String(error); });
  };
  const stopTerm = () => stop("SIGTERM"), stopInt = () => stop("SIGINT");
  process.on("SIGTERM", stopTerm); process.on("SIGINT", stopInt);
  const save = () => writeFile(resolve(output, "controller.json"), JSON.stringify(record, null, 2) + "\n");
  const holdLeaseForOuterTermination = async (proof) => {
    proof.leaseHeldForOuterTermination = true;
    try { await save(); }
    catch (error) { proof.evidenceSaveError = String(error); }
    finally { await delay(795000); }
  };
  const allowed = (info) => info && admittedScope.controllers.some((owner) => owner.pid === info.pid && owner.startTicks === info.startTicks && owner.cgroup === info.cgroup);
  const remember = (info, origin) => {
    if (!info || allowed(info)) return;
    const key = `${info.pid}:${info.startTicks}`;
    if (!knownChildren.has(key)) { check(knownChildren.size < 512, "Owned child identity bound exceeded"); knownChildren.set(key, { ...info, origin }); }
  };
  const discoverChildren = async () => {
    for (const pid of await scopeMembers(admittedScope)) {
      const info = await processInfo(pid);
      if (info?.cgroup === admittedScope.cgroup && !allowed(info)) remember(info, "post-admission-owned-scope");
    }
    const rows = [...knownChildren.values()];
    return (await Promise.all(rows.map(async (original) => {
      const current = await processInfo(original.pid);
      return current?.startTicks === original.startTicks ? current : null;
    }))).filter(Boolean);
  };
  const signalChildren = async (signal) => {
    record.childCleanup ||= { signals: [] };
    const live = await discoverChildren();
    const groupWitness = live.find((info) => info.pgid === child?.pid && info.session === child.pid && info.state !== "Z");
    if (groupWitness) {
      const current = await processInfo(groupWitness.pid);
      if (current?.startTicks === groupWitness.startTicks && current.pgid === child.pid && current.session === child.pid) {
        try { process.kill(-child.pid, signal); record.childCleanup.signals.push({ signal, pgid: child.pid, witness: current }); }
        catch (error) { if (error.code !== "ESRCH") throw error; }
      }
    }
    for (const original of live) {
      const current = await processInfo(original.pid);
      if (current?.startTicks !== original.startTicks || current.state === "Z") continue;
      try { process.kill(current.pid, signal); record.childCleanup.signals.push({ signal, process: current }); }
      catch (error) { if (error.code !== "ESRCH") throw error; }
    }
  };
  const cleanupChildren = () => cleanupPromise ||= (async () => {
    if (!admittedScope) return { scopeAdmitted: false };
    const proof = record.childCleanup = { scopeAdmitted: true, scope: values["scope-unit"], ownedPgid: child?.pid || null,
      termGraceMs: 600, maximumCleanupMs: 5000, signals: record.childCleanup?.signals || [], knownChildren: [], positiveZeroSurvivors: false };
    try {
      if (!record.page) record.page = await readFile(resolve(output, "browser.json"), "utf8").then(JSON.parse).catch(() => null);
      for (const snapshot of record.page?.runner?.containment?.snapshots || []) for (const info of snapshot.processes || []) remember(info, "adapter-browser-bookend");
      if (!child?.pid) check((await scopeMembers(admittedScope)).every((pid) => admittedScope.controllers.some((owner) => owner.pid === pid)), "Scope gained a process before any child was launched");
      const deadline = Date.now() + 5000;
      if (child?.pid) {
        await signalChildren("SIGTERM");
        await delay(600);
        while ((await discoverChildren()).length && Date.now() < deadline) { await signalChildren("SIGKILL"); await delay(200); }
        await Promise.race([childExit, delay(Math.max(0, deadline - Date.now())).then(() => { throw new Error("Owned child leader was not reaped before cleanup deadline"); })]);
        await Promise.race([childClose, delay(Math.max(0, deadline - Date.now())).then(() => { throw new Error("Owned child stdio remained open after cleanup"); })]);
      }
      const remainingKnown = child?.pid ? await discoverChildren() : [];
      const finalMembers = await scopeMembers(admittedScope), finalRows = (await Promise.all(finalMembers.map(processInfo))).filter(Boolean);
      const remainingScope = finalRows.filter((info) => !allowed(info));
      proof.knownChildren = [...knownChildren.values()]; proof.finalMembers = finalMembers; proof.finalProcesses = finalRows;
      proof.remainingKnownChildren = remainingKnown; proof.remainingScopeChildren = remainingScope;
      check(remainingKnown.length === 0 && remainingScope.length === 0 && finalMembers.includes(process.pid) && finalRows.some((info) => info.pid === process.pid && allowed(info)) && finalMembers.every((pid) => admittedScope.controllers.some((owner) => owner.pid === pid)), "Owned scope or known browser identities still contain survivors, or the Node owner left its scope");
      proof.leaderReaped = !child?.pid || child.exitCode !== null || child.signalCode !== null;
      check(proof.leaderReaped, "Owned child leader has not been reaped");
      proof.positiveZeroSurvivors = true;
      return proof;
    } catch (error) {
      proof.error = String(error); proof.knownChildren = [...knownChildren.values()]; record.status = "failed";
      // Keep this Node and its inherited lease alive for the admitted outer timeout's
      // bounded TERM/KILL path. Do not release the lease with unresolved owned work.
      await holdLeaseForOuterTermination(proof);
      throw error;
    }
  })();
  const closeHttp = async () => {
    if (!server) return;
    const proof = record.httpCleanup = { connectionsBefore: sockets.size, destroyedConnections: 0 };
    const closed = new Promise((resolveClosed) => server.close(() => resolveClosed(true)));
    server.closeIdleConnections();
    if (!await Promise.race([closed, delay(1000).then(() => false)])) {
      for (const socket of sockets) { socket.destroy(); proof.destroyedConnections++; }
      check(await Promise.race([closed, delay(500).then(() => false)]), "Owned HTTP server did not close after destroying its connections");
    }
    proof.remainingConnections = sockets.size; check(sockets.size === 0, "Owned HTTP connections remain"); proof.closed = true;
  };
  try {
    await mkdir(output, { recursive: false });
    outputCreated = true;
    for (const helper of packet.leaseHelpers) check(hash(await readFile(helper.path)) === helper.sha256, `Reviewed lease helper differs: ${helper.path}`);
    const { createLeaseGuard } = await import(pathToFileURL(packet.leaseHelpers[0].path).href);
    leaseGuard = createLeaseGuard({ canonicalPath: LOCK, recordEvidence: async (evidence) => {
      const name = `lease-${String(evidence.sequence).padStart(3, "0")}-${evidence.label}.json`, path = resolve(output, name);
      if (evidence.label === "after-owned-cleanup") evidence.childCleanup = record.childCleanup;
      const bytes = JSON.stringify(evidence, null, 2) + "\n", file = await open(path, "wx", 0o600);
      try { await file.writeFile(bytes); await file.sync(); } finally { await file.close(); }
      const directory = await open(output, "r");
      try { await directory.sync(); } finally { await directory.close(); }
      return { path, sha256: hash(bytes) };
    } });
    record.leaseBind = await leaseGuard.bind("node-own-bind-before-preflight");
    check(!Object.hasOwn(process.env, "VK_DRIVER_FILES"), "VK_DRIVER_FILES must be unset for the accepted isolated Chromium hardware launch");
    record.before = await containment();
    check(record.before.cgroup.endsWith(`/${values["scope-unit"]}`), "Runner scope unit differs from --scope-unit");
    const owner = await processInfo(process.pid), timeoutParent = await processInfo(process.ppid);
    check(owner.startTicks === leaseGuard.binding.startTicks && timeoutParent?.cgroup === owner.cgroup && await readlink(`/proc/${timeoutParent.pid}/exe`) === "/usr/bin/timeout", "Expected the admitted outer timeout as Node's direct parent");
    const timeoutArgs = (await readFile(`/proc/${timeoutParent.pid}/cmdline`, "utf8")).split("\0").filter(Boolean);
    check(JSON.stringify(timeoutArgs.slice(1, 4)) === JSON.stringify(["--signal=TERM", "--kill-after=15s", "780s"]) && timeoutArgs.includes(fileURLToPath(import.meta.url)) && timeoutArgs.includes("--no-fork") && timeoutArgs.includes(LOCK), "Outer timeout/flock envelope differs from the reviewed plan");
    admittedScope = { directory: resolve("/sys/fs/cgroup", `.${record.before.cgroup}`), cgroup: owner.cgroup, controllers: [owner, timeoutParent] };
    const initialMembers = await scopeMembers(admittedScope);
    check(JSON.stringify(initialMembers.sort((a, b) => a - b)) === JSON.stringify([owner.pid, timeoutParent.pid].sort((a, b) => a - b)), "Fresh scope must contain only this Node and its admitted outer timeout");
    record.scopeAdmission = { ...admittedScope, initialMembers, timeoutArgs };
    record.wrapper = { path: wrapper, sha256: hash(await readFile(wrapper)) };
    check(record.wrapper.sha256 === packet.sharedWrapper.sha256, "Shared browser wrapper hash differs");
    check(hash(await readFile(packet.emitterPreflight.path)) === packet.emitterPreflight.sha256, "Existing WASM-emitter preflight receipt differs");
    record.emitterPreflight = packet.emitterPreflight;
    for (const entry of packet.sourcePins) {
      check((await stat(resolve(ROOT, entry.path))).size === entry.bytes, `Reviewed execution source size differs: ${entry.path}`);
      check(hash(await readFile(resolve(ROOT, entry.path))) === entry.sha256, `Reviewed execution source differs: ${entry.path}`);
    }
    for (const [path, digest] of Object.entries(FROZEN)) check(hash(await readFile(resolve(ROOT, path))) === digest, `Frozen v2 file differs: ${path}`);
    const protocol = JSON.parse(await readFile(resolve(ROOT, "scripts/private/laya-geglu-protocol.json"), "utf8"));
    record.leaseBeforeExecutableRead = await leaseGuard.check("before-chrome-executable-read");
    record.browserExecutable = await verifiedBrowserExecutable(values.binary, packet.browserExecutable, abort.signal);
    record.leaseBeforeInputRead = await leaseGuard.check("before-pack-and-wasm-read");
    record.pack = await verifiedPack(resolve(values.pack), protocol.headerReceipt, abort.signal);
    const link = async (path, target, allowExisting = false) => {
      await mkdir(dirname(path), { recursive: true });
      if (await lstat(path).catch((e) => { if (e.code === "ENOENT") return null; throw e; })) { check(allowExisting, `Refusing to replace existing path: ${path}`); return; }
      await symlink(target, path); ownedLinks.push({ path, target });
    };
    record.wasm = [];
    for (const expected of protocol.wasmIdentity.expected) {
      const target = resolve(values["wasm-source"], `kevala-${expected.flavor}.wasm`), destination = resolve(ROOT, expected.path);
      check((await stat(target)).size === expected.bytes, `Actual CI80 WASM size differs: ${expected.flavor}`);
      const bytes = await readFile(target);
      check(bytes.length === expected.bytes && hash(bytes) === expected.sha256, `Actual CI80 WASM differs: ${expected.flavor}`);
      await link(destination, target, true);
      check((await stat(destination)).size === expected.bytes, `Served WASM size differs: ${expected.flavor}`);
      check(hash(await readFile(destination)) === expected.sha256, `Served WASM differs: ${expected.flavor}`);
      record.wasm.push({ ...expected, source: await realpath(target), served: await realpath(destination) });
    }
    const packPath = `tmp/laya-geglu-input-${tag}.kevala`;
    await link(resolve(ROOT, packPath), record.pack.path);
    server = createKevalaServer({ root: ROOT, isolate: true, models: {} });
    server.on("connection", (socket) => { sockets.add(socket); socket.once("close", () => sockets.delete(socket)); });
    await new Promise((resolveReady, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", resolveReady); });
    const base = `http://127.0.0.1:${server.address().port}`;
    const names = new Set([...packet.browserResources, ...protocol.wasmIdentity.expected.map((entry) => entry.path)]);
    const resources = [];
    let resourceBytes = 0;
    for (const path of [...names].sort()) {
      const bytes = await readFile(resolve(ROOT, path)); resourceBytes += bytes.length;
      check(resources.length < 256 && resourceBytes <= 64 * 1024 ** 2, "Runtime resource inventory exceeds its bound");
      resources.push({ path, bytes: bytes.length, sha256: hash(bytes) });
    }
    const manifest = { base: protocol.base, worktree: ROOT, resources, sourceFingerprint: hash(resources.map((entry) => `${entry.path}:${entry.sha256}\n`).join("")),
      pack: { ...record.pack, url: `${base}/${packPath}` }, controller: { sharedWrapper: record.wrapper, adapter: packet.sourcePins.find((entry) => entry.path.endsWith("laya-geglu-browser.py")) },
      gpuPolicy: "actual runtime defaults and gpuKernel:auto; do not force a generic or f32 baseline", cpuPolicy: "normal unpinned scheduling" };
    await writeFile(temporaryManifest, JSON.stringify(manifest, null, 2) + "\n", { flag: "wx" });
    manifestCreated = true;
    await writeFile(resolve(output, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
    record.url = `${base}/dev/laya-geglu-private.html?manifest=${encodeURIComponent(`/tmp/laya-geglu-manifest-${tag}.json`)}`;
    record.stage = "fresh-shared-wrapper";
    abort.signal.throwIfAborted();
    record.leaseBeforeBrowser = await leaseGuard.check("before-browser-controller");
    await save();
    const diagnosticsDir = resolve(output, "chrome-diagnostics");
    await mkdir(diagnosticsDir, { mode: 0o700 });
    const args = ["run", resolve(ROOT, "scripts/private/laya-geglu-browser.py"), "--url", record.url, "--output", resolve(output, "browser.json"), "--timeout", "660",
      "--shared-wrapper", wrapper, "--shared-wrapper-sha256", record.wrapper.sha256, "--session-bus-helper", packet.sessionBusHelper.path,
      "--session-bus-helper-sha256", packet.sessionBusHelper.sha256, "--scope-unit", values["scope-unit"],
      "--node-owner-pid", leaseGuard.binding.pid, "--node-owner-start-ticks", leaseGuard.binding.startTicks, "--chrome-diagnostics-dir", diagnosticsDir];
    if (values.binary) args.push("--binary", values.binary);
    wrapperLogPath = resolve(output, "wrapper.log");
    const logBudget = 4 * 1024 ** 2;
    record.wrapperLog = { path: wrapperLogPath, budgetBytes: logBudget, observedBytes: 0, retainedBytes: 0 };
    wrapperLog = createWriteStream(wrapperLogPath, { flags: "wx" });
    wrapperLog.on("error", (error) => {
      record.wrapperLog.error ||= { message: error.message, code: error.code || null };
      record.status = "failed"; abort.abort(error); failWrapperLog(error);
      if (child?.pid) void signalChildren("SIGTERM").catch((failure) => { record.signalCleanupError = String(failure); });
    });
    check((await scopeMembers(admittedScope)).every((pid) => admittedScope.controllers.some((owner) => owner.pid === pid)), "Scope gained an unadmitted process before browser launch");
    abort.signal.throwIfAborted();
    child = spawn("uv", args, { cwd: ROOT, detached: true, stdio: ["ignore", "pipe", "pipe"], env: { ...process.env } });
    childExit = new Promise((resolveExit) => { child.once("error", (error) => resolveExit({ code: null, spawnError: error.message })); child.once("exit", (code, signal) => resolveExit({ code, signal })); });
    childClose = new Promise((resolveClose) => child.once("close", resolveClose));
    if (child.pid) {
      const launched = await processInfo(child.pid); remember(launched, "spawned-uv-leader");
      record.ownedChild = { pid: child.pid, pgid: child.pid, session: child.pid, initialIdentity: launched };
      if (launched) check(launched.pgid === child.pid && launched.session === child.pid, "Detached child did not create its expected owned group/session");
    }
    const retain = (data) => {
      record.wrapperLog.observedBytes += data.length;
      const remaining = logBudget - record.wrapperLog.retainedBytes;
      if (remaining > 0 && !record.wrapperLog.error && !wrapperLog.destroyed && !wrapperLog.errored) { const bytes = data.subarray(0, remaining); wrapperLog.write(bytes); record.wrapperLog.retainedBytes += bytes.length; }
      if (record.wrapperLog.observedBytes > logBudget) { record.wrapperLog.budgetExceeded = true; void signalChildren("SIGTERM").catch((error) => { record.signalCleanupError = String(error); }); }
    };
    child.stdout.on("data", retain); child.stderr.on("data", retain);
    timeout = setTimeout(() => {
      record.controllerDeadlineExceeded = true;
      void signalChildren("SIGTERM").catch((error) => { record.signalCleanupError = String(error); });
      killTimer = setTimeout(() => { void signalChildren("SIGKILL").catch((error) => { record.signalCleanupError = String(error); }); }, 10000);
    }, 720000);
    record.wrapperExit = await Promise.race([childExit, wrapperLogFailure.then((error) => { throw error; })]);
    if (record.wrapperExit.spawnError) throw new Error(record.wrapperExit.spawnError);
    record.page = await readFile(resolve(output, "browser.json"), "utf8").then(JSON.parse).catch((error) => ({ status: "missing", error: error.message }));
    record.leaseAfterBrowser = await leaseGuard.check("after-browser-controller");
    record.after = await containment();
    check(JSON.stringify(identity(await stat(record.browserExecutable.path, { bigint: true }))) === JSON.stringify(record.browserExecutable.identity), "Chrome executable metadata changed across the campaign");
    check(JSON.stringify(identity(await stat(record.pack.path, { bigint: true }))) === JSON.stringify(record.pack.identity), "Pack file changed across the campaign");
    check(hash(await readFile(wrapper)) === record.wrapper.sha256, "Shared browser wrapper source changed");
    for (const helper of [...packet.leaseHelpers, packet.sessionBusHelper, packet.emitterPreflight]) check(hash(await readFile(helper.path)) === helper.sha256, `External reviewed helper or receipt changed: ${helper.path}`);
    for (const entry of resources) check(hash(await readFile(resolve(ROOT, entry.path))) === entry.sha256, `Resource changed across the campaign: ${entry.path}`);
    check(!record.wrapperLog.error && !record.wrapperLog.budgetExceeded && record.wrapperExit.code === 0 && record.page.status === "done" && record.page.correctness?.ok === true, "Private browser campaign failed or returned incomplete evidence");
    abort.signal.throwIfAborted();
    record.status = "complete";
  } catch (error) {
    record.status = "failed"; record.error = String(error?.message || error);
    if (outputCreated) await save();
    throw error;
  } finally {
    await cleanupChildren();
    clearTimeout(timeout); clearTimeout(killTimer);
    if (wrapperLog) {
      if (wrapperLog.errored || wrapperLog.destroyed) { if (!wrapperLog.destroyed) wrapperLog.destroy(); }
      else await new Promise((resolveLog) => wrapperLog.end((error) => { if (error) { record.wrapperLog.closeError = String(error); record.status = "failed"; } resolveLog(); }));
      try { const bytes = await readFile(wrapperLogPath); record.wrapperLog.actualRetainedBytes = bytes.length; record.wrapperLog.sha256 = hash(bytes); }
      catch (error) { record.wrapperLog.finalReadError = String(error); record.status = "failed"; }
    }
    await closeHttp();
    if (manifestCreated) await unlink(temporaryManifest).catch((e) => { if (e.code !== "ENOENT") throw e; });
    for (const entry of ownedLinks.reverse()) if (await readlink(entry.path).catch(() => null) === entry.target) await unlink(entry.path);
    if (admittedScope) {
      const members = await scopeMembers(admittedScope), processes = (await Promise.all(members.map(processInfo))).filter(Boolean);
      record.childCleanup.finalControllerBookend = { members, processes };
      if (!members.includes(process.pid) || !processes.some((info) => info.pid === process.pid && allowed(info)) || !members.every((pid) => admittedScope.controllers.some((owner) => owner.pid === pid)) || processes.some((info) => !allowed(info))) {
        record.status = "failed"; record.childCleanup.positiveZeroSurvivors = false;
        record.childCleanup.error = "Owned scope gained an extra process before the final lease proof";
        await holdLeaseForOuterTermination(record.childCleanup);
        throw new Error(record.childCleanup.error);
      }
    }
    if (leaseGuard?.binding) {
      try { record.leaseAfterCleanup = await leaseGuard.check("after-owned-cleanup"); }
      catch (error) { record.status = "failed"; record.cleanupLeaseError = String(error); if (outputCreated) await save(); throw error; }
    }
    if (outputCreated) await save();
    process.off("SIGTERM", stopTerm); process.off("SIGINT", stopInt);
  }
}

run().catch((error) => { console.error(error.message); process.exitCode = 1; });
