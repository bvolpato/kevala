import assert from 'node:assert/strict';
import {readdirSync, realpathSync, lstatSync, fstatSync, readlinkSync, openSync, readSync, closeSync} from 'node:fs';
import {deviceParts, parseFdinfo, validatePrimaryFd, verifyObservation} from './lease-parse.mjs';

function boundedText(path, limit) {
  const fd = openSync(path, 'r'), chunks = []; let bytes = 0;
  try {
    for (;;) {
      const block = Buffer.alloc(Math.min(16384, limit - bytes + 1)), n = readSync(fd, block, 0, block.length, null);
      if (!n) break; chunks.push(block.subarray(0, n)); bytes += n;
      if (bytes > limit) { const e = Error('bounded proc evidence exceeded limit: ' + path); e.rawPrefix = Buffer.concat(chunks).toString('utf8'); throw e; }
    }
  } finally { closeSync(fd); }
  return Buffer.concat(chunks).toString('utf8');
}
function processIdentity() {
  const raw = boundedText('/proc/self/stat', 16384), i = raw.lastIndexOf(')'), fields = raw.slice(i + 1).trim().split(/\s+/);
  assert.equal(raw.slice(0, raw.indexOf(' ')), String(process.pid)); assert(/^\d+$/.test(fields[19]));
  return {pid: String(process.pid), startTicks: fields[19], rawStat: raw};
}
const statIdentity = (s, fd) => ({...(fd === undefined ? {} : {fd}), device: String(s.dev), inode: String(s.ino), regular: s.isFile()});
function canonicalIdentity(path) {
  const realpath = realpathSync(path), s = lstatSync(path, {bigint: true});
  return {path, realpath, ...statIdentity(s), symlink: s.isSymbolicLink()};
}
function side(path, fd) {
  return {process: processIdentity(), canonical: canonicalIdentity(path), fd: statIdentity(fstatSync(fd, {bigint: true}), fd), fdTarget: readlinkSync(`/proc/self/fd/${fd}`)};
}
function observation(binding) {
  const v = {};
  try {
    v.before = side(binding.canonicalPath, binding.fd);
    v.fdinfoBefore = boundedText(`/proc/self/fdinfo/${binding.fd}`, 16384);
    v.globalRaw = boundedText('/proc/locks', 1048576);
    v.fdinfoAfter = boundedText(`/proc/self/fdinfo/${binding.fd}`, 16384);
    v.after = side(binding.canonicalPath, binding.fd);
    return v;
  } catch (error) { error.partialObservation = v; throw error; }
}
export function createLeaseGuard({canonicalPath, recordEvidence}) {
  assert.equal(typeof recordEvidence, 'function'); assert.equal(typeof canonicalPath, 'string');
  let binding = null, sequence = 0, pending = false;
  const finish = async (label, collect, commit) => {
    assert(!pending, 'await the preceding lease guard operation'); pending = true;
    try {
      const evidence = {sequence: sequence++, label, at: new Date().toISOString(), pid: process.pid};
      let proof, failure;
      try { proof = collect(evidence); evidence.passed = true; }
      catch (error) { failure = error; evidence.passed = false; evidence.error = {message: String(error), code: error.code ?? null,
        rawPrefix: error.rawPrefix ?? null, partialObservation: error.partialObservation ?? null}; }
      // Persist full bounded raw evidence before any validation exception leaves this helper.
      const evidenceRef = await recordEvidence(evidence) ?? null;
      if (failure) { failure.leaseEvidenceRef = evidenceRef; throw failure; }
      if (commit) commit(proof);
      return {...proof, evidenceRef};
    } finally { pending = false; }
  };
  return {
    async bind(label = 'bind-inherited-lease') {
      assert(binding === null, 'lease binding may not be replaced');
      const result = await finish(label, evidence => {
        const canonical = canonicalIdentity(canonicalPath), owner = processIdentity();
        evidence.canonical = canonical; evidence.process = owner;
        assert(canonical.regular && !canonical.symlink && canonical.realpath === canonicalPath);
        const base = {canonicalPath, pid: owner.pid, startTicks: owner.startTicks, device: canonical.device, inode: canonical.inode, ...deviceParts(canonical.device)};
        evidence.discoveryGlobalRaw = boundedText('/proc/locks', 1048576);
        const names = readdirSync('/proc/self/fd').filter(n => /^\d+$/.test(n)); evidence.fdInventory = names;
        assert(names.length <= 128, 'descriptor inventory cap exceeded');
        const candidates = []; evidence.canonicalDescriptors = candidates;
        for (const name of names) {
          const fd = Number(name); assert(Number.isSafeInteger(fd));
          let s; try { s = fstatSync(fd, {bigint: true}); } catch (error) { if (error.code === 'EBADF') continue; throw error; }
          if (String(s.dev) !== base.device || String(s.ino) !== base.inode) continue;
          const item = {fd, stat: statIdentity(s, fd)}; candidates.push(item);
          item.rawFdinfo = boundedText(`/proc/self/fdinfo/${fd}`, 16384);
          item.target = readlinkSync(`/proc/self/fd/${fd}`); item.parsedFdinfo = parseFdinfo(item.rawFdinfo);
          if (item.parsedFdinfo.locks.length) validatePrimaryFd(item.parsedFdinfo, base);
        }
        const held = candidates.filter(c => {
          try { validatePrimaryFd(c.parsedFdinfo, base); return c.stat.regular && c.target === canonicalPath; } catch { return false; }
        });
        assert.equal(held.length, 1, 'one unique inherited descriptor must prove exact current-PID ownership');
        const c = held[0], proposed = Object.freeze({...base, fd: c.fd, mntId: c.parsedFdinfo.mntId, flags: c.parsedFdinfo.flags});
        evidence.binding = {...proposed};
        evidence.observation = observation(proposed);
        const proof = verifyObservation(evidence.observation, proposed); evidence.proof = proof;
        return {proposed, proof};
      }, result => { binding = result.proposed; });
      return {...result.proof, evidenceRef: result.evidenceRef};
    },
    async check(label = 'check-bound-lease') {
      assert(binding !== null, 'bind the inherited descriptor first');
      return finish(label, evidence => { evidence.binding = {...binding}; evidence.observation = observation(binding); const proof = verifyObservation(evidence.observation, binding); evidence.proof = proof; return proof; });
    },
    get binding() { return binding && {...binding}; },
  };
}
