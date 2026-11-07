// Pure parsing/validation. No filesystem, process inspection or lock operations.
import assert from 'node:assert/strict';

const decimal = value => { assert(/^(?:0|[1-9]\d*)$/.test(value)); return BigInt(value).toString(); };
export function deviceParts(device) {
  const d = BigInt(device);
  return {major: String(d >> 8n & 0xfffn | d >> 32n & 0xfffff000n),
    minor: String(d & 0xffn | d >> 12n & 0xffffff00n)};
}
export function parseLockRow(raw) {
  const f = raw.trim().split(/\s+/); assert(/^\d+:$/.test(f[0]));
  const id = f.shift().slice(0, -1), blocked = f[0] === '->'; if (blocked) f.shift();
  assert.equal(f.length, 7, 'lock row must have exact kernel field count');
  const [kind, classification, mode, pid, dev, start, end] = f;
  assert([kind, classification, mode].every(s => /^[A-Z_+]+$/.test(s)));
  assert(/^(?:-1|0|[1-9]\d*)$/.test(pid));
  const d = dev.split(':'); assert.equal(d.length, 3);
  assert(d.slice(0, 2).every(s => /^[0-9a-fA-F]+$/.test(s)));
  assert(/^(?:0|[1-9]\d*)$/.test(start)); assert(end === 'EOF' || /^(?:0|[1-9]\d*)$/.test(end));
  return {id, blocked, kind, classification, mode, pid, major: BigInt('0x' + d[0]).toString(),
    minor: BigInt('0x' + d[1]).toString(), inode: decimal(d[2]), start, end, raw};
}
export function parseGlobal(text) {
  assert(text === '' || text.endsWith('\n'), 'truncated global lock text');
  return text.split('\n').filter(s => s.trim()).map(parseLockRow);
}
export function parseFdinfo(text) {
  assert(text.endsWith('\n'), 'truncated fdinfo text');
  const fields = {}, locks = [], extra = [];
  for (const line of text.split('\n').filter(Boolean)) {
    if (line.startsWith('lock:')) { locks.push(parseLockRow(line.slice(5).trim())); continue; }
    const m = /^(pos|flags|mnt_id|ino):\s*(\S+)\s*$/.exec(line);
    if (!m) { assert(!/^(pos|flags|mnt_id|ino):/.test(line), 'malformed fdinfo header'); extra.push(line); continue; }
    assert(!Object.hasOwn(fields, m[1]), 'duplicate fdinfo header'); fields[m[1]] = m[2];
  }
  for (const name of ['pos', 'flags', 'mnt_id', 'ino']) assert(Object.hasOwn(fields, name), 'missing fdinfo ' + name);
  assert(/^[0-7]+$/.test(fields.flags), 'fdinfo flags must be octal'); decimal(fields.pos);
  return {inode: decimal(fields.ino), mntId: decimal(fields.mnt_id), flags: BigInt('0o' + fields.flags).toString(),
    flagsOctal: fields.flags, locks, extra, raw: text};
}
export function sameFile(row, binding) {
  return row.major === binding.major && row.minor === binding.minor && row.inode === binding.inode;
}
export function exactOwnLock(row, binding) {
  return !row.blocked && row.kind === 'FLOCK' && row.classification === 'ADVISORY' && row.mode === 'WRITE'
    && row.pid === binding.pid && sameFile(row, binding) && row.start === '0' && row.end === 'EOF';
}
export function validatePrimaryFd(info, binding) {
  assert.equal(info.inode, binding.inode, 'fdinfo inode does not match bound fstat');
  if (binding.mntId !== undefined) assert.equal(info.mntId, binding.mntId, 'bound mount ID changed');
  if (binding.flags !== undefined) assert.equal(info.flags, binding.flags, 'bound descriptor flags changed');
  assert.equal(info.locks.length, 1, 'one exact own-fd lock row required; fdinfo duplicates are never deduplicated');
  assert(exactOwnLock(info.locks[0], binding), 'fdinfo must prove current-PID FLOCK ADVISORY WRITE 0..EOF');
  return info.locks[0];
}
export function verifyObservation(value, binding) {
  for (const side of ['before', 'after']) {
    const v = value[side]; assert.equal(v.process.pid, binding.pid); assert.equal(v.process.startTicks, binding.startTicks);
    assert.equal(v.canonical.path, binding.canonicalPath); assert.equal(v.canonical.realpath, binding.canonicalPath);
    assert(v.canonical.regular && !v.canonical.symlink); assert(v.fd.regular); assert.equal(v.fd.fd, binding.fd);
    for (const s of [v.canonical, v.fd]) { assert.equal(s.device, binding.device); assert.equal(s.inode, binding.inode); }
    assert.equal(v.fdTarget, binding.canonicalPath, 'bound fd target changed/deleted');
  }
  const first = parseFdinfo(value.fdinfoBefore), last = parseFdinfo(value.fdinfoAfter);
  validatePrimaryFd(first, binding); validatePrimaryFd(last, binding);
  const global = parseGlobal(value.globalRaw), canonicalGranted = global.filter(row => !row.blocked && sameFile(row, binding));
  const contradictions = canonicalGranted.filter(row => !exactOwnLock(row, binding));
  assert.equal(contradictions.length, 0, 'contradictory canonical granted lock; fail closed');
  const own = canonicalGranted.filter(row => exactOwnLock(row, binding));
  return {binding: {...binding}, primaryFdinfoRowsBefore: first.locks, primaryFdinfoRowsAfter: last.locks,
    globalCanonicalGrantedRows: canonicalGranted, globalOwnRowCount: own.length,
    globalCanonicalBlockedRows: global.filter(row => row.blocked && sameFile(row, binding)),
    globalEnumerationAnomaly: own.length !== 1, anomalyCauseEstablished: false,
    exactInheritedFdOwnershipProved: true, globalRowsAreDiagnosticsNotUniqueCountAuthority: true};
}
