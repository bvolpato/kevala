export function q8GlobalInventoryDue(nowMs, lastGlobalInventoryMs, intervalMs) {
  if (!Number.isFinite(nowMs) || (lastGlobalInventoryMs !== null && !Number.isFinite(lastGlobalInventoryMs)) ||
      !Number.isFinite(intervalMs) || intervalMs <= 0) {
    throw new TypeError("Q8 monitor cadence requires finite monotonic milliseconds and a positive interval");
  }
  return lastGlobalInventoryMs === null || nowMs - lastGlobalInventoryMs >= intervalMs;
}

export function q8KnownChildSample(known, current, expectedCgroup) {
  if (!current || current.startTicks !== known.startTicks) return null;
  if (current.cgroup !== expectedCgroup) {
    const error = new Error(`Q8 known child PID ${known.pid} escaped its exact cgroup`);
    error.scopeMonitorEvidence = {
      pid: known.pid,
      startTicks: known.startTicks,
      expectedCgroup,
      previousCgroup: known.cgroup,
      observedCgroup: current.cgroup,
    };
    throw error;
  }
  return current;
}

export function q8KnownChildIdentitySample(known, current) {
  if (!current || current.startTicks !== known.startTicks) return null;
  return current;
}
