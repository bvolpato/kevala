// CPU choices are private to the browser, WASM build family, and model configuration.
// Bump the revision when CPU kernels or the calibration procedure change.
const REVISION = 1;
const MAX_AGE = 7 * 24 * 60 * 60 * 1000;
const CACHE = "kevala-cpu-tuning-v1";
export const CPU_KERNELS = ["2x4", "4x4"];

export function cpuOptions({ cpuKernel = "auto", threads = "auto", retune = false } = {}) {
  if (cpuKernel !== "auto" && !CPU_KERNELS.includes(cpuKernel)) throw new Error('cpuKernel must be "auto", "2x4", or "4x4"');
  if (threads !== "auto" && (!Number.isInteger(threads) || threads < 1 || threads > 16)) {
    throw new Error('threads must be "auto" or an integer from 1 to 16');
  }
  if (typeof retune !== "boolean") throw new Error("retune must be a boolean");
  return { cpuKernel, threads, retune };
}

export function threadCandidates(hardware, limit) {
  const cap = Math.max(1, Math.min(16, Math.floor(hardware) || 1, Math.floor(limit) || 1));
  const counts = [1];
  for (let n = 2; n <= cap; n *= 2) counts.push(n);
  if (counts.at(-1) !== cap) counts.push(cap);
  return counts;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

/** Prefer fewer workers when their measured latency is within 10% of the fastest. */
export function selectThreadCount(measurements) {
  const scores = measurements.map(({ threads, samples }) => ({
    threads,
    ms: Math.exp(samples.reduce((sum, shape) => sum + Math.log(median(shape)), 0) / samples.length),
  })).filter(({ ms }) => Number.isFinite(ms) && ms > 0);
  if (!scores.length) throw new Error("CPU thread probe produced no usable timings");
  const fastest = Math.min(...scores.map(({ ms }) => ms));
  return scores.filter(({ ms }) => ms <= fastest * 1.1).sort((a, b) => a.threads - b.threads)[0].threads;
}

export async function tuningKey({ flavor, base, config, options, hardware, userAgent }) {
  const data = JSON.stringify([REVISION, flavor, base, config, options.cpuKernel, options.threads, hardware, userAgent]);
  const hash = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(data));
  return new URL(`./.kevala-cpu-tuning/${Array.from(new Uint8Array(hash), (b) => b.toString(16).padStart(2, "0")).join("")}`, base).href;
}

export function validProfile(profile, counts, options, now = Date.now()) {
  return profile?.revision === REVISION && profile.complete === true && Number.isFinite(profile.created) && profile.created <= now && now - profile.created < MAX_AGE &&
    counts.includes(profile.threads) && CPU_KERNELS.includes(profile.kernel) &&
    (options.cpuKernel === "auto" || options.cpuKernel === profile.kernel);
}

export async function readTuning(key, counts, options) {
  try {
    const response = await (await caches.open(CACHE)).match(key);
    const profile = response && await response.json();
    return validProfile(profile, counts, options) ? profile : null;
  } catch {
    return null; // Storage may be unavailable or full; tuning still works for this load.
  }
}

export async function writeTuning(key, profile) {
  try {
    await (await caches.open(CACHE)).put(key, new Response(JSON.stringify({ ...profile, revision: REVISION, created: Date.now() }), {
      headers: { "content-type": "application/json" },
    }));
  } catch {}
}

export async function clearTuning() {
  try {
    await caches.delete(CACHE);
  } catch {}
}
