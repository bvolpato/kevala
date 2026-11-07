import { spawnSync } from "node:child_process";
import { mkdirSync, realpathSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";

export const PROFILE_LOCK = join(homedir(), ".cache/kevala/gpu-profile.lock");

/** Re-execute under one OS-owned lock shared by every checkout and profiler. */
export async function withProfileLock(work, { path = PROFILE_LOCK } = {}) {
  const requested = resolve(path);
  mkdirSync(dirname(requested), { recursive: true });
  const canonical = join(realpathSync(dirname(requested)), basename(requested));
  if (process.env.KEVALA_PROFILE_LOCK_HELD === canonical) return work();
  const child = spawnSync("flock", ["--exclusive", "--nonblock", "--conflict-exit-code", "75", canonical, process.execPath, ...process.argv.slice(1)], {
    stdio: "inherit", env: { ...process.env, KEVALA_PROFILE_LOCK_HELD: canonical },
  });
  if (child.error) throw child.error;
  if (child.status === 75) throw new Error(`Another GPU profile holds ${canonical}; wait for it to finish. The lock is never removed to bypass an active writer.`);
  if (child.signal) throw new Error(`GPU profile terminated by ${child.signal}`);
  process.exitCode = child.status ?? 1;
}
