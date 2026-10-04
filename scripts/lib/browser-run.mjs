// What the suite and guard runners share: where the repository and its dev server are, and how to
// run one harness page in a browser through scripts/bench-gpu.py.
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

/** The repository root. The runners work from any directory. */
export const ROOT = fileURLToPath(new URL("../../", import.meta.url));

/** Where `node scripts/serve.mjs . --port 18086` serves the checkout. */
export const BASE_URL = process.env.KEVALA_BENCH_URL || "http://127.0.0.1:18086";

/**
 * Runs one page and writes its validated result to `output`. Returns the finished child process;
 * `failed(child)` says whether it failed.
 *
 * The browser is `browser`, else `KEVALA_BENCH_BROWSER`, else the default of bench-gpu.py.
 * `KEVALA_BENCH_CDP` attaches to a running Chrome instead (see bench-gpu.py).
 */
export function runPage({ result, url, output, backend, browser = process.env.KEVALA_BENCH_BROWSER, maxDp, timeoutSeconds = 600, cdpDefaultContext = false, stdio = "inherit" }) {
  const args = ["run", "scripts/bench-gpu.py", "--result", result, "--timeout", String(timeoutSeconds), "--url", url, "--output", output];
  if (backend) args.push("--backend", backend);
  if (browser) args.push("--browser", browser);
  if (maxDp) args.push("--max-dp", String(maxDp));
  if (cdpDefaultContext && process.env.KEVALA_BENCH_CDP) args.push("--cdp-default-context");
  return spawnSync("uv", args, { cwd: ROOT, stdio, encoding: "utf8", timeout: (timeoutSeconds + 50) * 1000 });
}

export const failed = (child) => Boolean(child.error) || child.status !== 0;

export function geometricMean(values) {
  return Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0) / values.length);
}
