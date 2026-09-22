import { spawnSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

const { values, positionals } = parseArgs({ allowPositionals: true, options: {
  browser: { type: "string", default: "firefox" },
  threads: { type: "string", default: "4" },
  flavor: { type: "string", default: "relaxed" },
} });
if (positionals.length > 1) throw new Error("expected one output directory");
const out = resolve(positionals[0] || "tmp/cpu-guard");
const base = process.env.KEVALA_BENCH_URL || "http://127.0.0.1:18086";
mkdirSync(out, { recursive: true });
const opts = new URLSearchParams({ auto: "1", backend: "wasm", pack: "local", threads: values.threads, flavor: values.flavor });
const checks = [
  ["laya", "parity", `/parity.html#${opts}`, "0.03"],
  ["kev", "parity", `/parity-kev.html#${opts}`, "0.011"],
  ["cache", "cache", `/dev/cache-test.html?${opts}`],
];
for (const [name, kind, path, tolerance] of checks) {
  const args = ["run", "scripts/bench-gpu.py", "--backend", "wasm", "--browser", values.browser,
    "--result", kind, "--timeout", "900", "--url", base + path, "--output", resolve(out, `${name}.json`)];
  if (tolerance) args.push("--max-dp", tolerance);
  const run = spawnSync("uv", args, { stdio: "inherit", timeout: 950000 });
  if (run.error || run.status !== 0) {
    console.error(`${name} failed`, run.error || "");
    process.exit(1);
  }
}
console.log("CPU model parity and cache probability checks passed; native kev_cache tests verify cache counters.");
