import { spawnSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";

const out = resolve(process.argv[2] || "tmp/gpu-guard");
const base = process.env.KEVALA_BENCH_URL || "http://127.0.0.1:18086";
mkdirSync(out, { recursive: true });
const checks = [
  ["tiles", "gpu", "/dev/gpu-bench.html?cases=7x132x96,17x132x96,45x132x96,45x132x1024&samples=3&warmups=1"],
  ["kernels", "kernels", "/dev/kernels.html"],
  ["laya", "parity", "/parity.html#backend=webgpu&pack=local", "0.024"],
  ["kev", "parity", "/parity-kev.html#backend=webgpu&pack=local", "0.010"],
];
for (const [name, kind, path, tolerance] of checks) {
  const args = ["run", "scripts/bench-gpu.py", "--result", kind, "--timeout", "600", "--url", base + path, "--output", resolve(out, `${name}.json`)];
  if (tolerance) args.push("--max-dp", tolerance);
  const run = spawnSync("uv", args, { stdio: "inherit", timeout: 650000 });
  if (run.error || run.status !== 0) {
    console.error(`${name} failed`, run.error || "");
    process.exit(1);
  }
}
console.log("GPU numerical guards passed");
