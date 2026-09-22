import { spawnSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";

const out = resolve(process.argv[2] || "tmp/gpu-guard");
const base = process.env.KEVALA_BENCH_URL || "http://127.0.0.1:18086";
mkdirSync(out, { recursive: true });
const checks = [
  ["tiles", "gpu", "/dev/gpu-bench.html?cases=7x132x96,17x132x96,45x132x96,45x132x1024&samples=3&warmups=1"],
  ["kernels", "kernels", "/dev/kernels.html"],
  ["attention", "gpu", "/dev/attn-bench.html?cases=1:0,15:0,16:0,17:0,31:0,32:0,33:0,63:0,64:0,65:0,127:0,128:0,129:0,47:0,512:0,512:64,150/40:64&kernels=attention,attention_subgroup,attention_tile,attention_tile_shared&samples=1&warmups=1&checks=3"],
  ["laya", "parity", "/parity.html#auto&backend=webgpu&pack=local", "0.024"],
  ["kev", "parity", "/parity-kev.html#backend=webgpu&pack=local", "0.011"],
  ["cache", "cache", "/dev/cache-test.html?backend=webgpu"],
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
