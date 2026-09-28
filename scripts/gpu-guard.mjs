import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const out = resolve(process.argv[2] || "tmp/gpu-guard");
const base = process.env.KEVALA_BENCH_URL || "http://127.0.0.1:18086";
mkdirSync(out, { recursive: true });
const registry = readFileSync(new URL("../crates/kevala/src/gpu.rs", import.meta.url), "utf8").match(/pub const KERNELS:.*?= &\[([\s\S]*?)\];/)?.[1];
const kernels = [...(registry || "").matchAll(/"([a-z0-9_]+)"/g)].map((match) => match[1]);
if (!kernels.length) throw new Error("GPU kernel registry is empty");
const compilePath = `/dev/kernel-compile-check.html?kernels=${encodeURIComponent(kernels.join(","))}`;
const checks = [
  ["compile", "gpu", compilePath],
  ["compile-baseline", "gpu", `${compilePath}&baseline=1`],
  ["activations", "gpu", "/dev/activation-check.html"],
  ["tiles", "gpu", "/dev/gpu-bench.html?cases=7x132x96,17x132x96,45x132x96,45x132x1024&samples=3&warmups=1"],
  ["row-tiles", "gpu", "/dev/gpu-bench.html?cases=65x68x96,80x132x96,96x70x160,129x68x96,140x132x96,144x70x160&kernels=matmul_h@256:4,matmul_h@256:3&samples=3&warmups=1", undefined, ["shaderF16", "timestampQuery"]],
  ["ordered-sums", "gpu", "/dev/kernel-bench.html?kernels=norm,kev_rms,kev_aprep,gemma4_rms,gemma4_ple&tokens=1,47,128&seed=42&samples=3&warmups=1", undefined, ["subgroup32", "timestampQuery"]],
  ...[2048, 2560, 4096].map((hidden) => [`ordered-rms-${hidden}`, "gpu", `/dev/kernel-bench.html?kernels=kev_rms&kevHidden=${hidden}&tokens=1,47,128&seed=42&samples=3&warmups=1`, undefined, ["subgroup32", "timestampQuery"]]),
  ["kernels", "kernels", "/dev/kernels.html"],
  ["gemma-gelu", "gpu", "/dev/gemma-gelu-check.html"],
  ["gemma-attention", "gpu", "/dev/gemma-attn-bench.html?samples=1&warmups=1"],
  ["gemma-rms", "gpu", "/dev/gemma-rms-check.html?samples=1&warmups=1"],
  ["gemma-rms-subgroups", "gpu", "/dev/gemma-rms-check.html?subgroups=1&samples=1&warmups=1", undefined, ["subgroup32", "timestampQuery"]],
  ["gemma-tail", "gpu", "/dev/gemma-tail-check.html"],
  ["recur4-shared", "gpu", "/dev/recur-check.html?candidate=current&subgroups=0&actual=1&samples=1&warmups=1", undefined, ["timestampQuery"]],
  ["recur8-shared", "gpu", "/dev/recur-check.html?candidate=recur8&subgroups=0&actual=1&samples=1&warmups=1", undefined, ["timestampQuery"]],
  ["recur16-shared", "gpu", "/dev/recur-check.html?candidate=recur16&subgroups=0&actual=1&samples=1&warmups=1", undefined, ["timestampQuery"]],
  ["kev-attention", "gpu", "/dev/kev-attn-bench.html?gateTails=1&samples=1&warmups=1", undefined, ["timestampQuery", "subgroup32"]],
  ["recur4-subgroups", "gpu", "/dev/recur-check.html?candidate=current&subgroups=1&actual=1&samples=1&warmups=1", undefined, ["timestampQuery", "subgroup4"]],
  ["recur8-subgroups", "gpu", "/dev/recur-check.html?candidate=recur8&subgroups=1&actual=1&samples=1&warmups=1", undefined, ["timestampQuery", "subgroup8"]],
  ["recur16-subgroups", "gpu", "/dev/recur-check.html?candidate=recur16&subgroups=1&actual=1&samples=1&warmups=1", undefined, ["timestampQuery", "subgroup16"]],
  ["attention", "gpu", "/dev/attn-bench.html?cases=1:0,15:0,16:0,17:0,31:0,32:0,33:0,63:0,64:0,65:0,127:0,128:0,129:0,47:0,512:0,512:64,150/40:64&kernels=attention,attention_subgroup,attention_tile,attention_tile_shared&samples=1&warmups=1&checks=3"],
  ["laya", "parity", "/parity.html#auto&backend=webgpu&pack=local", "0.024"],
  ["kev", "parity", "/parity-kev.html#backend=webgpu&pack=local", "0.011"],
  ["cache", "cache", "/dev/cache-test.html?backend=webgpu"],
];
let capabilities;
for (const [name, kind, path, tolerance, required = []] of checks) {
  const missing = required.filter((capability) => !capabilities?.[capability]);
  if (missing.length) {
    writeFileSync(resolve(out, `${name}.json`), JSON.stringify({ status: "skipped", missingCapabilities: missing }, null, 2) + "\n");
    console.log(`${name} skipped: missing ${missing.join(", ")}`);
    continue;
  }
  const args = ["run", "scripts/bench-gpu.py", "--result", kind, "--timeout", "600", "--url", base + path, "--output", resolve(out, `${name}.json`)];
  if (process.env.KEVALA_BENCH_CDP) args.push("--cdp-default-context");
  if (tolerance) args.push("--max-dp", tolerance);
  const run = spawnSync("uv", args, { stdio: "inherit", timeout: 650000 });
  if (run.error || run.status !== 0) {
    console.error(`${name} failed`, run.error || "");
    process.exit(1);
  }
  if (name === "compile") {
    capabilities = JSON.parse(readFileSync(resolve(out, "compile.json"), "utf8")).capabilities;
    if (!["timestampQuery", "shaderF16", "subgroup4", "subgroup8", "subgroup16", "subgroup32"].every((capability) => typeof capabilities?.[capability] === "boolean")) throw new Error("compile guard did not report device capabilities");
  }
}
console.log("All supported GPU numerical guards passed");
