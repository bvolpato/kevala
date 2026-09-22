// Run the two real models serially. Start scripts/serve.mjs and download the local packs first.
import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const out = resolve(process.argv[2] || "tmp/gpu-suite");
const base = process.env.KEVALA_BENCH_URL || "http://127.0.0.1:18086";
mkdirSync(out, { recursive: true });
const models = [];
for (const model of ["laya", "kev-0.8b"]) {
  const file = resolve(out, `${model}.json`);
  const url = `${base}/bench.html#auto&backend=webgpu&pack=local&model=${model}&runs=5&profile=1&unique=1&shapes=0,1,2`;
  const run = spawnSync("uv", ["run", "scripts/bench-gpu.py", "--result", "bench", "--timeout", "600", "--url", url, "--output", file], { encoding: "utf8", timeout: 650000 });
  if (run.error || run.status !== 0) {
    console.error(run.error || run.stderr || run.stdout);
    process.exit(1);
  }
  const result = JSON.parse(readFileSync(file, "utf8"));
  models.push(result);
  console.error(`${model}: ${result.metricMs.toFixed(3)} ms GPU geometric mean`);
}
const metricMs = Math.exp(models.reduce((sum, model) => sum + Math.log(model.metricMs), 0) / models.length);
writeFileSync(resolve(out, "summary.json"), JSON.stringify({ metricMs, models }, null, 2) + "\n");
console.log(metricMs);
