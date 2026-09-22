import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    browser: { type: "string", default: "firefox" },
    threads: { type: "string", default: "4" },
    flavor: { type: "string", default: "relaxed" },
    runs: { type: "string", default: "3" },
    warmups: { type: "string", default: "1" },
    models: { type: "string", default: "laya,kev-0.8b" },
    shapes: { type: "string", default: "0,1,2" },
  },
});
if (positionals.length > 1) throw new Error("expected one output directory");
const models = values.models.split(",");
if (!models.length || models.some((m) => !["laya", "kev-0.8b"].includes(m))) throw new Error("unknown model");
const out = resolve(positionals[0] || "tmp/cpu-suite");
const base = process.env.KEVALA_BENCH_URL || "http://127.0.0.1:18086";
mkdirSync(out, { recursive: true });
const results = [];
for (const model of models) {
  const output = resolve(out, `${model}.json`);
  const hash = new URLSearchParams({ auto: "1", backend: "wasm", pack: "local", model,
    threads: values.threads, flavor: values.flavor, runs: values.runs, warmups: values.warmups,
    unique: "1", shapes: values.shapes });
  const args = ["run", "scripts/bench-gpu.py", "--backend", "wasm", "--browser", values.browser,
    "--result", "latency", "--timeout", "900", "--url", `${base}/bench.html#${hash}`, "--output", output];
  const run = spawnSync("uv", args, { encoding: "utf8", timeout: 950000 });
  if (run.error || run.status !== 0) {
    console.error(run.error || run.stderr || run.stdout);
    process.exit(1);
  }
  const result = JSON.parse(readFileSync(output, "utf8"));
  if (result.flavor !== values.flavor) throw new Error(`expected ${values.flavor}, got ${result.flavor}`);
  results.push(result);
  console.error(`${model}: ${result.metricMs.toFixed(3)} ms, ${result.backend}, ${result.threads} workers`);
}
const medians = results.flatMap((result) => result.p50CaseMediansMs);
const metricMs = Math.exp(medians.reduce((sum, ms) => sum + Math.log(ms), 0) / medians.length);
writeFileSync(resolve(out, "summary.json"), JSON.stringify({ metricMs, options: values, models: results }, null, 2) + "\n");
console.log(metricMs);
