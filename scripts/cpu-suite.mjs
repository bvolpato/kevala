// Time Laya and Kev-0.8B on the CPU backend serially. Start scripts/serve.mjs and download the
// local packs first.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { BASE_URL, failed, geometricMean, runPage } from "./lib/browser-run.mjs";

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
mkdirSync(out, { recursive: true });
const results = [];
for (const model of models) {
  const output = resolve(out, `${model}.json`);
  const hash = new URLSearchParams({ auto: "1", backend: "wasm", pack: "local", model,
    threads: values.threads, flavor: values.flavor, runs: values.runs, warmups: values.warmups,
    unique: "1", shapes: values.shapes });
  const run = runPage({ result: "latency", url: `${BASE_URL}/bench.html#${hash}`, output, backend: "wasm", browser: values.browser, timeoutSeconds: 900, stdio: "pipe" });
  if (failed(run)) {
    console.error(run.error || run.stderr || run.stdout);
    process.exit(1);
  }
  const result = JSON.parse(readFileSync(output, "utf8"));
  if (result.flavor !== values.flavor) throw new Error(`expected ${values.flavor}, got ${result.flavor}`);
  results.push(result);
  console.error(`${model}: ${result.metricMs.toFixed(3)} ms, ${result.backend}, ${result.threads} workers`);
}
const medians = results.flatMap((result) => result.p50CaseMediansMs);
const metricMs = geometricMean(medians);
writeFileSync(resolve(out, "summary.json"), JSON.stringify({ metricMs, options: values, models: results }, null, 2) + "\n");
console.log(metricMs);
