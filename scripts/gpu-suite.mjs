// Run Laya and Kev-0.8B serially. Start scripts/serve.mjs and download the local packs first.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { BASE_URL, failed, geometricMean, runPage } from "./lib/browser-run.mjs";

const positional = process.argv.slice(2);
const outArg = positional[0]?.startsWith("--") ? undefined : positional.shift();
const wall = positional.includes("--wall");
// --baseline: no optional GPU features and default limits (the kernel paths of browsers without subgroups)
const baseline = positional.includes("--baseline");
const submitArg = positional.find((arg) => arg.startsWith("--submit="));
const submit = submitArg?.slice("--submit=".length) || "await";
const unknown = positional.filter((arg) => !["--wall", "--baseline", "--submit=await", "--submit=split", "--submit=none"].includes(arg));
if (unknown.length) {
  console.error(`unknown option: ${unknown.join(" ")}`);
  process.exit(2);
}
const out = resolve(outArg || "tmp/gpu-suite");
mkdirSync(out, { recursive: true });
const resultKind = wall ? "latency" : "bench";
const profile = wall ? 0 : 1;
const modeLabel = wall ? "wall-clock" : "GPU-profiled";
const models = [];
for (const model of ["laya", "kev-0.8b"]) {
  const file = resolve(out, `${model}.json`);
  const url = `${BASE_URL}/bench.html#auto&backend=webgpu&pack=local&model=${model}&runs=5&profile=${profile}&unique=1&shapes=0,1,2&submit=${submit}${baseline ? "&baseline=1" : ""}`;
  const run = runPage({ result: resultKind, url, output: file, stdio: "pipe" });
  if (failed(run)) {
    console.error(run.error || run.stderr || run.stdout);
    process.exit(1);
  }
  const result = JSON.parse(readFileSync(file, "utf8"));
  models.push(result);
  console.error(`${model}: ${result.metricMs.toFixed(3)} ms ${modeLabel} geometric mean`);
}
const metricMs = geometricMean(models.map((model) => model.metricMs));
writeFileSync(resolve(out, "summary.json"), JSON.stringify({ metricMs, mode: modeLabel, result: resultKind, submit, models }, null, 2) + "\n");
console.log(metricMs);
