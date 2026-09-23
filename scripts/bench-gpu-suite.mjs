#!/usr/bin/env node
// Run four model families serially on the same GPU.
import { spawnSync } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const [base = "http://127.0.0.1:18099/", output = "tmp/shared-gpu/current"] = process.argv.slice(2);
const destination = resolve(root, output);
await mkdir(destination, { recursive: true });
const models = ["laya", "kev-0.8b", "semif-qwen3.5-2b", "gemma-4-e2b"];
const results = [];
for (const model of models) {
  const url = new URL("bench.html", base.endsWith("/") ? base : `${base}/`);
  url.hash = `auto&backend=webgpu&pack=local&model=${model}&profile=1&runs=5&warmups=2&unique=1&shapes=0,1,2`;
  const path = resolve(destination, `${model}.json`);
  const child = spawnSync("uv", ["run", "scripts/bench-gpu.py", "--result", "bench", "--timeout", "600", "--url", url.href, "--output", path], { cwd: root, stdio: "inherit" });
  if (child.error) throw child.error;
  if (child.status !== 0) throw new Error(`${model}: benchmark exited ${child.status ?? child.signal}`);
  const result = JSON.parse(await readFile(path, "utf8"));
  if (result.model !== model || result.backend !== "webgpu" || !(result.metricMs > 0) || !Number.isFinite(result.metricMs)) {
    throw new Error(`${model}: invalid benchmark result`);
  }
  results.push({ model, metricMs: result.metricMs, source: path, cases: result.cases.map((c) => ({ tokens: c.last.tokens, wallP50Ms: c.p50 })) });
}
const metrics = { gpu_ms: Math.exp(results.reduce((sum, r) => sum + Math.log(r.metricMs), 0) / results.length) };
for (const { model, metricMs } of results) metrics[`${model}_ms`] = metricMs;
await writeFile(resolve(destination, "summary.json"), `${JSON.stringify({ base, metric: "geometric mean of GPU case medians across four model families", metrics, models: results }, null, 2)}\n`);
console.log(JSON.stringify(metrics));
