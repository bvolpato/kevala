// CPU backend guards: model parity and the Kev state cache in a browser. Start scripts/serve.mjs
// and download the local packs first.
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { BASE_URL, failed, runPage } from "./lib/browser-run.mjs";

const { values, positionals } = parseArgs({ allowPositionals: true, options: {
  browser: { type: "string", default: "firefox" },
  threads: { type: "string", default: "4" },
  flavor: { type: "string", default: "relaxed" },
} });
if (positionals.length > 1) throw new Error("expected one output directory");
const out = resolve(positionals[0] || "tmp/cpu-guard");
mkdirSync(out, { recursive: true });
const opts = new URLSearchParams({ auto: "1", backend: "wasm", pack: "local", threads: values.threads, flavor: values.flavor });
const checks = [
  ["laya", "parity", `/parity.html#${opts}`, "0.03"],
  ["kev", "parity", `/parity-kev.html#${opts}`, "0.011"],
  ["cache", "cache", `/dev/cache-test.html?${opts}`],
];
for (const [name, kind, path, tolerance] of checks) {
  const run = runPage({ result: kind, url: BASE_URL + path, output: resolve(out, `${name}.json`), backend: "wasm", browser: values.browser, maxDp: tolerance, timeoutSeconds: 900 });
  if (failed(run)) {
    console.error(`${name} failed`, run.error || "");
    process.exit(1);
  }
}
console.log("CPU model parity and cache probability checks passed; the native kev_model and kev_cache tests verify cache counters.");
