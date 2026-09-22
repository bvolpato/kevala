#!/usr/bin/env node
// Runs a .kevala pack through the WebAssembly engine in Node: a smoke test and a single-thread
// benchmark of the exact binary browsers load. usage: node scripts/node-smoke.mjs <pack> [flavor]
import { readFileSync } from "node:fs";
const [pack = "tmp/laya-q8.kevala", flavor = "relaxed"] = process.argv.slice(2);
const { instance } = await WebAssembly.instantiate(readFileSync(`js/src/kevala-${flavor}.wasm`), {});
const x = instance.exports;
x.kevala_init();
const mem = () => new Uint8Array(x.memory.buffer);
const bytes = readFileSync(pack);
let t = performance.now();
const ptr = x.kevala_alloc(bytes.length);
mem().set(bytes, ptr);
const err = () => new TextDecoder().decode(mem().subarray(x.kevala_error_ptr(), x.kevala_error_ptr() + x.kevala_error_len()));
const out = () => new TextDecoder().decode(mem().subarray(x.kevala_out_ptr(), x.kevala_out_ptr() + x.kevala_out_len()));
if (x.kevala_engine_load(ptr, bytes.length)) throw new Error(err());
console.log(`loaded ${pack} in ${(performance.now() - t).toFixed(0)} ms (${flavor})`);
const decide = (req) => {
  const b = new TextEncoder().encode(JSON.stringify(req));
  const p = x.kevala_alloc(b.length);
  mem().set(b, p);
  const rc = x.kevala_decide(p, b.length);
  x.kevala_free(p, b.length);
  if (rc) throw new Error(err());
  return JSON.parse(out());
};
const req = {
  state: { from: "user@acme.com", subject: "Duplicate charge on invoice #4411", body: "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan." },
  questions: {
    department: { type: "choice", instructions: "Which department should handle this request?", criteria: { billing: "invoices, payments, refunds", technical: "bugs, outages, system errors", sales: "pricing, new contracts", other: "everything else" } },
    churn_risk: { type: "noul", instructions: "Does the user threaten to cancel or leave?" },
  },
};
decide(req);
const ms = [];
let r;
for (let i = 0; i < 5; i++) {
  t = performance.now();
  r = decide(req);
  ms.push(performance.now() - t);
}
ms.sort((a, b) => a - b);
console.log(JSON.stringify(r[0].answers));
console.log(`tokens ${r[0].usage.input_tokens}  p50 ${ms[2].toFixed(1)} ms  min ${ms[0].toFixed(1)} ms`);
