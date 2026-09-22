#!/usr/bin/env node
// Single-thread WebAssembly benchmark of any .kevala pack, the exact binary browsers run.
// usage: node scripts/bench-node.mjs <pack> [flavor] [runs] [--warm]
// Each run starts its state with a run number, so a model with a state cache (Kev) does the full
// pass every time; --warm repeats the same state instead.
import { loadFile } from "../js/src/node.js";

const args = process.argv.slice(2).filter((a) => a !== "--warm");
const warm = process.argv.includes("--warm");
const [pack = "tmp/laya-q8.kevala", flavor, runs = "5"] = args;
const w = await loadFile(pack, { flavor });
const words = "the customer wrote in about a recurring charge they did not recognise and wants a refund today ";
const Q = { type: "noul", instructions: "Is the customer asking for money back?" };
const shapes = [
  ["1 question, ~40 tokens", "Refund the duplicate charge today or we cancel.", 1],
  ["1 question, ~128 tokens", words.repeat(6), 1],
  ["1 question, ~512 tokens", words.repeat(28), 1],
  ["5 questions, one state", words.repeat(2), 5],
];
console.log(`${w.info.arch} wasm-${w.info.flavor}, 1 thread${warm ? ", repeated state" : ""}`);
for (const [name, state, nq] of shapes) {
  const questions = Object.fromEntries(Array.from({ length: nq }, (_, i) => [`q${i}`, Q]));
  const ms = [];
  let tokens = 0;
  for (let i = 0; i < Number(runs) + 1; i++) {
    const t = performance.now();
    const r = w.decide(warm ? state : `Ticket ${i}. ${state}`, questions);
    if (i) ms.push(performance.now() - t);
    tokens = r.usage.input_tokens;
  }
  ms.sort((a, b) => a - b);
  console.log(`${name.padEnd(26)} ${String(tokens).padStart(4)} tok  p50 ${ms[ms.length >> 1].toFixed(0).padStart(6)} ms`);
}
