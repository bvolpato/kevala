#!/usr/bin/env node
// usage: node scripts/bench-guardrail.mjs <pack.kevala>
import { loadFile } from "../js/src/node.js";
import { QUESTIONS, assess } from "../app/views/guardrail-policy.js";
import { CASES } from "../benchmarks/guardrail-cases.mjs";

const pack = process.argv[2];
if (!pack) {
  console.error("usage: node scripts/bench-guardrail.mjs <pack.kevala>");
  process.exit(2);
}

const model = await loadFile(pack);
const counts = Object.fromEntries(["allow", "review", "block"].map((label) => [label, { allow: 0, review: 0, block: 0 }]));
const misses = [];
for (const { id, label, prompt } of CASES) {
  const { answers } = model.decide({ prompt }, QUESTIONS);
  const verdict = assess(answers, prompt).verdict;
  counts[label][verdict]++;
  if (verdict !== label) misses.push(`${id}: expected ${label}, got ${verdict}`);
}

console.log(`${model.info.arch} ${model.info.flavor}, ${CASES.length} cases`);
console.table(counts);
for (const miss of misses) console.log(miss);
