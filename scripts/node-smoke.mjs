#!/usr/bin/env node
// Runs a .kevala pack through the published Node entry point (js/src/node.js): the exact
// WebAssembly binary browsers load, in one instance. It fails unless every question gets a
// well-formed answer. For timings, use scripts/bench-node.mjs.
//
// usage: node scripts/node-smoke.mjs [pack] [flavor]
import { loadFile } from "../js/src/node.js";

const [pack = "tmp/laya-q8.kevala", flavor] = process.argv.slice(2);
const started = performance.now();
const model = await loadFile(pack, { flavor });
console.log(`loaded ${pack} in ${(performance.now() - started).toFixed(0)} ms (${model.info.arch}, wasm-${model.info.flavor})`);

const state = { from: "user@acme.com", subject: "Duplicate charge on invoice #4411", body: "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan." };
const questions = {
  department: { type: "choice", instructions: "Which department should handle this request?", criteria: { billing: "invoices, payments, refunds", technical: "bugs, outages, system errors", sales: "pricing, new contracts", other: "everything else" } },
  churn_risk: { type: "noul", instructions: "Does the user threaten to cancel or leave?" },
  urgency: { type: "score", instructions: "How urgent is this request?", criteria: ["can wait", "this week", "today"] },
};
const response = model.decide(state, questions);
const { answers } = response;

const fail = (message) => {
  console.error(`smoke test failed: ${message}`);
  process.exit(1);
};
const probability = (value, what) => (Number.isFinite(value) && value >= 0 && value <= 1 ? value : fail(`${what} is not a probability: ${value}`));
for (const id of Object.keys(questions)) if (answers?.[id]?.type !== questions[id].type) fail(`no ${questions[id].type} answer for ${id}`);
if (!Object.hasOwn(questions.department.criteria, answers.department.choice)) fail(`department chose an unknown option: ${answers.department.choice}`);
const total = Object.values(answers.department.probabilities).reduce((sum, p) => sum + probability(p, "a department probability"), 0);
// answers are rounded to two or four places, so the sum is 1 only to that precision
if (Math.abs(total - 1) > 0.03) fail(`department probabilities sum to ${total}`);
probability(answers.churn_risk.noul, "churn_risk");
if (!(answers.urgency.score >= 0 && answers.urgency.score <= 2)) fail(`urgency score ${answers.urgency.score} is outside its three levels`);
if (!(response.usage?.input_tokens > 0)) fail("the response reports no input tokens");
const again = model.decideMany([{ state, questions }, { state: "ok", questions: { churn_risk: questions.churn_risk } }]);
if (again.length !== 2 || JSON.stringify(again[0].answers) !== JSON.stringify(answers)) fail("a batched request answered differently from the same request alone");

console.log(JSON.stringify(answers));
console.log(`${response.usage.input_tokens} input tokens; smoke test passed`);
