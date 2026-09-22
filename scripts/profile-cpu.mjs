#!/usr/bin/env node
// Sampling evidence for the CPU WebAssembly path. This is a profiler, not a benchmark metric.

import inspector from "node:inspector";
import { mkdirSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { dirname, resolve } from "node:path";
import { parseArgs, promisify } from "node:util";

const WORDS = "the customer wrote in about a recurring charge they did not recognise and wants a refund today ";
const QUESTION = { type: "noul", instructions: "Is the customer asking for money back?" };

function workloadShapes(repeats) {
  return [
    { name: "medium", text: WORDS.repeat(6), wordRepeats: 6, approximateTokens: 128 },
    { name: "long", text: WORDS.repeat(repeats), wordRepeats: repeats, approximateTokens: Math.round((repeats / 28) * 512) },
  ];
}

function usage() {
  return [
    "usage: node scripts/profile-cpu.mjs --pack PATH [options]",
    "options: --runtime=PATH --flavor=relaxed --output=PREFIX --runs=3 --repeats=28",
  ].join("\n");
}

function positiveInteger(value, name) {
  const number = Number(value);
  if (!Number.isInteger(number) || number < 1) {
    throw new Error(`${name} must be a positive integer`);
  }
  return number;
}

function parseOptions(argv) {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: {
      runtime: { type: "string", default: "js/src/node.js" },
      pack: { type: "string" },
      flavor: { type: "string", default: "relaxed" },
      output: { type: "string", default: "tmp/cpu-profile" },
      runs: { type: "string", default: "3" },
      repeats: { type: "string", default: "28" },
    },
  });
  if (positionals.length) throw new Error(`unexpected arguments: ${positionals.join(" ")}\n${usage()}`);
  if (!values.pack) throw new Error(`--pack is required\n${usage()}`);
  return {
    runtime: resolve(process.cwd(), values.runtime),
    pack: resolve(process.cwd(), values.pack),
    flavor: values.flavor,
    output: resolve(process.cwd(), values.output),
    runs: positiveInteger(values.runs, "--runs"),
    repeats: positiveInteger(values.repeats, "--repeats"),
  };
}

function uniquePrefix(index) {
  let n = index;
  let word = "";
  do {
    word = String.fromCharCode(97 + (n % 26)) + word;
    n = Math.floor(n / 26) - 1;
  } while (n >= 0);
  return word;
}

function frameKey(frame) {
  return `${frame.functionName || "(anonymous)"}\u0000${frame.url || ""}`;
}

function frameDetails(node) {
  const frame = node?.callFrame || {};
  return {
    functionName: frame.functionName || "(anonymous)",
    url: frame.url || "",
  };
}

function aggregateProfile(profile) {
  const nodes = profile.nodes || [];
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const aggregate = new Map();
  const get = (details) => {
    const key = frameKey(details);
    let entry = aggregate.get(key);
    if (!entry) {
      entry = { ...details, sampledTimeUs: 0, hitCount: 0, sampleCount: 0 };
      aggregate.set(key, entry);
    }
    return entry;
  };

  for (const node of nodes) {
    const entry = get(frameDetails(node));
    entry.hitCount += Number(node.hitCount || 0);
  }

  const samples = profile.samples || [];
  const deltas = profile.timeDeltas || [];
  const fallbackDelta = samples.length && Number.isFinite(profile.startTime) && Number.isFinite(profile.endTime)
    ? Math.max(0, profile.endTime - profile.startTime) / samples.length
    : 0;
  for (let i = 0; i < samples.length; i++) {
    const node = nodeById.get(samples[i]);
    if (!node) continue;
    const entry = get(frameDetails(node));
    entry.sampledTimeUs += Number(deltas[i] ?? fallbackDelta);
    entry.sampleCount++;
  }

  const functions = [...aggregate.values()];
  if (!functions.some((entry) => entry.hitCount > 0)) {
    for (const entry of functions) entry.hitCount = entry.sampleCount;
  }
  for (const entry of functions) entry.sampledTimeMs = entry.sampledTimeUs / 1_000;
  functions.sort((a, b) => b.sampledTimeUs - a.sampledTimeUs || b.hitCount - a.hitCount);
  return functions;
}

async function loadRuntime(options) {
  const runtime = await import(pathToFileURL(options.runtime).href);
  if (typeof runtime.loadFile !== "function") throw new Error(`${options.runtime} does not export loadFile`);
  return runtime.loadFile;
}

function runWorkload(model, runs, shapes) {
  const questions = { q0: QUESTION };
  const byShape = Object.fromEntries(shapes.map((shape) => [shape.name, {
    requests: 0,
    inputTokens: [],
  }]));
  let requestIndex = 0;
  for (let run = 0; run < runs; run++) {
    for (const shape of shapes) {
      // Every call starts with a distinct word, avoiding state-cache hits in causal models.
      const state = `${uniquePrefix(requestIndex++)} ${shape.text}`;
      const result = model.decide(state, questions);
      const stats = byShape[shape.name];
      stats.requests++;
      if (Number.isInteger(result?.usage?.input_tokens)) stats.inputTokens.push(result.usage.input_tokens);
    }
  }
  return { requestCount: requestIndex, byShape, shapes };
}

async function profileWorkload(model, runs, repeats) {
  const session = new inspector.Session();
  session.connect();
  const post = promisify(session.post).bind(session);
  let profile;
  let workload;
  const shapes = workloadShapes(repeats);
  try {
    await post("Profiler.enable");
    await post("Profiler.start");
    try {
      workload = runWorkload(model, runs, shapes);
    } finally {
      const stopped = await post("Profiler.stop");
      profile = stopped?.profile || stopped;
    }
  } finally {
    try {
      await post("Profiler.disable");
    } finally {
      session.disconnect();
    }
  }
  if (!profile) throw new Error("Node CPU profiler did not return a profile");
  return { profile, workload };
}

function writeResults(profile, workload, options) {
  const functions = aggregateProfile(profile);
  const top = functions.slice(0, 10);
  const profilePath = `${options.output}.cpuprofile`;
  const summaryPath = `${options.output}.json`;
  mkdirSync(dirname(profilePath), { recursive: true });
  writeFileSync(profilePath, `${JSON.stringify(profile)}\n`);

  const summary = {
    kind: "kevala-cpu-profile",
    profiler: "Node inspector Profiler",
    evidenceType: "sampling CPU profile",
    runtime: options.runtime,
    pack: options.pack,
    flavor: options.flavor,
    runs: options.runs,
    repeats: options.repeats,
    warmup: "one medium request before profiling",
    requestCount: workload.requestCount,
    shapes: workload.shapes.map(({ name, wordRepeats, approximateTokens }) => ({ name, wordRepeats, approximateTokens })),
    profile: {
      startTime: profile.startTime,
      endTime: profile.endTime,
      durationMs: (profile.endTime - profile.startTime) / 1_000,
      sampleCount: profile.samples?.length || 0,
    },
    byShape: workload.byShape,
    functions,
    top10: top,
    profileFile: profilePath,
  };
  writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`);

  console.log(`Node CPU sampling profile written: ${profilePath}`);
  console.log(`Summary written: ${summaryPath}`);
  console.log(JSON.stringify({
    runtime: options.runtime,
    pack: options.pack,
    flavor: options.flavor,
    runs: options.runs,
    repeats: options.repeats,
    requestCount: workload.requestCount,
    sampleCount: summary.profile.sampleCount,
    profileDurationMs: summary.profile.durationMs,
  }, null, 2));
  console.log("Top 10 sampled functions:");
  for (const entry of top) {
    console.log(`${entry.sampledTimeMs.toFixed(3).padStart(10)} ms  ${String(entry.hitCount).padStart(6)} hits  ${entry.functionName}  ${entry.url}`);
  }
}

async function main() {
  const options = parseOptions(process.argv.slice(2));
  const loadFile = await loadRuntime(options);
  const model = await loadFile(options.pack, { flavor: options.flavor });
  const warmupQuestions = { q0: QUESTION };
  model.decide(`warmup ${workloadShapes(options.repeats)[0].text}`, warmupQuestions);
  const { profile, workload } = await profileWorkload(model, options.runs, options.repeats);
  writeResults(profile, workload, options);
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exitCode = 1;
});
