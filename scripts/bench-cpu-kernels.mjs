#!/usr/bin/env node
// Guard and benchmark the production Rust CPU linear kernel through a feature-gated WASM build.
// Build the artifacts first with scripts/build-cpu-bench.sh.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { parseArgs as parseNodeArgs } from "node:util";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SEED = 0x4b455641;
const ABS_TOL = 2e-4;
const REL_TOL = 1e-4;
const FLAVORS = new Set(["base", "simd", "relaxed"]);
const MODES = new Set(["guard", "benchmark"]);

function usage() {
  return [
    "usage: node scripts/bench-cpu-kernels.mjs [guard|benchmark] [base|simd|relaxed] [options]",
    "options: --artifact=FILE --output=FILE --samples=N --warmups=N",
  ].join("\n");
}

function parseCli(argv) {
  const { values, positionals } = parseNodeArgs({
    args: argv,
    allowPositionals: true,
    options: {
      artifact: { type: "string" },
      output: { type: "string" },
      samples: { type: "string", default: "5" },
      warmups: { type: "string", default: "2" },
    },
  });
  if (positionals.length > 2) throw new Error(`expected [guard|benchmark] [base|simd|relaxed]\n${usage()}`);
  const mode = positionals[0] || "guard";
  const flavor = positionals[1] || "base";
  if (!MODES.has(mode)) throw new Error(`unknown mode: ${mode}\n${usage()}`);
  if (!FLAVORS.has(flavor)) throw new Error(`unknown flavor: ${flavor}\n${usage()}`);
  const samples = Number(values.samples);
  const warmups = Number(values.warmups);
  if (!Number.isInteger(samples) || samples < 1) throw new Error("--samples must be a positive integer");
  if (!Number.isInteger(warmups) || warmups < 0) throw new Error("--warmups must be a non-negative integer");
  return { mode, flavor, artifact: values.artifact, output: values.output, samples, warmups };
}

function next(rng) {
  return (Math.imul(rng, 1_664_525) + 1_013_904_223) >>> 0;
}

function f32Values(length, tag, divisor) {
  let rng = (SEED ^ tag) >>> 0;
  const values = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    rng = next(rng);
    const centered = (rng % 2_001) - 1_000;
    values[i] = Math.fround(centered / divisor);
  }
  return values;
}

function q8Values(length) {
  let rng = (SEED ^ 2) >>> 0;
  const values = new Int8Array(length);
  for (let i = 0; i < length; i++) {
    rng = next(rng);
    values[i] = (rng % 255) - 127;
  }
  return values;
}

function scaleValues(length) {
  let rng = (SEED ^ 3) >>> 0;
  const values = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    rng = next(rng);
    values[i] = Math.fround(0.0005 + (rng % 251) / 1_000_000);
  }
  return values;
}

function makeInputs({ t, n, k, q8, bias }) {
  const x = f32Values(t * k, 1, 4_096);
  const weights = q8 ? undefined : f32Values(n * k, 2, 8_192);
  const q = q8 ? q8Values(n * k) : undefined;
  const scales = q8 ? scaleValues(n * (k / 32)) : undefined;
  const biases = bias ? f32Values(n, 4, 4_096) : undefined;
  return { x, weights, q, scales, biases };
}

function cpuReference(spec) {
  const { t, n, k, q8, bias } = spec;
  const data = makeInputs(spec);
  const out = new Float32Array(t * n);
  const blocks = q8 ? k / 32 : 0;
  for (let row = 0; row < t; row++) {
    for (let col = 0; col < n; col++) {
      let sum = 0;
      const x0 = row * k;
      const w0 = col * k;
      for (let i = 0; i < k; i++) {
        const weight = q8 ? data.q[w0 + i] * data.scales[col * blocks + Math.floor(i / 32)] : data.weights[w0 + i];
        sum += data.x[x0 + i] * weight;
      }
      out[row * n + col] = Math.fround(sum + (bias ? data.biases[col] : 0));
    }
  }
  return out;
}

function outputChecksum(values) {
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += Math.abs(values[i]) * ((i % 17) + 1);
  return sum;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length & 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function compareOutput(actual, expected) {
  if (actual.length !== expected.length) throw new Error(`output length ${actual.length}, expected ${expected.length}`);
  let maxAbs = 0;
  let maxRel = 0;
  let worst = -1;
  for (let i = 0; i < actual.length; i++) {
    const got = actual[i];
    const want = expected[i];
    const abs = Math.abs(got - want);
    const rel = abs / Math.max(Math.abs(want), 1e-12);
    if (!Number.isFinite(got) || abs > maxAbs) {
      maxAbs = abs;
      worst = i;
    }
    maxRel = Math.max(maxRel, rel);
    if (!Number.isFinite(got) || abs > ABS_TOL + REL_TOL * Math.abs(want)) {
      throw new Error(`output mismatch at ${i}: got ${got}, expected ${want}, abs ${abs} (limit ${ABS_TOL + REL_TOL * Math.abs(want)})`);
    }
  }
  return { maxAbs, maxRel, worst };
}

function findArtifact(flavor, requested) {
  const candidates = [
    requested,
    process.env.KEVALA_CPU_BENCH_WASM,
    resolve(ROOT, `tmp/cpu-bench/kevala-${flavor}.wasm`),
    resolve(ROOT, `tmp/kevala-cpu-bench-${flavor}.wasm`),
    resolve(ROOT, `tmp/cpu-bench-${flavor}.wasm`),
  ].filter(Boolean);
  const artifact = candidates.find((candidate) => existsSync(resolve(process.cwd(), candidate)));
  if (!artifact) {
    throw new Error(`missing ${flavor} CPU benchmark artifact; run scripts/build-cpu-bench.sh ${flavor}`);
  }
  return resolve(process.cwd(), artifact);
}

async function loadWasm(path) {
  const { instance } = await WebAssembly.instantiate(readFileSync(path), {});
  const x = instance.exports;
  for (const name of ["kevala_cpu_bench_prepare", "kevala_cpu_bench_run", "kevala_cpu_bench_out_ptr", "kevala_cpu_bench_out_len", "kevala_cpu_bench_checksum"]) {
    if (typeof x[name] !== "function") throw new Error(`${path} does not export ${name}`);
  }
  x.kevala_init?.();
  return { x, memory: x.memory };
}

function outputView(wasm) {
  const ptr = Number(wasm.x.kevala_cpu_bench_out_ptr());
  const len = Number(wasm.x.kevala_cpu_bench_out_len());
  return new Float32Array(wasm.memory.buffer, ptr, len);
}

function prepare(wasm, spec) {
  const rc = wasm.x.kevala_cpu_bench_prepare(spec.t, spec.n, spec.k, spec.q8 ? 1 : 0, spec.bias ? 1 : 0, spec.tile);
  if (rc !== 0) throw new Error(`prepare failed for ${JSON.stringify(spec)} (status ${rc})`);
}

function run(wasm, spec) {
  const rc = wasm.x.kevala_cpu_bench_run();
  if (rc !== 0) throw new Error(`run failed for ${JSON.stringify(spec)} (status ${rc})`);
}

function checkChecksum(wasm, values, spec) {
  const rust = Number(wasm.x.kevala_cpu_bench_checksum());
  const js = outputChecksum(values);
  if (!Number.isFinite(rust) || Math.abs(rust - js) > 1e-9 * Math.max(1, Math.abs(js))) {
    throw new Error(`checksum mismatch for ${JSON.stringify(spec)}: Rust ${rust}, JS ${js}`);
  }
  return js;
}

const guardTs = [1, 2, 3, 4, 7, 17, 31];
const guardNs = [1, 3, 4, 7, 12];
const guardF32Ks = [4, 12, 20, 28, 36, 52, 68];
const guardQ8Ks = [32, 64, 96];
const benchmarkShapes = [
  ["t32-n1024-k1024", 32, 1_024, 1_024],
  ["t128-n1024-k1024", 128, 1_024, 1_024],
  ["t32-n3584-k1024", 32, 3_584, 1_024],
  ["t128-n3584-k1024", 128, 3_584, 1_024],
];

async function guard(wasm, flavor, outputPath) {
  const cases = [];
  for (const q8 of [false, true]) {
    const ks = q8 ? guardQ8Ks : guardF32Ks;
    for (const k of ks) {
      for (const t of guardTs) {
        for (const n of guardNs) {
          for (const bias of [false, true]) {
            for (const tile of [0, 1]) {
              const spec = { t, n, k, q8, bias, tile };
              prepare(wasm, spec);
              run(wasm, spec);
              const actual = outputView(wasm);
              const expected = cpuReference(spec);
              const errors = compareOutput(actual, expected);
              const checksum = checkChecksum(wasm, actual, spec);
              cases.push({ ...spec, checksum, ...errors });
            }
          }
        }
      }
    }
  }
  const report = {
    kind: "kevala-cpu-linear",
    mode: "guard",
    flavor,
    seed: `0x${SEED.toString(16)}`,
    tolerance: { abs: ABS_TOL, relative: REL_TOL },
    cases,
    passed: cases.length,
  };
  mkdirSync(resolve(outputPath, ".."), { recursive: true });
  writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(`CPU numerical guards passed (${cases.length} cases)`);
}

async function benchmark(wasm, flavor, outputPath, samples, warmups) {
  const cases = [];
  for (const [name, t, n, k] of benchmarkShapes) {
    for (const bias of [false, true]) {
      for (const tile of [0, 1]) {
        const spec = { t, n, k, q8: true, bias, tile };
        prepare(wasm, spec);
        for (let i = 0; i < warmups; i++) run(wasm, spec);
        const timings = [];
        let checksum;
        for (let i = 0; i < samples; i++) {
          const start = performance.now();
          run(wasm, spec);
          const elapsed = performance.now() - start;
          const actual = outputView(wasm);
          checksum = checkChecksum(wasm, actual, spec);
          timings.push(elapsed);
        }
        const result = { name, ...spec, samples, warmups, timesMs: timings, medianMs: median(timings), checksum };
        cases.push(result);
        console.error(JSON.stringify(result));
      }
    }
  }
  const geometricMeanMs = Math.exp(cases.reduce((sum, result) => sum + Math.log(Math.max(result.medianMs, Number.MIN_VALUE)), 0) / cases.length);
  const report = {
    kind: "kevala-cpu-linear",
    mode: "benchmark",
    flavor,
    seed: `0x${SEED.toString(16)}`,
    samples,
    warmups,
    cases,
    geometricMeanMs,
  };
  mkdirSync(resolve(outputPath, ".."), { recursive: true });
  writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(geometricMeanMs.toString());
}

async function main() {
  const options = parseCli(process.argv.slice(2));
  const artifact = findArtifact(options.flavor, options.artifact);
  const outputPath = resolve(process.cwd(), options.output || resolve(ROOT, "tmp/cpu-bench", `${options.mode}-${options.flavor}.json`));
  const wasm = await loadWasm(artifact);
  if (options.mode === "guard") await guard(wasm, options.flavor, outputPath);
  else await benchmark(wasm, options.flavor, outputPath, options.samples, options.warmups);
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exitCode = 1;
});
