#!/usr/bin/env node
// Check the production vector exp through the development-only CPU benchmark WASM ABI.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: { artifact: { type: "string" }, output: { type: "string" } },
});
const flavor = positionals[0] || "relaxed";
if (positionals.length > 1 || !["base", "simd", "relaxed"].includes(flavor)) {
  throw new Error("usage: node scripts/check-cpu-exp.mjs [base|simd|relaxed] [--artifact=FILE] [--output=FILE]");
}
const artifact = resolve(values.artifact || process.env.KEVALA_CPU_BENCH_WASM || resolve(ROOT, `tmp/cpu-bench/kevala-${flavor}.wasm`));
const output = resolve(values.output || resolve(ROOT, `tmp/cpu-bench/exp-guard-${flavor}.json`));
const { instance } = await WebAssembly.instantiate(readFileSync(artifact), {});
const api = instance.exports;
for (const name of ["kevala_cpu_bench_exp_prepare", "kevala_cpu_bench_exp_run", "kevala_cpu_bench_exp_scalar", "kevala_cpu_bench_out_ptr", "kevala_cpu_bench_out_len"]) {
  if (typeof api[name] !== "function") throw new Error(`${artifact} does not export ${name}; rebuild with scripts/build-cpu-bench.sh ${flavor}`);
}
api.kevala_init?.();

const bitBuffer = new ArrayBuffer(4);
const floatBits = new Float32Array(bitBuffer);
const intBits = new Uint32Array(bitBuffer);
function bits(x) {
  floatBits[0] = x;
  return intBits[0];
}
function fromBits(x) {
  intBits[0] = x;
  return floatBits[0];
}

const cases = [];
function check(label, input) {
  const xs = Float32Array.from(input);
  const pointer = Number(api.kevala_cpu_bench_exp_prepare(xs.length));
  new Float32Array(api.memory.buffer, pointer, xs.length).set(xs);
  if (api.kevala_cpu_bench_exp_run() !== 0) throw new Error(`${label}: exp run failed`);
  const length = Number(api.kevala_cpu_bench_out_len());
  if (length !== xs.length) throw new Error(`${label}: output length ${length}, expected ${xs.length}`);
  const actual = new Float32Array(api.memory.buffer, Number(api.kevala_cpu_bench_out_ptr()), length);
  const vectorLength = length - (length % 4);
  const report = { label, inputs: length, vectorLanes: 0, scalarLanes: 0, maxUlps: 0, maxRelative: 0 };
  for (let first = 0; first < length; first += 4) {
    const end = Math.min(first + 4, length);
    let scalar = first >= vectorLength;
    for (let i = first; i < end; i++) scalar ||= !(Math.abs(xs[i]) <= 80);
    for (let i = first; i < end; i++) {
      const x = xs[i];
      const got = actual[i];
      const want = scalar ? api.kevala_cpu_bench_exp_scalar(x) : Math.fround(Math.exp(x));
      if (scalar) {
        report.scalarLanes++;
        if (Number.isNaN(want) ? !Number.isNaN(got) : bits(got) !== bits(want)) {
          throw new Error(`${label}[${i}]: scalar exp(${x}) got ${got}, expected ${want}`);
        }
      } else {
        report.vectorLanes++;
        const ulps = Math.abs(bits(got) - bits(want));
        const relative = Math.abs((got - want) / want);
        if (!Number.isFinite(got) || ulps > 4 || relative > 5e-7) {
          throw new Error(`${label}[${i}]: exp(${x}) got ${got}, expected ${want}; ${ulps} ULP, relative ${relative}`);
        }
        report.maxUlps = Math.max(report.maxUlps, ulps);
        report.maxRelative = Math.max(report.maxRelative, relative);
      }
    }
  }
  cases.push(report);
}

check("dense-range", Array.from({ length: 100_001 }, (_, i) => -80 + i / 625));

const boundaries = [];
for (let k = -115; k <= 115; k++) {
  for (const center of [k, k + 0.5]) {
    const middle = bits(center * Math.LN2);
    for (let offset = -8; offset <= 8; offset++) {
      const x = fromBits(middle + offset);
      if (Math.abs(x) <= 80) boundaries.push(x);
    }
  }
}
check("reduction-boundaries", boundaries);

let rng = 0x4b455641;
const random = [];
for (let i = 0; i < 32_768; i++) {
  rng ^= rng << 13;
  rng ^= rng >>> 17;
  rng ^= rng << 5;
  const x = fromBits(rng);
  if (Math.abs(x) <= 80) random.push(x);
}
check("normal-bit-patterns", random);

const outside = fromBits(bits(80) + 1);
const exceptional = [NaN, Infinity, -Infinity, -104, -90, 88, 90, outside, -outside];
check("exceptional-values", exceptional);
for (const value of exceptional) {
  for (let lane = 0; lane < 4; lane++) {
    const mixed = [-0.5, -0, 0.5, 2];
    mixed[lane] = value;
    check(`mixed-${String(value)}-lane-${lane}`, mixed);
  }
}
const tailValues = [80, -80, 0, -0, fromBits(1), Math.LN2 / 2, -Math.LN2 / 2, 0.125];
for (const length of [0, 1, 2, 3, 4, 5, 6, 7, 9, 17]) {
  check(`length-${length}`, Array.from({ length }, (_, i) => tailValues[i % tailValues.length]));
}

const report = {
  kind: "kevala-cpu-exp",
  flavor,
  artifact,
  tolerance: { ulps: 4, relative: 5e-7, fallback: "exact scalar Rust exp" },
  inputs: cases.reduce((sum, c) => sum + c.inputs, 0),
  maxUlps: Math.max(...cases.map((c) => c.maxUlps)),
  maxRelative: Math.max(...cases.map((c) => c.maxRelative)),
  cases,
};
mkdirSync(dirname(output), { recursive: true });
writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
console.log(`CPU exp guard passed (${flavor}, ${report.inputs} inputs, ${report.maxUlps} ULP, ${report.maxRelative} relative)`);
