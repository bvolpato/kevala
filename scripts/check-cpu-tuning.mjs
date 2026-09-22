#!/usr/bin/env node
// Check the shipping calibration ABI against independent small-matrix arithmetic.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  return sign * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a));
}

function linear(input, rows, n, k, tag) {
  const out = new Float64Array(rows * n);
  for (let row = 0; row < rows; row++) {
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < k; i++) {
        const q = (j * 31 + i * 17 + tag * 13) % 255 - 127;
        const scale = 0.001 + ((j * 7 + Math.floor(i / 32) * 11 + tag * 5) % 23) * 0.00001;
        out[row * n + j] += input[row * k + i] * q * scale;
      }
    }
  }
  return out;
}

const rows = 3, width = 32, inner = 64;
const input = Float32Array.from({ length: rows * width }, (_, i) => (i % 11 - 5) / 8);
const up = linear(input, rows, inner * 2, width, 1);
const gated = Float64Array.from({ length: rows * inner }, (_, i) => {
  const offset = Math.floor(i / inner) * inner * 2 + i % inner;
  const x = up[offset];
  return 0.5 * x * (1 + erf(x / Math.SQRT2)) * up[offset + inner];
});
const expected = linear(gated, rows, width, inner, 2);
const results = [];
for (const flavor of ["relaxed", "simd", "base"]) {
  const { instance } = await WebAssembly.instantiate(await readFile(new URL(`../js/src/kevala-${flavor}.wasm`, import.meta.url)), {});
  const x = instance.exports;
  x.kevala_init();
  assert.equal(x.kevala_cpu_tune_run(), 1);
  assert.equal(x.kevala_cpu_tune_prepare(129, width, inner), 1);
  assert.equal(x.kevala_cpu_tune_prepare(rows, 33, inner), 1);
  assert.equal(x.kevala_cpu_tune_prepare(rows, width, inner), 0);
  assert.equal(x.kevala_cpu_tune_input_len(), input.length);
  assert.equal(x.kevala_cpu_tune_output_len(), expected.length);
  new Float32Array(x.memory.buffer, x.kevala_cpu_tune_input_ptr(), input.length).set(input);
  for (const tile of [0, 1]) {
    x.kevala_set_tile(tile);
    const memoryBytes = x.memory.buffer.byteLength;
    assert.equal(x.kevala_cpu_tune_run(), 0);
    assert.equal(x.memory.buffer.byteLength, memoryBytes, "run must not grow memory");
    const actual = new Float32Array(x.memory.buffer, x.kevala_cpu_tune_output_ptr(), expected.length);
    const maxError = Math.max(...actual.map((v, i) => Math.abs(v - expected[i])));
    assert.ok(maxError < 1e-6, `${flavor}/${tile}: ${maxError}`);
    results.push({ flavor, tile, values: actual.length, maxError });
  }
  x.kevala_cpu_tune_drop();
  assert.equal(x.kevala_cpu_tune_input_ptr(), 0);
  assert.equal(x.kevala_cpu_tune_output_len(), 0);
  assert.equal(x.kevala_cpu_tune_run(), 1);
}
console.log(JSON.stringify(results, null, 2));
