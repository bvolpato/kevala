export const SOURCE_PROFILE_CASES = [
  { T: 7, R: 79, N: 68, K: 96 },
  { T: 17, R: 80, N: 132, K: 1056 },
  { T: 28, R: 64, N: 6144, K: 1536 },
  { T: 28, R: 64, N: 1536, K: 6144 },
];

export function sourceFixture({ T, N, K }, seed = 20261106) {
  if (![T, N, K].every((value) => Number.isSafeInteger(value) && value > 0) || K % 32) throw new Error("invalid source fixture dimensions");
  let state = seed >>> 0;
  const next = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state; };
  const weights = Int8Array.from({ length: N * K }, () => (next() >>> 24) - 128);
  const scales = Float32Array.from({ length: N * K / 32 }, () => ((next() % 5) + 1) / 16384);
  const input = Float32Array.from({ length: T * K }, () => ((next() % 17) - 8) / 64);
  const bias = Float32Array.from({ length: N }, () => ((next() % 7) - 3) / 128);
  const residual = Float32Array.from({ length: T * N }, () => ((next() % 11) - 5) / 256);
  return { T, N, K, weights, scales, input, bias, residual };
}

export function sourcePartials(fixture, splits) {
  const { T, N, K, weights, scales, input } = fixture;
  if (!Number.isSafeInteger(splits) || splits < 1 || splits > 8 || weights.length !== N * K || scales.length !== N * K / 32 || input.length !== T * K) throw new Error("invalid source partial reference input");
  const blocks = K / 32, per = Math.ceil(blocks / splits), output = new Float32Array(splits * T * N);
  for (let split = 0; split < splits; split++) for (let row = 0; row < T; row++) for (let column = 0; column < N; column++) {
    let sum = 0;
    for (let k = split * per * 32; k < Math.min(blocks, (split + 1) * per) * 32; k++) {
      const weight = Math.fround(weights[column * K + k] * scales[column * blocks + Math.floor(k / 32)]);
      sum = Math.fround(sum + Math.fround(input[row * K + k] * weight));
    }
    output[split * T * N + row * N + column] = sum;
  }
  return output;
}

export function reduceReference(partials, { T, N, splits, mode, bias, residual }) {
  if (partials.length !== splits * T * N || ![0, 1, 2].includes(mode) || bias && bias.length !== N || residual && residual.length !== T * N || mode === 1 && !residual) throw new Error("invalid reducer reference input");
  const output = new Float32Array(T * N);
  for (let index = 0; index < output.length; index++) {
    let value = 0;
    for (let split = 0; split < splits; split++) value = Math.fround(value + partials[split * output.length + index]);
    if (bias) value = Math.fround(value + bias[index % N]);
    if (mode === 1) value = Math.fround(value + residual[index]);
    else if (mode === 2) value = Math.max(value, 0);
    output[index] = value;
  }
  return output;
}

export function pleReference(values, ple) {
  if (values.length !== ple.length) throw new Error("PLE reference lengths differ");
  return Float32Array.from(values, (x, index) => {
    const gelu = x >= 10 ? x : x <= -10 ? 0 : 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
    return Math.fround(gelu * ple[index]);
  });
}
