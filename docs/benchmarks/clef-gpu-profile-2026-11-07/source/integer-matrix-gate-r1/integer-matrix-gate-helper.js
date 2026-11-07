// Private synthetic operator experiment. Importing this module does not request a device.
export const shape = Object.freeze({ M: 512, N: 5248, K: 1024, block: 32 });
export const geometry = Object.freeze({
  BM: 64, BN: 32, threads: 256, subgroupSize: 32, subgroups: 8,
  matrixM: 16, matrixN: 16, matrixK: 32, scratchBytes: 8192,
  outputGroups: [164, 8, 1], quantizeGroups: [2048, 1, 1],
  planeWords: 131072, outputValues: 2686976, activationValues: 524288,
  activationBlocks: 16384, guardWords: 64,
});
export const numericPolicy = Object.freeze({
  reconstructionMaxAbsPerBlockMax: 2 ** -20,
  absolute: 0.0005, relative: 0.0001,
  comparison: "abs(candidate-reference) <= absolute + relative*max(1,abs(reference))",
  exactFixtures: ["integer", "dyadic"],
});

export function requireCapabilities(adapter, device) {
  const info = adapter.info;
  if (info.subgroupMinSize !== 32 || info.subgroupMaxSize !== 32) throw new Error("Requires fixed subgroup size 32");
  const config = Array.from(info.subgroupMatrixConfigs ?? []).find((c) =>
    c.componentType === "i8" && c.resultComponentType === "i32" && c.M === 16 && c.N === 16 && c.K === 32);
  if (!config) throw new Error("Missing i8/i32 M16 N16 K32 configuration");
  for (const feature of ["subgroups", "chromium-experimental-subgroup-matrix"]) {
    if (!device.features.has(feature)) throw new Error(`Missing device feature ${feature}`);
  }
  if (device.features.has("shader-f16")) throw new Error("This experiment requires the baseline device without shader-f16");
  const limits = device.limits;
  if (limits.maxComputeInvocationsPerWorkgroup < 256 || limits.maxComputeWorkgroupSizeX < 256 ||
      limits.maxComputeWorkgroupStorageSize < 16384 || limits.maxStorageBufferBindingSize < 10748416) {
    throw new Error("Insufficient device limits for candidate and production FP32 baseline");
  }
  return { config: { componentType: config.componentType, resultComponentType: config.resultComponentType,
    M: config.M, N: config.N, K: config.K }, requiredLimits: {} };
}

export function packingFixture() {
  const values = [-128, -127, -1, 0, 1, 126, 127];
  const left = new Int8Array((72 + 15 * 256 + 8) * 4).fill(0x55);
  const right = new Int8Array((88 + 15 * 256 + 8) * 4).fill(0x55);
  const expected = new Int32Array(256);
  for (let r = 0; r < 16; r++) for (let k = 0; k < 32; k++) {
    left[72 * 4 + r * 1024 + k] = values[(r * 3 + k * 5) % 7];
    right[88 * 4 + r * 1024 + k] = values[(r * 5 + k * 3 + 1) % 7];
  }
  for (let r = 0; r < 16; r++) for (let n = 0; n < 16; n++) for (let k = 0; k < 32; k++) {
    expected[r * 16 + n] += left[72 * 4 + r * 1024 + k] * right[88 * 4 + n * 1024 + k];
  }
  return { left, right, expected };
}

const matrixEnable = `enable chromium_experimental_subgroup_matrix;
enable subgroups;
alias Left = subgroup_matrix_left<i8, 32, 16>;
alias Right = subgroup_matrix_right<i8, 16, 32>;
`;

// Nonzero offsets and a full K1024 row stride deliberately test the packed-word ABI.
// Pinned Dawn tests use array<i32> and divide i8 offsets/strides by four.
export const packingGateWGSL = `${matrixEnable}
@group(0) @binding(0) var<storage, read> A: array<i32>;
@group(0) @binding(1) var<storage, read> W: array<i32>;
@group(0) @binding(2) var<storage, read_write> Y: array<i32>;
var<workgroup> scratch: array<i32, 256>;
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) li: u32) {
  for (var j = 0u; j < 8u; j++) { scratch[li + j * 32u] = 0x6badc0de; }
  workgroupBarrier();
  let lhs = subgroupMatrixLoad<Left, row_major>(&A, 72u, 256u);
  let rhs = subgroupMatrixLoad<Right, col_major>(&W, 88u, 256u);
  let result = subgroupMatrixMultiply<i32>(lhs, rhs);
  subgroupMatrixStore<row_major>(&scratch, 0u, result, 16u);
  workgroupBarrier();
  for (var j = 0u; j < 8u; j++) {
    let slot = li + j * 32u;
    Y[64u + slot] = scratch[slot];
  }
}
`;

export const quantizeWGSL = `enable chromium_experimental_subgroup_matrix;
enable subgroups;
struct Block { scale: f32, maxAbs: f32, violations: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Q: array<i32>;
@group(0) @binding(2) var<storage, read_write> A: array<Block>;
fn finite(v: f32) -> bool { return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u; }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_id) sg: u32,
        @builtin(subgroup_invocation_id) lane: u32) {
  let blockIndex = wg.x * 8u + sg;
  let raw = X[blockIndex * 32u + lane];
  let valid = finite(raw);
  let x = select(0.0, raw, valid);
  let a = subgroupMax(abs(x));
  let domain = a == 0.0 || (a >= 0.0000152587890625 && a <= 128.0);
  let s = a / 127.0;
  let safeS = select(s, 1.0, a == 0.0);
  let r0 = round(x / safeS);
  let q0 = i32(clamp(r0, -127.0, 127.0));
  let residual1 = fma(-safeS, f32(q0), x);
  let r1 = round(residual1 / (safeS / 128.0));
  let q1 = i32(clamp(r1, -127.0, 127.0));
  let residual2 = fma(-(safeS / 128.0), f32(q1), residual1);
  let r2 = round(residual2 / (safeS / 16384.0));
  let q2 = i32(clamp(r2, -127.0, 127.0));
  let clipped = r0 != f32(q0) || r1 != f32(q1) || r2 != f32(q2);
  let bad = subgroupAdd(select(0u, 1u, !valid || !domain || clipped));
  let firstLane = lane & 0xfffffffcu;
  // Every invocation participates in every shuffle, including non-writing lanes.
  let p0 = (u32(subgroupShuffle(q0, firstLane)) & 255u) |
    ((u32(subgroupShuffle(q0, firstLane + 1u)) & 255u) << 8u) |
    ((u32(subgroupShuffle(q0, firstLane + 2u)) & 255u) << 16u) |
    ((u32(subgroupShuffle(q0, firstLane + 3u)) & 255u) << 24u);
  let p1 = (u32(subgroupShuffle(q1, firstLane)) & 255u) |
    ((u32(subgroupShuffle(q1, firstLane + 1u)) & 255u) << 8u) |
    ((u32(subgroupShuffle(q1, firstLane + 2u)) & 255u) << 16u) |
    ((u32(subgroupShuffle(q1, firstLane + 3u)) & 255u) << 24u);
  let p2 = (u32(subgroupShuffle(q2, firstLane)) & 255u) |
    ((u32(subgroupShuffle(q2, firstLane + 1u)) & 255u) << 8u) |
    ((u32(subgroupShuffle(q2, firstLane + 2u)) & 255u) << 16u) |
    ((u32(subgroupShuffle(q2, firstLane + 3u)) & 255u) << 24u);
  if ((lane & 3u) == 0u) {
    let word = blockIndex * 8u + lane / 4u;
    Q[word] = bitcast<i32>(p0);
    Q[131072u + word] = bitcast<i32>(p1);
    Q[262144u + word] = bitcast<i32>(p2);
  }
  if (lane == 0u) { A[blockIndex] = Block(s, a, bad, 0u); }
}
`;

export const integerMatmulWGSL = `${matrixEnable}
struct Block { scale: f32, maxAbs: f32, violations: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> Q: array<i32>;
@group(0) @binding(1) var<storage, read> W: array<i32>;
@group(0) @binding(2) var<storage, read> A: array<Block>;
@group(0) @binding(3) var<storage, read> S: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> scratch: array<i32, 2048>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_id) sg: u32,
        @builtin(local_invocation_index) li: u32) {
  let m0 = wg.y * 64u;
  let n0 = wg.x * 32u;
  let tileM = sg / 2u;
  let tileN = sg % 2u;
  let scratchBase = tileM * 512u + tileN * 16u;
  var acc: array<f32, 8>;
  for (var kb = 0u; kb < 32u; kb++) {
    let rhs = subgroupMatrixLoad<Right, col_major>(&W, (n0 + tileN * 16u) * 256u + kb * 8u, 256u);
    for (var plane = 0u; plane < 3u; plane++) {
      let lhs = subgroupMatrixLoad<Left, row_major>(&Q,
        plane * 131072u + (m0 + tileM * 16u) * 256u + kb * 8u, 256u);
      let result = subgroupMatrixMultiply<i32>(lhs, rhs);
      subgroupMatrixStore<row_major>(&scratch, scratchBase, result, 32u);
      workgroupBarrier();
      let factor = select(select(1.0, 0.0078125, plane == 1u), 0.00006103515625, plane == 2u);
      for (var j = 0u; j < 8u; j++) {
        let slot = li + j * 256u;
        let row = m0 + slot / 32u;
        let col = n0 + slot % 32u;
        let scale = (A[row * 32u + kb].scale * factor) * S[col * 32u + kb];
        acc[j] = acc[j] + f32(scratch[slot]) * scale;
      }
      // All 256 invocations finish consuming this plane before any subgroup reuses it.
      workgroupBarrier();
    }
  }
  for (var j = 0u; j < 8u; j++) {
    let slot = li + j * 256u;
    Y[(m0 + slot / 32u) * 5248u + n0 + slot % 32u] = acc[j] + 0.0;
  }
}
`;

const statusWGSL = `struct Status {
  outputs: atomic<u32>, mismatches: atomic<u32>, referenceNonfinite: atomic<u32>, candidateNonfinite: atomic<u32>,
  quantValues: atomic<u32>, quantBlocks: atomic<u32>, quantFailures: atomic<u32>,
  canaries: atomic<u32>, canaryFailures: atomic<u32>
}
fn finite(v: f32) -> bool { return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u; }
`;

export const compareOutputsWGSL = `${statusWGSL}
struct Params { exact: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> R: array<f32>;
@group(0) @binding(1) var<storage, read> C: array<f32>;
@group(0) @binding(2) var<storage, read_write> status: Status;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= 2686976u) { return; }
  let r = R[i]; let c = C[i];
  atomicAdd(&status.outputs, 1u);
  if (!finite(r)) { atomicAdd(&status.referenceNonfinite, 1u); }
  if (!finite(c)) { atomicAdd(&status.candidateNonfinite, 1u); }
  let tolerance = 0.0005 + 0.0001 * max(1.0, abs(r));
  if ((p.exact == 1u && bitcast<u32>(r) != bitcast<u32>(c)) ||
      (p.exact == 0u && abs(c - r) > tolerance)) { atomicAdd(&status.mismatches, 1u); }
}
`;

export const compareQuantizationWGSL = `${statusWGSL}
struct Block { scale: f32, maxAbs: f32, violations: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Q: array<i32>;
@group(0) @binding(2) var<storage, read> A: array<Block>;
@group(0) @binding(3) var<storage, read_write> status: Status;
fn byte(word: i32, lane: u32) -> i32 { return (word << (24u - lane * 8u)) >> 24u; }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= 524288u) { return; }
  let blockIndex = i / 32u;
  let blockData = A[blockIndex];
  let q0 = byte(Q[i / 4u], i % 4u);
  let q1 = byte(Q[131072u + i / 4u], i % 4u);
  let q2 = byte(Q[262144u + i / 4u], i % 4u);
  let reconstructed = (blockData.scale * f32(q0) + (blockData.scale / 128.0) * f32(q1)) +
    (blockData.scale / 16384.0) * f32(q2);
  let a = blockData.maxAbs;
  let domain = a == 0.0 || (a >= 0.0000152587890625 && a <= 128.0);
  var bad = !finite(X[i]) || !finite(blockData.scale) || !finite(a) || !domain || blockData.violations != 0u ||
    !finite(reconstructed) || abs(reconstructed - X[i]) > 0.00000095367431640625 * a ||
    abs(q0) > 127 || abs(q1) > 127 || abs(q2) > 127;
  if (a == 0.0 && (q0 != 0 || q1 != 0 || q2 != 0 || blockData.scale != 0.0)) { bad = true; }
  atomicAdd(&status.quantValues, 1u);
  if ((i % 32u) == 0u) {
    var expectedMax = 0.0;
    for (var k = 0u; k < 32u; k++) { expectedMax = max(expectedMax, abs(X[i + k])); }
    if (a != expectedMax || blockData.scale != a / 127.0 || blockData.pad != 0u) { bad = true; }
    atomicAdd(&status.quantBlocks, 1u);
  }
  if (bad) { atomicAdd(&status.quantFailures, 1u); }
}
`;

// Bind the entire guarded allocation here; operator bindings exclude the 64-word prefix.
export const compareCanariesWGSL = `${statusWGSL}
struct Params { payloadWords: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> buffer: array<u32>;
@group(0) @binding(1) var<storage, read_write> status: Status;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) li: u32) {
  let index = select(li, 64u + p.payloadWords + li - 64u, li >= 64u);
  atomicAdd(&status.canaries, 1u);
  if (buffer[index] != 0x5a17c9e3u || arrayLength(&buffer) != p.payloadWords + 128u) {
    atomicAdd(&status.canaryFailures, 1u);
  }
}
`;

export function validateStatus(words) {
  const expected = [2686976, 0, 0, 0, 524288, 16384, 0, 512, 0];
  if (words.length !== expected.length || expected.some((value, index) => words[index] !== value)) {
    throw new Error(`Integer operator validation failed: ${JSON.stringify(Array.from(words))}`);
  }
  return { outputs: words[0], quantValues: words[4], quantBlocks: words[5], canaries: words[7] };
}

// Each call is one candidate invocation. Pass boundaries provide dispatch ordering and
// storage visibility. Both passes, including quantization, belong inside the sample.
export function encodeCandidate(encoder, pipelines, bindGroups, timestampWrites) {
  for (const [name, dispatch] of [["quantize", geometry.quantizeGroups], ["integerMatmul", geometry.outputGroups]]) {
    const pass = encoder.beginComputePass({ label: `private-integer-${name}`, timestampWrites: timestampWrites?.[name] });
    pass.setPipeline(pipelines[name]);
    pass.setBindGroup(0, bindGroups[name]);
    pass.dispatchWorkgroups(...dispatch);
    pass.end();
  }
}
