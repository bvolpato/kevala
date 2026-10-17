import { matmulLayout, pipeline, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { selectMatmulKernel } from "../js/src/gpu-tuning.js";
import { kernelSource } from "./wgsl.js";

const LIMIT_BYTES = 64 * 1024 * 1024;
const UPLOAD_CHUNK_BYTES = 1024 * 1024;
const SENTINEL = 12345.25;
const T = 2;
const cases = [];
let metadata = {};

const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((b) => b.toString(16).padStart(2, "0")).join("");

function exact(a, b) {
  const left = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const right = new Uint32Array(b.buffer, b.byteOffset, b.length);
  let bitMismatches = 0, nonfinite = 0, maxAbs = 0, firstDifference = null;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) nonfinite++;
    else maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
    if (left[i] !== right[i]) {
      bitMismatches++;
      firstDifference ||= { index: i, baseline: a[i], candidate: b[i], baselineBits: left[i], candidateBits: right[i] };
    }
  }
  return { ok: a.length === b.length && bitMismatches === 0 && nonfinite === 0, samples: a.length, bitMismatches, nonfinite, maxAbs, firstDifference, signedZeroChecked: true };
}

export function logicalLaneProof(k4) {
  const coverage = new Uint8Array(k4);
  for (let logicalLane = 0; logicalLane < 64; logicalLane++) {
    const physicalLane = logicalLane & 31;
    const high = logicalLane >= 32;
    const original = [];
    for (let part = logicalLane; part < k4; part += 64) original.push(part);
    const mapped = [];
    for (let offset = 0; offset < k4; offset += 64) {
      const part = physicalLane + (high ? 32 : 0) + offset;
      if (part < k4) mapped.push(part);
    }
    if (original.length !== mapped.length || original.some((part, index) => part !== mapped[index])) {
      throw new Error(`logical K mapping differs for lane ${logicalLane}, K4=${k4}`);
    }
    for (const part of mapped) coverage[part]++;
  }
  if (coverage.some((count) => count !== 1)) throw new Error(`logical K mapping is not a partition for K4=${k4}`);

  const originalTree = Array.from({ length: 64 }, (_, lane) => `lane${lane}`);
  for (let stride = 32; stride > 0; stride >>= 1) {
    for (let lane = 0; lane < stride; lane++) originalTree[lane] = `(${originalTree[lane]}+${originalTree[lane + stride]})`;
  }
  const compactTree = Array.from({ length: 32 }, (_, lane) => `(lane${lane}+lane${lane + 32})`);
  for (let stride = 16; stride > 0; stride >>= 1) {
    for (let lane = 0; lane < stride; lane++) compactTree[lane] = `(${compactTree[lane]}+${compactTree[lane + stride]})`;
  }
  if (originalTree[0] !== compactTree[0]) throw new Error(`reduction tree differs for K4=${k4}`);
  return { K4: k4, originalLogicalLanes: 64, physicalSubgroupLanes: 32, everyPartOnce: true, perLanePartOrderPreserved: true, firstTreeAdditionPreserved: true, finalTreeEqual: true };
}

const proofK = [32, 1056, 1536, 6144, 12288];
const cpuProof = proofK.map((K) => logicalLaneProof(K / 4));

function replaceExactly(code, before, after, label) {
  if (code.split(before).length !== 2) throw new Error(`compact-lane source marker mismatch: ${label}`);
  return code.replace(before, after);
}

function replaceLoop(code, replacement) {
  const marker = "    for (var part = lane;";
  if (code.indexOf(marker) < 0 || code.indexOf(marker) !== code.lastIndexOf(marker)) throw new Error("compact-lane source marker mismatch: K loop");
  const start = code.indexOf(marker);
  const opening = code.indexOf("{", start);
  if (opening < 0) throw new Error("compact-lane K loop has no body");
  let depth = 0, end = -1;
  for (let index = opening; index < code.length; index++) {
    if (code[index] === "{") depth++;
    else if (code[index] === "}" && --depth === 0) { end = index + 1; break; }
  }
  if (end < 0) throw new Error("compact-lane K loop is unbalanced");
  return code.slice(0, start) + replacement + code.slice(end);
}

export function compactLaneCode(original, kind) {
  const partial = kind === "plain" ? "pair_partials" : "partials";
  const width = kind === "plain" ? 2 : 4;
  const accumulator = kind === "plain" ? "sum" : "values";
  const zero = `vec${width}<f32>(0.0)`;
  if (original.includes("enable f16;") || !original.includes("enable subgroups;") || !original.includes("@workgroup_size(256)")) {
    throw new Error("compact-lane guard requires the registered F32 WG256 subgroup kernel");
  }

  let code = original;
  code = replaceExactly(code, `var<workgroup> ${partial}: array<vec${width}<f32>, 256>;\n`, "", `${partial} declaration`);
  code = replaceExactly(code, "@compute @workgroup_size(256)", "@compute @workgroup_size(128)", "workgroup size");
  code = replaceExactly(code, "  let lane = li & 63u;\n  let col = wg.x * 4u + li / 64u;",
    "  let lane = li & 31u;\n  let highLane = lane + 32u;\n  let col = wg.x * 4u + li / 32u;", "logical lane and output column mapping");
  code = replaceExactly(code, `  var ${accumulator} = ${zero};`, `  var low = ${zero};\n  var high = ${zero};`, "accumulators");

  if (kind === "plain") {
    code = replaceLoop(code, `    for (var offset = 0u; offset < k4; offset += 64u) {
      let part = lane + offset;
      if (part < k4) {
        let packed = W[col * (K / 16u) + part / 4u];
        let word = packed[part & 3u];
        let weight = sx(word) * S[col * (K / 32u) + part / 8u];
        low += vec2<f32>(dot(X[part], weight), dot(X[k4 + part], weight));
      }
      let highPart = highLane + offset;
      if (highPart < k4) {
        let packed = W[col * (K / 16u) + highPart / 4u];
        let word = packed[highPart & 3u];
        let weight = sx(word) * S[col * (K / 32u) + highPart / 8u];
        high += vec2<f32>(dot(X[highPart], weight), dot(X[k4 + highPart], weight));
      }
    }`);
    code = replaceExactly(code, `  ${partial}[li] = ${accumulator};
  workgroupBarrier();
  var prefix = ${partial}[li];
  if (lane < 32u) { prefix += ${partial}[li + 32u]; }`, "  let prefix = low + high;", "plain logical-lane combine");
    if (!code.includes("let reduced = ordered_vector_sum32(prefix, subgroupLane);")) throw new Error("compact-lane plain subgroup tree marker mismatch");
    if (!code.includes("var value = reduced[row];")) throw new Error("compact-lane plain output marker mismatch");
    if (code.includes(`${partial}[`) || (code.match(/workgroupBarrier\(\)/g) || []).length !== 0) {
      throw new Error("compact-lane plain transform retained shared partials or a barrier");
    }
    return code;
  }

  code = replaceExactly(code, "  let K = p.K;\n  let lane = li & 31u;", "  let K = p.K;\n  let k4 = K / 4u;\n  let lane = li & 31u;", "GELU K4");
  code = replaceLoop(code, `    for (var offset = 0u; offset < k4; offset += 64u) {
      let part = lane + offset;
      if (part < k4) {
        let address = col * (K / 16u) + part / 4u;
        let gate_pack = G[address];
        let up_pack = UP[address];
        let gate_word = gate_pack[part & 3u];
        let up_word = up_pack[part & 3u];
        let gate_weight = sx(gate_word) * GS[col * (K / 32u) + part / 8u];
        let up_weight = sx(up_word) * US[col * (K / 32u) + part / 8u];
        let input0 = X[part];
        let input1 = X[k4 + part];
        low += vec4<f32>(dot(input0, gate_weight), dot(input0, up_weight), dot(input1, gate_weight), dot(input1, up_weight));
      }
      let highPart = highLane + offset;
      if (highPart < k4) {
        let address = col * (K / 16u) + highPart / 4u;
        let gate_pack = G[address];
        let up_pack = UP[address];
        let gate_word = gate_pack[highPart & 3u];
        let up_word = up_pack[highPart & 3u];
        let gate_weight = sx(gate_word) * GS[col * (K / 32u) + highPart / 8u];
        let up_weight = sx(up_word) * US[col * (K / 32u) + highPart / 8u];
        let input0 = X[highPart];
        let input1 = X[k4 + highPart];
        high += vec4<f32>(dot(input0, gate_weight), dot(input0, up_weight), dot(input1, gate_weight), dot(input1, up_weight));
      }
    }`);
  code = replaceExactly(code, `  ${partial}[li] = ${accumulator};
  workgroupBarrier();
  var prefix = ${partial}[li];
  if (lane < 32u) { prefix += ${partial}[li + 32u]; }`, "  let prefix = low + high;", "GELU logical-lane combine");
  if (!code.includes("let result = reduced;")) throw new Error("compact-lane GELU output marker mismatch");
  if (code.includes(`${partial}[`) || (code.match(/workgroupBarrier\(\)/g) || []).length !== 0) {
    throw new Error("compact-lane transform retained shared partials or a barrier");
  }
  return code;
}

function pairLayout(device) {
  const types = ["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage"];
  return device.createBindGroupLayout({ label: "compact-lane paired matvec GELU", entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
}

async function compile(device, source, kind) {
  const name = kind === "plain" ? "matvec_two" : "matvec_gelu_two";
  const spec = { weights: "q8", f16: false, rows: 1, groups: 1, splitTarget: 256, subgroups: true };
  const baselineCode = source(name, spec);
  if (!baselineCode.includes("ordered_vector_sum32") || !baselineCode.includes("enable subgroups;") || (baselineCode.match(/workgroupBarrier\(\)/g) || []).length !== 1) {
    throw new Error(`registered ${name} subgroup-vector baseline is unavailable; rebuild the WASM first`);
  }
  const candidateCode = compactLaneCode(baselineCode, kind);
  if ((candidateCode.match(/@workgroup_size\(128\)/g) || []).length !== 1 || !candidateCode.includes("let prefix = low + high;")) {
    throw new Error(`derived ${name} compact-lane markers are unavailable`);
  }
  const layout = kind === "plain" ? matmulLayout(device) : pairLayout(device);
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const [baselineHash, candidateHash] = await Promise.all([hash(baselineCode), hash(candidateCode)]);
  const [baseline, candidate] = await Promise.all([
    pipeline(device, baselineCode, `${name}.registered.subgroup32`, pipelineLayout),
    pipeline(device, candidateCode, `${name}.derived.compact64to32`, pipelineLayout),
  ]);
  return { kind, name, layout, baseline, candidate, baselineCode, candidateCode, shaderHashes: { registeredBaseline: baselineHash, candidate: candidateHash } };
}

async function drain(device, deadline, label) {
  const remaining = deadline - performance.now();
  if (remaining <= 0) throw new Error(`${label} deadline exceeded`);
  let timer;
  try {
    await Promise.race([device.queue.onSubmittedWorkDone(), new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${label} deadline exceeded`)), remaining);
    })]);
  } finally { clearTimeout(timer); }
}

async function uploadBuffer(device, buffer, byteLength, type, offset, fixture, deadline) {
  const maxElements = type === "i8" ? UPLOAD_CHUNK_BYTES : UPLOAD_CHUNK_BYTES / 4;
  const stageBuffer = uploadBuffer.stage;
  const stage = type === "i8" ? new Int8Array(stageBuffer) : new Float32Array(stageBuffer);
  const length = type === "i8" ? byteLength : byteLength / 4;
  for (let start = 0; start < length; start += maxElements) {
    const count = Math.min(maxElements, length - start);
    for (let j = 0; j < count; j++) {
      const index = start + j;
      if (type === "i8") {
        if (fixture === "saturation") stage[j] = 127;
        else if (fixture === "cancellation") stage[j] = offset ? -3 : 1;
        else stage[j] = (index * 31 + Math.floor(index / uploadBuffer.K) * 17 + offset) % 256 - 128;
      } else {
        if (fixture === "saturation") stage[j] = 0.05;
        else if (fixture === "cancellation") stage[j] = offset ? 0.0021 : 0.0037;
        else stage[j] = index % 257 === 0 ? -0 : (index % 2 ? -1 : 1) * (0.0002 + (index % 7) * 0.000037);
      }
    }
    const view = stage.subarray(0, count);
    device.queue.writeBuffer(buffer, type === "i8" ? start : start * 4, view);
    await drain(device, deadline, "compact-lane weight upload");
  }
}

uploadBuffer.stage = new ArrayBuffer(UPLOAD_CHUNK_BYTES);
uploadBuffer.K = 1;

function fixtureInput(K, fixture) {
  const values = new Float32Array(T * K + 4).fill(NaN);
  for (let index = 0; index < T * K; index++) {
    const row = Math.floor(index / K), col = index % K;
    if (fixture === "zeros") values[index] = row ? -0 : 0;
    else if (fixture === "saturation") values[index] = row ? -0.36 : 0.32;
    else if (fixture === "cancellation") values[index] = Math.fround((col % 8 < 4 ? 1024 : -1024) + (col % 4 + 1) * 0.03125 + row * 0.015625);
    else values[index] = Math.fround(0.037 * Math.sin(col * 0.127 + row * 1.731 + 0.3) + 0.011 * Math.cos(col * 0.071 - row * 0.417));
  }
  return values;
}

function fixtureBias(N, bias, fixture) {
  const values = new Float32Array(Math.max(4, N));
  for (let index = 0; index < values.length; index++) {
    if (!bias) values[index] = NaN;
    else if (fixture === "zeros") values[index] = index & 1 ? -0 : 0;
    else values[index] = Math.fround(0.003 * Math.sin(index + 0.1));
  }
  return values;
}

function initialOutput(N, mode, fixture) {
  const values = new Float32Array(T * N + 4).fill(NaN);
  if (mode === 1) for (let i = 0; i < T * N; i++) values[i] = fixture === "zeros" ? (i & 1 ? -0 : 0) : Math.fround(0.007 * Math.cos(i * 0.41));
  values.fill(SENTINEL, T * N);
  return values;
}

async function check(runtime, pipes, c) {
  const { device } = runtime;
  const U = GPUBufferUsage;
  const { N, K, mode = 0, bias = 0, fixture = "normal", timed = false } = c;
  const count = T * N, outputFloats = count + 4;
  const xBytes = (T * K + 4) * 4, yBytes = outputFloats * 4;
  const weightBytes = N * K, scaleBytes = N * K / 32 * 4;
  const biasBytes = Math.max(16, N * 4), readBytes = Math.max(256, yBytes * 2);
  const hostUploadBytes = UPLOAD_CHUNK_BYTES;
  const xValues = fixtureInput(K, fixture), biasValues = fixtureBias(N, bias, fixture), initialValues = initialOutput(N, mode, fixture);
  const hostWorkingBytes = hostUploadBytes + xValues.byteLength + biasValues.byteLength + initialValues.byteLength + readBytes + 96 + 16384;
  const deadline = performance.now() + 90000;
  const owned = [];
  let read, resolved, query, mapped = false;
  let allocatedGpuBytes = 0;
  const make = (size, usage, label) => {
    const actualSize = Math.max(16, size);
    const value = device.createBuffer({ label: `matvec-lanes.${label}`, size: actualSize, usage });
    owned.push(value);
    allocatedGpuBytes += actualSize;
    return value;
  };
  const workgroupCount = Math.ceil(N / 4);
  const weightGpuBytes = (pipes.kind === "gelu" ? 2 : 1) * (Math.max(16, weightBytes) + Math.max(16, scaleBytes));
  const expectedGpuBufferBytes = weightGpuBytes + 16 + 16 + xBytes + biasBytes + 2 * yBytes
    + (pipes.kind === "plain" ? 16 : 0) + readBytes + (timed ? 256 : 0);
  const expectedQuerySetBytes = timed ? 12 * 8 : 0;
  if (expectedGpuBufferBytes + hostWorkingBytes + expectedQuerySetBytes > LIMIT_BYTES) {
    throw new Error(`compact-lane ${c.shape} planned controlled memory ${expectedGpuBufferBytes + hostWorkingBytes + expectedQuerySetBytes} exceeds 64 MiB`);
  }
  if (Math.max(xBytes, yBytes, biasBytes, readBytes, weightBytes, scaleBytes) > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize)) {
    throw new Error(`compact-lane ${c.shape} exceeds device storage-buffer limits`);
  }
  if (workgroupCount > device.limits.maxComputeWorkgroupsPerDimension) throw new Error(`compact-lane ${c.shape} exceeds the workgroup-count limit`);

  try {
    const weight = async (offset, suffix) => {
      const values = make(weightBytes, U.STORAGE | U.COPY_DST, `${suffix}.q8`);
      const scales = make(scaleBytes, U.STORAGE | U.COPY_DST, `${suffix}.scales`);
      uploadBuffer.K = K;
      await uploadBuffer(device, values, weightBytes, "i8", offset, fixture, deadline);
      await uploadBuffer(device, scales, scaleBytes, "f32", offset, fixture, deadline);
      return { values, scales };
    };
    const gate = await weight(0, pipes.kind === "gelu" ? "gate" : "weight");
    const up = pipes.kind === "gelu" ? await weight(19, "up") : null;
    const globals = make(16, U.UNIFORM | U.COPY_DST, "globals");
    const params = make(16, U.UNIFORM | U.COPY_DST, "params");
    const x = make(xBytes, U.STORAGE | U.COPY_DST, "input");
    const b = make(biasBytes, U.STORAGE | U.COPY_DST, "bias");
    const baselineY = make(yBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "baseline-output");
    const candidateY = make(yBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "candidate-output");
    const part = pipes.kind === "plain" ? make(16, U.STORAGE, "unused-partials") : null;
    read = make(readBytes, U.MAP_READ | U.COPY_DST, "readback");
    device.queue.writeBuffer(globals, 0, new Uint32Array([T, 0, 0, 0]));
    device.queue.writeBuffer(params, 0, new Uint32Array([N, K, mode, bias]));
    device.queue.writeBuffer(x, 0, xValues);
    device.queue.writeBuffer(b, 0, biasValues);
    device.queue.writeBuffer(baselineY, 0, initialValues);
    device.queue.writeBuffer(candidateY, 0, initialValues);
    await drain(device, deadline, "compact-lane input upload");

    const bind = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const buffersFor = (output) => pipes.kind === "plain"
      ? [globals, params, x, gate.values, gate.scales, b, output, part]
      : [globals, params, x, gate.values, gate.scales, up.values, up.scales, output];
    const groups = [baselineY, candidateY].map((output) => bind(pipes.layout, buffersFor(output)));
    const dispatch = (pass, candidate) => {
      pass.setPipeline(candidate ? pipes.candidate : pipes.baseline);
      pass.setBindGroup(0, groups[candidate ? 1 : 0]);
      pass.dispatchWorkgroups(workgroupCount, 1, 1);
    };
    const copyOutputs = (encoder) => {
      encoder.copyBufferToBuffer(baselineY, 0, read, 0, yBytes);
      encoder.copyBufferToBuffer(candidateY, 0, read, yBytes, yBytes);
    };
    const readOutputs = async (copyBytes = readBytes) => {
      const within = await mapBeforeDeadline(read, copyBytes, deadline);
      mapped = true;
      if (!within) throw new Error(`compact-lane ${c.shape} output-map deadline exceeded`);
      const copy = read.getMappedRange(0, copyBytes).slice(0);
      const values = new Float32Array(copy);
      read.unmap(); mapped = false;
      return values;
    };
    const checkValues = (values) => {
      const stride = yBytes / 4;
      const baselineValues = values.subarray(0, count), candidateValues = values.subarray(stride, stride + count);
      const outputCheck = exact(baselineValues, candidateValues);
      const baselineFinite = baselineValues.every(Number.isFinite), candidateFinite = candidateValues.every(Number.isFinite);
      const baselinePadding = values.subarray(count, stride).every((value) => value === SENTINEL);
      const candidatePadding = values.subarray(stride + count, stride * 2).every((value) => value === SENTINEL);
      let fixtureCheck = true;
      if (pipes.kind === "gelu" && fixture === "saturation") {
        fixtureCheck = values[0] > 10 && values[N] === 0 && values[stride] > 10 && values[stride + N] === 0;
      }
      return { ok: outputCheck.ok && baselineFinite && candidateFinite && baselinePadding && candidatePadding && fixtureCheck,
        outputCheck, baselineFinite, candidateFinite, baselinePadding, candidatePadding, fixtureCheck, outputTailFloats: 4 };
    };
    const checkEncoder = device.createCommandEncoder();
    const checkPass = checkEncoder.beginComputePass(); dispatch(checkPass, false); dispatch(checkPass, true); checkPass.end();
    copyOutputs(checkEncoder); device.queue.submit([checkEncoder.finish()]);
    const before = checkValues(await readOutputs());
    if (!before.ok) return { ...c, kind: pipes.kind, ok: false, before, reason: "before-timing-exactness-failed", allocatedGpuBytes, hostWorkingBytes };

    let timing = null, after = null;
    if (timed) {
      query = device.createQuerySet({ type: "timestamp", count: 12 });
      resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamp-resolve");
      const encoder = device.createCommandEncoder();
      const warm = encoder.beginComputePass(); dispatch(warm, false); dispatch(warm, true); warm.end();
      const indices = { baseline: [], candidate: [] };
      let queryIndex = 0;
      for (let pair = 0; pair < 3; pair++) for (const candidate of pair % 2 ? [true, false] : [false, true]) {
        indices[candidate ? "candidate" : "baseline"].push(queryIndex);
        const pass = encoder.beginComputePass({ timestampWrites: { querySet: query, beginningOfPassWriteIndex: queryIndex, endOfPassWriteIndex: queryIndex + 1 } });
        for (let rep = 0; rep < 4; rep++) dispatch(pass, candidate);
        pass.end(); queryIndex += 2;
      }
      encoder.resolveQuerySet(query, 0, 12, resolved, 0);
      encoder.copyBufferToBuffer(resolved, 0, read, 0, 96);
      device.queue.submit([encoder.finish()]);
      const stampWithin = await mapBeforeDeadline(read, 96, deadline);
      mapped = true;
      if (!stampWithin) throw new Error(`compact-lane ${c.shape} timestamp-map deadline exceeded`);
      const stamps = new BigUint64Array(read.getMappedRange(0, 96).slice(0));
      read.unmap(); mapped = false;
      const samples = Object.fromEntries(Object.entries(indices).map(([label, positions]) => [label,
        positions.map((index) => Number(stamps[index + 1] - stamps[index]) / 1e6 / 4)]));
      const selection = selectMatmulKernel({ generic: samples.baseline, wide: samples.candidate });
      timing = { ...samples, pairs: 3, repetitions: 4, baselineMs: selection.genericMs, candidateMs: selection.wideMs,
        benefit: selection.benefit, stable: selection.stable, selectedByThreshold: selection.kernel === "wide", selectionReason: selection.reason };
      const finalEncoder = device.createCommandEncoder(); copyOutputs(finalEncoder); device.queue.submit([finalEncoder.finish()]);
      after = checkValues(await readOutputs());
    }

    const querySetBytes = query ? 12 * 8 : 0;
    const controlledBytes = allocatedGpuBytes + hostWorkingBytes + querySetBytes;
    if (controlledBytes > LIMIT_BYTES) throw new Error(`compact-lane ${c.shape} controlled memory ${controlledBytes} exceeds 64 MiB`);
    return { ...c, kind: pipes.kind, ok: before.ok && (!after || after.ok), before, after, timing,
      memory: { limitBytes: LIMIT_BYTES, controlledBytes, gpuBufferBytes: allocatedGpuBytes, querySetBytes,
        hostWorkingBytes, uploadChunkBytes: UPLOAD_CHUNK_BYTES, immutableWeightInputGpuBytes: weightGpuBytes,
        gpuScratchExcludingWeightInputsBytes: allocatedGpuBytes - weightGpuBytes },
      shaderHashes: pipes.shaderHashes };
  } finally {
    if (mapped) read?.unmap();
    query?.destroy();
    for (const buffer of owned) buffer.destroy();
  }
}

const actualCases = [
  { shape: "gelu-gate-wide", kind: "gelu", N: 12288, K: 1536, timed: true },
  { shape: "plain-down-wide", kind: "plain", N: 1536, K: 12288, timed: true },
  { shape: "gelu-gate", kind: "gelu", N: 6144, K: 1536, timed: true },
  { shape: "plain-down", kind: "plain", N: 1536, K: 6144, timed: true },
];
const modeBiasCases = [];
for (const K of [1056, 32]) for (const mode of [0, 1, 2]) for (const bias of [0, 1]) {
  modeBiasCases.push({ shape: K === 32 ? "plain-zero-padding" : "plain-mode-bias", kind: "plain", N: 17, K, mode, bias,
    fixture: K === 32 ? "zeros" : "normal" });
}
const geluFixtures = ["normal", "zeros", "cancellation", "saturation"].map((fixture) => ({
  shape: `gelu-ragged-${fixture}`, kind: "gelu", N: 17, K: 1056, fixture,
}));

async function main() {
  const started = performance.now();
  const params = new URLSearchParams(location.search);
  const shape = params.get("shape") || "all";
  const allowed = new Set(["all", ...actualCases.map((c) => c.shape), ...modeBiasCases.map((c) => c.shape), ...geluFixtures.map((c) => c.shape)]);
  if (!allowed.has(shape)) throw new Error(`unknown shape ${shape}; use shape=all or one of ${[...allowed].join(",")}`);
  const selected = shape === "all" ? [...actualCases, ...modeBiasCases, ...geluFixtures] : [...actualCases, ...modeBiasCases, ...geluFixtures].filter((c) => c.shape === shape);
  const runtime = await requestDevice({ features: ["timestamp-query", "subgroups"] });
  try {
    metadata = { precision: "f32", T, baseline: "registered subgroup32 WG256 vector reduction", candidate: "derived WG128 with two ordered logical-64 accumulators per physical lane",
      shape, subgroup32: runtime.subgroup32, adapter: { name: runtime.name, features: [...runtime.device.features], subgroupMinSize: runtime.adapter.info?.subgroupMinSize, subgroupMaxSize: runtime.adapter.info?.subgroupMaxSize },
      cpuProof };
    if (!runtime.subgroup32 || !runtime.device.features.has("subgroups") || !runtime.device.features.has("timestamp-query") || runtime.device.features.has("shader-f16")) {
      throw Object.assign(new Error("compact-lane probe requires fixed32 subgroups and timestamp-query without shader-f16"), { skip: true });
    }
    const source = await kernelSource();
    metadata.wasmUrl = source.wasmUrl;
    const pipes = new Map();
    for (const input of selected) {
      if (!pipes.has(input.kind)) {
        pipes.set(input.kind, await withGpuErrors(runtime, () => compile(runtime.device, source, input.kind), `compiling compact-lane ${input.kind}`));
        metadata[`registered${input.kind === "plain" ? "MatvecTwo" : "MatvecGeluTwo"}Sha256`] = pipes.get(input.kind).shaderHashes.registeredBaseline;
      }
      log({ status: "checking", shape: input.shape, kind: input.kind, N: input.N, K: input.K, mode: input.mode ?? 0, bias: input.bias ?? 0, fixture: input.fixture || "normal" });
      const result = await withGpuErrors(runtime, () => check(runtime, pipes.get(input.kind), input), "compact-lane exact/timing guard");
      cases.push(result); log(result);
      if (!result.ok) throw new Error(`compact-lane output/padding guard failed for ${input.shape}`);
      if (result.timing && result.timing.baselineMs === null) throw new Error(`compact-lane GPU timestamps were invalid for ${input.shape}`);
    }
    const timed = cases.filter((value) => value.timing);
    window.gpuBench = { status: "done", done: true, backend: "webgpu", ...metadata,
      metricMs: timed.length ? Math.exp(timed.reduce((sum, value) => sum + Math.log(value.timing.candidateMs), 0) / timed.length) : performance.now() - started,
      method: "registered fixed-32 F32 two-row subgroup-vector kernels versus derived WG128 logical-64-to-32 register mapping; full output bit/finiteness/padding checks before and after 3 alternating GPU-timestamp pairs x4 reps",
      elapsedMs: performance.now() - started, correctness: { ok: true, bitExact: true, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

export { cpuProof };

if (typeof window !== "undefined") {
  window.gpuBench = { status: "running", backend: "webgpu", cpuProof };
  main().catch((error) => {
    window.gpuBench = { status: error.skip ? "skipped" : "error", done: true, backend: "webgpu", ...metadata,
      ...(error.skip ? { skipped: true, skipReason: String(error.message) } : {}), correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) };
    log(error.stack || String(error));
  });
}
