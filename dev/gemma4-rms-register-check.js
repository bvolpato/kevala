// DEV-only exactness and timestamp guard for retaining the six D=1536 updates in registers.
// Candidates are derived from the registered rendered shaders; this page loads no model.
import { pipeline, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const U = GPUBufferUsage;
const MAP_READ = GPUMapMode.READ;
const D = 1536;
const TOKENS = [2, 12, 97];
const MODES = [0, 1, 2];
const PATTERNS = ["normal", "signed-zero", "cancellation", "large-finite"];
const TAIL = 4;
const SENTINEL = 12345.25;
const MEMORY_CAP = 64 * 1024 * 1024;
const query = new URLSearchParams(location.search);
const useSubgroups = query.get("subgroups") === "1";
const logNode = document.getElementById("log");
const log = (value) => { logNode.textContent += `${typeof value === "string" ? value : JSON.stringify(value)}\n`; };
const f32 = Math.fround;

function once(source, before, after, label) {
  const count = source.split(before).length - 1;
  if (count !== 1) throw new Error(`${label}: expected exactly one source marker, found ${count}`);
  return source.replace(before, after);
}

function retainedStatements(kind) {
  const output = [];
  for (let i = 0; i < 6; i++) {
    const d = `d${i}`;
    output.push(`  let ${d} = lane + ${i * 256}u;`);
    if (kind === "residualNorm") {
      output.push(`  let factor${i} = select(1.0, Wpost[${d}], p.mode == 1u || p.mode == 2u);`);
      output.push(`  let normalized${i} = X[base + ${d}] * inv1;`);
      output.push(`  let weighted${i} = normalized${i} * factor${i};`);
      output.push(`  var updated${i}: f32;`);
      output.push(`  if (p.mode == 2u) {`);
      output.push(`    updated${i} = Y[base + ${d}] + weighted${i};`);
      output.push("  } else {");
      output.push(`    updated${i} = weighted${i};`);
      output.push("  }");
      output.push(`  Y[base + ${d}] = updated${i};`);
      output.push(`  sum2 += updated${i} * updated${i};`);
    } else {
      output.push(`  let factor${i} = select(1.0, W[${d}], p.mode == 1u || p.mode == 2u);`);
      output.push(`  let normalized${i} = X[base + ${d}] * inv;`);
      output.push(`  let weighted${i} = normalized${i} * factor${i};`);
      output.push(`  var preScale${i}: f32;`);
      output.push(`  if (p.mode == 2u) {`);
      output.push(`    preScale${i} = Y[base + ${d}] + weighted${i};`);
      output.push("  } else {");
      output.push(`    preScale${i} = weighted${i};`);
      output.push("  }");
      output.push(`  let updated${i} = preScale${i} * scalar;`);
      output.push(`  Y[base + ${d}] = updated${i};`);
      output.push(`  sum2 += updated${i} * updated${i};`);
    }
  }
  return output.join("\n");
}

function originalChunk(kind) {
  if (kind === "residualNorm") {
    return `  for (var d = lane; d < p.D; d += 256u) {
    let factor = select(1.0, Wpost[d], p.mode == 1u || p.mode == 2u);
    let normalized = X[base + d] * inv1;
    let weighted = normalized * factor;
    if (p.mode == 2u) {
      Y[base + d] = Y[base + d] + weighted;
    } else {
      Y[base + d] = weighted;
    }
  }
  storageBarrier();
  var sum2 = 0.0;
  for (var d = lane; d < p.D; d += 256u) {
    let updated = Y[base + d];
    sum2 += updated * updated;
  }`;
  }
  return `  for (var d = lane; d < p.D; d += 256u) {
    let factor = select(1.0, W[d], p.mode == 1u || p.mode == 2u);
    let normalized = X[base + d] * inv;
    let weighted = normalized * factor;
    if (p.mode == 2u) {
      let updated = Y[base + d] + weighted;
      Y[base + d] = updated * scalar;
    } else {
      Y[base + d] = weighted * scalar;
    }
  }
  storageBarrier();
  var sum2 = 0.0;
  for (var d = lane; d < p.D; d += 256u) {
    let updated = Y[base + d];
    sum2 += updated * updated;
  }`;
}

function originalOutput(kind) {
  if (kind === "residualNorm") {
    return `  for (var d = lane; d < p.D; d += 256u) {
    let normalized = Y[base + d] * invResidual;
    H[base + d] = normalized * Wnext[d];
  }`;
  }
  return `  for (var d = lane; d < p.D; d += 256u) {
    let normalized = Y[base + d] * invResidual;
    H[base + d] = normalized * Wnext[d];
  }`;
}

function retainedOutputWithKind() {
  const output = [];
  for (let i = 0; i < 6; i++) {
    const d = `d${i}`;
    output.push(`  let normalizedOut${i} = updated${i} * invResidual;`);
    output.push(`  H[base + ${d}] = normalizedOut${i} * Wnext[${d}];`);
  }
  return output.join("\n");
}

function proveLaneMapping() {
  const owners = new Uint8Array(D);
  let ordered = true;
  for (let lane = 0; lane < 256; lane++) {
    let previous = -1;
    for (let i = 0; i < 6; i++) {
      const d = lane + 256 * i;
      if (d <= previous || d >= D) ordered = false;
      previous = d;
      owners[d]++;
    }
  }
  return {
    D, lanes: 256, valuesPerLane: 6,
    everyElementOwnedOnce: owners.every((count) => count === 1),
    perLaneDAscending: ordered,
    accumulationOrder: "d=lane, lane+256, lane+512, lane+768, lane+1024, lane+1280",
  };
}

function deriveCandidate(source, kind) {
  const invMarker = kind === "residualNorm" ? "  let inv1 = inverseSqrt(total / f32(p.D) + p.eps);" : "  let inv = inverseSqrt(total / f32(p.D) + p.eps);";
  const invAt = source.indexOf(invMarker);
  if (invAt < 0) throw new Error(`${kind}: first RMS inverse marker was not found`);
  const afterInverse = invAt + invMarker.length;
  const oldChunk = originalChunk(kind);
  const chunkStart = source.indexOf(oldChunk);
  if (chunkStart < afterInverse || source.indexOf(oldChunk, chunkStart + 1) !== -1) {
    throw new Error(`${kind}: registered update/storage-reload block did not match the reviewed source marker`);
  }
  let code = source.slice(0, chunkStart) + `  var sum2 = 0.0;\n${retainedStatements(kind)}` + source.slice(chunkStart + oldChunk.length);
  const oldOutput = originalOutput(kind);
  const outputAt = code.lastIndexOf(oldOutput);
  if (outputAt < 0 || code.indexOf(oldOutput) !== outputAt) throw new Error(`${kind}: expected one final H loop`);
  const newOutput = retainedOutputWithKind();
  code = code.slice(0, outputAt) + newOutput + code.slice(outputAt + oldOutput.length);

  const restoredOutput = once(code, newOutput, oldOutput, `${kind} output reverse transform`);
  const restored = once(restoredOutput, `  var sum2 = 0.0;\n${retainedStatements(kind)}`, oldChunk, `${kind} update reverse transform`);
  const firstTreeIdentical = source.slice(0, chunkStart) === code.slice(0, chunkStart);
  const secondTreeStart = source.indexOf("  sums[lane] = sum2;", chunkStart);
  const outputStart = source.lastIndexOf(oldOutput);
  const candidateSecondTreeStart = code.indexOf("  sums[lane] = sum2;", chunkStart);
  const candidateOutputStart = code.lastIndexOf(newOutput);
  const secondTreeIdentical = secondTreeStart >= 0 && outputStart > secondTreeStart
    && candidateSecondTreeStart >= 0 && candidateOutputStart > candidateSecondTreeStart
    && source.slice(secondTreeStart, outputStart) === code.slice(candidateSecondTreeStart, candidateOutputStart);
  const baselineBarriers = (source.match(/workgroupBarrier\(\);/g) || []).length;
  const candidateBarriers = (code.match(/workgroupBarrier\(\);/g) || []).length;
  const baselineStorageBarriers = (source.match(/storageBarrier\(\);/g) || []).length;
  const candidateStorageBarriers = (code.match(/storageBarrier\(\);/g) || []).length;
  const noAddedPrivateArrays = !/var<private>/.test(code) && !/retained\s*\[/.test(code) && !/array\s*<\s*f32\s*,\s*6\s*>/.test(code);
  const proof = {
    kind, sourceRestoresByteExactly: restored === source,
    firstRmsTreeIdentical: firstTreeIdentical,
    secondRmsTreeIdentical: secondTreeIdentical,
    workgroupBarrierCountPreserved: baselineBarriers === candidateBarriers,
    removedStorageBarrierCount: baselineStorageBarriers - candidateStorageBarriers,
    noAddedPrivateArrays,
    namedRegisters: Array.from({ length: 6 }, (_, i) => `updated${i}`),
    DOnly: D === 1536,
  };
  if (!proof.sourceRestoresByteExactly || !firstTreeIdentical || !secondTreeIdentical
      || !proof.workgroupBarrierCountPreserved || proof.removedStorageBarrierCount !== 1 || !noAddedPrivateArrays) {
    throw new Error(`${kind}: candidate structure proof failed: ${JSON.stringify(proof)}`);
  }
  return { code, proof };
}

function fixtures(T, mode, pattern, seed) {
  const count = T * D;
  const X = new Float32Array(count);
  const residual = new Float32Array(count);
  const W = new Float32Array(D);
  const Wnext = new Float32Array(D);
  for (let d = 0; d < D; d++) {
    if (pattern === "signed-zero") {
      W[d] = d % 2 ? -0 : 0;
      Wnext[d] = d % 2 ? -1 : 1;
    } else if (pattern === "large-finite") {
      W[d] = f32((((d * 7 + seed) % 19) - 9) * 0.127 + 0.19);
      Wnext[d] = f32((((d * 11 + seed + 3) % 17) - 8) * 0.113 + 0.27);
    } else {
      W[d] = f32((((d * 7 + seed * 3) % 19) - 9) * 0.0837);
      Wnext[d] = f32((((d * 11 + seed * 5 + 3) % 17) - 8) * 0.1173);
    }
  }
  for (let i = 0; i < count; i++) {
    const t = Math.floor(i / D);
    const d = i - t * D;
    if (pattern === "signed-zero") {
      X[i] = i % 2 ? -0 : 0;
      residual[i] = i % 3 ? -0 : 0;
    } else if (pattern === "cancellation") {
      const signed = ((d * 13 + t * 7 + seed) % 17) - 8;
      X[i] = f32(signed * 0.03125);
      residual[i] = f32(-X[i] * W[d] * 6.5 * 0.99997);
    } else if (pattern === "large-finite") {
      const signed = ((d * 17 + t * 3 + seed) % 19) - 9;
      X[i] = f32(signed * 1.0e14);
      residual[i] = f32((((d * 5 + t * 11 + seed) % 17) - 8) * 7.5e13);
    } else {
      X[i] = f32((((d * 17 + t * 3 + seed) % 17) - 8) * 0.00437);
      residual[i] = f32((((d * 7 + t * 13 + seed) % 19) - 9) * 0.0713);
    }
  }
  const scalar = pattern === "signed-zero" ? -0 : pattern === "cancellation" ? -1 : pattern === "large-finite" ? 0.25 : -1.337;
  const eps = pattern === "signed-zero" ? 1e-6 : 3.71e-5;
  return { T, D, count, mode, pattern, X, residual, W, Wnext, scalar, eps };
}

function writeUniform(device, values, label, owned, accounting) {
  const data = new ArrayBuffer(16);
  const view = new DataView(data);
  view.setUint32(0, values.D ?? values.T, true);
  view.setFloat32(4, values.eps ?? 0, true);
  view.setUint32(8, values.mode ?? 0, true);
  view.setUint32(12, 0, true);
  return trackedBuffer(device, data.byteLength, U.UNIFORM | U.COPY_DST, data, label, owned, accounting);
}

function trackedBuffer(device, size, usage, data, label, owned, accounting) {
  const actualSize = Math.max(16, size);
  const value = device.createBuffer({ label, size: actualSize, usage });
  owned.push(value);
  accounting.gpuBytes += actualSize;
  if (data) device.queue.writeBuffer(value, 0, data);
  return value;
}

function storage(device, values, label, owned, accounting, extra = U.COPY_DST | U.COPY_SRC) {
  return trackedBuffer(device, values.byteLength, U.STORAGE | extra, values, label, owned, accounting);
}

function bind(device, kernel, values, label) {
  return device.createBindGroup({
    label, layout: kernel.getBindGroupLayout(0),
    entries: values.map((value, binding) => ({ binding, resource: { buffer: value } })),
  });
}

function dispatch(encoder, kernel, group, T, timestamp = null) {
  const options = timestamp ? { timestampWrites: {
    querySet: timestamp.querySet,
    beginningOfPassWriteIndex: timestamp.start,
    endOfPassWriteIndex: timestamp.end,
  } } : undefined;
  const pass = encoder.beginComputePass(options);
  pass.setPipeline(kernel);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(T);
  pass.end();
}

function compareBits(label, baseline, candidate, logicalCount) {
  if (baseline.length !== candidate.length) throw new Error(`${label}: output length mismatch`);
  const a = new Uint32Array(baseline.buffer, baseline.byteOffset, baseline.length);
  const b = new Uint32Array(candidate.buffer, candidate.byteOffset, candidate.length);
  let differences = 0;
  let firstDifference = null;
  let finite = true;
  for (let i = 0; i < baseline.length; i++) {
    if (!Number.isFinite(baseline[i]) || !Number.isFinite(candidate[i])) finite = false;
    if (a[i] !== b[i]) {
      differences++;
      if (firstDifference === null) firstDifference = {
        index: i, baseline: baseline[i], candidate: candidate[i],
        baselineBits: `0x${a[i].toString(16).padStart(8, "0")}`,
        candidateBits: `0x${b[i].toString(16).padStart(8, "0")}`,
      };
    }
  }
  const tailA = baseline.slice(logicalCount);
  const tailB = candidate.slice(logicalCount);
  const tailGuard = tailA.length === TAIL && tailB.length === TAIL
    && [...tailA, ...tailB].every((value) => value === SENTINEL);
  return {
    label, logicalFloats: logicalCount, comparedFloatsIncludingTail: baseline.length,
    finite, bitExact: differences === 0, bitDifferences: differences,
    firstDifference, tailGuard, ok: finite && differences === 0 && tailGuard,
  };
}

async function readFloats(device, source, bytes, readback, label) {
  const encoder = device.createCommandEncoder({ label: `${label}.readback` });
  encoder.copyBufferToBuffer(source, 0, readback, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(MAP_READ, 0, bytes);
  try { return new Float32Array(readback.getMappedRange(0, bytes).slice(0)); }
  finally { readback.unmap(); }
}

async function compareBuffers(device, left, right, bytes, readback, label, logicalCount) {
  const a = await readFloats(device, left, bytes, readback, `${label}.baseline`);
  const b = await readFloats(device, right, bytes, readback, `${label}.candidate`);
  return compareBits(label, a, b, logicalCount);
}

function median(values) {
  const ordered = [...values].sort((a, b) => a - b);
  const middle = ordered.length >> 1;
  return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
}

async function timestampPairs(device, baselineKernel, candidateKernel, baselineGroup, candidateGroup, buffers, input, byteLength, label) {
  const blocks = 3;
  const repetitions = 4;
  const pairs = blocks * repetitions;
  const queryCount = pairs * 4;
  const queryBytes = queryCount * 8;
  const querySet = device.createQuerySet({ type: "timestamp", count: queryCount, label: `${label}.timestamps` });
  const resolved = device.createBuffer({ size: queryBytes, usage: U.QUERY_RESOLVE | U.COPY_SRC, label: `${label}.resolved` });
  const readback = device.createBuffer({ size: queryBytes, usage: U.MAP_READ | U.COPY_DST, label: `${label}.timestamp-readback` });
  const order = [];
  const encoder = device.createCommandEncoder({ label: `${label}.timed-pairs` });
  let q = 0;
  const run = (side) => {
    const y = side === "baseline" ? buffers.oldY : buffers.newY;
    encoder.copyBufferToBuffer(buffers.seed, 0, y, 0, byteLength);
    const start = q;
    q += 2;
    dispatch(encoder, side === "baseline" ? baselineKernel : candidateKernel,
      side === "baseline" ? baselineGroup : candidateGroup, input.T, { querySet, start, end: start + 1 });
    order.push({ side, start, end: start + 1 });
  };
  for (let block = 0; block < blocks; block++) {
    for (let repetition = 0; repetition < repetitions; repetition++) {
      const candidateFirst = (block + repetition) % 2 === 1;
      if (candidateFirst) { run("candidate"); run("baseline"); }
      else { run("baseline"); run("candidate"); }
    }
  }
  if (q !== queryCount) throw new Error(`${label}: timestamp query count mismatch ${q} != ${queryCount}`);
  encoder.resolveQuerySet(querySet, 0, queryCount, resolved, 0);
  encoder.copyBufferToBuffer(resolved, 0, readback, 0, queryBytes);
  try {
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(MAP_READ, 0, queryBytes);
    const stamps = new BigUint64Array(readback.getMappedRange(0, queryBytes).slice(0));
    const durations = { baseline: [], candidate: [] };
    for (const item of order) {
      if (stamps[item.end] <= stamps[item.start]) throw new Error(`${label}: nonpositive timestamp interval ${item.start}:${item.end}`);
      durations[item.side].push(Number(stamps[item.end] - stamps[item.start]) / 1e6);
    }
    return {
      method: "WebGPU timestamp-query; 3 alternating blocks x 4 paired repetitions",
      baseline: "registered shader", candidate: "six named retained F32 registers",
      pairedRuns: pairs,
      baselineMs: durations.baseline, candidateMs: durations.candidate,
      baselineMedianMs: median(durations.baseline), candidateMedianMs: median(durations.candidate),
      speedup: median(durations.baseline) / median(durations.candidate),
      queryBytes, queryResourcesBytes: queryBytes * 3,
    };
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    querySet.destroy(); resolved.destroy(); readback.destroy();
  }
}

async function checkCase(device, pipes, kind, input, timingEnabled) {
  const { T, count } = input;
  const outputFloats = count + TAIL;
  const outputBytes = outputFloats * Float32Array.BYTES_PER_ELEMENT;
  const logicalBytes = count * Float32Array.BYTES_PER_ELEMENT;
  const accounting = { gpuBytes: 0 };
  const seed = new Float32Array(outputFloats).fill(SENTINEL);
  seed.set(input.residual);
  const hSeed = new Float32Array(outputFloats).fill(SENTINEL);
  const fixtureBytes = input.X.byteLength + input.residual.byteLength + input.W.byteLength + input.Wnext.byteLength + 4;
  const uploadSourceBytes = input.X.byteLength + input.W.byteLength + input.Wnext.byteLength + seed.byteLength + hSeed.byteLength + 16 + 16 + (kind === "scaleNext" ? 4 : 0);
  // Include the CPU fixtures, upload-source staging, mapped/readback copies, and timing query storage.
  const hostBytes = fixtureBytes + uploadSourceBytes + 8 * outputBytes + 2048;
  const timed = timingEnabled && input.mode === 2 && input.pattern === "normal";
  const queryResourcesBytes = timed ? (3 * 4 * 4 * 8 * 3) : 0;
  const expectedGpuBytes = 16 + 16 + input.X.byteLength + input.W.byteLength + input.Wnext.byteLength
    + (kind === "scaleNext" ? 16 : 0) + 6 * outputBytes + queryResourcesBytes;
  const expectedCombinedBytes = expectedGpuBytes + hostBytes;
  if (expectedCombinedBytes > MEMORY_CAP) {
    throw new Error(`D=${D} T=${T} ${kind} exceeds 64 MiB controlled bound: GPU=${expectedGpuBytes}, host=${hostBytes}`);
  }
  const owned = [];
  const label = `rms-register.${kind}.T${T}.mode${input.mode}.${input.pattern}`;
  try {
    const globals = trackedBuffer(device, 16, U.UNIFORM | U.COPY_DST, new Uint32Array([T, 0, 0, 0]), `${label}.globals`, owned, accounting);
    const params = writeUniform(device, input, `${label}.params`, owned, accounting);
    const X = storage(device, input.X, `${label}.x`, owned, accounting, U.COPY_DST);
    const W = storage(device, input.W, `${label}.w`, owned, accounting, U.COPY_DST);
    const Wnext = storage(device, input.Wnext, `${label}.wnext`, owned, accounting, U.COPY_DST);
    const S = kind === "scaleNext" ? storage(device, new Float32Array([input.scalar]), `${label}.scalar`, owned, accounting, U.COPY_DST) : null;
    const seedBuffer = trackedBuffer(device, outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, seed, `${label}.seed`, owned, accounting);
    const oldY = trackedBuffer(device, outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, null, `${label}.old-y`, owned, accounting);
    const newY = trackedBuffer(device, outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, null, `${label}.new-y`, owned, accounting);
    const oldH = trackedBuffer(device, outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, hSeed, `${label}.old-h`, owned, accounting);
    const newH = trackedBuffer(device, outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, hSeed, `${label}.new-h`, owned, accounting);
    const readback = trackedBuffer(device, outputBytes, U.MAP_READ | U.COPY_DST, null, `${label}.readback`, owned, accounting);
    const baselineBindings = kind === "residualNorm"
      ? [globals, params, X, W, oldY, Wnext, oldH]
      : [globals, params, X, W, oldY, S, Wnext, oldH];
    const candidateBindings = kind === "residualNorm"
      ? [globals, params, X, W, newY, Wnext, newH]
      : [globals, params, X, W, newY, S, Wnext, newH];
    const baselineGroup = bind(device, pipes[kind].baseline, baselineBindings, `${label}.baseline-bind`);
    const candidateGroup = bind(device, pipes[kind].candidate, candidateBindings, `${label}.candidate-bind`);
    const encoder = device.createCommandEncoder({ label });
    encoder.copyBufferToBuffer(seedBuffer, 0, oldY, 0, outputBytes);
    dispatch(encoder, pipes[kind].baseline, baselineGroup, T);
    encoder.copyBufferToBuffer(seedBuffer, 0, newY, 0, outputBytes);
    dispatch(encoder, pipes[kind].candidate, candidateGroup, T);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const before = {
      Y: await compareBuffers(device, oldY, newY, outputBytes, readback, `${label}.before-timing.Y`, count),
      H: await compareBuffers(device, oldH, newH, outputBytes, readback, `${label}.before-timing.H`, count),
    };
    before.ok = before.Y.ok && before.H.ok;
    if (!before.ok) throw Object.assign(new Error(`${label}: before-output exactness failed`), { failure: before });
    let timing = null;
    let afterTiming = null;
    if (timed) {
      timing = await timestampPairs(device, pipes[kind].baseline, pipes[kind].candidate, baselineGroup, candidateGroup,
        { seed: seedBuffer, oldY, newY }, input, outputBytes, `${label}.timing`);
      afterTiming = {
        Y: await compareBuffers(device, oldY, newY, outputBytes, readback, `${label}.after-timing.Y`, count),
        H: await compareBuffers(device, oldH, newH, outputBytes, readback, `${label}.after-timing.H`, count),
      };
      afterTiming.ok = afterTiming.Y.ok && afterTiming.H.ok;
      if (!afterTiming.ok) throw Object.assign(new Error(`${label}: after-timing exactness failed`), { failure: afterTiming });
    }
    const gpuBytesAccounted = accounting.gpuBytes + (timed ? 3 * 4 * 4 * 8 * 3 : 0);
    if (gpuBytesAccounted !== expectedGpuBytes) throw new Error(`${label}: GPU allocation accounting mismatch: actual=${gpuBytesAccounted}, planned=${expectedGpuBytes}`);
    return {
      kind, T, D, mode: input.mode, pattern: input.pattern,
      eps: input.eps,
      scalar: Object.is(input.scalar, -0) ? "-0" : input.scalar,
      scalarBits: `0x${new Uint32Array(new Float32Array([input.scalar]).buffer)[0].toString(16).padStart(8, "0")}`,
      outputTailFloats: TAIL,
      beforeTiming: before, timing, afterTiming,
      memory: { plannedGpuBytes: expectedGpuBytes, allocatedGpuBytes: gpuBytesAccounted, plannedHostBytes: hostBytes, combinedBytes: expectedCombinedBytes, capBytes: MEMORY_CAP, underBound: expectedCombinedBytes <= MEMORY_CAP },
      ok: before.ok && (!afterTiming || afterTiming.ok) && expectedCombinedBytes <= MEMORY_CAP,
    };
  } finally {
    for (const value of owned) value.destroy();
  }
}

async function sha256(value) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function subgroup32(gpu) {
  const info = gpu.adapter.info || {};
  return gpu.subgroup4 && gpu.device.features.has("subgroups") && info.subgroupMinSize === 32 && info.subgroupMaxSize === 32;
}

function skipped(reason, gpu = null) {
  const state = {
    status: "skipped", done: true, success: false, backend: "webgpu",
    check: "gemma4-rms-register-retention", subgroups: useSubgroups,
    reason, adapter: gpu ? { name: gpu.name, features: [...gpu.device.features] } : null,
  };
  window.gpuBench = state;
  log(state);
}

async function main() {
  const started = performance.now();
  window.gpuBench = { status: "running", done: false, success: false, backend: "webgpu", check: "gemma4-rms-register-retention", subgroups: useSubgroups };
  const gpu = await requestDevice({ features: ["timestamp-query", ...(useSubgroups ? ["subgroups"] : [])] });
  const device = gpu.device;
  const uncaptured = [];
  device.addEventListener("uncapturederror", (event) => uncaptured.push(event.error?.message || String(event.error)));
  try {
    if (!device.features.has("timestamp-query")) return skipped("required timestamp-query feature is unavailable", gpu);
    if (useSubgroups && !subgroup32(gpu)) return skipped("requested subgroup variant requires the subgroups feature and subgroupMinSize=subgroupMaxSize=32", gpu);
    const source = await kernelSource();
    const baseline = {
      residualNorm: source("gemma4_rms_residual_norm", { subgroups: useSubgroups }),
      scaleNext: source("gemma4_rms_scale_next", { subgroups: useSubgroups }),
    };
    const derived = {
      residualNorm: deriveCandidate(baseline.residualNorm, "residualNorm"),
      scaleNext: deriveCandidate(baseline.scaleNext, "scaleNext"),
    };
    const hashes = {
      registeredResidualNorm: await sha256(baseline.residualNorm),
      retainedResidualNorm: await sha256(derived.residualNorm.code),
      registeredScaleNext: await sha256(baseline.scaleNext),
      retainedScaleNext: await sha256(derived.scaleNext.code),
    };
    const laneMapping = proveLaneMapping();
    if (!laneMapping.everyElementOwnedOnce || !laneMapping.perLaneDAscending) throw new Error(`static lane ownership proof failed: ${JSON.stringify(laneMapping)}`);
    const candidateMeta = {
      DOnly: D, T: TOKENS, methods: ["gemma4_rms_residual_norm", "gemma4_rms_scale_next"],
      transformations: {
        residualNorm: derived.residualNorm.proof,
        scaleNext: derived.scaleNext.proof,
        laneMapping,
      },
      arithmetic: "six named F32 values; ascending d accumulation; original first and second RMS trees; original normalized-then-Wnext multiplication order",
      hashes,
      wasmUrl: source.wasmUrl,
    };
    window.gpuBench = { ...window.gpuBench, adapter: { name: gpu.name, features: [...device.features], subgroup32: subgroup32(gpu) }, candidate: candidateMeta };
    log({ adapter: gpu.name, features: [...device.features], candidate: candidateMeta });

    const pipes = await withGpuErrors(gpu, async () => ({
      residualNorm: {
        baseline: await pipeline(device, baseline.residualNorm, "rms-register.baseline-residual-norm"),
        candidate: await pipeline(device, derived.residualNorm.code, "rms-register.retained-residual-norm"),
      },
      scaleNext: {
        baseline: await pipeline(device, baseline.scaleNext, "rms-register.baseline-scale-next"),
        candidate: await pipeline(device, derived.scaleNext.code, "rms-register.retained-scale-next"),
      },
    }), "compiling RMS register-retention pipelines");

    const results = [];
    let index = 0;
    const totalCases = 2 * TOKENS.length * MODES.length * PATTERNS.length;
    for (const kind of ["residualNorm", "scaleNext"]) {
      for (const T of TOKENS) {
        for (const mode of MODES) {
          for (const pattern of PATTERNS) {
            const input = fixtures(T, mode, pattern, (0x4b455641 + index * 0x9e3779b9) >>> 0);
            const result = await withGpuErrors(gpu, () => checkCase(device, pipes, kind, input, true), `checking ${kind} T${T} mode${mode} ${pattern}`);
            results.push(result);
            window.gpuBench = { ...window.gpuBench, status: "running", cases: results.length, expectedCases: totalCases, results };
            log(result);
            if (!result.ok) throw Object.assign(new Error(`${kind} output guard failed for T=${T} mode=${mode} pattern=${pattern}`), { failure: result });
            index++;
          }
        }
      }
    }
    if (uncaptured.length) throw new Error(`uncaptured WebGPU error: ${uncaptured.join("; ")}`);
    const allExact = results.length === totalCases && results.every((result) => result.ok);
    const allBounded = results.every((result) => result.memory.underBound && result.memory.allocatedGpuBytes === result.memory.plannedGpuBytes);
    const timedResults = results.filter((result) => result.timing);
    const allTimed = timedResults.length === 2 * TOKENS.length && timedResults.every((result) => result.afterTiming?.ok);
    const state = {
      ...window.gpuBench,
      status: allExact && allBounded && allTimed ? "done" : "error", done: true, success: allExact && allBounded && allTimed,
      metricMs: performance.now() - started,
      check: "gemma4-rms-register-retention", subgroups: useSubgroups,
      adapter: { name: gpu.name, isFallbackAdapter: Boolean(gpu.adapter.info?.isFallbackAdapter || gpu.adapter.isFallbackAdapter), features: [...device.features], subgroup32: subgroup32(gpu) },
      expectedCases: totalCases, cases: results.length,
      correctness: { ok: allExact, bitExact: allExact, finite: true, signedZeroComparedAsBits: true, tailGuards: true, laneMapping, results: results.map((result) => ({ kind: result.kind, T: result.T, mode: result.mode, pattern: result.pattern, beforeTiming: result.beforeTiming, afterTiming: result.afterTiming })) },
      timingCoverage: { expectedComparisons: 2 * TOKENS.length, completedComparisons: timedResults.length, allAfterTimingGuardsPassed: allTimed },
      timing: timedResults.map((result) => ({ kind: result.kind, T: result.T, pattern: result.pattern, mode: result.mode, ...result.timing })),
      memory: { capBytes: MEMORY_CAP, allCasesWithinBound: allBounded, maxCombinedBytes: Math.max(...results.map((result) => result.memory.combinedBytes)), maxGpuBytes: Math.max(...results.map((result) => result.memory.allocatedGpuBytes)), maxHostBytes: Math.max(...results.map((result) => result.memory.plannedHostBytes)) },
    };
    window.gpuBench = state;
    log({ done: state.done, success: state.success, cases: results.length, expectedCases: totalCases, timing: state.timing, memory: state.memory });
  } finally {
    device.destroy();
  }
}

window.gpuBench = { status: "running", done: false, success: false, backend: "webgpu", check: "gemma4-rms-register-retention", subgroups: useSubgroups };
main().catch((error) => {
  const state = { ...window.gpuBench, status: "error", done: true, success: false, backend: "webgpu", error: String(error?.message || error) };
  if (error?.failure) state.failure = error.failure;
  if (error?.code) state.code = error.code;
  window.gpuBench = state;
  log(error?.stack || error);
});
