import { matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { selectMatmulKernel } from "../js/src/gpu-tuning.js";
import { pairedMatmulGeluPipelines, pairedMatmulGeluBindGroup, encodePairedMatmulGelu } from "./gemma4-matmul-gelu.js";
import { kernelSource } from "./wgsl.js";

const CHUNK = 1024 * 1024, LIMIT = 64 * CHUNK, SENTINEL = 12345.25, TARGET = 256;
const cases = [];
let metadata = {};
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((b) => b.toString(16).padStart(2, "0")).join("");

function serialSplits(original) {
  let code = original;
  const replace = (from, to) => {
    if (code.split(from).length !== 2) throw new Error(`serial split transform marker mismatch: ${from}`);
    code = code.replace(from, to);
  };
  const stores = [...code.matchAll(/fn store_pair\(value: vec2<f32>, row: u32, col: u32, T: u32, N: u32, splits: u32, z: u32\) \{[\s\S]*?\n\}/g)];
  if (stores.length !== 1 || !stores[0][0].includes("PART[(splits + z) * T * N + o] = value.y;")) throw new Error("serial split paired store was not recognized");
  replace(stores[0][0], `fn store_pair(value: vec2<f32>, row: u32, col: u32, T: u32, N: u32, splits: u32, z: u32) {
  if (row >= T || col >= N) { return; }
  var gate = value.x;
  var up = value.y;
  if (splits == 1u) { gate += 0.0; up += 0.0; }
  Y[row * N + col] = gelu(gate) * up;
}`);
  const partition = "  let kb0 = wg.z * per;\n  let kb1 = min(nb, kb0 + per);\n", row = "  let row = m0 + lid.y;";
  if (code.split(partition).length !== 2 || code.split(row).length !== 2) throw new Error("serial split partition or output boundary was not recognized");
  const begin = code.indexOf(partition), end = code.indexOf(row, begin), inner = code.slice(begin + partition.length, end);
  const accumulators = [...inner.matchAll(/var acc(\d) = vec4<f32>\(0\.0\);/g)].map((match) => match[1]);
  if (![7, 8].includes(accumulators.length) || !inner.includes("for (var kb = kb0; kb < kb1; kb++)") || (inner.match(/workgroupBarrier\(\)/g) || []).length !== 2) throw new Error("serial split requires the original R4 K32 loop");
  for (let q = 0; q < 8; q++) if (inner.split(`accumulate(${q}u,`).length !== 2) throw new Error("serial split ordered dot call mismatch");
  const totals = accumulators.map((i) => `  var final${i} = vec4<f32>(0.0);`).join("\n");
  const add = accumulators.map((i) => `    if (splits == 1u) { final${i} = acc${i}; } else { final${i} += acc${i}; }`).join("\n");
  code = code.slice(0, begin) + `${totals}\n  for (var split = 0u; split < splits; split++) {\n    let kb0 = split * per;\n    let kb1 = min(nb, kb0 + per);\n${inner}${add}\n  }\n` + code.slice(end);
  if ((code.match(/store_pair\(acc\d\./g) || []).length !== 2 * accumulators.length) throw new Error("serial split output accumulator count mismatch");
  code = code.replace(/store_pair\(acc(\d)\./g, "store_pair(final$1.");
  if (/PART\s*\[/.test(code)) throw new Error("serial split candidate must not access PART");
  return code;
}

function exact(a, b, offset = 0) {
  const x = new Uint32Array(a.buffer, a.byteOffset, a.length), y = new Uint32Array(b.buffer, b.byteOffset, b.length);
  let bitMismatches = 0, nonfinite = 0, maxAbs = 0, firstDifference = null;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) nonfinite++;
    else maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
    if (x[i] !== y[i]) { bitMismatches++; firstDifference ||= { index: offset + i, baselineBits: x[i], candidateBits: y[i] }; }
  }
  return { ok: a.length === b.length && !bitMismatches && !nonfinite, samples: a.length, bitMismatches, nonfinite, maxAbs, firstDifference, signedZeroChecked: true };
}

async function drain(device, deadline) {
  const remaining = deadline - performance.now();
  if (remaining <= 0) throw new Error("serial MLP upload deadline exceeded");
  let timer;
  try { await Promise.race([device.queue.onSubmittedWorkDone(), new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("serial MLP upload deadline exceeded")), remaining); })]); }
  finally { clearTimeout(timer); }
}

async function compile(device, source, f16) {
  const spec = { weights: "q8", f16, rows: 4, row56: !f16, groups: 1, splitTarget: TARGET };
  const paired = await pairedMatmulGeluPipelines(device, source, spec);
  const candidateCode = serialSplits(paired.mmCode), pl = device.createPipelineLayout({ bindGroupLayouts: [paired.layout] });
  const candidate = await pipeline(device, candidateCode, "mlp_serial_split", pl);
  const hashes = { paired: await hash(paired.mmCode), pairedReduce: await hash(paired.reduceCode), candidate: await hash(candidateCode) };
  let unfused = null;
  if (!f16) {
    const layout = matmulLayout(device), rLayout = reduceLayout(device);
    const mmCode = source("matmul_gemma_batch", spec), reduceCode = source("reduce_gemma_batch", spec), geluCode = source("gemma4_gelu");
    const [mm, reduce, gelu] = await Promise.all([pipeline(device, mmCode, "mlp_serial_unfused_mm", device.createPipelineLayout({ bindGroupLayouts: [layout] })),
      pipeline(device, reduceCode, "mlp_serial_unfused_reduce", device.createPipelineLayout({ bindGroupLayouts: [rLayout] })), pipeline(device, geluCode, "mlp_serial_unfused_gelu")]);
    unfused = { mm, reduce, gelu, layout, rLayout };
    Object.assign(hashes, { unfused: await hash(mmCode), unfusedReduce: await hash(reduceCode), unfusedGelu: await hash(geluCode) });
  }
  return { paired, candidate, unfused, hashes, f16 };
}

async function check(runtime, pipes, c) {
  const { device } = runtime, U = GPUBufferUsage, { T, N, K, fixture = "normal" } = c;
  const splits = mmSplits(1, N, K, TARGET, 64, 64), count = T * N, xBytes = T * K * 4, yBytes = (count + 4) * 4;
  const planeCount = splits > 1 ? splits * count : 0, partBytes = (Math.max(4, planeCount) + 4) * 4, pairedPartBytes = (Math.max(4, planeCount * 2) + 4) * 4;
  const arenaPeak = Math.max(yBytes + pairedPartBytes, 3 * yBytes + partBytes), scratchBytes = 2 * xBytes + yBytes + arenaPeak + 8 * CHUNK + 4096;
  if (scratchBytes > LIMIT) throw new Error("serial MLP temporary scratch exceeds 64 MiB");
  if (Math.max(N * K, xBytes, yBytes, pairedPartBytes) > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize)) throw new Error("serial MLP case exceeds device buffer limits");
  const owned = [], deadline = performance.now() + 30000;
  let query, read, mapped = false;
  const make = (size, usage, data, target = owned) => {
    const buffer = device.createBuffer({ size: Math.max(16, size), usage }); target.push(buffer);
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const bind = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  try {
    const upload = new ArrayBuffer(CHUNK), bytes = new Int8Array(upload), scales = new Float32Array(upload);
    const weightsFor = async (offset) => {
      const w = make(N * K, U.STORAGE | U.COPY_DST), s = make(N * K / 8, U.STORAGE | U.COPY_DST);
      for (let at = 0; at < N * K; at += CHUNK) {
        const length = Math.min(CHUNK, N * K - at);
        for (let i = 0; i < length; i++) { const index = at + i, col = Math.floor(index / K); bytes[i] = fixture === "cancellation" ? offset ? -3 : 1 : fixture === "saturation" && col < 2 ? offset ? 17 : col ? -128 : 127 : (index * 31 + (index >>> 5) * 17 + offset) % 256 - 128; }
        device.queue.writeBuffer(w, at, upload, 0, length); await drain(device, deadline);
      }
      const blocks = N * K / 32;
      for (let at = 0; at < blocks; at += scales.length) {
        const length = Math.min(scales.length, blocks - at);
        for (let i = 0; i < length; i++) { const index = at + i, col = Math.floor(index / (K / 32)); scales[i] = fixture === "cancellation" ? offset ? 0.0021 : 0.0037 : fixture === "saturation" && col < 2 ? 0.05 : index % 257 === 0 ? -0 : (index % 2 ? -1 : 1) * (0.0002 + (index + offset) % 11 * 0.000037); }
        device.queue.writeBuffer(s, at * 4, upload, 0, length * 4); await drain(device, deadline);
      }
      return { values: w, scales: s };
    };
    const gate = await weightsFor(0), up = await weightsFor(19);
    const globals = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([T, 1, 1, 0])), params = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([N, K, 1, 0]));
    const ordinaryParams = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([N, K, 0, 0])), geluParams = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([N, 0, 0, 0]));
    const input = Float32Array.from({ length: T * K }, (_, i) => {
      const row = Math.floor(i / K), col = i % K;
      if (fixture === "zeros") return row & 1 ? -0 : 0;
      if (fixture === "saturation") return 0.32 + (row % 3) * 0.04;
      if (fixture === "cancellation") return (col % 8 < 4 ? 64 : -64) + (col % 4 + 1) * 0.0625 + (row % 3) * 0.03125;
      return Math.fround(0.04 * Math.sin(col * 0.127 + row * 0.71 + 0.3)) || 0.00390625;
    });
    const x = make(xBytes, U.STORAGE | U.COPY_DST, input), zero = make(16, U.STORAGE | U.COPY_DST, new Float32Array(4).fill(NaN));
    new Float32Array(upload).fill(NaN);
    const guarded = async (size, target) => {
      const buffer = make(size, U.STORAGE | U.COPY_SRC | U.COPY_DST, null, target);
      for (let offset = 0; offset < size - 16; offset += CHUNK) { device.queue.writeBuffer(buffer, offset, upload, 0, Math.min(CHUNK, size - 16 - offset)); await drain(device, deadline); }
      device.queue.writeBuffer(buffer, size - 16, new Float32Array(4).fill(SENTINEL)); return buffer;
    };
    const reference = await guarded(yBytes, owned); read = make(CHUNK, U.COPY_DST | U.MAP_READ);
    const snapshot = async (buffer, offset, length) => {
      const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, offset, read, 0, length); device.queue.submit([encoder.finish()]);
      const within = await mapBeforeDeadline(read, length, deadline); mapped = true;
      try { if (!within) throw new Error("serial MLP readback deadline exceeded"); return new Float32Array(read.getMappedRange(0, length).slice(0)); }
      finally { read.unmap(); mapped = false; }
    };
    const compare = async (other) => {
      const result = { ok: true, samples: 0, bitMismatches: 0, nonfinite: 0, maxAbs: 0, firstDifference: null, signedZeroChecked: true };
      for (let offset = 0; offset < count * 4; offset += CHUNK) {
        const length = Math.min(CHUNK, count * 4 - offset), a = await snapshot(reference, offset, length), b = await snapshot(other, offset, length), part = exact(a, b, offset / 4);
        result.ok &&= part.ok; result.samples += part.samples; result.bitMismatches += part.bitMismatches; result.nonfinite += part.nonfinite; result.maxAbs = Math.max(result.maxAbs, part.maxAbs); result.firstDifference ||= part.firstDifference;
      }
      return result;
    };
    const arena = async (kind) => {
      const local = [];
      try {
        const y = await guarded(yBytes, local), serial = kind === "candidate", separate = kind === "unfused";
        const partials = await guarded(serial ? 32 : separate ? partBytes : pairedPartBytes, local);
        let dispatch;
        const tails = [[y, yBytes], [partials, serial ? 32 : separate ? partBytes : pairedPartBytes]];
        if (!separate) {
          const op = { N, K, ...pairedMatmulGeluBindGroup(device, pipes.paired, { globals, params, input: x, gate: gate.values, gateScales: gate.scales, up: up.values, upScales: up.scales, output: y, partials }) };
          dispatch = (pass) => {
            if (!serial) return encodePairedMatmulGelu(pass, pipes.paired, op, T, 1);
            pass.setPipeline(pipes.candidate); pass.setBindGroup(0, op.group); pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(T / 64), 1);
          };
        } else {
          const g = await guarded(yBytes, local), u = await guarded(yBytes, local); tails.push([g, yBytes], [u, yBytes]);
          const p = pipes.unfused;
          const op = (weight, out) => ({ group: bind(p.layout, [globals, ordinaryParams, x, weight.values, weight.scales, zero, out, partials]), reduce: bind(p.rLayout, [globals, ordinaryParams, partials, zero, out]) });
          const ops = [op(gate, g), op(up, u)], gelu = bind(p.gelu.getBindGroupLayout(0), [globals, geluParams, g, u, y]);
          dispatch = (pass) => {
            for (const item of ops) {
              pass.setPipeline(p.mm); pass.setBindGroup(0, item.group); pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(T / 64), splits);
              if (splits > 1) { const work = Math.ceil(count / 256); pass.setPipeline(p.reduce); pass.setBindGroup(0, item.reduce); pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535)); }
            }
            const work = Math.ceil(count / 256); pass.setPipeline(p.gelu); pass.setBindGroup(0, gelu); pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
          };
        }
        const guards = async (scan = false) => {
          let tailGuard = true;
          for (const [buffer, size] of tails) tailGuard &&= (await snapshot(buffer, size - 16, 16)).every((value) => value === SENTINEL);
          const unused = serial || splits === 1;
          let partGuard = true, partValues = 0, fingerprint = 2166136261;
          if (unused || scan) {
            const total = unused ? 4 : planeCount * (separate ? 1 : 2);
            for (let offset = 0; offset < total * 4; offset += CHUNK) {
              const values = await snapshot(partials, offset, Math.min(CHUNK, total * 4 - offset)), bits = new Uint32Array(values.buffer);
              for (let i = 0; i < values.length; i++) { partGuard &&= unused ? Number.isNaN(values[i]) : Number.isFinite(values[i]); fingerprint = Math.imul(fingerprint ^ bits[i], 16777619); partValues++; }
            }
          }
          return { ok: tailGuard && partGuard, tailGuard, partGuard, candidatePartUnused: serial, partialValuesChecked: partValues, partialFingerprint: (fingerprint >>> 0).toString(16).padStart(8, "0") };
        };
        return { kind, y, dispatch, guards, destroy() { for (const buffer of local) buffer.destroy(); } };
      } catch (error) { for (const buffer of local) buffer.destroy(); throw error; }
    };
    const submit = (current, capture = false) => {
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); current.dispatch(pass); pass.end();
      if (capture) encoder.copyBufferToBuffer(current.y, 0, reference, 0, yBytes);
      device.queue.submit([encoder.finish()]);
    };
    const controls = pipes.f16 ? ["paired", "candidate"] : ["paired", "candidate", "unfused"], checks = {}, repeatedChecks = [];
    for (const kind of controls) {
      const current = await arena(kind);
      try {
        submit(current, kind === "paired");
        const output = await compare(current.y), guards = await current.guards(true);
        let fixtureCheck = true;
        if (fixture === "saturation") {
          const values = await snapshot(current.y, 0, count * 4);
          for (let row = 0; row < T; row++) fixtureCheck &&= values[row * N] > 10 && values[row * N + 1] === 0;
        }
        checks[kind] = { ok: output.ok && guards.ok && fixtureCheck, output, guards, fixtureCheck };
      } finally { current.destroy(); }
      if (!checks[kind].ok) return { ...c, f16: pipes.f16, splits, ok: false, checks, shaderHashes: pipes.hashes };
    }
    const samples = Object.fromEntries(controls.map((kind) => [kind, []]));
    if (fixture === "normal" && N >= 256) {
      query = device.createQuerySet({ type: "timestamp", count: 2 });
      const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC);
      for (let pair = 0; pair < 3; pair++) for (const kind of pair % 2 ? [...controls].reverse() : controls) {
        const current = await arena(kind);
        try {
          submit(current); await drain(device, deadline);
          const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass({ timestampWrites: { querySet: query, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
          for (let rep = 0; rep < 4; rep++) current.dispatch(pass); pass.end();
          encoder.resolveQuerySet(query, 0, 2, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, read, 0, 16); device.queue.submit([encoder.finish()]);
          const within = await mapBeforeDeadline(read, 16, deadline); mapped = true;
          if (!within) throw new Error("serial MLP timestamp deadline exceeded");
          const stamps = new BigUint64Array(read.getMappedRange(0, 16).slice(0)); read.unmap(); mapped = false;
          samples[kind].push(Number(stamps[1] - stamps[0]) / 1e6 / 4);
          const output = await compare(current.y), guards = await current.guards(); repeatedChecks.push({ pair, kind, ok: output.ok && guards.ok, output, guards });
          if (!output.ok || !guards.ok) throw new Error("serial MLP repeated output or tail parity failed");
        } finally { current.destroy(); }
      }
    }
    if (samples.candidate.length && Object.values(samples).some((values) => values.length !== 3 || values.some((value) => !Number.isFinite(value) || value <= 0))) throw new Error("serial MLP paired timestamps are invalid");
    const choices = samples.candidate.length ? Object.fromEntries(controls.filter((kind) => kind !== "candidate").map((kind) => [kind, selectMatmulKernel({ generic: samples[kind], wide: samples.candidate })])) : null;
    const nb = K / 32, per = Math.ceil(nb / splits), partitions = Array.from({ length: splits }, (_, split) => [split * per, Math.min(nb, (split + 1) * per)]);
    return { ...c, f16: pipes.f16, rows: 4, physicalBM: pipes.f16 ? 64 : 56, splitT: 1, splits, partitions, ok: true, checks, repeatedChecks,
      timing: choices ? { samples, choices, pairs: 3, repetitions: 4, allocationsAndUploadsTimed: false, everyArenaWarmed: true } : null,
      dispatches: { paired: splits > 1 ? 2 : 1, candidate: 1, ...(pipes.unfused ? { unfused: splits > 1 ? 5 : 3 } : {}) },
      matmulWorkgroups: { paired: Math.ceil(N / 64) * Math.ceil(T / 64) * splits, candidate: Math.ceil(N / 64) * Math.ceil(T / 64) },
      scratchBytes, scratchLimitBytes: LIMIT, immutableGpuWeightBytes: 2 * N * K * 9 / 8, boundedWeightUploadBytes: 2 * CHUNK, stagedReadbackBytes: CHUNK,
      shaderHashes: pipes.hashes };
  } finally { if (mapped) read?.unmap(); query?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const params = new URLSearchParams(location.search), started = performance.now(), f16 = params.get("f16") !== "0", T = Number(params.get("T") || 97);
  if (T !== 97) throw new Error("serial split probe supports T97 only");
  const shapes = [{ shape: "gate6144", N: 6144, K: 1536 }, { shape: "gate12288", N: 12288, K: 1536 }, { shape: "ragged", N: 65, K: 1056 },
    { shape: "zeros", N: 65, K: 1056, fixture: "zeros" }, { shape: "zeros-unsplit", N: 17, K: 32, fixture: "zeros" },
    { shape: "saturation", N: 17, K: 1056, fixture: "saturation" }, { shape: "cancellation", N: 65, K: 1056, fixture: "cancellation" }].filter((c) => !params.has("shape") || c.shape === params.get("shape"));
  if (!shapes.length) throw new Error("unknown serial split shape");
  const runtime = await requestDevice({ features: f16 ? ["shader-f16", "timestamp-query"] : ["timestamp-query"] });
  try {
    metadata = { precision: f16 ? "f16" : "f32", adapter: { name: runtime.name, isFallbackAdapter: false, features: [...runtime.device.features] } };
    if (!runtime.device.features.has("timestamp-query") || f16 && !runtime.device.features.has("shader-f16")) throw new Error("requested shader-f16 or timestamp-query feature is unavailable");
    const source = await kernelSource(); metadata.wasmUrl = source.wasmUrl;
    const pipes = await withGpuErrors(runtime, () => compile(runtime.device, source, f16), "compiling serial split MLP guard"); metadata.shaderHashes = pipes.hashes;
    for (const shape of shapes) {
      log({ status: "checking", T, ...shape });
      const result = await withGpuErrors(runtime, () => check(runtime, pipes, { T, ...shape }), "serial split MLP exact guard"); cases.push(result); log(result);
      if (!result.ok) throw new Error("serial split MLP bit-exact parity failed");
    }
    const timed = cases.filter((c) => c.timing), medians = timed.map((c) => c.timing.choices.paired.wideMs);
    if (medians.some((value) => !Number.isFinite(value) || value <= 0)) throw new Error("serial split MLP timestamps are invalid");
    window.gpuBench = { status: "done", done: true, backend: "webgpu", ...metadata,
      metricMs: medians.length ? Math.exp(medians.reduce((sum, value) => sum + Math.log(value), 0) / medians.length) : performance.now() - started,
      method: "GPU timestamps, serial original K partitions versus paired split/reduce; F32 also checks/times current generic unfused MLP", elapsedMs: performance.now() - started,
      correctness: { ok: true, bitExact: true, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

window.gpuBench = { status: "running", backend: "webgpu" };
main().catch((error) => { window.gpuBench = { status: "error", done: true, backend: "webgpu", ...metadata, correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) }; log(error.stack || String(error)); });
