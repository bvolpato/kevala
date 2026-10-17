// Exact address-only probe. No model weights, runtime changes, or approximate arithmetic.
import { pipeline, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const search = new URLSearchParams(location.search);
const state = { status: "running", backend: "webgpu", kernel: "gemma4_attention_split_probe" };
window.gpuBench = state;
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const U = GPUBufferUsage;
const BUDGET = 32 * 1024 * 1024;
const SENTINELS = new Float32Array([12345.25, -1731.5, 913.75, -0]);
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((value) => value.toString(16).padStart(2, "0")).join("");

function branched(lengths, prefix, dim, window, { tail = false, selected = lengths.map((_, i) => i), batch = true, causal = true, label = tail ? "tail" : "suffix", timing = false, pattern = "signed" } = {}) {
  const positions = [], offsets = [];
  let offset = 0;
  for (const [branch, length] of lengths.entries()) {
    if (!tail || selected.includes(branch)) for (const row of tail ? [length - 1] : Array.from({ length }, (_, i) => i)) { positions.push(prefix + row); offsets.push(batch ? offset : 0); }
    offset += length;
  }
  return { label, lengths, prefix, suffixRows: offset, T: positions.length, S: prefix + offset, heads: 8, kvHeads: 1, dim, window, causal, batch, timing, pattern,
    positions: Uint32Array.from(positions), offsets: Uint32Array.from(offsets) };
}

function cases() {
  const out = [];
  for (const dim of [256, 512]) {
    if (dim === 256) out.push(branched([42, 55], 89, dim, 512, { timing: true, label: "actual-local-suffix" }));
    for (const prefix of [66, 89, 117]) for (const window of [0, 65]) {
      out.push(branched([42, 55], prefix, dim, window, { timing: prefix === 89 && dim === 512 && window === 0 }));
      out.push(branched([42, 55], prefix, dim, window, { tail: true }));
    }
    for (const pattern of ["zero", "cancellation", "peaked"]) out.push(branched([42, 55], 89, dim, 65, { pattern }));
    for (const window of [0, 65]) out.push(branched([97], 89, dim, window, { batch: false, causal: false, label: "unbatched-noncausal" }));
    for (const tail of [false, true]) out.push(branched([1, 17], 117, dim, 129, { tail, label: tail ? "ragged-tail" : "ragged-suffix" }));
    out.push(branched([42, 55], 0, dim, 65, { label: "suffix-only" }));
    out.push({ ...branched([89], 0, dim, 0, { batch: false, label: "prefix-only" }), prefix: 89, suffixRows: 0 });
    out.push(branched([42, 55], 66, dim, 65, { tail: true, selected: [1], label: "selected-second-tail" }));
    out.push({ ...branched([3, 5, 9], 66, dim, 8, { label: "three-ragged-branches" }), heads: 4, kvHeads: 2 });
  }
  if (search.get("dim128") === "1") out.push({ ...branched([3, 5, 9], 66, 128, 8, { label: "optional-dim128" }), heads: 4, kvHeads: 2 });
  return search.get("cases") === "timing" ? out.filter((c) => c.timing) : out;
}

function splitSource(original) {
  const key = "@group(0) @binding(3) var<storage, read> K: array<f32>;";
  const value = "@group(0) @binding(4) var<storage, read> V: array<f32>;";
  const keyNew = key.replace(" K:", " KP:"), valueNew = value.replace(" V:", " VP:");
  const added = "@group(0) @binding(7) var<storage, read> KS: array<f32>;\n@group(0) @binding(8) var<storage, read> VS: array<f32>;\n";
  const helpers = `fn split_key(at: u32) -> f32 {
  let boundary = g._b * p.kv_heads * p.head_dim;
  if (at < boundary) { return KP[at]; }
  return KS[at - boundary];
}
fn split_value(at: u32) -> f32 {
  let boundary = g._b * p.kv_heads * p.head_dim;
  if (at < boundary) { return VP[at]; }
  return VS[at - boundary];
}
`;
  for (const declaration of [key, value, "var<workgroup> qtile:"]) if (original.split(declaration).length !== 2) throw new Error(`attention source drift: expected exactly one ${declaration}`);
  let keyLoads = 0, valueLoads = 0;
  let code = original.replace(key, keyNew).replace(value, valueNew)
    .replace(/\bK\[([^\]\n]+)\]/g, (_match, at) => { keyLoads++; return `split_key(${at})`; })
    .replace(/\bV\[([^\]\n]+)\]/g, (_match, at) => { valueLoads++; return `split_value(${at})`; });
  if (keyLoads !== 5 || valueLoads !== 2) throw new Error(`attention load layout changed: K=${keyLoads}, V=${valueLoads}`);
  code = code.replace("var<workgroup> qtile:", added + helpers + "var<workgroup> qtile:");
  const restored = code.replace(added + helpers, "").replace(keyNew, key).replace(valueNew, value)
    .replace(/\bsplit_key\(([^()\n]*)\)/g, (_match, at) => `K[${at}]`).replace(/\bsplit_value\(([^()\n]*)\)/g, (_match, at) => `V[${at}]`);
  if (restored !== original) throw new Error("split attention changed source outside K/V bindings and load addresses");
  return { code, keyLoads, valueLoads, exactArithmeticSource: true, storageBindings: 7 };
}

function bracedRange(source, marker) {
  const start = source.indexOf(marker);
  if (start < 0 || source.indexOf(marker, start + marker.length) >= 0) throw new Error(`attention source drift: expected one ${marker}`);
  const open = source.indexOf("{", start), stack = [];
  for (let at = open; at < source.length; at++) {
    if (source[at] === "{") stack.push(at);
    else if (source[at] === "}" && stack.pop() === open) return { start, open, close: at };
  }
  throw new Error("attention source block is unterminated");
}

function hoistedSplitSource(original) {
  const key = "@group(0) @binding(3) var<storage, read> K: array<f32>;", value = "@group(0) @binding(4) var<storage, read> V: array<f32>;";
  const keyNew = key.replace(" K:", " KP:"), valueNew = value.replace(" V:", " VP:");
  const added = "@group(0) @binding(7) var<storage, read> KS: array<f32>;\n@group(0) @binding(8) var<storage, read> VS: array<f32>;\n";
  for (const declaration of [key, value, "var<workgroup> qtile:"]) if (original.split(declaration).length !== 2) throw new Error(`attention source drift: expected exactly one ${declaration}`);
  const keyBase = "      let kbase = physical_key * p.kv_heads * H + kv_head * H;";
  const keyRange = bracedRange(original, "      if (j < end) {");
  const keyStart = original.lastIndexOf(keyBase, keyRange.start);
  if (keyStart < 0 || original.slice(keyStart + keyBase.length, keyRange.start) !== "\n      var sum = 0.0;\n") throw new Error("attention K accumulation layout changed");
  const oldKey = original.slice(keyStart, keyRange.close + 1), keyBody = original.slice(keyRange.open + 1, keyRange.close).trimEnd();
  const keyLoads = (keyBody.match(/\bK\[/g) || []).length;
  const indent = (body) => body.split("\n").map((line) => line ? `  ${line}` : line).join("\n");
  const prefixKey = indent(keyBody.replace(/\bK\[/g, "KP[")), suffixKey = indent(keyBody.replace(/\bK\[/g, "KS["));
  const newKey = `      var sum = 0.0;
      if (j < end) {
        if (j < g._b) {
          let kbase = j * p.kv_heads * H + kv_head * H;${prefixKey}
        } else {
          let kbase = (physical_key - g._b) * p.kv_heads * H + kv_head * H;${suffixKey}
        }
      }`;
  const valueRange = bracedRange(original, "    for (var j = 0u; j < min(128u, end - j0); j++) {");
  const oldValue = original.slice(valueRange.start, valueRange.close + 1), valueBody = original.slice(valueRange.open + 1, valueRange.close).trimEnd();
  const valueBase = "let vbase = physical_key * p.kv_heads * H + kv_head * H;";
  if (valueBody.split(valueBase).length !== 2 || /workgroupBarrier|subgroup/.test(keyBody + valueBody)) throw new Error("attention load blocks contain an unexpected layout or collective operation");
  const valueLoads = (valueBody.match(/\bV\[/g) || []).length;
  const prefixValue = valueBody.replace(valueBase, "let vbase = logical_key * p.kv_heads * H + kv_head * H;").replace(/\bV\[/g, "VP[");
  const suffixValue = valueBody.replace(valueBase, "let vbase = (physical_key - g._b) * p.kv_heads * H + kv_head * H;").replace(/\bV\[/g, "VS[");
  const newValue = `    let tile_keys = min(128u, end - j0);
    var prefix_keys = 0u;
    if (j0 < g._b) { prefix_keys = min(tile_keys, g._b - j0); }
    for (var j = 0u; j < prefix_keys; j++) {${prefixValue}
    }
    for (var j = prefix_keys; j < tile_keys; j++) {${suffixValue}
    }`;
  if (keyLoads !== 5 || valueLoads !== 2) throw new Error(`attention load layout changed: K=${keyLoads}, V=${valueLoads}`);
  const code = original.replace(key, keyNew).replace(value, valueNew).replace("var<workgroup> qtile:", added + "var<workgroup> qtile:").replace(oldKey, newKey).replace(oldValue, newValue);
  const restored = code.replace(newKey, oldKey).replace(newValue, oldValue).replace(added, "").replace(keyNew, key).replace(valueNew, value);
  if (restored !== original) throw new Error("hoisted split attention changed source outside the K/V load blocks and bindings");
  return { code, keyLoads, valueLoads, exactArithmeticSource: true, storageBindings: 7, keyBufferChoice: "once per key before dot loop", valueBufferChoice: "ordered prefix loop then suffix loop" };
}

function sizes(c) {
  const qCount = c.T * c.heads * c.dim, kvWidth = c.kvHeads * c.dim, kvCount = c.S * kvWidth;
  const qBytes = (qCount + SENTINELS.length) * 4, kvBytes = (kvCount + SENTINELS.length) * 4, posBytes = (c.T * 2 + SENTINELS.length) * 4;
  // Bound GPU buffers, host joined/split fixtures, both readback copies, and upload reserves.
  const estimateBytes = 14 * qBytes + 13 * kvBytes + 4 * posBytes + 8192;
  if (estimateBytes > BUDGET) throw new Error(`attention probe estimate ${estimateBytes} exceeds ${BUDGET}`);
  return { qCount, kvWidth, kvCount, qBytes, kvBytes, posBytes, estimateBytes };
}

function fixture(c, size, seed) {
  let state = seed >>> 0;
  const random = () => { state = Math.imul(state ^ (state >>> 16), 0x45d9f3b); state = Math.imul(state ^ (state >>> 16), 0x45d9f3b); state ^= state >>> 16; return (state >>> 0) / 0x100000000; };
  const q = new Float32Array(size.qBytes / 4).fill(NaN), k = new Float32Array(size.kvBytes / 4).fill(NaN), v = k.slice();
  for (let i = 0; i < size.qCount; i++) q[i] = c.pattern === "zero" ? i % 2 ? -0 : 0 : c.pattern === "peaked" ? 4 + i % 4 / 8 : (random() - 0.5) * 0.5;
  for (let i = 0; i < size.kvCount; i++) {
    k[i] = c.pattern === "zero" ? i % 3 ? 0 : -0 : c.pattern === "peaked" ? 4 + (Math.floor(i / size.kvWidth) % 7 - 3) / 128 : c.pattern === "cancellation" ? (i % 2 ? -1 : 1) * (0.1731 + (i % 13) * 0.00737) : (random() - 0.5) * 0.5;
    v[i] = c.pattern === "zero" ? i % 5 ? -0 : 0 : c.pattern === "cancellation" ? (Math.floor(i / size.kvWidth) % 2 ? -1 : 1) * (0.317 + (i % 19) * 0.00437) : (random() - 0.5) * 0.5;
  }
  const prefixCount = c.prefix * size.kvWidth;
  const prefixK = k.slice(0, prefixCount + SENTINELS.length), prefixV = v.slice(0, prefixCount + SENTINELS.length);
  prefixK.fill(NaN, prefixCount); prefixV.fill(NaN, prefixCount);
  const suffixK = k.slice(prefixCount), suffixV = v.slice(prefixCount);
  const positions = new Uint32Array(size.posBytes / 4).fill(0xffffffff); positions.set(c.positions); positions.set(c.offsets, c.T);
  return { q, k, v, prefixK, prefixV, suffixK, suffixV, positions };
}

function checkAddresses(c, size, input) {
  const selected = [...new Set([0, Math.floor(c.T / 2), c.T - 1])];
  const planes = [[input.k, input.prefixK, input.suffixK], [input.v, input.prefixV, input.suffixV]];
  let scalarAddresses = 0;
  for (const t of selected) for (const head of [0, c.heads - 1]) {
    const position = c.positions[t], start = c.window ? position + 1 - Math.min(position + 1, c.window) : 0, end = c.causal ? position + 1 : c.S;
    const kvHead = Math.floor(head / (c.heads / c.kvHeads));
    for (let j = start; j < end; j++) {
      const physical = j + (c.batch && j >= c.prefix ? c.offsets[t] : 0);
      if (physical >= c.S) throw new Error(`fixture physical key ${physical} exceeds ${c.S}`);
      const base = (physical * c.kvHeads + kvHead) * c.dim;
      for (let d = 0; d < c.dim; d++) {
        const at = base + d, prefix = at < c.prefix * size.kvWidth, local = prefix ? at : at - c.prefix * size.kvWidth;
        for (const [joined, a, b] of planes) {
          const expected = joined[at], actual = (prefix ? a : b)[local];
          if (!Number.isFinite(expected) || !Number.isFinite(actual) || !Object.is(expected, actual)) throw new Error(`split address mismatch at ${at}`);
          scalarAddresses++;
        }
      }
    }
  }
  return { ok: true, bitExact: true, scalarAddresses, selectedQueryRows: selected };
}

function checkHoistedAddresses(c, size, input) {
  const selected = [...new Set([0, Math.floor(c.T / 2), c.T - 1])], planes = [[input.k, input.prefixK, input.suffixK], [input.v, input.prefixV, input.suffixV]];
  let scalarAddresses = 0, valueKeysInOriginalOrder = 0;
  for (const t of selected) for (const head of [0, c.heads - 1]) {
    const position = c.positions[t], start = c.window ? position + 1 - Math.min(position + 1, c.window) : 0, end = c.causal ? position + 1 : c.S;
    const kvHead = Math.floor(head / (c.heads / c.kvHeads));
    let expectedKey = start;
    for (let j0 = start; j0 < end; j0 += 128) {
      const tileKeys = Math.min(128, end - j0), prefixKeys = j0 < c.prefix ? Math.min(tileKeys, c.prefix - j0) : 0;
      for (const [first, last, prefix] of [[0, prefixKeys, true], [prefixKeys, tileKeys, false]]) for (let j = first; j < last; j++) {
        const logical = j0 + j, physical = logical + (c.batch && logical >= c.prefix ? c.offsets[t] : 0);
        if (logical !== expectedKey++ || prefix !== (logical < c.prefix)) throw new Error("hoisted V loop changed key order");
        const localBase = ((prefix ? logical : physical - c.prefix) * c.kvHeads + kvHead) * c.dim;
        const joinedBase = (physical * c.kvHeads + kvHead) * c.dim;
        for (let d = 0; d < c.dim; d++) for (const [joined, a, b] of planes) {
          const expected = joined[joinedBase + d], actual = (prefix ? a : b)[localBase + d];
          if (!Number.isFinite(expected) || !Number.isFinite(actual) || !Object.is(expected, actual)) throw new Error("hoisted K/V address differs from joined storage");
          scalarAddresses++;
        }
        valueKeysInOriginalOrder++;
      }
    }
    if (expectedKey !== end) throw new Error("hoisted V loop omitted keys");
  }
  return { ok: true, bitExact: true, scalarAddresses, valueKeysInOriginalOrder, selectedQueryRows: selected };
}

function compare(label, expected, actual) {
  if (expected.length !== actual.length) throw new Error(`${label} length mismatch`);
  const a = new Uint32Array(expected.buffer, expected.byteOffset, expected.length), b = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  let differences = 0, maxAbs = 0, firstDifference = null;
  for (let i = 0; i < expected.length; i++) {
    if (!Number.isFinite(expected[i]) || !Number.isFinite(actual[i])) throw new Error(`${label} nonfinite output at ${i}`);
    maxAbs = Math.max(maxAbs, Math.abs(expected[i] - actual[i]));
    if (a[i] !== b[i]) { differences++; firstDifference ||= { index: i, expected: expected[i], actual: actual[i], expectedBits: a[i], actualBits: b[i] }; }
  }
  const result = { label, bitExact: differences === 0, differences, maxAbs, firstDifference, checkedOutputs: expected.length };
  if (differences) throw new Error(`${label}: ${JSON.stringify(result)}`);
  return result;
}

function median(values) {
  const sorted = values.toSorted((a, b) => a - b), mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

async function checkCase(gpu, kernels, c, seed) {
  const size = sizes(c), input = fixture(c, size, seed), cpuAddresses = checkAddresses(c, size, input), device = gpu.device;
  const cpuHoistedAddresses = search.get("address") === "hoisted" ? checkHoistedAddresses(c, size, input) : null;
  const owned = [];
  let allocatedBytes = 0, queries, read, mapped = false;
  const make = (bytes, usage, value, label) => {
    const count = Math.max(16, bytes); allocatedBytes += count;
    if (allocatedBytes > BUDGET) throw new Error("attention GPU scratch limit exceeded");
    const buffer = device.createBuffer({ size: count, usage, label }); owned.push(buffer);
    if (value) device.queue.writeBuffer(buffer, 0, value);
    return buffer;
  };
  const storage = (value, label, extra = 0) => make(value.byteLength, U.STORAGE | U.COPY_DST | extra, value, label);
  const group = (kernel, buffers) => device.createBindGroup({ layout: kernel.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  try {
    const globals = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([c.T, 0, c.S, c.prefix]), "attention globals");
    const params = make(32, U.UNIFORM | U.COPY_DST, new Uint32Array([c.heads, c.kvHeads, c.dim, c.window, Number(c.causal), Number(c.batch), 0, 0]), "attention params");
    const Q = storage(input.q, "query with poisoned tail"), POS = storage(input.positions, "logical positions and physical branch offsets");
    const KP = storage(input.prefixK, "cached prefix K", U.COPY_SRC), VP = storage(input.prefixV, "cached prefix V", U.COPY_SRC);
    const KS = storage(input.suffixK, "suffix K", U.COPY_SRC), VS = storage(input.suffixV, "suffix V", U.COPY_SRC);
    const joinedPoison = new Float32Array(size.kvBytes / 4).fill(NaN);
    const K = storage(joinedPoison, "joined K with poisoned tail"), V = storage(joinedPoison, "joined V with poisoned tail");
    const poison = new Float32Array(size.qBytes / 4).fill(NaN); poison.set(SENTINELS, size.qCount);
    const outputs = ["joined attention output", "split attention output"].map((label) => make(size.qBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, poison, label));
    const baselineGroup = group(kernels.baseline, [globals, params, Q, K, V, outputs[0], POS]);
    const candidateGroup = group(kernels.candidate, [globals, params, Q, KP, VP, outputs[1], POS, KS, VS]);
    const prefixBytes = c.prefix * size.kvWidth * 4, suffixBytes = c.suffixRows * size.kvWidth * 4;
    const join = (encoder) => {
      if (prefixBytes) { encoder.copyBufferToBuffer(KP, 0, K, 0, prefixBytes); encoder.copyBufferToBuffer(VP, 0, V, 0, prefixBytes); }
      if (suffixBytes) { encoder.copyBufferToBuffer(KS, 0, K, prefixBytes, suffixBytes); encoder.copyBufferToBuffer(VS, 0, V, prefixBytes, suffixBytes); }
    };
    const dispatch = (pass, next) => { pass.setPipeline(next ? kernels.candidate : kernels.baseline); pass.setBindGroup(0, next ? candidateGroup : baselineGroup); pass.dispatchWorkgroups(c.T, c.heads); };
    const copyOutputs = (encoder) => outputs.forEach((output, i) => encoder.copyBufferToBuffer(output, 0, read, i * size.qBytes, size.qBytes));
    const copyInclusive = search.get("copies") === "1", samples = 3, repetitions = 4, queryCount = samples * repetitions * 2 * (copyInclusive ? 4 : 2), queryBytes = queryCount * 8;
    read = make(Math.max(size.qBytes * 2, queryBytes), U.COPY_DST | U.MAP_READ, null, "attention split guard readback");
    const check = async () => {
      await read.mapAsync(GPUMapMode.READ, 0, size.qBytes * 2); mapped = true;
      const actual = new Float32Array(read.getMappedRange(0, size.qBytes * 2).slice(0)); read.unmap(); mapped = false;
      const stride = size.qBytes / 4;
      for (let i = 0; i < 2; i++) compare(`attention output${i} tail`, SENTINELS, actual.subarray(i * stride + size.qCount, (i + 1) * stride));
      return compare("joined vs split attention", actual.subarray(0, size.qCount), actual.subarray(stride, stride + size.qCount));
    };
    const encoder = device.createCommandEncoder(); join(encoder);
    const pass = encoder.beginComputePass(); dispatch(pass, false); dispatch(pass, true); pass.end();
    copyOutputs(encoder); device.queue.submit([encoder.finish()]);
    const parity = await check();
    let timing = { scope: copyInclusive ? "join-copies-plus-kernel" : "kernel-only", available: false }, repeatedParity = null;
    if (c.timing && search.get("timing") !== "0" && device.features.has("timestamp-query")) {
      queries = device.createQuerySet({ type: "timestamp", count: queryCount });
      const resolved = make(Math.ceil(queryBytes / 256) * 256, U.QUERY_RESOLVE | U.COPY_SRC, null, "attention timestamps"), encoder = device.createCommandEncoder(), records = [];
      let index = 0;
      for (let sample = 0; sample < samples; sample++) for (let repetition = 0; repetition < repetitions; repetition++) for (const next of (sample + repetition) % 2 ? [true, false] : [false, true]) {
        const start = index;
        if (copyInclusive) {
          const marker = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } }); marker.end(); index += 2;
          if (!next) join(encoder);
        }
        const pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } }); dispatch(pass, next); pass.end();
        records.push({ sample, repetition, path: next ? "candidate" : "baseline", start, end: index + 1 }); index += 2;
      }
      if (index !== queryCount) throw new Error("attention timestamp layout mismatch");
      encoder.resolveQuerySet(queries, 0, queryCount, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, read, 0, queryBytes); device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ, 0, queryBytes); mapped = true;
      const stamps = new BigUint64Array(read.getMappedRange(0, queryBytes).slice(0)); read.unmap(); mapped = false;
      const pairs = Array.from({ length: samples }, (_, sample) => ({ sample, baseline: [], candidate: [] }));
      for (const record of records) {
        if (stamps[record.end] <= stamps[record.start]) throw new Error("attention timestamp interval is nonpositive");
        pairs[record.sample][record.path].push(Number(stamps[record.end] - stamps[record.start]) / 1e6);
      }
      for (const pair of pairs) { pair.baselineMedianMs = median(pair.baseline); pair.candidateMedianMs = median(pair.candidate); }
      const baselineMedianMs = median(pairs.map((p) => p.baselineMedianMs)), candidateMedianMs = median(pairs.map((p) => p.candidateMedianMs));
      const encoderAfter = device.createCommandEncoder(); copyOutputs(encoderAfter); device.queue.submit([encoderAfter.finish()]); repeatedParity = await check();
      timing = { scope: copyInclusive ? "four join copies plus attention; identical marker-pass overhead included on both paths" : "attention kernels only; initial join copies excluded",
        available: true, samples, repetitions, pairs, baselineMedianMs, candidateMedianMs, speedup: baselineMedianMs / candidateMedianMs,
        baselineJoinCopyCommands: Number(prefixBytes > 0) * 2 + Number(suffixBytes > 0) * 2, candidateJoinCopyCommands: 0 };
    }
    const { positions, offsets, ...summary } = c;
    return { ...summary, positionRange: [Math.min(...positions), Math.max(...positions)], branchOffsets: [...new Set(offsets)], cpuAddresses, ...(cpuHoistedAddresses ? { cpuHoistedAddresses } : {}), parity, repeatedParity, outputTailGuard: true, inputTailsPoisoned: true,
      timing, ownedGpuBufferBytes: allocatedBytes, estimatedCaseBufferBytes: size.estimateBytes, ok: true };
  } finally { if (mapped) read?.unmap(); queries?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const started = performance.now(), subgroupOption = search.get("subgroups"), allowSubgroups = subgroupOption !== "0";
  const addressMode = search.get("address") || "scalar";
  if (!["scalar", "hoisted"].includes(addressMode)) throw new Error("address must be scalar or hoisted");
  const gpu = await requestDevice({ features: ["timestamp-query", ...(allowSubgroups ? ["subgroups"] : [])] }), uncaptured = [];
  gpu.device.addEventListener("uncapturederror", (event) => uncaptured.push(event.error?.message || String(event.error)));
  try {
    const info = gpu.adapter.info || {}, subgroupId = navigator.gpu.wgslLanguageFeatures?.has("subgroup_id") === true;
    const subgroupAvailable = gpu.device.features.has("subgroups") && gpu.subgroup4 && subgroupId && info.subgroupMinSize === 32 && info.subgroupMaxSize === 32;
    if (subgroupOption === "1" && !subgroupAvailable) throw new Error("split subgroup attention requires compiled subgroups, WGSL subgroup_id, and fixed32 lanes");
    const names = subgroupOption === "1" ? ["subgroup"] : ["portable", ...(allowSubgroups && subgroupAvailable ? ["subgroup"] : [])];
    const source = await kernelSource(), results = [], compilation = [];
    for (const variant of names) {
      const registered = variant === "subgroup" ? "gemma4_attention_subgroup" : "gemma4_attention", baseline = source(registered), candidate = addressMode === "hoisted" ? hoistedSplitSource(baseline) : splitSource(baseline);
      const kernels = await withGpuErrors(gpu, async () => ({ baseline: await pipeline(gpu.device, baseline, `${registered}.joined`), candidate: await pipeline(gpu.device, candidate.code, `${registered}.split`) }), "compiling split-input decoder attention");
      compilation.push({ variant, registered, addressMode, keyLoads: candidate.keyLoads, valueLoads: candidate.valueLoads, exactArithmeticSource: candidate.exactArithmeticSource, storageBindings: candidate.storageBindings,
        baselineSha256: await hash(baseline), candidateSha256: await hash(candidate.code),
        ...(candidate.keyBufferChoice ? { keyBufferChoice: candidate.keyBufferChoice, valueBufferChoice: candidate.valueBufferChoice } : {}) });
      for (const [index, c] of cases().entries()) {
        state.currentCase = { variant, addressMode, label: c.label, T: c.T, dim: c.dim, prefix: c.prefix, window: c.window, pattern: c.pattern };
        const result = await withGpuErrors(gpu, () => checkCase(gpu, kernels, c, (0x4b455641 + Math.imul(index, 0x9e3779b9)) >>> 0), "checking split-input decoder attention");
        results.push({ variant, ...result }); log({ variant, ...result });
      }
    }
    if (uncaptured.length) throw new Error(`uncaptured WebGPU error: ${uncaptured.join("; ")}`);
    const timed = results.filter((result) => result.timing.available);
    const elapsedMs = performance.now() - started;
    window.gpuBench = { status: "done", done: true, backend: state.backend, kernel: state.kernel, addressMode, elapsedMs,
      metricMs: timed.length ? Math.exp(timed.reduce((sum, result) => sum + Math.log(result.timing.candidateMedianMs), 0) / timed.length) : elapsedMs,
      method: timed.length ? `geometric mean candidate GPU milliseconds; ${timed[0].timing.scope}` : "guard elapsed milliseconds; GPU timestamps unavailable or disabled",
      adapter: { name: gpu.name, features: [...gpu.device.features], subgroupMinSize: info.subgroupMinSize, subgroupMaxSize: info.subgroupMaxSize, subgroupId },
      compile: { ok: true, source: "current registered attention WGSL with verified reversible K/V binding/load substitution", variants: compilation },
      subgroups: { available: Boolean(subgroupAvailable), requested: subgroupOption || "auto", skipped: allowSubgroups && !subgroupAvailable },
      resourceBounds: { maxCaseBufferEstimateBytes: Math.max(...results.map((r) => r.estimatedCaseBufferBytes)), budgetBytes: BUDGET, sequentialCases: true },
      correctness: { ok: true, bitExact: true, tolerance: { absolute: 0, relative: 0 }, cases: results } };
  } finally { gpu.device.destroy(); }
}

main().catch((error) => { window.gpuBench = { ...state, status: "error", done: true, error: String(error?.message || error), code: error?.code }; log(error.stack || String(error)); });
