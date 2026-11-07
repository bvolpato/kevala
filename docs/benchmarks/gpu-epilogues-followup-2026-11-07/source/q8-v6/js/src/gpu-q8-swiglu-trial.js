import { mmSplits, pipeline } from "./gpu.js";

export const KEV_Q8_SWIGLU_GUARD_BYTES = 256;
export const KEV_Q8_SWIGLU_GUARD_WORDS = KEV_Q8_SWIGLU_GUARD_BYTES / 4;
export const KEV_Q8_SWIGLU_CANARY = 0x7fc0cafe;
export const KEV_Q8_SWIGLU_LAYER_WORDS = 6;

const SPEC = { weights: "q8", f16: false, rows: 4, groups: 1, splitTarget: 256 };

function functionSource(source, name) {
  const marker = `fn ${name}(`;
  const start = source.indexOf(marker);
  if (start < 0 || source.indexOf(marker, start + marker.length) >= 0) throw new Error(`Q8 SwiGLU trial expected one ${marker.trim()} function`);
  const open = source.indexOf("{", start + marker.length);
  if (open < 0) throw new Error(`Q8 SwiGLU trial found no body for ${name}`);
  let depth = 0;
  for (let at = open; at < source.length; at++) {
    if (source[at] === "{") depth++;
    else if (source[at] === "}" && --depth === 0) return source.slice(start, at + 1);
  }
  throw new Error(`Q8 SwiGLU trial found an unterminated ${name} function`);
}

export function kevQ8SwiGluWgsl(matmulSource, silumulSource) {
  const gelu = functionSource(matmulSource, "gelu");
  const sigmoid = functionSource(silumulSource, "sigmoid");
  const silu = functionSource(silumulSource, "silu");
  const store = "Y[o] = gelu(gate) * up;";
  if (matmulSource.split(store).length !== 2) throw new Error("Q8 SwiGLU trial expected one paired GELU store");
  const source = matmulSource.replace(gelu, `${sigmoid}\n\n${silu}`).replace(store, "Y[o] = silu(gate) * up;");
  if (/\bgelu\s*\(/.test(source) || source.split("Y[o] = silu(gate) * up;").length !== 2) throw new Error("Q8 SwiGLU trial SiLU source transform was incomplete");
  return source;
}

export function kevQ8SwiGluCompatible({ model, config, entry, device }) {
  const [rows, K] = entry?.info?.shape || [];
  const I = Number(config?.intermediate_size);
  const hidden = Number(config?.hidden_size);
  const offset = I * K;
  const scaleOffset = I * (K / 32) * 4;
  const alignment = device.limits.minStorageBufferOffsetAlignment || 256;
  const sizeOk = Number.isSafeInteger(offset) && offset > 0 && offset + offset <= entry.buf?.size &&
    Number.isSafeInteger(scaleOffset) && scaleOffset > 0 && scaleOffset + scaleOffset <= entry.sbuf?.size;
  const workgroupStorageBytes = Number(device.limits.maxComputeWorkgroupStorageSize);
  return {
    eligible: model?.name === "kev-9b" && model?.precision === "q8" && hidden === 4096 && I === 12288 &&
      Number(config?.layer_types?.length) === 32 && rows === 2 * I && K === hidden && entry?.info?.dtype === "q8" &&
      Number(entry?.info?.block) === 32 && entry.buf && entry.sbuf && sizeOk && offset % alignment === 0 && scaleOffset % alignment === 0 &&
      workgroupStorageBytes >= 24576,
    model: model?.name || null,
    precision: model?.precision || null,
    hidden,
    intermediate: I,
    layers: Number(config?.layer_types?.length) || null,
    rows,
    K,
    weightOffsetBytes: offset,
    scaleOffsetBytes: scaleOffset,
    alignment,
    workgroupStorageBytes,
    requiredWorkgroupStorageBytes: 24576,
  };
}

export function kevQ8SwiGluEligible({ enabled, profiler, mode, stage, tokens, expectedTokens, plannedLayers, matvecSelected, op, cfg, baseline, tuning }) {
  const I = cfg?.intermediate;
  const K = cfg?.hidden;
  return enabled === true && profiler == null && ["verify", "candidate"].includes(mode) && stage === "prefill" &&
    Number.isSafeInteger(tokens) && tokens > 0 && tokens === expectedTokens &&
    (!Array.isArray(plannedLayers) || plannedLayers.includes(op?.q8SwiGluLayer)) && op?.label?.endsWith("gate_up") &&
    op?.N === 2 * I && op?.K === K && op?.dtype === "q8" && K === 4096 && I === 12288 &&
    Number(op?.q8SwiGluEntry?.info?.block) === 32 && op?.q8SwiGluEntry?.buf && op?.q8SwiGluEntry?.sbuf &&
    baseline === "generic" && matvecSelected !== true && tuning?.f16 === false && tuning?.groups === 1 && tuning?.splitTarget === 256 &&
    mmSplits(tokens, op.N, op.K, 256, 64, 64) === 1 && mmSplits(tokens, I, K, 256, 64, 64) === 1;
}

export async function createKevQ8SwiGluTrial(device, wgsl) {
  const code = kevQ8SwiGluWgsl(wgsl("matmul_gelu", SPEC), wgsl("kev_silumul"));
  const gemmLayout = device.createBindGroupLayout({
    label: "Q8 Kev SwiGLU trial",
    entries: ["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage", "storage"].map((type, binding) => ({
      binding, visibility: GPUShaderStage.COMPUTE, buffer: { type },
    })),
  });
  const compareLayout = device.createBindGroupLayout({
    label: "Q8 Kev SwiGLU full activation comparison",
    entries: ["uniform", "read-only-storage", "read-only-storage", "storage"].map((type, binding) => ({
      binding, visibility: GPUShaderStage.COMPUTE, buffer: { type },
    })),
  });
  const fillLayout = device.createBindGroupLayout({
    label: "Q8 Kev SwiGLU validation output initialization",
    entries: ["uniform", "storage"].map((type, binding) => ({
      binding, visibility: GPUShaderStage.COMPUTE, buffer: { type },
    })),
  });
  const compareCode = `
struct P { T: u32, I: u32, layer: u32, guard_words: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> baseline: array<u32>;
@group(0) @binding(2) var<storage, read> candidate: array<u32>;
@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * 65535u * 256u;
  let count = p.T * p.I;
  let base = p.layer * ${KEV_Q8_SWIGLU_LAYER_WORDS}u;
  if (i < count) {
    let a = baseline[i];
    let b = candidate[p.guard_words + i];
    atomicAdd(&status[base + 4u], 1u);
    if (a != b) { atomicAdd(&status[base], 1u); atomicMin(&status[base + 3u], i); }
    if ((a & 0x7f800000u) == 0x7f800000u || (b & 0x7f800000u) == 0x7f800000u) { atomicAdd(&status[base + 1u], 1u); }
  }
  if (i < p.guard_words) {
    let suffix = p.guard_words + count + i;
    atomicAdd(&status[base + 5u], 2u);
    if (candidate[i] != ${KEV_Q8_SWIGLU_CANARY}u || candidate[suffix] != ${KEV_Q8_SWIGLU_CANARY}u) { atomicAdd(&status[base + 2u], 1u); }
  }
}`;
  const fillCode = `
struct P { T: u32, I: u32, layer: u32, guard_words: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> candidate: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * 65535u * 256u;
  if (i < p.T * p.I) { candidate[p.guard_words + i] = ${KEV_Q8_SWIGLU_CANARY}u; }
}`;
  const [matmul, compare, fill] = await Promise.all([
    pipeline(device, code, "kev_q8_swiglu_trial", device.createPipelineLayout({ bindGroupLayouts: [gemmLayout] })),
    pipeline(device, compareCode, "kev_q8_swiglu_compare", device.createPipelineLayout({ bindGroupLayouts: [compareLayout] })),
    pipeline(device, fillCode, "kev_q8_swiglu_fill", device.createPipelineLayout({ bindGroupLayouts: [fillLayout] })),
  ]);
  return { matmul, compare, fill, gemmLayout, compareLayout, fillLayout, source: code, spec: { ...SPEC } };
}

export function kevQ8SwiGluLayerWeights(device, layout, globals, entry, params, input, output, partials, I, K) {
  const weightBytes = I * K;
  const scaleBytes = I * (K / 32) * 4;
  const weightUp = { buffer: entry.buf, offset: weightBytes, size: weightBytes };
  const scaleUp = { buffer: entry.sbuf, offset: scaleBytes, size: scaleBytes };
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: { buffer: globals } },
      { binding: 1, resource: { buffer: params } },
      { binding: 2, resource: { buffer: input } },
      { binding: 3, resource: { buffer: entry.buf, offset: 0, size: weightBytes } },
      { binding: 4, resource: { buffer: entry.sbuf, offset: 0, size: scaleBytes } },
      { binding: 5, resource: weightUp },
      { binding: 6, resource: scaleUp },
      { binding: 7, resource: { buffer: output.buffer, offset: output.offset || 0, size: output.size } },
      { binding: 8, resource: { buffer: partials } },
    ],
  });
}

export function kevQ8SwiGluCompareGroup(device, layout, params, baseline, candidate, status) {
  return device.createBindGroup({
    layout,
    entries: [params, baseline, candidate, status].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
}

export function kevQ8SwiGluFillGroup(device, layout, params, candidate) {
  return device.createBindGroup({
    layout,
    entries: [params, candidate].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
}

export function kevQ8SwiGluStatusInit(layers) {
  const status = new Uint32Array(layers * KEV_Q8_SWIGLU_LAYER_WORDS);
  for (let layer = 0; layer < layers; layer++) status[layer * KEV_Q8_SWIGLU_LAYER_WORDS + 3] = 0xffffffff;
  return status;
}
