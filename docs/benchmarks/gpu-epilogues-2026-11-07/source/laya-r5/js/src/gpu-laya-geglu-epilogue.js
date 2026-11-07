// Private source-only experiment. No calibration or automatic selection is performed here.
import { mmSplits, pipeline, rowsPerThread } from "./gpu.js";
import { selectedMatmulKernel } from "./gpu-tuning.js";

export const LAYA_GEGLU_SHAPE = Object.freeze({ tokens: 512, I: 2624, K: 1024, layers: 28, role: "long:first" });
const WORDS = "the customer wrote in about a recurring charge they did not recognise and wants a refund today ";
const PINNED_LONG_JSON = JSON.stringify([{ state: `Run 0. ${WORDS.repeat(30)}`, questions: { q: { type: "noul", instructions: "Is the customer asking for money back?" } } }]);
const STATUS_WORDS = 6;
const STATUS_BYTES = STATUS_WORDS * 4;
const SENTINEL = 0x4640e500; // finite 12345.25, compared as bits
const POISON = 0x7fc00000;

function replaceOnce(source, from, to) {
  if (source.split(from).length !== 2) throw new Error("Private Laya GeGLU source contract changed");
  return source.replace(from, to);
}

/** Transform only the activation and epilogue of the existing paired Q8 template. */
export function layaGeGluSource(paired, original) {
  const gelu = `fn gelu(x: f32) -> f32 {
  if (x >= 10.0) { return x; }
  if (x <= -10.0) { return 0.0; }
  let c = 0.7978845608028654;
  return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}`;
  const erf = original.match(/fn erf\(x: f32\) -> f32 \{[\s\S]*?\n\}/g);
  const expression = "0.5 * u * (1.0 + erf(u * 0.70710678118)) * gate";
  if (erf?.length !== 1 || !original.includes(`A[idx] = ${expression};`) || /enable f16|vec4<f16>/.test(paired)) {
    throw new Error("Private Laya GeGLU requires the original erf and FP32 paired source");
  }
  const output = replaceOnce(paired, gelu, erf[0]);
  return replaceOnce(output, `  let gate = value.x + 0.0;
  let up = value.y + 0.0;
  Y[o] = gelu(gate) * up;`, `  let u = value.x + 0.0;
  let gate = value.y + 0.0;
  Y[o] = ${expression};`);
}

function tuningKey(trunk) {
  return JSON.stringify({ kernel: trunk.gpuKernel, selection: [...trunk.tuning.selection].sort(),
    kernelName: trunk.mm.kernel, f16: trunk.mm.f16, groups: trunk.mm.groups, splitTarget: trunk.mm.splitTarget });
}

/** This inspects the real selected baseline, without replacing or recalibrating it. */
export function layaGeGluIneligibleReason(trunk, tokens) {
  const s = LAYA_GEGLU_SHAPE, cfg = trunk.cfg, mm = trunk.mm;
  if (tokens !== s.tokens) return "outside-active-T512";
  if (cfg.hidden !== s.K || cfg.intermediate !== s.I || cfg.layers !== s.layers || cfg.head_ff !== 4096) return "outside-recorded-shape";
  if (!mm || mm.kernel !== "matmul" || mm.f16 !== false || mm.groups !== 1 || mm.splitTarget !== 256 || rowsPerThread(tokens, mm) !== 4) return "unsupported-baseline";
  if (trunk.device.limits.maxComputeWorkgroupStorageSize < 24576) return "paired-shared-memory-limit";
  const selected = trunk.gpuKernel === "auto" ? selectedMatmulKernel(trunk.tuning.selection, 2 * s.I, s.K, tokens) : trunk.gpuKernel;
  if (selected !== "generic") return "baseline-is-not-selected-generic";
  if (mmSplits(tokens, 2 * s.I, s.K, mm.splitTarget, 64, 64) !== 1 || mmSplits(tokens, s.I, s.K, mm.splitTarget, 64, 64) !== 1) return "split-baseline-or-candidate";
  for (let layer = 0; layer < s.layers; layer++) {
    const e = trunk.tensors.get(`enc.${layer}.wi`), t = e?.info;
    if (t?.dtype !== "q8" || t.block !== 32 || t.shape?.length !== 2 || t.shape[0] !== 2 * s.I || t.shape[1] !== s.K ||
      t.size !== 2 * s.I * s.K || t.scales_size !== 2 * s.I * (s.K / 32) * 4 || !e.buf || !e.sbuf) return "unsupported-wi-layout";
    const alignment = trunk.device.limits.minStorageBufferOffsetAlignment;
    if ((s.I * s.K) % alignment || (s.I * (s.K / 32) * 4) % alignment) return "unaligned-weight-half";
  }
  return null;
}

/** The caller arms this with the actual requests immediately before its decide call. */
export function armLayaGeGluRequest(trunk, { role, requests } = {}) {
  trunk.layaGeGluPermit = null;
  if (role !== LAYA_GEGLU_SHAPE.role) return { armed: false, reason: "outside-first-long-role" };
  let requestJson;
  try { requestJson = JSON.stringify(requests); }
  catch { return { armed: false, reason: "invalid-request-context" }; }
  if (requestJson !== PINNED_LONG_JSON) return { armed: false, reason: "outside-pinned-long-request" };
  trunk.layaGeGluPermit = Object.freeze({ role, requestJson });
  return { armed: true, role };
}

async function validatorPipelines(device) {
  const params = "struct P { len: u32, activeCount: u32, sentinel: u32, poison: u32 }\n@group(0) @binding(0) var<uniform> p: P;\n";
  const fill = params + `@group(0) @binding(1) var<storage, read_write> values: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * 65535u * 256u;
  if (i >= p.len) { return; }
  values[i] = select(p.sentinel, p.poison, i < p.activeCount);
}`;
  const compare = params + `@group(0) @binding(1) var<storage, read> baseline: array<u32>;
@group(0) @binding(2) var<storage, read> candidate: array<u32>;
@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * 65535u * 256u;
  if (i >= p.len) { return; }
  let a = baseline[i]; let b = candidate[i];
  if (i < p.activeCount) {
    atomicAdd(&status[4], 1u);
    if (a != b) { atomicAdd(&status[0], 1u); }
    if ((a & 0x7f800000u) == 0x7f800000u) { atomicAdd(&status[1], 1u); }
    if ((b & 0x7f800000u) == 0x7f800000u) { atomicAdd(&status[2], 1u); }
  } else {
    atomicAdd(&status[5], 1u);
    if (a != p.sentinel) { atomicAdd(&status[3], 1u); }
    if (b != p.sentinel) { atomicAdd(&status[3], 1u); }
  }
}`;
  const [pFill, pCompare] = await Promise.all([pipeline(device, fill, "private_laya_geglu_fill"), pipeline(device, compare, "private_laya_geglu_compare")]);
  return { fill: pFill, compare: pCompare };
}

/** Call on the already loaded trunk. Switching lanes never changes its tuning state. */
export async function configureLayaGeGlu(trunk, { mode = "off" } = {}) {
  trunk.layaGeGluPermit = null;
  if (!["off", "on", "validate"].includes(mode)) throw new Error("Private Laya GeGLU mode must be off, on, or validate");
  if (mode === "off") {
    if (trunk.layaGeGlu) trunk.layaGeGlu.mode = "off";
    return { mode: "off" };
  }
  const reason = layaGeGluIneligibleReason(trunk, LAYA_GEGLU_SHAPE.tokens);
  if (reason) throw new Error(`Private Laya GeGLU is ineligible: ${reason}`);
  let state = trunk.layaGeGlu;
  let rebuild = false;
  if (!state) {
    const d = trunk.device;
    const spec = { weights: "q8", f16: false, rows: 4, groups: 1, splitTarget: trunk.mm.splitTarget };
    const code = layaGeGluSource(trunk.wgsl("matmul_gelu", spec), trunk.wgsl("geglu"));
    const types = ["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage", "storage"];
    const layout = d.createBindGroupLayout({ label: "private Laya GeGLU", entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
    const p = await pipeline(d, code, "private_laya_geglu_r4_f32", d.createPipelineLayout({ bindGroupLayouts: [layout] }));
    state = { pipeline: p, layout, tuning: trunk.tuning, mm: trunk.mm, tuningKey: tuningKey(trunk), mode: "off", owned: [], statusBytes: STATUS_BYTES };
    trunk.layaGeGlu = state;
    rebuild = true;
  }
  if (state.tuning !== trunk.tuning || state.mm !== trunk.mm || state.tuningKey !== tuningKey(trunk)) throw new Error("Private Laya GeGLU tuning state changed");
  if (mode === "validate" && !state.validator) {
    state.validator = await validatorPipelines(trunk.device);
    rebuild = true;
  }
  state.mode = mode;
  if (rebuild) buildLayaGeGluGroups(trunk);
  return { mode, shape: LAYA_GEGLU_SHAPE, tuningKey: state.tuningKey };
}

export function buildLayaGeGluGroups(trunk) {
  const state = trunk.layaGeGlu;
  if (!state) return;
  for (const b of state.owned) b.destroy();
  state.owned = [];
  const d = trunk.device, U = GPUBufferUsage, { I, K, layers, tokens } = LAYA_GEGLU_SHAPE;
  const make = (size, usage) => { const b = d.createBuffer({ size, usage }); state.owned.push(b); return b; };
  // The existing act allocation is cap * max(I, head_ff), larger than cap * I here.
  // Check its entire inactive tail, including padding when active T is below cap.
  const length = trunk.act.size / 4;
  const group = (layout, buffers) => d.createBindGroup({ layout, entries: buffers.map((resource, binding) => ({ binding, resource: resource.buffer ? resource : { buffer: resource } })) });
  const params = trunk.uniform([I, K, 0, 0]);
  const checkParams = state.validator ? trunk.uniform([length, tokens * I, SENTINEL, POISON]) : null;
  state.shadow = state.validator ? make(length * 4, U.STORAGE) : null;
  state.read = state.validator ? make(layers * STATUS_BYTES, U.MAP_READ | U.COPY_DST) : null;
  state.length = length;
  state.ops = Array.from({ length: layers }, (_, layer) => {
    const e = trunk.tensors.get(`enc.${layer}.wi`), weightBytes = I * K, scaleBytes = I * (K / 32) * 4;
    const resources = [trunk.globals, params, trunk.h, { buffer: e.buf, offset: 0, size: weightBytes }, { buffer: e.sbuf, offset: 0, size: scaleBytes },
      { buffer: e.buf, offset: weightBytes, size: weightBytes }, { buffer: e.sbuf, offset: scaleBytes, size: scaleBytes }];
    const op = { group: group(state.layout, [...resources, trunk.act, trunk.part]) };
    if (state.validator) {
      op.status = make(STATUS_BYTES, U.STORAGE | U.COPY_SRC | U.COPY_DST);
      op.shadow = group(state.layout, [...resources, state.shadow, trunk.part]);
      op.fill = [trunk.act, state.shadow].map((b) => group(state.validator.fill.getBindGroupLayout(0), [checkParams, b]));
      op.compare = group(state.validator.compare.getBindGroupLayout(0), [checkParams, trunk.act, state.shadow, op.status]);
    }
    return op;
  });
}

export function activeLayaGeGlu(trunk, tokens) {
  // Consume before every early return, including off mode and ineligible active lengths.
  const permit = trunk.layaGeGluPermit;
  trunk.layaGeGluPermit = null;
  const state = trunk.layaGeGlu;
  if (!state || state.mode === "off") return null;
  if (state.tuning !== trunk.tuning || state.mm !== trunk.mm || state.tuningKey !== tuningKey(trunk)) throw new Error("Private Laya GeGLU tuning state changed");
  const requestReason = !permit ? "missing-one-shot-request-permit" :
    permit.role !== LAYA_GEGLU_SHAPE.role || permit.requestJson !== PINNED_LONG_JSON ? "invalid-one-shot-request-permit" : null;
  const reason = requestReason || layaGeGluIneligibleReason(trunk, tokens);
  state.lastEligibility = { tokens, reason, enabled: !reason, mode: state.mode };
  return reason ? null : state;
}

export function encodeLayaGeGlu(pass, state, op, shadow = false) {
  pass.setPipeline(state.pipeline); pass.setBindGroup(0, shadow ? op.shadow : op.group);
  pass.dispatchWorkgroups(Math.ceil(LAYA_GEGLU_SHAPE.I / 64), 8, 1);
}

function validationWork(pass, length) {
  const n = Math.ceil(length / 256);
  pass.dispatchWorkgroups(Math.min(n, 65535), Math.ceil(n / 65535));
}

export function fillLayaGeGlu(pass, state, op) {
  pass.setPipeline(state.validator.fill);
  for (const group of op.fill) { pass.setBindGroup(0, group); validationWork(pass, state.length); }
}

export function compareLayaGeGlu(pass, state, op) {
  encodeLayaGeGlu(pass, state, op, true);
  pass.setPipeline(state.validator.compare); pass.setBindGroup(0, op.compare);
  validationWork(pass, state.length);
}

export async function collectLayaGeGlu(state) {
  await state.read.mapAsync(GPUMapMode.READ);
  try {
    const counts = new Uint32Array(state.read.getMappedRange());
    const activeElements = LAYA_GEGLU_SHAPE.tokens * LAYA_GEGLU_SHAPE.I;
    const inactiveTailElements = state.length - activeElements;
    const layers = state.ops.map((_, layer) => ({ layer, expectedActiveCompared: activeElements, expectedTailCompared: inactiveTailElements,
      bitMismatches: counts[layer * STATUS_WORDS], baselineNonFinite: counts[layer * STATUS_WORDS + 1],
      candidateNonFinite: counts[layer * STATUS_WORDS + 2], tailMismatches: counts[layer * STATUS_WORDS + 3],
      activeCompared: counts[layer * STATUS_WORDS + 4], tailCompared: counts[layer * STATUS_WORDS + 5] }));
    const ok = layers.every((layer) => layer.bitMismatches === 0 && layer.baselineNonFinite === 0 && layer.candidateNonFinite === 0 &&
      layer.tailMismatches === 0 && layer.activeCompared === activeElements && layer.tailCompared === inactiveTailElements);
    const result = { ok, method: "gpu-array-u32-exact-with-positive-coverage", signedZeroChecked: true,
      expectedComparedElementsPerLayer: state.length, expectedInactiveTailElementsPerLayer: inactiveTailElements,
      readbackBytes: counts.byteLength, layers };
    state.lastValidation = result;
    if (!result.ok) throw Object.assign(new Error("Private Laya GeGLU bitwise validation failed"), { validation: result });
    return result;
  } finally { state.read.unmap(); }
}
