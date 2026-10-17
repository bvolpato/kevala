// Gemma 4's vision tower and multimodal projector. Companion packs retain the source BF16
// matrices and position table; normalization, clipping, pooling and all arithmetic stay F32.
import { GpuWeights, encodeMatmul, endChunk, matmulPipelines, mmSplits, pipeline, rowsPerThread, SPLIT_SCRATCH, withGpuErrors } from "./gpu.js";
import { clearGemma4VisionImageData, consumeGemma4VisionImage, disposeGemma4VisionImage, materializeGemma4VisionPixels } from "./gemma4-image.js";
import { createGemma4ImageGpu } from "./gpu-gemma4-image.js";
import { calibrateVisionQkv, fuseVisionQkvOps } from "./gpu-gemma4-vision-qkv.js";

const f32 = (value) => ({ f: value });
const positive = (value) => Number.isSafeInteger(value) && value > 0;
const fail = (detail) => new Error(`Gemma 4 vision ${detail}`);
const PREFIX = "vision_tower.encoder.layers.";
const geometryCache = new WeakMap();
const gpuProjections = new WeakMap();

function encodeVisionMatmul(pass, item) {
  const { pipes, op, tokens } = item;
  if (!item.partialOnly) return encodeMatmul(pass, pipes, op, tokens);
  const rows = rowsPerThread(tokens, pipes);
  const row56 = rows === 4 && pipes.f16 === false && pipes.groups === 1 && Math.ceil(tokens / 56) === Math.ceil(tokens / 64);
  const splits = mmSplits(tokens, op.N, op.K, pipes.splitTarget, 16 * rows, 64 * pipes.groups);
  pass.setPipeline((row56 && pipes.mm56) || pipes.mm[rows]);
  pass.setBindGroup(0, op.group);
  pass.dispatchWorkgroups(Math.ceil(op.N / (64 * pipes.groups)), Math.ceil(tokens / (16 * rows)), splits);
}

/** Returns shared real input clipping bounds, or null when callers must clip separately. */
export function sharedInputClipBounds(...bounds) {
  if (!bounds.length || bounds.some((value) => !Array.isArray(value) || value.length < 2)) return null;
  const [lo, hi] = bounds[0];
  if (lo === -Infinity && hi === Infinity) return null;
  return bounds.every((value) => Object.is(value[0], lo) && Object.is(value[1], hi)) ? [lo, hi] : null;
}

/** Fits both ordered MLP partial planes in the existing, aligned split scratch. */
export function visionMlpReductionPlan(rows, N, K, pipes, capacity, alignment = 256) {
  if (![rows, N, K, capacity, alignment].every(positive) || N % 4 || K % 32 || pipes?.f16 !== false) return null;
  const splits = mmSplits(rows, N, K, pipes.splitTarget, 16 * rowsPerThread(rows, pipes), 64 * pipes.groups);
  if (splits <= 1 || splits > 8) return null;
  const partBytes = splits * rows * N * 4;
  const upOffset = Math.ceil(partBytes / alignment) * alignment;
  const requiredBytes = upOffset + partBytes;
  return Number.isSafeInteger(requiredBytes) && requiredBytes <= capacity ? { splits, partBytes, upOffset, requiredBytes } : null;
}

/** Internal same-device F32 rows. The caller owns their lifetime until decoder readback. */
export function gpuVisionProjection(rows, device, byteLength) {
  const projection = gpuProjections.get(rows);
  return projection && projection.device === device && projection.byteLength === byteLength ? projection : null;
}

function ownProjection(device, byteLength) {
  const buffer = device.createBuffer({ label: "Gemma vision projected rows", size: byteLength, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  let status;
  try {
    status = device.createBuffer({ label: "Gemma vision projection status", size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const rows = Object.freeze({ byteLength, destroy() {
      if (!gpuProjections.delete(rows)) return;
      buffer.destroy();
      status.destroy();
    } });
    gpuProjections.set(rows, Object.freeze({ device, buffer, status, byteLength }));
    return rows;
  } catch (error) { buffer.destroy(); status?.destroy(); throw error; }
}

export function gemma4VisionConfig(header) {
  const config = header.config || {};
  const c = config.vision_config || config;
  const hidden = c.hidden_size;
  const intermediate = c.intermediate_size;
  const layers = c.num_hidden_layers;
  const heads = c.num_attention_heads;
  const kvHeads = c.num_key_value_heads;
  const headDim = c.head_dim;
  const patch = c.patch_size;
  const pool = c.pooling_kernel_size;
  const positionSize = c.position_embedding_size;
  const textHidden = config.text_hidden_size || config.text_config?.hidden_size;
  if (![hidden, intermediate, layers, heads, kvHeads, headDim, patch, pool, positionSize, textHidden].every(positive)) throw fail("config has invalid dimensions");
  if (hidden % 32 || intermediate % 32 || (3 * patch * patch) % 32 || (heads * headDim) % 32 || headDim % 4 || textHidden % 4 || heads % kvHeads) throw fail("config has unsupported matrix or head dimensions");
  if (headDim > 256 || c.hidden_activation !== "gelu_pytorch_tanh" || c.attention_bias || c.attention_dropout) throw fail("config requires GELU, unbiased projections, and zero attention dropout");
  const eps = c.rms_norm_eps;
  const theta = c.rope_parameters?.rope_theta;
  if (!Number.isFinite(eps) || eps <= 0 || !Number.isFinite(theta) || theta <= 0 || !["axial", "default"].includes(c.rope_parameters?.rope_type)) throw fail("config requires axial RoPE and positive normalization parameters");
  return { hidden, intermediate, layers, heads, kvHeads, headDim, patch, pool, positionSize, textHidden, eps, theta, clipped: c.use_clipped_linears === true, standardize: c.standardize === true };
}

/** Validates the processor's padded patch grid and maps each real spatial pool to its patches. */
export function visionImagePlan(image, cfg, compact = true) {
  return imagePlan(image, cfg, compact, false);
}

function imagePlan(image, cfg, compact, allowRaw) {
  const prepared = consumeGemma4VisionImage(image, cfg, compact);
  try {
    const { patches, positions, softTokens } = prepared || image;
    const patchWidth = 3 * cfg.patch * cfg.patch;
    if ((!prepared && !(patches instanceof Float32Array)) || !(positions instanceof Int32Array) || positions.length % 2) throw fail("expects Float32 patches and signed Int32 position pairs");
    const rows = positions.length / 2;
    if (!rows || (!prepared && patches.length !== rows * patchWidth) || rows % (cfg.pool * cfg.pool)) throw fail("patches and padded positions have inconsistent dimensions");
    const key = [compact, cfg.patch, cfg.pool, cfg.positionSize, cfg.headDim, cfg.theta].join(":");
    let geometry = geometryCache.get(cfg);
    if (!geometry || geometry.key !== key || geometry.source.length !== positions.length || !positions.every((value, i) => value === geometry.source[i])) {
      geometry = { ...imageGeometry(positions, cfg, compact), key, source: positions.slice() };
      geometryCache.set(cfg, geometry);
    }
    const { sourceRows, softRows, activePositions, rope, pools } = geometry;
    if (softTokens != null && softTokens !== softRows) throw fail(`processor reports ${softTokens} soft tokens but positions produce ${softRows}`);
    const raw = allowRaw && prepared?.raw && sourceRows.every((row, i) => row === i) ? prepared.raw : null;
    const pixels = prepared ? (raw ? null : materializeGemma4VisionPixels(prepared)) : new Float32Array(sourceRows.length * patchWidth);
    if (!prepared) {
      for (let i = 0; i < patches.length; i++) {
        if (!Number.isFinite(patches[i]) || patches[i] < 0 || patches[i] > 1) {
          pixels.fill(0);
          throw fail("patch pixels must be finite values between zero and one");
        }
      }
      for (let row = 0; row < sourceRows.length; row++) for (let col = 0; col < patchWidth; col++) pixels[row * patchWidth + col] = 2 * (patches[sourceRows[row] * patchWidth + col] - 0.5);
    }
    return { rows: sourceRows.length, softRows, pixels, positions: activePositions.slice(), rope: rope.slice(), pools: pools.slice(), ...(allowRaw && prepared ? { ownedImage: prepared, raw } : {}) };
  } catch (error) { clearGemma4VisionImageData(prepared); throw error; }
}

function imageGeometry(positions, cfg, compact) {
  const rows = positions.length / 2;
  let width = 0;
  let height = 0;
  const seen = new Set();
  const validRows = [];
  for (let row = 0; row < rows; row++) {
    const x = positions[row * 2], y = positions[row * 2 + 1];
    if (x === -1 && y === -1) continue;
    if (x < 0 || y < 0 || x >= cfg.positionSize || y >= cfg.positionSize) throw fail("patch position is outside the learned position table");
    const key = `${x}:${y}`;
    if (seen.has(key)) throw fail("patch positions contain a duplicate");
    seen.add(key);
    validRows.push(row);
    width = Math.max(width, x + 1);
    height = Math.max(height, y + 1);
  }
  if (!width || width % cfg.pool || height % cfg.pool || seen.size !== width * height) throw fail("real patch positions must form a complete rectangular pooling grid");
  const softRows = width * height / (cfg.pool * cfg.pool);
  const sourceRows = compact ? validRows : Array.from({ length: rows }, (_, row) => row);
  const activeRows = sourceRows.length;
  const activePositions = activeRows === rows ? positions : Int32Array.from(sourceRows.flatMap((row) => [positions[row * 2], positions[row * 2 + 1]]));
  const pools = Array.from({ length: softRows }, () => []);
  for (let row = 0; row < activeRows; row++) {
    const x = activePositions[row * 2], y = activePositions[row * 2 + 1];
    if (x < 0) continue;
    pools[Math.floor(y / cfg.pool) * (width / cfg.pool) + Math.floor(x / cfg.pool)].push(row);
  }
  const rope = new Float32Array(activeRows * cfg.headDim);
  const quarter = cfg.headDim / 4;
  for (let row = 0; row < activeRows; row++) for (let axis = 0; axis < 2; axis++) for (let pair = 0; pair < quarter; pair++) {
    const inverse = Math.fround(1 / Math.fround(Math.pow(cfg.theta, pair / quarter)));
    const angle = Math.fround(activePositions[row * 2 + axis] * inverse);
    const at = row * cfg.headDim + (axis * quarter + pair) * 2;
    rope[at] = Math.cos(angle);
    rope[at + 1] = Math.sin(angle);
  }
  return { sourceRows, softRows, activePositions: activePositions.slice(), rope, pools: Uint32Array.from(pools.flat()) };
}

export class GpuGemma4Vision {
  constructor(gpu, layout, header, { compactPatches = true, submitMode = "split", fusedAttention = true, gpuKernel = "auto" } = {}) {
    this.runtime = gpu;
    this.gpu = gpu;
    this.device = gpu.device;
    this.wgsl = gpu.wgsl;
    this.cfg = gemma4VisionConfig(header);
    this.compactPatches = compactPatches;
    this.prepareCompactPixels = true;
    this.prepareGpuPixels = true;
    this.lastGpuPixelDispatches = 0;
    this.gpuPixelDiagnostics = { enabled: false, reason: "disabled" };
    this.fusedAttention = fusedAttention === true && this.cfg.headDim === 64;
    this.fuseInputClips = true;
    this.fuseVisionOps = true;
    this.fuseResidualNorm = true;
    this.fuseReduceNorm = true;
    this.fuseReduceGelu = true;
    this.fuseReduceResidual = true;
    this.gpuKernel = gpuKernel;
    this.fuseQkv = gpuKernel === "auto";
    this.lastQkvFusions = 0;
    this.ensuredFuseQkv = null;
    this.ensuredProfiler = null;
    this.ensuredFuseInputClips = null;
    this.ensuredFuseVisionOps = null;
    this.ensuredFuseResidualNorm = null;
    this.ensuredFuseReduceNorm = null;
    this.ensuredFuseReduceGelu = null;
    this.ensuredFuseReduceResidual = null;
    this.fusionStats = null;
    this.clipFusionStats = null;
    if (!["await", "split", "none"].includes(submitMode)) throw fail("submission mode must be await, split, or none");
    this.submitMode = submitMode;
    this.weights = new GpuWeights(this.device, layout);
    this.owned = [];
    for (const entry of this.weights.tensors.values()) {
      this.checkBuffer(entry.info.size, entry.info.name);
      if (!["bf16", "f32"].includes(entry.info.dtype)) throw fail(`companion tensor ${entry.info.name} must retain BF16 or F32 precision`);
      if (/\.(input_min|input_max|output_min|output_max)$/.test(entry.info.name)) this.weights.keepHost.add(entry.info.name);
    }
  }

  write(dst, bytes) { this.weights.write(dst, bytes); }
  missing() { return this.weights.missing(); }

  checkBuffer(size, label) {
    const limit = Math.min(this.device.limits.maxBufferSize, this.device.limits.maxStorageBufferBindingSize);
    if (!Number.isSafeInteger(size) || size < 0 || Math.max(16, size) > limit) throw Object.assign(fail(`buffer ${label} needs ${Math.ceil(size / 1048576)} MiB; this device allows ${Math.floor(limit / 1048576)} MiB. Use imageTokens: 70.`), { code: "WEBGPU_INIT" });
  }

  tensor(name, shape, dtype) {
    const entry = this.weights.get(name);
    if (entry.info.shape.length !== shape.length || shape.some((value, i) => value !== entry.info.shape[i]) || (dtype && entry.info.dtype !== dtype)) throw fail(`tensor ${name} has an incompatible shape or dtype`);
    return entry;
  }

  bounds(name) {
    if (!this.cfg.clipped) return [-Infinity, Infinity, -Infinity, Infinity];
    const values = ["input_min", "input_max", "output_min", "output_max"].map((field) => this.tensor(`${name}.${field}`, [1], "f32").host?.[0]);
    if (values.some((value) => typeof value !== "number" || Number.isNaN(value)) || values[0] > values[1] || values[2] > values[3]) throw fail(`projection ${name} has invalid clipping bounds`);
    return values;
  }

  async init() {
    const missing = this.missing();
    if (missing.length) throw fail(`companion is missing tensor ${missing[0]}`);
    const c = this.cfg;
    this.tensor("vision_tower.patch_embedder.input_proj", [c.hidden, 3 * c.patch * c.patch]);
    this.tensor("vision_tower.patch_embedder.position_embedding_table", [2, c.positionSize, c.hidden]);
    this.tensor("embed_vision.embedding_projection", [c.textHidden, c.hidden]);
    this.clips = new Map();
    const matrixTypes = new Set();
    for (const name of ["vision_tower.patch_embedder.input_proj", "embed_vision.embedding_projection"]) matrixTypes.add(this.weights.get(name).info.dtype);
    for (let layer = 0; layer < c.layers; layer++) {
      const prefix = `${PREFIX}${layer}.`;
      for (const name of ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"]) this.tensor(prefix + name, [c.hidden], "f32");
      for (const name of ["q_norm", "k_norm"]) this.tensor(prefix + "self_attn." + name, [c.headDim], "f32");
      for (const [name, n, k] of [
        ["self_attn.q_proj", c.heads * c.headDim, c.hidden], ["self_attn.k_proj", c.kvHeads * c.headDim, c.hidden],
        ["self_attn.v_proj", c.kvHeads * c.headDim, c.hidden], ["self_attn.o_proj", c.hidden, c.heads * c.headDim],
        ["mlp.gate_proj", c.intermediate, c.hidden], ["mlp.up_proj", c.intermediate, c.hidden], ["mlp.down_proj", c.hidden, c.intermediate],
      ]) {
        const base = prefix + name;
        matrixTypes.add(this.tensor(base + ".linear", [n, k]).info.dtype);
        this.clips.set(base, this.bounds(base));
      }
    }
    if (c.standardize) for (const name of ["std_bias", "std_scale"]) this.tensor("vision_tower." + name, [c.hidden], "f32");
    this.mm = new Map();
    for (const dtype of matrixTypes) this.mm.set(dtype, await matmulPipelines(this.device, this.wgsl, "matmul", dtype));
    const kinds = {
      position: ["uniform", "read-only-storage", "read-only-storage", "storage"],
      clip: ["uniform", "read-only-storage", "storage"],
      norm: ["uniform", "read-only-storage", "read-only-storage", "read-only-storage", "storage"],
      norm_rope: ["uniform", "read-only-storage", "read-only-storage", "read-only-storage", "storage"],
      residual_norm: ["uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage", "storage"],
      reduce_residual: ["uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage", "storage"],
      reduce_norm_rope: ["uniform", "read-only-storage", "read-only-storage", "read-only-storage", "storage"],
      rope: ["uniform", "read-only-storage", "read-only-storage", "storage"],
      scores: ["uniform", "read-only-storage", "read-only-storage", "read-only-storage", "storage"],
      attention: ["uniform", "storage", "read-only-storage", "storage"],
      gelu: ["uniform", "read-only-storage", "read-only-storage", "storage"],
      reduce_gelu: ["uniform", "read-only-storage", "read-only-storage", "storage"],
      pool: ["uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage"],
      pool_norm: ["uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage"],
      finite: ["uniform", "read-only-storage", "storage"],
    };
    if (this.fusedAttention) kinds.attention_fused = ["uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage"];
    this.p = {};
    for (const [name, types] of Object.entries(kinds)) {
      const layout = this.device.createBindGroupLayout({ entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
      const weights = this.weights.get("vision_tower.patch_embedder.position_embedding_table").info.dtype;
      const kernel = name === "scores" && c.headDim === 64 ? "gemma4_vision_scores_tiled" : "gemma4_vision_" + name;
      this.p[name] = await pipeline(this.device, this.wgsl(kernel, { weights }), kernel, this.device.createPipelineLayout({ bindGroupLayouts: [layout] }));
    }
    this.imageGpu = await createGemma4ImageGpu(this.runtime, this.wgsl);
    this.qkv = await calibrateVisionQkv(this.runtime, this.wgsl, { cfg: c, weights: this.weights, clips: this.clips, matmul: this.mm.get("bf16"), norm: this.p.reduce_norm_rope, enabled: this.gpuKernel === "auto" });
  }

  reset() {
    this.imageGpu?.reset();
    for (const buffer of this.owned) buffer.destroy();
    this.owned = [];
    this.shape = null;
    this.attentionMode = null;
    this.ensuredFuseInputClips = null;
    this.ensuredFuseVisionOps = null;
    this.ensuredFuseResidualNorm = null;
    this.ensuredFuseReduceNorm = null;
    this.ensuredFuseReduceGelu = null;
    this.ensuredFuseReduceResidual = null;
    this.ensuredFuseQkv = null;
    this.ensuredProfiler = null;
    this.fusionStats = null;
    this.clipFusionStats = null;
  }

  ensure(rows, softRows) {
    const fuseInputClips = this.fuseInputClips === true;
    const fuseVisionOps = this.fuseVisionOps === true;
    const fuseResidualNorm = fuseVisionOps && this.fuseResidualNorm === true && this.cfg.hidden <= 1024;
    const fuseReduceNorm = fuseVisionOps && this.fuseReduceNorm === true && this.cfg.headDim === 64;
    const fuseReduceGelu = fuseVisionOps && this.fuseReduceGelu === true;
    const fuseReduceResidual = fuseResidualNorm && this.fuseReduceResidual === true;
    const profiler = !!this.profiler;
    const fuseQkv = this.fuseQkv === true && this.gpuKernel === "auto" && this.qkv?.selected === true && !profiler && fuseInputClips && fuseReduceNorm && rows === 108;
    if (this.shape?.[0] === rows && this.shape[1] === softRows && this.ensuredFuseInputClips === fuseInputClips && this.ensuredFuseVisionOps === fuseVisionOps && this.ensuredFuseResidualNorm === fuseResidualNorm && this.ensuredFuseReduceNorm === fuseReduceNorm && this.ensuredFuseReduceGelu === fuseReduceGelu && this.ensuredFuseReduceResidual === fuseReduceResidual && this.ensuredFuseQkv === fuseQkv && this.ensuredProfiler === profiler) return;
    this.reset();
    const d = this.device, c = this.cfg, U = GPUBufferUsage;
    const make = (size, usage = U.STORAGE, label = "scratch") => {
      this.checkBuffer(size, label);
      const buffer = d.createBuffer({ size: Math.max(16, size), usage, label: `Gemma vision ${label}` });
      this.owned.push(buffer);
      return buffer;
    };
    const floats = (size, usage = 0, label) => make(size * 4, U.STORAGE | usage, label);
    const uniform = (values) => {
      const bytes = new ArrayBuffer(Math.max(16, values.length * 4));
      const ints = new Uint32Array(bytes), vals = new Float32Array(bytes);
      values.forEach((value, i) => typeof value === "object" ? vals[i] = value.f : ints[i] = value);
      const buffer = make(bytes.byteLength, U.UNIFORM | U.COPY_DST, "parameters");
      d.queue.writeBuffer(buffer, 0, bytes);
      return buffer;
    };
    const zeros = floats(4, 0, "zeros");
    this.pixels = floats(rows * 3 * c.patch * c.patch, U.COPY_DST, "pixels");
    this.positions = make(rows * 8, U.STORAGE | U.COPY_DST, "positions");
    this.rope = floats(rows * c.headDim, U.COPY_DST, "rotary positions");
    this.pools = make(softRows * c.pool * c.pool * 4, U.STORAGE | U.COPY_DST, "pool indices");
    const x = floats(rows * c.hidden), residual = floats(rows * c.hidden), normed = floats(rows * c.hidden);
    const clip = floats(rows * Math.max(c.hidden, c.intermediate, c.heads * c.headDim));
    const q = floats(rows * c.heads * c.headDim), nq = floats(rows * c.heads * c.headDim);
    const k = floats(rows * c.kvHeads * c.headDim), nk = floats(rows * c.kvHeads * c.headDim);
    const v = floats(rows * c.kvHeads * c.headDim), nv = floats(rows * c.kvHeads * c.headDim);
    const attention = floats(rows * c.heads * c.headDim), projected = floats(rows * c.hidden);
    const fusedAttention = this.fusedAttention && rows <= 630;
    const scores = fusedAttention ? null : floats(rows * rows * c.heads, 0, "attention scores");
    const gate = floats(rows * c.intermediate), up = floats(rows * c.intermediate), activated = floats(rows * c.intermediate);
    const pooled = floats(softRows * c.hidden), poolNorm = floats(softRows * c.hidden);
    const output = floats(softRows * c.textHidden, U.COPY_SRC, "projected image rows");
    this.finiteParams = uniform([softRows * c.textHidden, 0, 0, 0]);
    const part = floats(SPLIT_SCRATCH, 0, "split matmul");
    this.readback = make(softRows * c.textHidden * 4, U.MAP_READ | U.COPY_DST, "readback");
    const globals = (tokens) => uniform([tokens, 0, 0, 0]);
    const g = globals(rows), gp = globals(softRows);
    const ops = this.ops = [];
    let reduceNormPasses = 0;
    let reduceGeluPasses = 0;
    const clipFusionStats = { enabled: fuseInputClips, originalInputClipPasses: 0, removedInputClipPasses: 0, qkvLayers: 0, gateUpLayers: 0, downLayers: 0, attentionLayers: 0 };
    const resource = (buffer) => buffer.buffer ? buffer : { buffer };
    const dispatch = (name, values, buffers, groups, label) => {
      const p = this.p[name], params = uniform(values);
      const group = d.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [params, ...buffers].map((buffer, binding) => ({ binding, resource: resource(buffer) })) });
      ops.push({ p, group, groups, label });
      if (name === "reduce_norm_rope") ops.at(-1).qkvNorm = { params, values, buffers };
      return ops.at(-1);
    };
    const mm = (name, input, result, tokens, bounds, label, skipInputClip = false, partials = part) => {
      const e = this.weights.get(name), [N, K] = e.info.shape, pipes = this.mm.get(e.info.dtype);
      if (bounds) {
        clipFusionStats.originalInputClipPasses++;
        if (!skipInputClip) {
          dispatch("clip", [tokens * K, 0, 0, 0, f32(bounds[0]), f32(bounds[1]), f32(0), f32(0)], [input, clip], [Math.ceil(tokens * K / 1024)], "vision.clip");
          input = clip;
        }
      }
      const u = uniform([N, K, 0, 0]);
      const buffers = [tokens === rows ? g : gp, u, input, e.buf, zeros, zeros, result, partials];
      const op = { N, K,
        group: d.createBindGroup({ layout: pipes.layout, entries: buffers.map((buffer, binding) => ({ binding, resource: resource(buffer) })) }),
        reduce: d.createBindGroup({ layout: pipes.reduceLayout, entries: [buffers[0], u, partials, zeros, result].map((buffer, binding) => ({ binding, resource: resource(buffer) })) }),
      };
      ops.push({ pipes, op, tokens, label });
      if (["vision.mm.q", "vision.mm.k", "vision.mm.v"].includes(label)) ops.at(-1).qkvMatrix = { dtype: e.info.dtype, mode: 0, bias: 0, buffers };
      return ops.at(-1);
    };
    const norm = (input, result, width, tokens, weight = null, add = null, bounds = [-Infinity, Infinity, -Infinity, Infinity], label, clampAfter = null) => {
      const useClampAfter = !!clampAfter && bounds[2] === -Infinity && bounds[3] === Infinity;
      const lo = useClampAfter ? clampAfter[0] : bounds[2];
      const hi = useClampAfter ? clampAfter[1] : bounds[3];
      const values = [tokens, width, weight ? 1 : 0, add ? 1 : 0, f32(c.eps), f32(lo), f32(hi), useClampAfter ? 1 : 0];
      const buffers = [input, weight ? this.weights.get(weight).buf : zeros, add || zeros, result];
      const op = dispatch("norm", values, buffers, [tokens], label);
      op.norm = { values, buffers };
      return op;
    };
    mm("vision_tower.patch_embedder.input_proj", this.pixels, x, rows, null, "vision.mm.patch_embed");
    dispatch("position", [rows, c.hidden, c.positionSize, 0], [this.positions, this.weights.get("vision_tower.patch_embedder.position_embedding_table").buf, x], [Math.ceil(rows * c.hidden / 256)], "vision.position");
    for (let layer = 0; layer < c.layers; layer++) {
      const prefix = `${PREFIX}${layer}.`;
      const attn = prefix + "self_attn.", mlp = prefix + "mlp.";
      const qBounds = this.clips.get(attn + "q_proj");
      const kBounds = this.clips.get(attn + "k_proj");
      const vBounds = this.clips.get(attn + "v_proj");
      const qkvInputBounds = fuseInputClips ? sharedInputClipBounds(qBounds, kBounds, vBounds) : null;
      if (qkvInputBounds) clipFusionStats.qkvLayers++;
      const inputNormBounds = qkvInputBounds ? qkvInputBounds : null;
      norm(x, normed, c.hidden, rows, prefix + "input_layernorm", null, undefined, "vision.norm.input", inputNormBounds);
      for (const [name, result, normalized, heads, weight] of [["q", q, nq, c.heads, attn + "q_norm"], ["k", k, nk, c.kvHeads, attn + "k_norm"], ["v", v, nv, c.kvHeads, null]]) {
        const base = attn + name + "_proj", bounds = this.clips.get(base);
        const projection = mm(base + ".linear", normed, result, rows, bounds, `vision.mm.${name}`, !!qkvInputBounds);
        const { pipes, op } = projection;
        const splits = mmSplits(rows, op.N, op.K, pipes.splitTarget, 16 * rowsPerThread(rows, pipes), 64 * pipes.groups);
        if (fuseReduceNorm && splits > 1) {
          projection.partialOnly = true;
          dispatch("reduce_norm_rope", [rows, heads, c.headDim, Number(Boolean(weight)), f32(c.eps), f32(bounds[2]), f32(bounds[3]), Number(name === "v"), splits, 0, 0, 0], [part, weight ? this.weights.get(weight).buf : zeros, this.rope, normalized], [rows * heads], `vision.reduce_norm_rope.${name}`);
          reduceNormPasses++;
        } else if (fuseVisionOps) {
          dispatch("norm_rope", [rows, heads, c.headDim, Number(Boolean(weight)), f32(c.eps), f32(bounds[2]), f32(bounds[3]), Number(name === "v")], [result, weight ? this.weights.get(weight).buf : zeros, this.rope, normalized], [rows * heads], name === "v" ? "vision.norm_head.v" : `vision.norm_rope.${name}`);
        } else {
          norm(result, normalized, c.headDim, rows * heads, weight, null, bounds, `vision.norm.${name}`);
          if (name !== "v") dispatch("rope", [rows, heads, c.headDim, 0], [this.rope, normalized, result], [Math.ceil(rows * heads * c.headDim / 256)], `vision.rope.${name}`);
        }
      }
      const oBounds = this.clips.get(attn + "o_proj");
      const attentionInputBounds = fuseInputClips && fusedAttention ? sharedInputClipBounds(oBounds) : null;
      if (attentionInputBounds) clipFusionStats.attentionLayers++;
      if (fusedAttention) dispatch("attention_fused", [rows, c.heads, c.kvHeads, c.headDim, f32(attentionInputBounds?.[0] ?? 0), f32(attentionInputBounds?.[1] ?? 0), attentionInputBounds ? 1 : 0, 0], [fuseVisionOps ? nq : q, fuseVisionOps ? nk : k, nv, this.positions, attention], [Math.ceil(rows / 16), c.heads], "vision.attention_fused");
      else {
        const groups = c.headDim === 64 ? Math.ceil(rows / 16) ** 2 * c.heads : Math.ceil(rows * rows * c.heads / 256);
        dispatch("scores", [rows, c.heads, c.kvHeads, c.headDim], [fuseVisionOps ? nq : q, fuseVisionOps ? nk : k, this.positions, scores], [Math.min(65535, groups), Math.ceil(groups / 65535)], "vision.scores");
        dispatch("attention", [rows, c.heads, c.kvHeads, c.headDim], [scores, nv, attention], [rows, c.heads], "vision.attention");
      }
      const oProjection = mm(attn + "o_proj.linear", attention, projected, rows, oBounds, "vision.mm.o", !!attentionInputBounds);
      norm(projected, residual, c.hidden, rows, prefix + "post_attention_layernorm", x, oBounds, "vision.norm.post_attention").projection = oProjection;
      const gateBounds = this.clips.get(mlp + "gate_proj"), upBounds = this.clips.get(mlp + "up_proj");
      const gateUpInputBounds = fuseInputClips ? sharedInputClipBounds(gateBounds, upBounds) : null;
      if (gateUpInputBounds) clipFusionStats.gateUpLayers++;
      norm(residual, normed, c.hidden, rows, prefix + "pre_feedforward_layernorm", null, undefined, "vision.norm.pre_feedforward", gateUpInputBounds);
      const gateWeight = this.weights.get(mlp + "gate_proj.linear"), upWeight = this.weights.get(mlp + "up_proj.linear");
      const reduceGelu = fuseReduceGelu && gateWeight.info.dtype === upWeight.info.dtype
        ? visionMlpReductionPlan(rows, c.intermediate, c.hidden, this.mm.get(gateWeight.info.dtype), part.size, d.limits.minStorageBufferOffsetAlignment) : null;
      const gatePart = reduceGelu ? { buffer: part, offset: 0, size: reduceGelu.partBytes } : part;
      const upPart = reduceGelu ? { buffer: part, offset: reduceGelu.upOffset, size: reduceGelu.partBytes } : part;
      const gateProjection = mm(mlp + "gate_proj.linear", normed, gate, rows, gateBounds, "vision.mm.gate", !!gateUpInputBounds, gatePart);
      const upProjection = mm(mlp + "up_proj.linear", normed, up, rows, upBounds, "vision.mm.up", !!gateUpInputBounds, upPart);
      if (reduceGelu) gateProjection.partialOnly = upProjection.partialOnly = true;
      const downBounds = this.clips.get(mlp + "down_proj");
      const downInputBounds = fuseInputClips ? sharedInputClipBounds(downBounds) : null;
      if (downInputBounds) clipFusionStats.downLayers++;
      const geluValues = [rows * c.intermediate, f32(downInputBounds?.[0] ?? 0), f32(downInputBounds?.[1] ?? 0), downInputBounds ? 1 : 0, f32(gateBounds[2]), f32(gateBounds[3]), f32(upBounds[2]), f32(upBounds[3])];
      if (reduceGelu) {
        dispatch("reduce_gelu", [...geluValues, reduceGelu.splits, 0, 0, 0], [gatePart, upPart, activated], [Math.ceil(rows * c.intermediate / 1024)], "vision.reduce_gelu");
        reduceGeluPasses += 2;
      } else dispatch("gelu", geluValues, [gate, up, activated], [Math.ceil(rows * c.intermediate / 1024)], "vision.gelu");
      const downProjection = mm(mlp + "down_proj.linear", activated, projected, rows, downBounds, "vision.mm.down", !!downInputBounds);
      norm(projected, x, c.hidden, rows, prefix + "post_feedforward_layernorm", residual, this.clips.get(mlp + "down_proj"), "vision.norm.post_feedforward").projection = downProjection;
      ops.push({ boundary: true });
    }
    const poolNormFused = fuseVisionOps && c.hidden <= 1024;
    if (poolNormFused) {
      dispatch("pool_norm", [softRows, c.hidden, c.pool * c.pool, c.standardize ? 1 : 0, f32(c.eps), f32(-Infinity), f32(Infinity), 0], [x, this.pools, c.standardize ? this.weights.get("vision_tower.std_bias").buf : zeros, c.standardize ? this.weights.get("vision_tower.std_scale").buf : zeros, poolNorm], [softRows], "vision.pool_norm");
    } else {
      dispatch("pool", [softRows, c.hidden, c.pool * c.pool, c.standardize ? 1 : 0], [x, this.pools, c.standardize ? this.weights.get("vision_tower.std_bias").buf : zeros, c.standardize ? this.weights.get("vision_tower.std_scale").buf : zeros, pooled], [Math.ceil(softRows * c.hidden / 1024)], "vision.pool");
      norm(pooled, poolNorm, c.hidden, softRows, null, null, undefined, "vision.norm.pool");
    }
    mm("embed_vision.embedding_projection", poolNorm, output, softRows, null, "vision.mm.project");
    let residualNormPasses = 0;
    let reduceResidualPasses = 0;
    if (fuseResidualNorm) for (let index = 0; index < ops.length - 1; index++) {
      const first = ops[index];
      if (!["vision.norm.post_attention", "vision.norm.post_feedforward"].includes(first.label)) continue;
      const nextIndex = index + (ops[index + 1].boundary ? 2 : 1);
      const second = ops[nextIndex];
      const a = first.norm, b = second?.norm;
      const nextLabel = first.label === "vision.norm.post_attention" ? "vision.norm.pre_feedforward" : "vision.norm.input";
      if (!a || !b || second.label !== nextLabel || a.values[0] !== b.values[0] || a.values[1] !== b.values[1] || b.values[3] !== 0 || a.buffers[3] !== b.buffers[0]) continue;
      const values = [...a.values, b.values[2], b.values[7], 0, 0, ...b.values.slice(4, 7), 0];
      const buffers = [a.buffers[0], a.buffers[1], a.buffers[2], b.buffers[1], a.buffers[3], b.buffers[3]];
      const projection = first.projection;
      const splits = projection && mmSplits(rows, projection.op.N, projection.op.K, projection.pipes.splitTarget, 16 * rowsPerThread(rows, projection.pipes), 64 * projection.pipes.groups);
      const reduceResidual = fuseReduceResidual && splits > 1 && splits <= 8;
      if (reduceResidual) {
        projection.partialOnly = true;
        buffers[0] = part;
        values[10] = splits;
        reduceResidualPasses++;
      }
      const fused = dispatch(reduceResidual ? "reduce_residual" : "residual_norm", values, buffers, first.groups, first.label + (reduceResidual ? ".reduce.next_norm" : ".next_norm"));
      ops.pop();
      ops[index] = fused;
      ops.splice(nextIndex, 1);
      residualNormPasses++;
    }
    const qkvPasses = fuseQkv ? fuseVisionQkvOps(d, this.qkv, ops) : 0;
    this.output = output;
    this.shape = [rows, softRows];
    this.attentionMode = fusedAttention ? "fused-f32" : "split-f32";
    this.ensuredFuseInputClips = fuseInputClips;
    this.ensuredFuseVisionOps = fuseVisionOps;
    this.ensuredFuseResidualNorm = fuseResidualNorm;
    this.ensuredFuseReduceNorm = fuseReduceNorm;
    this.ensuredFuseReduceGelu = fuseReduceGelu;
    this.ensuredFuseReduceResidual = fuseReduceResidual;
    this.ensuredFuseQkv = fuseQkv;
    this.ensuredProfiler = profiler;
    this.fusionStats = { enabled: fuseVisionOps, normRope: fuseVisionOps, poolNorm: poolNormFused, residualNormPasses, reduceNormPasses, reduceGeluPasses, reduceResidualPasses, qkvPasses, qkvRemovedPasses: qkvPasses * 2, removedPasses: (fuseVisionOps ? c.layers * 2 : 0) + Number(poolNormFused) + residualNormPasses + reduceNormPasses + reduceGeluPasses + reduceResidualPasses + qkvPasses * 2 };
    clipFusionStats.removedInputClipPasses = clipFusionStats.qkvLayers * 3 + clipFusionStats.gateUpLayers * 2 + clipFusionStats.downLayers + clipFusionStats.attentionLayers;
    clipFusionStats.remainingInputClipPasses = clipFusionStats.originalInputClipPasses - clipFusionStats.removedInputClipPasses;
    this.clipFusionStats = clipFusionStats;
  }

  async forward(image, { gpuOutput = false } = {}) {
    if (this.running) {
      disposeGemma4VisionImage(image);
      throw fail("encoder is already running an image");
    }
    this.running = true;
    this.lastProfile = null;
    this.lastGpuPixelDispatches = 0;
    this.lastQkvFusions = 0;
    this.gpuPixelDiagnostics = { enabled: false, reason: "disabled" };
    let plan;
    let projection;
    const profiler = this.profiler;
    gpuOutput &&= !profiler;
    try {
      const allowRaw = this.prepareGpuPixels === true && !profiler && this.imageGpu?.enabled === true;
      plan = imagePlan(image, this.cfg, this.compactPatches !== false, allowRaw);
      return await withGpuErrors(this.runtime, async () => {
        this.ensure(plan.rows, plan.softRows);
        const d = this.device;
        const gpuPixels = plan.raw ? await this.imageGpu.prepare(plan.raw, this.pixels) : null;
        if (plan.raw && !gpuPixels) plan.pixels = materializeGemma4VisionPixels(plan.ownedImage);
        this.gpuPixelDiagnostics = { ...this.imageGpu?.diagnostics, enabled: !!gpuPixels, reason: gpuPixels ? "ready" : profiler ? "profiler" : this.prepareGpuPixels ? this.imageGpu?.diagnostics?.lastFallback || this.imageGpu?.diagnostics?.reason || "cpu-input" : "disabled" };
        const bytes = plan.softRows * this.cfg.textHidden * 4;
        if (gpuOutput) projection = ownProjection(d, bytes);
        if (!gpuPixels) d.queue.writeBuffer(this.pixels, 0, plan.pixels);
        for (const [buffer, values] of [[this.positions, plan.positions], [this.rope, plan.rope], [this.pools, plan.pools]]) d.queue.writeBuffer(buffer, 0, values);
        let enc = d.createCommandEncoder({ label: "Gemma vision forward" });
        if (gpuPixels) { this.imageGpu.encode(enc, gpuPixels); this.lastGpuPixelDispatches = 2; }
        let pass = profiler ? null : enc.beginComputePass(), layers = 0;
        for (const item of this.ops) {
          if (item.boundary) {
            if (++layers % 2 === 0) {
              pass?.end();
              enc = await endChunk(d, enc, this.submitMode);
              pass = profiler ? null : enc.beginComputePass();
            }
          } else if (item.op) {
            if (profiler) {
              const measured = profiler.pass(enc, item.label);
              encodeVisionMatmul(measured, item);
              measured.end();
            } else encodeVisionMatmul(pass, item);
          }
          else {
            const measured = profiler ? profiler.pass(enc, item.label) : pass;
            measured.setPipeline(item.p);
            measured.setBindGroup(0, item.group);
            measured.dispatchWorkgroups(...item.groups);
            if (item.qkv === true) this.lastQkvFusions++;
            if (profiler) measured.end();
          }
        }
        if (projection) {
          const rows = gpuVisionProjection(projection, d, bytes);
          const finite = this.p.finite;
          const group = d.createBindGroup({ layout: finite.getBindGroupLayout(0), entries: [this.finiteParams, this.output, rows.status].map((buffer, binding) => ({ binding, resource: { buffer } })) });
          pass.setPipeline(finite);
          pass.setBindGroup(0, group);
          pass.dispatchWorkgroups(Math.ceil((bytes / 4) / 1024));
        }
        pass?.end();
        enc.copyBufferToBuffer(this.output, 0, projection ? gpuVisionProjection(projection, d, bytes).buffer : this.readback, 0, bytes);
        enc.clearBuffer(this.pixels);
        if (gpuPixels) this.imageGpu.clear(enc);
        profiler?.finish(enc);
        d.queue.submit([enc.finish()]);
        if (projection) return projection;
        const [profile, mapping] = await Promise.allSettled([profiler?.collect(), this.readback.mapAsync(GPUMapMode.READ, 0, bytes)]);
        if (profile.status === "rejected") throw profile.reason;
        if (mapping.status === "rejected") throw mapping.reason;
        if (profiler) this.lastProfile = profile.value;
        const result = new Float32Array(this.readback.getMappedRange(0, bytes).slice(0));
        this.readback.unmap();
        return result;
      }, "running Gemma 4 image encoding");
    } catch (error) {
      projection?.destroy();
      this.lastProfile = null;
      this.reset();
      throw error;
    } finally {
      plan?.pixels?.fill(0);
      clearGemma4VisionImageData(plan?.ownedImage);
      disposeGemma4VisionImage(image);
      this.running = false;
    }
  }

  destroy() {
    this.reset();
    for (const entry of this.weights.tensors.values()) { entry.buf?.destroy(); entry.sbuf?.destroy(); }
  }
}
