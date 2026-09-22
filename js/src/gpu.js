// The transformer trunk on WebGPU: 28 ModernBERT layers and 2 decision-head layers, int8 weights
// widened to f32 inside the matmul tiles. The coordinator (WebAssembly) embeds tokens before and
// scores markers after; everything in between is one command buffer.

const WGSL_COMMON = /* wgsl */ `
struct Globals { T: u32, R: u32, S: u32, _b: u32 }
@group(0) @binding(0) var<uniform> g: Globals;
`;

// Narrow matmuls at short lengths launch too few 64x64 tiles to fill a GPU (a 1024-wide output
// at 64 tokens is 16 workgroups), so K is split across up to 8 workgroups whose partial tiles a
// second pass sums. The shader and the host compute the same split count.
const SPLIT_TARGET = 128;
const splitsWGSL = (target, bm = 64) => /* wgsl */ `
fn mm_splits(T: u32, N: u32, K: u32) -> u32 {
  let tiles = ((N + 63u) / 64u) * ((T + ${bm - 1}u) / ${bm}u);
  if (tiles >= 96u) { return 1u; }
  return max(1u, min(min(8u, (${target}u + tiles - 1u) / tiles), K / 128u));
}
`;

/** Host side of `mm_splits`. */
export function mmSplits(T, N, K, target = SPLIT_TARGET, bm = 64) {
  const tiles = Math.ceil(N / 64) * Math.ceil(T / bm);
  if (tiles >= 96) return 1;
  return Math.max(1, Math.min(8, Math.ceil(target / tiles), Math.floor(K / 128)));
}

/** Rows per thread for an input of T tokens: a short input gets a shorter tile (16 R rows). */
export const rowsPerThread = (T) => (T >= 64 ? 4 : Math.max(1, Math.ceil(T / 16)));

/** Encodes a matmul op (bind groups `group` and `reduce`), splitting K when it helps. */
export function dispatchMatmul(pass, pMatmul, pReduce, op, T, target = SPLIT_TARGET, R = 4) {
  const bm = 16 * R;
  const splits = mmSplits(T, op.N, op.K, target, bm);
  pass.setPipeline(pMatmul);
  pass.setBindGroup(0, op.group);
  pass.dispatchWorkgroups(Math.ceil(op.N / 64), Math.ceil(T / bm), splits);
  if (splits > 1) {
    const n = Math.ceil((T * op.N) / 256);
    pass.setPipeline(pReduce);
    pass.setBindGroup(0, op.reduce);
    pass.dispatchWorkgroups(Math.min(n, 65535), Math.ceil(n / 65535));
  }
}

/** Encodes a matmul with the kernel variant for T. `pipes` comes from `matmulPipelines`. */
export function encodeMatmul(pass, pipes, op, T) {
  const R = rowsPerThread(T);
  dispatchMatmul(pass, pipes.mm[R], pipes.reduce[R], op, T, SPLIT_TARGET, R);
}

/** The matmul bind group layout, explicit so every kernel variant shares one bind group. */
export function matmulLayout(device) {
  const e = (binding, type) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
  return device.createBindGroupLayout({
    label: "matmul",
    entries: [e(0, "uniform"), e(1, "uniform"), e(2, "read-only-storage"), e(3, "read-only-storage"), e(4, "read-only-storage"), e(5, "read-only-storage"), e(6, "storage"), e(7, "storage")],
  });
}

/** The split-K reduce layout, explicit for the same reason. */
export function reduceLayout(device) {
  const e = (binding, type) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
  return device.createBindGroupLayout({ label: "reduce", entries: [e(0, "uniform"), e(1, "uniform"), e(2, "read-only-storage"), e(3, "read-only-storage"), e(4, "storage")] });
}

/**
 * Builds the matmul variants (1 to 4 rows per thread) and their split-K reduces; resolves to
 * { layout, reduceLayout, mm: [, R1..R4], reduce: [, R1..R4] }. With `shader-f16` the tiles live
 * in workgroup memory as f16, which halves the traffic that limits this kernel (1.6-1.75x
 * faster on Apple GPUs); products and sums stay f32, and the rounding (about 3e-4 relative) is
 * far below the int8 weight quantization.
 */
export async function matmulPipelines(device) {
  const layout = matmulLayout(device);
  const rlayout = reduceLayout(device);
  const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const rpl = device.createPipelineLayout({ bindGroupLayouts: [rlayout] });
  const h = device.features.has("shader-f16");
  const mm = [];
  const reduce = [];
  await Promise.all(
    [1, 2, 3, 4].flatMap((R) => [
      pipeline(device, matmulKernel(h, SPLIT_TARGET, R), `matmul_r${R}`, pl).then((x) => (mm[R] = x)),
      pipeline(device, reduceKernel(SPLIT_TARGET, 16 * R), `reduce_r${R}`, rpl).then((x) => (reduce[R] = x)),
    ]),
  );
  return { layout, reduceLayout: rlayout, mm, reduce, f16: h };
}

/** Floats of scratch the split-K partials need at most (tiles < 96, at most 8 splits). */
export const SPLIT_SCRATCH = 8 * 96 * 64 * 64;

// Y[T,N] = X[T,K] . W[N,K]^T (+ bias), W int8 in u32 words with one f32 scale per 32 weights.
// 64x64 output tile per workgroup; thread (x, y) owns rows y + 16i and columns x + 16j, so
// output stores coalesce. Tiles hold one quantization block of K as vec4 rows; every thread
// writes whole vectors (sub-vector writes from several threads race on some GPUs), and the
// weight tile is XOR-swizzled so neighbouring threads read different banks.
/** `h` stores the tiles as f16 (half the workgroup-memory traffic; math stays f32). */
const matmulKernel = (h, target = SPLIT_TARGET, R = 4) => /* wgsl */ `
${h ? "enable f16;" : ""}
${WGSL_COMMON}
struct P { N: u32, K: u32, mode: u32, bias: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> S: array<f32>;
@group(0) @binding(5) var<storage, read> B: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
@group(0) @binding(7) var<storage, read_write> PART: array<f32>;

var<workgroup> xs: array<vec4<${h ? "f16" : "f32"}>, 512>; // [m][k/4], 64 x 8
var<workgroup> ws: array<vec4<${h ? "f16" : "f32"}>, 512>; // [n][(k/4) ^ (n & 7)], 64 x 8

fn TV(v: vec4<f32>) -> ${h ? "vec4<f16>" : "vec4<f32>"} { return ${h ? "vec4<f16>" : "vec4<f32>"}(v); }

fn sx(w: u32) -> vec4<f32> {
  return vec4<f32>(vec4<i32>(bitcast<i32>(w << 24u), bitcast<i32>(w << 16u), bitcast<i32>(w << 8u), bitcast<i32>(w)) >> vec4<u32>(24u));
}

${splitsWGSL(target, 16 * R)}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let T = g.T;
  let N = p.N;
  let K = p.K;
  let nb = K / 32u;
  let m0 = wg.y * ${16 * R}u;
  let n0 = wg.x * 64u;
  // split-K: this workgroup covers blocks kb0..kb1 and, when split, writes a partial tile
  let splits = mm_splits(T, N, K);
  let per = (nb + splits - 1u) / splits;
  let kb0 = wg.z * per;
  let kb1 = min(nb, kb0 + per);
  var acc: array<vec4<f32>, ${R}>;
  // loader roles: ${16 * R} rows of X and 64 of W x 32 k per tile, 4 threads per row, 8 values each
  let lr = li / 4u;
  let lq = li % 4u;
  let xrow = m0 + lr;
  let wrow = n0 + lr;
  let xok = lr < ${16 * R}u && xrow < T;
  let wok = wrow < N;
  let sw = lr & 7u;
  for (var kb = kb0; kb < kb1; kb++) {
    let k0 = kb * 32u;
    if (lr >= ${16 * R}u) {
      // this thread only loads weights
    } else if (xok) {
      let base = (xrow * K + k0) / 4u + lq * 2u;
      xs[lr * 8u + lq * 2u] = TV(X[base]);
      xs[lr * 8u + lq * 2u + 1u] = TV(X[base + 1u]);
    } else {
      xs[lr * 8u + lq * 2u] = TV(vec4<f32>(0.0));
      xs[lr * 8u + lq * 2u + 1u] = TV(vec4<f32>(0.0));
    }
    if (wok) {
      let v = W[(wrow * K + k0) / 16u + lq / 2u];
      let s = S[wrow * nb + kb];
      var w0 = v.x;
      var w1 = v.y;
      if ((lq & 1u) == 1u) { w0 = v.z; w1 = v.w; }
      ws[lr * 8u + ((lq * 2u) ^ sw)] = TV(sx(w0) * s);
      ws[lr * 8u + ((lq * 2u + 1u) ^ sw)] = TV(sx(w1) * s);
    } else {
      ws[lr * 8u + ((lq * 2u) ^ sw)] = TV(vec4<f32>(0.0));
      ws[lr * 8u + ((lq * 2u + 1u) ^ sw)] = TV(vec4<f32>(0.0));
    }
    workgroupBarrier();
    for (var kq = 0u; kq < 8u; kq++) {
      let nx = lid.x & 7u;
      let b0 = vec4<f32>(ws[lid.x * 8u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(lid.x + 16u) * 8u + (kq ^ nx)]);
      let b2 = vec4<f32>(ws[(lid.x + 32u) * 8u + (kq ^ nx)]);
      let b3 = vec4<f32>(ws[(lid.x + 48u) * 8u + (kq ^ nx)]);
      for (var i = 0u; i < ${R}u; i++) {
        let a = vec4<f32>(xs[(lid.y + 16u * i) * 8u + kq]);
        acc[i] += vec4<f32>(dot(a, b0), dot(a, b1), dot(a, b2), dot(a, b3));
      }
    }
    workgroupBarrier();
  }
  for (var j = 0u; j < 4u; j++) {
    let col = n0 + lid.x + 16u * j;
    if (col >= N) { break; }
    var bias = 0.0;
    if (p.bias == 1u) { bias = B[col]; }
    for (var i = 0u; i < ${R}u; i++) {
      let row = m0 + lid.y + 16u * i;
      if (row >= T) { break; }
      let o = row * N + col;
      if (splits > 1u) {
        PART[wg.z * T * N + o] = acc[i][j];
        continue;
      }
      var v = acc[i][j] + bias;
      if (p.mode == 1u) { v += Y[o]; } else if (p.mode == 2u) { v = max(v, 0.0); }
      Y[o] = v;
    }
  }
}
`;
const WGSL_MATMUL = matmulKernel(false);

// Sums split-K partials and applies the matmul epilogue (bias, residual add or ReLU).
const reduceKernel = (target = SPLIT_TARGET, bm = 64) => /* wgsl */ `
${WGSL_COMMON}
struct P { N: u32, K: u32, mode: u32, bias: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> PART: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
${splitsWGSL(target, bm)}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let T = g.T;
  let N = p.N;
  let o = gid.x + gid.y * 65535u * 256u;
  if (o >= T * N) { return; }
  let splits = mm_splits(T, N, p.K);
  var v = 0.0;
  for (var s = 0u; s < splits; s++) { v += PART[s * T * N + o]; }
  if (p.bias == 1u) { v += B[o % N]; }
  if (p.mode == 1u) { v += Y[o]; } else if (p.mode == 2u) { v = max(v, 0.0); }
  Y[o] = v;
}
`;
const WGSL_REDUCE = reduceKernel();


// Row layer norm: one workgroup per token, D = 1024 features.
const WGSL_NORM = /* wgsl */ `
${WGSL_COMMON}
struct P { D: u32, bias: u32, typed: u32, eps: f32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> Wt: array<f32>;
@group(0) @binding(4) var<storage, read> Bs: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;
@group(0) @binding(6) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read> TE: array<f32>;
var<workgroup> red: array<f32, 256>;

fn reduce(v: f32, l: u32) -> f32 {
  red[l] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (l < s) { red[l] += red[l + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let t = wg.x;
  if (t >= g.T) { return; }
  let D = p.D;
  let base = t * D;
  var v: array<f32, 4>;
  var s = 0.0;
  for (var i = 0u; i < 4u; i++) { v[i] = X[base + l + i * 256u]; s += v[i]; }
  let mean = reduce(s, l) / f32(D);
  var q = 0.0;
  for (var i = 0u; i < 4u; i++) { let d = v[i] - mean; q += d * d; }
  let inv = inverseSqrt(reduce(q, l) / f32(D) + p.eps);
  let qt = tok[t].w;
  for (var i = 0u; i < 4u; i++) {
    let c = l + i * 256u;
    var o = (v[i] - mean) * inv * Wt[c];
    if (p.bias == 1u) { o += Bs[c]; }
    if (p.typed == 1u) { o += TE[qt * D + c]; }
    Y[base + c] = o;
  }
}
`;

// Rotary embedding on q and k, in place in the qkv buffer (rotate_half convention).
const WGSL_ROPE = /* wgsl */ `
${WGSL_COMMON}
struct P { heads: u32, stride: u32, table: u32, _a: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read_write> QKV: array<f32>;
@group(0) @binding(3) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> CS: array<vec2<f32>>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) i: u32) {
  let t = wg.x;
  let h = wg.y; // 0..2*heads: q heads then k heads
  if (t >= g.T) { return; }
  let pos = tok[t].z;
  let cs = CS[p.table * 512u * 32u + pos * 32u + i];
  let base = t * p.stride + h * 64u;
  let x1 = QKV[base + i];
  let x2 = QKV[base + i + 32u];
  QKV[base + i] = x1 * cs.x - x2 * cs.y;
  QKV[base + i + 32u] = x2 * cs.x + x1 * cs.y;
}
`;

// Blocked attention: one workgroup per (block of 16 queries of one segment, head). Keys and
// values stream through workgroup memory 16 at a time and serve all 16 queries, with an online
// softmax per query, so K and V are read once per block instead of once per query.
const WGSL_ATTN_BLOCK = /* wgsl */ `
${WGSL_COMMON}
struct P { width: u32, stride: u32, window: u32, _a: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> QKV: array<f32>;
@group(0) @binding(3) var<storage, read> blocks: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> CTX: array<f32>;
var<workgroup> qs: array<f32, 1024>;  // [16 queries][64]
var<workgroup> ks: array<f32, 1024>;  // [16 keys][64]
var<workgroup> vs: array<f32, 1024>;  // [16 keys][64]
var<workgroup> ps: array<f32, 256>;   // [16 queries][16 keys]
var<workgroup> corr: array<f32, 16>;
var<workgroup> info: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let h = wg.y;
  if (wg.x >= g.S) { return; }
  if (t == 0u) { info = blocks[wg.x]; }
  let bi = workgroupUniformLoad(&info);
  let s0 = bi.x;   // segment start (token index)
  let L = bi.y;    // segment length
  let q0 = bi.z;   // first query of the block, within the segment
  let nq = min(16u, L - q0);
  let w = p.window;
  var lo = 0u;
  var hi = L;
  if (w > 0u) {
    if (q0 > w) { lo = q0 - w; }
    hi = min(L, q0 + nq - 1u + w + 1u);
  }
  // queries of the block
  for (var e = t; e < 1024u; e += 64u) {
    let qi = e / 64u;
    let d = e % 64u;
    var v = 0.0;
    if (qi < nq) { v = QKV[(s0 + q0 + qi) * p.stride + h * 64u + d]; }
    qs[e] = v;
  }
  var m = -3.0e38;  // running max, for query t (t < 16)
  var l = 0.0;      // running sum, for query t
  var o: array<f32, 16>; // this thread's dim t, for each query
  for (var i = 0u; i < 16u; i++) { o[i] = 0.0; }
  for (var j0 = lo; j0 < hi; j0 += 16u) {
    workgroupBarrier();
    for (var e = t; e < 1024u; e += 64u) {
      let kj = e / 64u;
      let d = e % 64u;
      var kv = 0.0;
      var vv = 0.0;
      if (j0 + kj < hi) {
        let row = (s0 + j0 + kj) * p.stride + h * 64u + d;
        kv = QKV[row + p.width];
        vv = QKV[row + 2u * p.width];
      }
      ks[e] = kv;
      vs[e] = vv;
    }
    workgroupBarrier();
    // scores: thread t computes 4 of them, query t / 4 against keys (t % 4) * 4 .. + 4
    let qi = t / 4u;
    for (var c = 0u; c < 4u; c++) {
      let kj = (t % 4u) * 4u + c;
      let j = j0 + kj;
      let i = q0 + qi;
      var sc = -3.0e38;
      let inwin = w == 0u || (max(i, j) - min(i, j)) <= w;
      if (qi < nq && j < hi && inwin) {
        var acc = 0.0;
        for (var d = 0u; d < 64u; d++) { acc += qs[qi * 64u + d] * ks[kj * 64u + d]; }
        sc = acc * 0.125;
      }
      ps[qi * 16u + kj] = sc;
    }
    workgroupBarrier();
    // online softmax bookkeeping, one thread per query
    if (t < 16u) {
      var mx = m;
      for (var kj = 0u; kj < 16u; kj++) { mx = max(mx, ps[t * 16u + kj]); }
      let cf = exp(m - mx);
      var sum = 0.0;
      for (var kj = 0u; kj < 16u; kj++) {
        let sc = ps[t * 16u + kj];
        var e = 0.0;
        if (sc > -1.0e38) { e = exp(sc - mx); }
        ps[t * 16u + kj] = e;
        sum += e;
      }
      l = l * cf + sum;
      m = mx;
      corr[t] = cf;
    }
    workgroupBarrier();
    // output: thread t owns dim t for every query of the block
    for (var qi2 = 0u; qi2 < 16u; qi2++) {
      var acc = o[qi2] * corr[qi2];
      for (var kj = 0u; kj < 16u; kj++) { acc += ps[qi2 * 16u + kj] * vs[kj * 64u + t]; }
      o[qi2] = acc;
    }
  }
  // share each query's final sum, then write
  workgroupBarrier();
  if (t < 16u) { corr[t] = l; }
  workgroupBarrier();
  for (var qi2 = 0u; qi2 < nq; qi2++) {
    CTX[(s0 + q0 + qi2) * p.width + h * 64u + t] = o[qi2] / corr[qi2];
  }
}
`;

// GeGLU: A[t, i] = gelu(U[t, i]) * U[t, I + i], exact erf GELU.
const WGSL_GEGLU = /* wgsl */ `
${WGSL_COMMON}
struct P { I: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> U: array<f32>;
@group(0) @binding(3) var<storage, read_write> A: array<f32>;

// Abramowitz and Stegun 7.1.26 is only good to 1.5e-7 near 0; use erf's Taylor series there
fn erf(x: f32) -> f32 {
  let a = abs(x);
  if (a < 0.5) {
    let z = x * x;
    return x * (1.1283791671 + z * (-0.3761263890 + z * (0.1128379167 + z * (-0.0268661706 + z * 0.0052239776))));
  }
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return select(-y, y, x >= 0.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let I = p.I;
  let idx = gid.x + gid.y * 65535u * 256u;
  if (idx >= g.T * I) { return; }
  let t = idx / I;
  let i = idx % I;
  let u = U[t * 2u * I + i];
  let gate = U[t * 2u * I + I + i];
  A[idx] = 0.5 * u * (1.0 + erf(u * 0.70710678118)) * gate;
}
`;

// Copies the rows the scorer needs (markers and segment starts) into a compact buffer.
const WGSL_GATHER = /* wgsl */ `
${WGSL_COMMON}
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> rows: array<u32>;
@group(0) @binding(3) var<storage, read_write> O: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let r = wg.x;
  if (r >= g.R) { return; }
  let src = rows[r];
  O[r * 256u + l] = X[src * 256u + l];
}
`;


function parseSubpackHeader(prefix) {
  const dv = new DataView(prefix.buffer, prefix.byteOffset, prefix.byteLength);
  const n = dv.getUint32(8, true);
  return JSON.parse(new TextDecoder().decode(prefix.subarray(16, 16 + n)));
}

/** Detects a usable GPU. Returns null when WebGPU is missing or the adapter is too small. */
export async function requestDevice() {
  if (typeof navigator === "undefined" || !navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return null;
  const want = {
    maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1 << 30),
    maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1 << 30),
    maxComputeWorkgroupStorageSize: Math.min(adapter.limits.maxComputeWorkgroupStorageSize, 32768),
  };
  // timestamps only feed the optional profiler (`Kevala.load({ profile: true })`)
  const requiredFeatures = ["timestamp-query", "shader-f16"].filter((f) => adapter.features.has(f));
  const device = await adapter.requestDevice({ requiredLimits: want, requiredFeatures });
  const info = adapter.info || {};
  return { device, adapter, name: [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" ") || "WebGPU" };
}

let U;

/** The shader sources, exported for the kernel tests. */
export const SHADERS = { attention_block: WGSL_ATTN_BLOCK, matmul: WGSL_MATMUL, matmul_h: matmulKernel(true), matmulKernel, reduceKernel, reduce: WGSL_REDUCE, norm: WGSL_NORM, rope: WGSL_ROPE, geglu: WGSL_GEGLU, gather: WGSL_GATHER };

/**
 * Trunk weights streamed straight into GPU buffers, one buffer per tensor (and one for q8
 * scales), as the pack goes by.
 */
export class GpuWeights {
  constructor(device, layout) {
    U = GPUBufferUsage;
    this.device = device;
    this.header = parseSubpackHeader(layout.prefix);
    this.tensors = new Map();
    // destination ranges -> tensor parts, for routing streamed fragments
    this.parts = [];
    for (const t of this.header.tensors) {
      const e = { info: t, data: null, scales: null, filled: 0, need: t.size + (t.scales_size || 0), buf: null, sbuf: null, host: null };
      this.tensors.set(t.name, e);
      this.parts.push({ lo: t.offset, hi: t.offset + t.size, e, part: "data" });
      if (t.dtype === "q8") this.parts.push({ lo: t.scales_offset, hi: t.scales_offset + t.scales_size, e, part: "scales" });
    }
    this.parts.sort((a, b) => a.lo - b.lo);
    this.cursor = 0;
    this.keepHost = new Set();
  }

  /** Receives a fragment of the trunk sub-pack. */
  write(dst, bytes) {
    let off = 0;
    while (off < bytes.byteLength) {
      const at = dst + off;
      while (this.cursor < this.parts.length && this.parts[this.cursor].hi <= at) this.cursor++;
      const part = this.parts[this.cursor];
      if (!part || part.lo > at) {
        // padding or the header: skip to the next part
        const next = part ? part.lo : Infinity;
        off += Math.min(bytes.byteLength - off, next - at);
        continue;
      }
      const n = Math.min(bytes.byteLength - off, part.hi - at);
      const e = part.e;
      const key = part.part;
      if (!e[key]) e[key] = new Uint8Array(part.hi - part.lo);
      e[key].set(bytes.subarray(off, off + n), at - part.lo);
      e.filled += n;
      off += n;
      if (e.filled === e.need) this.upload(e);
    }
  }

  upload(e) {
    const mk = (bytes) => {
      const size = Math.max(16, Math.ceil(bytes.byteLength / 16) * 16);
      const b = this.device.createBuffer({ size, usage: U.STORAGE | U.COPY_DST });
      this.device.queue.writeBuffer(b, 0, bytes.buffer, bytes.byteOffset, bytes.byteLength & ~3);
      return b;
    };
    e.buf = mk(e.data);
    if (e.scales) e.sbuf = mk(e.scales);
    // small f32 tensors some kernels combine on the host (Kev's gate projections)
    if (this.keepHost.has(e.info.name)) e.host = new Float32Array(e.data.buffer, e.data.byteOffset, e.data.byteLength / 4);
    e.data = e.scales = null;
  }

  get(name) {
    const e = this.tensors.get(name);
    if (!e) throw new Error(`GPU trunk has no tensor ${name}`);
    return e;
  }

  missing() {
    return [...this.tensors.values()].filter((e) => !e.buf).map((e) => e.info.name);
  }
}

/**
 * Times every dispatch in its own compute pass and totals the milliseconds by label. Slower than
 * a normal pass (one pass per dispatch), so only for finding where the time goes.
 */
export class Profiler {
  constructor(device, max = 2048) {
    this.device = device;
    this.max = max;
    this.qs = device.createQuerySet({ type: "timestamp", count: 2 * max });
    this.resolved = device.createBuffer({ size: 16 * max, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this.read = device.createBuffer({ size: 16 * max, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.labels = [];
  }

  pass(enc, label) {
    const i = this.labels.length;
    if (i >= this.max) return enc.beginComputePass();
    this.labels.push(label);
    return enc.beginComputePass({ timestampWrites: { querySet: this.qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
  }

  finish(enc) {
    const n = this.labels.length;
    if (!n) return;
    enc.resolveQuerySet(this.qs, 0, 2 * n, this.resolved, 0);
    enc.copyBufferToBuffer(this.resolved, 0, this.read, 0, 16 * n);
  }

  async collect() {
    const n = this.labels.length;
    const out = {};
    if (n) {
      await this.read.mapAsync(GPUMapMode.READ, 0, 16 * n);
      const t = new BigInt64Array(this.read.getMappedRange(0, 16 * n));
      this.labels.forEach((l, i) => (out[l] = (out[l] || 0) + Number(t[2 * i + 1] - t[2 * i]) / 1e6));
      this.read.unmap();
    }
    this.labels = [];
    for (const k of Object.keys(out)) out[k] = Math.round(out[k] * 1000) / 1000;
    return out;
  }
}

/**
 * Passes over more than this many tokens are submitted in several command buffers, waiting for
 * the queue to drain in between: one long submit starves the page's compositor (frames stall for
 * hundreds of milliseconds while the GPU is busy).
 */
export const YIELD_TOKENS = 256;
export const YIELD_CHUNKS = 6;

/**
 * How long passes are submitted: "await" waits for each chunk to finish before sending the next,
 * "split" sends the chunks as separate command buffers back to back, "none" sends one.
 */
export const SUBMIT = { mode: "await" };

/** Ends a chunk of a long pass according to SUBMIT.mode. */
export async function endChunk(device, enc) {
  if (SUBMIT.mode === "none") return enc;
  device.queue.submit([enc.finish()]);
  if (SUBMIT.mode === "await") await device.queue.onSubmittedWorkDone();
  return device.createCommandEncoder();
}

/** Splits `ops` into `n` contiguous chunks. */
export function chunks(ops, n) {
  const out = [];
  const size = Math.ceil(ops.length / n);
  for (let i = 0; i < ops.length; i += size) out.push(ops.slice(i, i + size));
  return out;
}

/** Compiles a compute pipeline, turning WGSL errors into exceptions. */
export async function pipeline(device, code, label, layout = "auto") {
  const m = device.createShaderModule({ code, label });
  const info = await m.getCompilationInfo?.();
  const errs = (info?.messages || []).filter((x) => x.type === "error");
  if (errs.length) throw new Error(`${label}: ${errs.map((x) => `${x.lineNum}:${x.linePos} ${x.message}`).join("; ")}`);
  return device.createComputePipelineAsync({ layout, compute: { module: m, entryPoint: "main" }, label });
}

export class GpuTrunk {
  /**
   * `layout` is the single-shard trunk layout from `kevala_layouts`; tensors arrive through
   * `write(dst, bytes)` as the pack streams by.
   */
  constructor(gpu, layout, cfg) {
    U = GPUBufferUsage;
    this.device = gpu.device;
    this.name = gpu.name;
    this.cfg = cfg;
    this.weights = new GpuWeights(gpu.device, layout);
    this.tensors = this.weights.tensors;
    this.capacity = 0;
  }

  write(dst, bytes) {
    this.weights.write(dst, bytes);
  }

  missing() {
    return this.weights.missing();
  }

  /** Builds pipelines and static buffers once all weights are uploaded. */
  async init(finalNorm, typeEmb) {
    const miss = this.missing();
    if (miss.length) throw new Error(`GPU trunk is missing ${miss.length} tensors (${miss[0]}...)`);
    const d = this.device;
    const cfg = this.cfg;
    const pipe = (code, label) => pipeline(d, code, label);
    [this.mm, this.pNorm, this.pRope, this.pAttn, this.pGeglu, this.pGather] = await Promise.all([
      matmulPipelines(d),
      pipe(WGSL_NORM, "norm"),
      pipe(WGSL_ROPE, "rope"),
      pipe(WGSL_ATTN_BLOCK, "attention"),
      pipe(WGSL_GEGLU, "geglu"),
      pipe(WGSL_GATHER, "gather"),
    ]);
    this.globals = d.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    const f32buf = (arr, usage = U.STORAGE) => {
      const b = d.createBuffer({ size: Math.max(16, arr.byteLength), usage: usage | U.COPY_DST });
      d.queue.writeBuffer(b, 0, arr);
      return b;
    };
    this.finalNorm = f32buf(finalNorm);
    this.typeEmb = f32buf(typeEmb);
    this.zeros = f32buf(new Float32Array(4096));
    // rotary tables: [table][pos][32] of (cos, sin), same math as the CPU path
    const maxPos = 512;
    const cs = new Float32Array(2 * maxPos * 32 * 2);
    [cfg.rope_global, cfg.rope_local].forEach((theta, ti) => {
      for (let i = 0; i < 32; i++) {
        const inv = Math.fround(1 / Math.fround(Math.pow(theta, (2 * i) / 64)));
        for (let pos = 0; pos < maxPos; pos++) {
          const a = Math.fround(pos * inv);
          const o = ((ti * maxPos + pos) * 32 + i) * 2;
          cs[o] = Math.cos(a);
          cs[o + 1] = Math.sin(a);
        }
      }
    });
    this.rope = f32buf(cs);
    this.uniforms = [];
    this.ensure(64);
  }

  uniform(values) {
    const b = this.device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    const a = new ArrayBuffer(16);
    const u = new Uint32Array(a);
    const f = new Float32Array(a);
    values.forEach((v, i) => {
      if (typeof v === "object") f[i] = v.f;
      else u[i] = v;
    });
    this.device.queue.writeBuffer(b, 0, a);
    this.uniforms.push(b);
    return b;
  }

  /** Grows activation buffers to hold `tokens` and rebuilds the bind groups. */
  ensure(tokens) {
    if (tokens <= this.capacity) return;
    const cap = Math.max(64, 1 << Math.ceil(Math.log2(tokens)));
    const d = this.device;
    for (const b of [this.x, this.h, this.qkv, this.ctx, this.up, this.act, this.tok, this.blocks, this.rowsBuf, this.gathered, this.readback, this.part]) b?.destroy();
    for (const b of this.uniforms) b.destroy();
    this.uniforms = [];
    const D = this.cfg.hidden;
    const buf = (floats, extra = 0) => d.createBuffer({ size: floats * 4, usage: U.STORAGE | extra });
    this.x = buf(cap * D, U.COPY_DST | U.COPY_SRC);
    this.h = buf(cap * D, U.COPY_SRC);
    this.qkv = buf(cap * 3 * D);
    this.ctx = buf(cap * D);
    this.up = buf(cap * 2 * this.cfg.intermediate);
    this.act = buf(cap * Math.max(this.cfg.intermediate, this.cfg.head_ff));
    this.tok = buf(cap * 4, U.COPY_DST);
    // attention query blocks: at most one per token
    this.blocks = buf(cap * 4, U.COPY_DST);
    this.rowsBuf = buf(cap, U.COPY_DST);
    this.gathered = buf(cap * D, U.COPY_SRC);
    this.readback = d.createBuffer({ size: cap * D * 4, usage: U.MAP_READ | U.COPY_DST });
    this.part = buf(SPLIT_SCRATCH);
    this.capacity = cap;
    this.buildGroups();
  }

  buildGroups() {
    const d = this.device;
    const cfg = this.cfg;
    const D = cfg.hidden;
    const T = (n) => this.tensors.get(n);
    const bg = (pipeline, entries) =>
      d.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: entries.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const mm = (w, bias, input, output, mode) => {
      const e = T(w);
      const [N, K] = e.info.shape;
      const b = bias ? T(bias).buf : this.zeros;
      const u = this.uniform([N, K, mode, bias ? 1 : 0]);
      return {
        kind: "mm",
        label: "mm." + w.split(".").pop(),
        N,
        K,
        group: d.createBindGroup({ layout: this.mm.layout, entries: [this.globals, u, input, e.buf, e.sbuf, b, output, this.part].map((b, i) => ({ binding: i, resource: { buffer: b } })) }),
        reduce: d.createBindGroup({ layout: this.mm.reduceLayout, entries: [this.globals, u, this.part, b, output].map((b, i) => ({ binding: i, resource: { buffer: b } })) }),
      };
    };
    const norm = (w, b, input, output, typed = false) => ({
      kind: "norm",
      group: bg(this.pNorm, [this.globals, this.uniform([D, b ? 1 : 0, typed ? 1 : 0, { f: typed ? cfg.norm_eps : b ? cfg.head_norm_eps : cfg.norm_eps }]), input, w, b || this.zeros, output, this.tok, this.typeEmb]),
    });
    const ops = [];
    for (let i = 0; i < cfg.layers; i++) {
      const global = i % cfg.global_every === 0;
      const n = (s) => `enc.${i}.${s}`;
      // layer 0 has no attention norm: its input is the normalized embedding itself
      const attnIn = i === 0 ? this.x : this.h;
      if (i > 0) ops.push(norm(T(n("attn_norm")).buf, null, this.x, this.h));
      ops.push(mm(n("wqkv"), null, attnIn, this.qkv, 0));
      ops.push({ kind: "rope", group: bg(this.pRope, [this.globals, this.uniform([cfg.heads, 3 * D, global ? 0 : 1, 0]), this.qkv, this.tok, this.rope]) });
      ops.push({ kind: "attn", group: bg(this.pAttn, [this.globals, this.uniform([D, 3 * D, global ? 0 : cfg.window, 0]), this.qkv, this.blocks, this.ctx]) });
      ops.push(mm(n("wo"), null, this.ctx, this.x, 1));
      ops.push(norm(T(n("mlp_norm")).buf, null, this.x, this.h));
      ops.push(mm(n("wi"), null, this.h, this.up, 0));
      ops.push({ kind: "geglu", I: cfg.intermediate, group: bg(this.pGeglu, [this.globals, this.uniform([cfg.intermediate, 0, 0, 0]), this.up, this.act]) });
      ops.push(mm(n("wo2"), null, this.act, this.x, 1));
    }
    // bridge: final encoder norm plus the question-type embedding, in place
    ops.push(norm(this.finalNorm, null, this.x, this.h, true));
    ops.push({ kind: "copy" });
    for (let i = 0; i < cfg.head_layers; i++) {
      const n = (s) => `head.${i}.${s}`;
      ops.push(norm(T(n("norm1.w")).buf, T(n("norm1.b")).buf, this.x, this.h));
      ops.push(mm(n("in_proj"), n("in_proj.b"), this.h, this.qkv, 0));
      ops.push({ kind: "attn", group: bg(this.pAttn, [this.globals, this.uniform([D, 3 * D, 0, 0]), this.qkv, this.blocks, this.ctx]) });
      ops.push(mm(n("out_proj"), n("out_proj.b"), this.ctx, this.x, 1));
      ops.push(norm(T(n("norm2.w")).buf, T(n("norm2.b")).buf, this.x, this.h));
      ops.push(mm(n("lin1"), n("lin1.b"), this.h, this.act, 2));
      ops.push(mm(n("lin2"), n("lin2.b"), this.act, this.x, 1));
    }
    ops.push({ kind: "gather", group: bg(this.pGather, [this.globals, this.x, this.rowsBuf, this.gathered]) });
    this.ops = ops;
  }

  /**
   * Runs the trunk. `x` is the embedded batch (`tokens * hidden` f32), `segs` the segment table,
   * `rows` the token indices whose final states the scorer needs. Resolves to those rows.
   */
  async forward(x, segs, rows) {
    const d = this.device;
    const D = this.cfg.hidden;
    const tokens = x.length / D;
    this.ensure(Math.max(tokens, rows.length));
    const tok = new Uint32Array(tokens * 4);
    for (const s of segs) {
      for (let p = 0; p < s.len; p++) {
        const t = s.start + p;
        tok.set([s.start, s.len, p, s.qtype], t * 4);
      }
    }
    const q = d.queue;
    // the first passes run inside error scopes so a validation failure surfaces instead of
    // silently producing garbage
    // every pass runs in error scopes: a rejected command buffer must be an error, not the
    // previous pass's rows read back
    const checking = true;
    if (checking) {
      d.pushErrorScope("validation");
      d.pushErrorScope("out-of-memory");
    }
    const blocks = [];
    for (const sg of segs) for (let q0 = 0; q0 < sg.len; q0 += 16) blocks.push(sg.start, sg.len, q0, 0);
    const nblocks = blocks.length / 4;
    q.writeBuffer(this.globals, 0, new Uint32Array([tokens, rows.length, nblocks, 0]));
    q.writeBuffer(this.blocks, 0, new Uint32Array(blocks));
    q.writeBuffer(this.x, 0, x);
    q.writeBuffer(this.tok, 0, tok);
    q.writeBuffer(this.rowsBuf, 0, new Uint32Array(rows));
    const prof = this.profiler;
    const heads = this.cfg.heads;
    const groups = tokens > YIELD_TOKENS && !prof ? chunks(this.ops, YIELD_CHUNKS) : [this.ops];
    let enc = d.createCommandEncoder();
    for (let gi = 0; gi < groups.length; gi++) {
    // let the page render between chunks of a long pass
    if (gi) enc = await endChunk(d, enc);
    let pass = prof ? null : enc.beginComputePass();
    for (const op of groups[gi]) {
      if (op.kind === "copy") {
        pass?.end();
        enc.copyBufferToBuffer(this.h, 0, this.x, 0, tokens * D * 4);
        pass = prof ? null : enc.beginComputePass();
        continue;
      }
      if (prof) pass = prof.pass(enc, op.label || op.kind);
      pass.setBindGroup(0, op.group);
      switch (op.kind) {
        case "mm":
          encodeMatmul(pass, this.mm, op, tokens);
          break;
        case "norm":
          pass.setPipeline(this.pNorm);
          pass.dispatchWorkgroups(tokens);
          break;
        case "rope":
          pass.setPipeline(this.pRope);
          pass.dispatchWorkgroups(tokens, 2 * heads);
          break;
        case "attn":
          pass.setPipeline(this.pAttn);
          pass.dispatchWorkgroups(nblocks, heads);
          break;
        case "geglu": {
          pass.setPipeline(this.pGeglu);
          const n = Math.ceil((tokens * op.I) / 256);
          pass.dispatchWorkgroups(Math.min(n, 65535), Math.ceil(n / 65535));
          break;
        }
        case "gather":
          pass.setPipeline(this.pGather);
          pass.dispatchWorkgroups(rows.length);
          break;
      }
      if (prof) {
        pass.end();
        pass = null;
      }
    }
    pass?.end();
    }
    prof?.finish(enc);
    const bytes = rows.length * D * 4;
    enc.copyBufferToBuffer(this.gathered, 0, this.readback, 0, bytes);
    q.submit([enc.finish()]);
    if (prof) this.lastProfile = await prof.collect();
    if (checking) {
      const [oom, invalid] = [await d.popErrorScope(), await d.popErrorScope()];
      if (oom || invalid) throw new Error(`WebGPU: ${(oom || invalid).message}`);
    }
    await this.readback.mapAsync(GPUMapMode.READ, 0, bytes);
    const out = new Float32Array(this.readback.getMappedRange(0, bytes).slice(0));
    this.readback.unmap();
    return out;
  }
}
