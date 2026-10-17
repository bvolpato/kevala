// DEV-only triple BF16 vision Q/K/V path; registered FP32 arithmetic and split order.
//
// A workgroup of 32 x 8 threads covers 56 rows and 64 columns, retaining a virtual
// 64-row grid and split rule. Each thread owns seven rows and two columns.
// The K32 loaders and shared weight swizzle are identical to the 16 x 16 path.
// Laya trunk: the per-pass globals every kernel reads.
struct Globals { T: u32, R: u32, S: u32, _b: u32 }
@group(0) @binding(0) var<uniform> g: Globals;

struct P { N: u32, K: u32, mode: u32, bias: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> WQ: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> WK: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> WV: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read_write> PART: array<f32>;

var<workgroup> xs: array<vec4<f32>, 512>; // [m][k/4]
var<workgroup> wsQ: array<vec2<u32>, 512>;
var<workgroup> wsK: array<vec2<u32>, 512>;
var<workgroup> wsV: array<vec2<u32>, 512>; // [n][(k/4) ^ (n & 7)], swizzled

fn bf16(w: vec2<u32>) -> vec4<f32> {
  return bitcast<vec4<f32>>(vec4<u32>(w.x << 16u, w.x & 0xffff0000u, w.y << 16u, w.y & 0xffff0000u));
}

fn qkv_store(value: f32, row: u32, col: u32, T: u32, N: u32, split: u32, splits: u32, projection: u32) {
  if (row >= T) { return; }
  PART[(projection * splits + split) * T * N + row * N + col] = value;
}

// Split-K: narrow matmuls at short lengths launch too few tiles to fill a GPU (a 1024-wide output
// at 64 tokens is 16 workgroups), so K is split across up to 8 workgroups whose partial tiles a
// second pass sums. The host computes the same count (runtime `mm_splits` twin in gpu.js).
fn mm_splits(T: u32, N: u32, K: u32) -> u32 {
  let tiles = ((N + 63u) / 64u) * ((T + 63u) / 64u);
  if (tiles >= (256u * 3u + 3u) / 4u) { return 1u; }
  return max(1u, min(min(8u, (256u + tiles - 1u) / tiles), K / 128u));
}

@compute @workgroup_size(32, 8)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let T = g.T;
  let N = p.N;
  let K = p.K;
  let nb = K / 32u;
  let m0 = wg.y * 56u;
  let n0 = wg.x * 64u;
  // split-K: this workgroup covers blocks kb0..kb1 and, when split, writes a partial tile
  let splits = mm_splits(T, N, K);
  let per = (nb + splits - 1u) / splits;
  let kb0 = wg.z * per;
  let kb1 = min(nb, kb0 + per);
  var qacc0 = vec2<f32>(0.0);
  var kacc0 = vec2<f32>(0.0);
  var vacc0 = vec2<f32>(0.0);
  var qacc1 = vec2<f32>(0.0);
  var kacc1 = vec2<f32>(0.0);
  var vacc1 = vec2<f32>(0.0);
  var qacc2 = vec2<f32>(0.0);
  var kacc2 = vec2<f32>(0.0);
  var vacc2 = vec2<f32>(0.0);
  var qacc3 = vec2<f32>(0.0);
  var kacc3 = vec2<f32>(0.0);
  var vacc3 = vec2<f32>(0.0);
  var qacc4 = vec2<f32>(0.0);
  var kacc4 = vec2<f32>(0.0);
  var vacc4 = vec2<f32>(0.0);
  var qacc5 = vec2<f32>(0.0);
  var kacc5 = vec2<f32>(0.0);
  var vacc5 = vec2<f32>(0.0);
  var qacc6 = vec2<f32>(0.0);
  var kacc6 = vec2<f32>(0.0);
  var vacc6 = vec2<f32>(0.0);

  // loaders: 4 threads per row, 8 values each; X rows 0..56, W rows lr + 64 g
  let lr = li / 4u;
  let lq = li % 4u;
  let xrow = m0 + lr;
  let sw = (lr >> 1u) & 7u;
  var xv0 = vec4<f32>(0.0);
  var xv1 = vec4<f32>(0.0);
  var wqv: array<vec2<u32>, 2>;
  var wkv: array<vec2<u32>, 2>;
  var wvv: array<vec2<u32>, 2>;
  for (var kb = kb0; kb < kb1; kb++) {
    xv0 = vec4<f32>(0.0);
    xv1 = vec4<f32>(0.0);
    if (lr < 56u && xrow < T) {
      let base = (xrow * K + kb * 32u) / 4u + lq * 2u;
      xv0 = X[base];
      xv1 = X[base + 1u];
    }
    for (var gg = 0u; gg < 1u; gg++) {
      let wrow = n0 + lr + 64u * gg;
      wqv[2u * gg] = vec2<u32>(0u);
      wqv[2u * gg + 1u] = vec2<u32>(0u);
      wkv[2u * gg] = vec2<u32>(0u);
      wkv[2u * gg + 1u] = vec2<u32>(0u);
      wvv[2u * gg] = vec2<u32>(0u);
      wvv[2u * gg + 1u] = vec2<u32>(0u);
      if (wrow < N) {
        let wi = (wrow * K + kb * 32u) / 8u + lq;
        let q = WQ[wi];
        let k = WK[wi];
        let v = WV[wi];
        wqv[2u * gg] = q.xy;
        wqv[2u * gg + 1u] = q.zw;
        wkv[2u * gg] = k.xy;
        wkv[2u * gg + 1u] = k.zw;
        wvv[2u * gg] = v.xy;
        wvv[2u * gg + 1u] = v.zw;
      }
    }
    if (lr < 56u) {
      xs[lr * 8u + lq * 2u] = vec4<f32>(xv0);
      xs[lr * 8u + lq * 2u + 1u] = vec4<f32>(xv1);
    }
    for (var gg = 0u; gg < 1u; gg++) {
      let r = lr + 64u * gg;
      wsQ[r * 8u + ((lq * 2u) ^ sw)] = wqv[2u * gg];
      wsQ[r * 8u + ((lq * 2u + 1u) ^ sw)] = wqv[2u * gg + 1u];
      wsK[r * 8u + ((lq * 2u) ^ sw)] = wkv[2u * gg];
      wsK[r * 8u + ((lq * 2u + 1u) ^ sw)] = wkv[2u * gg + 1u];
      wsV[r * 8u + ((lq * 2u) ^ sw)] = wvv[2u * gg];
      wsV[r * 8u + ((lq * 2u + 1u) ^ sw)] = wvv[2u * gg + 1u];
    }
    workgroupBarrier();
    let nx = (lid.x >> 1u) & 7u;
    // Keep the eight K/4 steps explicit, reusing two weight vectors across eight rows.
    let c = lid.x;
    {
      let kq = 0u;
      let b0q = bf16(wsQ[c * 8u + (kq ^ nx)]);
      let b0k = bf16(wsK[c * 8u + (kq ^ nx)]);
      let b0v = bf16(wsV[c * 8u + (kq ^ nx)]);
      let b1q = bf16(wsQ[(c + 32u) * 8u + (kq ^ nx)]);
      let b1k = bf16(wsK[(c + 32u) * 8u + (kq ^ nx)]);
      let b1v = bf16(wsV[(c + 32u) * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      qacc0 += vec2<f32>(dot(a0, b0q), dot(a0, b1q));
      kacc0 += vec2<f32>(dot(a0, b0k), dot(a0, b1k));
      vacc0 += vec2<f32>(dot(a0, b0v), dot(a0, b1v));
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      qacc1 += vec2<f32>(dot(a1, b0q), dot(a1, b1q));
      kacc1 += vec2<f32>(dot(a1, b0k), dot(a1, b1k));
      vacc1 += vec2<f32>(dot(a1, b0v), dot(a1, b1v));
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      qacc2 += vec2<f32>(dot(a2, b0q), dot(a2, b1q));
      kacc2 += vec2<f32>(dot(a2, b0k), dot(a2, b1k));
      vacc2 += vec2<f32>(dot(a2, b0v), dot(a2, b1v));
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      qacc3 += vec2<f32>(dot(a3, b0q), dot(a3, b1q));
      kacc3 += vec2<f32>(dot(a3, b0k), dot(a3, b1k));
      vacc3 += vec2<f32>(dot(a3, b0v), dot(a3, b1v));
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      qacc4 += vec2<f32>(dot(a4, b0q), dot(a4, b1q));
      kacc4 += vec2<f32>(dot(a4, b0k), dot(a4, b1k));
      vacc4 += vec2<f32>(dot(a4, b0v), dot(a4, b1v));
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      qacc5 += vec2<f32>(dot(a5, b0q), dot(a5, b1q));
      kacc5 += vec2<f32>(dot(a5, b0k), dot(a5, b1k));
      vacc5 += vec2<f32>(dot(a5, b0v), dot(a5, b1v));
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      qacc6 += vec2<f32>(dot(a6, b0q), dot(a6, b1q));
      kacc6 += vec2<f32>(dot(a6, b0k), dot(a6, b1k));
      vacc6 += vec2<f32>(dot(a6, b0v), dot(a6, b1v));
    }
    {
      let kq = 1u;
      let b0q = bf16(wsQ[c * 8u + (kq ^ nx)]);
      let b0k = bf16(wsK[c * 8u + (kq ^ nx)]);
      let b0v = bf16(wsV[c * 8u + (kq ^ nx)]);
      let b1q = bf16(wsQ[(c + 32u) * 8u + (kq ^ nx)]);
      let b1k = bf16(wsK[(c + 32u) * 8u + (kq ^ nx)]);
      let b1v = bf16(wsV[(c + 32u) * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      qacc0 += vec2<f32>(dot(a0, b0q), dot(a0, b1q));
      kacc0 += vec2<f32>(dot(a0, b0k), dot(a0, b1k));
      vacc0 += vec2<f32>(dot(a0, b0v), dot(a0, b1v));
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      qacc1 += vec2<f32>(dot(a1, b0q), dot(a1, b1q));
      kacc1 += vec2<f32>(dot(a1, b0k), dot(a1, b1k));
      vacc1 += vec2<f32>(dot(a1, b0v), dot(a1, b1v));
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      qacc2 += vec2<f32>(dot(a2, b0q), dot(a2, b1q));
      kacc2 += vec2<f32>(dot(a2, b0k), dot(a2, b1k));
      vacc2 += vec2<f32>(dot(a2, b0v), dot(a2, b1v));
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      qacc3 += vec2<f32>(dot(a3, b0q), dot(a3, b1q));
      kacc3 += vec2<f32>(dot(a3, b0k), dot(a3, b1k));
      vacc3 += vec2<f32>(dot(a3, b0v), dot(a3, b1v));
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      qacc4 += vec2<f32>(dot(a4, b0q), dot(a4, b1q));
      kacc4 += vec2<f32>(dot(a4, b0k), dot(a4, b1k));
      vacc4 += vec2<f32>(dot(a4, b0v), dot(a4, b1v));
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      qacc5 += vec2<f32>(dot(a5, b0q), dot(a5, b1q));
      kacc5 += vec2<f32>(dot(a5, b0k), dot(a5, b1k));
      vacc5 += vec2<f32>(dot(a5, b0v), dot(a5, b1v));
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      qacc6 += vec2<f32>(dot(a6, b0q), dot(a6, b1q));
      kacc6 += vec2<f32>(dot(a6, b0k), dot(a6, b1k));
      vacc6 += vec2<f32>(dot(a6, b0v), dot(a6, b1v));
    }
    {
      let kq = 2u;
      let b0q = bf16(wsQ[c * 8u + (kq ^ nx)]);
      let b0k = bf16(wsK[c * 8u + (kq ^ nx)]);
      let b0v = bf16(wsV[c * 8u + (kq ^ nx)]);
      let b1q = bf16(wsQ[(c + 32u) * 8u + (kq ^ nx)]);
      let b1k = bf16(wsK[(c + 32u) * 8u + (kq ^ nx)]);
      let b1v = bf16(wsV[(c + 32u) * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      qacc0 += vec2<f32>(dot(a0, b0q), dot(a0, b1q));
      kacc0 += vec2<f32>(dot(a0, b0k), dot(a0, b1k));
      vacc0 += vec2<f32>(dot(a0, b0v), dot(a0, b1v));
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      qacc1 += vec2<f32>(dot(a1, b0q), dot(a1, b1q));
      kacc1 += vec2<f32>(dot(a1, b0k), dot(a1, b1k));
      vacc1 += vec2<f32>(dot(a1, b0v), dot(a1, b1v));
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      qacc2 += vec2<f32>(dot(a2, b0q), dot(a2, b1q));
      kacc2 += vec2<f32>(dot(a2, b0k), dot(a2, b1k));
      vacc2 += vec2<f32>(dot(a2, b0v), dot(a2, b1v));
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      qacc3 += vec2<f32>(dot(a3, b0q), dot(a3, b1q));
      kacc3 += vec2<f32>(dot(a3, b0k), dot(a3, b1k));
      vacc3 += vec2<f32>(dot(a3, b0v), dot(a3, b1v));
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      qacc4 += vec2<f32>(dot(a4, b0q), dot(a4, b1q));
      kacc4 += vec2<f32>(dot(a4, b0k), dot(a4, b1k));
      vacc4 += vec2<f32>(dot(a4, b0v), dot(a4, b1v));
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      qacc5 += vec2<f32>(dot(a5, b0q), dot(a5, b1q));
      kacc5 += vec2<f32>(dot(a5, b0k), dot(a5, b1k));
      vacc5 += vec2<f32>(dot(a5, b0v), dot(a5, b1v));
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      qacc6 += vec2<f32>(dot(a6, b0q), dot(a6, b1q));
      kacc6 += vec2<f32>(dot(a6, b0k), dot(a6, b1k));
      vacc6 += vec2<f32>(dot(a6, b0v), dot(a6, b1v));
    }
    {
      let kq = 3u;
      let b0q = bf16(wsQ[c * 8u + (kq ^ nx)]);
      let b0k = bf16(wsK[c * 8u + (kq ^ nx)]);
      let b0v = bf16(wsV[c * 8u + (kq ^ nx)]);
      let b1q = bf16(wsQ[(c + 32u) * 8u + (kq ^ nx)]);
      let b1k = bf16(wsK[(c + 32u) * 8u + (kq ^ nx)]);
      let b1v = bf16(wsV[(c + 32u) * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      qacc0 += vec2<f32>(dot(a0, b0q), dot(a0, b1q));
      kacc0 += vec2<f32>(dot(a0, b0k), dot(a0, b1k));
      vacc0 += vec2<f32>(dot(a0, b0v), dot(a0, b1v));
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      qacc1 += vec2<f32>(dot(a1, b0q), dot(a1, b1q));
      kacc1 += vec2<f32>(dot(a1, b0k), dot(a1, b1k));
      vacc1 += vec2<f32>(dot(a1, b0v), dot(a1, b1v));
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      qacc2 += vec2<f32>(dot(a2, b0q), dot(a2, b1q));
      kacc2 += vec2<f32>(dot(a2, b0k), dot(a2, b1k));
      vacc2 += vec2<f32>(dot(a2, b0v), dot(a2, b1v));
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      qacc3 += vec2<f32>(dot(a3, b0q), dot(a3, b1q));
      kacc3 += vec2<f32>(dot(a3, b0k), dot(a3, b1k));
      vacc3 += vec2<f32>(dot(a3, b0v), dot(a3, b1v));
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      qacc4 += vec2<f32>(dot(a4, b0q), dot(a4, b1q));
      kacc4 += vec2<f32>(dot(a4, b0k), dot(a4, b1k));
      vacc4 += vec2<f32>(dot(a4, b0v), dot(a4, b1v));
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      qacc5 += vec2<f32>(dot(a5, b0q), dot(a5, b1q));
      kacc5 += vec2<f32>(dot(a5, b0k), dot(a5, b1k));
      vacc5 += vec2<f32>(dot(a5, b0v), dot(a5, b1v));
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      qacc6 += vec2<f32>(dot(a6, b0q), dot(a6, b1q));
      kacc6 += vec2<f32>(dot(a6, b0k), dot(a6, b1k));
      vacc6 += vec2<f32>(dot(a6, b0v), dot(a6, b1v));
    }
    {
      let kq = 4u;
      let b0q = bf16(wsQ[c * 8u + (kq ^ nx)]);
      let b0k = bf16(wsK[c * 8u + (kq ^ nx)]);
      let b0v = bf16(wsV[c * 8u + (kq ^ nx)]);
      let b1q = bf16(wsQ[(c + 32u) * 8u + (kq ^ nx)]);
      let b1k = bf16(wsK[(c + 32u) * 8u + (kq ^ nx)]);
      let b1v = bf16(wsV[(c + 32u) * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      qacc0 += vec2<f32>(dot(a0, b0q), dot(a0, b1q));
      kacc0 += vec2<f32>(dot(a0, b0k), dot(a0, b1k));
      vacc0 += vec2<f32>(dot(a0, b0v), dot(a0, b1v));
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      qacc1 += vec2<f32>(dot(a1, b0q), dot(a1, b1q));
      kacc1 += vec2<f32>(dot(a1, b0k), dot(a1, b1k));
      vacc1 += vec2<f32>(dot(a1, b0v), dot(a1, b1v));
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      qacc2 += vec2<f32>(dot(a2, b0q), dot(a2, b1q));
      kacc2 += vec2<f32>(dot(a2, b0k), dot(a2, b1k));
      vacc2 += vec2<f32>(dot(a2, b0v), dot(a2, b1v));
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      qacc3 += vec2<f32>(dot(a3, b0q), dot(a3, b1q));
      kacc3 += vec2<f32>(dot(a3, b0k), dot(a3, b1k));
      vacc3 += vec2<f32>(dot(a3, b0v), dot(a3, b1v));
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      qacc4 += vec2<f32>(dot(a4, b0q), dot(a4, b1q));
      kacc4 += vec2<f32>(dot(a4, b0k), dot(a4, b1k));
      vacc4 += vec2<f32>(dot(a4, b0v), dot(a4, b1v));
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      qacc5 += vec2<f32>(dot(a5, b0q), dot(a5, b1q));
      kacc5 += vec2<f32>(dot(a5, b0k), dot(a5, b1k));
      vacc5 += vec2<f32>(dot(a5, b0v), dot(a5, b1v));
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      qacc6 += vec2<f32>(dot(a6, b0q), dot(a6, b1q));
      kacc6 += vec2<f32>(dot(a6, b0k), dot(a6, b1k));
      vacc6 += vec2<f32>(dot(a6, b0v), dot(a6, b1v));
    }
    {
      let kq = 5u;
      let b0q = bf16(wsQ[c * 8u + (kq ^ nx)]);
      let b0k = bf16(wsK[c * 8u + (kq ^ nx)]);
      let b0v = bf16(wsV[c * 8u + (kq ^ nx)]);
      let b1q = bf16(wsQ[(c + 32u) * 8u + (kq ^ nx)]);
      let b1k = bf16(wsK[(c + 32u) * 8u + (kq ^ nx)]);
      let b1v = bf16(wsV[(c + 32u) * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      qacc0 += vec2<f32>(dot(a0, b0q), dot(a0, b1q));
      kacc0 += vec2<f32>(dot(a0, b0k), dot(a0, b1k));
      vacc0 += vec2<f32>(dot(a0, b0v), dot(a0, b1v));
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      qacc1 += vec2<f32>(dot(a1, b0q), dot(a1, b1q));
      kacc1 += vec2<f32>(dot(a1, b0k), dot(a1, b1k));
      vacc1 += vec2<f32>(dot(a1, b0v), dot(a1, b1v));
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      qacc2 += vec2<f32>(dot(a2, b0q), dot(a2, b1q));
      kacc2 += vec2<f32>(dot(a2, b0k), dot(a2, b1k));
      vacc2 += vec2<f32>(dot(a2, b0v), dot(a2, b1v));
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      qacc3 += vec2<f32>(dot(a3, b0q), dot(a3, b1q));
      kacc3 += vec2<f32>(dot(a3, b0k), dot(a3, b1k));
      vacc3 += vec2<f32>(dot(a3, b0v), dot(a3, b1v));
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      qacc4 += vec2<f32>(dot(a4, b0q), dot(a4, b1q));
      kacc4 += vec2<f32>(dot(a4, b0k), dot(a4, b1k));
      vacc4 += vec2<f32>(dot(a4, b0v), dot(a4, b1v));
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      qacc5 += vec2<f32>(dot(a5, b0q), dot(a5, b1q));
      kacc5 += vec2<f32>(dot(a5, b0k), dot(a5, b1k));
      vacc5 += vec2<f32>(dot(a5, b0v), dot(a5, b1v));
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      qacc6 += vec2<f32>(dot(a6, b0q), dot(a6, b1q));
      kacc6 += vec2<f32>(dot(a6, b0k), dot(a6, b1k));
      vacc6 += vec2<f32>(dot(a6, b0v), dot(a6, b1v));
    }
    {
      let kq = 6u;
      let b0q = bf16(wsQ[c * 8u + (kq ^ nx)]);
      let b0k = bf16(wsK[c * 8u + (kq ^ nx)]);
      let b0v = bf16(wsV[c * 8u + (kq ^ nx)]);
      let b1q = bf16(wsQ[(c + 32u) * 8u + (kq ^ nx)]);
      let b1k = bf16(wsK[(c + 32u) * 8u + (kq ^ nx)]);
      let b1v = bf16(wsV[(c + 32u) * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      qacc0 += vec2<f32>(dot(a0, b0q), dot(a0, b1q));
      kacc0 += vec2<f32>(dot(a0, b0k), dot(a0, b1k));
      vacc0 += vec2<f32>(dot(a0, b0v), dot(a0, b1v));
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      qacc1 += vec2<f32>(dot(a1, b0q), dot(a1, b1q));
      kacc1 += vec2<f32>(dot(a1, b0k), dot(a1, b1k));
      vacc1 += vec2<f32>(dot(a1, b0v), dot(a1, b1v));
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      qacc2 += vec2<f32>(dot(a2, b0q), dot(a2, b1q));
      kacc2 += vec2<f32>(dot(a2, b0k), dot(a2, b1k));
      vacc2 += vec2<f32>(dot(a2, b0v), dot(a2, b1v));
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      qacc3 += vec2<f32>(dot(a3, b0q), dot(a3, b1q));
      kacc3 += vec2<f32>(dot(a3, b0k), dot(a3, b1k));
      vacc3 += vec2<f32>(dot(a3, b0v), dot(a3, b1v));
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      qacc4 += vec2<f32>(dot(a4, b0q), dot(a4, b1q));
      kacc4 += vec2<f32>(dot(a4, b0k), dot(a4, b1k));
      vacc4 += vec2<f32>(dot(a4, b0v), dot(a4, b1v));
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      qacc5 += vec2<f32>(dot(a5, b0q), dot(a5, b1q));
      kacc5 += vec2<f32>(dot(a5, b0k), dot(a5, b1k));
      vacc5 += vec2<f32>(dot(a5, b0v), dot(a5, b1v));
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      qacc6 += vec2<f32>(dot(a6, b0q), dot(a6, b1q));
      kacc6 += vec2<f32>(dot(a6, b0k), dot(a6, b1k));
      vacc6 += vec2<f32>(dot(a6, b0v), dot(a6, b1v));
    }
    {
      let kq = 7u;
      let b0q = bf16(wsQ[c * 8u + (kq ^ nx)]);
      let b0k = bf16(wsK[c * 8u + (kq ^ nx)]);
      let b0v = bf16(wsV[c * 8u + (kq ^ nx)]);
      let b1q = bf16(wsQ[(c + 32u) * 8u + (kq ^ nx)]);
      let b1k = bf16(wsK[(c + 32u) * 8u + (kq ^ nx)]);
      let b1v = bf16(wsV[(c + 32u) * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      qacc0 += vec2<f32>(dot(a0, b0q), dot(a0, b1q));
      kacc0 += vec2<f32>(dot(a0, b0k), dot(a0, b1k));
      vacc0 += vec2<f32>(dot(a0, b0v), dot(a0, b1v));
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      qacc1 += vec2<f32>(dot(a1, b0q), dot(a1, b1q));
      kacc1 += vec2<f32>(dot(a1, b0k), dot(a1, b1k));
      vacc1 += vec2<f32>(dot(a1, b0v), dot(a1, b1v));
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      qacc2 += vec2<f32>(dot(a2, b0q), dot(a2, b1q));
      kacc2 += vec2<f32>(dot(a2, b0k), dot(a2, b1k));
      vacc2 += vec2<f32>(dot(a2, b0v), dot(a2, b1v));
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      qacc3 += vec2<f32>(dot(a3, b0q), dot(a3, b1q));
      kacc3 += vec2<f32>(dot(a3, b0k), dot(a3, b1k));
      vacc3 += vec2<f32>(dot(a3, b0v), dot(a3, b1v));
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      qacc4 += vec2<f32>(dot(a4, b0q), dot(a4, b1q));
      kacc4 += vec2<f32>(dot(a4, b0k), dot(a4, b1k));
      vacc4 += vec2<f32>(dot(a4, b0v), dot(a4, b1v));
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      qacc5 += vec2<f32>(dot(a5, b0q), dot(a5, b1q));
      kacc5 += vec2<f32>(dot(a5, b0k), dot(a5, b1k));
      vacc5 += vec2<f32>(dot(a5, b0v), dot(a5, b1v));
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      qacc6 += vec2<f32>(dot(a6, b0q), dot(a6, b1q));
      kacc6 += vec2<f32>(dot(a6, b0k), dot(a6, b1k));
      vacc6 += vec2<f32>(dot(a6, b0v), dot(a6, b1v));
    }
    workgroupBarrier();
  }

  let row = m0 + lid.y;
  {
    let col = n0 + lid.x + 0u;
    if (col < N) {
      qkv_store(qacc0.x, row + 0u, col, T, N, wg.z, splits, 0u);
      qkv_store(kacc0.x, row + 0u, col, T, N, wg.z, splits, 1u);
      qkv_store(vacc0.x, row + 0u, col, T, N, wg.z, splits, 2u);
      qkv_store(qacc1.x, row + 8u, col, T, N, wg.z, splits, 0u);
      qkv_store(kacc1.x, row + 8u, col, T, N, wg.z, splits, 1u);
      qkv_store(vacc1.x, row + 8u, col, T, N, wg.z, splits, 2u);
      qkv_store(qacc2.x, row + 16u, col, T, N, wg.z, splits, 0u);
      qkv_store(kacc2.x, row + 16u, col, T, N, wg.z, splits, 1u);
      qkv_store(vacc2.x, row + 16u, col, T, N, wg.z, splits, 2u);
      qkv_store(qacc3.x, row + 24u, col, T, N, wg.z, splits, 0u);
      qkv_store(kacc3.x, row + 24u, col, T, N, wg.z, splits, 1u);
      qkv_store(vacc3.x, row + 24u, col, T, N, wg.z, splits, 2u);
      qkv_store(qacc4.x, row + 32u, col, T, N, wg.z, splits, 0u);
      qkv_store(kacc4.x, row + 32u, col, T, N, wg.z, splits, 1u);
      qkv_store(vacc4.x, row + 32u, col, T, N, wg.z, splits, 2u);
      qkv_store(qacc5.x, row + 40u, col, T, N, wg.z, splits, 0u);
      qkv_store(kacc5.x, row + 40u, col, T, N, wg.z, splits, 1u);
      qkv_store(vacc5.x, row + 40u, col, T, N, wg.z, splits, 2u);
      qkv_store(qacc6.x, row + 48u, col, T, N, wg.z, splits, 0u);
      qkv_store(kacc6.x, row + 48u, col, T, N, wg.z, splits, 1u);
      qkv_store(vacc6.x, row + 48u, col, T, N, wg.z, splits, 2u);
    }
  }
  {
    let col = n0 + lid.x + 32u;
    if (col < N) {
      qkv_store(qacc0.y, row + 0u, col, T, N, wg.z, splits, 0u);
      qkv_store(kacc0.y, row + 0u, col, T, N, wg.z, splits, 1u);
      qkv_store(vacc0.y, row + 0u, col, T, N, wg.z, splits, 2u);
      qkv_store(qacc1.y, row + 8u, col, T, N, wg.z, splits, 0u);
      qkv_store(kacc1.y, row + 8u, col, T, N, wg.z, splits, 1u);
      qkv_store(vacc1.y, row + 8u, col, T, N, wg.z, splits, 2u);
      qkv_store(qacc2.y, row + 16u, col, T, N, wg.z, splits, 0u);
      qkv_store(kacc2.y, row + 16u, col, T, N, wg.z, splits, 1u);
      qkv_store(vacc2.y, row + 16u, col, T, N, wg.z, splits, 2u);
      qkv_store(qacc3.y, row + 24u, col, T, N, wg.z, splits, 0u);
      qkv_store(kacc3.y, row + 24u, col, T, N, wg.z, splits, 1u);
      qkv_store(vacc3.y, row + 24u, col, T, N, wg.z, splits, 2u);
      qkv_store(qacc4.y, row + 32u, col, T, N, wg.z, splits, 0u);
      qkv_store(kacc4.y, row + 32u, col, T, N, wg.z, splits, 1u);
      qkv_store(vacc4.y, row + 32u, col, T, N, wg.z, splits, 2u);
      qkv_store(qacc5.y, row + 40u, col, T, N, wg.z, splits, 0u);
      qkv_store(kacc5.y, row + 40u, col, T, N, wg.z, splits, 1u);
      qkv_store(vacc5.y, row + 40u, col, T, N, wg.z, splits, 2u);
      qkv_store(qacc6.y, row + 48u, col, T, N, wg.z, splits, 0u);
      qkv_store(kacc6.y, row + 48u, col, T, N, wg.z, splits, 1u);
      qkv_store(vacc6.y, row + 48u, col, T, N, wg.z, splits, 2u);
    }
  }
}
