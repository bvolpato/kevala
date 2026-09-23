// Experimental FP32 64 x 32 matrix tile, 32 x 8 threads, eight scalar outputs per thread.
// Physical BN32 uses the retained virtual BM64/BN64/target256 split schedule and reducer.
//#include common

struct P { N: u32, K: u32, mode: u32, bias: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> S: array<f32>;
@group(0) @binding(5) var<storage, read> B: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
@group(0) @binding(7) var<storage, read_write> PART: array<f32>;

var<workgroup> xs: array<vec4<f32>, 512>; // [m][k/4]
var<workgroup> ws: array<vec4<f32>, 256>; // [n][(k/4) ^ (n & 7)], swizzled

// Sign-extend each packed int8 lane, then convert the exact integer values to f32.
fn sx(w: u32) -> vec4<f32> {
  let bytes = vec4<i32>(bitcast<i32>(w << 24u), bitcast<i32>(w << 16u), bitcast<i32>(w << 8u), bitcast<i32>(w)) >> vec4<u32>(24u);
  return vec4<f32>(bytes);
}

fn mm_store(value: f32, row: u32, col: u32, bias: f32, T: u32, N: u32, splits: u32, z: u32) {
  if (row >= T) { return; }
  let o = row * N + col;
  if (splits > 1u) {
    PART[z * T * N + o] = value;
    return;
  }
  var v = value;
  v += bias;
  if (p.mode == 1u) { v += Y[o]; } else if (p.mode == 2u) { v = max(v, 0.0); }
  Y[o] = v;
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
  let m0 = wg.y * 64u;
  let n0 = wg.x * 32u;
  // split-K: this workgroup covers blocks kb0..kb1 and, when split, writes a partial tile
  let splits = mm_splits(T, N, K);
  let per = (nb + splits - 1u) / splits;
  let kb0 = wg.z * per;
  let kb1 = min(nb, kb0 + per);
  var acc0 = 0.0;
  var acc1 = 0.0;
  var acc2 = 0.0;
  var acc3 = 0.0;
  var acc4 = 0.0;
  var acc5 = 0.0;
  var acc6 = 0.0;
  var acc7 = 0.0;

  // Four threads per row stage X rows 0..64 and W rows 0..32.
  let lr = li / 4u;
  let lq = li % 4u;
  let xrow = m0 + lr;
  let sw = lr & 7u;
  var xv0 = vec4<f32>(0.0);
  var xv1 = vec4<f32>(0.0);
  var wv: array<vec4<f32>, 2>;
  for (var kb = kb0; kb < kb1; kb++) {
    xv0 = vec4<f32>(0.0);
    xv1 = vec4<f32>(0.0);
    if (lr < 64u && xrow < T) {
      let base = (xrow * K + kb * 32u) / 4u + lq * 2u;
      xv0 = X[base];
      xv1 = X[base + 1u];
    }
    for (var gg = 0u; gg < 1u; gg++) {
      let wrow = n0 + lr + 64u * gg;
      wv[2u * gg] = vec4<f32>(0.0);
      wv[2u * gg + 1u] = vec4<f32>(0.0);
      if (lr < 32u && wrow < N) {
        let v = W[(wrow * K + kb * 32u) / 16u + lq / 2u];
        let s = S[wrow * nb + kb];
        let lo = select(v.xy, v.zw, (lq & 1u) == 1u);
        wv[2u * gg] = sx(lo.x) * s;
        wv[2u * gg + 1u] = sx(lo.y) * s;
      }
    }
    if (lr < 64u) {
      xs[lr * 8u + lq * 2u] = vec4<f32>(xv0);
      xs[lr * 8u + lq * 2u + 1u] = vec4<f32>(xv1);
    }
    if (lr < 32u) {
      for (var gg = 0u; gg < 1u; gg++) {
        let r = lr + 64u * gg;
        ws[r * 8u + ((lq * 2u) ^ sw)] = vec4<f32>(wv[2u * gg]);
        ws[r * 8u + ((lq * 2u + 1u) ^ sw)] = vec4<f32>(wv[2u * gg + 1u]);
      }
    }
    workgroupBarrier();
    let nx = lid.x & 7u;
    // Keep the eight K/4 steps explicit, reusing one weight vector across eight rows.
    let c = lid.x;
    {
      let kq = 0u;
      let b0 = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += dot(a0, b0);
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc1 += dot(a1, b0);
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc2 += dot(a2, b0);
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc3 += dot(a3, b0);
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc4 += dot(a4, b0);
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc5 += dot(a5, b0);
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc6 += dot(a6, b0);
      let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc7 += dot(a7, b0);
    }
    {
      let kq = 1u;
      let b0 = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += dot(a0, b0);
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc1 += dot(a1, b0);
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc2 += dot(a2, b0);
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc3 += dot(a3, b0);
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc4 += dot(a4, b0);
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc5 += dot(a5, b0);
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc6 += dot(a6, b0);
      let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc7 += dot(a7, b0);
    }
    {
      let kq = 2u;
      let b0 = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += dot(a0, b0);
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc1 += dot(a1, b0);
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc2 += dot(a2, b0);
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc3 += dot(a3, b0);
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc4 += dot(a4, b0);
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc5 += dot(a5, b0);
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc6 += dot(a6, b0);
      let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc7 += dot(a7, b0);
    }
    {
      let kq = 3u;
      let b0 = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += dot(a0, b0);
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc1 += dot(a1, b0);
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc2 += dot(a2, b0);
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc3 += dot(a3, b0);
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc4 += dot(a4, b0);
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc5 += dot(a5, b0);
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc6 += dot(a6, b0);
      let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc7 += dot(a7, b0);
    }
    {
      let kq = 4u;
      let b0 = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += dot(a0, b0);
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc1 += dot(a1, b0);
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc2 += dot(a2, b0);
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc3 += dot(a3, b0);
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc4 += dot(a4, b0);
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc5 += dot(a5, b0);
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc6 += dot(a6, b0);
      let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc7 += dot(a7, b0);
    }
    {
      let kq = 5u;
      let b0 = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += dot(a0, b0);
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc1 += dot(a1, b0);
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc2 += dot(a2, b0);
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc3 += dot(a3, b0);
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc4 += dot(a4, b0);
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc5 += dot(a5, b0);
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc6 += dot(a6, b0);
      let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc7 += dot(a7, b0);
    }
    {
      let kq = 6u;
      let b0 = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += dot(a0, b0);
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc1 += dot(a1, b0);
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc2 += dot(a2, b0);
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc3 += dot(a3, b0);
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc4 += dot(a4, b0);
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc5 += dot(a5, b0);
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc6 += dot(a6, b0);
      let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc7 += dot(a7, b0);
    }
    {
      let kq = 7u;
      let b0 = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += dot(a0, b0);
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc1 += dot(a1, b0);
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc2 += dot(a2, b0);
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc3 += dot(a3, b0);
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc4 += dot(a4, b0);
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc5 += dot(a5, b0);
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc6 += dot(a6, b0);
      let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc7 += dot(a7, b0);
    }
    workgroupBarrier();
  }

  let row = m0 + lid.y;
  let col = n0 + lid.x;
  if (col < N) {
    var bias = 0.0;
    if (p.bias == 1u) { bias = B[col]; }
    mm_store(acc0, row + 0u, col, bias, T, N, splits, wg.z);
    mm_store(acc1, row + 8u, col, bias, T, N, splits, wg.z);
    mm_store(acc2, row + 16u, col, bias, T, N, splits, wg.z);
    mm_store(acc3, row + 24u, col, bias, T, N, splits, wg.z);
    mm_store(acc4, row + 32u, col, bias, T, N, splits, wg.z);
    mm_store(acc5, row + 40u, col, bias, T, N, splits, wg.z);
    mm_store(acc6, row + 48u, col, bias, T, N, splits, wg.z);
    mm_store(acc7, row + 56u, col, bias, T, N, splits, wg.z);
  }
}
