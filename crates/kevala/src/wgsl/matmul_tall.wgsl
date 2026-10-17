// Y[T, N] = X[T, K] . W[N, K]^T (+ bias), with Q8 stored weights.
//
// A workgroup of 32 x 8 threads covers 104 rows and 64 columns. Each thread owns rows
// y + 8 i and columns x and x + 32, with thirteen named vec2 FP32 accumulators. It keeps
// the Q8/F16 K32 staging, swizzle, casts, and barriers from the registered single-matmul kernel.
enable f16;
//#include common

struct P { N: u32, K: u32, mode: u32, bias: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> S: array<f32>;
@group(0) @binding(5) var<storage, read> B: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
@group(0) @binding(7) var<storage, read_write> PART: array<f32>;

var<workgroup> xs: array<vec4<f16>, 832>; // [m][k/4]
var<workgroup> ws: array<vec4<f16>, 512>; // [n][(k/4) ^ (n & 7)], swizzled

// four int8 weights as f32, exactly and without int-to-float conversions (Marlin's trick): the
// xor makes each byte b + 128, and a byte in the low mantissa bits of 2^23 reads as 2^23 + b + 128
fn sx(w: u32) -> vec4<f32> {
  let u = w ^ 0x80808080u;
  let bytes = vec4<u32>(u, u >> 8u, u >> 16u, u >> 24u) & vec4<u32>(0xFFu);
  return bitcast<vec4<f32>>(bytes | vec4<u32>(0x4B000000u)) - vec4<f32>(8388736.0);
}

//#include splits

fn mm_store(value: f32, row: u32, col: u32, bias: f32, T: u32, N: u32, splits: u32, z: u32) {
  if (row >= T) { return; }
  let o = row * N + col;
  if (splits > 1u) { PART[z * T * N + o] = value; return; }
  var v = value + bias;
  if (p.mode == 1u) { v += Y[o]; } else if (p.mode == 2u) { v = max(v, 0.0); }
  Y[o] = v;
}

@compute @workgroup_size(32, 8)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let T = g.T;
  let N = p.N;
  let K = p.K;
  let nb = K / 32u;
  let m0 = wg.y * 104u;
  let n0 = wg.x * 64u;
  // split-K: this workgroup covers blocks kb0..kb1 and, when split, writes a partial tile
  let splits = mm_splits(T, N, K);
  let per = (nb + splits - 1u) / splits;
  let kb0 = wg.z * per;
  let kb1 = min(nb, kb0 + per);
  var acc0 = vec2<f32>(0.0);
  var acc1 = vec2<f32>(0.0);
  var acc2 = vec2<f32>(0.0);
  var acc3 = vec2<f32>(0.0);
  var acc4 = vec2<f32>(0.0);
  var acc5 = vec2<f32>(0.0);
  var acc6 = vec2<f32>(0.0);
  var acc7 = vec2<f32>(0.0);
  var acc8 = vec2<f32>(0.0);
  var acc9 = vec2<f32>(0.0);
  var acc10 = vec2<f32>(0.0);
  var acc11 = vec2<f32>(0.0);
  var acc12 = vec2<f32>(0.0);

  // loaders: 4 threads per row, 8 values each; X rows 0..104, W rows lr + 64 g
  let lr = li / 4u;
  let lq = li % 4u;
  let xrow = m0 + lr;
  let sw = lr & 7u;
  var xv0 = vec4<f32>(0.0);
  var xv1 = vec4<f32>(0.0);
  var xv2 = vec4<f32>(0.0);
  var xv3 = vec4<f32>(0.0);
  var wv: array<vec4<f32>, 2>;
  for (var kb = kb0; kb < kb1; kb++) {
    xv0 = vec4<f32>(0.0);
    xv1 = vec4<f32>(0.0);
    xv2 = vec4<f32>(0.0);
    xv3 = vec4<f32>(0.0);
    if (lr < 64u && xrow < T) {
      let base = (xrow * K + kb * 32u) / 4u + lq * 2u;
      xv0 = X[base];
      xv1 = X[base + 1u];
    }
    let xrow2 = xrow + 64u;
    if (lr + 64u < 104u && xrow2 < T) {
      let base2 = (xrow2 * K + kb * 32u) / 4u + lq * 2u;
      xv2 = X[base2];
      xv3 = X[base2 + 1u];
    }
    for (var gg = 0u; gg < 1u; gg++) {
      let wrow = n0 + lr + 64u * gg;
      wv[2u * gg] = vec4<f32>(0.0);
      wv[2u * gg + 1u] = vec4<f32>(0.0);
      if (wrow < N) {
        let v = W[(wrow * K + kb * 32u) / 16u + lq / 2u];
        let s = S[wrow * nb + kb];
        let lo = select(v.xy, v.zw, (lq & 1u) == 1u);
        wv[2u * gg] = sx(lo.x) * s;
        wv[2u * gg + 1u] = sx(lo.y) * s;
      }
    }
    if (lr < 64u) {
      xs[lr * 8u + lq * 2u] = vec4<f16>(xv0);
      xs[lr * 8u + lq * 2u + 1u] = vec4<f16>(xv1);
    }
    if (lr + 64u < 104u) {
      xs[(lr + 64u) * 8u + lq * 2u] = vec4<f16>(xv2);
      xs[(lr + 64u) * 8u + lq * 2u + 1u] = vec4<f16>(xv3);
    }
    for (var gg = 0u; gg < 1u; gg++) {
      let r = lr + 64u * gg;
      ws[r * 8u + ((lq * 2u) ^ sw)] = vec4<f16>(wv[2u * gg]);
      ws[r * 8u + ((lq * 2u + 1u) ^ sw)] = vec4<f16>(wv[2u * gg + 1u]);
    }
    workgroupBarrier();
    let nx = lid.x & 7u;
    {
      let kq = 0u;
      let b0 = vec4<f32>(ws[lid.x * 8u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(lid.x + 32u) * 8u + (kq ^ nx)]);
    let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + 0u]);
    acc0 += vec2<f32>(dot(a0, b0), dot(a0, b1));
    let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + 0u]);
    acc1 += vec2<f32>(dot(a1, b0), dot(a1, b1));
    let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + 0u]);
    acc2 += vec2<f32>(dot(a2, b0), dot(a2, b1));
    let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + 0u]);
    acc3 += vec2<f32>(dot(a3, b0), dot(a3, b1));
    let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + 0u]);
    acc4 += vec2<f32>(dot(a4, b0), dot(a4, b1));
    let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + 0u]);
    acc5 += vec2<f32>(dot(a5, b0), dot(a5, b1));
    let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + 0u]);
    acc6 += vec2<f32>(dot(a6, b0), dot(a6, b1));
    let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + 0u]);
    acc7 += vec2<f32>(dot(a7, b0), dot(a7, b1));
    let a8 = vec4<f32>(xs[(lid.y + 64u) * 8u + 0u]);
    acc8 += vec2<f32>(dot(a8, b0), dot(a8, b1));
    let a9 = vec4<f32>(xs[(lid.y + 72u) * 8u + 0u]);
    acc9 += vec2<f32>(dot(a9, b0), dot(a9, b1));
    let a10 = vec4<f32>(xs[(lid.y + 80u) * 8u + 0u]);
    acc10 += vec2<f32>(dot(a10, b0), dot(a10, b1));
    let a11 = vec4<f32>(xs[(lid.y + 88u) * 8u + 0u]);
    acc11 += vec2<f32>(dot(a11, b0), dot(a11, b1));
    let a12 = vec4<f32>(xs[(lid.y + 96u) * 8u + 0u]);
    acc12 += vec2<f32>(dot(a12, b0), dot(a12, b1));
    }
    {
      let kq = 1u;
      let b0 = vec4<f32>(ws[lid.x * 8u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(lid.x + 32u) * 8u + (kq ^ nx)]);
    let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + 1u]);
    acc0 += vec2<f32>(dot(a0, b0), dot(a0, b1));
    let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + 1u]);
    acc1 += vec2<f32>(dot(a1, b0), dot(a1, b1));
    let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + 1u]);
    acc2 += vec2<f32>(dot(a2, b0), dot(a2, b1));
    let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + 1u]);
    acc3 += vec2<f32>(dot(a3, b0), dot(a3, b1));
    let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + 1u]);
    acc4 += vec2<f32>(dot(a4, b0), dot(a4, b1));
    let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + 1u]);
    acc5 += vec2<f32>(dot(a5, b0), dot(a5, b1));
    let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + 1u]);
    acc6 += vec2<f32>(dot(a6, b0), dot(a6, b1));
    let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + 1u]);
    acc7 += vec2<f32>(dot(a7, b0), dot(a7, b1));
    let a8 = vec4<f32>(xs[(lid.y + 64u) * 8u + 1u]);
    acc8 += vec2<f32>(dot(a8, b0), dot(a8, b1));
    let a9 = vec4<f32>(xs[(lid.y + 72u) * 8u + 1u]);
    acc9 += vec2<f32>(dot(a9, b0), dot(a9, b1));
    let a10 = vec4<f32>(xs[(lid.y + 80u) * 8u + 1u]);
    acc10 += vec2<f32>(dot(a10, b0), dot(a10, b1));
    let a11 = vec4<f32>(xs[(lid.y + 88u) * 8u + 1u]);
    acc11 += vec2<f32>(dot(a11, b0), dot(a11, b1));
    let a12 = vec4<f32>(xs[(lid.y + 96u) * 8u + 1u]);
    acc12 += vec2<f32>(dot(a12, b0), dot(a12, b1));
    }
    {
      let kq = 2u;
      let b0 = vec4<f32>(ws[lid.x * 8u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(lid.x + 32u) * 8u + (kq ^ nx)]);
    let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + 2u]);
    acc0 += vec2<f32>(dot(a0, b0), dot(a0, b1));
    let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + 2u]);
    acc1 += vec2<f32>(dot(a1, b0), dot(a1, b1));
    let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + 2u]);
    acc2 += vec2<f32>(dot(a2, b0), dot(a2, b1));
    let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + 2u]);
    acc3 += vec2<f32>(dot(a3, b0), dot(a3, b1));
    let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + 2u]);
    acc4 += vec2<f32>(dot(a4, b0), dot(a4, b1));
    let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + 2u]);
    acc5 += vec2<f32>(dot(a5, b0), dot(a5, b1));
    let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + 2u]);
    acc6 += vec2<f32>(dot(a6, b0), dot(a6, b1));
    let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + 2u]);
    acc7 += vec2<f32>(dot(a7, b0), dot(a7, b1));
    let a8 = vec4<f32>(xs[(lid.y + 64u) * 8u + 2u]);
    acc8 += vec2<f32>(dot(a8, b0), dot(a8, b1));
    let a9 = vec4<f32>(xs[(lid.y + 72u) * 8u + 2u]);
    acc9 += vec2<f32>(dot(a9, b0), dot(a9, b1));
    let a10 = vec4<f32>(xs[(lid.y + 80u) * 8u + 2u]);
    acc10 += vec2<f32>(dot(a10, b0), dot(a10, b1));
    let a11 = vec4<f32>(xs[(lid.y + 88u) * 8u + 2u]);
    acc11 += vec2<f32>(dot(a11, b0), dot(a11, b1));
    let a12 = vec4<f32>(xs[(lid.y + 96u) * 8u + 2u]);
    acc12 += vec2<f32>(dot(a12, b0), dot(a12, b1));
    }
    {
      let kq = 3u;
      let b0 = vec4<f32>(ws[lid.x * 8u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(lid.x + 32u) * 8u + (kq ^ nx)]);
    let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + 3u]);
    acc0 += vec2<f32>(dot(a0, b0), dot(a0, b1));
    let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + 3u]);
    acc1 += vec2<f32>(dot(a1, b0), dot(a1, b1));
    let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + 3u]);
    acc2 += vec2<f32>(dot(a2, b0), dot(a2, b1));
    let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + 3u]);
    acc3 += vec2<f32>(dot(a3, b0), dot(a3, b1));
    let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + 3u]);
    acc4 += vec2<f32>(dot(a4, b0), dot(a4, b1));
    let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + 3u]);
    acc5 += vec2<f32>(dot(a5, b0), dot(a5, b1));
    let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + 3u]);
    acc6 += vec2<f32>(dot(a6, b0), dot(a6, b1));
    let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + 3u]);
    acc7 += vec2<f32>(dot(a7, b0), dot(a7, b1));
    let a8 = vec4<f32>(xs[(lid.y + 64u) * 8u + 3u]);
    acc8 += vec2<f32>(dot(a8, b0), dot(a8, b1));
    let a9 = vec4<f32>(xs[(lid.y + 72u) * 8u + 3u]);
    acc9 += vec2<f32>(dot(a9, b0), dot(a9, b1));
    let a10 = vec4<f32>(xs[(lid.y + 80u) * 8u + 3u]);
    acc10 += vec2<f32>(dot(a10, b0), dot(a10, b1));
    let a11 = vec4<f32>(xs[(lid.y + 88u) * 8u + 3u]);
    acc11 += vec2<f32>(dot(a11, b0), dot(a11, b1));
    let a12 = vec4<f32>(xs[(lid.y + 96u) * 8u + 3u]);
    acc12 += vec2<f32>(dot(a12, b0), dot(a12, b1));
    }
    {
      let kq = 4u;
      let b0 = vec4<f32>(ws[lid.x * 8u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(lid.x + 32u) * 8u + (kq ^ nx)]);
    let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + 4u]);
    acc0 += vec2<f32>(dot(a0, b0), dot(a0, b1));
    let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + 4u]);
    acc1 += vec2<f32>(dot(a1, b0), dot(a1, b1));
    let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + 4u]);
    acc2 += vec2<f32>(dot(a2, b0), dot(a2, b1));
    let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + 4u]);
    acc3 += vec2<f32>(dot(a3, b0), dot(a3, b1));
    let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + 4u]);
    acc4 += vec2<f32>(dot(a4, b0), dot(a4, b1));
    let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + 4u]);
    acc5 += vec2<f32>(dot(a5, b0), dot(a5, b1));
    let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + 4u]);
    acc6 += vec2<f32>(dot(a6, b0), dot(a6, b1));
    let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + 4u]);
    acc7 += vec2<f32>(dot(a7, b0), dot(a7, b1));
    let a8 = vec4<f32>(xs[(lid.y + 64u) * 8u + 4u]);
    acc8 += vec2<f32>(dot(a8, b0), dot(a8, b1));
    let a9 = vec4<f32>(xs[(lid.y + 72u) * 8u + 4u]);
    acc9 += vec2<f32>(dot(a9, b0), dot(a9, b1));
    let a10 = vec4<f32>(xs[(lid.y + 80u) * 8u + 4u]);
    acc10 += vec2<f32>(dot(a10, b0), dot(a10, b1));
    let a11 = vec4<f32>(xs[(lid.y + 88u) * 8u + 4u]);
    acc11 += vec2<f32>(dot(a11, b0), dot(a11, b1));
    let a12 = vec4<f32>(xs[(lid.y + 96u) * 8u + 4u]);
    acc12 += vec2<f32>(dot(a12, b0), dot(a12, b1));
    }
    {
      let kq = 5u;
      let b0 = vec4<f32>(ws[lid.x * 8u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(lid.x + 32u) * 8u + (kq ^ nx)]);
    let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + 5u]);
    acc0 += vec2<f32>(dot(a0, b0), dot(a0, b1));
    let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + 5u]);
    acc1 += vec2<f32>(dot(a1, b0), dot(a1, b1));
    let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + 5u]);
    acc2 += vec2<f32>(dot(a2, b0), dot(a2, b1));
    let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + 5u]);
    acc3 += vec2<f32>(dot(a3, b0), dot(a3, b1));
    let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + 5u]);
    acc4 += vec2<f32>(dot(a4, b0), dot(a4, b1));
    let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + 5u]);
    acc5 += vec2<f32>(dot(a5, b0), dot(a5, b1));
    let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + 5u]);
    acc6 += vec2<f32>(dot(a6, b0), dot(a6, b1));
    let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + 5u]);
    acc7 += vec2<f32>(dot(a7, b0), dot(a7, b1));
    let a8 = vec4<f32>(xs[(lid.y + 64u) * 8u + 5u]);
    acc8 += vec2<f32>(dot(a8, b0), dot(a8, b1));
    let a9 = vec4<f32>(xs[(lid.y + 72u) * 8u + 5u]);
    acc9 += vec2<f32>(dot(a9, b0), dot(a9, b1));
    let a10 = vec4<f32>(xs[(lid.y + 80u) * 8u + 5u]);
    acc10 += vec2<f32>(dot(a10, b0), dot(a10, b1));
    let a11 = vec4<f32>(xs[(lid.y + 88u) * 8u + 5u]);
    acc11 += vec2<f32>(dot(a11, b0), dot(a11, b1));
    let a12 = vec4<f32>(xs[(lid.y + 96u) * 8u + 5u]);
    acc12 += vec2<f32>(dot(a12, b0), dot(a12, b1));
    }
    {
      let kq = 6u;
      let b0 = vec4<f32>(ws[lid.x * 8u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(lid.x + 32u) * 8u + (kq ^ nx)]);
    let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + 6u]);
    acc0 += vec2<f32>(dot(a0, b0), dot(a0, b1));
    let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + 6u]);
    acc1 += vec2<f32>(dot(a1, b0), dot(a1, b1));
    let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + 6u]);
    acc2 += vec2<f32>(dot(a2, b0), dot(a2, b1));
    let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + 6u]);
    acc3 += vec2<f32>(dot(a3, b0), dot(a3, b1));
    let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + 6u]);
    acc4 += vec2<f32>(dot(a4, b0), dot(a4, b1));
    let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + 6u]);
    acc5 += vec2<f32>(dot(a5, b0), dot(a5, b1));
    let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + 6u]);
    acc6 += vec2<f32>(dot(a6, b0), dot(a6, b1));
    let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + 6u]);
    acc7 += vec2<f32>(dot(a7, b0), dot(a7, b1));
    let a8 = vec4<f32>(xs[(lid.y + 64u) * 8u + 6u]);
    acc8 += vec2<f32>(dot(a8, b0), dot(a8, b1));
    let a9 = vec4<f32>(xs[(lid.y + 72u) * 8u + 6u]);
    acc9 += vec2<f32>(dot(a9, b0), dot(a9, b1));
    let a10 = vec4<f32>(xs[(lid.y + 80u) * 8u + 6u]);
    acc10 += vec2<f32>(dot(a10, b0), dot(a10, b1));
    let a11 = vec4<f32>(xs[(lid.y + 88u) * 8u + 6u]);
    acc11 += vec2<f32>(dot(a11, b0), dot(a11, b1));
    let a12 = vec4<f32>(xs[(lid.y + 96u) * 8u + 6u]);
    acc12 += vec2<f32>(dot(a12, b0), dot(a12, b1));
    }
    {
      let kq = 7u;
      let b0 = vec4<f32>(ws[lid.x * 8u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(lid.x + 32u) * 8u + (kq ^ nx)]);
    let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + 7u]);
    acc0 += vec2<f32>(dot(a0, b0), dot(a0, b1));
    let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + 7u]);
    acc1 += vec2<f32>(dot(a1, b0), dot(a1, b1));
    let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + 7u]);
    acc2 += vec2<f32>(dot(a2, b0), dot(a2, b1));
    let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + 7u]);
    acc3 += vec2<f32>(dot(a3, b0), dot(a3, b1));
    let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + 7u]);
    acc4 += vec2<f32>(dot(a4, b0), dot(a4, b1));
    let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + 7u]);
    acc5 += vec2<f32>(dot(a5, b0), dot(a5, b1));
    let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + 7u]);
    acc6 += vec2<f32>(dot(a6, b0), dot(a6, b1));
    let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + 7u]);
    acc7 += vec2<f32>(dot(a7, b0), dot(a7, b1));
    let a8 = vec4<f32>(xs[(lid.y + 64u) * 8u + 7u]);
    acc8 += vec2<f32>(dot(a8, b0), dot(a8, b1));
    let a9 = vec4<f32>(xs[(lid.y + 72u) * 8u + 7u]);
    acc9 += vec2<f32>(dot(a9, b0), dot(a9, b1));
    let a10 = vec4<f32>(xs[(lid.y + 80u) * 8u + 7u]);
    acc10 += vec2<f32>(dot(a10, b0), dot(a10, b1));
    let a11 = vec4<f32>(xs[(lid.y + 88u) * 8u + 7u]);
    acc11 += vec2<f32>(dot(a11, b0), dot(a11, b1));
    let a12 = vec4<f32>(xs[(lid.y + 96u) * 8u + 7u]);
    acc12 += vec2<f32>(dot(a12, b0), dot(a12, b1));
    }
    workgroupBarrier();
  }

  {
    let col = n0 + lid.x + 0u;
    if (col < N) {
      var bias = 0.0;
      if (p.bias == 1u) { bias = B[col]; }
      let row = m0 + lid.y;
      mm_store(acc0.x, row + 0u, col, bias, T, N, splits, wg.z);
      mm_store(acc1.x, row + 8u, col, bias, T, N, splits, wg.z);
      mm_store(acc2.x, row + 16u, col, bias, T, N, splits, wg.z);
      mm_store(acc3.x, row + 24u, col, bias, T, N, splits, wg.z);
      mm_store(acc4.x, row + 32u, col, bias, T, N, splits, wg.z);
      mm_store(acc5.x, row + 40u, col, bias, T, N, splits, wg.z);
      mm_store(acc6.x, row + 48u, col, bias, T, N, splits, wg.z);
      mm_store(acc7.x, row + 56u, col, bias, T, N, splits, wg.z);
      mm_store(acc8.x, row + 64u, col, bias, T, N, splits, wg.z);
      mm_store(acc9.x, row + 72u, col, bias, T, N, splits, wg.z);
      mm_store(acc10.x, row + 80u, col, bias, T, N, splits, wg.z);
      mm_store(acc11.x, row + 88u, col, bias, T, N, splits, wg.z);
      mm_store(acc12.x, row + 96u, col, bias, T, N, splits, wg.z);
    }
  }
  {
    let col = n0 + lid.x + 32u;
    if (col < N) {
      var bias = 0.0;
      if (p.bias == 1u) { bias = B[col]; }
      let row = m0 + lid.y;
      mm_store(acc0.y, row + 0u, col, bias, T, N, splits, wg.z);
      mm_store(acc1.y, row + 8u, col, bias, T, N, splits, wg.z);
      mm_store(acc2.y, row + 16u, col, bias, T, N, splits, wg.z);
      mm_store(acc3.y, row + 24u, col, bias, T, N, splits, wg.z);
      mm_store(acc4.y, row + 32u, col, bias, T, N, splits, wg.z);
      mm_store(acc5.y, row + 40u, col, bias, T, N, splits, wg.z);
      mm_store(acc6.y, row + 48u, col, bias, T, N, splits, wg.z);
      mm_store(acc7.y, row + 56u, col, bias, T, N, splits, wg.z);
      mm_store(acc8.y, row + 64u, col, bias, T, N, splits, wg.z);
      mm_store(acc9.y, row + 72u, col, bias, T, N, splits, wg.z);
      mm_store(acc10.y, row + 80u, col, bias, T, N, splits, wg.z);
      mm_store(acc11.y, row + 88u, col, bias, T, N, splits, wg.z);
      mm_store(acc12.y, row + 96u, col, bias, T, N, splits, wg.z);
    }
  }
}
