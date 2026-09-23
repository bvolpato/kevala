// Y[T, N] = X[T, K] . W[N, K]^T (+ bias), W int8 in u32 words with one f32 scale per 32 weights.
//
//#if GENERIC_UNROLL
// A workgroup of 64 x 4 threads covers {{BM}} rows and 64 columns. Each thread owns
// rows y + 4 i for i < 4 * {{ROWS}}, and column x, in named scalar FP32 accumulators.
// The K32 loaders and shared weight swizzle are identical to the 16 x 16 path.
//#else
// A workgroup of 16 x 16 threads covers {{BM}} rows ({{ROWS}} per thread) and {{BN}} columns
// ({{GROUPS}} groups of 64); thread (x, y) owns rows y + 16 i and columns x + 16 j + 64 g, so
// output stores coalesce. Each step stages one quantization block of K (32 values) for X and W
// in workgroup memory as {{TILE}}; products and sums stay f32. The weight tile is XOR-swizzled
// so neighbouring threads read different banks, and every thread writes whole vectors
// (sub-vector writes from several threads race on some GPUs).
//#endif
//#if F16
enable f16;
//#endif
//#include common

struct P { N: u32, K: u32, mode: u32, bias: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> S: array<f32>;
@group(0) @binding(5) var<storage, read> B: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
@group(0) @binding(7) var<storage, read_write> PART: array<f32>;

var<workgroup> xs: array<vec4<{{TILE}}>, 512>; // [m][k/4]
var<workgroup> ws: array<vec4<{{TILE}}>, {{WS_LEN}}>; // [n][(k/4) ^ (n & 7)], swizzled

// four int8 weights as f32, exactly and without int-to-float conversions (Marlin's trick): the
// xor makes each byte b + 128, and a byte in the low mantissa bits of 2^23 reads as 2^23 + b + 128
fn sx(w: u32) -> vec4<f32> {
  let u = w ^ 0x80808080u;
  let bytes = vec4<u32>(u, u >> 8u, u >> 16u, u >> 24u) & vec4<u32>(0xFFu);
  return bitcast<vec4<f32>>(bytes | vec4<u32>(0x4B000000u)) - vec4<f32>(8388736.0);
}

//#include splits

//#if GENERIC_UNROLL
@compute @workgroup_size(64, 4)
//#else
@compute @workgroup_size(16, 16)
//#endif
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let T = g.T;
//#if SHAPE
  let N = {{N}}u;
  let K = {{K}}u;
//#else
  let N = p.N;
  let K = p.K;
//#endif
  let nb = K / 32u;
  let m0 = wg.y * {{BM}}u;
  let n0 = wg.x * {{BN}}u;
  // split-K: this workgroup covers blocks kb0..kb1 and, when split, writes a partial tile
  let splits = mm_splits(T, N, K);
  let per = (nb + splits - 1u) / splits;
  let kb0 = wg.z * per;
  let kb1 = min(nb, kb0 + per);
//#if GENERIC_UNROLL
  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  var acc2: f32 = 0.0;
  var acc3: f32 = 0.0;
//#if ROWS_GE_2
  var acc4: f32 = 0.0;
  var acc5: f32 = 0.0;
  var acc6: f32 = 0.0;
  var acc7: f32 = 0.0;
//#endif
//#if ROWS_GE_3
  var acc8: f32 = 0.0;
  var acc9: f32 = 0.0;
  var acc10: f32 = 0.0;
  var acc11: f32 = 0.0;
//#endif
//#if ROWS_GE_4
  var acc12: f32 = 0.0;
  var acc13: f32 = 0.0;
  var acc14: f32 = 0.0;
  var acc15: f32 = 0.0;
//#endif
//#else
  var acc: array<vec4<f32>, {{ACC_LEN}}>; // [i][g]: row y + 16 i, columns x + 16 j + 64 g for j < 4
//#endif

  // loaders: 4 threads per row, 8 values each; X rows 0..{{BM}}, W rows lr + 64 g
  let lr = li / 4u;
  let lq = li % 4u;
  let xrow = m0 + lr;
  let sw = lr & 7u;
  var xv0 = vec4<f32>(0.0);
  var xv1 = vec4<f32>(0.0);
  var wv: array<vec4<f32>, {{WV_LEN}}>;
  for (var kb = kb0; kb < kb1; kb++) {
    xv0 = vec4<f32>(0.0);
    xv1 = vec4<f32>(0.0);
    if (lr < {{BM}}u && xrow < T) {
      let base = (xrow * K + kb * 32u) / 4u + lq * 2u;
      xv0 = X[base];
      xv1 = X[base + 1u];
    }
    for (var gg = 0u; gg < {{GROUPS}}u; gg++) {
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
    if (lr < {{BM}}u) {
      xs[lr * 8u + lq * 2u] = vec4<{{TILE}}>(xv0);
      xs[lr * 8u + lq * 2u + 1u] = vec4<{{TILE}}>(xv1);
    }
    for (var gg = 0u; gg < {{GROUPS}}u; gg++) {
      let r = lr + 64u * gg;
      ws[r * 8u + ((lq * 2u) ^ sw)] = vec4<{{TILE}}>(wv[2u * gg]);
      ws[r * 8u + ((lq * 2u + 1u) ^ sw)] = vec4<{{TILE}}>(wv[2u * gg + 1u]);
    }
    workgroupBarrier();
    let nx = lid.x & 7u;
//#if GENERIC_UNROLL
    // Keep the eight K/4 steps explicit, reusing one weight vector across the rows.
    let c = lid.x;
    {
      let kq = 0u;
      let b = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += dot(a0, b);
      let a1 = vec4<f32>(xs[(lid.y + 4u) * 8u + kq]);
      acc1 += dot(a1, b);
      let a2 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc2 += dot(a2, b);
      let a3 = vec4<f32>(xs[(lid.y + 12u) * 8u + kq]);
      acc3 += dot(a3, b);
//#if ROWS_GE_2
      let a4 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc4 += dot(a4, b);
      let a5 = vec4<f32>(xs[(lid.y + 20u) * 8u + kq]);
      acc5 += dot(a5, b);
      let a6 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc6 += dot(a6, b);
      let a7 = vec4<f32>(xs[(lid.y + 28u) * 8u + kq]);
      acc7 += dot(a7, b);
//#endif
//#if ROWS_GE_3
      let a8 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc8 += dot(a8, b);
      let a9 = vec4<f32>(xs[(lid.y + 36u) * 8u + kq]);
      acc9 += dot(a9, b);
      let a10 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc10 += dot(a10, b);
      let a11 = vec4<f32>(xs[(lid.y + 44u) * 8u + kq]);
      acc11 += dot(a11, b);
//#endif
//#if ROWS_GE_4
      let a12 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc12 += dot(a12, b);
      let a13 = vec4<f32>(xs[(lid.y + 52u) * 8u + kq]);
      acc13 += dot(a13, b);
      let a14 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc14 += dot(a14, b);
      let a15 = vec4<f32>(xs[(lid.y + 60u) * 8u + kq]);
      acc15 += dot(a15, b);
//#endif
    }
    {
      let kq = 1u;
      let b = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += dot(a0, b);
      let a1 = vec4<f32>(xs[(lid.y + 4u) * 8u + kq]);
      acc1 += dot(a1, b);
      let a2 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc2 += dot(a2, b);
      let a3 = vec4<f32>(xs[(lid.y + 12u) * 8u + kq]);
      acc3 += dot(a3, b);
//#if ROWS_GE_2
      let a4 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc4 += dot(a4, b);
      let a5 = vec4<f32>(xs[(lid.y + 20u) * 8u + kq]);
      acc5 += dot(a5, b);
      let a6 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc6 += dot(a6, b);
      let a7 = vec4<f32>(xs[(lid.y + 28u) * 8u + kq]);
      acc7 += dot(a7, b);
//#endif
//#if ROWS_GE_3
      let a8 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc8 += dot(a8, b);
      let a9 = vec4<f32>(xs[(lid.y + 36u) * 8u + kq]);
      acc9 += dot(a9, b);
      let a10 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc10 += dot(a10, b);
      let a11 = vec4<f32>(xs[(lid.y + 44u) * 8u + kq]);
      acc11 += dot(a11, b);
//#endif
//#if ROWS_GE_4
      let a12 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc12 += dot(a12, b);
      let a13 = vec4<f32>(xs[(lid.y + 52u) * 8u + kq]);
      acc13 += dot(a13, b);
      let a14 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc14 += dot(a14, b);
      let a15 = vec4<f32>(xs[(lid.y + 60u) * 8u + kq]);
      acc15 += dot(a15, b);
//#endif
    }
    {
      let kq = 2u;
      let b = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += dot(a0, b);
      let a1 = vec4<f32>(xs[(lid.y + 4u) * 8u + kq]);
      acc1 += dot(a1, b);
      let a2 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc2 += dot(a2, b);
      let a3 = vec4<f32>(xs[(lid.y + 12u) * 8u + kq]);
      acc3 += dot(a3, b);
//#if ROWS_GE_2
      let a4 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc4 += dot(a4, b);
      let a5 = vec4<f32>(xs[(lid.y + 20u) * 8u + kq]);
      acc5 += dot(a5, b);
      let a6 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc6 += dot(a6, b);
      let a7 = vec4<f32>(xs[(lid.y + 28u) * 8u + kq]);
      acc7 += dot(a7, b);
//#endif
//#if ROWS_GE_3
      let a8 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc8 += dot(a8, b);
      let a9 = vec4<f32>(xs[(lid.y + 36u) * 8u + kq]);
      acc9 += dot(a9, b);
      let a10 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc10 += dot(a10, b);
      let a11 = vec4<f32>(xs[(lid.y + 44u) * 8u + kq]);
      acc11 += dot(a11, b);
//#endif
//#if ROWS_GE_4
      let a12 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc12 += dot(a12, b);
      let a13 = vec4<f32>(xs[(lid.y + 52u) * 8u + kq]);
      acc13 += dot(a13, b);
      let a14 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc14 += dot(a14, b);
      let a15 = vec4<f32>(xs[(lid.y + 60u) * 8u + kq]);
      acc15 += dot(a15, b);
//#endif
    }
    {
      let kq = 3u;
      let b = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += dot(a0, b);
      let a1 = vec4<f32>(xs[(lid.y + 4u) * 8u + kq]);
      acc1 += dot(a1, b);
      let a2 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc2 += dot(a2, b);
      let a3 = vec4<f32>(xs[(lid.y + 12u) * 8u + kq]);
      acc3 += dot(a3, b);
//#if ROWS_GE_2
      let a4 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc4 += dot(a4, b);
      let a5 = vec4<f32>(xs[(lid.y + 20u) * 8u + kq]);
      acc5 += dot(a5, b);
      let a6 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc6 += dot(a6, b);
      let a7 = vec4<f32>(xs[(lid.y + 28u) * 8u + kq]);
      acc7 += dot(a7, b);
//#endif
//#if ROWS_GE_3
      let a8 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc8 += dot(a8, b);
      let a9 = vec4<f32>(xs[(lid.y + 36u) * 8u + kq]);
      acc9 += dot(a9, b);
      let a10 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc10 += dot(a10, b);
      let a11 = vec4<f32>(xs[(lid.y + 44u) * 8u + kq]);
      acc11 += dot(a11, b);
//#endif
//#if ROWS_GE_4
      let a12 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc12 += dot(a12, b);
      let a13 = vec4<f32>(xs[(lid.y + 52u) * 8u + kq]);
      acc13 += dot(a13, b);
      let a14 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc14 += dot(a14, b);
      let a15 = vec4<f32>(xs[(lid.y + 60u) * 8u + kq]);
      acc15 += dot(a15, b);
//#endif
    }
    {
      let kq = 4u;
      let b = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += dot(a0, b);
      let a1 = vec4<f32>(xs[(lid.y + 4u) * 8u + kq]);
      acc1 += dot(a1, b);
      let a2 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc2 += dot(a2, b);
      let a3 = vec4<f32>(xs[(lid.y + 12u) * 8u + kq]);
      acc3 += dot(a3, b);
//#if ROWS_GE_2
      let a4 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc4 += dot(a4, b);
      let a5 = vec4<f32>(xs[(lid.y + 20u) * 8u + kq]);
      acc5 += dot(a5, b);
      let a6 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc6 += dot(a6, b);
      let a7 = vec4<f32>(xs[(lid.y + 28u) * 8u + kq]);
      acc7 += dot(a7, b);
//#endif
//#if ROWS_GE_3
      let a8 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc8 += dot(a8, b);
      let a9 = vec4<f32>(xs[(lid.y + 36u) * 8u + kq]);
      acc9 += dot(a9, b);
      let a10 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc10 += dot(a10, b);
      let a11 = vec4<f32>(xs[(lid.y + 44u) * 8u + kq]);
      acc11 += dot(a11, b);
//#endif
//#if ROWS_GE_4
      let a12 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc12 += dot(a12, b);
      let a13 = vec4<f32>(xs[(lid.y + 52u) * 8u + kq]);
      acc13 += dot(a13, b);
      let a14 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc14 += dot(a14, b);
      let a15 = vec4<f32>(xs[(lid.y + 60u) * 8u + kq]);
      acc15 += dot(a15, b);
//#endif
    }
    {
      let kq = 5u;
      let b = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += dot(a0, b);
      let a1 = vec4<f32>(xs[(lid.y + 4u) * 8u + kq]);
      acc1 += dot(a1, b);
      let a2 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc2 += dot(a2, b);
      let a3 = vec4<f32>(xs[(lid.y + 12u) * 8u + kq]);
      acc3 += dot(a3, b);
//#if ROWS_GE_2
      let a4 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc4 += dot(a4, b);
      let a5 = vec4<f32>(xs[(lid.y + 20u) * 8u + kq]);
      acc5 += dot(a5, b);
      let a6 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc6 += dot(a6, b);
      let a7 = vec4<f32>(xs[(lid.y + 28u) * 8u + kq]);
      acc7 += dot(a7, b);
//#endif
//#if ROWS_GE_3
      let a8 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc8 += dot(a8, b);
      let a9 = vec4<f32>(xs[(lid.y + 36u) * 8u + kq]);
      acc9 += dot(a9, b);
      let a10 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc10 += dot(a10, b);
      let a11 = vec4<f32>(xs[(lid.y + 44u) * 8u + kq]);
      acc11 += dot(a11, b);
//#endif
//#if ROWS_GE_4
      let a12 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc12 += dot(a12, b);
      let a13 = vec4<f32>(xs[(lid.y + 52u) * 8u + kq]);
      acc13 += dot(a13, b);
      let a14 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc14 += dot(a14, b);
      let a15 = vec4<f32>(xs[(lid.y + 60u) * 8u + kq]);
      acc15 += dot(a15, b);
//#endif
    }
    {
      let kq = 6u;
      let b = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += dot(a0, b);
      let a1 = vec4<f32>(xs[(lid.y + 4u) * 8u + kq]);
      acc1 += dot(a1, b);
      let a2 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc2 += dot(a2, b);
      let a3 = vec4<f32>(xs[(lid.y + 12u) * 8u + kq]);
      acc3 += dot(a3, b);
//#if ROWS_GE_2
      let a4 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc4 += dot(a4, b);
      let a5 = vec4<f32>(xs[(lid.y + 20u) * 8u + kq]);
      acc5 += dot(a5, b);
      let a6 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc6 += dot(a6, b);
      let a7 = vec4<f32>(xs[(lid.y + 28u) * 8u + kq]);
      acc7 += dot(a7, b);
//#endif
//#if ROWS_GE_3
      let a8 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc8 += dot(a8, b);
      let a9 = vec4<f32>(xs[(lid.y + 36u) * 8u + kq]);
      acc9 += dot(a9, b);
      let a10 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc10 += dot(a10, b);
      let a11 = vec4<f32>(xs[(lid.y + 44u) * 8u + kq]);
      acc11 += dot(a11, b);
//#endif
//#if ROWS_GE_4
      let a12 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc12 += dot(a12, b);
      let a13 = vec4<f32>(xs[(lid.y + 52u) * 8u + kq]);
      acc13 += dot(a13, b);
      let a14 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc14 += dot(a14, b);
      let a15 = vec4<f32>(xs[(lid.y + 60u) * 8u + kq]);
      acc15 += dot(a15, b);
//#endif
    }
    {
      let kq = 7u;
      let b = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += dot(a0, b);
      let a1 = vec4<f32>(xs[(lid.y + 4u) * 8u + kq]);
      acc1 += dot(a1, b);
      let a2 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc2 += dot(a2, b);
      let a3 = vec4<f32>(xs[(lid.y + 12u) * 8u + kq]);
      acc3 += dot(a3, b);
//#if ROWS_GE_2
      let a4 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc4 += dot(a4, b);
      let a5 = vec4<f32>(xs[(lid.y + 20u) * 8u + kq]);
      acc5 += dot(a5, b);
      let a6 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc6 += dot(a6, b);
      let a7 = vec4<f32>(xs[(lid.y + 28u) * 8u + kq]);
      acc7 += dot(a7, b);
//#endif
//#if ROWS_GE_3
      let a8 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc8 += dot(a8, b);
      let a9 = vec4<f32>(xs[(lid.y + 36u) * 8u + kq]);
      acc9 += dot(a9, b);
      let a10 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc10 += dot(a10, b);
      let a11 = vec4<f32>(xs[(lid.y + 44u) * 8u + kq]);
      acc11 += dot(a11, b);
//#endif
//#if ROWS_GE_4
      let a12 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc12 += dot(a12, b);
      let a13 = vec4<f32>(xs[(lid.y + 52u) * 8u + kq]);
      acc13 += dot(a13, b);
      let a14 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc14 += dot(a14, b);
      let a15 = vec4<f32>(xs[(lid.y + 60u) * 8u + kq]);
      acc15 += dot(a15, b);
//#endif
    }
//#else
    // The f32 R4/G1 path keeps eight K/4 steps and four row accumulators visible to the compiler.
    for (var kq = 0u; kq < 8u; kq++) {
      for (var gg = 0u; gg < {{GROUPS}}u; gg++) {
        let c = lid.x + 64u * gg;
        let b0 = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
        let b1 = vec4<f32>(ws[(c + 16u) * 8u + (kq ^ nx)]);
        let b2 = vec4<f32>(ws[(c + 32u) * 8u + (kq ^ nx)]);
        let b3 = vec4<f32>(ws[(c + 48u) * 8u + (kq ^ nx)]);
        for (var i = 0u; i < {{ROWS}}u; i++) {
          let a = vec4<f32>(xs[(lid.y + 16u * i) * 8u + kq]);
          acc[i * {{GROUPS}}u + gg] += vec4<f32>(dot(a, b0), dot(a, b1), dot(a, b2), dot(a, b3));
        }
      }
    }
//#endif
    workgroupBarrier();
  }

//#if GENERIC_UNROLL
  let col = n0 + lid.x;
  if (col < N) {
    var bias = 0.0;
    if (p.bias == 1u) { bias = B[col]; }
    for (var i = 0u; i < 4u * {{ROWS}}u; i++) {
      let row = m0 + lid.y + 4u * i;
      if (row >= T) { break; }
      let o = row * N + col;
      var v = acc0;
      if (i == 1u) { v = acc1; }
      else if (i == 2u) { v = acc2; }
      else if (i == 3u) { v = acc3; }
//#if ROWS_GE_2
      else if (i == 4u) { v = acc4; }
      else if (i == 5u) { v = acc5; }
      else if (i == 6u) { v = acc6; }
      else if (i == 7u) { v = acc7; }
//#endif
//#if ROWS_GE_3
      else if (i == 8u) { v = acc8; }
      else if (i == 9u) { v = acc9; }
      else if (i == 10u) { v = acc10; }
      else if (i == 11u) { v = acc11; }
//#endif
//#if ROWS_GE_4
      else if (i == 12u) { v = acc12; }
      else if (i == 13u) { v = acc13; }
      else if (i == 14u) { v = acc14; }
      else if (i == 15u) { v = acc15; }
//#endif
      if (splits > 1u) {
        PART[wg.z * T * N + o] = v;
        continue;
      }
      v += bias;
      if (p.mode == 1u) { v += Y[o]; } else if (p.mode == 2u) { v = max(v, 0.0); }
      Y[o] = v;
    }
  }
//#else
  for (var gg = 0u; gg < {{GROUPS}}u; gg++) {
    for (var j = 0u; j < 4u; j++) {
      let col = n0 + lid.x + 16u * j + 64u * gg;
      if (col >= N) { break; }
      var bias = 0.0;
      if (p.bias == 1u) { bias = B[col]; }
      for (var i = 0u; i < {{ROWS}}u; i++) {
        let row = m0 + lid.y + 16u * i;
        if (row >= T) { break; }
        let o = row * N + col;
        // Separate the array and vector reads: nested indexing loses components on NVIDIA/Naga.
        let values = acc[i * {{GROUPS}}u + gg];
        let v0 = values[j];
        if (splits > 1u) {
          PART[wg.z * T * N + o] = v0;
          continue;
        }
        var v = v0 + bias;
        if (p.mode == 1u) { v += Y[o]; } else if (p.mode == 2u) { v = max(v, 0.0); }
        Y[o] = v;
      }
    }
  }
//#endif
}
