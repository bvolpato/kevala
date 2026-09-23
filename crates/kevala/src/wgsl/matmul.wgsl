// Y[T, N] = X[T, K] . W[N, K]^T (+ bias), W int8 in u32 words with one f32 scale per 32 weights.
//
//#if GENERIC_UNROLL
//#if ROWS_GE_4
// A workgroup of 32 x 8 threads covers 64 rows and 64 columns. Each thread owns
// rows y + 8 i and columns x and x + 32, with eight named vec2 FP32 accumulators.
//#else
// A workgroup of 32 x 8 threads covers {{BM}} rows and 64 columns. Each thread owns
// rows y + 8 i for i < 2 * {{ROWS}}, and columns x and x + 32, in named vec2 accumulators.
//#endif
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
@compute @workgroup_size(32, 8)
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
  var acc0 = vec2<f32>(0.0);
  var acc1 = vec2<f32>(0.0);
//#if ROWS_GE_2
  var acc2 = vec2<f32>(0.0);
  var acc3 = vec2<f32>(0.0);
//#endif
//#if ROWS_GE_3
  var acc4 = vec2<f32>(0.0);
  var acc5 = vec2<f32>(0.0);
//#endif
//#if ROWS_GE_4
  var acc6 = vec2<f32>(0.0);
  var acc7 = vec2<f32>(0.0);
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
    // Keep the eight K/4 steps explicit, reusing two weight vectors across eight rows.
    let c = lid.x;
    {
      let kq = 0u;
      let b0 = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 32u) * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += vec2<f32>(dot(a0, b0), dot(a0, b1));
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc1 += vec2<f32>(dot(a1, b0), dot(a1, b1));
//#if ROWS_GE_2
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc2 += vec2<f32>(dot(a2, b0), dot(a2, b1));
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc3 += vec2<f32>(dot(a3, b0), dot(a3, b1));
//#endif
//#if ROWS_GE_3
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc4 += vec2<f32>(dot(a4, b0), dot(a4, b1));
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc5 += vec2<f32>(dot(a5, b0), dot(a5, b1));
//#endif
//#if ROWS_GE_4
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc6 += vec2<f32>(dot(a6, b0), dot(a6, b1));
      let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc7 += vec2<f32>(dot(a7, b0), dot(a7, b1));
//#endif
    }
    {
      let kq = 1u;
      let b0 = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 32u) * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += vec2<f32>(dot(a0, b0), dot(a0, b1));
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc1 += vec2<f32>(dot(a1, b0), dot(a1, b1));
//#if ROWS_GE_2
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc2 += vec2<f32>(dot(a2, b0), dot(a2, b1));
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc3 += vec2<f32>(dot(a3, b0), dot(a3, b1));
//#endif
//#if ROWS_GE_3
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc4 += vec2<f32>(dot(a4, b0), dot(a4, b1));
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc5 += vec2<f32>(dot(a5, b0), dot(a5, b1));
//#endif
//#if ROWS_GE_4
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc6 += vec2<f32>(dot(a6, b0), dot(a6, b1));
      let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc7 += vec2<f32>(dot(a7, b0), dot(a7, b1));
//#endif
    }
    {
      let kq = 2u;
      let b0 = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 32u) * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += vec2<f32>(dot(a0, b0), dot(a0, b1));
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc1 += vec2<f32>(dot(a1, b0), dot(a1, b1));
//#if ROWS_GE_2
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc2 += vec2<f32>(dot(a2, b0), dot(a2, b1));
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc3 += vec2<f32>(dot(a3, b0), dot(a3, b1));
//#endif
//#if ROWS_GE_3
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc4 += vec2<f32>(dot(a4, b0), dot(a4, b1));
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc5 += vec2<f32>(dot(a5, b0), dot(a5, b1));
//#endif
//#if ROWS_GE_4
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc6 += vec2<f32>(dot(a6, b0), dot(a6, b1));
      let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc7 += vec2<f32>(dot(a7, b0), dot(a7, b1));
//#endif
    }
    {
      let kq = 3u;
      let b0 = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 32u) * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += vec2<f32>(dot(a0, b0), dot(a0, b1));
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc1 += vec2<f32>(dot(a1, b0), dot(a1, b1));
//#if ROWS_GE_2
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc2 += vec2<f32>(dot(a2, b0), dot(a2, b1));
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc3 += vec2<f32>(dot(a3, b0), dot(a3, b1));
//#endif
//#if ROWS_GE_3
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc4 += vec2<f32>(dot(a4, b0), dot(a4, b1));
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc5 += vec2<f32>(dot(a5, b0), dot(a5, b1));
//#endif
//#if ROWS_GE_4
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc6 += vec2<f32>(dot(a6, b0), dot(a6, b1));
      let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc7 += vec2<f32>(dot(a7, b0), dot(a7, b1));
//#endif
    }
    {
      let kq = 4u;
      let b0 = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 32u) * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += vec2<f32>(dot(a0, b0), dot(a0, b1));
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc1 += vec2<f32>(dot(a1, b0), dot(a1, b1));
//#if ROWS_GE_2
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc2 += vec2<f32>(dot(a2, b0), dot(a2, b1));
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc3 += vec2<f32>(dot(a3, b0), dot(a3, b1));
//#endif
//#if ROWS_GE_3
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc4 += vec2<f32>(dot(a4, b0), dot(a4, b1));
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc5 += vec2<f32>(dot(a5, b0), dot(a5, b1));
//#endif
//#if ROWS_GE_4
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc6 += vec2<f32>(dot(a6, b0), dot(a6, b1));
      let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc7 += vec2<f32>(dot(a7, b0), dot(a7, b1));
//#endif
    }
    {
      let kq = 5u;
      let b0 = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 32u) * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += vec2<f32>(dot(a0, b0), dot(a0, b1));
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc1 += vec2<f32>(dot(a1, b0), dot(a1, b1));
//#if ROWS_GE_2
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc2 += vec2<f32>(dot(a2, b0), dot(a2, b1));
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc3 += vec2<f32>(dot(a3, b0), dot(a3, b1));
//#endif
//#if ROWS_GE_3
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc4 += vec2<f32>(dot(a4, b0), dot(a4, b1));
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc5 += vec2<f32>(dot(a5, b0), dot(a5, b1));
//#endif
//#if ROWS_GE_4
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc6 += vec2<f32>(dot(a6, b0), dot(a6, b1));
      let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc7 += vec2<f32>(dot(a7, b0), dot(a7, b1));
//#endif
    }
    {
      let kq = 6u;
      let b0 = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 32u) * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += vec2<f32>(dot(a0, b0), dot(a0, b1));
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc1 += vec2<f32>(dot(a1, b0), dot(a1, b1));
//#if ROWS_GE_2
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc2 += vec2<f32>(dot(a2, b0), dot(a2, b1));
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc3 += vec2<f32>(dot(a3, b0), dot(a3, b1));
//#endif
//#if ROWS_GE_3
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc4 += vec2<f32>(dot(a4, b0), dot(a4, b1));
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc5 += vec2<f32>(dot(a5, b0), dot(a5, b1));
//#endif
//#if ROWS_GE_4
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc6 += vec2<f32>(dot(a6, b0), dot(a6, b1));
      let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc7 += vec2<f32>(dot(a7, b0), dot(a7, b1));
//#endif
    }
    {
      let kq = 7u;
      let b0 = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 32u) * 8u + (kq ^ nx)]);
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 8u + kq]);
      acc0 += vec2<f32>(dot(a0, b0), dot(a0, b1));
      let a1 = vec4<f32>(xs[(lid.y + 8u) * 8u + kq]);
      acc1 += vec2<f32>(dot(a1, b0), dot(a1, b1));
//#if ROWS_GE_2
      let a2 = vec4<f32>(xs[(lid.y + 16u) * 8u + kq]);
      acc2 += vec2<f32>(dot(a2, b0), dot(a2, b1));
      let a3 = vec4<f32>(xs[(lid.y + 24u) * 8u + kq]);
      acc3 += vec2<f32>(dot(a3, b0), dot(a3, b1));
//#endif
//#if ROWS_GE_3
      let a4 = vec4<f32>(xs[(lid.y + 32u) * 8u + kq]);
      acc4 += vec2<f32>(dot(a4, b0), dot(a4, b1));
      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);
      acc5 += vec2<f32>(dot(a5, b0), dot(a5, b1));
//#endif
//#if ROWS_GE_4
      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);
      acc6 += vec2<f32>(dot(a6, b0), dot(a6, b1));
      let a7 = vec4<f32>(xs[(lid.y + 56u) * 8u + kq]);
      acc7 += vec2<f32>(dot(a7, b0), dot(a7, b1));
//#endif
    }
//#else
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
  for (var j = 0u; j < 2u; j++) {
    let col = n0 + lid.x + 32u * j;
    if (col >= N) { break; }
    var bias = 0.0;
    if (p.bias == 1u) { bias = B[col]; }
    let v0 = acc0[j];
    let v1 = acc1[j];
//#if ROWS_GE_2
    let v2 = acc2[j];
    let v3 = acc3[j];
//#endif
//#if ROWS_GE_3
    let v4 = acc4[j];
    let v5 = acc5[j];
//#endif
//#if ROWS_GE_4
    let v6 = acc6[j];
    let v7 = acc7[j];
//#endif
//#if ROWS_GE_4
    for (var i = 0u; i < 8u; i++) {
//#else
    for (var i = 0u; i < 2u * {{ROWS}}u; i++) {
//#endif
      let row = m0 + lid.y + 8u * i;
      if (row >= T) { break; }
      let o = row * N + col;
      var v = v0;
      if (i == 1u) { v = v1; }
//#if ROWS_GE_2
      else if (i == 2u) { v = v2; }
      else if (i == 3u) { v = v3; }
//#endif
//#if ROWS_GE_3
      else if (i == 4u) { v = v4; }
      else if (i == 5u) { v = v5; }
//#endif
//#if ROWS_GE_4
      else if (i == 6u) { v = v6; }
      else if (i == 7u) { v = v7; }
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
