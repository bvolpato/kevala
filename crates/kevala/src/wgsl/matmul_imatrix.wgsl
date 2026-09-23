// Experimental Chrome 149 integer subgroup matrices, M16/N16/K32, one 32-lane subgroup.
// API pinned to Dawn 54b4153cfef88e048f365f99b962478f0087dfe8:
// https://dawn.googlesource.com/dawn/+/54b4153cfef88e048f365f99b962478f0087dfe8/src/tint/lang/wgsl/wgsl.def
enable chromium_experimental_subgroup_matrix;
//#include common

struct P { N: u32, K: u32, mode: u32, bias: u32 }
struct XScales { p0: f32, p1: f32, p2: f32 }
struct XPlanes { p0: u32, p1: u32, p2: u32 }
struct QBlock { scales: XScales, planes: array<XPlanes, 8> }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> Q: array<QBlock>;
@group(0) @binding(3) var<storage, read> W: array<u32>;
@group(0) @binding(4) var<storage, read> S: array<f32>;
@group(0) @binding(5) var<storage, read> B: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
@group(0) @binding(7) var<storage, read_write> PART: array<f32>;

// i8 operands reinterpret four packed signed bytes in each i32 word. This Dawn revision
// counts load offsets and strides in i8 elements, so a 32-value row has stride 32, not 8.
var<workgroup> a0: array<i32, 128>;
var<workgroup> a1: array<i32, 128>;
var<workgroup> a2: array<i32, 128>;
var<workgroup> weights: array<i32, 128>;
var<workgroup> xscale: array<XScales, 16>;
var<workgroup> wscale: array<f32, 16>;
var<workgroup> out0: array<i32, 256>;
var<workgroup> out1: array<i32, 256>;
var<workgroup> out2: array<i32, 256>;

// Match the generic R4/G1 split count and reduction order for the same target. The matrix
// dispatch uses 16x16 tiles; the occupancy estimate deliberately stays at the baseline's 64x64.
//#include splits

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let T = g.T;
  let N = p.N;
  let K = p.K;
  let nb = K / 32u;
  let m0 = wg.y * 16u;
  let n0 = wg.x * 16u;
  let splits = mm_splits(T, N, K);
  let per = (nb + splits - 1u) / splits;
  let kb0 = wg.z * per;
  let kb1 = min(nb, kb0 + per);
  // Lane li owns eight outputs li + 32*i, so stores from adjacent lanes coalesce.
  var acc_lo = vec4<f32>(0.0);
  var acc_hi = vec4<f32>(0.0);
  for (var kb = kb0; kb < kb1; kb++) {
    for (var part = 0u; part < 4u; part++) {
      let word = li + part * 32u;
      let row = word / 8u;
      let kq = word & 7u;
      var plane = XPlanes(0u, 0u, 0u);
      var scale = XScales(0.0, 0.0, 0.0);
      if (m0 + row < T) {
        let block = (m0 + row) * nb + kb;
        plane = Q[block].planes[kq];
        scale = Q[block].scales;
      }
      a0[word] = bitcast<i32>(plane.p0);
      a1[word] = bitcast<i32>(plane.p1);
      a2[word] = bitcast<i32>(plane.p2);
      if (kq == 0u) { xscale[row] = scale; }
      var packed_weight = 0u;
      var weight_scale = 0.0;
      if (n0 + row < N) {
        packed_weight = W[((n0 + row) * K + kb * 32u) / 4u + kq];
        weight_scale = S[(n0 + row) * nb + kb];
      }
      weights[word] = bitcast<i32>(packed_weight);
      if (kq == 0u) { wscale[row] = weight_scale; }
    }
    workgroupBarrier();
    // WGSL matrix dimensions are columns then rows. W[N,K] is B[K,N] in column-major form.
    let lhs0 = subgroupMatrixLoad<subgroup_matrix_left<i8, 32, 16>>(&a0, 0u, false, 32u);
    let lhs1 = subgroupMatrixLoad<subgroup_matrix_left<i8, 32, 16>>(&a1, 0u, false, 32u);
    let lhs2 = subgroupMatrixLoad<subgroup_matrix_left<i8, 32, 16>>(&a2, 0u, false, 32u);
    let rhs = subgroupMatrixLoad<subgroup_matrix_right<i8, 16, 32>>(&weights, 0u, true, 32u);
    // Fresh multiply results explicitly start from zero for every K32 block; no integer
    // accumulator survives into a block with different activation or weight scales.
    let result0 = subgroupMatrixMultiply<i32>(lhs0, rhs);
    let result1 = subgroupMatrixMultiply<i32>(lhs1, rhs);
    let result2 = subgroupMatrixMultiply<i32>(lhs2, rhs);
    subgroupMatrixStore(&out0, 0u, result0, false, 16u);
    subgroupMatrixStore(&out1, 0u, result1, false, 16u);
    subgroupMatrixStore(&out2, 0u, result2, false, 16u);
    workgroupBarrier();
    for (var i = 0u; i < 4u; i++) {
      let o0 = li + i * 32u;
      let o1 = o0 + 128u;
      let s0 = xscale[o0 / 16u];
      let s1 = xscale[o1 / 16u];
      let sw = wscale[li % 16u];
      acc_lo[i] += (f32(out0[o0]) * s0.p0 + f32(out1[o0]) * s0.p1 + f32(out2[o0]) * s0.p2) * sw;
      acc_hi[i] += (f32(out0[o1]) * s1.p0 + f32(out1[o1]) * s1.p1 + f32(out2[o1]) * s1.p2) * sw;
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 8u; i++) {
    let local = li + i * 32u;
    let row = m0 + local / 16u;
    let col = n0 + local % 16u;
    if (row >= T || col >= N) { continue; }
    let o = row * N + col;
    var v = select(acc_lo[i % 4u], acc_hi[i % 4u], i >= 4u);
    if (splits > 1u) {
      PART[wg.z * T * N + o] = v;
      continue;
    }
    if (p.bias == 1u) { v += B[col]; }
    if (p.mode == 1u) { v += Y[o]; } else if (p.mode == 2u) { v = max(v, 0.0); }
    Y[o] = v;
  }
}
