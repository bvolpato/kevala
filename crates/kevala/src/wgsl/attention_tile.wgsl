// Laya attention with register tiles, for GPUs with subgroups of 16 or more lanes and f16
// (FlashAttention-2's work split, without matrix units). A workgroup of 256 threads takes 64
// queries of one head and walks the keys 32 at a time. Thread t owns queries 4 (t / 16) .. + 3:
// first it scores them against keys 2 (t % 16) and 2 (t % 16) + 1, then it accumulates output
// dimensions 4 (t % 16) .. + 3 for them. The 16 threads sharing those queries are consecutive
// lanes of one subgroup, so the running max and sum of each query come from lane shuffles and
// stay in registers, and so does the output. Q, K and V tiles are f16 in workgroup memory (the
// math is f32); rows are padded so the lanes of a subgroup read different banks.
enable f16;
enable subgroups;

//#include common

struct P { width: u32, stride: u32, window: u32, _a: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> QKV: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> blocks: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> CTX: array<vec4<f32>>;

const QROW = 17u; // vec4s per query row of qs (16 and a pad)
const KROW = 17u; // vec4s per key row of ks
const PROW = 33u; // floats per query row of ps
var<workgroup> qs: array<vec4<f16>, 1088>; // [64 queries][QROW]
var<workgroup> ks: array<vec4<f16>, 544>;  // [32 keys][KROW]
var<workgroup> vs: array<vec4<f16>, 512>;  // [32 keys][16]
var<workgroup> ps: array<f32, 2112>;       // [64 queries][PROW]: probabilities of this step
var<workgroup> info: vec4<u32>;

// max and sum over the 16 lanes that share a query
fn row_max(x: f32) -> f32 {
  var m = max(x, subgroupShuffleXor(x, 1u));
  m = max(m, subgroupShuffleXor(m, 2u));
  m = max(m, subgroupShuffleXor(m, 4u));
  return max(m, subgroupShuffleXor(m, 8u));
}
fn row_sum(x: f32) -> f32 {
  var s = x + subgroupShuffleXor(x, 1u);
  s += subgroupShuffleXor(s, 2u);
  s += subgroupShuffleXor(s, 4u);
  return s + subgroupShuffleXor(s, 8u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let h = wg.y;
  if (wg.x >= g.S) { return; }
  if (t == 0u) { info = blocks[wg.x]; }
  let bi = workgroupUniformLoad(&info);
  let s0 = bi.x; // segment start (token index)
  let L = bi.y;  // segment length
  let q0 = bi.z; // first query of the block, within the segment
  let nq = min(64u, L - q0);
  let w = p.window;
  var lo = 0u;
  var hi = L;
  if (w > 0u) {
    if (q0 > w) { lo = q0 - w; }
    hi = min(L, q0 + nq - 1u + w + 1u);
  }
  let rs = p.stride / 4u; // vec4s per token row of QKV
  let hv = h * 16u;       // this head's first vec4 in a row
  for (var e = t; e < 1024u; e += 256u) {
    let qi = e / 16u;
    var v = vec4<f32>(0.0);
    if (qi < nq) { v = QKV[(s0 + q0 + qi) * rs + hv + e % 16u]; }
    qs[qi * QROW + e % 16u] = vec4<f16>(v);
  }

  let tq = t / 16u; // queries 4 tq .. 4 tq + 3
  let tk = t % 16u; // keys 2 tk, 2 tk + 1 when scoring; dimensions 4 tk .. + 3 when accumulating
  var mx = vec4<f32>(-3.0e38);
  var sum = vec4<f32>(0.0);
  var o: array<vec4<f32>, 4>;
  let kv = p.width / 4u;
  for (var j0 = lo; j0 < hi; j0 += 32u) {
    workgroupBarrier();
    for (var e = t; e < 512u; e += 256u) {
      let kj = e / 16u;
      let d = e % 16u;
      var kk = vec4<f32>(0.0);
      var vv = vec4<f32>(0.0);
      if (j0 + kj < hi) {
        let row = (s0 + j0 + kj) * rs + hv + d;
        kk = QKV[row + kv];
        vv = QKV[row + 2u * kv];
      }
      ks[kj * KROW + d] = vec4<f16>(kk);
      vs[e] = vec4<f16>(vv);
    }
    workgroupBarrier();

    // scores of 4 queries x 2 keys
    var s: array<vec2<f32>, 4>;
    for (var d = 0u; d < 16u; d++) {
      let k0 = vec4<f32>(ks[(2u * tk) * KROW + d]);
      let k1 = vec4<f32>(ks[(2u * tk + 1u) * KROW + d]);
      for (var c = 0u; c < 4u; c++) {
        let q = vec4<f32>(qs[(4u * tq + c) * QROW + d]);
        s[c] += vec2<f32>(dot(q, k0), dot(q, k1));
      }
    }
    // online softmax: every lane of a query's 16 gets the same statistics
    var scale = vec4<f32>(1.0);
    for (var c = 0u; c < 4u; c++) {
      let i = q0 + 4u * tq + c;
      let j = j0 + 2u * tk;
      let v0 = 4u * tq + c < nq && j < hi && (w == 0u || max(i, j) - min(i, j) <= w);
      let v1 = 4u * tq + c < nq && j + 1u < hi && (w == 0u || max(i, j + 1u) - min(i, j + 1u) <= w);
      let a = select(-3.0e38, s[c].x * 0.125, v0);
      let b = select(-3.0e38, s[c].y * 0.125, v1);
      let top = max(mx[c], row_max(max(a, b)));
      let ea = select(0.0, exp(a - top), v0);
      let eb = select(0.0, exp(b - top), v1);
      scale[c] = exp(mx[c] - top);
      sum[c] = sum[c] * scale[c] + row_sum(ea + eb);
      mx[c] = top;
      ps[(4u * tq + c) * PROW + 2u * tk] = ea;
      ps[(4u * tq + c) * PROW + 2u * tk + 1u] = eb;
    }
    for (var c = 0u; c < 4u; c++) { o[c] *= scale[c]; }
    workgroupBarrier();

    // output: 4 queries x 4 dimensions
    for (var k = 0u; k < 32u; k++) {
      let v = vec4<f32>(vs[k * 16u + tk]);
      for (var c = 0u; c < 4u; c++) { o[c] += ps[(4u * tq + c) * PROW + k] * v; }
    }
  }

  for (var c = 0u; c < 4u; c++) {
    let qi = 4u * tq + c;
    if (qi < nq) { CTX[(s0 + q0 + qi) * kv + hv + tk] = o[c] / sum[c]; }
  }
}
