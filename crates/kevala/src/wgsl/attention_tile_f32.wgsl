// Laya FP32 attention with register tiles. A workgroup of 256 threads takes 64 queries of one
// head and walks the keys 16 at a time. Each group of 16 lanes owns four queries: it scores those
// queries against one key per lane, then accumulates four output dimensions per lane. The subgroup
// path uses subgroup IDs to make that logical mapping independent of implementation-specific local
// lane numbering. The same 30,224-byte tile also has a workgroup reduction fallback for devices or
// probes that cannot use subgroup IDs; the probability table is reused as reduction scratch before
// each row is stored.
//#if SUBGROUPS
enable subgroups;
requires subgroup_id;
//#endif

//#include common

struct P { width: u32, stride: u32, window: u32, _a: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> QKV: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> blocks: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> CTX: array<vec4<f32>>;

const QROW = 17u; // vec4s per query row of qs (16 and a pad)
const KROW = 17u; // vec4s per key row of ks
const PROW = 17u; // floats per query row of ps (16 keys and a pad)
var<workgroup> qs: array<vec4<f32>, 1088>; // [64 queries][QROW]
var<workgroup> ks: array<vec4<f32>, 272>;  // [16 keys][KROW]
var<workgroup> vs: array<vec4<f32>, 256>;  // [16 keys][16]
var<workgroup> ps: array<f32, 1088>;       // [64 queries][PROW]: probabilities of this step
var<workgroup> info: vec4<u32>;

// A group of 16 lanes uses one probability row as a four-stage workgroup reduction. Every
// invocation calls these barriers, including lanes in the other query groups, so the control flow
// remains uniform. The final barrier lets all callers read the broadcast result before the row is
// reused for probabilities.
fn shared_row_max(x: f32, qi: u32, tk: u32) -> f32 {
  let base = qi * PROW;
  ps[base + tk] = x;
  workgroupBarrier();
  if (tk < 8u) { ps[base + tk] = max(ps[base + tk], ps[base + tk + 8u]); }
  workgroupBarrier();
  if (tk < 4u) { ps[base + tk] = max(ps[base + tk], ps[base + tk + 4u]); }
  workgroupBarrier();
  if (tk < 2u) { ps[base + tk] = max(ps[base + tk], ps[base + tk + 2u]); }
  workgroupBarrier();
  if (tk < 1u) { ps[base] = max(ps[base], ps[base + 1u]); }
  workgroupBarrier();
  let result = ps[base];
  workgroupBarrier();
  return result;
}

fn shared_row_sum(x: f32, qi: u32, tk: u32) -> f32 {
  let base = qi * PROW;
  ps[base + tk] = x;
  workgroupBarrier();
  if (tk < 8u) { ps[base + tk] += ps[base + tk + 8u]; }
  workgroupBarrier();
  if (tk < 4u) { ps[base + tk] += ps[base + tk + 4u]; }
  workgroupBarrier();
  if (tk < 2u) { ps[base + tk] += ps[base + tk + 2u]; }
  workgroupBarrier();
  if (tk < 1u) { ps[base] += ps[base + 1u]; }
  workgroupBarrier();
  let result = ps[base];
  workgroupBarrier();
  return result;
}

//#if SUBGROUPS
// The 16 lanes that share a query combine row statistics. With 32-lane subgroups, XOR offsets
// 1..8 stay within either half of the subgroup.
fn row_max_subgroup(x: f32) -> f32 {
  var m = max(x, subgroupShuffleXor(x, 1u));
  m = max(m, subgroupShuffleXor(m, 2u));
  m = max(m, subgroupShuffleXor(m, 4u));
  return max(m, subgroupShuffleXor(m, 8u));
}
fn row_sum_subgroup(x: f32) -> f32 {
  var s = x + subgroupShuffleXor(x, 1u);
  s += subgroupShuffleXor(s, 2u);
  s += subgroupShuffleXor(s, 4u);
  return s + subgroupShuffleXor(s, 8u);
}
//#endif

fn row_max(x: f32, qi: u32, tk: u32, fast: bool) -> f32 {
  //#if SUBGROUPS
  if (fast) { return row_max_subgroup(x); }
  //#endif
  return shared_row_max(x, qi, tk);
}
fn row_sum(x: f32, qi: u32, tk: u32, fast: bool) -> f32 {
  //#if SUBGROUPS
  if (fast) { return row_sum_subgroup(x); }
  //#endif
  return shared_row_sum(x, qi, tk);
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_index) local_t: u32
//#if SUBGROUPS
  , @builtin(subgroup_id) sgid: u32
  , @builtin(subgroup_invocation_id) lane: u32
  , @builtin(num_subgroups) nsg: u32
//#endif
) {
  let h = wg.y;
  if (wg.x >= g.S) { return; }
  //#if SUBGROUPS
  let fast = nsg == 8u;
  let t = select(local_t, sgid * 32u + lane, fast);
  //#else
  let fast = false;
  let t = local_t;
  //#endif
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
    qs[qi * QROW + e % 16u] = v;
  }

  let tq = t / 16u; // queries 4 tq .. 4 tq + 3
  let tk = t % 16u; // one key per lane; dimensions 4 tk .. + 3 for output
  var mx = vec4<f32>(-3.0e38);
  var sum = vec4<f32>(0.0);
  var o: array<vec4<f32>, 4>;
  for (var c = 0u; c < 4u; c++) { o[c] = vec4<f32>(0.0); }
  let kv = p.width / 4u;
  for (var j0 = lo; j0 < hi; j0 += 16u) {
    workgroupBarrier();
    for (var e = t; e < 256u; e += 256u) {
      let kj = e / 16u;
      let d = e % 16u;
      var kk = vec4<f32>(0.0);
      var vv = vec4<f32>(0.0);
      if (j0 + kj < hi) {
        let row = (s0 + j0 + kj) * rs + hv + d;
        kk = QKV[row + kv];
        vv = QKV[row + 2u * kv];
      }
      ks[kj * KROW + d] = kk;
      vs[e] = vv;
    }
    workgroupBarrier();

    // scores of 4 queries x 1 key
    var s: array<f32, 4>;
    for (var c = 0u; c < 4u; c++) { s[c] = 0.0; }
    for (var d = 0u; d < 16u; d++) {
      let k = vec4<f32>(ks[tk * KROW + d]);
      for (var c = 0u; c < 4u; c++) {
        let q = qs[(4u * tq + c) * QROW + d];
        s[c] += dot(q, k);
      }
    }

    // Online softmax: every lane of a query's 16 gets the same running max and sum.
    var top: vec4<f32>;
    var ea: vec4<f32>;
    let j = j0 + tk;
    for (var c = 0u; c < 4u; c++) {
      let i = q0 + 4u * tq + c;
      let valid = 4u * tq + c < nq && j < hi && (w == 0u || max(i, j) - min(i, j) <= w);
      ea[c] = select(-3.0e38, s[c] * 0.125, valid);
      top[c] = max(mx[c], row_max(ea[c], 4u * tq + c, tk, fast));
    }
    for (var c = 0u; c < 4u; c++) {
      let pa = select(0.0, exp(ea[c] - top[c]), ea[c] > -1.0e38);
      let scale = exp(mx[c] - top[c]);
      sum[c] = sum[c] * scale + row_sum(pa, 4u * tq + c, tk, fast);
      mx[c] = top[c];
      ps[(4u * tq + c) * PROW + tk] = pa;
      o[c] *= scale;
    }
    workgroupBarrier();

    // output: 4 queries x 4 dimensions
    for (var k = 0u; k < 16u; k++) {
      let v = vec4<f32>(vs[k * 16u + tk]);
      for (var c = 0u; c < 4u; c++) { o[c] += ps[(4u * tq + c) * PROW + k] * v; }
    }
  }

  for (var c = 0u; c < 4u; c++) {
    let qi = 4u * tq + c;
    if (qi < nq) { CTX[(s0 + q0 + qi) * kv + hv + tk] = o[c] / sum[c]; }
  }
}
