// The same attention with 32 keys per step and 128 threads, for GPUs whose subgroups are exactly
// 32 lanes wide. Scores are laid out one key per lane, so each query's running max and sum come
// from subgroupMax and subgroupAdd instead of a serial loop; the K tile is stored transposed so the
// 32 lanes read 32 consecutive keys; and a step needs three barriers for 32 keys instead of four
// for 16.
enable subgroups;

//#include common

struct P { width: u32, stride: u32, window: u32, _a: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> QKV: array<f32>;
@group(0) @binding(3) var<storage, read> blocks: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> CTX: array<f32>;
var<workgroup> qs: array<f32, 1024>; // [16 queries][64 dims]
var<workgroup> kt: array<f32, 2048>; // [64 dims][32 keys]
var<workgroup> vs: array<f32, 2048>; // [32 keys][64 dims]
var<workgroup> ps: array<f32, 512>;  // [16 queries][32 keys]: exp(score - running max)
var<workgroup> scale: array<f32, 16>; // per query: the rescale factor this step, then the final sum
var<workgroup> info: vec4<u32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let h = wg.y;
  if (wg.x >= g.S) { return; }
  if (t == 0u) { info = blocks[wg.x]; }
  let bi = workgroupUniformLoad(&info);
  let s0 = bi.x; // segment start (token index)
  let L = bi.y;  // segment length
  let q0 = bi.z; // first query of the block, within the segment
  let nq = min(16u, L - q0);
  let w = p.window;
  var lo = 0u;
  var hi = L;
  if (w > 0u) {
    if (q0 > w) { lo = q0 - w; }
    hi = min(L, q0 + nq - 1u + w + 1u);
  }
  for (var e = t; e < 1024u; e += 128u) {
    let qi = e / 64u;
    var v = 0.0;
    if (qi < nq) { v = QKV[(s0 + q0 + qi) * p.stride + h * 64u + e % 64u]; }
    qs[e] = v;
  }

  // scores and softmax: key "lane" of the step, for queries 4 qg .. 4 qg + 3
  let lane = t % 32u;
  let qg = t / 32u;
  // output: dimension "dim", for queries 8 qh .. 8 qh + 7
  let dim = t % 64u;
  let qh = t / 64u;
  var mx: array<f32, 4>;
  var sum: array<f32, 4>;
  for (var c = 0u; c < 4u; c++) {
    mx[c] = -3.0e38;
    sum[c] = 0.0;
  }
  var o: array<f32, 8>;
  for (var c = 0u; c < 8u; c++) { o[c] = 0.0; }

  for (var j0 = lo; j0 < hi; j0 += 32u) {
    workgroupBarrier();
    for (var e = t; e < 2048u; e += 128u) {
      let kj = e / 64u;
      let d = e % 64u;
      var kv = 0.0;
      var vv = 0.0;
      if (j0 + kj < hi) {
        let row = (s0 + j0 + kj) * p.stride + h * 64u + d;
        kv = QKV[row + p.width];
        vv = QKV[row + 2u * p.width];
      }
      kt[d * 32u + kj] = kv;
      vs[e] = vv;
    }
    workgroupBarrier();
    let j = j0 + lane;
    for (var c = 0u; c < 4u; c++) {
      let qi = qg * 4u + c;
      let i = q0 + qi;
      let valid = qi < nq && j < hi && (w == 0u || max(i, j) - min(i, j) <= w);
      var dot = 0.0;
      for (var d = 0u; d < 64u; d++) { dot += qs[qi * 64u + d] * kt[d * 32u + lane]; }
      let score = select(-3.0e38, dot * 0.125, valid);
      let top = max(mx[c], subgroupMax(score));
      let e = select(0.0, exp(score - top), valid);
      let rescale = exp(mx[c] - top);
      sum[c] = sum[c] * rescale + subgroupAdd(e);
      mx[c] = top;
      ps[qi * 32u + lane] = e;
      if (lane == 0u) { scale[qi] = rescale; }
    }
    workgroupBarrier();
    for (var c = 0u; c < 8u; c++) {
      let qi = qh * 8u + c;
      var acc = o[c] * scale[qi];
      for (var kj = 0u; kj < 32u; kj++) { acc += ps[qi * 32u + kj] * vs[kj * 64u + dim]; }
      o[c] = acc;
    }
  }

  workgroupBarrier();
  if (lane == 0u) {
    for (var c = 0u; c < 4u; c++) { scale[qg * 4u + c] = sum[c]; }
  }
  workgroupBarrier();
  for (var c = 0u; c < 8u; c++) {
    let qi = qh * 8u + c;
    if (qi < nq) { CTX[(s0 + q0 + qi) * p.width + h * 64u + dim] = o[c] / scale[qi]; }
  }
}
