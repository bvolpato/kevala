// Blocked attention: one workgroup per (block of 16 queries of one segment, head). Keys and
// values stream through workgroup memory 16 at a time and serve all 16 queries, with an online
// softmax per query, so K and V are read once per block instead of once per query.
//#include common

struct P { width: u32, stride: u32, window: u32, _a: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> QKV: array<f32>;
@group(0) @binding(3) var<storage, read> blocks: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> CTX: array<f32>;
var<workgroup> qs: array<f32, 1040>;  // [16 queries][65], padded to spread rows across memory banks
var<workgroup> ks: array<f32, 1040>;  // [16 keys][65]
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
    qs[qi * 65u + d] = v;
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
      ks[kj * 65u + d] = kv;
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
        var acc = vec4<f32>(0.0);
        for (var d = 0u; d < 64u; d += 4u) {
          let q = qi * 65u + d;
          let k = kj * 65u + d;
          acc += vec4<f32>(qs[q], qs[q + 1u], qs[q + 2u], qs[q + 3u])
            * vec4<f32>(ks[k], ks[k + 1u], ks[k + 2u], ks[k + 3u]);
        }
        sc = (acc.x + acc.y + acc.z + acc.w) * 0.125;
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
