// Causal GQA attention, one workgroup per query token and head, 256 threads (one key each per
// step). With 32-lane subgroups the per-step max and sum reduce inside each subgroup first, so a
// reduction takes two barriers instead of eight.
//#if SUBGROUPS
enable subgroups;
//#endif
//#include kev_common

@group(0) @binding(1) var<storage, read> PROJ: array<f32>;
@group(0) @binding(2) var<storage, read> KV: array<f32>;
@group(0) @binding(3) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> segs: array<Seg>;
@group(0) @binding(5) var<storage, read_write> OUT: array<f32>;
var<workgroup> q: array<vec4<f32>, 64>;
var<workgroup> sc: array<f32, 256>;
var<workgroup> red: array<f32, 256>;
var<workgroup> info: vec4<u32>;
var<workgroup> info2: vec4<u32>;

//#if SUBGROUPS
// each of the 8 subgroups reduces its 32 lanes, then every thread combines the 8 partial results
fn rmax(x: f32, l: u32) -> f32 {
  let part = subgroupMax(x);
  if (l % 32u == 0u) { red[l / 32u] = part; }
  workgroupBarrier();
  var r = red[0];
  for (var i = 1u; i < 8u; i++) { r = max(r, red[i]); }
  workgroupBarrier();
  return r;
}
fn rsum(x: f32, l: u32) -> f32 {
  let part = subgroupAdd(x);
  if (l % 32u == 0u) { red[l / 32u] = part; }
  workgroupBarrier();
  var r = 0.0;
  for (var i = 0u; i < 8u; i++) { r += red[i]; }
  workgroupBarrier();
  return r;
}
//#else
fn rmax(x: f32, l: u32) -> f32 {
  red[l] = x;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (l < k) { red[l] = max(red[l], red[l + k]); } workgroupBarrier(); }
  let r = red[0];
  workgroupBarrier();
  return r;
}
fn rsum(x: f32, l: u32) -> f32 {
  red[l] = x;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (l < k) { red[l] += red[l + k]; } workgroupBarrier(); }
  let r = red[0];
  workgroupBarrier();
  return r;
}
//#endif
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) d: u32) {
  let t = wg.x;
  let h = wg.y;
  if (t >= g.T) { return; }
  if (d == 0u) {
    let tk = tok[t];
    let sg = segs[tk.x];
    var plen = 0u;
    if (sg.parent != 0xffffffffu) { plen = sg.plen; }
    info = vec4<u32>(sg.start, tk.y, plen, sg.pstart);
  }
  let sp = workgroupUniformLoad(&info);
  let start = sp.x;
  let r = sp.y;
  let plen = sp.z;
  let pstart = sp.w;
  let n = plen + r + 1u;
  let kvh = h / 4u;
  if (d < 64u) {
    let b = t * 5120u + h * 512u + d * 4u;
    q[d] = vec4<f32>(PROJ[b], PROJ[b + 1u], PROJ[b + 2u], PROJ[b + 3u]);
  }
  workgroupBarrier();
  var m = -3.0e38;
  var lsum = 0.0;
  var o = 0.0;
  for (var j0 = 0u; j0 < n; j0 += 256u) {
    let j = j0 + d;
    var s = -3.0e38;
    if (j < n) {
      var kb: u32;
      if (j < plen) { kb = (pstart + j) * 1024u + kvh * 256u; } else { kb = (start + j - plen) * 5120u + 4096u + kvh * 256u; }
      var acc = vec4<f32>(0.0);
      if (j < plen) {
        for (var c = 0u; c < 64u; c++) { acc += q[c] * vec4<f32>(KV[kb + c * 4u], KV[kb + c * 4u + 1u], KV[kb + c * 4u + 2u], KV[kb + c * 4u + 3u]); }
      } else {
        for (var c = 0u; c < 64u; c++) { acc += q[c] * vec4<f32>(PROJ[kb + c * 4u], PROJ[kb + c * 4u + 1u], PROJ[kb + c * 4u + 2u], PROJ[kb + c * 4u + 3u]); }
      }
      s = (acc.x + acc.y + acc.z + acc.w) * 0.0625;
    }
    let tm = rmax(s, d);
    let mn = max(m, tm);
    let corr = exp(m - mn);
    var pj = 0.0;
    if (j < n) { pj = exp(s - mn); }
    sc[d] = pj;
    let ts = rsum(pj, d);
    lsum = lsum * corr + ts;
    o = o * corr;
    let cnt = min(256u, n - j0);
    for (var jj = 0u; jj < cnt; jj++) {
      let key = j0 + jj;
      var vv: f32;
      if (key < plen) { vv = KV[(pstart + key) * 1024u + 512u + kvh * 256u + d]; } else { vv = PROJ[(start + key - plen) * 5120u + 4608u + kvh * 256u + d]; }
      o += sc[jj] * vv;
    }
    m = mn;
    workgroupBarrier();
  }
  let gate = PROJ[t * 5120u + h * 512u + 256u + d];
  OUT[t * 2048u + h * 256u + d] = o / lsum / (1.0 + exp(-gate));
}
