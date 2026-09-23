// Causal GQA attention, one workgroup per query token and head. Eight lanes cooperate on each
// key score, producing four 32-key score blocks per 128-key online-softmax tile. With 32-lane
// subgroups the tile max and sum reduce inside each subgroup first, so a reduction takes two
// barriers instead of eight.
//#if SUBGROUPS
enable subgroups;
//#endif
//#include kev_common

@group(0) @binding(1) var<storage, read> PROJ: array<f32>;
@group(0) @binding(2) var<storage, read> KV: array<f32>;
@group(0) @binding(3) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> segs: array<Seg>;
@group(0) @binding(5) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(6) var<storage, read> KEYS: array<f32>;
var<workgroup> q: array<vec4<f32>, 64>;
var<workgroup> sc: array<f32, 256>;
var<workgroup> dotp: array<f32, 256>;
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
  let kvh = h / (HEADS / KV_HEADS);
  if (d < 64u) {
    let b = t * ATTN_WIDTH + h * 512u + d * 4u;
    q[d] = vec4<f32>(PROJ[b], PROJ[b + 1u], PROJ[b + 2u], PROJ[b + 3u]);
  }
  workgroupBarrier();
  var m = -3.0e38;
  var lsum = 0.0;
  var o = 0.0;
  for (var j0 = 0u; j0 < n; j0 += 128u) {
    // A workgroup has 256 lanes, so eight lanes can cover one 256-dimensional key while
    // 32 keys are scored in parallel. Four such blocks form the softmax tile.
    for (var block = 0u; block < 4u; block++) {
      let lane = d % 8u;
      let key_slot = d / 8u;
      let j = j0 + block * 32u + key_slot;
      var acc = 0.0;
      if (j < n) {
        if (j < plen) {
          let key_base = (pstart + j) * ATTN_KV + kvh * 256u;
          for (var c = lane * 4u; c < 256u; c += 32u) {
            acc += dot(q[c >> 2u], vec4<f32>(
              KV[key_base + c], KV[key_base + c + 1u],
              KV[key_base + c + 2u], KV[key_base + c + 3u]));
          }
        } else {
          let kt = start + j - plen;
          for (var c = lane * 4u; c < 256u; c += 32u) {
            let key_base = (kvh * 256u + c) * g.T + kt;
            acc += dot(q[c >> 2u], vec4<f32>(
              KEYS[key_base], KEYS[key_base + g.T],
              KEYS[key_base + 2u * g.T], KEYS[key_base + 3u * g.T]));
          }
        }
      }
      dotp[d] = acc;
      workgroupBarrier();
      for (var stride = 4u; stride > 0u; stride >>= 1u) {
        if (lane < stride) { dotp[d] += dotp[d + stride]; }
        workgroupBarrier();
      }
      if (lane == 0u) {
        if (j < n) { sc[block * 32u + key_slot] = dotp[d] * 0.0625; }
        else { sc[block * 32u + key_slot] = -3.0e38; }
      }
      workgroupBarrier();
    }
    var tile_score = -3.0e38;
    if (d < 128u) { tile_score = sc[d]; }
    let tm = rmax(tile_score, d);
    let mn = max(m, tm);
    let corr = exp(m - mn);
    var pj = 0.0;
    if (d < 128u && j0 + d < n) { pj = exp(sc[d] - mn); }
    if (d < 128u) { sc[d] = pj; }
    let ts = rsum(pj, d);
    lsum = lsum * corr + ts;
    o = o * corr;
    let cnt = min(128u, n - j0);
    for (var jj = 0u; jj < cnt; jj++) {
      let key = j0 + jj;
      var vv: f32;
      if (key < plen) { vv = KV[(pstart + key) * ATTN_KV + ATTN_K + kvh * 256u + d]; } else { vv = PROJ[(start + key - plen) * ATTN_WIDTH + 2u * ATTN_Q + ATTN_K + kvh * 256u + d]; }
      o += sc[jj] * vv;
    }
    m = mn;
    workgroupBarrier();
  }
  let gate = PROJ[t * ATTN_WIDTH + h * 512u + 256u + d];
  OUT[t * ATTN_Q + h * 256u + d] = o / lsum / (1.0 + exp(-gate));
}
