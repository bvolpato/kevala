// Row layer norm: one workgroup per token, D = 1024 features.
//#if SUBGROUPS
enable subgroups;
requires subgroup_id;
//#endif
//#include common

struct P { D: u32, bias: u32, typed: u32, eps: f32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> Wt: array<f32>;
@group(0) @binding(4) var<storage, read> Bs: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;
@group(0) @binding(6) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read> TE: array<f32>;
var<workgroup> red: array<f32, 256>;

//#if SUBGROUPS
fn reduce(v: f32, l: u32, subgroup: u32, lane: u32) -> f32 {
//#else
fn reduce(v: f32, l: u32) -> f32 {
//#endif
  red[l] = v;
  workgroupBarrier();
//#if SUBGROUPS
  // Preserve the tree's 128, 64, 32, 16, 8, 4, 2, 1 addition order.
  for (var s = 128u; s > 32u; s >>= 1u) {
    if (l < s) { red[l] += red[l + s]; }
    workgroupBarrier();
  }
  var tail = 0.0;
  if (subgroup == 0u) { tail = red[lane] + red[lane + 32u]; }
  for (var s = 16u; s > 0u; s >>= 1u) {
    let other = subgroupShuffleXor(tail, s);
    if (lane < s) { tail += other; }
  }
  if (subgroup == 0u && lane == 0u) { red[0] = tail; }
  workgroupBarrier();
//#else
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (l < s) { red[l] += red[l + s]; }
    workgroupBarrier();
  }
//#endif
  let r = red[0];
  workgroupBarrier();
  return r;
}

@compute @workgroup_size(256)
//#if SUBGROUPS
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) l: u32,
        @builtin(subgroup_id) subgroup: u32, @builtin(subgroup_invocation_id) lane: u32) {
//#else
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) l: u32) {
//#endif
  let t = wg.x;
  if (t >= g.T) { return; }
  let D = p.D;
  let base = t * D;
  var v: array<f32, 4>;
  var s = 0.0;
  for (var i = 0u; i < 4u; i++) { v[i] = X[base + l + i * 256u]; s += v[i]; }
//#if SUBGROUPS
  let mean = reduce(s, l, subgroup, lane) / f32(D);
//#else
  let mean = reduce(s, l) / f32(D);
//#endif
  var q = 0.0;
  for (var i = 0u; i < 4u; i++) { let d = v[i] - mean; q += d * d; }
//#if SUBGROUPS
  let inv = inverseSqrt(reduce(q, l, subgroup, lane) / f32(D) + p.eps);
//#else
  let inv = inverseSqrt(reduce(q, l) / f32(D) + p.eps);
//#endif
  let qt = tok[t].w;
  for (var i = 0u; i < 4u; i++) {
    let c = l + i * 256u;
    var o = (v[i] - mean) * inv * Wt[c];
    if (p.bias == 1u) { o += Bs[c]; }
    if (p.typed == 1u) { o += TE[qt * D + c]; }
    Y[base + c] = o;
  }
}
