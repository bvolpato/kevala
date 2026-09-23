// RMS norm over the hidden dimension (zero-centred weights stored as 1 + w).
//#if SUBGROUPS
enable subgroups;
requires subgroup_id;
//#endif
//#include kev_common

struct P { D: u32, _a: u32, _b: u32, eps: f32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> Wt: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
//#if SUBGROUPS
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) l: u32,
        @builtin(subgroup_id) subgroup: u32, @builtin(subgroup_invocation_id) lane: u32,
        @builtin(num_subgroups) subgroup_count: u32) {
//#else
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) l: u32) {
//#endif
  let t = wg.x;
  if (t >= g.T) { return; }
  let base = t * p.D;
  var v: array<f32, HIDDEN / 256u>;
  var s = 0.0;
  for (var i = 0u; i < HIDDEN / 256u; i++) { v[i] = X[base + l + i * 256u]; s += v[i] * v[i]; }
  red[l] = s;
  workgroupBarrier();
//#if SUBGROUPS
  // Preserve the tree's 128, 64, 32, 16, 8, 4, 2, 1 addition order.
  for (var k = 128u; k > 32u; k >>= 1u) { if (l < k) { red[l] += red[l + k]; } workgroupBarrier(); }
  // Eight 32-lane subgroups account for every invocation, with no inactive lanes.
  if (subgroup_count == 8u) {
    var tail = 0.0;
    if (subgroup == 0u) { tail = red[lane] + red[lane + 32u]; }
    for (var k = 16u; k > 0u; k >>= 1u) {
      let other = subgroupShuffleXor(tail, k);
      if (lane < k) { tail += other; }
    }
    if (subgroup == 0u && lane == 0u) { red[0] = tail; }
    workgroupBarrier();
  } else {
    for (var k = 32u; k > 0u; k >>= 1u) { if (l < k) { red[l] += red[l + k]; } workgroupBarrier(); }
  }
//#else
  for (var k = 128u; k > 0u; k >>= 1u) { if (l < k) { red[l] += red[l + k]; } workgroupBarrier(); }
//#endif
  let inv = inverseSqrt(red[0] / f32(p.D) + p.eps);
  for (var i = 0u; i < HIDDEN / 256u; i++) { let c = l + i * 256u; Y[base + c] = v[i] * inv * Wt[c]; }
}
