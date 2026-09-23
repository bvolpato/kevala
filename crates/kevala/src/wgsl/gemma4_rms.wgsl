// Gemma RMSNorm uses the checkpoint weight as a direct multiplicative tensor.
// Mode 0 is unscaled, mode 1 is weighted, and mode 2 adds the weighted result to Y.
//#if SUBGROUPS
enable subgroups;
requires subgroup_id;
//#endif
//#include common

struct P { D: u32, eps: f32, mode: u32, _a: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> W: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> sums: array<f32, 256>;

@compute @workgroup_size(256)
//#if SUBGROUPS
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32,
        @builtin(subgroup_id) subgroup: u32, @builtin(subgroup_invocation_id) sublane: u32,
        @builtin(num_subgroups) subgroup_count: u32) {
//#else
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
//#endif
  let t = wg.x;
  if (t >= g.T) { return; }
  let base = t * p.D;
  var sum = 0.0;
  for (var d = lane; d < p.D; d += 256u) {
    let v = X[base + d];
    sum += v * v;
  }
  sums[lane] = sum;
  workgroupBarrier();
//#if SUBGROUPS
  // Preserve the tree's 128, 64, 32, 16, 8, 4, 2, 1 addition order.
  for (var stride = 128u; stride > 32u; stride >>= 1u) {
    if (lane < stride) { sums[lane] += sums[lane + stride]; }
    workgroupBarrier();
  }
  // Eight 32-lane subgroups account for every invocation, with no inactive lanes.
  if (subgroup_count == 8u) {
    var tail = 0.0;
    if (subgroup == 0u) { tail = sums[sublane] + sums[sublane + 32u]; }
    for (var stride = 16u; stride > 0u; stride >>= 1u) {
      let other = subgroupShuffleXor(tail, stride);
      if (sublane < stride) { tail += other; }
    }
    if (subgroup == 0u && sublane == 0u) { sums[0] = tail; }
    workgroupBarrier();
  } else {
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
      if (lane < stride) { sums[lane] += sums[lane + stride]; }
      workgroupBarrier();
    }
  }
//#else
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (lane < stride) { sums[lane] += sums[lane + stride]; }
    workgroupBarrier();
  }
//#endif
  let inv = inverseSqrt(sums[0] / f32(p.D) + p.eps);
  for (var d = lane; d < p.D; d += 256u) {
    let factor = select(1.0, W[d], p.mode == 1u || p.mode == 2u);
    let normalized = X[base + d] * inv;
    let weighted = normalized * factor;
    if (p.mode == 2u) {
      Y[base + d] = Y[base + d] + weighted;
    } else {
      Y[base + d] = weighted;
    }
  }
}
