// Ordered decoder split reduction, then the unchanged Gemma RMSNorm epilogue.
//#if SUBGROUPS
//#include ordered_sum
//#endif
//#include common

struct P { D: u32, splits: u32, eps: f32, mode: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> PART: array<f32>;
@group(0) @binding(3) var<storage, read> W: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> raw: array<f32, 2048>;
var<workgroup> sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32
//#if SUBGROUPS
  , @builtin(subgroup_invocation_id) subgroupLane: u32
//#endif
) {
  let t = wg.x;
  if (t >= g.T || p.D == 0u || p.D > 2048u || p.splits == 0u || p.splits > 8u || p.mode > 2u) { return; }
  let base = t * p.D;
  for (var d = lane; d < p.D; d += 256u) {
    let o = base + d;
    var v = 0.0;
    for (var s = 0u; s < p.splits; s++) { v += PART[s * g.T * p.D + o]; }
    raw[d] = v;
  }
  // Retain the original reducer's F32 store before the RMS products and reduction.
  workgroupBarrier();
  var sum = 0.0;
  for (var d = lane; d < p.D; d += 256u) {
    let v = raw[d];
    sum += v * v;
  }
  sums[lane] = sum;
  workgroupBarrier();
//#if SUBGROUPS
  for (var stride = 128u; stride >= 32u; stride >>= 1u) {
//#else
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
//#endif
    if (lane < stride) { sums[lane] += sums[lane + stride]; }
    workgroupBarrier();
  }
//#if SUBGROUPS
  let total = ordered_sum32(sums[subgroupLane], subgroupLane);
//#else
  let total = sums[0];
//#endif
  let inv = inverseSqrt(total / f32(p.D) + p.eps);
  for (var d = lane; d < p.D; d += 256u) {
    let factor = select(1.0, W[d], p.mode == 1u || p.mode == 2u);
    let normalized = raw[d] * inv;
    let weighted = normalized * factor;
    if (p.mode == 2u) {
      Y[base + d] = Y[base + d] + weighted;
    } else {
      Y[base + d] = weighted;
    }
  }
}
