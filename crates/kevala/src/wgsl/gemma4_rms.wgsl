// Gemma RMSNorm uses the checkpoint weight as a direct multiplicative tensor.
// Mode 0 is unscaled, mode 1 is weighted, and mode 2 adds the weighted result to Y.
//#include common

struct P { D: u32, eps: f32, mode: u32, _a: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> W: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
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
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (lane < stride) { sums[lane] += sums[lane + stride]; }
    workgroupBarrier();
  }
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
