// Combines the token identity and context projection parts of PLE. The context
// projection has already been computed by a Q8 matmul from the original embedding.
//#include common

struct P { width: u32, input_dim: u32, eps: f32, input_scale: f32, projection_scale: f32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> token: array<f32>;
@group(0) @binding(3) var<storage, read> projected: array<f32>;
@group(0) @binding(4) var<storage, read> W: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;
var<workgroup> sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let t = wg.x;
  if (t >= g.T) { return; }
  let base = t * p.width;
  var sum = 0.0;
  for (var i = lane; i < p.width; i += 256u) {
    let v = projected[base + i] * p.projection_scale;
    sum += v * v;
  }
  sums[lane] = sum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (lane < stride) { sums[lane] += sums[lane + stride]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(sums[0] / f32(p.width) + p.eps);
  for (var i = lane; i < p.width; i += 256u) {
    let context = projected[base + i] * p.projection_scale * inv * W[i];
    Y[base + i] = (token[base + i] + context) * p.input_scale;
  }
}
