// Experimental activation quantization for matmul_dp4a. Each block reconstructs 32 f32
// values as scales.x * plane0 + scales.y * plane1. The host must reject non-finite inputs.
//#include common

struct P { N: u32, K: u32, mode: u32, bias: u32 }
struct QBlock { scales: vec2<f32>, planes: array<vec2<u32>, 8> }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Q: array<QBlock>;

var<workgroup> magnitudes: array<f32, 256>;

fn pack_i8(v: vec4<i32>) -> u32 {
  let b = bitcast<vec4<u32>>(v) & vec4<u32>(255u);
  return b.x | (b.y << 8u) | (b.z << 16u) | (b.w << 24u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let block = (wg.x + wg.y * 65535u) * 32u + li / 8u;
  let lane = li & 7u;
  let first = li & ~7u;
  let valid = block < g.T * (p.K / 32u);
  var x = vec4<f32>(0.0);
  if (valid) { x = X[block * 8u + lane]; }
  let a = abs(x);
  magnitudes[li] = max(max(a.x, a.y), max(a.z, a.w));
  workgroupBarrier();
  for (var stride = 4u; stride > 0u; stride /= 2u) {
    if (lane < stride) { magnitudes[li] = max(magnitudes[li], magnitudes[li + stride]); }
    workgroupBarrier();
  }
  let s0 = magnitudes[first] / 127.0;
  let q0 = vec4<i32>(clamp(round(x / select(1.0, s0, s0 > 0.0)), vec4<f32>(-127.0), vec4<f32>(127.0)));
  let residual = x - vec4<f32>(q0) * s0;
  let r = abs(residual);
  // All lanes finish reading the first reduction before its storage is reused.
  workgroupBarrier();
  magnitudes[li] = max(max(r.x, r.y), max(r.z, r.w));
  workgroupBarrier();
  for (var stride = 4u; stride > 0u; stride /= 2u) {
    if (lane < stride) { magnitudes[li] = max(magnitudes[li], magnitudes[li + stride]); }
    workgroupBarrier();
  }
  let s1 = magnitudes[first] / 127.0;
  let q1 = vec4<i32>(clamp(round(residual / select(1.0, s1, s1 > 0.0)), vec4<f32>(-127.0), vec4<f32>(127.0)));
  if (valid) {
    Q[block].planes[lane] = vec2<u32>(pack_i8(q0), pack_i8(q1));
    if (lane == 0u) { Q[block].scales = vec2<f32>(s0, s1); }
  }
}
