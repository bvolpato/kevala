// per-head RMS norm of the DeltaNet output, gated by silu(z)
//#include kev_common

struct P { eps: f32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> PROJ: array<f32>;
@group(0) @binding(3) var<storage, read> GW: array<f32>;
@group(0) @binding(4) var<storage, read_write> CORE: array<f32>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) j: u32) {
  let t = wg.x;
  if (t >= g.T) { return; }
  let idx = t * LIN_OUT + wg.y * 128u + j;
  let o = CORE[idx];
  red[j] = o * o;
  workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (j < k) { red[j] += red[j + k]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / 128.0 + p.eps);
  let z = PROJ[t * LIN_WIDTH + LIN_DIM + wg.y * 128u + j];
  CORE[idx] = o * inv * GW[j] * (z / (1.0 + exp(-z)));
}
