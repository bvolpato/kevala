// l2-normalize q and k per head in place (q also scaled by 1/sqrt(128))
//#include kev_common

@group(0) @binding(1) var<storage, read_write> C: array<f32>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) j: u32) {
  let t = wg.x;
  if (t >= g.T) { return; }
  let idx = t * LIN_DIM + wg.y * 128u + j;
  let v = C[idx];
  red[j] = v * v;
  workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (j < k) { red[j] += red[j + k]; } workgroupBarrier(); }
  var inv = inverseSqrt(red[0] + 1e-6);
  if (wg.y < LIN_KEY_HEADS) { inv = inv * 0.08838834764831845; }
  C[idx] = v * inv;
}
