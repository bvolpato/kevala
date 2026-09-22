// In-place residual addition or layer scalar. Keeping this separate from matmul
// avoids relying on aliasing two storage bindings in a bind group.
//#include common

struct P { D: u32, mode: u32, _a: u32, _b: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read_write> X: array<f32>;
@group(0) @binding(3) var<storage, read> Y: array<f32>;
@group(0) @binding(4) var<storage, read> S: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let t = wg.x;
  if (t >= g.T) { return; }
  let base = t * p.D;
  let scalar = S[0];
  for (var d = lane; d < p.D; d += 256u) {
    if (p.mode == 0u) { X[base + d] += Y[base + d]; }
    else { X[base + d] *= scalar; }
  }
}
