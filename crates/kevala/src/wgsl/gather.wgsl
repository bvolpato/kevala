// Copies the rows the scorer needs (markers and segment starts) into a compact buffer.
//#include common

@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> rows: array<u32>;
@group(0) @binding(3) var<storage, read_write> O: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let r = wg.x;
  if (r >= g.R) { return; }
  let src = rows[r];
  O[r * 256u + l] = X[src * 256u + l];
}
