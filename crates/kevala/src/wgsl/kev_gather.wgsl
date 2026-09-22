// Copies scorer rows for the model's hidden width.
//#include kev_common

@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> rows: array<u32>;
@group(0) @binding(3) var<storage, read_write> O: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let r = wg.x;
  if (r >= g.R) { return; }
  let src = rows[r];
  for (var c = l; c < HIDDEN / 4u; c += 256u) {
    O[r * (HIDDEN / 4u) + c] = X[src * (HIDDEN / 4u) + c];
  }
}
