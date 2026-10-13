struct P { rows: u32, width: u32, patches: u32, standardize: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> I: array<u32>;
@group(0) @binding(3) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> S: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> Y: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let at = g.x;
  let width4 = p.width / 4u;
  if (at >= p.rows * width4) { return; }
  let row = at / width4;
  let col = at % width4;
  var value = vec4<f32>(0.0);
  for (var index = 0u; index < p.patches; index++) {
    value += X[I[row * p.patches + index] * width4 + col] * (1.0 / f32(p.patches));
  }
  value *= sqrt(f32(p.width));
  if (p.standardize != 0u) { value = (value - B[col]) * S[col]; }
  Y[at] = value;
}
