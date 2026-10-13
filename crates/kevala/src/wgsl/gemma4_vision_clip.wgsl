struct P { count: u32, pad0: u32, pad1: u32, pad2: u32, lo: f32, hi: f32, pad3: f32, pad4: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Y: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x * 4u >= p.count) { return; }
  Y[g.x] = clamp(X[g.x], vec4<f32>(p.lo), vec4<f32>(p.hi));
}
