struct P { count: u32, down_lo: f32, down_hi: f32, clamp_down: u32, gate_lo: f32, gate_hi: f32, up_lo: f32, up_hi: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> G: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> U: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Y: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x * 4u >= p.count) { return; }
  let x = clamp(G[g.x], vec4<f32>(p.gate_lo), vec4<f32>(p.gate_hi));
  let up = clamp(U[g.x], vec4<f32>(p.up_lo), vec4<f32>(p.up_hi));
  let activated = 0.5 * x * (vec4<f32>(1.0) + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
  var output = activated * up;
  if (p.clamp_down != 0u) { output = clamp(output, vec4<f32>(p.down_lo), vec4<f32>(p.down_hi)); }
  Y[g.x] = output;
}
