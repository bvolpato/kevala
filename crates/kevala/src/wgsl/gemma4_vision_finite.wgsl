// Preserve nonfinite image-projection rejection without a separate CPU readback.
struct P { count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> status: atomic<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x * 4u >= p.count) { return; }
  let bits = bitcast<vec4<u32>>(X[g.x]);
  if (any((bits & vec4<u32>(0x7f800000u)) == vec4<u32>(0x7f800000u))) {
    atomicOr(&status, 1u);
  }
}
