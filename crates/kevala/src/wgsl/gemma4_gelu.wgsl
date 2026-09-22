// Gemma's gated GELU-tanh MLP and PLE gate. The two inputs are separate buffers:
// the MLP uses gate and up, while the PLE gate uses gate and the per-layer input.
//#include common

struct P { I: u32, mode: u32, _a: u32, _b: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> U: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> A: array<f32>;

fn gelu(x: f32) -> f32 {
  let c = 0.7978845608028654;
  return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x + gid.y * 65535u * 256u;
  if (idx >= g.T * p.I) { return; }
  let t = idx / p.I;
  let i = idx % p.I;
  let x = U[t * p.I + i];
  A[idx] = gelu(x) * B[t * p.I + i];
}
