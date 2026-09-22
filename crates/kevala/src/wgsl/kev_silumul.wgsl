// SwiGLU: A[t, i] = silu(U[t, i]) * U[t, I + i]
//#include kev_common

struct P { I: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> Uu: array<f32>;
@group(0) @binding(3) var<storage, read_write> A: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let I = p.I;
  let idx = gid.x + gid.y * 65535u * 256u;
  if (idx >= g.T * I) { return; }
  let t = idx / I;
  let i = idx % I;
  let u = Uu[t * 2u * I + i];
  A[idx] = u / (1.0 + exp(-u)) * Uu[t * 2u * I + I + i];
}
