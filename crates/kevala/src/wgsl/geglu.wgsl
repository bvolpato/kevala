// GeGLU: A[t, i] = gelu(U[t, i]) * U[t, I + i], exact erf GELU.
//#include common

struct P { I: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> U: array<f32>;
@group(0) @binding(3) var<storage, read_write> A: array<f32>;

// Abramowitz and Stegun 7.1.26 is only good to 1.5e-7 near 0; use erf's Taylor series there
fn erf(x: f32) -> f32 {
  let a = abs(x);
  if (a < 0.5) {
    let z = x * x;
    return x * (1.1283791671 + z * (-0.3761263890 + z * (0.1128379167 + z * (-0.0268661706 + z * 0.0052239776))));
  }
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return select(-y, y, x >= 0.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let I = p.I;
  let idx = gid.x + gid.y * 65535u * 256u;
  if (idx >= g.T * I) { return; }
  let t = idx / I;
  let i = idx % I;
  let u = U[t * 2u * I + i];
  let gate = U[t * 2u * I + I + i];
  A[idx] = 0.5 * u * (1.0 + erf(u * 0.70710678118)) * gate;
}
