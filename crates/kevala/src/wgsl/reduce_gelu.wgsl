// Ordered split-K reduction for matmul_gelu's separate gate/up partial planes.
//#include common

struct P { N: u32, K: u32, splitT: u32, _pad: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> PART: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

fn gelu(x: f32) -> f32 {
  if (x >= 10.0) { return x; }
  if (x <= -10.0) { return 0.0; }
  let c = 0.7978845608028654;
  return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

//#include splits

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let T = g.T;
  let N = p.N;
  let o = gid.x + gid.y * 65535u * 256u;
  if (o >= T * N) { return; }
  let splitT = select(p.splitT, T, p.splitT == 0u);
  let splits = mm_splits(splitT, N, p.K);
  var gate = 0.0;
  var up = 0.0;
  for (var s = 0u; s < splits; s++) {
    gate += PART[s * T * N + o];
    up += PART[(splits + s) * T * N + o];
  }
  Y[o] = gelu(gate) * up;
}
