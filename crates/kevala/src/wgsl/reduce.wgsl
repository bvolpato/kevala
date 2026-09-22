// Sums the split-K partial tiles of a matmul and applies its epilogue (bias, residual add or
// ReLU). Paired with the matmul variant of the same row and column tiling.
//#include common

struct P { N: u32, K: u32, mode: u32, bias: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> PART: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

//#include splits

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let T = g.T;
  let N = p.N;
  let o = gid.x + gid.y * 65535u * 256u;
  if (o >= T * N) { return; }
  let splits = mm_splits(T, N, p.K);
  var v = 0.0;
  for (var s = 0u; s < splits; s++) { v += PART[s * T * N + o]; }
  if (p.bias == 1u) { v += B[o % N]; }
  if (p.mode == 1u) { v += Y[o]; } else if (p.mode == 2u) { v = max(v, 0.0); }
  Y[o] = v;
}
