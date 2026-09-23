// Decode packed int8 weights once per matrix operation using the generic matmul's exact
// sx(word) * scale arithmetic. D[N, K] is f32 scratch; K must be a multiple of 32.
//#include common

struct P { N: u32, K: u32, mode: u32, bias: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> S: array<f32>;
@group(0) @binding(4) var<storage, read_write> D: array<vec4<f32>>;

fn sx(w: u32) -> vec4<f32> {
  let u = w ^ 0x80808080u;
  let bytes = vec4<u32>(u, u >> 8u, u >> 16u, u >> 24u) & vec4<u32>(0xFFu);
  return bitcast<vec4<f32>>(bytes | vec4<u32>(0x4B000000u)) - vec4<f32>(8388736.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x + gid.y * 65535u * 256u;
  if (q >= p.N * (p.K / 4u)) { return; }
  D[q] = sx(W[q]) * S[q / 8u];
}
