struct P { rows: u32, heads: u32, kv_heads: u32, dim: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> positions: array<vec2<i32>>;
@group(0) @binding(4) var<storage, read_write> S: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let at = (wg.y * 65535u + wg.x) * 256u + lane;
  if (at >= p.heads * p.rows * p.rows) { return; }
  let key = at % p.rows;
  let row = (at / p.rows) % p.rows;
  let head = at / (p.rows * p.rows);
  if (positions[key].x < 0) { S[at] = -1e30; return; }
  let head4 = p.dim / 4u;
  let q = (row * p.heads + head) * head4;
  let k = (key * p.kv_heads + head / (p.heads / p.kv_heads)) * head4;
  var score = 0.0;
  for (var col = 0u; col < head4; col++) { score += dot(Q[q + col], K[k + col]); }
  // Vision Q/K norms supply the scale. The upstream attention multiplier is exactly one.
  S[at] = score;
}
