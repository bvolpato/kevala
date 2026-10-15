// E2B's 64-dimensional heads: one score per lane, with 16-way Q/K tile reuse.
struct P { rows: u32, heads: u32, kv_heads: u32, dim: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> positions: array<vec2<i32>>;
@group(0) @binding(4) var<storage, read_write> S: array<f32>;
var<workgroup> queries: array<vec4<f32>, 272>; // 16 rows with a padded vector stride
var<workgroup> keys: array<vec4<f32>, 272>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let group = wg.y * 65535u + wg.x;
  let tiles = (p.rows + 15u) / 16u;
  let head = group / (tiles * tiles);
  if (head >= p.heads) { return; }
  let query_tile = (group / tiles) % tiles;
  let key_tile = group % tiles;
  let local_row = lane / 16u;
  let local_key = lane % 16u;
  let row = query_tile * 16u + local_row;
  let key = key_tile * 16u + local_key;
  let load_key = key_tile * 16u + local_row;
  let kv_head = head / (p.heads / p.kv_heads);
  var q = vec4<f32>(0.0);
  var k = vec4<f32>(0.0);
  if (row < p.rows) { q = Q[(row * p.heads + head) * 16u + local_key]; }
  if (load_key < p.rows) { k = K[(load_key * p.kv_heads + kv_head) * 16u + local_key]; }
  queries[local_row * 17u + local_key] = q;
  keys[local_row * 17u + local_key] = k;
  workgroupBarrier();
  if (row >= p.rows || key >= p.rows) { return; }
  let at = (head * p.rows + row) * p.rows + key;
  if (positions[key].x < 0) { S[at] = -1e30; return; }
  var score = 0.0;
  for (var col = 0u; col < 16u; col++) {
    score += dot(queries[local_row * 17u + col], keys[local_key * 17u + col]);
  }
  // Vision Q/K norms supply the scale. The upstream attention multiplier is exactly one.
  S[at] = score;
}
