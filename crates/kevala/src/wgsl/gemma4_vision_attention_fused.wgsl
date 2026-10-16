// F32 16-query x 16-key online attention for Gemma 4's 64-dimensional vision heads.
// Q/K/V tiles and scores use 14,080 bytes; no quadratic storage buffer is materialized.
struct P { rows: u32, heads: u32, kv_heads: u32, dim: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> positions: array<vec2<i32>>;
@group(0) @binding(5) var<storage, read_write> Y: array<vec4<f32>>;
var<workgroup> queries: array<vec4<f32>, 272>;
var<workgroup> keys: array<vec4<f32>, 272>;
var<workgroup> values: array<vec4<f32>, 272>;
var<workgroup> weights: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let row_lane = lane / 16u;
  let col = lane % 16u;
  let row = wg.x * 16u + row_lane;
  let head = wg.y;
  let kv_head = head / (p.heads / p.kv_heads);
  var query = vec4<f32>(0.0);
  if (row < p.rows) { query = Q[(row * p.heads + head) * 16u + col]; }
  queries[row_lane * 17u + col] = query;
  var maximum = -1e30;
  var denominator = 0.0;
  var result = vec4<f32>(0.0);
  for (var key_base = 0u; key_base < p.rows; key_base += 16u) {
    let load_key = key_base + row_lane;
    var key = vec4<f32>(0.0);
    var value = vec4<f32>(0.0);
    if (load_key < p.rows) {
      let at = (load_key * p.kv_heads + kv_head) * 16u + col;
      key = K[at];
      value = V[at];
    }
    keys[row_lane * 17u + col] = key;
    values[row_lane * 17u + col] = value;
    workgroupBarrier();
    let key_row = key_base + col;
    var score = -1e30;
    if (key_row < p.rows && positions[key_row].x >= 0) {
      score = 0.0;
      for (var part = 0u; part < 16u; part++) {
        score += dot(queries[row_lane * 17u + part], keys[col * 17u + part]);
      }
    }
    weights[lane] = score;
    workgroupBarrier();
    var tile_maximum = -1e30;
    for (var part = 0u; part < 16u; part++) { tile_maximum = max(tile_maximum, weights[row_lane * 16u + part]); }
    let next_maximum = max(maximum, tile_maximum);
    let correction = exp(maximum - next_maximum);
    var probability = 0.0;
    if (key_row < p.rows) { probability = exp(score - next_maximum); }
    // Every lane has finished reading the raw score tile before it is overwritten.
    workgroupBarrier();
    weights[lane] = probability;
    workgroupBarrier();
    result *= correction;
    denominator *= correction;
    for (var part = 0u; part < 16u; part++) {
      let weight = weights[row_lane * 16u + part];
      result += values[part * 17u + col] * weight;
      denominator += weight;
    }
    maximum = next_maximum;
    workgroupBarrier();
  }
  if (row < p.rows) { Y[(row * p.heads + head) * 16u + col] = result / denominator; }
}
