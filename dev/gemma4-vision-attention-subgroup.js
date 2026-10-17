// Experimental candidate. The portable fused kernel remains selected at runtime.
export default String.raw`
// F32 16-query x 16-key online attention, with two 16-lane rows per subgroup.
// Requires fixed 32-lane subgroups. Q/K/V tiles use 13,056 bytes of shared storage.
enable subgroups;
requires subgroup_id;

struct P { rows: u32, heads: u32, kv_heads: u32, dim: u32, lo: f32, hi: f32, clamp_output: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> positions: array<vec2<i32>>;
@group(0) @binding(5) var<storage, read_write> Y: array<vec4<f32>>;
var<workgroup> queries: array<vec4<f32>, 272>;
var<workgroup> keys: array<vec4<f32>, 272>;
var<workgroup> values: array<vec4<f32>, 272>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(subgroup_id) sgid: u32,
  @builtin(subgroup_invocation_id) lane: u32,
  @builtin(num_subgroups) nsg: u32
) {
  // This guard is uniform across the workgroup, including before every shuffle/barrier.
  if (nsg != 8u || p.dim != 64u) { return; }
  let row_lane = sgid * 2u + lane / 16u;
  let col = lane % 16u;
  let half_base = lane & 16u;
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
    // Max chooses an existing finite value, so its tree does not round or reorder sums.
    // XOR masks below 16 keep the two query rows independent within each subgroup.
    var tile_maximum = max(-1e30, score);
    tile_maximum = max(tile_maximum, subgroupShuffleXor(tile_maximum, 1u));
    tile_maximum = max(tile_maximum, subgroupShuffleXor(tile_maximum, 2u));
    tile_maximum = max(tile_maximum, subgroupShuffleXor(tile_maximum, 4u));
    tile_maximum = max(tile_maximum, subgroupShuffleXor(tile_maximum, 8u));
    let next_maximum = max(maximum, tile_maximum);
    let correction = exp(maximum - next_maximum);
    var probability = 0.0;
    if (key_row < p.rows) { probability = exp(score - next_maximum); }
    result *= correction;
    denominator *= correction;
    for (var part = 0u; part < 16u; part++) {
      let weight = subgroupShuffle(probability, half_base + part);
      result += values[part * 17u + col] * weight;
      denominator += weight;
    }
    maximum = next_maximum;
    // Padded query rows remain active until all subgroup operations and V reads finish.
    workgroupBarrier();
  }
  if (row < p.rows) {
    var output = result / denominator;
    if (p.clamp_output != 0u) { output = clamp(output, vec4<f32>(p.lo), vec4<f32>(p.hi)); }
    Y[(row * p.heads + head) * 16u + col] = output;
  }
}
`;
