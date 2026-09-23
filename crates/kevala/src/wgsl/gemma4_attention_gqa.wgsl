// Gemma attention tile for grouped-query heads. A workgroup covers 32 query rows
// for one KV head: four tokens x eight query heads for E2B, or eight tokens x four
// query heads for E4B. One KV tile is staged once for every group of eight keys at
// head_dim=256 and four keys at head_dim=512.
//
// The shader uses ordinary workgroup barriers for staging and subgroup XORs for
// row reduction. It keeps one online-softmax state per query row. The host must dispatch
// ceil(T / (32 / (heads / kv_heads))) by kv_heads workgroups and pass token_rows
// plus key_count in the candidate uniform.
enable subgroups;
requires subgroup_id;
//#include common

struct P {
  heads: u32,
  kv_heads: u32,
  head_dim: u32,
  window: u32,
  causal: u32,
  token_rows: u32,
  key_count: u32,
  subgroup32: u32,
}
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> Q: array<f32>;
@group(0) @binding(3) var<storage, read> K: array<f32>;
@group(0) @binding(4) var<storage, read> V: array<f32>;
@group(0) @binding(5) var<storage, read_write> O: array<f32>;
@group(0) @binding(6) var<storage, read> POS: array<u32>;

const NEG_INF = -3.402823466e+38;
const SCORE_ROUNDS = 2u;

// H=256,K=8 and H=512,K=4 both use exactly 512 vec4 slots per staged array.
var<workgroup> ktile: array<vec4<f32>, 512>;
var<workgroup> vtile: array<vec4<f32>, 512>;
// Fallback score reduction for subgroup sizes other than 32. The fast path
// leaves this storage unused and reduces eight lanes with subgroup XORs.
var<workgroup> score_part: array<f32, 256>;
var<workgroup> ranges: array<vec2<u32>, 32>;
var<workgroup> tile_range: vec2<u32>;

// With eight fixed 32-lane subgroups, subgroup_id and subgroup_invocation_id
// provide the logical row/lane mapping without assuming local invocation order.
// A device with another subgroup size uses local indices and this shared fallback.
fn row_sum(x: f32, qrow: u32, lane: u32, li: u32, use_subgroups: bool) -> f32 {
  if (use_subgroups) {
    var s = x + subgroupShuffleXor(x, 1u);
    s += subgroupShuffleXor(s, 2u);
    return s + subgroupShuffleXor(s, 4u);
  }
  score_part[li] = x;
  workgroupBarrier();
  for (var stride = 4u; stride > 0u; stride >>= 1u) {
    if (lane < stride) { score_part[li] += score_part[li + stride]; }
    workgroupBarrier();
  }
  let result = score_part[qrow * 8u];
  workgroupBarrier();
  return result;
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_index) li: u32,
  @builtin(subgroup_id) subgroup_id: u32,
  @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
  @builtin(num_subgroups) num_subgroups: u32
) {
  let group = p.heads / p.kv_heads;
  let block = wg.x * p.token_rows;
  let use_subgroups = p.subgroup32 != 0u && num_subgroups == 8u;
  let logical_li = select(li, subgroup_id * 32u + subgroup_invocation_id, use_subgroups);
  let qrow = logical_li / 8u;
  let lane = logical_li % 8u;
  let token_row = qrow / group;
  let head_in_group = qrow % group;
  let t = block + token_row;
  let kv_head = wg.y;
  let head = kv_head * group + head_in_group;
  let qvalid = qrow < 32u && token_row < p.token_rows && t < g.T && head < p.heads;
  let H = p.head_dim;
  let vecs = H / 4u;
  let qbase = (t * p.heads + head) * H;

  // Keep the query and output vectors in registers. The supported 256/512 head
  // dimensions require at most 8/16 vec4s per lane.
  var q: array<vec4<f32>, 16>;
  var acc: array<vec4<f32>, 16>;
  for (var i = 0u; i < 16u; i++) {
    let d = (lane + 8u * i) * 4u;
    q[i] = vec4<f32>(0.0);
    acc[i] = vec4<f32>(0.0);
    if (qvalid && d + 3u < H) {
      q[i] = vec4<f32>(Q[qbase + d], Q[qbase + d + 1u], Q[qbase + d + 2u], Q[qbase + d + 3u]);
    }
  }

  var row_start = select(g.S, 0u, qvalid);
  var row_end = 0u;
  if (qvalid) {
    let position = POS[t];
    row_start = select(0u, position + 1u - min(position + 1u, p.window), p.window > 0u);
    row_end = select(g.S, position + 1u, p.causal != 0u);
  }
  if (lane == 0u) { ranges[qrow] = vec2<u32>(row_start, row_end); }
  workgroupBarrier();

  // Use the union of row ranges so arbitrary POS values remain correct while a
  // sliding window does not force every row to scan the entire sequence.
  if (li == 0u) {
    var first = g.S;
    var last = 0u;
    for (var r = 0u; r < 32u; r++) {
      first = min(first, ranges[r].x);
      last = max(last, ranges[r].y);
    }
    tile_range = vec2<u32>(first, min(last, g.S));
  }
  workgroupBarrier();
  let tile_start = tile_range.x;
  let tile_end = tile_range.y;

  var m = NEG_INF;
  var l = 0.0;
  for (var j0 = tile_start; j0 < tile_end; j0 += p.key_count) {
    // Stage one KV head once. The host only selects key_count=8 for H=256 and
    // key_count=4 for H=512, so key_count * vecs is always 512.
    for (var e = li; e < 512u; e += 256u) {
      let key = e / vecs;
      let col = e % vecs;
      let d = col * 4u;
      let j = j0 + key;
      let valid = key < p.key_count && j < g.S && d + 3u < H;
      var kk = vec4<f32>(0.0);
      var vv = vec4<f32>(0.0);
      if (valid) {
        let base = (j * p.kv_heads + kv_head) * H + d;
        kk = vec4<f32>(K[base], K[base + 1u], K[base + 2u], K[base + 3u]);
        vv = vec4<f32>(V[base], V[base + 1u], V[base + 2u], V[base + 3u]);
      }
      ktile[e] = kk;
      vtile[e] = vv;
    }
    workgroupBarrier();

    var top = m;
    var tile_scores: array<f32, 8>;
    for (var round = 0u; round < SCORE_ROUNDS; round++) {
      var part = vec4<f32>(0.0);
      for (var key4 = 0u; key4 < 4u; key4++) {
        let key = round * 4u + key4;
        if (key < p.key_count) {
          var sum = 0.0;
          for (var i = 0u; i < 16u; i++) {
            let d = lane + 8u * i;
            if (d < vecs) { sum += dot(q[i], ktile[key * vecs + d]); }
          }
          part[key4] = sum;
        }
      }
      for (var key4 = 0u; key4 < 4u; key4++) {
        let key = round * 4u + key4;
        let j = j0 + key;
        let valid = qvalid && key < p.key_count && j < g.S && j >= row_start && j < row_end;
        let value = select(NEG_INF, row_sum(part[key4], qrow, lane, li, use_subgroups), valid);
        tile_scores[key] = value;
        top = max(top, value);
      }
    }

    let has_old = m > -3.0e37;
    let has_new = top > -3.0e37;
    let correction = select(1.0, exp(m - top), has_old && has_new);
    l *= correction;
    for (var i = 0u; i < 16u; i++) { acc[i] *= correction; }

    for (var round = 0u; round < SCORE_ROUNDS; round++) {
      for (var key4 = 0u; key4 < 4u; key4++) {
        let key = round * 4u + key4;
        let j = j0 + key;
        let valid = qvalid && key < p.key_count && j < g.S && j >= row_start && j < row_end;
        if (valid && has_new) {
          let score = tile_scores[key];
          let weight = exp(score - top);
          l += weight;
          for (var i = 0u; i < 16u; i++) {
            let d = lane + 8u * i;
            if (d < vecs) { acc[i] += weight * vtile[key * vecs + d]; }
          }
        }
      }
    }
    m = select(m, top, has_new);
    workgroupBarrier();
  }

  if (qvalid) {
    for (var i = 0u; i < 16u; i++) {
      let d = (lane + 8u * i) * 4u;
      if (d + 3u < H) {
        let obase = qbase + d;
        let value = acc[i] / l;
        O[obase] = value.x;
        O[obase + 1u] = value.y;
        O[obase + 2u] = value.z;
        O[obase + 3u] = value.w;
      }
    }
  }
}
