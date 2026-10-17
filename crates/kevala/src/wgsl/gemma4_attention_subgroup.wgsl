// Gemma decoder attention with the original dot, softmax, and V accumulation trees.
// Fixed 32-lane subgroups replace shared reductions, retaining 128-key rescale boundaries.
enable subgroups;
requires subgroup_id;
//#include common

struct P { heads: u32, kv_heads: u32, head_dim: u32, window: u32, causal: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> Q: array<f32>;
@group(0) @binding(3) var<storage, read> K: array<f32>;
@group(0) @binding(4) var<storage, read> V: array<f32>;
@group(0) @binding(5) var<storage, read_write> O: array<f32>;
@group(0) @binding(6) var<storage, read> POS: array<u32>;
var<workgroup> qtile: array<f32, 512>;
var<workgroup> scores: array<f32, 128>;
var<workgroup> state: array<f32, 4>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(subgroup_id) sgid: u32,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(num_subgroups) nsg: u32
) {
  let t = wg.x;
  let head = wg.y;
  if (nsg != 8u || t >= g.T || head >= p.heads || p.head_dim == 0u || p.head_dim > 512u) { return; }
  let lane = sgid * 32u + subgroup_lane;
  let H = p.head_dim;
  let qbase = t * p.heads * H + head * H;
  let kv_head = head / (p.heads / p.kv_heads);
  let position = POS[t];
  var branch_offset = 0u;
  if (p._a != 0u) { branch_offset = POS[g.T + t]; }
  for (var d = lane; d < H; d += 256u) { qtile[d] = Q[qbase + d]; }
  if (lane == 0u) { state[0] = -3.402823466e+38; state[1] = 0.0; }
  workgroupBarrier();
  var acc0 = 0.0;
  var acc1 = 0.0;
  let start = select(0u, position + 1u - min(position + 1u, p.window), p.window > 0u);
  let end = select(g.S, position + 1u, p.causal != 0u);
  let key_lane = lane % 8u;
  for (var j0 = start; j0 < end; j0 += 128u) {
    for (var block = 0u; block < 4u; block++) {
      let j = j0 + block * 32u + lane / 8u;
      let physical_key = j + select(0u, branch_offset, j >= g._b);
      let kbase = physical_key * p.kv_heads * H + kv_head * H;
      var sum = 0.0;
      if (j < end) {
        if (H % 4u == 0u) {
          for (var d = key_lane * 4u; d < H; d += 32u) {
            let qv = vec4<f32>(qtile[d], qtile[d + 1u], qtile[d + 2u], qtile[d + 3u]);
            let kv = vec4<f32>(K[kbase + d], K[kbase + d + 1u], K[kbase + d + 2u], K[kbase + d + 3u]);
            sum += dot(qv, kv);
          }
        } else {
          for (var d = key_lane; d < H; d += 8u) { sum += qtile[d] * K[kbase + d]; }
        }
      }
      // Lane zero of each eight-lane key keeps the old stride4/2/1 addition order.
      sum += subgroupShuffleXor(sum, 4u);
      sum += subgroupShuffleXor(sum, 2u);
      sum += subgroupShuffleXor(sum, 1u);
      if (key_lane == 0u) { scores[block * 32u + lane / 8u] = select(-3.402823466e+38, sum, j < end); }
    }
    workgroupBarrier();
    if (sgid == 0u) {
      // The old stride128 step only paired scores with the lowest finite F32 value.
      var top = max(max(scores[subgroup_lane], scores[subgroup_lane + 64u]), max(scores[subgroup_lane + 32u], scores[subgroup_lane + 96u]));
      top = max(top, subgroupShuffleXor(top, 16u));
      top = max(top, subgroupShuffleXor(top, 8u));
      top = max(top, subgroupShuffleXor(top, 4u));
      top = max(top, subgroupShuffleXor(top, 2u));
      top = max(top, subgroupShuffleXor(top, 1u));
      if (subgroup_lane == 0u) {
        let next_top = max(state[0], top);
        state[2] = exp(state[0] - next_top);
        state[0] = next_top;
      }
    }
    workgroupBarrier();
    var weight = 0.0;
    if (lane < 128u && j0 + lane < end) { weight = exp(scores[lane] - state[0]); }
    if (lane < 128u) { scores[lane] = weight; }
    workgroupBarrier();
    if (sgid == 0u) {
      // Match the old stride128/64/32 prefix, then the stride16/8/4/2/1 tree.
      // The removed stride128 addition was weight + 0, with nonnegative weights.
      var total = (scores[subgroup_lane] + scores[subgroup_lane + 64u]) + (scores[subgroup_lane + 32u] + scores[subgroup_lane + 96u]);
      total += subgroupShuffleXor(total, 16u);
      total += subgroupShuffleXor(total, 8u);
      total += subgroupShuffleXor(total, 4u);
      total += subgroupShuffleXor(total, 2u);
      total += subgroupShuffleXor(total, 1u);
      if (subgroup_lane == 0u) { state[1] = state[1] * state[2] + total; }
    }
    let correction = state[2];
    acc0 *= correction;
    acc1 *= correction;
    for (var j = 0u; j < min(128u, end - j0); j++) {
      let logical_key = j0 + j;
      let physical_key = logical_key + select(0u, branch_offset, logical_key >= g._b);
      let vbase = physical_key * p.kv_heads * H + kv_head * H;
      let w = scores[j];
      if (lane < H) { acc0 += w * V[vbase + lane]; }
      if (lane + 256u < H) { acc1 += w * V[vbase + lane + 256u]; }
    }
    workgroupBarrier();
  }
  if (lane < H) { O[qbase + lane] = acc0 / state[1]; }
  if (lane + 256u < H) { O[qbase + lane + 256u] = acc1 / state[1]; }
}
