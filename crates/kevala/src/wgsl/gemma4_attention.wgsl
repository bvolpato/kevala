// Gemma attention: eight lanes cooperate on each of 32 keys, then reduce a
// 128-key softmax tile. Output dimensions stay in registers across tiles.
//#include common

struct P { heads: u32, kv_heads: u32, head_dim: u32, window: u32, causal: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> Q: array<f32>;
@group(0) @binding(3) var<storage, read> K: array<f32>;
@group(0) @binding(4) var<storage, read> V: array<f32>;
@group(0) @binding(5) var<storage, read_write> O: array<f32>;
var<workgroup> qtile: array<f32, 512>;
var<workgroup> partial: array<f32, 256>;
var<workgroup> scores: array<f32, 128>;
var<workgroup> state: array<f32, 4>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let t = wg.x;
  let head = wg.y;
  if (t >= g.T || head >= p.heads) { return; }
  let H = p.head_dim;
  let qbase = t * p.heads * H + head * H;
  let kv_head = head / (p.heads / p.kv_heads);
  for (var d = lane; d < H; d += 256u) { qtile[d] = Q[qbase + d]; }
  if (lane == 0u) { state[0] = -3.402823466e+38; state[1] = 0.0; }
  workgroupBarrier();
  var acc0 = 0.0;
  var acc1 = 0.0;
  let start = select(0u, t + 1u - min(t + 1u, p.window), p.window > 0u);
  let end = select(g.T, t + 1u, p.causal != 0u);
  let key_lane = lane % 8u;
  for (var j0 = start; j0 < end; j0 += 128u) {
    for (var block = 0u; block < 4u; block++) {
      let j = j0 + block * 32u + lane / 8u;
      let kbase = j * p.kv_heads * H + kv_head * H;
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
      partial[lane] = sum;
      workgroupBarrier();
      for (var stride = 4u; stride > 0u; stride >>= 1u) {
        if (key_lane < stride) { partial[lane] += partial[lane + stride]; }
        workgroupBarrier();
      }
      if (key_lane == 0u) { scores[block * 32u + lane / 8u] = select(-3.402823466e+38, partial[lane], j < end); }
      workgroupBarrier();
    }
    partial[lane] = -3.402823466e+38;
    if (lane < 128u) { partial[lane] = scores[lane]; }
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
      if (lane < stride) { partial[lane] = max(partial[lane], partial[lane + stride]); }
      workgroupBarrier();
    }
    if (lane == 0u) {
      let top = max(state[0], partial[0]);
      state[2] = exp(state[0] - top);
      state[0] = top;
    }
    workgroupBarrier();
    var weight = 0.0;
    if (lane < 128u && j0 + lane < end) { weight = exp(scores[lane] - state[0]); }
    if (lane < 128u) { scores[lane] = weight; }
    partial[lane] = weight;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
      if (lane < stride) { partial[lane] += partial[lane + stride]; }
      workgroupBarrier();
    }
    if (lane == 0u) { state[1] = state[1] * state[2] + partial[0]; }
    let correction = state[2];
    acc0 *= correction;
    acc1 *= correction;
    for (var j = 0u; j < min(128u, end - j0); j++) {
      let vbase = (j0 + j) * p.kv_heads * H + kv_head * H;
      let w = scores[j];
      if (lane < H) { acc0 += w * V[vbase + lane]; }
      if (lane + 256u < H) { acc1 += w * V[vbase + lane + 256u]; }
    }
    workgroupBarrier();
  }
  if (lane < H) { O[qbase + lane] = acc0 / state[1]; }
  if (lane + 256u < H) { O[qbase + lane + 256u] = acc1 / state[1]; }
}
