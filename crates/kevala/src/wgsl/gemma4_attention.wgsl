// Correctness-first causal online attention for Gemma's local and global heads.
// A workgroup computes one query head. Its lanes cooperate on the dot product,
// then stream the keys and values once while maintaining an online softmax.
//#include common

struct P { heads: u32, kv_heads: u32, head_dim: u32, window: u32, causal: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> Q: array<f32>;
@group(0) @binding(3) var<storage, read> K: array<f32>;
@group(0) @binding(4) var<storage, read> V: array<f32>;
@group(0) @binding(5) var<storage, read_write> O: array<f32>;
var<workgroup> qtile: array<f32, 512>;
var<workgroup> partial: array<f32, 256>;
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
  if (lane == 0u) {
    state[0] = -3.402823466e+38;
    state[1] = 0.0;
  }
  workgroupBarrier();

  var acc0 = 0.0;
  var acc1 = 0.0;
  let start = select(0u, t + 1u - min(t + 1u, p.window), p.window > 0u);
  let end = select(g.T, t + 1u, p.causal != 0u);
  for (var j = start; j < end; j++) {
    let kbase = j * p.kv_heads * H + kv_head * H;
    var dot = 0.0;
    for (var d = lane; d < H; d += 256u) { dot += qtile[d] * K[kbase + d]; }
    partial[lane] = dot;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
      if (lane < stride) { partial[lane] += partial[lane + stride]; }
      workgroupBarrier();
    }
    if (lane == 0u) {
      let score = partial[0];
      let oldM = state[0];
      let newM = max(oldM, score);
      let correction = select(0.0, exp(oldM - newM), oldM > -3.0e38);
      let weight = exp(score - newM);
      state[0] = newM;
      state[1] = state[1] * correction + weight;
      state[2] = correction;
      state[3] = weight;
    }
    workgroupBarrier();
    let correction = state[2];
    let weight = state[3];
    for (var d = lane; d < H; d += 256u) {
      let value = V[kbase + d];
      if (d < 256u) { acc0 = acc0 * correction + weight * value; }
      else { acc1 = acc1 * correction + weight * value; }
    }
    workgroupBarrier();
  }
  let denom = state[1];
  let obase = t * p.heads * H + head * H;
  for (var d = lane; d < H; d += 256u) {
    O[obase + d] = select(acc1, acc0, d < 256u) / denom;
  }
}
