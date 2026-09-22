// Q/K learned RMSNorm, V unweighted RMSNorm, and layer-specific RoPE.
// One workgroup owns one token and one head. This supports the 256-wide local heads
// and 512-wide global heads used by Gemma 4 E2B/E4B.
//#include common

struct P {
  heads: u32,
  kv_heads: u32,
  head_dim: u32,
  rotary_dim: u32,
  table: u32,
  table_stride: u32,
  has_kv: u32,
  _a: u32,
  eps: f32,
  _b: u32,
  _c: u32,
  _d: u32,
}
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read_write> Q: array<f32>;
@group(0) @binding(3) var<storage, read_write> K: array<f32>;
@group(0) @binding(4) var<storage, read_write> V: array<f32>;
@group(0) @binding(5) var<storage, read> QN: array<f32>;
@group(0) @binding(6) var<storage, read> KN: array<f32>;
@group(0) @binding(7) var<storage, read> CS: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read> pos: array<u32>;
var<workgroup> sums: array<f32, 256>;
var<workgroup> normed: array<f32, 512>;

fn rotate(x: f32, y: f32, c: f32, s: f32, first: bool) -> f32 {
  return select(y * c + x * s, x * c - y * s, first);
}

fn rope_index(position: u32, pair: u32) -> u32 {
  // 128 pairs is the largest table (local heads). The host keeps the second table
  // immediately after the first table and leaves unused entries at the end of each row.
  return p.table * p.table_stride + position * 128u + pair;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let t = wg.x;
  let group = wg.y;
  if (t >= g.T) { return; }
  let position = pos[t];
  let H = p.head_dim;
  // RoPE pairs the first half of the head with the second half. Global heads
  // have 512 dimensions but only the first 64 frequency pairs are nontrivial;
  // the remaining 192 pairs use the padded zero frequencies from HF's
  // proportional implementation (cos=1, sin=0).
  let half = H / 2u;
  var kind = 0u;
  var head = group;
  if (group >= p.heads && group < p.heads + p.kv_heads) {
    kind = 1u;
    head = group - p.heads;
  } else if (group >= p.heads + p.kv_heads) {
    if (p.has_kv == 0u) { return; }
    kind = 2u;
    head = group - p.heads - p.kv_heads;
  }
  if (kind > 0u && p.has_kv == 0u) { return; }

  var sum = 0.0;
  var base = 0u;
  if (kind == 0u) { base = t * p.heads * H + head * H; }
  else { base = t * p.kv_heads * H + head * H; }
  for (var d = lane; d < H; d += 256u) {
    var x = 0.0;
    if (kind == 0u) { x = Q[base + d]; }
    else if (kind == 1u) { x = K[base + d]; }
    else { x = V[base + d]; }
    sum += x * x;
  }
  sums[lane] = sum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (lane < stride) { sums[lane] += sums[lane + stride]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(sums[0] / f32(H) + p.eps);
  for (var d = lane; d < H; d += 256u) {
    var raw = 0.0;
    if (kind == 0u) { raw = Q[base + d]; }
    else if (kind == 1u) { raw = K[base + d]; }
    else { raw = V[base + d]; }
    var learned = 1.0;
    if (kind == 0u) { learned = QN[d]; }
    else if (kind == 1u) { learned = KN[d]; }
    normed[d] = raw * inv * select(learned, 1.0, kind == 2u);
  }
  workgroupBarrier();
  for (var d = lane; d < H; d += 256u) {
    var value = normed[d];
    if (kind < 2u && d < H) {
      let pair = d % half;
      if (pair < p.rotary_dim / 2u) {
        let cs = CS[rope_index(position, pair)];
        let a = normed[pair];
        let b = normed[pair + half];
        value = select(b * cs.x + a * cs.y, a * cs.x - b * cs.y, d < half);
      }
    }
    if (kind == 0u) { Q[base + d] = value; }
    else if (kind == 1u) { K[base + d] = value; }
    else { V[base + d] = value; }
  }
}
