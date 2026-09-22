// Attention prep: RMS-normalize q and k heads, then apply partial rotary embeddings.
//#include kev_common

struct P { eps: f32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read_write> PROJ: array<f32>;
@group(0) @binding(3) var<storage, read> QN: array<f32>;
@group(0) @binding(4) var<storage, read> KN: array<f32>;
@group(0) @binding(5) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> CS: array<vec2<f32>>;
var<workgroup> red: array<f32, 256>;
var<workgroup> v: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) d: u32) {
  let t = wg.x;
  if (t >= g.T) { return; }
  let h = wg.y;
  var off = t * ATTN_WIDTH + h * 512u;
  if (h >= HEADS) { off = t * ATTN_WIDTH + 2u * ATTN_Q + (h - HEADS) * 256u; }
  let x = PROJ[off + d];
  red[d] = x * x;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (d < k) { red[d] += red[d + k]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / 256.0 + p.eps);
  var w = QN[d];
  if (h >= HEADS) { w = KN[d]; }
  v[d] = x * inv * w;
  workgroupBarrier();
  let pos = tok[t].z;
  if (d < ROTARY / 2u) {
    let cs = CS[pos * (ROTARY / 2u) + d];
    let x1 = v[d];
    let x2 = v[d + ROTARY / 2u];
    PROJ[off + d] = x1 * cs.x - x2 * cs.y;
    PROJ[off + d + ROTARY / 2u] = x2 * cs.x + x1 * cs.y;
  } else if (d >= ROTARY) {
    PROJ[off + d] = v[d];
  }
}
