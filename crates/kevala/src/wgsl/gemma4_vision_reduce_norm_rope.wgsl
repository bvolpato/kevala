// Exact split-K reduction followed by the vision head64 RMSNorm and optional axial RoPE.
struct P {
  rows: u32, heads: u32, dim: u32, weighted: u32,
  eps: f32, lo: f32, hi: f32, skip_rope: u32,
  splits: u32, mode: u32, bias: u32, pad: u32
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> PART: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> C: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> raw: array<f32, 64>;
var<workgroup> sums: array<f32, 256>;
var<workgroup> normalized: array<vec4<f32>, 16>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let row_head = wg.x;
  if (row_head >= p.rows * p.heads || p.dim != 64u || p.splits == 0u || p.splits > 8u || p.mode != 0u || p.bias != 0u) { return; }
  let plane_stride = p.rows * p.heads * p.dim;
  let o = row_head * p.dim + lane;
  var v = 0.0;
  for (var s = 0u; s < p.splits; s++) { v += PART[s * plane_stride + o]; }
  // Match the original reducer's F32 output store before clipping or squaring the projection.
  raw[lane] = v;
  workgroupBarrier();
  let width4 = p.dim / 4u;
  for (var at = lane + 64u; at < 256u; at += 64u) { sums[at] = 0.0; }
  var sum = 0.0;
  for (var col = lane; col < width4; col += 256u) {
    let at = col * 4u;
    let x = clamp(vec4<f32>(raw[at], raw[at + 1u], raw[at + 2u], raw[at + 3u]), vec4<f32>(p.lo), vec4<f32>(p.hi));
    sum += dot(x, x);
  }
  sums[lane] = sum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (lane < stride) { sums[lane] += sums[lane + stride]; }
    workgroupBarrier();
  }
  let scale = pow(sums[0] / f32(p.dim) + p.eps, -0.5);
  for (var col = lane; col < width4; col += 256u) {
    let at = col * 4u;
    var y = clamp(vec4<f32>(raw[at], raw[at + 1u], raw[at + 2u], raw[at + 3u]), vec4<f32>(p.lo), vec4<f32>(p.hi)) * scale;
    if (p.weighted != 0u) { y *= W[col]; }
    normalized[col] = y;
  }
  workgroupBarrier();
  for (var col = lane; col < p.dim; col += 64u) {
    if (p.skip_rope != 0u) {
      Y[row_head * p.dim + col] = normalized[col / 4u][col % 4u];
    } else {
      let row = row_head / p.heads;
      let spatial = p.dim / 2u;
      let half = spatial / 2u;
      let axis = col / spatial;
      let part = col % spatial;
      let pair = part % half;
      let cs = C[row * spatial + axis * half + pair];
      let partner = select(col + half, col - half, part >= half);
      let x = normalized[col / 4u][col % 4u];
      let other = normalized[partner / 4u][partner % 4u];
      let rotated = select(-other, other, part >= half);
      Y[row_head * p.dim + col] = x * cs.x + rotated * cs.y;
    }
  }
}
