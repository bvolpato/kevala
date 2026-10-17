// Gemma vision per-head RMSNorm, optionally followed by axial RoPE, in the original F32 order.
struct P { rows: u32, heads: u32, dim: u32, weighted: u32, eps: f32, lo: f32, hi: f32, skip_rope: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> W: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> C: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> sums: array<f32, 256>;
var<workgroup> normalized: array<vec4<f32>, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let row_head = wg.x;
  if (row_head >= p.rows * p.heads || p.dim == 0u || p.dim > 256u || p.dim % 4u != 0u) { return; }
  let width4 = p.dim / 4u;
  // At most 64 vec4 lanes contribute. Keep the old 256-lane reduction's inactive sums zero.
  for (var at = lane + 64u; at < 256u; at += 64u) { sums[at] = 0.0; }
  var sum = 0.0;
  for (var col = lane; col < width4; col += 256u) {
    let x = clamp(X[row_head * width4 + col], vec4<f32>(p.lo), vec4<f32>(p.hi));
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
    var y = clamp(X[row_head * width4 + col], vec4<f32>(p.lo), vec4<f32>(p.hi)) * scale;
    if (p.weighted != 0u) { y *= W[col]; }
    // Materialize the learned multiply in F32 before the rotation, as the old norm output did.
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
