struct P { rows: u32, width: u32, weighted: u32, residual: u32, eps: f32, lo: f32, hi: f32, clamp_after: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> W: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> R: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> Y: array<vec4<f32>>;
var<workgroup> sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let row = wg.x;
  let width4 = p.width / 4u;
  var sum = 0.0;
  for (var col = lane; col < width4; col += 256u) {
    var x = X[row * width4 + col];
    if (p.clamp_after == 0u) { x = clamp(x, vec4<f32>(p.lo), vec4<f32>(p.hi)); }
    sum += dot(x, x);
  }
  sums[lane] = sum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (lane < stride) { sums[lane] += sums[lane + stride]; }
    workgroupBarrier();
  }
  let scale = pow(sums[0] / f32(p.width) + p.eps, -0.5);
  for (var col = lane; col < width4; col += 256u) {
    let at = row * width4 + col;
    var y: vec4<f32>;
    if (p.clamp_after == 0u) {
      y = clamp(X[at], vec4<f32>(p.lo), vec4<f32>(p.hi)) * scale;
    } else {
      y = X[at] * scale;
    }
    if (p.weighted != 0u) { y *= W[col]; }
    if (p.residual != 0u) { y += R[at]; }
    if (p.clamp_after != 0u) { y = clamp(y, vec4<f32>(p.lo), vec4<f32>(p.hi)); }
    Y[at] = y;
  }
}
