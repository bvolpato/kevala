// Fused vision pool and its following unweighted RMS norm. A workgroup store
// retains the F32 rounding boundary between the original two dispatches.
// The host selects this path only for widths divisible by four and <= 1024.
struct P { rows: u32, width: u32, patches: u32, standardize: u32, eps: f32, lo: f32, hi: f32, _pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> I: array<u32>;
@group(0) @binding(3) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> S: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> Y: array<vec4<f32>>;

var<workgroup> pooled: array<vec4<f32>, 256>;
var<workgroup> sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let row = wg.x;
  if (row >= p.rows || p.width == 0u || p.width > 1024u || p.width % 4u != 0u || p.patches == 0u) { return; }
  let width4 = p.width / 4u;
  for (var col = lane; col < width4; col += 256u) {
    var value = vec4<f32>(0.0);
    for (var index = 0u; index < p.patches; index++) {
      value += X[I[row * p.patches + index] * width4 + col] * (1.0 / f32(p.patches));
    }
    value *= sqrt(f32(p.width));
    if (p.standardize != 0u) { value = (value - B[col]) * S[col]; }
    pooled[col] = value;
  }
  workgroupBarrier();

  var sum = 0.0;
  for (var col = lane; col < width4; col += 256u) {
    let x = clamp(pooled[col], vec4<f32>(p.lo), vec4<f32>(p.hi));
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
    Y[at] = clamp(pooled[col], vec4<f32>(p.lo), vec4<f32>(p.hi)) * scale;
  }
}
