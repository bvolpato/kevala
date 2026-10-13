struct P { rows: u32, heads: u32, kv_heads: u32, dim: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> S: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Y: array<vec4<f32>>;
var<workgroup> sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let row = wg.x;
  let head = wg.y;
  let base = (head * p.rows + row) * p.rows;
  var maximum = -1e30;
  for (var key = lane; key < p.rows; key += 256u) { maximum = max(maximum, S[base + key]); }
  sums[lane] = maximum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (lane < stride) { sums[lane] = max(sums[lane], sums[lane + stride]); }
    workgroupBarrier();
  }
  maximum = sums[0];
  workgroupBarrier();
  var total = 0.0;
  for (var key = lane; key < p.rows; key += 256u) {
    let weight = exp(S[base + key] - maximum);
    S[base + key] = weight;
    total += weight;
  }
  storageBarrier();
  sums[lane] = total;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (lane < stride) { sums[lane] += sums[lane + stride]; }
    workgroupBarrier();
  }
  let inverse = 1.0 / sums[0];
  let head4 = p.dim / 4u;
  let kv_head = head / (p.heads / p.kv_heads);
  for (var col = lane; col < head4; col += 256u) {
    var value = vec4<f32>(0.0);
    for (var key = 0u; key < p.rows; key++) {
      value += V[(key * p.kv_heads + kv_head) * head4 + col] * (S[base + key] * inverse);
    }
    Y[(row * p.heads + head) * head4 + col] = value;
  }
}
