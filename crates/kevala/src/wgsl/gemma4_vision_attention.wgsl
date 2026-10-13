struct P { rows: u32, heads: u32, kv_heads: u32, dim: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> S: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Y: array<vec4<f32>>;
var<workgroup> sums: array<f32, 256>;
var<workgroup> partial_values: array<vec4<f32>, 256>;

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
  let key_lanes = 256u / head4;
  let active_lanes = key_lanes * head4;
  let col = lane % head4;
  let key_lane = lane / head4;
  var value = vec4<f32>(0.0);
  if (lane < active_lanes) {
    for (var key = key_lane; key < p.rows; key += key_lanes) {
      value += V[(key * p.kv_heads + kv_head) * head4 + col] * (S[base + key] * inverse);
    }
  }
  partial_values[lane] = value;
  workgroupBarrier();
  if (lane < head4) {
    var result = vec4<f32>(0.0);
    for (var contributor = 0u; contributor < key_lanes; contributor++) {
      result += partial_values[contributor * head4 + lane];
    }
    Y[(row * p.heads + head) * head4 + lane] = result;
  }
}
