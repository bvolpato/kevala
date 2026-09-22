// DeltaNet gates: a and b projections (f32, interleaved [D/4, 32, 4]) -> decay = exp(-exp(A_log) *
// softplus(a + dt_bias)), beta = sigmoid(b)
//#include kev_common

@group(0) @binding(1) var<storage, read> H: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> Wab: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> DT: array<f32>;
@group(0) @binding(4) var<storage, read> NA: array<f32>;
@group(0) @binding(5) var<storage, read_write> AB: array<f32>;
// 256 threads per token: 32 outputs x 8 slices of the 1024-long dot product (32 vec4s each), so
// the dependent chain is 32 loads long rather than 256 and a short request still fills the GPU
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = wg.x;
  if (t >= g.T) { return; }
  let j = li % 32u;
  let slice = li / 32u;
  var acc = vec4<f32>(0.0);
  for (var k = 32u * slice; k < 32u * slice + 32u; k++) { acc += H[t * 256u + k] * Wab[k * 32u + j]; }
  part[li] = acc.x + acc.y + acc.z + acc.w;
  workgroupBarrier();
  if (li >= 32u) { return; }
  var v = 0.0;
  for (var q = 0u; q < 8u; q++) { v += part[q * 32u + j]; }
  var o: f32;
  if (j < 16u) {
    let x = v + DT[j];
    let sp = select(log(1.0 + exp(x)), x, x > 20.0);
    o = exp(NA[j] * sp);
  } else {
    o = 1.0 / (1.0 + exp(-v));
  }
  AB[t * 32u + j] = o;
}
