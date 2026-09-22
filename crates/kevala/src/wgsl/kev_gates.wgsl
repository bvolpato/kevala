// DeltaNet gates: a and b projections (f32, stacked [32, D]) -> decay = exp(-exp(A_log) *
// softplus(a + dt_bias)), beta = sigmoid(b)
//#include kev_common

@group(0) @binding(1) var<storage, read> H: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> Wab: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> DT: array<f32>;
@group(0) @binding(4) var<storage, read> NA: array<f32>;
@group(0) @binding(5) var<storage, read_write> AB: array<f32>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) j: u32) {
  let t = wg.x;
  if (t >= g.T) { return; }
  var acc = vec4<f32>(0.0);
  for (var k = 0u; k < 256u; k++) { acc += H[t * 256u + k] * Wab[j * 256u + k]; }
  let v = acc.x + acc.y + acc.z + acc.w;
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
