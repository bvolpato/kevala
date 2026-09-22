// DeltaNet gates: a and b projections (f32, interleaved [D/4, 2*heads, 4]) -> decay = exp(-exp(A_log) *
// softplus(a + dt_bias)), beta = sigmoid(b)
//#include kev_common

@group(0) @binding(1) var<storage, read> H: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> Wab: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> DT: array<f32>;
@group(0) @binding(4) var<storage, read> NA: array<f32>;
@group(0) @binding(5) var<storage, read_write> AB: array<f32>;
@compute @workgroup_size(2u * LIN_HEADS)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) j: u32) {
  let t = wg.x;
  if (t >= g.T) { return; }
  var acc = vec4<f32>(0.0);
  for (var k = 0u; k < HIDDEN / 4u; k++) { acc += H[t * (HIDDEN / 4u) + k] * Wab[k * (2u * LIN_HEADS) + j]; }
  let v = acc.x + acc.y + acc.z + acc.w;
  var o: f32;
  if (j < LIN_HEADS) {
    let x = v + DT[j];
    let sp = select(log(1.0 + exp(x)), x, x > 20.0);
    o = exp(NA[j] * sp);
  } else {
    o = 1.0 / (1.0 + exp(-v));
  }
  AB[t * (2u * LIN_HEADS) + j] = o;
}
