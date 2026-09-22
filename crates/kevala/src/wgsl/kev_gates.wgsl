// DeltaNet gates: a and b projections (f32, interleaved [D/4, 2 * heads, 4]) -> decay =
// exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)
//#include kev_common

@group(0) @binding(1) var<storage, read> H: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> Wab: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> DT: array<f32>;
@group(0) @binding(4) var<storage, read> NA: array<f32>;
@group(0) @binding(5) var<storage, read_write> AB: array<f32>;
// 256 threads per token: the 2 * heads outputs x the slices of the D-long dot product that fit
// (32 outputs x 8 slices of 32 vec4s for Kev-0.8B), so the dependent chain is D / 4 / slices loads
// long rather than D / 4 and a short request still fills the GPU
const OUTS = 2u * LIN_HEADS;
const SLICES = 256u / OUTS;
const PER = (HIDDEN / 4u + SLICES - 1u) / SLICES;
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = wg.x;
  if (t >= g.T) { return; }
  let j = li % OUTS;
  let slice = li / OUTS;
  var acc = vec4<f32>(0.0);
  if (slice < SLICES) {
    for (var k = PER * slice; k < min(HIDDEN / 4u, PER * slice + PER); k++) { acc += H[t * (HIDDEN / 4u) + k] * Wab[k * OUTS + j]; }
  }
  part[li] = acc.x + acc.y + acc.z + acc.w;
  workgroupBarrier();
  if (li >= OUTS) { return; }
  var v = 0.0;
  for (var q = 0u; q < SLICES; q++) { v += part[q * OUTS + j]; }
  var o: f32;
  if (j < LIN_HEADS) {
    let x = v + DT[j];
    let sp = select(log(1.0 + exp(x)), x, x > 20.0);
    o = exp(NA[j] * sp);
  } else {
    o = 1.0 / (1.0 + exp(-v));
  }
  AB[t * OUTS + j] = o;
}
