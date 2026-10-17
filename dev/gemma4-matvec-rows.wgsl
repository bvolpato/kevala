// Appended to the rendered F32 matvec_gelu bindings, sx and GELU helpers.
// Each vec4 component retains one original lane64 dot and reduction sequence.
var<workgroup> partials: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  if (g.T != 2u || wg.y != 0u) { return; }
  let N = p.N;
  let K = p.K;
  let lane = li & 63u;
  let col = wg.x * 4u + li / 64u;
  var values = vec4<f32>(0.0);
  if (col < N) {
    for (var part = lane; part < K / 4u; part += 64u) {
      let address = col * (K / 16u) + part / 4u;
      let gate_pack = G[address];
      let up_pack = UP[address];
      let gate_word = gate_pack[part & 3u];
      let up_word = up_pack[part & 3u];
      let gate_weight = sx(gate_word) * GS[col * (K / 32u) + part / 8u];
      let up_weight = sx(up_word) * US[col * (K / 32u) + part / 8u];
      let input0 = X[part];
      let input1 = X[K / 4u + part];
      values += vec4<f32>(dot(input0, gate_weight), dot(input0, up_weight), dot(input1, gate_weight), dot(input1, up_weight));
    }
  }
  partials[li] = values;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lane < stride) { partials[li] += partials[li + stride]; }
    workgroupBarrier();
  }
  if (lane == 0u && col < N) {
    let result = partials[li];
    Y[col] = gelu(result.x) * result.y;
    Y[N + col] = gelu(result.z) * result.w;
  }
}
