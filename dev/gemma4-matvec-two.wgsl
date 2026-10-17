// Appended to the rendered production Q8/F32 matvec prelude.
var<workgroup> pair_partials: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  if (g.T != 2u) { return; }
  let N = p.N;
  let K = p.K;
  let lane = li & 63u;
  let col = wg.x * 4u + li / 64u;
  let k4 = K / 4u;
  var sum = vec2<f32>(0.0);
  if (col < N) {
    for (var part = lane; part < k4; part += 64u) {
      let packed = W[col * (K / 16u) + part / 4u];
      let word = packed[part & 3u];
      let weight = sx(word) * S[col * (K / 32u) + part / 8u];
      sum += vec2<f32>(dot(X[part], weight), dot(X[k4 + part], weight));
    }
  }
  pair_partials[li] = sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lane < stride) { pair_partials[li] += pair_partials[li + stride]; }
    workgroupBarrier();
  }
  if (lane == 0u && col < N) {
    for (var row = 0u; row < 2u; row++) {
      let at = row * N + col;
      var value = pair_partials[li][row];
      if (p.bias == 1u) { value += B[col]; }
      if (p.mode == 1u) { value += Y[at]; } else if (p.mode == 2u) { value = max(value, 0.0); }
      Y[at] = value;
    }
  }
}
