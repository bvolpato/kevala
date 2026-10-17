// Appended to the rendered production Q8/F16 matvec prelude and weight tile.
var<workgroup> pair_inputs: array<vec4<f16>, 32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  if (g.T != 2u) { return; }
  let N = p.N;
  let K = p.K;
  let nb = K / 32u;
  let splits = mm_splits(g.T, N, K);
  let per = (nb + splits - 1u) / splits;
  let load_row = li / 4u;
  let load_col = li & 3u;
  let load_output = wg.x * 16u + load_row;
  let row = li / 16u;
  let output_col = li & 15u;
  var total = 0.0;
  for (var split = 0u; split < splits; split++) {
    let kb0 = split * per;
    let kb1 = min(nb, kb0 + per);
    var partial = 0.0;
    for (var kb_base = kb0; kb_base < kb1; kb_base += 2u) {
      for (var half = 0u; half < 2u; half++) {
        let kb = kb_base + half;
        let at = half * 8u + load_col * 2u;
        var w0 = vec4<f32>(0.0);
        var w1 = vec4<f32>(0.0);
        if (load_output < N && kb < kb1) {
          let packed = W[(load_output * K + kb * 32u) / 16u + load_col / 2u];
          let words = select(packed.xy, packed.zw, (load_col & 1u) == 1u);
          let scale = S[load_output * nb + kb];
          w0 = sx(words.x) * scale;
          w1 = sx(words.y) * scale;
        }
        let swizzle = load_row & 7u;
        weights[load_row * 16u + (at ^ swizzle)] = vec4<f16>(w0);
        weights[load_row * 16u + ((at + 1u) ^ swizzle)] = vec4<f16>(w1);
        if (li < 8u) {
          let input_row = li / 4u;
          var x0 = vec4<f32>(0.0);
          var x1 = vec4<f32>(0.0);
          if (kb < kb1) {
            let base = (input_row * K + kb * 32u) / 4u + load_col * 2u;
            x0 = X[base];
            x1 = X[base + 1u];
          }
          pair_inputs[input_row * 16u + at] = vec4<f16>(x0);
          pair_inputs[input_row * 16u + at + 1u] = vec4<f16>(x1);
        }
      }
      workgroupBarrier();
      if (li < 32u) {
        let count = min(16u, (kb1 - kb_base) * 8u);
        for (var kq = 0u; kq < count; kq++) {
          let input = vec4<f32>(pair_inputs[row * 16u + kq]);
          let weight = vec4<f32>(weights[output_col * 16u + (kq ^ (output_col & 7u))]);
          partial += dot(input, weight);
        }
      }
      workgroupBarrier();
    }
    // Match the production ascending split reduction and unsplit signed-zero path.
    if (splits == 1u) { total = partial; } else { total += partial; }
  }
  let col = wg.x * 16u + output_col;
  if (li < 32u && col < N) {
    let at = row * N + col;
    if (splits == 1u) {
      var bias = 0.0;
      if (p.bias == 1u) { bias = B[col]; }
      total += bias;
    } else if (p.bias == 1u) { total += B[col]; }
    if (p.mode == 1u) { total += Y[at]; } else if (p.mode == 2u) { total = max(total, 0.0); }
    Y[at] = total;
  }
}
