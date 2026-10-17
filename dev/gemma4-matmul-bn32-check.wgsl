// Loader fragment for WG16x8/BN32, preserving the original Q8/F32 operands.
    xv0 = vec4<f32>(0.0);
    xv1 = vec4<f32>(0.0);
    if (lr < {{BM}}u && xrow < T) {
      let base = (xrow * K + kb * 32u) / 4u + lq * 2u;
      xv0 = X[base];
      xv1 = X[base + 1u];
    }
    wv[0] = vec4<f32>(0.0);
    wv[1] = vec4<f32>(0.0);
    let wrow = n0 + lr;
    if (wrow < N) {
      let v = W[(wrow * K + kb * 32u) / 16u + lq / 2u];
      let s = S[wrow * nb + kb];
      let lo = select(v.xy, v.zw, (lq & 1u) == 1u);
      wv[0] = sx(lo.x) * s;
      wv[1] = sx(lo.y) * s;
    }
    if (lr < {{BM}}u) {
      xs[lr * 8u + lq * 2u] = xv0;
      xs[lr * 8u + lq * 2u + 1u] = xv1;
    }
    if (lr + 32u < {{BM}}u) {
      var next0 = vec4<f32>(0.0);
      var next1 = vec4<f32>(0.0);
      if (xrow + 32u < T) {
        let base = ((xrow + 32u) * K + kb * 32u) / 4u + lq * 2u;
        next0 = X[base];
        next1 = X[base + 1u];
      }
      xs[(lr + 32u) * 8u + lq * 2u] = next0;
      xs[(lr + 32u) * 8u + lq * 2u + 1u] = next1;
    }
    ws[lr * 8u + ((lq * 2u) ^ sw)] = wv[0];
    ws[lr * 8u + ((lq * 2u + 1u) ^ sw)] = wv[1];
