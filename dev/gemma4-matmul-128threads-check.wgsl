// Loader fragment for F32 WG32x4/BM28, replacing the production weight/X stores.
// Each of 128 lanes loads eight weights from both its lr and lr+32 rows.
    for (var gg = 0u; gg < 2u; gg++) {
      let wrow = n0 + lr + 32u * gg;
      wv[2u * gg] = vec4<f32>(0.0);
      wv[2u * gg + 1u] = vec4<f32>(0.0);
      if (wrow < N) {
        let v = W[(wrow * K + kb * 32u) / 16u + lq / 2u];
        let s = S[wrow * nb + kb];
        let lo = select(v.xy, v.zw, (lq & 1u) == 1u);
        wv[2u * gg] = sx(lo.x) * s;
        wv[2u * gg + 1u] = sx(lo.y) * s;
      }
    }
    if (lr < 28u) {
      xs[lr * 8u + lq * 2u] = vec4<f32>(xv0);
      xs[lr * 8u + lq * 2u + 1u] = vec4<f32>(xv1);
    }
    for (var gg = 0u; gg < 2u; gg++) {
      let r = lr + 32u * gg;
      ws[r * 8u + ((lq * 2u) ^ sw)] = vec4<f32>(wv[2u * gg]);
      ws[r * 8u + ((lq * 2u + 1u) ^ sw)] = vec4<f32>(wv[2u * gg + 1u]);
    }
