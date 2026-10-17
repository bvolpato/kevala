// Paired Q8 gate/up projections followed by Gemma's GELU-tanh gate. The two
// projections retain independent ordered FP32 accumulators. groups must be 1.
// One K32 input tile and two weight tiles use 24 KiB F32 or 12 KiB F16 shared memory.
//#if F16
enable f16;
//#endif
//#include common

struct P { N: u32, K: u32, splitT: u32, _pad: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> WG: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> SG: array<f32>;
@group(0) @binding(5) var<storage, read> WU: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> SU: array<f32>;
@group(0) @binding(7) var<storage, read_write> Y: array<f32>;
@group(0) @binding(8) var<storage, read_write> PART: array<f32>;

var<workgroup> xs: array<vec4<{{TILE}}>, {{MLP_XS_LEN}}>;
var<workgroup> gs: array<vec4<{{TILE}}>, 512>;
var<workgroup> us: array<vec4<{{TILE}}>, 512>;

fn sx(w: u32) -> vec4<f32> {
  let bytes = vec4<i32>(bitcast<i32>(w << 24u), bitcast<i32>(w << 16u), bitcast<i32>(w << 8u), bitcast<i32>(w)) >> vec4<u32>(24u);
  return vec4<f32>(bytes);
}

fn gelu(x: f32) -> f32 {
  if (x >= 10.0) { return x; }
  if (x <= -10.0) { return 0.0; }
  let c = 0.7978845608028654;
  return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

fn accumulate(kq: u32, c: u32, y: u32,
  acc0: ptr<function, vec4<f32>>, acc1: ptr<function, vec4<f32>>,
//#if ROWS_GE_2
  acc2: ptr<function, vec4<f32>>, acc3: ptr<function, vec4<f32>>,
//#endif
//#if ROWS_GE_3
  acc4: ptr<function, vec4<f32>>, acc5: ptr<function, vec4<f32>>,
//#endif
//#if ROWS_GE_4
  acc6: ptr<function, vec4<f32>>,
//#if ACC7
  acc7: ptr<function, vec4<f32>>,
//#if TALL104
  acc8: ptr<function, vec4<f32>>,
  acc9: ptr<function, vec4<f32>>,
  acc10: ptr<function, vec4<f32>>,
  acc11: ptr<function, vec4<f32>>,
  acc12: ptr<function, vec4<f32>>,
//#endif
//#endif
//#endif
) {
  let nx = c & 7u;
  let g0 = vec4<f32>(gs[c * 8u + (kq ^ nx)]);
  let g1 = vec4<f32>(gs[(c + 32u) * 8u + (kq ^ nx)]);
  let u0 = vec4<f32>(us[c * 8u + (kq ^ nx)]);
  let u1 = vec4<f32>(us[(c + 32u) * 8u + (kq ^ nx)]);
  let a0 = vec4<f32>(xs[y * 8u + kq]);
  *acc0 += vec4<f32>(dot(a0, g0), dot(a0, g1), dot(a0, u0), dot(a0, u1));
  let a1 = vec4<f32>(xs[(y + 8u) * 8u + kq]);
  *acc1 += vec4<f32>(dot(a1, g0), dot(a1, g1), dot(a1, u0), dot(a1, u1));
//#if ROWS_GE_2
  let a2 = vec4<f32>(xs[(y + 16u) * 8u + kq]);
  *acc2 += vec4<f32>(dot(a2, g0), dot(a2, g1), dot(a2, u0), dot(a2, u1));
  let a3 = vec4<f32>(xs[(y + 24u) * 8u + kq]);
  *acc3 += vec4<f32>(dot(a3, g0), dot(a3, g1), dot(a3, u0), dot(a3, u1));
//#endif
//#if ROWS_GE_3
  let a4 = vec4<f32>(xs[(y + 32u) * 8u + kq]);
  *acc4 += vec4<f32>(dot(a4, g0), dot(a4, g1), dot(a4, u0), dot(a4, u1));
  let a5 = vec4<f32>(xs[(y + 40u) * 8u + kq]);
  *acc5 += vec4<f32>(dot(a5, g0), dot(a5, g1), dot(a5, u0), dot(a5, u1));
//#endif
//#if ROWS_GE_4
  let a6 = vec4<f32>(xs[(y + 48u) * 8u + kq]);
  *acc6 += vec4<f32>(dot(a6, g0), dot(a6, g1), dot(a6, u0), dot(a6, u1));
//#if ACC7
  let a7 = vec4<f32>(xs[(y + 56u) * 8u + kq]);
  *acc7 += vec4<f32>(dot(a7, g0), dot(a7, g1), dot(a7, u0), dot(a7, u1));
//#if TALL104
  let a8 = vec4<f32>(xs[(y + 64u) * 8u + kq]);
  *acc8 += vec4<f32>(dot(a8, g0), dot(a8, g1), dot(a8, u0), dot(a8, u1));
  let a9 = vec4<f32>(xs[(y + 72u) * 8u + kq]);
  *acc9 += vec4<f32>(dot(a9, g0), dot(a9, g1), dot(a9, u0), dot(a9, u1));
  let a10 = vec4<f32>(xs[(y + 80u) * 8u + kq]);
  *acc10 += vec4<f32>(dot(a10, g0), dot(a10, g1), dot(a10, u0), dot(a10, u1));
  let a11 = vec4<f32>(xs[(y + 88u) * 8u + kq]);
  *acc11 += vec4<f32>(dot(a11, g0), dot(a11, g1), dot(a11, u0), dot(a11, u1));
  let a12 = vec4<f32>(xs[(y + 96u) * 8u + kq]);
  *acc12 += vec4<f32>(dot(a12, g0), dot(a12, g1), dot(a12, u0), dot(a12, u1));
//#endif
//#endif
//#endif
}

fn store_pair(value: vec2<f32>, row: u32, col: u32, T: u32, N: u32, splits: u32, z: u32) {
  if (row >= T || col >= N) { return; }
  let o = row * N + col;
  if (splits > 1u) {
    PART[z * T * N + o] = value.x;
    PART[(splits + z) * T * N + o] = value.y;
    return;
  }
  // The unfused zero-bias matmul also adds positive zero before its output store.
  let gate = value.x + 0.0;
  let up = value.y + 0.0;
  Y[o] = gelu(gate) * up;
}

//#include splits

@compute @workgroup_size(32, 8)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let T = g.T;
  let N = p.N;
  let K = p.K;
  let nb = K / 32u;
  let m0 = wg.y * {{MLP_PHYSICAL_BM}}u;
  let n0 = wg.x * 64u;
  // splitT=0 uses the active row count; splitT=1 retains independent suffix partitions.
  let splitT = select(p.splitT, T, p.splitT == 0u);
  let splits = mm_splits(splitT, N, K);
  let per = (nb + splits - 1u) / splits;
  let kb0 = wg.z * per;
  let kb1 = min(nb, kb0 + per);
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
//#if ROWS_GE_2
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);
//#endif
//#if ROWS_GE_3
  var acc4 = vec4<f32>(0.0);
  var acc5 = vec4<f32>(0.0);
//#endif
//#if ROWS_GE_4
  var acc6 = vec4<f32>(0.0);
//#if ACC7
  var acc7 = vec4<f32>(0.0);
//#if TALL104
  var acc8 = vec4<f32>(0.0);
  var acc9 = vec4<f32>(0.0);
  var acc10 = vec4<f32>(0.0);
  var acc11 = vec4<f32>(0.0);
  var acc12 = vec4<f32>(0.0);
//#endif
//#endif
//#endif
  let lr = li / 4u;
  let lq = li % 4u;
  let xrow = m0 + lr;
  let wrow = n0 + lr;
  let sw = lr & 7u;
  for (var kb = kb0; kb < kb1; kb++) {
    var xv0 = vec4<f32>(0.0);
    var xv1 = vec4<f32>(0.0);
    if (lr < {{MLP_LOAD_ROWS}}u && xrow < T) {
      let base = (xrow * K + kb * 32u) / 4u + lq * 2u;
      xv0 = X[base];
      xv1 = X[base + 1u];
    }
    var gv0 = vec4<f32>(0.0);
    var gv1 = vec4<f32>(0.0);
    var uv0 = vec4<f32>(0.0);
    var uv1 = vec4<f32>(0.0);
    if (wrow < N) {
      let at = (wrow * K + kb * 32u) / 16u + lq / 2u;
      let gate = WG[at];
      let up = WU[at];
      let scaleG = SG[wrow * nb + kb];
      let scaleU = SU[wrow * nb + kb];
      let gateBytes = select(gate.xy, gate.zw, (lq & 1u) == 1u);
      let upBytes = select(up.xy, up.zw, (lq & 1u) == 1u);
      gv0 = sx(gateBytes.x) * scaleG;
      gv1 = sx(gateBytes.y) * scaleG;
      uv0 = sx(upBytes.x) * scaleU;
      uv1 = sx(upBytes.y) * scaleU;
    }
    if (lr < {{MLP_LOAD_ROWS}}u) {
      xs[lr * 8u + lq * 2u] = vec4<{{TILE}}>(xv0);
      xs[lr * 8u + lq * 2u + 1u] = vec4<{{TILE}}>(xv1);
    }
//#if TALL104
    var xv2 = vec4<f32>(0.0);
    var xv3 = vec4<f32>(0.0);
    let xrow2 = xrow + 64u;
    if (lr + 64u < 104u && xrow2 < T) {
      let base2 = (xrow2 * K + kb * 32u) / 4u + lq * 2u;
      xv2 = X[base2];
      xv3 = X[base2 + 1u];
    }
    if (lr + 64u < 104u) {
      xs[(lr + 64u) * 8u + lq * 2u] = vec4<f16>(xv2);
      xs[(lr + 64u) * 8u + lq * 2u + 1u] = vec4<f16>(xv3);
    }
//#endif
    gs[lr * 8u + ((lq * 2u) ^ sw)] = vec4<{{TILE}}>(gv0);
    gs[lr * 8u + ((lq * 2u + 1u) ^ sw)] = vec4<{{TILE}}>(gv1);
    us[lr * 8u + ((lq * 2u) ^ sw)] = vec4<{{TILE}}>(uv0);
    us[lr * 8u + ((lq * 2u + 1u) ^ sw)] = vec4<{{TILE}}>(uv1);
    workgroupBarrier();
    accumulate(0u, lid.x, lid.y, &acc0, &acc1,
//#if ROWS_GE_2
      &acc2, &acc3,
//#endif
//#if ROWS_GE_3
      &acc4, &acc5,
//#endif
//#if ROWS_GE_4
      &acc6,
//#if ACC7
      &acc7,
//#if TALL104
      &acc8,
      &acc9,
      &acc10,
      &acc11,
      &acc12,
//#endif
//#endif
//#endif
    );
    accumulate(1u, lid.x, lid.y, &acc0, &acc1,
//#if ROWS_GE_2
      &acc2, &acc3,
//#endif
//#if ROWS_GE_3
      &acc4, &acc5,
//#endif
//#if ROWS_GE_4
      &acc6,
//#if ACC7
      &acc7,
//#if TALL104
      &acc8,
      &acc9,
      &acc10,
      &acc11,
      &acc12,
//#endif
//#endif
//#endif
    );
    accumulate(2u, lid.x, lid.y, &acc0, &acc1,
//#if ROWS_GE_2
      &acc2, &acc3,
//#endif
//#if ROWS_GE_3
      &acc4, &acc5,
//#endif
//#if ROWS_GE_4
      &acc6,
//#if ACC7
      &acc7,
//#if TALL104
      &acc8,
      &acc9,
      &acc10,
      &acc11,
      &acc12,
//#endif
//#endif
//#endif
    );
    accumulate(3u, lid.x, lid.y, &acc0, &acc1,
//#if ROWS_GE_2
      &acc2, &acc3,
//#endif
//#if ROWS_GE_3
      &acc4, &acc5,
//#endif
//#if ROWS_GE_4
      &acc6,
//#if ACC7
      &acc7,
//#if TALL104
      &acc8,
      &acc9,
      &acc10,
      &acc11,
      &acc12,
//#endif
//#endif
//#endif
    );
    accumulate(4u, lid.x, lid.y, &acc0, &acc1,
//#if ROWS_GE_2
      &acc2, &acc3,
//#endif
//#if ROWS_GE_3
      &acc4, &acc5,
//#endif
//#if ROWS_GE_4
      &acc6,
//#if ACC7
      &acc7,
//#if TALL104
      &acc8,
      &acc9,
      &acc10,
      &acc11,
      &acc12,
//#endif
//#endif
//#endif
    );
    accumulate(5u, lid.x, lid.y, &acc0, &acc1,
//#if ROWS_GE_2
      &acc2, &acc3,
//#endif
//#if ROWS_GE_3
      &acc4, &acc5,
//#endif
//#if ROWS_GE_4
      &acc6,
//#if ACC7
      &acc7,
//#if TALL104
      &acc8,
      &acc9,
      &acc10,
      &acc11,
      &acc12,
//#endif
//#endif
//#endif
    );
    accumulate(6u, lid.x, lid.y, &acc0, &acc1,
//#if ROWS_GE_2
      &acc2, &acc3,
//#endif
//#if ROWS_GE_3
      &acc4, &acc5,
//#endif
//#if ROWS_GE_4
      &acc6,
//#if ACC7
      &acc7,
//#if TALL104
      &acc8,
      &acc9,
      &acc10,
      &acc11,
      &acc12,
//#endif
//#endif
//#endif
    );
    accumulate(7u, lid.x, lid.y, &acc0, &acc1,
//#if ROWS_GE_2
      &acc2, &acc3,
//#endif
//#if ROWS_GE_3
      &acc4, &acc5,
//#endif
//#if ROWS_GE_4
      &acc6,
//#if ACC7
      &acc7,
//#if TALL104
      &acc8,
      &acc9,
      &acc10,
      &acc11,
      &acc12,
//#endif
//#endif
//#endif
    );
    workgroupBarrier();
  }
  let row = m0 + lid.y;
  store_pair(acc0.xz, row + 0u, n0 + lid.x, T, N, splits, wg.z);
  store_pair(acc0.yw, row + 0u, n0 + lid.x + 32u, T, N, splits, wg.z);
  store_pair(acc1.xz, row + 8u, n0 + lid.x, T, N, splits, wg.z);
  store_pair(acc1.yw, row + 8u, n0 + lid.x + 32u, T, N, splits, wg.z);
//#if ROWS_GE_2
  store_pair(acc2.xz, row + 16u, n0 + lid.x, T, N, splits, wg.z);
  store_pair(acc2.yw, row + 16u, n0 + lid.x + 32u, T, N, splits, wg.z);
  store_pair(acc3.xz, row + 24u, n0 + lid.x, T, N, splits, wg.z);
  store_pair(acc3.yw, row + 24u, n0 + lid.x + 32u, T, N, splits, wg.z);
//#endif
//#if ROWS_GE_3
  store_pair(acc4.xz, row + 32u, n0 + lid.x, T, N, splits, wg.z);
  store_pair(acc4.yw, row + 32u, n0 + lid.x + 32u, T, N, splits, wg.z);
  store_pair(acc5.xz, row + 40u, n0 + lid.x, T, N, splits, wg.z);
  store_pair(acc5.yw, row + 40u, n0 + lid.x + 32u, T, N, splits, wg.z);
//#endif
//#if ROWS_GE_4
  store_pair(acc6.xz, row + 48u, n0 + lid.x, T, N, splits, wg.z);
  store_pair(acc6.yw, row + 48u, n0 + lid.x + 32u, T, N, splits, wg.z);
//#if ACC7
  store_pair(acc7.xz, row + 56u, n0 + lid.x, T, N, splits, wg.z);
  store_pair(acc7.yw, row + 56u, n0 + lid.x + 32u, T, N, splits, wg.z);
//#if TALL104
  store_pair(acc8.xz, row + 64u, n0 + lid.x, T, N, splits, wg.z);
  store_pair(acc8.yw, row + 64u, n0 + lid.x + 32u, T, N, splits, wg.z);
  store_pair(acc9.xz, row + 72u, n0 + lid.x, T, N, splits, wg.z);
  store_pair(acc9.yw, row + 72u, n0 + lid.x + 32u, T, N, splits, wg.z);
  store_pair(acc10.xz, row + 80u, n0 + lid.x, T, N, splits, wg.z);
  store_pair(acc10.yw, row + 80u, n0 + lid.x + 32u, T, N, splits, wg.z);
  store_pair(acc11.xz, row + 88u, n0 + lid.x, T, N, splits, wg.z);
  store_pair(acc11.yw, row + 88u, n0 + lid.x + 32u, T, N, splits, wg.z);
  store_pair(acc12.xz, row + 96u, n0 + lid.x, T, N, splits, wg.z);
  store_pair(acc12.yw, row + 96u, n0 + lid.x + 32u, T, N, splits, wg.z);
//#endif
//#endif
//#endif
}
