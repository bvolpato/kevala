// Y[T, N] = X[T, K] . W[N, K]^T (+ bias), with Q8, BF16, or F32 weights.
// A 32 x 8 workgroup covers {{PHYSICAL_BM}} rows and 64 columns. It retains
// matmul's named vec2 accumulators and stages two K32 blocks per barrier pair.
// This variant requires groups = 1 and 32 KiB of workgroup storage for F32.
//#if F16
enable f16;
//#endif
//#include common

struct P { N: u32, K: u32, mode: u32, bias: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> S: array<f32>;
@group(0) @binding(5) var<storage, read> B: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
@group(0) @binding(7) var<storage, read_write> PART: array<f32>;

var<workgroup> xs: array<vec4<{{TILE}}>, 1024>; // [row][k/4]
var<workgroup> ws: array<vec4<{{TILE}}>, 1024>; // [column][(k/4) ^ (column & 7)]

//#if WEIGHT_Q8
fn sx(w: u32) -> vec4<f32> {
  let bytes = vec4<i32>(bitcast<i32>(w << 24u), bitcast<i32>(w << 16u), bitcast<i32>(w << 8u), bitcast<i32>(w)) >> vec4<u32>(24u);
  return vec4<f32>(bytes);
}
//#endif
//#if WEIGHT_BF16
fn bf16(w: vec2<u32>) -> vec4<f32> {
  return bitcast<vec4<f32>>(vec4<u32>(w.x << 16u, w.x & 0xffff0000u, w.y << 16u, w.y & 0xffff0000u));
}
//#endif

// Each call has a constant kq. The compiler can inline the shared loads while
// keeping the same increasing-K dot and accumulation order as matmul.
fn accumulate(kq: u32, c: u32, y: u32,
  acc0: ptr<function, vec2<f32>>, acc1: ptr<function, vec2<f32>>,
//#if ROWS_GE_2
  acc2: ptr<function, vec2<f32>>, acc3: ptr<function, vec2<f32>>,
//#endif
//#if ROWS_GE_3
  acc4: ptr<function, vec2<f32>>, acc5: ptr<function, vec2<f32>>,
//#endif
//#if ROWS_GE_4
  acc6: ptr<function, vec2<f32>>,
//#if ACC7
  acc7: ptr<function, vec2<f32>>,
//#endif
//#endif
) {
  let nx = c & 7u;
  let b0 = vec4<f32>(ws[c * 16u + (kq ^ nx)]);
  let b1 = vec4<f32>(ws[(c + 32u) * 16u + (kq ^ nx)]);
  let a0 = vec4<f32>(xs[y * 16u + kq]);
  *acc0 += vec2<f32>(dot(a0, b0), dot(a0, b1));
  let a1 = vec4<f32>(xs[(y + 8u) * 16u + kq]);
  *acc1 += vec2<f32>(dot(a1, b0), dot(a1, b1));
//#if ROWS_GE_2
  let a2 = vec4<f32>(xs[(y + 16u) * 16u + kq]);
  *acc2 += vec2<f32>(dot(a2, b0), dot(a2, b1));
  let a3 = vec4<f32>(xs[(y + 24u) * 16u + kq]);
  *acc3 += vec2<f32>(dot(a3, b0), dot(a3, b1));
//#endif
//#if ROWS_GE_3
  let a4 = vec4<f32>(xs[(y + 32u) * 16u + kq]);
  *acc4 += vec2<f32>(dot(a4, b0), dot(a4, b1));
  let a5 = vec4<f32>(xs[(y + 40u) * 16u + kq]);
  *acc5 += vec2<f32>(dot(a5, b0), dot(a5, b1));
//#endif
//#if ROWS_GE_4
  let a6 = vec4<f32>(xs[(y + 48u) * 16u + kq]);
  *acc6 += vec2<f32>(dot(a6, b0), dot(a6, b1));
//#if ACC7
  let a7 = vec4<f32>(xs[(y + 56u) * 16u + kq]);
  *acc7 += vec2<f32>(dot(a7, b0), dot(a7, b1));
//#endif
//#endif
}

fn mm_store(value: f32, row: u32, col: u32, bias: f32, T: u32, N: u32, splits: u32, z: u32) {
  if (row >= T) { return; }
  let o = row * N + col;
  if (splits > 1u) {
    PART[z * T * N + o] = value;
    return;
  }
  var v = value;
  v += bias;
  if (p.mode == 1u) { v += Y[o]; } else if (p.mode == 2u) { v = max(v, 0.0); }
  Y[o] = v;
}

//#include splits

@compute @workgroup_size(32, 8)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let T = g.T;
//#if SHAPE
  let N = {{N}}u;
  let K = {{K}}u;
//#else
  let N = p.N;
  let K = p.K;
//#endif
  let nb = K / 32u;
  let m0 = wg.y * {{PHYSICAL_BM}}u;
  let n0 = wg.x * 64u;
  let splits = mm_splits(T, N, K);
  let per = (nb + splits - 1u) / splits;
  let kb0 = wg.z * per;
  let kb1 = min(nb, kb0 + per);
  var acc0 = vec2<f32>(0.0);
  var acc1 = vec2<f32>(0.0);
//#if ROWS_GE_2
  var acc2 = vec2<f32>(0.0);
  var acc3 = vec2<f32>(0.0);
//#endif
//#if ROWS_GE_3
  var acc4 = vec2<f32>(0.0);
  var acc5 = vec2<f32>(0.0);
//#endif
//#if ROWS_GE_4
  var acc6 = vec2<f32>(0.0);
//#if ACC7
  var acc7 = vec2<f32>(0.0);
//#endif
//#endif

  let lr = li / 4u;
  let lq = li % 4u;
  let xrow = m0 + lr;
  let wrow = n0 + lr;
  let sw = lr & 7u;
  for (var block = kb0; block < kb1; block += 2u) {
    for (var step = 0u; step < 2u; step++) {
      let kb = block + step;
      var xv0 = vec4<f32>(0.0);
      var xv1 = vec4<f32>(0.0);
      if (kb < kb1 && lr < {{PHYSICAL_BM}}u && xrow < T) {
        let base = (xrow * K + kb * 32u) / 4u + lq * 2u;
        xv0 = X[base];
        xv1 = X[base + 1u];
      }
      var wv0 = vec4<f32>(0.0);
      var wv1 = vec4<f32>(0.0);
      if (kb < kb1 && wrow < N) {
//#if WEIGHT_Q8
        let v = W[(wrow * K + kb * 32u) / 16u + lq / 2u];
        let scale = S[wrow * nb + kb];
        let lo = select(v.xy, v.zw, (lq & 1u) == 1u);
        wv0 = sx(lo.x) * scale;
        wv1 = sx(lo.y) * scale;
//#endif
//#if WEIGHT_BF16
        let v = W[(wrow * K + kb * 32u) / 8u + lq];
        wv0 = bf16(v.xy);
        wv1 = bf16(v.zw);
//#endif
//#if WEIGHT_F32
        let base = (wrow * K + kb * 32u) / 4u + lq * 2u;
        wv0 = bitcast<vec4<f32>>(W[base]);
        wv1 = bitcast<vec4<f32>>(W[base + 1u]);
//#endif
      }
      if (lr < {{PHYSICAL_BM}}u) {
        xs[lr * 16u + step * 8u + lq * 2u] = vec4<{{TILE}}>(xv0);
        xs[lr * 16u + step * 8u + lq * 2u + 1u] = vec4<{{TILE}}>(xv1);
      }
      ws[lr * 16u + ((step * 8u + lq * 2u) ^ sw)] = vec4<{{TILE}}>(wv0);
      ws[lr * 16u + ((step * 8u + lq * 2u + 1u) ^ sw)] = vec4<{{TILE}}>(wv1);
    }
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
//#endif
//#endif
    );
    if (block + 1u < kb1) {
      accumulate(8u, lid.x, lid.y, &acc0, &acc1,
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
//#endif
//#endif
      );
      accumulate(9u, lid.x, lid.y, &acc0, &acc1,
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
//#endif
//#endif
      );
      accumulate(10u, lid.x, lid.y, &acc0, &acc1,
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
//#endif
//#endif
      );
      accumulate(11u, lid.x, lid.y, &acc0, &acc1,
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
//#endif
//#endif
      );
      accumulate(12u, lid.x, lid.y, &acc0, &acc1,
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
//#endif
//#endif
      );
      accumulate(13u, lid.x, lid.y, &acc0, &acc1,
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
//#endif
//#endif
      );
      accumulate(14u, lid.x, lid.y, &acc0, &acc1,
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
//#endif
//#endif
      );
      accumulate(15u, lid.x, lid.y, &acc0, &acc1,
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
//#endif
//#endif
      );
    }
    workgroupBarrier();
  }

  let row = m0 + lid.y;
  {
    let col = n0 + lid.x;
    if (col < N) {
      var bias = 0.0;
      if (p.bias == 1u) { bias = B[col]; }
      mm_store(acc0.x, row, col, bias, T, N, splits, wg.z);
      mm_store(acc1.x, row + 8u, col, bias, T, N, splits, wg.z);
//#if ROWS_GE_2
      mm_store(acc2.x, row + 16u, col, bias, T, N, splits, wg.z);
      mm_store(acc3.x, row + 24u, col, bias, T, N, splits, wg.z);
//#endif
//#if ROWS_GE_3
      mm_store(acc4.x, row + 32u, col, bias, T, N, splits, wg.z);
      mm_store(acc5.x, row + 40u, col, bias, T, N, splits, wg.z);
//#endif
//#if ROWS_GE_4
      mm_store(acc6.x, row + 48u, col, bias, T, N, splits, wg.z);
//#if ACC7
      mm_store(acc7.x, row + 56u, col, bias, T, N, splits, wg.z);
//#endif
//#endif
    }
  }
  {
    let col = n0 + lid.x + 32u;
    if (col < N) {
      var bias = 0.0;
      if (p.bias == 1u) { bias = B[col]; }
      mm_store(acc0.y, row, col, bias, T, N, splits, wg.z);
      mm_store(acc1.y, row + 8u, col, bias, T, N, splits, wg.z);
//#if ROWS_GE_2
      mm_store(acc2.y, row + 16u, col, bias, T, N, splits, wg.z);
      mm_store(acc3.y, row + 24u, col, bias, T, N, splits, wg.z);
//#endif
//#if ROWS_GE_3
      mm_store(acc4.y, row + 32u, col, bias, T, N, splits, wg.z);
      mm_store(acc5.y, row + 40u, col, bias, T, N, splits, wg.z);
//#endif
//#if ROWS_GE_4
      mm_store(acc6.y, row + 48u, col, bias, T, N, splits, wg.z);
//#if ACC7
      mm_store(acc7.y, row + 56u, col, bias, T, N, splits, wg.z);
//#endif
//#endif
    }
  }
}
