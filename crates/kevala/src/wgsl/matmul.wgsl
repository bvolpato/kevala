// Y[T, N] = X[T, K] . W[N, K]^T (+ bias), W int8 in u32 words with one f32 scale per 32 weights.
//
// A workgroup of 16 x 16 threads covers {{BM}} rows ({{ROWS}} per thread) and {{BN}} columns
// ({{GROUPS}} groups of 64); thread (x, y) owns rows y + 16 i and columns x + 16 j + 64 g, so
// output stores coalesce. Each step stages one quantization block of K (32 values) for X and W
// in workgroup memory as {{TILE}}; products and sums stay f32. The weight tile is XOR-swizzled
// so neighbouring threads read different banks, and every thread writes whole vectors
// (sub-vector writes from several threads race on some GPUs).
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

var<workgroup> xs: array<vec4<{{TILE}}>, 512>; // [m][k/4]
var<workgroup> ws: array<vec4<{{TILE}}>, {{WS_LEN}}>; // [n][(k/4) ^ (n & 7)], swizzled

// four int8 weights, sign-extended
fn sx(w: u32) -> vec4<f32> {
  return vec4<f32>(vec4<i32>(bitcast<i32>(w << 24u), bitcast<i32>(w << 16u), bitcast<i32>(w << 8u), bitcast<i32>(w)) >> vec4<u32>(24u));
}

//#include splits

@compute @workgroup_size(16, 16)
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
  let m0 = wg.y * {{BM}}u;
  let n0 = wg.x * {{BN}}u;
  // split-K: this workgroup covers blocks kb0..kb1 and, when split, writes a partial tile
  let splits = mm_splits(T, N, K);
  let per = (nb + splits - 1u) / splits;
  let kb0 = wg.z * per;
  let kb1 = min(nb, kb0 + per);
  var acc: array<vec4<f32>, {{ACC_LEN}}>; // [i][g]: row y + 16 i, columns x + 16 j + 64 g for j < 4

  // loaders: 4 threads per row, 8 values each; X rows 0..{{BM}}, W rows lr + 64 g
  let lr = li / 4u;
  let lq = li % 4u;
  let xrow = m0 + lr;
  let sw = lr & 7u;
  var xv0 = vec4<f32>(0.0);
  var xv1 = vec4<f32>(0.0);
  var wv: array<vec4<f32>, {{WV_LEN}}>;
  for (var kb = kb0; kb < kb1; kb++) {
    xv0 = vec4<f32>(0.0);
    xv1 = vec4<f32>(0.0);
    if (lr < {{BM}}u && xrow < T) {
      let base = (xrow * K + kb * 32u) / 4u + lq * 2u;
      xv0 = X[base];
      xv1 = X[base + 1u];
    }
    for (var gg = 0u; gg < {{GROUPS}}u; gg++) {
      let wrow = n0 + lr + 64u * gg;
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
    if (lr < {{BM}}u) {
      xs[lr * 8u + lq * 2u] = vec4<{{TILE}}>(xv0);
      xs[lr * 8u + lq * 2u + 1u] = vec4<{{TILE}}>(xv1);
    }
    for (var gg = 0u; gg < {{GROUPS}}u; gg++) {
      let r = lr + 64u * gg;
      ws[r * 8u + ((lq * 2u) ^ sw)] = vec4<{{TILE}}>(wv[2u * gg]);
      ws[r * 8u + ((lq * 2u + 1u) ^ sw)] = vec4<{{TILE}}>(wv[2u * gg + 1u]);
    }
    workgroupBarrier();
    let nx = lid.x & 7u;
    for (var kq = 0u; kq < 8u; kq++) {
      for (var gg = 0u; gg < {{GROUPS}}u; gg++) {
        let c = lid.x + 64u * gg;
        let b0 = vec4<f32>(ws[c * 8u + (kq ^ nx)]);
        let b1 = vec4<f32>(ws[(c + 16u) * 8u + (kq ^ nx)]);
        let b2 = vec4<f32>(ws[(c + 32u) * 8u + (kq ^ nx)]);
        let b3 = vec4<f32>(ws[(c + 48u) * 8u + (kq ^ nx)]);
        for (var i = 0u; i < {{ROWS}}u; i++) {
          let a = vec4<f32>(xs[(lid.y + 16u * i) * 8u + kq]);
          acc[i * {{GROUPS}}u + gg] += vec4<f32>(dot(a, b0), dot(a, b1), dot(a, b2), dot(a, b3));
        }
      }
    }
    workgroupBarrier();
  }

  for (var gg = 0u; gg < {{GROUPS}}u; gg++) {
    for (var j = 0u; j < 4u; j++) {
      let col = n0 + lid.x + 16u * j + 64u * gg;
      if (col >= N) { break; }
      var bias = 0.0;
      if (p.bias == 1u) { bias = B[col]; }
      for (var i = 0u; i < {{ROWS}}u; i++) {
        let row = m0 + lid.y + 16u * i;
        if (row >= T) { break; }
        let o = row * N + col;
        let values = acc[i * {{GROUPS}}u + gg];
        let v0 = values[j];
        if (splits > 1u) {
          PART[wg.z * T * N + o] = v0;
          continue;
        }
        var v = v0 + bias;
        if (p.mode == 1u) { v += Y[o]; } else if (p.mode == 2u) { v = max(v, 0.0); }
        Y[o] = v;
      }
    }
  }
}
