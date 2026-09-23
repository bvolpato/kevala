// Experimental two-plane activation DP4A. Default BM64/BN64/K32, four rows per thread.
// Integer sums are exact within each 32-value block; scaling and accumulation stay f32.
requires packed_4x8_integer_dot_product;
//#include common

struct P { N: u32, K: u32, mode: u32, bias: u32 }
struct QBlock { scales: vec2<f32>, planes: array<vec2<u32>, 8> }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> Q: array<QBlock>;
@group(0) @binding(3) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> S: array<f32>;
@group(0) @binding(5) var<storage, read> B: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
@group(0) @binding(7) var<storage, read_write> PART: array<f32>;

var<workgroup> xs: array<vec2<u32>, 512>;
var<workgroup> ws: array<u32, {{WS_LEN}}>;
var<workgroup> xscale: array<vec2<f32>, 64>;
var<workgroup> wscale: array<f32, {{BN}}>;

//#include splits

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let T = g.T;
  let N = p.N;
  let K = p.K;
  let nb = K / 32u;
  let m0 = wg.y * {{BM}}u;
  let n0 = wg.x * {{BN}}u;
  let splits = mm_splits(T, N, K);
  let per = (nb + splits - 1u) / splits;
  let kb0 = wg.z * per;
  let kb1 = min(nb, kb0 + per);
  let lr = li / 4u;
  let lq = li % 4u;
  let sw = lr & 7u;
  var acc: array<vec4<f32>, {{ACC_LEN}}>;
  var d0: array<vec4<i32>, {{ACC_LEN}}>;
  var d1: array<vec4<i32>, {{ACC_LEN}}>;

  for (var kb = kb0; kb < kb1; kb++) {
    if (lr < {{BM}}u) {
      var a0 = vec2<u32>(0u);
      var a1 = vec2<u32>(0u);
      var scale = vec2<f32>(0.0);
      if (m0 + lr < T) {
        let block = (m0 + lr) * nb + kb;
        a0 = Q[block].planes[lq * 2u];
        a1 = Q[block].planes[lq * 2u + 1u];
        scale = Q[block].scales;
      }
      xs[lr * 8u + lq * 2u] = a0;
      xs[lr * 8u + lq * 2u + 1u] = a1;
      if (lq == 0u) { xscale[lr] = scale; }
    }
    for (var gg = 0u; gg < {{GROUPS}}u; gg++) {
      let r = lr + 64u * gg;
      let col = n0 + r;
      var words = vec2<u32>(0u);
      var scale = 0.0;
      if (col < N) {
        let v = W[(col * K + kb * 32u) / 16u + lq / 2u];
        words = select(v.xy, v.zw, (lq & 1u) == 1u);
        scale = S[col * nb + kb];
      }
      ws[r * 8u + ((lq * 2u) ^ sw)] = words.x;
      ws[r * 8u + ((lq * 2u + 1u) ^ sw)] = words.y;
      if (lq == 0u) { wscale[r] = scale; }
    }
    workgroupBarrier();
    // Clear explicitly: Chrome 149/NVIDIA retained implicit array initialization across
    // loop iterations, carrying integer sums into the next block's different scales.
    for (var ix = 0u; ix < {{ACC_LEN}}u; ix++) {
      d0[ix] = vec4<i32>(0);
      d1[ix] = vec4<i32>(0);
    }
    let nx = lid.x & 7u;
    for (var kq = 0u; kq < 8u; kq++) {
      for (var gg = 0u; gg < {{GROUPS}}u; gg++) {
        let c = lid.x + 64u * gg;
        let b0 = ws[c * 8u + (kq ^ nx)];
        let b1 = ws[(c + 16u) * 8u + (kq ^ nx)];
        let b2 = ws[(c + 32u) * 8u + (kq ^ nx)];
        let b3 = ws[(c + 48u) * 8u + (kq ^ nx)];
        for (var i = 0u; i < {{ROWS}}u; i++) {
          let a = xs[(lid.y + 16u * i) * 8u + kq];
          let ix = i * {{GROUPS}}u + gg;
          d0[ix] += vec4<i32>(dot4I8Packed(a.x, b0), dot4I8Packed(a.x, b1), dot4I8Packed(a.x, b2), dot4I8Packed(a.x, b3));
          d1[ix] += vec4<i32>(dot4I8Packed(a.y, b0), dot4I8Packed(a.y, b1), dot4I8Packed(a.y, b2), dot4I8Packed(a.y, b3));
        }
      }
    }
    for (var gg = 0u; gg < {{GROUPS}}u; gg++) {
      let c = lid.x + 64u * gg;
      let sw4 = vec4<f32>(wscale[c], wscale[c + 16u], wscale[c + 32u], wscale[c + 48u]);
      for (var i = 0u; i < {{ROWS}}u; i++) {
        let scale = xscale[lid.y + 16u * i];
        let ix = i * {{GROUPS}}u + gg;
        acc[ix] += (vec4<f32>(d0[ix]) * scale.x + vec4<f32>(d1[ix]) * scale.y) * sw4;
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
