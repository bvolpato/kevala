// Y[T, N] = X[T, K] . W[N, K]^T (+ bias), W int8 in u32 words with one f32 scale per 32 weights.
//
// Shared Q8 projection kernel with 64-value staging and named accumulator vectors.
// Keeping each row in a named vector avoids dynamic private-array indexing.
// This variant supports one 64-column group per workgroup.
//
// A workgroup covers {{BM}} rows and 64 columns. Each thread owns four columns
// separated by 16, as in the generic kernel. X and W tiles use {{TILE}} while
// every dot product and accumulator stays f32.
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

var<workgroup> xs: array<vec4<{{TILE}}>, 1024>; // [m][k/4], two K blocks
var<workgroup> ws: array<vec4<{{TILE}}>, {{WIDE_WS_LEN}}>; // [n][(k/4) ^ (n & W_SWIZZLE)], two K blocks

// FP16 vectors span two 32-bit banks. Spread full row tiles over all 16 slots;
// shorter tiles keep the original layout, which wins on some adapters.
//#if F16
const W_SWIZZLE: u32 = select(7u, 15u, {{ROWS}}u == 4u);
//#else
const W_SWIZZLE: u32 = 7u;
//#endif

// Four int8 weights as f32, exactly and without int-to-float conversions (Marlin's trick): the
// xor makes each byte b + 128, and a byte in the low mantissa bits of 2^23 reads as 2^23 + b + 128
fn sx(w: u32) -> vec4<f32> {
  let u = w ^ 0x80808080u;
  let bytes = vec4<u32>(u, u >> 8u, u >> 16u, u >> 24u) & vec4<u32>(0xFFu);
  return bitcast<vec4<f32>>(bytes | vec4<u32>(0x4B000000u)) - vec4<f32>(8388736.0);
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
  let splits = mm_splits(T, N, K);
  let per = (nb + splits - 1u) / splits;
  let kb0 = wg.z * per;
  let kb1 = min(nb, kb0 + per);

  // Named vectors let the compiler remove unused rows without private-array indexing.
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  // Four threads per row load eight values. The wide variant stages two 32-value
  // quantization blocks before the barrier.
  let lr = li / 4u;
  let lq = li % 4u;
  let xrow = m0 + lr;
  let sw = lr & W_SWIZZLE;
  var xv0 = vec4<f32>(0.0);
  var xv1 = vec4<f32>(0.0);
  var wv0 = vec4<f32>(0.0);
  var wv1 = vec4<f32>(0.0);
  for (var block = kb0; block < kb1; block += 2u) {
    for (var step = 0u; step < 2u; step++) {
      let kb = block + step;
      xv0 = vec4<f32>(0.0);
      xv1 = vec4<f32>(0.0);
      if (kb < kb1 && lr < {{BM}}u && xrow < T) {
        let base = (xrow * K + kb * 32u) / 4u + lq * 2u;
        xv0 = X[base];
        xv1 = X[base + 1u];
      }

      wv0 = vec4<f32>(0.0);
      wv1 = vec4<f32>(0.0);
      let wrow = n0 + lr;
      if (kb < kb1 && wrow < N) {
        let v = W[(wrow * K + kb * 32u) / 16u + lq / 2u];
        let scale = S[wrow * nb + kb];
        let lo = select(v.xy, v.zw, (lq & 1u) == 1u);
        wv0 = sx(lo.x) * scale;
        wv1 = sx(lo.y) * scale;
      }

      if (lr < {{BM}}u) {
        xs[lr * 16u + step * 8u + lq * 2u] = vec4<{{TILE}}>(xv0);
        xs[lr * 16u + step * 8u + lq * 2u + 1u] = vec4<{{TILE}}>(xv1);
      }
      let r = lr;
      ws[r * 16u + ((step * 8u + lq * 2u) ^ sw)] = vec4<{{TILE}}>(wv0);
      ws[r * 16u + ((step * 8u + lq * 2u + 1u) ^ sw)] = vec4<{{TILE}}>(wv1);
    }
    workgroupBarrier();

    let nx = lid.x & W_SWIZZLE;
    for (var kq = 0u; kq < 16u; kq++) {
      let c = lid.x;
      let b0 = vec4<f32>(ws[c * 16u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 16u) * 16u + (kq ^ nx)]);
      let b2 = vec4<f32>(ws[(c + 32u) * 16u + (kq ^ nx)]);
      let b3 = vec4<f32>(ws[(c + 48u) * 16u + (kq ^ nx)]);

      // Keep the same row and dot-product order as matmul. Each row block
      // is guarded by a compile-time constant so unused registers disappear.
      let a0 = vec4<f32>(xs[(lid.y + 0u) * 16u + kq]);
      acc0 += vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
      if ({{ROWS}}u > 1u) {
        let a1 = vec4<f32>(xs[(lid.y + 16u) * 16u + kq]);
        acc1 += vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
      }
      if ({{ROWS}}u > 2u) {
        let a2 = vec4<f32>(xs[(lid.y + 32u) * 16u + kq]);
        acc2 += vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
      }
      if ({{ROWS}}u > 3u) {
        let a3 = vec4<f32>(xs[(lid.y + 48u) * 16u + kq]);
        acc3 += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
      }
    }
    workgroupBarrier();
  }

  for (var j = 0u; j < 4u; j++) {
    let col = n0 + lid.x + 16u * j;
    if (col >= N) { break; }
    var bias = 0.0;
    if (p.bias == 1u) { bias = B[col]; }
    let values0 = acc0;
    let v0 = values0[j];
    let values1 = acc1;
    let v1 = values1[j];
    let values2 = acc2;
    let v2 = values2[j];
    let values3 = acc3;
    let v3 = values3[j];
    for (var i = 0u; i < {{ROWS}}u; i++) {
      let row = m0 + lid.y + 16u * i;
      if (row >= T) { break; }
      let o = row * N + col;
      var v = v0;
      if (i == 1u) { v = v1; } else if (i == 2u) { v = v2; } else if (i == 3u) { v = v3; }
      if (splits > 1u) {
        PART[wg.z * T * N + o] = v;
        continue;
      }
      v += bias;
      if (p.mode == 1u) { v += Y[o]; } else if (p.mode == 2u) { v = max(v, 0.0); }
      Y[o] = v;
    }
  }
}
