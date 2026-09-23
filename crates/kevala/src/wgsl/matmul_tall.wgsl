// Y[T, N] = X[T, K] . W[N, K]^T (+ bias), W int8 in u32 words with one f32 scale per 32 weights.
//
// Tall Q8 projection kernel. A 16 x 16 workgroup covers 128 rows and 64 columns. Each
// thread owns four columns and eight rows: four rows in each half of the tile. The full
// X tile and one W tile are kept in workgroup memory so both row halves reuse the same W
// values. Products and accumulators stay f32; only the shared tiles use {{TILE}}.
//
// This source is intended for ROWS=8, GROUPS=1, and shader-f16. The host benchmark gates
// those specializations because the shared tile is 24 KiB with f16 and 48 KiB with f32.
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

// 128 rows x 16 vec4 slots and 64 columns x 16 vec4 slots. With f16 this is 24 KiB.
var<workgroup> xs: array<vec4<{{TILE}}>, 2048>;
var<workgroup> ws: array<vec4<{{TILE}}>, 1024>;

// The full 64-row W tile uses all 16 slots. This keeps the two K blocks distributed across
// the shared-memory banks on adapters where vec4<f16> spans two 32-bit banks.
const W_SWIZZLE: u32 = 15u;

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

  // Named vectors avoid dynamic private-array indexing in the accumulator hot loop.
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);
  var acc4 = vec4<f32>(0.0);
  var acc5 = vec4<f32>(0.0);
  var acc6 = vec4<f32>(0.0);
  var acc7 = vec4<f32>(0.0);

  // Four lanes load each row. The 64 loader rows populate both halves of the 128-row X tile;
  // each loader also dequantizes one row of the single 64-column W tile.
  let lr = li / 4u;
  let lq = li % 4u;
  let xrow0 = m0 + lr;
  let xrow1 = m0 + 64u + lr;
  let sw = lr & W_SWIZZLE;
  var xv0 = vec4<f32>(0.0);
  var xv1 = vec4<f32>(0.0);
  var wv0 = vec4<f32>(0.0);
  var wv1 = vec4<f32>(0.0);
  for (var block = kb0; block < kb1; block += 2u) {
    // Stage two quantization blocks (64 values) before the barrier, matching matmul_wide.
    for (var step = 0u; step < 2u; step++) {
      let kb = block + step;
      xv0 = vec4<f32>(0.0);
      xv1 = vec4<f32>(0.0);
      if (kb < kb1 && xrow0 < T) {
        let base = (xrow0 * K + kb * 32u) / 4u + lq * 2u;
        xv0 = X[base];
        xv1 = X[base + 1u];
      }
      if (kb < kb1) {
        let base = (xrow1 * K + kb * 32u) / 4u + lq * 2u;
        if (xrow1 < T) {
          xs[(lr + 64u) * 16u + step * 8u + lq * 2u] = vec4<{{TILE}}>(X[base]);
          xs[(lr + 64u) * 16u + step * 8u + lq * 2u + 1u] = vec4<{{TILE}}>(X[base + 1u]);
        }
      }

      if (kb < kb1 && xrow0 < T) {
        xs[lr * 16u + step * 8u + lq * 2u] = vec4<{{TILE}}>(xv0);
        xs[lr * 16u + step * 8u + lq * 2u + 1u] = vec4<{{TILE}}>(xv1);
      } else {
        xs[lr * 16u + step * 8u + lq * 2u] = vec4<{{TILE}}>(0.0);
        xs[lr * 16u + step * 8u + lq * 2u + 1u] = vec4<{{TILE}}>(0.0);
      }
      if (!(kb < kb1 && xrow1 < T)) {
        xs[(lr + 64u) * 16u + step * 8u + lq * 2u] = vec4<{{TILE}}>(0.0);
        xs[(lr + 64u) * 16u + step * 8u + lq * 2u + 1u] = vec4<{{TILE}}>(0.0);
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
      ws[lr * 16u + ((step * 8u + lq * 2u) ^ sw)] = vec4<{{TILE}}>(wv0);
      ws[lr * 16u + ((step * 8u + lq * 2u + 1u) ^ sw)] = vec4<{{TILE}}>(wv1);
    }
    workgroupBarrier();

    let nx = lid.x & W_SWIZZLE;
    // Keep the exact sixteen-step order while exposing each step to the compiler.
    {
      let kq = 0u;
      let c = lid.x;
      let b0 = vec4<f32>(ws[c * 16u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 16u) * 16u + (kq ^ nx)]);
      let b2 = vec4<f32>(ws[(c + 32u) * 16u + (kq ^ nx)]);
      let b3 = vec4<f32>(ws[(c + 48u) * 16u + (kq ^ nx)]);

      let a0 = vec4<f32>(xs[(lid.y + 0u) * 16u + kq]);
      let a1 = vec4<f32>(xs[(lid.y + 16u) * 16u + kq]);
      let a2 = vec4<f32>(xs[(lid.y + 32u) * 16u + kq]);
      let a3 = vec4<f32>(xs[(lid.y + 48u) * 16u + kq]);
      let a4 = vec4<f32>(xs[(lid.y + 64u) * 16u + kq]);
      let a5 = vec4<f32>(xs[(lid.y + 80u) * 16u + kq]);
      let a6 = vec4<f32>(xs[(lid.y + 96u) * 16u + kq]);
      let a7 = vec4<f32>(xs[(lid.y + 112u) * 16u + kq]);

      acc0 += vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
      acc1 += vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
      acc2 += vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
      acc3 += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
      acc4 += vec4<f32>(dot(a4, b0), dot(a4, b1), dot(a4, b2), dot(a4, b3));
      acc5 += vec4<f32>(dot(a5, b0), dot(a5, b1), dot(a5, b2), dot(a5, b3));
      acc6 += vec4<f32>(dot(a6, b0), dot(a6, b1), dot(a6, b2), dot(a6, b3));
      acc7 += vec4<f32>(dot(a7, b0), dot(a7, b1), dot(a7, b2), dot(a7, b3));
    }
    {
      let kq = 1u;
      let c = lid.x;
      let b0 = vec4<f32>(ws[c * 16u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 16u) * 16u + (kq ^ nx)]);
      let b2 = vec4<f32>(ws[(c + 32u) * 16u + (kq ^ nx)]);
      let b3 = vec4<f32>(ws[(c + 48u) * 16u + (kq ^ nx)]);

      let a0 = vec4<f32>(xs[(lid.y + 0u) * 16u + kq]);
      let a1 = vec4<f32>(xs[(lid.y + 16u) * 16u + kq]);
      let a2 = vec4<f32>(xs[(lid.y + 32u) * 16u + kq]);
      let a3 = vec4<f32>(xs[(lid.y + 48u) * 16u + kq]);
      let a4 = vec4<f32>(xs[(lid.y + 64u) * 16u + kq]);
      let a5 = vec4<f32>(xs[(lid.y + 80u) * 16u + kq]);
      let a6 = vec4<f32>(xs[(lid.y + 96u) * 16u + kq]);
      let a7 = vec4<f32>(xs[(lid.y + 112u) * 16u + kq]);

      acc0 += vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
      acc1 += vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
      acc2 += vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
      acc3 += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
      acc4 += vec4<f32>(dot(a4, b0), dot(a4, b1), dot(a4, b2), dot(a4, b3));
      acc5 += vec4<f32>(dot(a5, b0), dot(a5, b1), dot(a5, b2), dot(a5, b3));
      acc6 += vec4<f32>(dot(a6, b0), dot(a6, b1), dot(a6, b2), dot(a6, b3));
      acc7 += vec4<f32>(dot(a7, b0), dot(a7, b1), dot(a7, b2), dot(a7, b3));
    }
    {
      let kq = 2u;
      let c = lid.x;
      let b0 = vec4<f32>(ws[c * 16u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 16u) * 16u + (kq ^ nx)]);
      let b2 = vec4<f32>(ws[(c + 32u) * 16u + (kq ^ nx)]);
      let b3 = vec4<f32>(ws[(c + 48u) * 16u + (kq ^ nx)]);

      let a0 = vec4<f32>(xs[(lid.y + 0u) * 16u + kq]);
      let a1 = vec4<f32>(xs[(lid.y + 16u) * 16u + kq]);
      let a2 = vec4<f32>(xs[(lid.y + 32u) * 16u + kq]);
      let a3 = vec4<f32>(xs[(lid.y + 48u) * 16u + kq]);
      let a4 = vec4<f32>(xs[(lid.y + 64u) * 16u + kq]);
      let a5 = vec4<f32>(xs[(lid.y + 80u) * 16u + kq]);
      let a6 = vec4<f32>(xs[(lid.y + 96u) * 16u + kq]);
      let a7 = vec4<f32>(xs[(lid.y + 112u) * 16u + kq]);

      acc0 += vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
      acc1 += vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
      acc2 += vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
      acc3 += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
      acc4 += vec4<f32>(dot(a4, b0), dot(a4, b1), dot(a4, b2), dot(a4, b3));
      acc5 += vec4<f32>(dot(a5, b0), dot(a5, b1), dot(a5, b2), dot(a5, b3));
      acc6 += vec4<f32>(dot(a6, b0), dot(a6, b1), dot(a6, b2), dot(a6, b3));
      acc7 += vec4<f32>(dot(a7, b0), dot(a7, b1), dot(a7, b2), dot(a7, b3));
    }
    {
      let kq = 3u;
      let c = lid.x;
      let b0 = vec4<f32>(ws[c * 16u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 16u) * 16u + (kq ^ nx)]);
      let b2 = vec4<f32>(ws[(c + 32u) * 16u + (kq ^ nx)]);
      let b3 = vec4<f32>(ws[(c + 48u) * 16u + (kq ^ nx)]);

      let a0 = vec4<f32>(xs[(lid.y + 0u) * 16u + kq]);
      let a1 = vec4<f32>(xs[(lid.y + 16u) * 16u + kq]);
      let a2 = vec4<f32>(xs[(lid.y + 32u) * 16u + kq]);
      let a3 = vec4<f32>(xs[(lid.y + 48u) * 16u + kq]);
      let a4 = vec4<f32>(xs[(lid.y + 64u) * 16u + kq]);
      let a5 = vec4<f32>(xs[(lid.y + 80u) * 16u + kq]);
      let a6 = vec4<f32>(xs[(lid.y + 96u) * 16u + kq]);
      let a7 = vec4<f32>(xs[(lid.y + 112u) * 16u + kq]);

      acc0 += vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
      acc1 += vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
      acc2 += vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
      acc3 += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
      acc4 += vec4<f32>(dot(a4, b0), dot(a4, b1), dot(a4, b2), dot(a4, b3));
      acc5 += vec4<f32>(dot(a5, b0), dot(a5, b1), dot(a5, b2), dot(a5, b3));
      acc6 += vec4<f32>(dot(a6, b0), dot(a6, b1), dot(a6, b2), dot(a6, b3));
      acc7 += vec4<f32>(dot(a7, b0), dot(a7, b1), dot(a7, b2), dot(a7, b3));
    }
    {
      let kq = 4u;
      let c = lid.x;
      let b0 = vec4<f32>(ws[c * 16u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 16u) * 16u + (kq ^ nx)]);
      let b2 = vec4<f32>(ws[(c + 32u) * 16u + (kq ^ nx)]);
      let b3 = vec4<f32>(ws[(c + 48u) * 16u + (kq ^ nx)]);

      let a0 = vec4<f32>(xs[(lid.y + 0u) * 16u + kq]);
      let a1 = vec4<f32>(xs[(lid.y + 16u) * 16u + kq]);
      let a2 = vec4<f32>(xs[(lid.y + 32u) * 16u + kq]);
      let a3 = vec4<f32>(xs[(lid.y + 48u) * 16u + kq]);
      let a4 = vec4<f32>(xs[(lid.y + 64u) * 16u + kq]);
      let a5 = vec4<f32>(xs[(lid.y + 80u) * 16u + kq]);
      let a6 = vec4<f32>(xs[(lid.y + 96u) * 16u + kq]);
      let a7 = vec4<f32>(xs[(lid.y + 112u) * 16u + kq]);

      acc0 += vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
      acc1 += vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
      acc2 += vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
      acc3 += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
      acc4 += vec4<f32>(dot(a4, b0), dot(a4, b1), dot(a4, b2), dot(a4, b3));
      acc5 += vec4<f32>(dot(a5, b0), dot(a5, b1), dot(a5, b2), dot(a5, b3));
      acc6 += vec4<f32>(dot(a6, b0), dot(a6, b1), dot(a6, b2), dot(a6, b3));
      acc7 += vec4<f32>(dot(a7, b0), dot(a7, b1), dot(a7, b2), dot(a7, b3));
    }
    {
      let kq = 5u;
      let c = lid.x;
      let b0 = vec4<f32>(ws[c * 16u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 16u) * 16u + (kq ^ nx)]);
      let b2 = vec4<f32>(ws[(c + 32u) * 16u + (kq ^ nx)]);
      let b3 = vec4<f32>(ws[(c + 48u) * 16u + (kq ^ nx)]);

      let a0 = vec4<f32>(xs[(lid.y + 0u) * 16u + kq]);
      let a1 = vec4<f32>(xs[(lid.y + 16u) * 16u + kq]);
      let a2 = vec4<f32>(xs[(lid.y + 32u) * 16u + kq]);
      let a3 = vec4<f32>(xs[(lid.y + 48u) * 16u + kq]);
      let a4 = vec4<f32>(xs[(lid.y + 64u) * 16u + kq]);
      let a5 = vec4<f32>(xs[(lid.y + 80u) * 16u + kq]);
      let a6 = vec4<f32>(xs[(lid.y + 96u) * 16u + kq]);
      let a7 = vec4<f32>(xs[(lid.y + 112u) * 16u + kq]);

      acc0 += vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
      acc1 += vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
      acc2 += vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
      acc3 += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
      acc4 += vec4<f32>(dot(a4, b0), dot(a4, b1), dot(a4, b2), dot(a4, b3));
      acc5 += vec4<f32>(dot(a5, b0), dot(a5, b1), dot(a5, b2), dot(a5, b3));
      acc6 += vec4<f32>(dot(a6, b0), dot(a6, b1), dot(a6, b2), dot(a6, b3));
      acc7 += vec4<f32>(dot(a7, b0), dot(a7, b1), dot(a7, b2), dot(a7, b3));
    }
    {
      let kq = 6u;
      let c = lid.x;
      let b0 = vec4<f32>(ws[c * 16u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 16u) * 16u + (kq ^ nx)]);
      let b2 = vec4<f32>(ws[(c + 32u) * 16u + (kq ^ nx)]);
      let b3 = vec4<f32>(ws[(c + 48u) * 16u + (kq ^ nx)]);

      let a0 = vec4<f32>(xs[(lid.y + 0u) * 16u + kq]);
      let a1 = vec4<f32>(xs[(lid.y + 16u) * 16u + kq]);
      let a2 = vec4<f32>(xs[(lid.y + 32u) * 16u + kq]);
      let a3 = vec4<f32>(xs[(lid.y + 48u) * 16u + kq]);
      let a4 = vec4<f32>(xs[(lid.y + 64u) * 16u + kq]);
      let a5 = vec4<f32>(xs[(lid.y + 80u) * 16u + kq]);
      let a6 = vec4<f32>(xs[(lid.y + 96u) * 16u + kq]);
      let a7 = vec4<f32>(xs[(lid.y + 112u) * 16u + kq]);

      acc0 += vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
      acc1 += vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
      acc2 += vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
      acc3 += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
      acc4 += vec4<f32>(dot(a4, b0), dot(a4, b1), dot(a4, b2), dot(a4, b3));
      acc5 += vec4<f32>(dot(a5, b0), dot(a5, b1), dot(a5, b2), dot(a5, b3));
      acc6 += vec4<f32>(dot(a6, b0), dot(a6, b1), dot(a6, b2), dot(a6, b3));
      acc7 += vec4<f32>(dot(a7, b0), dot(a7, b1), dot(a7, b2), dot(a7, b3));
    }
    {
      let kq = 7u;
      let c = lid.x;
      let b0 = vec4<f32>(ws[c * 16u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 16u) * 16u + (kq ^ nx)]);
      let b2 = vec4<f32>(ws[(c + 32u) * 16u + (kq ^ nx)]);
      let b3 = vec4<f32>(ws[(c + 48u) * 16u + (kq ^ nx)]);

      let a0 = vec4<f32>(xs[(lid.y + 0u) * 16u + kq]);
      let a1 = vec4<f32>(xs[(lid.y + 16u) * 16u + kq]);
      let a2 = vec4<f32>(xs[(lid.y + 32u) * 16u + kq]);
      let a3 = vec4<f32>(xs[(lid.y + 48u) * 16u + kq]);
      let a4 = vec4<f32>(xs[(lid.y + 64u) * 16u + kq]);
      let a5 = vec4<f32>(xs[(lid.y + 80u) * 16u + kq]);
      let a6 = vec4<f32>(xs[(lid.y + 96u) * 16u + kq]);
      let a7 = vec4<f32>(xs[(lid.y + 112u) * 16u + kq]);

      acc0 += vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
      acc1 += vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
      acc2 += vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
      acc3 += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
      acc4 += vec4<f32>(dot(a4, b0), dot(a4, b1), dot(a4, b2), dot(a4, b3));
      acc5 += vec4<f32>(dot(a5, b0), dot(a5, b1), dot(a5, b2), dot(a5, b3));
      acc6 += vec4<f32>(dot(a6, b0), dot(a6, b1), dot(a6, b2), dot(a6, b3));
      acc7 += vec4<f32>(dot(a7, b0), dot(a7, b1), dot(a7, b2), dot(a7, b3));
    }
    {
      let kq = 8u;
      let c = lid.x;
      let b0 = vec4<f32>(ws[c * 16u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 16u) * 16u + (kq ^ nx)]);
      let b2 = vec4<f32>(ws[(c + 32u) * 16u + (kq ^ nx)]);
      let b3 = vec4<f32>(ws[(c + 48u) * 16u + (kq ^ nx)]);

      let a0 = vec4<f32>(xs[(lid.y + 0u) * 16u + kq]);
      let a1 = vec4<f32>(xs[(lid.y + 16u) * 16u + kq]);
      let a2 = vec4<f32>(xs[(lid.y + 32u) * 16u + kq]);
      let a3 = vec4<f32>(xs[(lid.y + 48u) * 16u + kq]);
      let a4 = vec4<f32>(xs[(lid.y + 64u) * 16u + kq]);
      let a5 = vec4<f32>(xs[(lid.y + 80u) * 16u + kq]);
      let a6 = vec4<f32>(xs[(lid.y + 96u) * 16u + kq]);
      let a7 = vec4<f32>(xs[(lid.y + 112u) * 16u + kq]);

      acc0 += vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
      acc1 += vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
      acc2 += vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
      acc3 += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
      acc4 += vec4<f32>(dot(a4, b0), dot(a4, b1), dot(a4, b2), dot(a4, b3));
      acc5 += vec4<f32>(dot(a5, b0), dot(a5, b1), dot(a5, b2), dot(a5, b3));
      acc6 += vec4<f32>(dot(a6, b0), dot(a6, b1), dot(a6, b2), dot(a6, b3));
      acc7 += vec4<f32>(dot(a7, b0), dot(a7, b1), dot(a7, b2), dot(a7, b3));
    }
    {
      let kq = 9u;
      let c = lid.x;
      let b0 = vec4<f32>(ws[c * 16u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 16u) * 16u + (kq ^ nx)]);
      let b2 = vec4<f32>(ws[(c + 32u) * 16u + (kq ^ nx)]);
      let b3 = vec4<f32>(ws[(c + 48u) * 16u + (kq ^ nx)]);

      let a0 = vec4<f32>(xs[(lid.y + 0u) * 16u + kq]);
      let a1 = vec4<f32>(xs[(lid.y + 16u) * 16u + kq]);
      let a2 = vec4<f32>(xs[(lid.y + 32u) * 16u + kq]);
      let a3 = vec4<f32>(xs[(lid.y + 48u) * 16u + kq]);
      let a4 = vec4<f32>(xs[(lid.y + 64u) * 16u + kq]);
      let a5 = vec4<f32>(xs[(lid.y + 80u) * 16u + kq]);
      let a6 = vec4<f32>(xs[(lid.y + 96u) * 16u + kq]);
      let a7 = vec4<f32>(xs[(lid.y + 112u) * 16u + kq]);

      acc0 += vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
      acc1 += vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
      acc2 += vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
      acc3 += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
      acc4 += vec4<f32>(dot(a4, b0), dot(a4, b1), dot(a4, b2), dot(a4, b3));
      acc5 += vec4<f32>(dot(a5, b0), dot(a5, b1), dot(a5, b2), dot(a5, b3));
      acc6 += vec4<f32>(dot(a6, b0), dot(a6, b1), dot(a6, b2), dot(a6, b3));
      acc7 += vec4<f32>(dot(a7, b0), dot(a7, b1), dot(a7, b2), dot(a7, b3));
    }
    {
      let kq = 10u;
      let c = lid.x;
      let b0 = vec4<f32>(ws[c * 16u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 16u) * 16u + (kq ^ nx)]);
      let b2 = vec4<f32>(ws[(c + 32u) * 16u + (kq ^ nx)]);
      let b3 = vec4<f32>(ws[(c + 48u) * 16u + (kq ^ nx)]);

      let a0 = vec4<f32>(xs[(lid.y + 0u) * 16u + kq]);
      let a1 = vec4<f32>(xs[(lid.y + 16u) * 16u + kq]);
      let a2 = vec4<f32>(xs[(lid.y + 32u) * 16u + kq]);
      let a3 = vec4<f32>(xs[(lid.y + 48u) * 16u + kq]);
      let a4 = vec4<f32>(xs[(lid.y + 64u) * 16u + kq]);
      let a5 = vec4<f32>(xs[(lid.y + 80u) * 16u + kq]);
      let a6 = vec4<f32>(xs[(lid.y + 96u) * 16u + kq]);
      let a7 = vec4<f32>(xs[(lid.y + 112u) * 16u + kq]);

      acc0 += vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
      acc1 += vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
      acc2 += vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
      acc3 += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
      acc4 += vec4<f32>(dot(a4, b0), dot(a4, b1), dot(a4, b2), dot(a4, b3));
      acc5 += vec4<f32>(dot(a5, b0), dot(a5, b1), dot(a5, b2), dot(a5, b3));
      acc6 += vec4<f32>(dot(a6, b0), dot(a6, b1), dot(a6, b2), dot(a6, b3));
      acc7 += vec4<f32>(dot(a7, b0), dot(a7, b1), dot(a7, b2), dot(a7, b3));
    }
    {
      let kq = 11u;
      let c = lid.x;
      let b0 = vec4<f32>(ws[c * 16u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 16u) * 16u + (kq ^ nx)]);
      let b2 = vec4<f32>(ws[(c + 32u) * 16u + (kq ^ nx)]);
      let b3 = vec4<f32>(ws[(c + 48u) * 16u + (kq ^ nx)]);

      let a0 = vec4<f32>(xs[(lid.y + 0u) * 16u + kq]);
      let a1 = vec4<f32>(xs[(lid.y + 16u) * 16u + kq]);
      let a2 = vec4<f32>(xs[(lid.y + 32u) * 16u + kq]);
      let a3 = vec4<f32>(xs[(lid.y + 48u) * 16u + kq]);
      let a4 = vec4<f32>(xs[(lid.y + 64u) * 16u + kq]);
      let a5 = vec4<f32>(xs[(lid.y + 80u) * 16u + kq]);
      let a6 = vec4<f32>(xs[(lid.y + 96u) * 16u + kq]);
      let a7 = vec4<f32>(xs[(lid.y + 112u) * 16u + kq]);

      acc0 += vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
      acc1 += vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
      acc2 += vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
      acc3 += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
      acc4 += vec4<f32>(dot(a4, b0), dot(a4, b1), dot(a4, b2), dot(a4, b3));
      acc5 += vec4<f32>(dot(a5, b0), dot(a5, b1), dot(a5, b2), dot(a5, b3));
      acc6 += vec4<f32>(dot(a6, b0), dot(a6, b1), dot(a6, b2), dot(a6, b3));
      acc7 += vec4<f32>(dot(a7, b0), dot(a7, b1), dot(a7, b2), dot(a7, b3));
    }
    {
      let kq = 12u;
      let c = lid.x;
      let b0 = vec4<f32>(ws[c * 16u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 16u) * 16u + (kq ^ nx)]);
      let b2 = vec4<f32>(ws[(c + 32u) * 16u + (kq ^ nx)]);
      let b3 = vec4<f32>(ws[(c + 48u) * 16u + (kq ^ nx)]);

      let a0 = vec4<f32>(xs[(lid.y + 0u) * 16u + kq]);
      let a1 = vec4<f32>(xs[(lid.y + 16u) * 16u + kq]);
      let a2 = vec4<f32>(xs[(lid.y + 32u) * 16u + kq]);
      let a3 = vec4<f32>(xs[(lid.y + 48u) * 16u + kq]);
      let a4 = vec4<f32>(xs[(lid.y + 64u) * 16u + kq]);
      let a5 = vec4<f32>(xs[(lid.y + 80u) * 16u + kq]);
      let a6 = vec4<f32>(xs[(lid.y + 96u) * 16u + kq]);
      let a7 = vec4<f32>(xs[(lid.y + 112u) * 16u + kq]);

      acc0 += vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
      acc1 += vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
      acc2 += vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
      acc3 += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
      acc4 += vec4<f32>(dot(a4, b0), dot(a4, b1), dot(a4, b2), dot(a4, b3));
      acc5 += vec4<f32>(dot(a5, b0), dot(a5, b1), dot(a5, b2), dot(a5, b3));
      acc6 += vec4<f32>(dot(a6, b0), dot(a6, b1), dot(a6, b2), dot(a6, b3));
      acc7 += vec4<f32>(dot(a7, b0), dot(a7, b1), dot(a7, b2), dot(a7, b3));
    }
    {
      let kq = 13u;
      let c = lid.x;
      let b0 = vec4<f32>(ws[c * 16u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 16u) * 16u + (kq ^ nx)]);
      let b2 = vec4<f32>(ws[(c + 32u) * 16u + (kq ^ nx)]);
      let b3 = vec4<f32>(ws[(c + 48u) * 16u + (kq ^ nx)]);

      let a0 = vec4<f32>(xs[(lid.y + 0u) * 16u + kq]);
      let a1 = vec4<f32>(xs[(lid.y + 16u) * 16u + kq]);
      let a2 = vec4<f32>(xs[(lid.y + 32u) * 16u + kq]);
      let a3 = vec4<f32>(xs[(lid.y + 48u) * 16u + kq]);
      let a4 = vec4<f32>(xs[(lid.y + 64u) * 16u + kq]);
      let a5 = vec4<f32>(xs[(lid.y + 80u) * 16u + kq]);
      let a6 = vec4<f32>(xs[(lid.y + 96u) * 16u + kq]);
      let a7 = vec4<f32>(xs[(lid.y + 112u) * 16u + kq]);

      acc0 += vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
      acc1 += vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
      acc2 += vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
      acc3 += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
      acc4 += vec4<f32>(dot(a4, b0), dot(a4, b1), dot(a4, b2), dot(a4, b3));
      acc5 += vec4<f32>(dot(a5, b0), dot(a5, b1), dot(a5, b2), dot(a5, b3));
      acc6 += vec4<f32>(dot(a6, b0), dot(a6, b1), dot(a6, b2), dot(a6, b3));
      acc7 += vec4<f32>(dot(a7, b0), dot(a7, b1), dot(a7, b2), dot(a7, b3));
    }
    {
      let kq = 14u;
      let c = lid.x;
      let b0 = vec4<f32>(ws[c * 16u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 16u) * 16u + (kq ^ nx)]);
      let b2 = vec4<f32>(ws[(c + 32u) * 16u + (kq ^ nx)]);
      let b3 = vec4<f32>(ws[(c + 48u) * 16u + (kq ^ nx)]);

      let a0 = vec4<f32>(xs[(lid.y + 0u) * 16u + kq]);
      let a1 = vec4<f32>(xs[(lid.y + 16u) * 16u + kq]);
      let a2 = vec4<f32>(xs[(lid.y + 32u) * 16u + kq]);
      let a3 = vec4<f32>(xs[(lid.y + 48u) * 16u + kq]);
      let a4 = vec4<f32>(xs[(lid.y + 64u) * 16u + kq]);
      let a5 = vec4<f32>(xs[(lid.y + 80u) * 16u + kq]);
      let a6 = vec4<f32>(xs[(lid.y + 96u) * 16u + kq]);
      let a7 = vec4<f32>(xs[(lid.y + 112u) * 16u + kq]);

      acc0 += vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
      acc1 += vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
      acc2 += vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
      acc3 += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
      acc4 += vec4<f32>(dot(a4, b0), dot(a4, b1), dot(a4, b2), dot(a4, b3));
      acc5 += vec4<f32>(dot(a5, b0), dot(a5, b1), dot(a5, b2), dot(a5, b3));
      acc6 += vec4<f32>(dot(a6, b0), dot(a6, b1), dot(a6, b2), dot(a6, b3));
      acc7 += vec4<f32>(dot(a7, b0), dot(a7, b1), dot(a7, b2), dot(a7, b3));
    }
    {
      let kq = 15u;
      let c = lid.x;
      let b0 = vec4<f32>(ws[c * 16u + (kq ^ nx)]);
      let b1 = vec4<f32>(ws[(c + 16u) * 16u + (kq ^ nx)]);
      let b2 = vec4<f32>(ws[(c + 32u) * 16u + (kq ^ nx)]);
      let b3 = vec4<f32>(ws[(c + 48u) * 16u + (kq ^ nx)]);

      let a0 = vec4<f32>(xs[(lid.y + 0u) * 16u + kq]);
      let a1 = vec4<f32>(xs[(lid.y + 16u) * 16u + kq]);
      let a2 = vec4<f32>(xs[(lid.y + 32u) * 16u + kq]);
      let a3 = vec4<f32>(xs[(lid.y + 48u) * 16u + kq]);
      let a4 = vec4<f32>(xs[(lid.y + 64u) * 16u + kq]);
      let a5 = vec4<f32>(xs[(lid.y + 80u) * 16u + kq]);
      let a6 = vec4<f32>(xs[(lid.y + 96u) * 16u + kq]);
      let a7 = vec4<f32>(xs[(lid.y + 112u) * 16u + kq]);

      acc0 += vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
      acc1 += vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
      acc2 += vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
      acc3 += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
      acc4 += vec4<f32>(dot(a4, b0), dot(a4, b1), dot(a4, b2), dot(a4, b3));
      acc5 += vec4<f32>(dot(a5, b0), dot(a5, b1), dot(a5, b2), dot(a5, b3));
      acc6 += vec4<f32>(dot(a6, b0), dot(a6, b1), dot(a6, b2), dot(a6, b3));
      acc7 += vec4<f32>(dot(a7, b0), dot(a7, b1), dot(a7, b2), dot(a7, b3));
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
    let values4 = acc4;
    let v4 = values4[j];
    let values5 = acc5;
    let v5 = values5[j];
    let values6 = acc6;
    let v6 = values6[j];
    let values7 = acc7;
    let v7 = values7[j];
    for (var i = 0u; i < {{ROWS}}u; i++) {
      let row = m0 + lid.y + 16u * i;
      if (row >= T) { break; }
      let o = row * N + col;
      var v = v0;
      if (i == 1u) { v = v1; } else if (i == 2u) { v = v2; } else if (i == 3u) { v = v3; }
      else if (i == 4u) { v = v4; } else if (i == 5u) { v = v5; } else if (i == 6u) { v = v6; } else if (i == 7u) { v = v7; }
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
