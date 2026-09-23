// Fused Q8 gate/up projections followed by Gemma's GELU-tanh gate.
//
// A workgroup covers 64 rows and 64 columns with the same 16 x 16 mapping as
// the generic R4/G1 matmul. X is staged once per K block. The two Q8 weight
// tiles are staged independently, but their dot products use the same K/4
// order and FP32 accumulators as matmul.wgsl. This prototype intentionally
// has no split-K path; the harness rejects cases for which the runtime split
// rule would choose more than one split.
//#include common

struct P { N: u32, K: u32, _a: u32, _b: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> WG: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> SG: array<f32>;
@group(0) @binding(5) var<storage, read> WU: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> SU: array<f32>;
@group(0) @binding(7) var<storage, read_write> Y: array<f32>;

var<workgroup> xs: array<vec4<f32>, 512>; // [row][k/4]
var<workgroup> wg: array<vec4<f32>, 512>; // [column][(k/4) ^ (column & 7)]
var<workgroup> wu: array<vec4<f32>, 512>;

// Four signed int8 weights as exact f32 values without integer-to-float
// conversion. This is the same unpacking used by matmul.wgsl.
fn sx(w: u32) -> vec4<f32> {
  let u = w ^ 0x80808080u;
  let bytes = vec4<u32>(u, u >> 8u, u >> 16u, u >> 24u) & vec4<u32>(0xFFu);
  return bitcast<vec4<f32>>(bytes | vec4<u32>(0x4B000000u)) - vec4<f32>(8388736.0);
}

fn gelu(x: f32) -> f32 {
  let c = 0.7978845608028654;
  return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(local_invocation_index) li: u32,
) {
  let T = g.T;
  let N = p.N;
  let K = p.K;
  let nb = K / 32u;
  let m0 = wid.y * 64u;
  let n0 = wid.x * 64u;

  var gate0 = vec4<f32>(0.0);
  var gate1 = vec4<f32>(0.0);
  var gate2 = vec4<f32>(0.0);
  var gate3 = vec4<f32>(0.0);
  var up0 = vec4<f32>(0.0);
  var up1 = vec4<f32>(0.0);
  var up2 = vec4<f32>(0.0);
  var up3 = vec4<f32>(0.0);

  // Four loaders per row, each bringing two vec4 values from a K=32 block.
  let lr = li / 4u;
  let lq = li % 4u;
  let xrow = m0 + lr;
  let sw = lr & 7u;
  for (var kb = 0u; kb < nb; kb++) {
    var xv0 = vec4<f32>(0.0);
    var xv1 = vec4<f32>(0.0);
    if (lr < 64u && xrow < T) {
      let base = (xrow * K + kb * 32u) / 4u + lq * 2u;
      xv0 = X[base];
      xv1 = X[base + 1u];
    }

    var g0 = vec4<f32>(0.0);
    var g1 = vec4<f32>(0.0);
    var u0 = vec4<f32>(0.0);
    var u1 = vec4<f32>(0.0);
    let wrow = n0 + lr;
    if (wrow < N) {
      let gi = (wrow * K + kb * 32u) / 16u + lq / 2u;
      let gv = WG[gi];
      let uv = WU[gi];
      let glo = select(gv.xy, gv.zw, (lq & 1u) == 1u);
      let ulo = select(uv.xy, uv.zw, (lq & 1u) == 1u);
      let gs = SG[wrow * nb + kb];
      let us = SU[wrow * nb + kb];
      g0 = sx(glo.x) * gs;
      g1 = sx(glo.y) * gs;
      u0 = sx(ulo.x) * us;
      u1 = sx(ulo.y) * us;
    }

    if (lr < 64u) {
      xs[lr * 8u + lq * 2u] = xv0;
      xs[lr * 8u + lq * 2u + 1u] = xv1;
    }
    let wr = lr;
    wg[wr * 8u + ((lq * 2u) ^ sw)] = g0;
    wg[wr * 8u + ((lq * 2u + 1u) ^ sw)] = g1;
    wu[wr * 8u + ((lq * 2u) ^ sw)] = u0;
    wu[wr * 8u + ((lq * 2u + 1u) ^ sw)] = u1;
    workgroupBarrier();

    let c = lid.x;
    let nx = lid.x & 7u;
    {
      let kq = 0u;
      let bg0 = wg[c * 8u + (kq ^ nx)];
      let bg1 = wg[(c + 16u) * 8u + (kq ^ nx)];
      let bg2 = wg[(c + 32u) * 8u + (kq ^ nx)];
      let bg3 = wg[(c + 48u) * 8u + (kq ^ nx)];
      let bu0 = wu[c * 8u + (kq ^ nx)];
      let bu1 = wu[(c + 16u) * 8u + (kq ^ nx)];
      let bu2 = wu[(c + 32u) * 8u + (kq ^ nx)];
      let bu3 = wu[(c + 48u) * 8u + (kq ^ nx)];
      let a0 = xs[(lid.y + 0u) * 8u + kq];
      let a1 = xs[(lid.y + 16u) * 8u + kq];
      let a2 = xs[(lid.y + 32u) * 8u + kq];
      let a3 = xs[(lid.y + 48u) * 8u + kq];
      gate0 += vec4<f32>(dot(a0, bg0), dot(a0, bg1), dot(a0, bg2), dot(a0, bg3));
      gate1 += vec4<f32>(dot(a1, bg0), dot(a1, bg1), dot(a1, bg2), dot(a1, bg3));
      gate2 += vec4<f32>(dot(a2, bg0), dot(a2, bg1), dot(a2, bg2), dot(a2, bg3));
      gate3 += vec4<f32>(dot(a3, bg0), dot(a3, bg1), dot(a3, bg2), dot(a3, bg3));
      up0 += vec4<f32>(dot(a0, bu0), dot(a0, bu1), dot(a0, bu2), dot(a0, bu3));
      up1 += vec4<f32>(dot(a1, bu0), dot(a1, bu1), dot(a1, bu2), dot(a1, bu3));
      up2 += vec4<f32>(dot(a2, bu0), dot(a2, bu1), dot(a2, bu2), dot(a2, bu3));
      up3 += vec4<f32>(dot(a3, bu0), dot(a3, bu1), dot(a3, bu2), dot(a3, bu3));
    }
    {
      let kq = 1u;
      let bg0 = wg[c * 8u + (kq ^ nx)];
      let bg1 = wg[(c + 16u) * 8u + (kq ^ nx)];
      let bg2 = wg[(c + 32u) * 8u + (kq ^ nx)];
      let bg3 = wg[(c + 48u) * 8u + (kq ^ nx)];
      let bu0 = wu[c * 8u + (kq ^ nx)];
      let bu1 = wu[(c + 16u) * 8u + (kq ^ nx)];
      let bu2 = wu[(c + 32u) * 8u + (kq ^ nx)];
      let bu3 = wu[(c + 48u) * 8u + (kq ^ nx)];
      let a0 = xs[(lid.y + 0u) * 8u + kq];
      let a1 = xs[(lid.y + 16u) * 8u + kq];
      let a2 = xs[(lid.y + 32u) * 8u + kq];
      let a3 = xs[(lid.y + 48u) * 8u + kq];
      gate0 += vec4<f32>(dot(a0, bg0), dot(a0, bg1), dot(a0, bg2), dot(a0, bg3));
      gate1 += vec4<f32>(dot(a1, bg0), dot(a1, bg1), dot(a1, bg2), dot(a1, bg3));
      gate2 += vec4<f32>(dot(a2, bg0), dot(a2, bg1), dot(a2, bg2), dot(a2, bg3));
      gate3 += vec4<f32>(dot(a3, bg0), dot(a3, bg1), dot(a3, bg2), dot(a3, bg3));
      up0 += vec4<f32>(dot(a0, bu0), dot(a0, bu1), dot(a0, bu2), dot(a0, bu3));
      up1 += vec4<f32>(dot(a1, bu0), dot(a1, bu1), dot(a1, bu2), dot(a1, bu3));
      up2 += vec4<f32>(dot(a2, bu0), dot(a2, bu1), dot(a2, bu2), dot(a2, bu3));
      up3 += vec4<f32>(dot(a3, bu0), dot(a3, bu1), dot(a3, bu2), dot(a3, bu3));
    }
    {
      let kq = 2u;
      let bg0 = wg[c * 8u + (kq ^ nx)];
      let bg1 = wg[(c + 16u) * 8u + (kq ^ nx)];
      let bg2 = wg[(c + 32u) * 8u + (kq ^ nx)];
      let bg3 = wg[(c + 48u) * 8u + (kq ^ nx)];
      let bu0 = wu[c * 8u + (kq ^ nx)];
      let bu1 = wu[(c + 16u) * 8u + (kq ^ nx)];
      let bu2 = wu[(c + 32u) * 8u + (kq ^ nx)];
      let bu3 = wu[(c + 48u) * 8u + (kq ^ nx)];
      let a0 = xs[(lid.y + 0u) * 8u + kq];
      let a1 = xs[(lid.y + 16u) * 8u + kq];
      let a2 = xs[(lid.y + 32u) * 8u + kq];
      let a3 = xs[(lid.y + 48u) * 8u + kq];
      gate0 += vec4<f32>(dot(a0, bg0), dot(a0, bg1), dot(a0, bg2), dot(a0, bg3));
      gate1 += vec4<f32>(dot(a1, bg0), dot(a1, bg1), dot(a1, bg2), dot(a1, bg3));
      gate2 += vec4<f32>(dot(a2, bg0), dot(a2, bg1), dot(a2, bg2), dot(a2, bg3));
      gate3 += vec4<f32>(dot(a3, bg0), dot(a3, bg1), dot(a3, bg2), dot(a3, bg3));
      up0 += vec4<f32>(dot(a0, bu0), dot(a0, bu1), dot(a0, bu2), dot(a0, bu3));
      up1 += vec4<f32>(dot(a1, bu0), dot(a1, bu1), dot(a1, bu2), dot(a1, bu3));
      up2 += vec4<f32>(dot(a2, bu0), dot(a2, bu1), dot(a2, bu2), dot(a2, bu3));
      up3 += vec4<f32>(dot(a3, bu0), dot(a3, bu1), dot(a3, bu2), dot(a3, bu3));
    }
    {
      let kq = 3u;
      let bg0 = wg[c * 8u + (kq ^ nx)];
      let bg1 = wg[(c + 16u) * 8u + (kq ^ nx)];
      let bg2 = wg[(c + 32u) * 8u + (kq ^ nx)];
      let bg3 = wg[(c + 48u) * 8u + (kq ^ nx)];
      let bu0 = wu[c * 8u + (kq ^ nx)];
      let bu1 = wu[(c + 16u) * 8u + (kq ^ nx)];
      let bu2 = wu[(c + 32u) * 8u + (kq ^ nx)];
      let bu3 = wu[(c + 48u) * 8u + (kq ^ nx)];
      let a0 = xs[(lid.y + 0u) * 8u + kq];
      let a1 = xs[(lid.y + 16u) * 8u + kq];
      let a2 = xs[(lid.y + 32u) * 8u + kq];
      let a3 = xs[(lid.y + 48u) * 8u + kq];
      gate0 += vec4<f32>(dot(a0, bg0), dot(a0, bg1), dot(a0, bg2), dot(a0, bg3));
      gate1 += vec4<f32>(dot(a1, bg0), dot(a1, bg1), dot(a1, bg2), dot(a1, bg3));
      gate2 += vec4<f32>(dot(a2, bg0), dot(a2, bg1), dot(a2, bg2), dot(a2, bg3));
      gate3 += vec4<f32>(dot(a3, bg0), dot(a3, bg1), dot(a3, bg2), dot(a3, bg3));
      up0 += vec4<f32>(dot(a0, bu0), dot(a0, bu1), dot(a0, bu2), dot(a0, bu3));
      up1 += vec4<f32>(dot(a1, bu0), dot(a1, bu1), dot(a1, bu2), dot(a1, bu3));
      up2 += vec4<f32>(dot(a2, bu0), dot(a2, bu1), dot(a2, bu2), dot(a2, bu3));
      up3 += vec4<f32>(dot(a3, bu0), dot(a3, bu1), dot(a3, bu2), dot(a3, bu3));
    }
    {
      let kq = 4u;
      let bg0 = wg[c * 8u + (kq ^ nx)];
      let bg1 = wg[(c + 16u) * 8u + (kq ^ nx)];
      let bg2 = wg[(c + 32u) * 8u + (kq ^ nx)];
      let bg3 = wg[(c + 48u) * 8u + (kq ^ nx)];
      let bu0 = wu[c * 8u + (kq ^ nx)];
      let bu1 = wu[(c + 16u) * 8u + (kq ^ nx)];
      let bu2 = wu[(c + 32u) * 8u + (kq ^ nx)];
      let bu3 = wu[(c + 48u) * 8u + (kq ^ nx)];
      let a0 = xs[(lid.y + 0u) * 8u + kq];
      let a1 = xs[(lid.y + 16u) * 8u + kq];
      let a2 = xs[(lid.y + 32u) * 8u + kq];
      let a3 = xs[(lid.y + 48u) * 8u + kq];
      gate0 += vec4<f32>(dot(a0, bg0), dot(a0, bg1), dot(a0, bg2), dot(a0, bg3));
      gate1 += vec4<f32>(dot(a1, bg0), dot(a1, bg1), dot(a1, bg2), dot(a1, bg3));
      gate2 += vec4<f32>(dot(a2, bg0), dot(a2, bg1), dot(a2, bg2), dot(a2, bg3));
      gate3 += vec4<f32>(dot(a3, bg0), dot(a3, bg1), dot(a3, bg2), dot(a3, bg3));
      up0 += vec4<f32>(dot(a0, bu0), dot(a0, bu1), dot(a0, bu2), dot(a0, bu3));
      up1 += vec4<f32>(dot(a1, bu0), dot(a1, bu1), dot(a1, bu2), dot(a1, bu3));
      up2 += vec4<f32>(dot(a2, bu0), dot(a2, bu1), dot(a2, bu2), dot(a2, bu3));
      up3 += vec4<f32>(dot(a3, bu0), dot(a3, bu1), dot(a3, bu2), dot(a3, bu3));
    }
    {
      let kq = 5u;
      let bg0 = wg[c * 8u + (kq ^ nx)];
      let bg1 = wg[(c + 16u) * 8u + (kq ^ nx)];
      let bg2 = wg[(c + 32u) * 8u + (kq ^ nx)];
      let bg3 = wg[(c + 48u) * 8u + (kq ^ nx)];
      let bu0 = wu[c * 8u + (kq ^ nx)];
      let bu1 = wu[(c + 16u) * 8u + (kq ^ nx)];
      let bu2 = wu[(c + 32u) * 8u + (kq ^ nx)];
      let bu3 = wu[(c + 48u) * 8u + (kq ^ nx)];
      let a0 = xs[(lid.y + 0u) * 8u + kq];
      let a1 = xs[(lid.y + 16u) * 8u + kq];
      let a2 = xs[(lid.y + 32u) * 8u + kq];
      let a3 = xs[(lid.y + 48u) * 8u + kq];
      gate0 += vec4<f32>(dot(a0, bg0), dot(a0, bg1), dot(a0, bg2), dot(a0, bg3));
      gate1 += vec4<f32>(dot(a1, bg0), dot(a1, bg1), dot(a1, bg2), dot(a1, bg3));
      gate2 += vec4<f32>(dot(a2, bg0), dot(a2, bg1), dot(a2, bg2), dot(a2, bg3));
      gate3 += vec4<f32>(dot(a3, bg0), dot(a3, bg1), dot(a3, bg2), dot(a3, bg3));
      up0 += vec4<f32>(dot(a0, bu0), dot(a0, bu1), dot(a0, bu2), dot(a0, bu3));
      up1 += vec4<f32>(dot(a1, bu0), dot(a1, bu1), dot(a1, bu2), dot(a1, bu3));
      up2 += vec4<f32>(dot(a2, bu0), dot(a2, bu1), dot(a2, bu2), dot(a2, bu3));
      up3 += vec4<f32>(dot(a3, bu0), dot(a3, bu1), dot(a3, bu2), dot(a3, bu3));
    }
    {
      let kq = 6u;
      let bg0 = wg[c * 8u + (kq ^ nx)];
      let bg1 = wg[(c + 16u) * 8u + (kq ^ nx)];
      let bg2 = wg[(c + 32u) * 8u + (kq ^ nx)];
      let bg3 = wg[(c + 48u) * 8u + (kq ^ nx)];
      let bu0 = wu[c * 8u + (kq ^ nx)];
      let bu1 = wu[(c + 16u) * 8u + (kq ^ nx)];
      let bu2 = wu[(c + 32u) * 8u + (kq ^ nx)];
      let bu3 = wu[(c + 48u) * 8u + (kq ^ nx)];
      let a0 = xs[(lid.y + 0u) * 8u + kq];
      let a1 = xs[(lid.y + 16u) * 8u + kq];
      let a2 = xs[(lid.y + 32u) * 8u + kq];
      let a3 = xs[(lid.y + 48u) * 8u + kq];
      gate0 += vec4<f32>(dot(a0, bg0), dot(a0, bg1), dot(a0, bg2), dot(a0, bg3));
      gate1 += vec4<f32>(dot(a1, bg0), dot(a1, bg1), dot(a1, bg2), dot(a1, bg3));
      gate2 += vec4<f32>(dot(a2, bg0), dot(a2, bg1), dot(a2, bg2), dot(a2, bg3));
      gate3 += vec4<f32>(dot(a3, bg0), dot(a3, bg1), dot(a3, bg2), dot(a3, bg3));
      up0 += vec4<f32>(dot(a0, bu0), dot(a0, bu1), dot(a0, bu2), dot(a0, bu3));
      up1 += vec4<f32>(dot(a1, bu0), dot(a1, bu1), dot(a1, bu2), dot(a1, bu3));
      up2 += vec4<f32>(dot(a2, bu0), dot(a2, bu1), dot(a2, bu2), dot(a2, bu3));
      up3 += vec4<f32>(dot(a3, bu0), dot(a3, bu1), dot(a3, bu2), dot(a3, bu3));
    }
    {
      let kq = 7u;
      let bg0 = wg[c * 8u + (kq ^ nx)];
      let bg1 = wg[(c + 16u) * 8u + (kq ^ nx)];
      let bg2 = wg[(c + 32u) * 8u + (kq ^ nx)];
      let bg3 = wg[(c + 48u) * 8u + (kq ^ nx)];
      let bu0 = wu[c * 8u + (kq ^ nx)];
      let bu1 = wu[(c + 16u) * 8u + (kq ^ nx)];
      let bu2 = wu[(c + 32u) * 8u + (kq ^ nx)];
      let bu3 = wu[(c + 48u) * 8u + (kq ^ nx)];
      let a0 = xs[(lid.y + 0u) * 8u + kq];
      let a1 = xs[(lid.y + 16u) * 8u + kq];
      let a2 = xs[(lid.y + 32u) * 8u + kq];
      let a3 = xs[(lid.y + 48u) * 8u + kq];
      gate0 += vec4<f32>(dot(a0, bg0), dot(a0, bg1), dot(a0, bg2), dot(a0, bg3));
      gate1 += vec4<f32>(dot(a1, bg0), dot(a1, bg1), dot(a1, bg2), dot(a1, bg3));
      gate2 += vec4<f32>(dot(a2, bg0), dot(a2, bg1), dot(a2, bg2), dot(a2, bg3));
      gate3 += vec4<f32>(dot(a3, bg0), dot(a3, bg1), dot(a3, bg2), dot(a3, bg3));
      up0 += vec4<f32>(dot(a0, bu0), dot(a0, bu1), dot(a0, bu2), dot(a0, bu3));
      up1 += vec4<f32>(dot(a1, bu0), dot(a1, bu1), dot(a1, bu2), dot(a1, bu3));
      up2 += vec4<f32>(dot(a2, bu0), dot(a2, bu1), dot(a2, bu2), dot(a2, bu3));
      up3 += vec4<f32>(dot(a3, bu0), dot(a3, bu1), dot(a3, bu2), dot(a3, bu3));
    }
    workgroupBarrier();
  }

  for (var j = 0u; j < 4u; j++) {
    let col = n0 + lid.x + 16u * j;
    if (col >= N) { break; }
    let g0 = gate0[j];
    let g1 = gate1[j];
    let g2 = gate2[j];
    let g3 = gate3[j];
    let u0 = up0[j];
    let u1 = up1[j];
    let u2 = up2[j];
    let u3 = up3[j];
    for (var i = 0u; i < 4u; i++) {
      let row = m0 + lid.y + 16u * i;
      if (row >= T) { break; }
      var gate = g0;
      var up = u0;
      if (i == 1u) { gate = g1; up = u1; }
      else if (i == 2u) { gate = g2; up = u2; }
      else if (i == 3u) { gate = g3; up = u3; }
      Y[row * N + col] = gelu(gate) * up;
    }
  }
}
