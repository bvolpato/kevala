// Fused Q8 paired projections followed by a gated activation.
//
// A 32 x 8 workgroup covers 64 rows and 64 columns. Each thread owns eight
// rows y + 8 i and two columns x and x + 32. X is staged once per K block. The Q8
// tiles are staged independently, but their dot products use the same K/4
// order and FP32 accumulators as matmul.wgsl. This prototype intentionally
// has no split-K path; the harness rejects cases for which the runtime split
// rule would choose more than one split.
//#include common

struct P { N: u32, K: u32, upRowOffset: u32, activation: u32 }
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

fn erf(x: f32) -> f32 {
  let a = abs(x);
  if (a < 0.5) {
    let z = x * x;
    return x * (1.1283791671 + z * (-0.3761263890 + z * (0.1128379167 + z * (-0.0268661706 + z * 0.0052239776))));
  }
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return select(-y, y, x >= 0.0);
}

fn activate(x: f32) -> f32 {
  if (p.activation == 1u) { return 0.5 * x * (1.0 + erf(x * 0.70710678118)); }
  if (p.activation == 2u) { return x / (1.0 + exp(-x)); }
  let c = 0.7978845608028654;
  return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

fn pair_store(gate: f32, up: f32, row: u32, col: u32) {
  if (row >= g.T || col >= p.N) { return; }
  Y[row * p.N + col] = activate(gate) * up;
}

@compute @workgroup_size(32, 8)
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

  var gate0 = vec2<f32>(0.0);
  var gate1 = vec2<f32>(0.0);
  var gate2 = vec2<f32>(0.0);
  var gate3 = vec2<f32>(0.0);
  var gate4 = vec2<f32>(0.0);
  var gate5 = vec2<f32>(0.0);
  var gate6 = vec2<f32>(0.0);
  var gate7 = vec2<f32>(0.0);
  var up0 = vec2<f32>(0.0);
  var up1 = vec2<f32>(0.0);
  var up2 = vec2<f32>(0.0);
  var up3 = vec2<f32>(0.0);
  var up4 = vec2<f32>(0.0);
  var up5 = vec2<f32>(0.0);
  var up6 = vec2<f32>(0.0);
  var up7 = vec2<f32>(0.0);

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
      let uv = WU[gi + p.upRowOffset * K / 16u];
      let glo = select(gv.xy, gv.zw, (lq & 1u) == 1u);
      let ulo = select(uv.xy, uv.zw, (lq & 1u) == 1u);
      let gs = SG[wrow * nb + kb];
      let us = SU[(wrow + p.upRowOffset) * nb + kb];
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
      let bg1 = wg[(c + 32u) * 8u + (kq ^ nx)];
      let bu0 = wu[c * 8u + (kq ^ nx)];
      let bu1 = wu[(c + 32u) * 8u + (kq ^ nx)];
      let a0 = xs[(lid.y + 0u) * 8u + kq];
      gate0 += vec2<f32>(dot(a0, bg0), dot(a0, bg1));
      up0 += vec2<f32>(dot(a0, bu0), dot(a0, bu1));
      let a1 = xs[(lid.y + 8u) * 8u + kq];
      gate1 += vec2<f32>(dot(a1, bg0), dot(a1, bg1));
      up1 += vec2<f32>(dot(a1, bu0), dot(a1, bu1));
      let a2 = xs[(lid.y + 16u) * 8u + kq];
      gate2 += vec2<f32>(dot(a2, bg0), dot(a2, bg1));
      up2 += vec2<f32>(dot(a2, bu0), dot(a2, bu1));
      let a3 = xs[(lid.y + 24u) * 8u + kq];
      gate3 += vec2<f32>(dot(a3, bg0), dot(a3, bg1));
      up3 += vec2<f32>(dot(a3, bu0), dot(a3, bu1));
      let a4 = xs[(lid.y + 32u) * 8u + kq];
      gate4 += vec2<f32>(dot(a4, bg0), dot(a4, bg1));
      up4 += vec2<f32>(dot(a4, bu0), dot(a4, bu1));
      let a5 = xs[(lid.y + 40u) * 8u + kq];
      gate5 += vec2<f32>(dot(a5, bg0), dot(a5, bg1));
      up5 += vec2<f32>(dot(a5, bu0), dot(a5, bu1));
      let a6 = xs[(lid.y + 48u) * 8u + kq];
      gate6 += vec2<f32>(dot(a6, bg0), dot(a6, bg1));
      up6 += vec2<f32>(dot(a6, bu0), dot(a6, bu1));
      let a7 = xs[(lid.y + 56u) * 8u + kq];
      gate7 += vec2<f32>(dot(a7, bg0), dot(a7, bg1));
      up7 += vec2<f32>(dot(a7, bu0), dot(a7, bu1));
    }
    {
      let kq = 1u;
      let bg0 = wg[c * 8u + (kq ^ nx)];
      let bg1 = wg[(c + 32u) * 8u + (kq ^ nx)];
      let bu0 = wu[c * 8u + (kq ^ nx)];
      let bu1 = wu[(c + 32u) * 8u + (kq ^ nx)];
      let a0 = xs[(lid.y + 0u) * 8u + kq];
      gate0 += vec2<f32>(dot(a0, bg0), dot(a0, bg1));
      up0 += vec2<f32>(dot(a0, bu0), dot(a0, bu1));
      let a1 = xs[(lid.y + 8u) * 8u + kq];
      gate1 += vec2<f32>(dot(a1, bg0), dot(a1, bg1));
      up1 += vec2<f32>(dot(a1, bu0), dot(a1, bu1));
      let a2 = xs[(lid.y + 16u) * 8u + kq];
      gate2 += vec2<f32>(dot(a2, bg0), dot(a2, bg1));
      up2 += vec2<f32>(dot(a2, bu0), dot(a2, bu1));
      let a3 = xs[(lid.y + 24u) * 8u + kq];
      gate3 += vec2<f32>(dot(a3, bg0), dot(a3, bg1));
      up3 += vec2<f32>(dot(a3, bu0), dot(a3, bu1));
      let a4 = xs[(lid.y + 32u) * 8u + kq];
      gate4 += vec2<f32>(dot(a4, bg0), dot(a4, bg1));
      up4 += vec2<f32>(dot(a4, bu0), dot(a4, bu1));
      let a5 = xs[(lid.y + 40u) * 8u + kq];
      gate5 += vec2<f32>(dot(a5, bg0), dot(a5, bg1));
      up5 += vec2<f32>(dot(a5, bu0), dot(a5, bu1));
      let a6 = xs[(lid.y + 48u) * 8u + kq];
      gate6 += vec2<f32>(dot(a6, bg0), dot(a6, bg1));
      up6 += vec2<f32>(dot(a6, bu0), dot(a6, bu1));
      let a7 = xs[(lid.y + 56u) * 8u + kq];
      gate7 += vec2<f32>(dot(a7, bg0), dot(a7, bg1));
      up7 += vec2<f32>(dot(a7, bu0), dot(a7, bu1));
    }
    {
      let kq = 2u;
      let bg0 = wg[c * 8u + (kq ^ nx)];
      let bg1 = wg[(c + 32u) * 8u + (kq ^ nx)];
      let bu0 = wu[c * 8u + (kq ^ nx)];
      let bu1 = wu[(c + 32u) * 8u + (kq ^ nx)];
      let a0 = xs[(lid.y + 0u) * 8u + kq];
      gate0 += vec2<f32>(dot(a0, bg0), dot(a0, bg1));
      up0 += vec2<f32>(dot(a0, bu0), dot(a0, bu1));
      let a1 = xs[(lid.y + 8u) * 8u + kq];
      gate1 += vec2<f32>(dot(a1, bg0), dot(a1, bg1));
      up1 += vec2<f32>(dot(a1, bu0), dot(a1, bu1));
      let a2 = xs[(lid.y + 16u) * 8u + kq];
      gate2 += vec2<f32>(dot(a2, bg0), dot(a2, bg1));
      up2 += vec2<f32>(dot(a2, bu0), dot(a2, bu1));
      let a3 = xs[(lid.y + 24u) * 8u + kq];
      gate3 += vec2<f32>(dot(a3, bg0), dot(a3, bg1));
      up3 += vec2<f32>(dot(a3, bu0), dot(a3, bu1));
      let a4 = xs[(lid.y + 32u) * 8u + kq];
      gate4 += vec2<f32>(dot(a4, bg0), dot(a4, bg1));
      up4 += vec2<f32>(dot(a4, bu0), dot(a4, bu1));
      let a5 = xs[(lid.y + 40u) * 8u + kq];
      gate5 += vec2<f32>(dot(a5, bg0), dot(a5, bg1));
      up5 += vec2<f32>(dot(a5, bu0), dot(a5, bu1));
      let a6 = xs[(lid.y + 48u) * 8u + kq];
      gate6 += vec2<f32>(dot(a6, bg0), dot(a6, bg1));
      up6 += vec2<f32>(dot(a6, bu0), dot(a6, bu1));
      let a7 = xs[(lid.y + 56u) * 8u + kq];
      gate7 += vec2<f32>(dot(a7, bg0), dot(a7, bg1));
      up7 += vec2<f32>(dot(a7, bu0), dot(a7, bu1));
    }
    {
      let kq = 3u;
      let bg0 = wg[c * 8u + (kq ^ nx)];
      let bg1 = wg[(c + 32u) * 8u + (kq ^ nx)];
      let bu0 = wu[c * 8u + (kq ^ nx)];
      let bu1 = wu[(c + 32u) * 8u + (kq ^ nx)];
      let a0 = xs[(lid.y + 0u) * 8u + kq];
      gate0 += vec2<f32>(dot(a0, bg0), dot(a0, bg1));
      up0 += vec2<f32>(dot(a0, bu0), dot(a0, bu1));
      let a1 = xs[(lid.y + 8u) * 8u + kq];
      gate1 += vec2<f32>(dot(a1, bg0), dot(a1, bg1));
      up1 += vec2<f32>(dot(a1, bu0), dot(a1, bu1));
      let a2 = xs[(lid.y + 16u) * 8u + kq];
      gate2 += vec2<f32>(dot(a2, bg0), dot(a2, bg1));
      up2 += vec2<f32>(dot(a2, bu0), dot(a2, bu1));
      let a3 = xs[(lid.y + 24u) * 8u + kq];
      gate3 += vec2<f32>(dot(a3, bg0), dot(a3, bg1));
      up3 += vec2<f32>(dot(a3, bu0), dot(a3, bu1));
      let a4 = xs[(lid.y + 32u) * 8u + kq];
      gate4 += vec2<f32>(dot(a4, bg0), dot(a4, bg1));
      up4 += vec2<f32>(dot(a4, bu0), dot(a4, bu1));
      let a5 = xs[(lid.y + 40u) * 8u + kq];
      gate5 += vec2<f32>(dot(a5, bg0), dot(a5, bg1));
      up5 += vec2<f32>(dot(a5, bu0), dot(a5, bu1));
      let a6 = xs[(lid.y + 48u) * 8u + kq];
      gate6 += vec2<f32>(dot(a6, bg0), dot(a6, bg1));
      up6 += vec2<f32>(dot(a6, bu0), dot(a6, bu1));
      let a7 = xs[(lid.y + 56u) * 8u + kq];
      gate7 += vec2<f32>(dot(a7, bg0), dot(a7, bg1));
      up7 += vec2<f32>(dot(a7, bu0), dot(a7, bu1));
    }
    {
      let kq = 4u;
      let bg0 = wg[c * 8u + (kq ^ nx)];
      let bg1 = wg[(c + 32u) * 8u + (kq ^ nx)];
      let bu0 = wu[c * 8u + (kq ^ nx)];
      let bu1 = wu[(c + 32u) * 8u + (kq ^ nx)];
      let a0 = xs[(lid.y + 0u) * 8u + kq];
      gate0 += vec2<f32>(dot(a0, bg0), dot(a0, bg1));
      up0 += vec2<f32>(dot(a0, bu0), dot(a0, bu1));
      let a1 = xs[(lid.y + 8u) * 8u + kq];
      gate1 += vec2<f32>(dot(a1, bg0), dot(a1, bg1));
      up1 += vec2<f32>(dot(a1, bu0), dot(a1, bu1));
      let a2 = xs[(lid.y + 16u) * 8u + kq];
      gate2 += vec2<f32>(dot(a2, bg0), dot(a2, bg1));
      up2 += vec2<f32>(dot(a2, bu0), dot(a2, bu1));
      let a3 = xs[(lid.y + 24u) * 8u + kq];
      gate3 += vec2<f32>(dot(a3, bg0), dot(a3, bg1));
      up3 += vec2<f32>(dot(a3, bu0), dot(a3, bu1));
      let a4 = xs[(lid.y + 32u) * 8u + kq];
      gate4 += vec2<f32>(dot(a4, bg0), dot(a4, bg1));
      up4 += vec2<f32>(dot(a4, bu0), dot(a4, bu1));
      let a5 = xs[(lid.y + 40u) * 8u + kq];
      gate5 += vec2<f32>(dot(a5, bg0), dot(a5, bg1));
      up5 += vec2<f32>(dot(a5, bu0), dot(a5, bu1));
      let a6 = xs[(lid.y + 48u) * 8u + kq];
      gate6 += vec2<f32>(dot(a6, bg0), dot(a6, bg1));
      up6 += vec2<f32>(dot(a6, bu0), dot(a6, bu1));
      let a7 = xs[(lid.y + 56u) * 8u + kq];
      gate7 += vec2<f32>(dot(a7, bg0), dot(a7, bg1));
      up7 += vec2<f32>(dot(a7, bu0), dot(a7, bu1));
    }
    {
      let kq = 5u;
      let bg0 = wg[c * 8u + (kq ^ nx)];
      let bg1 = wg[(c + 32u) * 8u + (kq ^ nx)];
      let bu0 = wu[c * 8u + (kq ^ nx)];
      let bu1 = wu[(c + 32u) * 8u + (kq ^ nx)];
      let a0 = xs[(lid.y + 0u) * 8u + kq];
      gate0 += vec2<f32>(dot(a0, bg0), dot(a0, bg1));
      up0 += vec2<f32>(dot(a0, bu0), dot(a0, bu1));
      let a1 = xs[(lid.y + 8u) * 8u + kq];
      gate1 += vec2<f32>(dot(a1, bg0), dot(a1, bg1));
      up1 += vec2<f32>(dot(a1, bu0), dot(a1, bu1));
      let a2 = xs[(lid.y + 16u) * 8u + kq];
      gate2 += vec2<f32>(dot(a2, bg0), dot(a2, bg1));
      up2 += vec2<f32>(dot(a2, bu0), dot(a2, bu1));
      let a3 = xs[(lid.y + 24u) * 8u + kq];
      gate3 += vec2<f32>(dot(a3, bg0), dot(a3, bg1));
      up3 += vec2<f32>(dot(a3, bu0), dot(a3, bu1));
      let a4 = xs[(lid.y + 32u) * 8u + kq];
      gate4 += vec2<f32>(dot(a4, bg0), dot(a4, bg1));
      up4 += vec2<f32>(dot(a4, bu0), dot(a4, bu1));
      let a5 = xs[(lid.y + 40u) * 8u + kq];
      gate5 += vec2<f32>(dot(a5, bg0), dot(a5, bg1));
      up5 += vec2<f32>(dot(a5, bu0), dot(a5, bu1));
      let a6 = xs[(lid.y + 48u) * 8u + kq];
      gate6 += vec2<f32>(dot(a6, bg0), dot(a6, bg1));
      up6 += vec2<f32>(dot(a6, bu0), dot(a6, bu1));
      let a7 = xs[(lid.y + 56u) * 8u + kq];
      gate7 += vec2<f32>(dot(a7, bg0), dot(a7, bg1));
      up7 += vec2<f32>(dot(a7, bu0), dot(a7, bu1));
    }
    {
      let kq = 6u;
      let bg0 = wg[c * 8u + (kq ^ nx)];
      let bg1 = wg[(c + 32u) * 8u + (kq ^ nx)];
      let bu0 = wu[c * 8u + (kq ^ nx)];
      let bu1 = wu[(c + 32u) * 8u + (kq ^ nx)];
      let a0 = xs[(lid.y + 0u) * 8u + kq];
      gate0 += vec2<f32>(dot(a0, bg0), dot(a0, bg1));
      up0 += vec2<f32>(dot(a0, bu0), dot(a0, bu1));
      let a1 = xs[(lid.y + 8u) * 8u + kq];
      gate1 += vec2<f32>(dot(a1, bg0), dot(a1, bg1));
      up1 += vec2<f32>(dot(a1, bu0), dot(a1, bu1));
      let a2 = xs[(lid.y + 16u) * 8u + kq];
      gate2 += vec2<f32>(dot(a2, bg0), dot(a2, bg1));
      up2 += vec2<f32>(dot(a2, bu0), dot(a2, bu1));
      let a3 = xs[(lid.y + 24u) * 8u + kq];
      gate3 += vec2<f32>(dot(a3, bg0), dot(a3, bg1));
      up3 += vec2<f32>(dot(a3, bu0), dot(a3, bu1));
      let a4 = xs[(lid.y + 32u) * 8u + kq];
      gate4 += vec2<f32>(dot(a4, bg0), dot(a4, bg1));
      up4 += vec2<f32>(dot(a4, bu0), dot(a4, bu1));
      let a5 = xs[(lid.y + 40u) * 8u + kq];
      gate5 += vec2<f32>(dot(a5, bg0), dot(a5, bg1));
      up5 += vec2<f32>(dot(a5, bu0), dot(a5, bu1));
      let a6 = xs[(lid.y + 48u) * 8u + kq];
      gate6 += vec2<f32>(dot(a6, bg0), dot(a6, bg1));
      up6 += vec2<f32>(dot(a6, bu0), dot(a6, bu1));
      let a7 = xs[(lid.y + 56u) * 8u + kq];
      gate7 += vec2<f32>(dot(a7, bg0), dot(a7, bg1));
      up7 += vec2<f32>(dot(a7, bu0), dot(a7, bu1));
    }
    {
      let kq = 7u;
      let bg0 = wg[c * 8u + (kq ^ nx)];
      let bg1 = wg[(c + 32u) * 8u + (kq ^ nx)];
      let bu0 = wu[c * 8u + (kq ^ nx)];
      let bu1 = wu[(c + 32u) * 8u + (kq ^ nx)];
      let a0 = xs[(lid.y + 0u) * 8u + kq];
      gate0 += vec2<f32>(dot(a0, bg0), dot(a0, bg1));
      up0 += vec2<f32>(dot(a0, bu0), dot(a0, bu1));
      let a1 = xs[(lid.y + 8u) * 8u + kq];
      gate1 += vec2<f32>(dot(a1, bg0), dot(a1, bg1));
      up1 += vec2<f32>(dot(a1, bu0), dot(a1, bu1));
      let a2 = xs[(lid.y + 16u) * 8u + kq];
      gate2 += vec2<f32>(dot(a2, bg0), dot(a2, bg1));
      up2 += vec2<f32>(dot(a2, bu0), dot(a2, bu1));
      let a3 = xs[(lid.y + 24u) * 8u + kq];
      gate3 += vec2<f32>(dot(a3, bg0), dot(a3, bg1));
      up3 += vec2<f32>(dot(a3, bu0), dot(a3, bu1));
      let a4 = xs[(lid.y + 32u) * 8u + kq];
      gate4 += vec2<f32>(dot(a4, bg0), dot(a4, bg1));
      up4 += vec2<f32>(dot(a4, bu0), dot(a4, bu1));
      let a5 = xs[(lid.y + 40u) * 8u + kq];
      gate5 += vec2<f32>(dot(a5, bg0), dot(a5, bg1));
      up5 += vec2<f32>(dot(a5, bu0), dot(a5, bu1));
      let a6 = xs[(lid.y + 48u) * 8u + kq];
      gate6 += vec2<f32>(dot(a6, bg0), dot(a6, bg1));
      up6 += vec2<f32>(dot(a6, bu0), dot(a6, bu1));
      let a7 = xs[(lid.y + 56u) * 8u + kq];
      gate7 += vec2<f32>(dot(a7, bg0), dot(a7, bg1));
      up7 += vec2<f32>(dot(a7, bu0), dot(a7, bu1));
    }
    workgroupBarrier();
  }

  pair_store(gate0.x, up0.x, m0 + lid.y + 0u, n0 + lid.x + 0u);
  pair_store(gate1.x, up1.x, m0 + lid.y + 8u, n0 + lid.x + 0u);
  pair_store(gate2.x, up2.x, m0 + lid.y + 16u, n0 + lid.x + 0u);
  pair_store(gate3.x, up3.x, m0 + lid.y + 24u, n0 + lid.x + 0u);
  pair_store(gate4.x, up4.x, m0 + lid.y + 32u, n0 + lid.x + 0u);
  pair_store(gate5.x, up5.x, m0 + lid.y + 40u, n0 + lid.x + 0u);
  pair_store(gate6.x, up6.x, m0 + lid.y + 48u, n0 + lid.x + 0u);
  pair_store(gate7.x, up7.x, m0 + lid.y + 56u, n0 + lid.x + 0u);
  pair_store(gate0.y, up0.y, m0 + lid.y + 0u, n0 + lid.x + 32u);
  pair_store(gate1.y, up1.y, m0 + lid.y + 8u, n0 + lid.x + 32u);
  pair_store(gate2.y, up2.y, m0 + lid.y + 16u, n0 + lid.x + 32u);
  pair_store(gate3.y, up3.y, m0 + lid.y + 24u, n0 + lid.x + 32u);
  pair_store(gate4.y, up4.y, m0 + lid.y + 32u, n0 + lid.x + 32u);
  pair_store(gate5.y, up5.y, m0 + lid.y + 40u, n0 + lid.x + 32u);
  pair_store(gate6.y, up6.y, m0 + lid.y + 48u, n0 + lid.x + 32u);
  pair_store(gate7.y, up7.y, m0 + lid.y + 56u, n0 + lid.x + 32u);
}
