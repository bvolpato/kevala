// Direct Y[T, N] = X[T, K] . W[N, K]^T for short inputs, with the matmul layout.
// Four output columns share a workgroup, with 64 lanes reducing K for each column.
// Stored Q8/BF16/F32 values expand to F32; no split-K scratch is needed.
// Q8 optionally rounds operands through F16 to match the existing F16-tiled matmul.
// BF16/F32 operands and all arithmetic remain F32.
// The Q8/F16 variant also retains the legacy split-K and serial dot accumulation order.
//#if WEIGHT_Q8
//#if F16
enable f16;
//#endif
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
//#if MATVEC_ORDERED
var<workgroup> inputs: array<vec4<f16>, 16>;
var<workgroup> weights: array<vec4<f16>, 256>;
//#else
var<workgroup> partials: array<f32, 256>;
//#endif

//#if WEIGHT_Q8
fn sx(word: u32) -> vec4<f32> {
  let bytes = vec4<i32>(bitcast<i32>(word << 24u), bitcast<i32>(word << 16u), bitcast<i32>(word << 8u), bitcast<i32>(word)) >> vec4<u32>(24u);
  return vec4<f32>(bytes);
}
//#endif
//#if WEIGHT_BF16
fn bf16(words: vec2<u32>) -> vec4<f32> {
  return bitcast<vec4<f32>>(vec4<u32>(words.x << 16u, words.x & 0xffff0000u, words.y << 16u, words.y & 0xffff0000u));
}
//#endif

//#if MATVEC_ORDERED
//#include splits

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
//#if SHAPE
  let N = {{N}}u;
  let K = {{K}}u;
//#else
  let N = p.N;
  let K = p.K;
//#endif
  let row = wg.y;
  if (row >= g.T) { return; }
  let nb = K / 32u;
  let splits = mm_splits(g.T, N, K);
  let per = (nb + splits - 1u) / splits;
  let load_row = li / 4u;
  let load_col = li & 3u;
  let load_output = wg.x * 16u + load_row;
  var total = 0.0;
  for (var split = 0u; split < splits; split++) {
    let kb0 = split * per;
    let kb1 = min(nb, kb0 + per);
    var partial = 0.0;
    for (var kb_base = kb0; kb_base < kb1; kb_base += 2u) {
      for (var half = 0u; half < 2u; half++) {
        let kb = kb_base + half;
        let at = half * 8u + load_col * 2u;
        var w0 = vec4<f32>(0.0);
        var w1 = vec4<f32>(0.0);
        if (load_output < N && kb < kb1) {
          let packed = W[(load_output * K + kb * 32u) / 16u + load_col / 2u];
          let words = select(packed.xy, packed.zw, (load_col & 1u) == 1u);
          let scale = S[load_output * nb + kb];
          w0 = sx(words.x) * scale;
          w1 = sx(words.y) * scale;
        }
        let swizzle = load_row & 7u;
        weights[load_row * 16u + (at ^ swizzle)] = vec4<f16>(w0);
        weights[load_row * 16u + ((at + 1u) ^ swizzle)] = vec4<f16>(w1);
        if (li < 4u) {
          var x0 = vec4<f32>(0.0);
          var x1 = vec4<f32>(0.0);
          if (kb < kb1) {
            let base = (row * K + kb * 32u) / 4u + load_col * 2u;
            x0 = X[base];
            x1 = X[base + 1u];
          }
          inputs[at] = vec4<f16>(x0);
          inputs[at + 1u] = vec4<f16>(x1);
        }
      }
      workgroupBarrier();
      if (li < 16u) {
        let count = min(16u, (kb1 - kb_base) * 8u);
        for (var kq = 0u; kq < count; kq++) {
          let input = vec4<f32>(inputs[kq]);
          let weight = vec4<f32>(weights[li * 16u + (kq ^ (li & 7u))]);
          partial += dot(input, weight);
        }
      }
      workgroupBarrier();
    }
    // The old reducer adds the original split totals in ascending split order.
    if (splits == 1u) { total = partial; } else { total += partial; }
  }
  let col = wg.x * 16u + li;
  if (li < 16u && col < N) {
    let at = row * N + col;
    if (splits == 1u) {
      var bias = 0.0;
      if (p.bias == 1u) { bias = B[col]; }
      total += bias;
    } else if (p.bias == 1u) { total += B[col]; }
    if (p.mode == 1u) { total += Y[at]; } else if (p.mode == 2u) { total = max(total, 0.0); }
    Y[at] = total;
  }
}
//#else
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
//#if SHAPE
  let N = {{N}}u;
  let K = {{K}}u;
//#else
  let N = p.N;
  let K = p.K;
//#endif
  let lane = li & 63u;
  let col = wg.x * 4u + li / 64u;
  let row = wg.y;
  let k4 = K / 4u;
  var sum = 0.0;
  if (col < N && row < g.T) {
    // Adjacent lanes cover adjacent K/4 values. Q8 scales are shared by eight lanes.
    for (var part = lane; part < k4; part += 64u) {
      var weight = vec4<f32>(0.0);
//#if WEIGHT_Q8
      let packed = W[col * (K / 16u) + part / 4u];
      let word = packed[part & 3u];
      weight = sx(word) * S[col * (K / 32u) + part / 8u];
//#if F16
      weight = vec4<f32>(vec4<f16>(weight));
//#endif
//#endif
//#if WEIGHT_BF16
      let packed = W[col * (K / 8u) + part / 2u];
      weight = bf16(select(packed.xy, packed.zw, (part & 1u) == 1u));
//#endif
//#if WEIGHT_F32
      weight = bitcast<vec4<f32>>(W[col * k4 + part]);
//#endif
      var input = X[row * k4 + part];
//#if WEIGHT_Q8
//#if F16
      input = vec4<f32>(vec4<f16>(input));
//#endif
//#endif
      sum += dot(input, weight);
    }
  }
  partials[li] = sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lane < stride) { partials[li] += partials[li + stride]; }
    workgroupBarrier();
  }
  if (lane == 0u && col < N && row < g.T) {
    let at = row * N + col;
    var value = partials[li];
    if (p.bias == 1u) { value += B[col]; }
    if (p.mode == 1u) { value += Y[at]; } else if (p.mode == 2u) { value = max(value, 0.0); }
    Y[at] = value;
  }
}
//#endif
