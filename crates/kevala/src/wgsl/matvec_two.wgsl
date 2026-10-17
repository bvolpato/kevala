// Q8 projections for exactly two independent rows, sharing each weight tile.
//#if F16
enable f16;
//#else
//#if SUBGROUPS
enable subgroups;
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

fn sx(word: u32) -> vec4<f32> {
  let bytes = vec4<i32>(bitcast<i32>(word << 24u), bitcast<i32>(word << 16u), bitcast<i32>(word << 8u), bitcast<i32>(word)) >> vec4<u32>(24u);
  return vec4<f32>(bytes);
}

//#if F16
var<workgroup> weights: array<vec4<f16>, 256>;
//#include splits
var<workgroup> pair_inputs: array<vec4<f16>, 32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  if (g.T != 2u) { return; }
  let N = p.N;
  let K = p.K;
  let nb = K / 32u;
  let splits = mm_splits(g.T, N, K);
  let per = (nb + splits - 1u) / splits;
  let load_row = li / 4u;
  let load_col = li & 3u;
  let load_output = wg.x * 16u + load_row;
  let row = li / 16u;
  let output_col = li & 15u;
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
        if (li < 8u) {
          let input_row = li / 4u;
          var x0 = vec4<f32>(0.0);
          var x1 = vec4<f32>(0.0);
          if (kb < kb1) {
            let base = (input_row * K + kb * 32u) / 4u + load_col * 2u;
            x0 = X[base];
            x1 = X[base + 1u];
          }
          pair_inputs[input_row * 16u + at] = vec4<f16>(x0);
          pair_inputs[input_row * 16u + at + 1u] = vec4<f16>(x1);
        }
      }
      workgroupBarrier();
      if (li < 32u) {
        let count = min(16u, (kb1 - kb_base) * 8u);
        for (var kq = 0u; kq < count; kq++) {
          let input = vec4<f32>(pair_inputs[row * 16u + kq]);
          let weight = vec4<f32>(weights[output_col * 16u + (kq ^ (output_col & 7u))]);
          partial += dot(input, weight);
        }
      }
      workgroupBarrier();
    }
    // Match the production ascending split reduction and unsplit signed-zero path.
    if (splits == 1u) { total = partial; } else { total += partial; }
  }
  let col = wg.x * 16u + output_col;
  if (li < 32u && col < N) {
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
//#if SUBGROUPS
fn ordered_vector_sum32(prefix: vec2<f32>, subgroupLane: u32) -> vec2<f32> {
  var aggregate = prefix;
  for (var step = 16u; step > 0u; step >>= 1u) {
    let other = subgroupShuffleDown(aggregate, step);
    if (subgroupLane < step) { aggregate += other; }
  }
  return aggregate;
}
//#endif
var<workgroup> pair_partials: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
//#if SUBGROUPS
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(subgroup_invocation_id) subgroupLane: u32) {
//#else
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
//#endif
  if (g.T != 2u) { return; }
  let N = p.N;
  let K = p.K;
  let lane = li & 63u;
  let col = wg.x * 4u + li / 64u;
  let k4 = K / 4u;
  var sum = vec2<f32>(0.0);
  if (col < N) {
    for (var part = lane; part < k4; part += 64u) {
      let packed = W[col * (K / 16u) + part / 4u];
      let word = packed[part & 3u];
      let weight = sx(word) * S[col * (K / 32u) + part / 8u];
      sum += vec2<f32>(dot(X[part], weight), dot(X[k4 + part], weight));
    }
  }
  pair_partials[li] = sum;
  workgroupBarrier();
//#if SUBGROUPS
  var prefix = pair_partials[li];
  if (lane < 32u) { prefix += pair_partials[li + 32u]; }
  let reduced = ordered_vector_sum32(prefix, subgroupLane);
//#else
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lane < stride) { pair_partials[li] += pair_partials[li + stride]; }
    workgroupBarrier();
  }
//#endif
  if (lane == 0u && col < N) {
    for (var row = 0u; row < 2u; row++) {
      let at = row * N + col;
//#if SUBGROUPS
      var value = reduced[row];
//#else
      var value = pair_partials[li][row];
//#endif
      if (p.bias == 1u) { value += B[col]; }
      if (p.mode == 1u) { value += Y[at]; } else if (p.mode == 2u) { value = max(value, 0.0); }
      Y[at] = value;
    }
  }
}

//#endif
