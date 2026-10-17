// Paired Q8 projections for two independent rows, sharing weights before exact gated GELU-tanh.
// Each output preserves its selected matvec's dot and reduction order independently.
//#if F16
enable f16;
//#else
//#if SUBGROUPS
enable subgroups;
//#endif
//#endif
//#include common

struct P { N: u32, K: u32, _a: u32, _b: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> G: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> GS: array<f32>;
@group(0) @binding(5) var<storage, read> UP: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> US: array<f32>;
@group(0) @binding(7) var<storage, read_write> Y: array<f32>;

fn sx(word: u32) -> vec4<f32> {
  let bytes = vec4<i32>(bitcast<i32>(word << 24u), bitcast<i32>(word << 16u), bitcast<i32>(word << 8u), bitcast<i32>(word)) >> vec4<u32>(24u);
  return vec4<f32>(bytes);
}

fn gelu(x: f32) -> f32 {
  if (x >= 10.0) { return x; }
  if (x <= -10.0) { return 0.0; }
  let c = 0.7978845608028654;
  return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

//#if F16
//#include splits
var<workgroup> inputs: array<vec4<f16>, 32>;
var<workgroup> gate_weights: array<vec4<f16>, 256>;
var<workgroup> up_weights: array<vec4<f16>, 256>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  if (g.T != 2u || wg.y != 0u) { return; }
  let N = p.N;
  let K = p.K;
  let nb = K / 32u;
  let splits = mm_splits(1u, N, K);
  let per = (nb + splits - 1u) / splits;
  let load_row = li / 4u;
  let load_col = li & 3u;
  let load_output = wg.x * 16u + load_row;
  let output_col = li & 15u;
  let row = li / 16u;
  var total_gate = 0.0;
  var total_up = 0.0;
  for (var split = 0u; split < splits; split++) {
    let kb0 = split * per;
    let kb1 = min(nb, kb0 + per);
    var partial_gate = 0.0;
    var partial_up = 0.0;
    for (var kb_base = kb0; kb_base < kb1; kb_base += 2u) {
      for (var half = 0u; half < 2u; half++) {
        let kb = kb_base + half;
        let at = half * 8u + load_col * 2u;
        var g0 = vec4<f32>(0.0);
        var g1 = vec4<f32>(0.0);
        var u0 = vec4<f32>(0.0);
        var u1 = vec4<f32>(0.0);
        if (load_output < N && kb < kb1) {
          let address = (load_output * K + kb * 32u) / 16u + load_col / 2u;
          let packed_gate = G[address];
          let packed_up = UP[address];
          let gate_words = select(packed_gate.xy, packed_gate.zw, (load_col & 1u) == 1u);
          let up_words = select(packed_up.xy, packed_up.zw, (load_col & 1u) == 1u);
          let gate_scale = GS[load_output * nb + kb];
          let up_scale = US[load_output * nb + kb];
          g0 = sx(gate_words.x) * gate_scale;
          g1 = sx(gate_words.y) * gate_scale;
          u0 = sx(up_words.x) * up_scale;
          u1 = sx(up_words.y) * up_scale;
        }
        let swizzle = load_row & 7u;
        gate_weights[load_row * 16u + (at ^ swizzle)] = vec4<f16>(g0);
        gate_weights[load_row * 16u + ((at + 1u) ^ swizzle)] = vec4<f16>(g1);
        up_weights[load_row * 16u + (at ^ swizzle)] = vec4<f16>(u0);
        up_weights[load_row * 16u + ((at + 1u) ^ swizzle)] = vec4<f16>(u1);
        if (li < 4u) {
          var x00 = vec4<f32>(0.0);
          var x01 = vec4<f32>(0.0);
          var x10 = vec4<f32>(0.0);
          var x11 = vec4<f32>(0.0);
          if (kb < kb1) {
            let base = (kb * 32u) / 4u + load_col * 2u;
            x00 = X[base];
            x01 = X[base + 1u];
            x10 = X[K / 4u + base];
            x11 = X[K / 4u + base + 1u];
          }
          inputs[at] = vec4<f16>(x00);
          inputs[at + 1u] = vec4<f16>(x01);
          inputs[16u + at] = vec4<f16>(x10);
          inputs[16u + at + 1u] = vec4<f16>(x11);
        }
      }
      workgroupBarrier();
      if (li < 32u) {
        let count = min(16u, (kb1 - kb_base) * 8u);
        for (var kq = 0u; kq < count; kq++) {
          let input = vec4<f32>(inputs[row * 16u + kq]);
          let gate_weight = vec4<f32>(gate_weights[output_col * 16u + (kq ^ (output_col & 7u))]);
          let up_weight = vec4<f32>(up_weights[output_col * 16u + (kq ^ (output_col & 7u))]);
          partial_gate += dot(input, gate_weight);
          partial_up += dot(input, up_weight);
        }
      }
      workgroupBarrier();
    }
    if (splits == 1u) {
      total_gate = partial_gate;
      total_up = partial_up;
    } else {
      total_gate += partial_gate;
      total_up += partial_up;
    }
  }
  let col = wg.x * 16u + output_col;
  if (li < 32u && col < N) {
    if (splits == 1u) { total_gate += 0.0; total_up += 0.0; }
    Y[row * N + col] = gelu(total_gate) * total_up;
  }
}

//#else
//#if SUBGROUPS
fn ordered_vector_sum32(prefix: vec4<f32>, subgroupLane: u32) -> vec4<f32> {
  var aggregate = prefix;
  for (var step = 16u; step > 0u; step >>= 1u) {
    let other = subgroupShuffleDown(aggregate, step);
    if (subgroupLane < step) { aggregate += other; }
  }
  return aggregate;
}
//#endif
// Each vec4 component retains one original lane64 dot and reduction sequence.
var<workgroup> partials: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
//#if SUBGROUPS
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(subgroup_invocation_id) subgroupLane: u32) {
//#else
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
//#endif
  if (g.T != 2u || wg.y != 0u) { return; }
  let N = p.N;
  let K = p.K;
  let lane = li & 63u;
  let col = wg.x * 4u + li / 64u;
  var values = vec4<f32>(0.0);
  if (col < N) {
    for (var part = lane; part < K / 4u; part += 64u) {
      let address = col * (K / 16u) + part / 4u;
      let gate_pack = G[address];
      let up_pack = UP[address];
      let gate_word = gate_pack[part & 3u];
      let up_word = up_pack[part & 3u];
      let gate_weight = sx(gate_word) * GS[col * (K / 32u) + part / 8u];
      let up_weight = sx(up_word) * US[col * (K / 32u) + part / 8u];
      let input0 = X[part];
      let input1 = X[K / 4u + part];
      values += vec4<f32>(dot(input0, gate_weight), dot(input0, up_weight), dot(input1, gate_weight), dot(input1, up_weight));
    }
  }
  partials[li] = values;
  workgroupBarrier();
//#if SUBGROUPS
  var prefix = partials[li];
  if (lane < 32u) { prefix += partials[li + 32u]; }
  let reduced = ordered_vector_sum32(prefix, subgroupLane);
//#else
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lane < stride) { partials[li] += partials[li + stride]; }
    workgroupBarrier();
  }
//#endif
  if (lane == 0u && col < N) {
//#if SUBGROUPS
    let result = reduced;
//#else
    let result = partials[li];
//#endif
    Y[col] = gelu(result.x) * result.y;
    Y[N + col] = gelu(result.z) * result.w;
  }
}

//#endif
