// Clef tile-0 F32 projection. The first staged request validates every output bit.
struct Params { T: u32, N: u32, K: u32, output_offset: u32, tail_count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> W: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read> TAIL_ROWS: array<u32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;

fn mac(acc: vec4<f32>, x: vec4<f32>, w: vec4<f32>) -> vec4<f32> {
  return fma(x, w, acc);
}
fn hsum(acc: vec4<f32>) -> f32 {
  return (acc.x + acc.z) + (acc.y + acc.w);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let output = id.x;
  if (output >= p.tail_count * p.N) { return; }
  let row = TAIL_ROWS[output / p.N];
  let col = output % p.N;
  let stride = p.K / 4u;
  let xb = row * stride;
  let wb = col * stride;
  var low = vec4<f32>(0.0);
  var high = vec4<f32>(0.0);
    // Production dot4: two accumulators over alternating four-value chunks.
    var c = 0u;
    while (c + 8u <= p.K) {
      let block = c / 4u;
      low = mac(low, X[xb + block], W[wb + block]);
      high = mac(high, X[xb + block + 1u], W[wb + block + 1u]);
      c += 8u;
    }
    if (c < p.K) {
      low = mac(low, X[xb + c / 4u], W[wb + c / 4u]);
    }
    low = low + high;
  Y[p.output_offset + row * p.N + col] = hsum(low) + B[col];
}
