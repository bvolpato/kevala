// Quantized Gemma word embedding lookup. The converter stores rows as signed int8
// values with one f32 scale for every 32 values. Keeping the lookup on the device avoids
// copying the 262k-row embedding through Wasm for every request.
//#include common

struct P { width: u32, blocks: u32, scale: f32, _a: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read> W: array<u32>;
@group(0) @binding(4) var<storage, read> S: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;

fn q8(word: u32, lane: u32) -> f32 {
  let q = (word >> (lane * 8u)) & 0xffu;
  return select(f32(q), f32(q) - 256.0, q >= 128u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let t = wg.x;
  if (t >= g.T) { return; }
  let id = ids[t];
  let row = id * p.width;
  for (var j = lane; j < p.width; j += 256u) {
    let packed = W[(row + j) >> 2u];
    let q = q8(packed, (row + j) & 3u);
    Y[t * p.width + j] = q * S[id * p.blocks + j / 32u] * p.scale;
  }
}
