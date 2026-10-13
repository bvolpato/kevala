struct P { rows: u32, width: u32, positions: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> ids: array<vec2<i32>>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;

fn weight(at: u32) -> f32 {
//#if WEIGHT_BF16
  let word = W[at / 2u];
  let bits = select(word << 16u, word & 0xffff0000u, (at & 1u) == 1u);
  return bitcast<f32>(bits);
//#endif
//#if WEIGHT_F32
  return bitcast<f32>(W[at]);
//#endif
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let at = g.x;
  if (at >= p.rows * p.width) { return; }
  let row = at / p.width;
  let col = at % p.width;
  let xy = ids[row];
  if (xy.x < 0 || xy.y < 0) { return; }
  X[at] += weight(u32(xy.x) * p.width + col) + weight((p.positions + u32(xy.y)) * p.width + col);
}
