// Gather complete rows of Q8, BF16, or F32 weights into F32 activations.
struct P { rows: u32, width: u32, block: u32, first: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> S: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let row = wg.x;
  if (row >= p.rows) { return; }
  let src = ids[row] - p.first;
  for (var col = lane * 4u; col < p.width; col += 1024u) {
    let at = src * p.width + col;
    var value: vec4<f32>;
//#if WEIGHT_Q8
    let words = W[at / 16u];
    let word = words[(at % 16u) / 4u];
    let signed = vec4<i32>(bitcast<i32>(word << 24u), bitcast<i32>(word << 16u), bitcast<i32>(word << 8u), bitcast<i32>(word)) >> vec4<u32>(24u);
    value = vec4<f32>(signed) * S[at / p.block];
//#endif
//#if WEIGHT_BF16
    let words = W[at / 8u];
    let pair = select(words.xy, words.zw, (at % 8u) == 4u);
    value = bitcast<vec4<f32>>(vec4<u32>(pair.x << 16u, pair.x & 0xffff0000u, pair.y << 16u, pair.y & 0xffff0000u));
//#endif
//#if WEIGHT_F32
    value = bitcast<vec4<f32>>(W[at / 4u]);
//#endif
    O[(row * p.width + col) / 4u] = value;
  }
}
