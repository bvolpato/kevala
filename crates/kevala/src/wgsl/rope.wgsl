// Rotary embedding on q and k, in place in the qkv buffer (rotate_half convention).
//#include common

struct P { heads: u32, stride: u32, table: u32, _a: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read_write> QKV: array<f32>;
@group(0) @binding(3) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> CS: array<vec2<f32>>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) i: u32) {
  let t = wg.x;
  let h = wg.y; // 0..2*heads: q heads then k heads
  if (t >= g.T) { return; }
  let pos = tok[t].z;
  let cs = CS[p.table * 512u * 32u + pos * 32u + i];
  let base = t * p.stride + h * 64u;
  let x1 = QKV[base + i];
  let x2 = QKV[base + i + 32u];
  QKV[base + i] = x1 * cs.x - x2 * cs.y;
  QKV[base + i + 32u] = x2 * cs.x + x1 * cs.y;
}
