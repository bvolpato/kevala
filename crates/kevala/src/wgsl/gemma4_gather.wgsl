// Gathers selected normalized hidden rows for CPU readout. Gemma hidden sizes are
// larger than the fixed 256-wide Laya gather, so each lane walks the row.
//#include common

struct P { D: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> rows: array<u32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let row = wg.x;
  if (row >= g.R) { return; }
  let src = rows[row] * p.D;
  let dst = row * p.D;
  for (var d = lane; d < p.D; d += 256u) { O[dst + d] = X[src + d]; }
}
