// stage 1: remember the last three pre-conv inputs of every state (a short extension takes the
// rest from its parent's tail)
//#include kev_common

@group(0) @binding(1) var<storage, read> PROJ: array<f32>;
@group(0) @binding(2) var<storage, read> segs: array<Seg>;
@group(0) @binding(3) var<storage, read_write> TAIL: array<f32>;
const CD = LIN_DIM;
const PW = LIN_WIDTH;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let s = wg.x;
  if (s >= g.S) { return; }
  let i = wg.y / (CD / 256u);
  let c = (wg.y % (CD / 256u)) * 256u + l;
  let sg = segs[s];
  let rr = i32(sg.len) - 3 + i32(i);
  var x = 0.0;
  if (rr >= 0) {
    x = PROJ[(sg.start + u32(rr)) * PW + c];
  } else if (sg.parent != 0xffffffffu) {
    x = TAIL[(sg.parent * 3u + u32(3 + rr)) * CD + c];
  }
  // dst is never the parent's slot, so reading one while writing the other cannot race
  TAIL[(sg.dst * 3u + i) * CD + c] = x;
}
