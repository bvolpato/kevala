// stage 1: keep every state token's keys and values for the branches (and later requests)
//#include kev_common

@group(0) @binding(1) var<storage, read> PROJ: array<f32>;
@group(0) @binding(2) var<storage, read_write> KV: array<f32>;
@group(0) @binding(3) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> segs: array<Seg>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let t = wg.x;
  if (t >= g.T) { return; }
  let c = wg.y * 256u + l;
  let row = segs[tok[t].x].kvdst + tok[t].y;
  KV[row * 1024u + c] = PROJ[t * 5120u + 4096u + c];
}
