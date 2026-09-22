// Transpose the current-token key projection for coalesced attention reads. Each workgroup
// handles a 16x16 tile, padding shared memory to avoid bank conflicts during the transpose.
//#include kev_common

@group(0) @binding(1) var<storage, read> PROJ: array<f32>;
@group(0) @binding(2) var<storage, read_write> KEYS: array<f32>;
var<workgroup> tile: array<array<f32, 17>, 16>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let lx = l & 15u;
  let ly = l >> 4u;
  let token = wg.x * 16u + ly;
  let dim = wg.y * 16u + lx;
  tile[ly][lx] = 0.0;
  if (token < g.T) {
    tile[ly][lx] = PROJ[token * ATTN_WIDTH + 2u * ATTN_Q + dim];
  }
  workgroupBarrier();
  let out_token = wg.x * 16u + lx;
  let out_dim = wg.y * 16u + ly;
  if (out_token < g.T) {
    KEYS[out_dim * g.T + out_token] = tile[lx][ly];
  }
}
