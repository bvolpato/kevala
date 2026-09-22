// The same recurrence with four lanes per value column, each holding a quarter of the key
// dimension (32 of 128). Partial dot products are combined with subgroup shuffles, so each
// token's serial chain is 32 multiply-adds instead of 128, and a thread keeps 32 state values in
// registers instead of 128. A workgroup covers half of a head's 128 columns (z picks which).
// Needs the `subgroups` feature and lanes numbered consecutively within a subgroup.
enable subgroups;

//#include kev_common

@group(0) @binding(1) var<storage, read> C: array<f32>;
@group(0) @binding(2) var<storage, read> AB: array<f32>;
@group(0) @binding(3) var<storage, read> segs: array<Seg>;
@group(0) @binding(4) var<storage, read_write> STATE: array<f32>;
@group(0) @binding(5) var<storage, read_write> CORE: array<f32>;
var<workgroup> qs: array<array<f32, 128>, 2>;
var<workgroup> ks: array<array<f32, 128>, 2>;
var<workgroup> info: vec4<u32>;

// the token's q and k into buffer b: threads 0..127 copy q, 128..255 copy k
fn stage(t: u32, h: u32, b: u32, li: u32) {
  if (li < 128u) {
    qs[b][li] = C[t * 6144u + h * 128u + li];
  } else {
    ks[b][li - 128u] = C[t * 6144u + 2048u + h * 128u + li - 128u];
  }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let s = wg.x;
  let h = wg.y;
  if (s >= g.S) { return; }
  let column = wg.z * 64u + li / 4u;
  let k0 = (li % 4u) * 32u;
  if (li == 0u) {
    let sg = segs[s];
    info = vec4<u32>(sg.start, sg.len, sg.parent, sg.dst);
  }
  let sp = workgroupUniformLoad(&info);
  var col: array<f32, 32>;
  if (sp.z != 0xffffffffu) {
    let base = (sp.z * 16u + h) * 16384u + column;
    for (var i = 0u; i < 32u; i++) { col[i] = STATE[base + (k0 + i) * 128u]; }
  } else {
    for (var i = 0u; i < 32u; i++) { col[i] = 0.0; }
  }
  var v = 0.0;
  var decay = 0.0;
  var beta = 0.0;
  if (sp.y > 0u) {
    stage(sp.x, h, 0u, li);
    v = C[sp.x * 6144u + 4096u + h * 128u + column];
    decay = AB[sp.x * 32u + h];
    beta = AB[sp.x * 32u + 16u + h];
  }
  workgroupBarrier();
  for (var r = 0u; r < sp.y; r++) {
    let t = sp.x + r;
    let cur = r & 1u;
    // stage the next token in the other buffer; nobody reads it until after the barrier
    var nv = 0.0;
    var nd = 0.0;
    var nb = 0.0;
    if (r + 1u < sp.y) {
      stage(t + 1u, h, cur ^ 1u, li);
      nv = C[(t + 1u) * 6144u + 4096u + h * 128u + column];
      nd = AB[(t + 1u) * 32u + h];
      nb = AB[(t + 1u) * 32u + 16u + h];
    }
    var kv = 0.0;
    for (var i = 0u; i < 32u; i++) { kv += col[i] * ks[cur][k0 + i]; }
    kv += subgroupShuffleXor(kv, 1u);
    kv += subgroupShuffleXor(kv, 2u);
    let delta = (v - decay * kv) * beta;
    var o = 0.0;
    for (var i = 0u; i < 32u; i++) {
      let x = decay * col[i] + ks[cur][k0 + i] * delta;
      col[i] = x;
      o += x * qs[cur][k0 + i];
    }
    o += subgroupShuffleXor(o, 1u);
    o += subgroupShuffleXor(o, 2u);
    if (k0 == 0u) { CORE[t * 2048u + h * 128u + column] = o; }
    v = nv;
    decay = nd;
    beta = nb;
    workgroupBarrier();
  }
  if (g.stage == 1u) {
    let base = (sp.w * 16u + h) * 16384u + column;
    for (var i = 0u; i < 32u; i++) { STATE[base + (k0 + i) * 128u] = col[i]; }
  }
}
