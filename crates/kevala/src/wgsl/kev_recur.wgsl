// the gated delta rule, one workgroup per (segment, head), one thread per value column. The
// loop is sequential in time, so per-token latency is the cost: the next token's inputs load
// into registers while the current one computes, and q/k ping-pong between two workgroup
// buffers so each token needs one barrier instead of two.
//#include kev_common

@group(0) @binding(1) var<storage, read> C: array<f32>;
@group(0) @binding(2) var<storage, read> AB: array<f32>;
@group(0) @binding(3) var<storage, read> segs: array<Seg>;
@group(0) @binding(4) var<storage, read_write> STATE: array<f32>;
@group(0) @binding(5) var<storage, read_write> CORE: array<f32>;
var<workgroup> qs: array<array<vec4<f32>, 32>, 2>;
var<workgroup> ks: array<array<vec4<f32>, 32>, 2>;
var<workgroup> info: vec4<u32>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) j: u32) {
  let s = wg.x;
  let h = wg.y;
  if (s >= g.S) { return; }
  if (j == 0u) {
    let sg = segs[s];
    info = vec4<u32>(sg.start, sg.len, sg.parent, sg.dst);
  }
  let sp = workgroupUniformLoad(&info);
  var col: array<vec4<f32>, 32>;
  if (sp.z != 0xffffffffu) {
    let base = (sp.z * 16u + h) * 16384u + j;
    for (var i = 0u; i < 32u; i++) {
      let row = base + i * 4u * 128u;
      col[i] = vec4<f32>(
        STATE[row],
        STATE[row + 128u],
        STATE[row + 2u * 128u],
        STATE[row + 3u * 128u]
      );
    }
  } else {
    for (var i = 0u; i < 32u; i++) { col[i] = vec4<f32>(0.0); }
  }
  var v = 0.0;
  var decay = 0.0;
  var beta = 0.0;
  if (sp.y > 0u) {
    let t = sp.x;
    if (j < 32u) {
      let q = t * 6144u + h * 128u + j * 4u;
      qs[0][j] = vec4<f32>(C[q], C[q + 1u], C[q + 2u], C[q + 3u]);
      let k = t * 6144u + 2048u + h * 128u + j * 4u;
      ks[0][j] = vec4<f32>(C[k], C[k + 1u], C[k + 2u], C[k + 3u]);
    }
    v = C[t * 6144u + 4096u + h * 128u + j];
    decay = AB[t * 32u + h];
    beta = AB[t * 32u + 16u + h];
  }
  workgroupBarrier();
  for (var r = 0u; r < sp.y; r++) {
    let t = sp.x + r;
    let cur = r & 1u;
    let nxt = cur ^ 1u;
    // stage the next token in the other buffer; nobody reads it until after the barrier
    var nv = 0.0;
    var nd = 0.0;
    var nb = 0.0;
    if (r + 1u < sp.y) {
      let tn = t + 1u;
      if (j < 32u) {
        let q = tn * 6144u + h * 128u + j * 4u;
        qs[nxt][j] = vec4<f32>(C[q], C[q + 1u], C[q + 2u], C[q + 3u]);
        let k = tn * 6144u + 2048u + h * 128u + j * 4u;
        ks[nxt][j] = vec4<f32>(C[k], C[k + 1u], C[k + 2u], C[k + 3u]);
      }
      nv = C[tn * 6144u + 4096u + h * 128u + j];
      nd = AB[tn * 32u + h];
      nb = AB[tn * 32u + 16u + h];
    }
    var kv4 = vec4<f32>(0.0);
    for (var i = 0u; i < 32u; i++) { kv4 += col[i] * ks[cur][i]; }
    let kv = kv4.x + kv4.y + kv4.z + kv4.w;
    let delta = (v - decay * kv) * beta;
    var o4 = vec4<f32>(0.0);
    for (var i = 0u; i < 32u; i++) {
      let x = decay * col[i] + ks[cur][i] * delta;
      col[i] = x;
      o4 += x * qs[cur][i];
    }
    let o = o4.x + o4.y + o4.z + o4.w;
    CORE[t * 2048u + h * 128u + j] = o;
    v = nv;
    decay = nd;
    beta = nb;
    workgroupBarrier();
  }
  if (g.stage == 1u) {
    let base = (sp.w * 16u + h) * 16384u + j;
    for (var i = 0u; i < 32u; i++) {
      let row = base + i * 4u * 128u;
      let value = col[i];
      STATE[row] = value.x;
      STATE[row + 128u] = value.y;
      STATE[row + 2u * 128u] = value.z;
      STATE[row + 3u * 128u] = value.w;
    }
  }
}
