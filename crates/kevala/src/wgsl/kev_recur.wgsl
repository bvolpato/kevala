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
var<workgroup> qs: array<array<f32, 128>, 2>;
var<workgroup> ks: array<array<f32, 128>, 2>;
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
  var col: array<f32, 128>;
  if (sp.z != 0xffffffffu) {
    let base = (sp.z * 16u + h) * 16384u + j;
    for (var i = 0u; i < 128u; i++) { col[i] = STATE[base + i * 128u]; }
  } else {
    for (var i = 0u; i < 128u; i++) { col[i] = 0.0; }
  }
  var v = 0.0;
  var decay = 0.0;
  var beta = 0.0;
  if (sp.y > 0u) {
    let t = sp.x;
    qs[0][j] = C[t * 6144u + h * 128u + j];
    ks[0][j] = C[t * 6144u + 2048u + h * 128u + j];
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
      qs[nxt][j] = C[tn * 6144u + h * 128u + j];
      ks[nxt][j] = C[tn * 6144u + 2048u + h * 128u + j];
      nv = C[tn * 6144u + 4096u + h * 128u + j];
      nd = AB[tn * 32u + h];
      nb = AB[tn * 32u + 16u + h];
    }
    var kv = 0.0;
    for (var i = 0u; i < 128u; i++) { kv += col[i] * ks[cur][i]; }
    let delta = (v - decay * kv) * beta;
    var o = 0.0;
    for (var i = 0u; i < 128u; i++) {
      let x = decay * col[i] + ks[cur][i] * delta;
      col[i] = x;
      o += x * qs[cur][i];
    }
    CORE[t * 2048u + h * 128u + j] = o;
    v = nv;
    decay = nd;
    beta = nb;
    workgroupBarrier();
  }
  if (g.stage == 1u) {
    let base = (sp.w * 16u + h) * 16384u + j;
    for (var i = 0u; i < 128u; i++) { STATE[base + i * 128u] = col[i]; }
  }
}
