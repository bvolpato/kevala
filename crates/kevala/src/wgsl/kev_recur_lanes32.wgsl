// The same recurrence with thirty-two lanes per value column, each holding 4 of 128 key
// values. The private recurrence state uses 4 named scalar values, avoiding dynamic private-array
// indexing. A workgroup covers 8 columns (z picks one of 16 tiles in a head). With subgroups of
// at least 32 lanes, the 32 partial dot products combine with xor shuffles; without, through
// workgroup memory with the same balanced reduction tree. The state/cache layout is unchanged
// from kev_recur_lanes.
//#if SUBGROUPS
enable subgroups;
//#endif

//#include kev_common

@group(0) @binding(1) var<storage, read> C: array<f32>;
@group(0) @binding(2) var<storage, read> AB: array<f32>;
@group(0) @binding(3) var<storage, read> segs: array<Seg>;
@group(0) @binding(4) var<storage, read_write> STATE: array<f32>;
@group(0) @binding(5) var<storage, read_write> CORE: array<f32>;
var<workgroup> qs: array<array<f32, 128>, 2>;
var<workgroup> ks: array<array<f32, 128>, 2>;
var<workgroup> info: vec4<u32>;
//#if SUBGROUPS
//#else
var<workgroup> kvs: array<f32, 256>; // partial k.S of each thread
var<workgroup> os: array<f32, 256>;  // partial outputs of the token before
//#endif

// The token's q and k (key head kh) into buffer b: threads 0..127 copy q, 128..255 copy k.
fn stage(t: u32, kh: u32, b: u32, li: u32) {
  if (li < 128u) {
    qs[b][li] = C[t * LIN_DIM + kh * 128u + li];
  } else {
    ks[b][li - 128u] = C[t * LIN_DIM + LIN_QK + kh * 128u + li - 128u];
  }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let s = wg.x;
  let h = wg.y;
//#if KEV_GROUPED
  let kh = h / (LIN_HEADS / LIN_KEY_HEADS); // value heads share key heads in groups
//#else
  let kh = h;
//#endif
  if (s >= g.S) { return; }
  let column = wg.z * 8u + li / 32u;
  let lane = li % 32u;
  let k0 = lane * 4u;
  if (li == 0u) {
    let sg = segs[s];
    info = vec4<u32>(sg.start, sg.len, sg.parent, sg.dst);
  }
  let sp = workgroupUniformLoad(&info);
  var c0 = 0.0;
  var c1 = 0.0;
  var c2 = 0.0;
  var c3 = 0.0;
  if (sp.z != 0xffffffffu) {
    let base = (sp.z * LIN_HEADS + h) * 16384u + column;
    c0 = STATE[base + (k0 + 0u) * 128u];
    c1 = STATE[base + (k0 + 1u) * 128u];
    c2 = STATE[base + (k0 + 2u) * 128u];
    c3 = STATE[base + (k0 + 3u) * 128u];
  }
  var v = 0.0;
  var decay = 0.0;
  var beta = 0.0;
  if (sp.y > 0u) {
    stage(sp.x, kh, 0u, li);
    v = C[sp.x * LIN_DIM + 2u * LIN_QK + h * 128u + column];
    decay = AB[sp.x * (2u * LIN_HEADS) + h];
    beta = AB[sp.x * (2u * LIN_HEADS) + LIN_HEADS + h];
  }
  workgroupBarrier();
  for (var r = 0u; r < sp.y; r++) {
    let t = sp.x + r;
    let cur = r & 1u;
    // Stage the next token in the other buffer; nobody reads it until after the barrier.
    var nv = 0.0;
    var nd = 0.0;
    var nb = 0.0;
    if (r + 1u < sp.y) {
      stage(t + 1u, kh, cur ^ 1u, li);
      nv = C[(t + 1u) * LIN_DIM + 2u * LIN_QK + h * 128u + column];
      nd = AB[(t + 1u) * (2u * LIN_HEADS) + h];
      nb = AB[(t + 1u) * (2u * LIN_HEADS) + LIN_HEADS + h];
    }
    var kv = 0.0;
    kv += c0 * ks[cur][k0 + 0u];
    kv += c1 * ks[cur][k0 + 1u];
    kv += c2 * ks[cur][k0 + 2u];
    kv += c3 * ks[cur][k0 + 3u];
//#if SUBGROUPS
    kv += subgroupShuffleXor(kv, 1u);
    kv += subgroupShuffleXor(kv, 2u);
    kv += subgroupShuffleXor(kv, 4u);
    kv += subgroupShuffleXor(kv, 8u);
    kv += subgroupShuffleXor(kv, 16u);
//#else
    // The previous token's output, summed by its column's first lane before anyone overwrites it.
    if (r > 0u && lane == 0u) {
      let q32 = li;
      let p0 = (os[q32] + os[q32 + 1u]) + (os[q32 + 2u] + os[q32 + 3u]);
      let p1 = (os[q32 + 4u] + os[q32 + 5u]) + (os[q32 + 6u] + os[q32 + 7u]);
      let p2 = (os[q32 + 8u] + os[q32 + 9u]) + (os[q32 + 10u] + os[q32 + 11u]);
      let p3 = (os[q32 + 12u] + os[q32 + 13u]) + (os[q32 + 14u] + os[q32 + 15u]);
      let p4 = (os[q32 + 16u] + os[q32 + 17u]) + (os[q32 + 18u] + os[q32 + 19u]);
      let p5 = (os[q32 + 20u] + os[q32 + 21u]) + (os[q32 + 22u] + os[q32 + 23u]);
      let p6 = (os[q32 + 24u] + os[q32 + 25u]) + (os[q32 + 26u] + os[q32 + 27u]);
      let p7 = (os[q32 + 28u] + os[q32 + 29u]) + (os[q32 + 30u] + os[q32 + 31u]);
      let q0 = p0 + p1;
      let q1 = p2 + p3;
      let q2 = p4 + p5;
      let q3 = p6 + p7;
      let left = q0 + q1;
      let right = q2 + q3;
      CORE[(t - 1u) * LIN_OUT + h * 128u + column] = left + right;
    }
    kvs[li] = kv;
    workgroupBarrier();
    let q32 = li & ~31u;
    let p0 = (kvs[q32] + kvs[q32 + 1u]) + (kvs[q32 + 2u] + kvs[q32 + 3u]);
    let p1 = (kvs[q32 + 4u] + kvs[q32 + 5u]) + (kvs[q32 + 6u] + kvs[q32 + 7u]);
    let p2 = (kvs[q32 + 8u] + kvs[q32 + 9u]) + (kvs[q32 + 10u] + kvs[q32 + 11u]);
    let p3 = (kvs[q32 + 12u] + kvs[q32 + 13u]) + (kvs[q32 + 14u] + kvs[q32 + 15u]);
    let p4 = (kvs[q32 + 16u] + kvs[q32 + 17u]) + (kvs[q32 + 18u] + kvs[q32 + 19u]);
    let p5 = (kvs[q32 + 20u] + kvs[q32 + 21u]) + (kvs[q32 + 22u] + kvs[q32 + 23u]);
    let p6 = (kvs[q32 + 24u] + kvs[q32 + 25u]) + (kvs[q32 + 26u] + kvs[q32 + 27u]);
    let p7 = (kvs[q32 + 28u] + kvs[q32 + 29u]) + (kvs[q32 + 30u] + kvs[q32 + 31u]);
    let q0 = p0 + p1;
    let q1 = p2 + p3;
    let q2 = p4 + p5;
    let q3 = p6 + p7;
    let left = q0 + q1;
    let right = q2 + q3;
    kv = left + right;
//#endif
    let delta = (v - decay * kv) * beta;
    var o = 0.0;
    let x0 = decay * c0 + ks[cur][k0 + 0u] * delta;
    c0 = x0;
    o += x0 * qs[cur][k0 + 0u];
    let x1 = decay * c1 + ks[cur][k0 + 1u] * delta;
    c1 = x1;
    o += x1 * qs[cur][k0 + 1u];
    let x2 = decay * c2 + ks[cur][k0 + 2u] * delta;
    c2 = x2;
    o += x2 * qs[cur][k0 + 2u];
    let x3 = decay * c3 + ks[cur][k0 + 3u] * delta;
    c3 = x3;
    o += x3 * qs[cur][k0 + 3u];
//#if SUBGROUPS
    o += subgroupShuffleXor(o, 1u);
    o += subgroupShuffleXor(o, 2u);
    o += subgroupShuffleXor(o, 4u);
    o += subgroupShuffleXor(o, 8u);
    o += subgroupShuffleXor(o, 16u);
    if (lane == 0u) { CORE[t * LIN_OUT + h * 128u + column] = o; }
//#else
    os[li] = o;
//#endif
    v = nv;
    decay = nd;
    beta = nb;
    workgroupBarrier();
  }
//#if SUBGROUPS
//#else
  if (sp.y > 0u && lane == 0u) {
    let q32 = li;
    let p0 = (os[q32] + os[q32 + 1u]) + (os[q32 + 2u] + os[q32 + 3u]);
    let p1 = (os[q32 + 4u] + os[q32 + 5u]) + (os[q32 + 6u] + os[q32 + 7u]);
    let p2 = (os[q32 + 8u] + os[q32 + 9u]) + (os[q32 + 10u] + os[q32 + 11u]);
    let p3 = (os[q32 + 12u] + os[q32 + 13u]) + (os[q32 + 14u] + os[q32 + 15u]);
    let p4 = (os[q32 + 16u] + os[q32 + 17u]) + (os[q32 + 18u] + os[q32 + 19u]);
    let p5 = (os[q32 + 20u] + os[q32 + 21u]) + (os[q32 + 22u] + os[q32 + 23u]);
    let p6 = (os[q32 + 24u] + os[q32 + 25u]) + (os[q32 + 26u] + os[q32 + 27u]);
    let p7 = (os[q32 + 28u] + os[q32 + 29u]) + (os[q32 + 30u] + os[q32 + 31u]);
    let q0 = p0 + p1;
    let q1 = p2 + p3;
    let q2 = p4 + p5;
    let q3 = p6 + p7;
    let left = q0 + q1;
    let right = q2 + q3;
    CORE[(sp.x + sp.y - 1u) * LIN_OUT + h * 128u + column] = left + right;
  }
//#endif
  if (g.stage == 1u) {
    let base = (sp.w * LIN_HEADS + h) * 16384u + column;
    STATE[base + (k0 + 0u) * 128u] = c0;
    STATE[base + (k0 + 1u) * 128u] = c1;
    STATE[base + (k0 + 2u) * 128u] = c2;
    STATE[base + (k0 + 3u) * 128u] = c3;
  }
}
