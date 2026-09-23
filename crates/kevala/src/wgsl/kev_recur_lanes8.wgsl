// The same recurrence with eight lanes per value column, each holding 16 of 128 key values.
// The private recurrence state uses 16 named scalar values, avoiding dynamic private-array
// indexing. A workgroup covers one quarter of a head's 128 columns (z picks which). With
// subgroups, the eight partial dot products combine with xor shuffles; without, through workgroup
// memory. The state/cache layout is unchanged from kev_recur_lanes.
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

// the token's q and k (key head kh) into buffer b: threads 0..127 copy q, 128..255 copy k
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
  let column = wg.z * 32u + li / 8u;
  let k0 = (li % 8u) * 16u;
  if (li == 0u) {
    let sg = segs[s];
    info = vec4<u32>(sg.start, sg.len, sg.parent, sg.dst);
  }
  let sp = workgroupUniformLoad(&info);
  var c0 = 0.0;
  var c1 = 0.0;
  var c2 = 0.0;
  var c3 = 0.0;
  var c4 = 0.0;
  var c5 = 0.0;
  var c6 = 0.0;
  var c7 = 0.0;
  var c8 = 0.0;
  var c9 = 0.0;
  var c10 = 0.0;
  var c11 = 0.0;
  var c12 = 0.0;
  var c13 = 0.0;
  var c14 = 0.0;
  var c15 = 0.0;
  if (sp.z != 0xffffffffu) {
    let base = (sp.z * LIN_HEADS + h) * 16384u + column;
    c0 = STATE[base + (k0 + 0u) * 128u];
    c1 = STATE[base + (k0 + 1u) * 128u];
    c2 = STATE[base + (k0 + 2u) * 128u];
    c3 = STATE[base + (k0 + 3u) * 128u];
    c4 = STATE[base + (k0 + 4u) * 128u];
    c5 = STATE[base + (k0 + 5u) * 128u];
    c6 = STATE[base + (k0 + 6u) * 128u];
    c7 = STATE[base + (k0 + 7u) * 128u];
    c8 = STATE[base + (k0 + 8u) * 128u];
    c9 = STATE[base + (k0 + 9u) * 128u];
    c10 = STATE[base + (k0 + 10u) * 128u];
    c11 = STATE[base + (k0 + 11u) * 128u];
    c12 = STATE[base + (k0 + 12u) * 128u];
    c13 = STATE[base + (k0 + 13u) * 128u];
    c14 = STATE[base + (k0 + 14u) * 128u];
    c15 = STATE[base + (k0 + 15u) * 128u];
  } else {
    c0 = 0.0;
    c1 = 0.0;
    c2 = 0.0;
    c3 = 0.0;
    c4 = 0.0;
    c5 = 0.0;
    c6 = 0.0;
    c7 = 0.0;
    c8 = 0.0;
    c9 = 0.0;
    c10 = 0.0;
    c11 = 0.0;
    c12 = 0.0;
    c13 = 0.0;
    c14 = 0.0;
    c15 = 0.0;
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
    // stage the next token in the other buffer; nobody reads it until after the barrier
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
    kv += c4 * ks[cur][k0 + 4u];
    kv += c5 * ks[cur][k0 + 5u];
    kv += c6 * ks[cur][k0 + 6u];
    kv += c7 * ks[cur][k0 + 7u];
    kv += c8 * ks[cur][k0 + 8u];
    kv += c9 * ks[cur][k0 + 9u];
    kv += c10 * ks[cur][k0 + 10u];
    kv += c11 * ks[cur][k0 + 11u];
    kv += c12 * ks[cur][k0 + 12u];
    kv += c13 * ks[cur][k0 + 13u];
    kv += c14 * ks[cur][k0 + 14u];
    kv += c15 * ks[cur][k0 + 15u];
//#if SUBGROUPS
    kv += subgroupShuffleXor(kv, 1u);
    kv += subgroupShuffleXor(kv, 2u);
    kv += subgroupShuffleXor(kv, 4u);
//#else
    // the previous token's output, summed by its column's first lane before anyone overwrites it
    if (r > 0u && k0 == 0u) {
      let q8 = li;
      let left = (os[q8] + os[q8 + 1u]) + (os[q8 + 2u] + os[q8 + 3u]);
      let right = (os[q8 + 4u] + os[q8 + 5u]) + (os[q8 + 6u] + os[q8 + 7u]);
      CORE[(t - 1u) * LIN_OUT + h * 128u + column] = left + right;
    }
    kvs[li] = kv;
    workgroupBarrier();
    let q8 = li & ~7u;
    let left = (kvs[q8] + kvs[q8 + 1u]) + (kvs[q8 + 2u] + kvs[q8 + 3u]);
    let right = (kvs[q8 + 4u] + kvs[q8 + 5u]) + (kvs[q8 + 6u] + kvs[q8 + 7u]);
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
    let x4 = decay * c4 + ks[cur][k0 + 4u] * delta;
    c4 = x4;
    o += x4 * qs[cur][k0 + 4u];
    let x5 = decay * c5 + ks[cur][k0 + 5u] * delta;
    c5 = x5;
    o += x5 * qs[cur][k0 + 5u];
    let x6 = decay * c6 + ks[cur][k0 + 6u] * delta;
    c6 = x6;
    o += x6 * qs[cur][k0 + 6u];
    let x7 = decay * c7 + ks[cur][k0 + 7u] * delta;
    c7 = x7;
    o += x7 * qs[cur][k0 + 7u];
    let x8 = decay * c8 + ks[cur][k0 + 8u] * delta;
    c8 = x8;
    o += x8 * qs[cur][k0 + 8u];
    let x9 = decay * c9 + ks[cur][k0 + 9u] * delta;
    c9 = x9;
    o += x9 * qs[cur][k0 + 9u];
    let x10 = decay * c10 + ks[cur][k0 + 10u] * delta;
    c10 = x10;
    o += x10 * qs[cur][k0 + 10u];
    let x11 = decay * c11 + ks[cur][k0 + 11u] * delta;
    c11 = x11;
    o += x11 * qs[cur][k0 + 11u];
    let x12 = decay * c12 + ks[cur][k0 + 12u] * delta;
    c12 = x12;
    o += x12 * qs[cur][k0 + 12u];
    let x13 = decay * c13 + ks[cur][k0 + 13u] * delta;
    c13 = x13;
    o += x13 * qs[cur][k0 + 13u];
    let x14 = decay * c14 + ks[cur][k0 + 14u] * delta;
    c14 = x14;
    o += x14 * qs[cur][k0 + 14u];
    let x15 = decay * c15 + ks[cur][k0 + 15u] * delta;
    c15 = x15;
    o += x15 * qs[cur][k0 + 15u];
//#if SUBGROUPS
    o += subgroupShuffleXor(o, 1u);
    o += subgroupShuffleXor(o, 2u);
    o += subgroupShuffleXor(o, 4u);
    if (k0 == 0u) { CORE[t * LIN_OUT + h * 128u + column] = o; }
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
  if (sp.y > 0u && k0 == 0u) {
    let q8 = li;
    let left = (os[q8] + os[q8 + 1u]) + (os[q8 + 2u] + os[q8 + 3u]);
    let right = (os[q8 + 4u] + os[q8 + 5u]) + (os[q8 + 6u] + os[q8 + 7u]);
    CORE[(sp.x + sp.y - 1u) * LIN_OUT + h * 128u + column] = left + right;
  }
//#endif
  if (g.stage == 1u) {
    let base = (sp.w * LIN_HEADS + h) * 16384u + column;
    STATE[base + (k0 + 0u) * 128u] = c0;
    STATE[base + (k0 + 1u) * 128u] = c1;
    STATE[base + (k0 + 2u) * 128u] = c2;
    STATE[base + (k0 + 3u) * 128u] = c3;
    STATE[base + (k0 + 4u) * 128u] = c4;
    STATE[base + (k0 + 5u) * 128u] = c5;
    STATE[base + (k0 + 6u) * 128u] = c6;
    STATE[base + (k0 + 7u) * 128u] = c7;
    STATE[base + (k0 + 8u) * 128u] = c8;
    STATE[base + (k0 + 9u) * 128u] = c9;
    STATE[base + (k0 + 10u) * 128u] = c10;
    STATE[base + (k0 + 11u) * 128u] = c11;
    STATE[base + (k0 + 12u) * 128u] = c12;
    STATE[base + (k0 + 13u) * 128u] = c13;
    STATE[base + (k0 + 14u) * 128u] = c14;
    STATE[base + (k0 + 15u) * 128u] = c15;
  }
}
