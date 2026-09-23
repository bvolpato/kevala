// The same recurrence with four lanes per value column, each holding a quarter of the key
// dimension (32 of 128), so each token's serial chain is 32 multiply-adds instead of 128. The
// private recurrence state uses 32 named scalar values, avoiding dynamic private-array indexing
// while preserving the original scalar load, reduction, update, output, and store order. A
// workgroup covers half of a head's 128 columns (z picks which). With subgroups (lanes numbered
// consecutively within one) the four partial dot products combine with shuffles; without, through
// workgroup memory: one more barrier per token for k.S, while a column's output sums after the
// barrier that ends its token.
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
  let column = wg.z * 64u + li / 4u;
  let k0 = (li % 4u) * 32u;
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
  var c16 = 0.0;
  var c17 = 0.0;
  var c18 = 0.0;
  var c19 = 0.0;
  var c20 = 0.0;
  var c21 = 0.0;
  var c22 = 0.0;
  var c23 = 0.0;
  var c24 = 0.0;
  var c25 = 0.0;
  var c26 = 0.0;
  var c27 = 0.0;
  var c28 = 0.0;
  var c29 = 0.0;
  var c30 = 0.0;
  var c31 = 0.0;
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
    c16 = STATE[base + (k0 + 16u) * 128u];
    c17 = STATE[base + (k0 + 17u) * 128u];
    c18 = STATE[base + (k0 + 18u) * 128u];
    c19 = STATE[base + (k0 + 19u) * 128u];
    c20 = STATE[base + (k0 + 20u) * 128u];
    c21 = STATE[base + (k0 + 21u) * 128u];
    c22 = STATE[base + (k0 + 22u) * 128u];
    c23 = STATE[base + (k0 + 23u) * 128u];
    c24 = STATE[base + (k0 + 24u) * 128u];
    c25 = STATE[base + (k0 + 25u) * 128u];
    c26 = STATE[base + (k0 + 26u) * 128u];
    c27 = STATE[base + (k0 + 27u) * 128u];
    c28 = STATE[base + (k0 + 28u) * 128u];
    c29 = STATE[base + (k0 + 29u) * 128u];
    c30 = STATE[base + (k0 + 30u) * 128u];
    c31 = STATE[base + (k0 + 31u) * 128u];
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
    c16 = 0.0;
    c17 = 0.0;
    c18 = 0.0;
    c19 = 0.0;
    c20 = 0.0;
    c21 = 0.0;
    c22 = 0.0;
    c23 = 0.0;
    c24 = 0.0;
    c25 = 0.0;
    c26 = 0.0;
    c27 = 0.0;
    c28 = 0.0;
    c29 = 0.0;
    c30 = 0.0;
    c31 = 0.0;
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
    kv += c16 * ks[cur][k0 + 16u];
    kv += c17 * ks[cur][k0 + 17u];
    kv += c18 * ks[cur][k0 + 18u];
    kv += c19 * ks[cur][k0 + 19u];
    kv += c20 * ks[cur][k0 + 20u];
    kv += c21 * ks[cur][k0 + 21u];
    kv += c22 * ks[cur][k0 + 22u];
    kv += c23 * ks[cur][k0 + 23u];
    kv += c24 * ks[cur][k0 + 24u];
    kv += c25 * ks[cur][k0 + 25u];
    kv += c26 * ks[cur][k0 + 26u];
    kv += c27 * ks[cur][k0 + 27u];
    kv += c28 * ks[cur][k0 + 28u];
    kv += c29 * ks[cur][k0 + 29u];
    kv += c30 * ks[cur][k0 + 30u];
    kv += c31 * ks[cur][k0 + 31u];
//#if SUBGROUPS
    kv += subgroupShuffleXor(kv, 1u);
    kv += subgroupShuffleXor(kv, 2u);
//#else
    // the previous token's output, summed by its column's first lane before anyone overwrites it
    if (r > 0u && k0 == 0u) { CORE[(t - 1u) * LIN_OUT + h * 128u + column] = (os[li] + os[li + 1u]) + (os[li + 2u] + os[li + 3u]); }
    kvs[li] = kv;
    workgroupBarrier();
    let q4 = li & ~3u;
    // paired as the shuffles pair them, so both variants give the same bits
    kv = (kvs[q4] + kvs[q4 + 1u]) + (kvs[q4 + 2u] + kvs[q4 + 3u]);
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
    let x16 = decay * c16 + ks[cur][k0 + 16u] * delta;
    c16 = x16;
    o += x16 * qs[cur][k0 + 16u];
    let x17 = decay * c17 + ks[cur][k0 + 17u] * delta;
    c17 = x17;
    o += x17 * qs[cur][k0 + 17u];
    let x18 = decay * c18 + ks[cur][k0 + 18u] * delta;
    c18 = x18;
    o += x18 * qs[cur][k0 + 18u];
    let x19 = decay * c19 + ks[cur][k0 + 19u] * delta;
    c19 = x19;
    o += x19 * qs[cur][k0 + 19u];
    let x20 = decay * c20 + ks[cur][k0 + 20u] * delta;
    c20 = x20;
    o += x20 * qs[cur][k0 + 20u];
    let x21 = decay * c21 + ks[cur][k0 + 21u] * delta;
    c21 = x21;
    o += x21 * qs[cur][k0 + 21u];
    let x22 = decay * c22 + ks[cur][k0 + 22u] * delta;
    c22 = x22;
    o += x22 * qs[cur][k0 + 22u];
    let x23 = decay * c23 + ks[cur][k0 + 23u] * delta;
    c23 = x23;
    o += x23 * qs[cur][k0 + 23u];
    let x24 = decay * c24 + ks[cur][k0 + 24u] * delta;
    c24 = x24;
    o += x24 * qs[cur][k0 + 24u];
    let x25 = decay * c25 + ks[cur][k0 + 25u] * delta;
    c25 = x25;
    o += x25 * qs[cur][k0 + 25u];
    let x26 = decay * c26 + ks[cur][k0 + 26u] * delta;
    c26 = x26;
    o += x26 * qs[cur][k0 + 26u];
    let x27 = decay * c27 + ks[cur][k0 + 27u] * delta;
    c27 = x27;
    o += x27 * qs[cur][k0 + 27u];
    let x28 = decay * c28 + ks[cur][k0 + 28u] * delta;
    c28 = x28;
    o += x28 * qs[cur][k0 + 28u];
    let x29 = decay * c29 + ks[cur][k0 + 29u] * delta;
    c29 = x29;
    o += x29 * qs[cur][k0 + 29u];
    let x30 = decay * c30 + ks[cur][k0 + 30u] * delta;
    c30 = x30;
    o += x30 * qs[cur][k0 + 30u];
    let x31 = decay * c31 + ks[cur][k0 + 31u] * delta;
    c31 = x31;
    o += x31 * qs[cur][k0 + 31u];
//#if SUBGROUPS
    o += subgroupShuffleXor(o, 1u);
    o += subgroupShuffleXor(o, 2u);
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
  if (sp.y > 0u && k0 == 0u) { CORE[(sp.x + sp.y - 1u) * LIN_OUT + h * 128u + column] = (os[li] + os[li + 1u]) + (os[li + 2u] + os[li + 3u]); }
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
    STATE[base + (k0 + 16u) * 128u] = c16;
    STATE[base + (k0 + 17u) * 128u] = c17;
    STATE[base + (k0 + 18u) * 128u] = c18;
    STATE[base + (k0 + 19u) * 128u] = c19;
    STATE[base + (k0 + 20u) * 128u] = c20;
    STATE[base + (k0 + 21u) * 128u] = c21;
    STATE[base + (k0 + 22u) * 128u] = c22;
    STATE[base + (k0 + 23u) * 128u] = c23;
    STATE[base + (k0 + 24u) * 128u] = c24;
    STATE[base + (k0 + 25u) * 128u] = c25;
    STATE[base + (k0 + 26u) * 128u] = c26;
    STATE[base + (k0 + 27u) * 128u] = c27;
    STATE[base + (k0 + 28u) * 128u] = c28;
    STATE[base + (k0 + 29u) * 128u] = c29;
    STATE[base + (k0 + 30u) * 128u] = c30;
    STATE[base + (k0 + 31u) * 128u] = c31;
  }
}
