// Gemma attention in query tiles, for the f32 Firefox path. A workgroup covers
// eight consecutive queries of one head. Each query owns 32 lanes, so the
// query dimensions stay in registers while the four keys in a tile are loaded
// once for all eight rows. Key and value tiles are f32: this path is a
// correctness-first companion to gemma4_attention, not a reduced-precision
// replacement.
//#include common

struct P { heads: u32, kv_heads: u32, head_dim: u32, window: u32, causal: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> Q: array<f32>;
@group(0) @binding(3) var<storage, read> K: array<f32>;
@group(0) @binding(4) var<storage, read> V: array<f32>;
@group(0) @binding(5) var<storage, read_write> O: array<f32>;

// Four keys x 128 vec4s is enough for Gemma's 512-wide global head. K and V
// together therefore use 16 KiB of workgroup storage at the maximum width.
const KEY_COUNT = 4u;
const MAX_VECS = 128u;
var<workgroup> ktile: array<vec4<f32>, 512>;
var<workgroup> vtile: array<vec4<f32>, 512>;

// One vec4 score contribution per lane and one reduced score vector per query.
// The 32 lanes belonging to a query reduce independently of the other seven.
var<workgroup> score_part: array<vec4<f32>, 256>;
var<workgroup> scores: array<vec4<f32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let block = wg.x * 8u;
  let head = wg.y;
  if (block >= g.T || head >= p.heads) { return; }

  let H = p.head_dim;
  let qrow = li / 32u;
  let lane = li % 32u;
  let kv_head = head / (p.heads / p.kv_heads);
  let nq = min(8u, g.T - block);
  let qindex = block + qrow;
  let qvalid = qrow < nq;
  let qbase = qindex * p.heads * H + head * H;

  // Each lane owns four consecutive dimensions, plus the same four-vector at
  // offsets 128, 256, and 384 when the head is wide enough.
  var q0 = vec4<f32>(0.0);
  var q1 = vec4<f32>(0.0);
  var q2 = vec4<f32>(0.0);
  var q3 = vec4<f32>(0.0);
  let d0 = lane * 4u;
  if (qvalid && d0 + 3u < H) { q0 = vec4<f32>(Q[qbase + d0], Q[qbase + d0 + 1u], Q[qbase + d0 + 2u], Q[qbase + d0 + 3u]); }
  if (qvalid && d0 + 128u + 3u < H) { q1 = vec4<f32>(Q[qbase + d0 + 128u], Q[qbase + d0 + 129u], Q[qbase + d0 + 130u], Q[qbase + d0 + 131u]); }
  if (qvalid && d0 + 256u + 3u < H) { q2 = vec4<f32>(Q[qbase + d0 + 256u], Q[qbase + d0 + 257u], Q[qbase + d0 + 258u], Q[qbase + d0 + 259u]); }
  if (qvalid && d0 + 384u + 3u < H) { q3 = vec4<f32>(Q[qbase + d0 + 384u], Q[qbase + d0 + 385u], Q[qbase + d0 + 386u], Q[qbase + d0 + 387u]); }

  // All eight rows participate in the same key range. Per-row masks below
  // retain causal and sliding-window semantics for rows with different ends.
  let tile_start = select(0u, block + 1u - min(block + 1u, p.window), p.window > 0u);
  let last = block + nq - 1u;
  let tile_end = select(g.T, min(g.T, last + 1u), p.causal != 0u);
  let row_start = select(0u, qindex + 1u - min(qindex + 1u, p.window), p.window > 0u);
  let row_end = select(g.T, qindex + 1u, p.causal != 0u);

  var m = -3.402823466e+38;
  var l = 0.0;
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  for (var j0 = tile_start; j0 < tile_end; j0 += KEY_COUNT) {
    // 256 lanes cooperatively stage four K and V rows. At H=512 each array
    // uses 8 KiB, and all rows are available to all eight query groups.
    for (var e = li; e < KEY_COUNT * MAX_VECS; e += 256u) {
      let key = e / MAX_VECS;
      let col = e % MAX_VECS;
      let j = j0 + key;
      let d = col * 4u;
      let valid = j < tile_end && d + 3u < H;
      var kk = vec4<f32>(0.0);
      var vv = vec4<f32>(0.0);
      if (valid) {
        let base = j * p.kv_heads * H + kv_head * H + d;
        kk = vec4<f32>(K[base], K[base + 1u], K[base + 2u], K[base + 3u]);
        vv = vec4<f32>(V[base], V[base + 1u], V[base + 2u], V[base + 3u]);
      }
      ktile[e] = kk;
      vtile[e] = vv;
    }
    workgroupBarrier();

    // Each lane computes its four key dot-product pieces. The five barrier
    // reduction steps below are confined logically to this query's 32 lanes.
    let k0 = lane;
    let k1 = MAX_VECS + lane;
    let k2 = 2u * MAX_VECS + lane;
    let k3 = 3u * MAX_VECS + lane;
    var part = vec4<f32>(
      dot(q0, ktile[k0]) + dot(q1, ktile[k0 + 32u]) + dot(q2, ktile[k0 + 64u]) + dot(q3, ktile[k0 + 96u]),
      dot(q0, ktile[k1]) + dot(q1, ktile[k1 + 32u]) + dot(q2, ktile[k1 + 64u]) + dot(q3, ktile[k1 + 96u]),
      dot(q0, ktile[k2]) + dot(q1, ktile[k2 + 32u]) + dot(q2, ktile[k2 + 64u]) + dot(q3, ktile[k2 + 96u]),
      dot(q0, ktile[k3]) + dot(q1, ktile[k3 + 32u]) + dot(q2, ktile[k3 + 64u]) + dot(q3, ktile[k3 + 96u])
    );
    score_part[li] = part;
    workgroupBarrier();
    let row_base = qrow * 32u;
    for (var stride = 16u; stride > 0u; stride >>= 1u) {
      if (lane < stride) { score_part[row_base + lane] += score_part[row_base + lane + stride]; }
      workgroupBarrier();
    }
    if (lane == 0u) { scores[qrow] = score_part[row_base]; }
    workgroupBarrier();

    let s = scores[qrow];
    let j = j0;
    let v0 = qvalid && j >= row_start && j < row_end;
    let v1 = qvalid && j + 1u >= row_start && j + 1u < row_end;
    let v2 = qvalid && j + 2u >= row_start && j + 2u < row_end;
    let v3 = qvalid && j + 3u >= row_start && j + 3u < row_end;
    let s0 = select(-3.402823466e+38, s.x, v0);
    let s1 = select(-3.402823466e+38, s.y, v1);
    let s2 = select(-3.402823466e+38, s.z, v2);
    let s3 = select(-3.402823466e+38, s.w, v3);
    let top = max(m, max(max(s0, s1), max(s2, s3)));
    let correction = select(0.0, exp(m - top), m > -3.0e38);
    let w0 = select(0.0, exp(s0 - top), v0);
    let w1 = select(0.0, exp(s1 - top), v1);
    let w2 = select(0.0, exp(s2 - top), v2);
    let w3 = select(0.0, exp(s3 - top), v3);
    l = l * correction + w0 + w1 + w2 + w3;
    acc0 = acc0 * correction;
    acc1 = acc1 * correction;
    acc2 = acc2 * correction;
    acc3 = acc3 * correction;
    acc0 += w0 * vtile[lane] + w1 * vtile[MAX_VECS + lane] + w2 * vtile[2u * MAX_VECS + lane] + w3 * vtile[3u * MAX_VECS + lane];
    acc1 += w0 * vtile[32u + lane] + w1 * vtile[MAX_VECS + 32u + lane] + w2 * vtile[2u * MAX_VECS + 32u + lane] + w3 * vtile[3u * MAX_VECS + 32u + lane];
    acc2 += w0 * vtile[64u + lane] + w1 * vtile[MAX_VECS + 64u + lane] + w2 * vtile[2u * MAX_VECS + 64u + lane] + w3 * vtile[3u * MAX_VECS + 64u + lane];
    acc3 += w0 * vtile[96u + lane] + w1 * vtile[MAX_VECS + 96u + lane] + w2 * vtile[2u * MAX_VECS + 96u + lane] + w3 * vtile[3u * MAX_VECS + 96u + lane];
    m = top;
    workgroupBarrier();
  }

  if (!qvalid) { return; }
  let obase = qbase;
  if (d0 + 3u < H) {
    O[obase + d0] = acc0.x / l;
    O[obase + d0 + 1u] = acc0.y / l;
    O[obase + d0 + 2u] = acc0.z / l;
    O[obase + d0 + 3u] = acc0.w / l;
  }
  if (d0 + 128u + 3u < H) {
    O[obase + d0 + 128u] = acc1.x / l;
    O[obase + d0 + 129u] = acc1.y / l;
    O[obase + d0 + 130u] = acc1.z / l;
    O[obase + d0 + 131u] = acc1.w / l;
  }
  if (d0 + 256u + 3u < H) {
    O[obase + d0 + 256u] = acc2.x / l;
    O[obase + d0 + 257u] = acc2.y / l;
    O[obase + d0 + 258u] = acc2.z / l;
    O[obase + d0 + 259u] = acc2.w / l;
  }
  if (d0 + 384u + 3u < H) {
    O[obase + d0 + 384u] = acc3.x / l;
    O[obase + d0 + 385u] = acc3.y / l;
    O[obase + d0 + 386u] = acc3.z / l;
    O[obase + d0 + 387u] = acc3.w / l;
  }
}
