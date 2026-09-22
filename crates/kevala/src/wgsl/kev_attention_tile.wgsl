// Kev's causal GQA attention in query tiles, for GPUs with subgroups of 8 or more lanes and f16
// (FlashInfer's grouping of the query heads that share a key/value head). A workgroup takes up to
// 8 consecutive tokens of one segment and one key/value head: 32 query rows (8 tokens x the 4
// query heads of that key/value head), 8 lanes per row. A lane keeps 32 of its row's 256 query and
// output dimensions in registers (vec4s lane, lane + 8, ...), so the keys and values the rows
// share are staged once per 16 keys in workgroup memory, as f16, instead of once per row. A score
// is finished across its row's 8 lanes with three shuffles, so every lane has the softmax state of
// its row; the 32 lanes of a subgroup are one token, so the causal mask is uniform in a subgroup.
// Keys come from the parent's cache rows first, then from this segment's own projections.
enable f16;
enable subgroups;

//#include kev_common

@group(0) @binding(1) var<storage, read> PROJ: array<vec4<f32>>; // [T][ATTN_WIDTH / 4]: q, gate per head; k; v
@group(0) @binding(2) var<storage, read> KV: array<vec4<f32>>;   // cache rows [ATTN_KV / 4]: k then v
@group(0) @binding(3) var<storage, read> blocks: array<vec4<u32>>; // [0].x: count; then (segment, first position, tokens, 0)
@group(0) @binding(4) var<storage, read> segs: array<Seg>;
@group(0) @binding(5) var<storage, read_write> OUT: array<vec4<f32>>; // [T][ATTN_Q / 4]
const_assert HEADS == 4u * KV_HEADS; // a workgroup's rows are the 4 query heads of one key/value head
const PROW = ATTN_WIDTH / 4u; // vec4s per projection row
const KROW = ATTN_KV / 4u;    // vec4s per cache row
const KOFF = ATTN_Q / 2u;     // first key vec4 of a projection row (after the q and gate halves)
const VOFF = KOFF + ATTN_K / 4u;
var<workgroup> ks: array<vec4<f16>, 1024>; // [16 keys][64]
var<workgroup> vs: array<vec4<f16>, 1024>; // [16 keys][64]
var<workgroup> info: vec4<u32>;
var<workgroup> info2: vec4<u32>;

// a score's 8 partial sums, one per lane of its row
fn row_sum(x: f32) -> f32 {
  var s = x + subgroupShuffleXor(x, 1u);
  s += subgroupShuffleXor(s, 2u);
  return s + subgroupShuffleXor(s, 4u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let kvh = wg.y;
  if (li == 0u) {
    info = vec4<u32>(0u);
    if (g.T > 0u && wg.x < blocks[0].x) {
      let b = blocks[1u + wg.x];
      let sg = segs[b.x];
      var plen = 0u;
      if (sg.parent != 0xffffffffu) { plen = sg.plen; }
      info = vec4<u32>(sg.start, b.y, b.z, 1u);
      info2 = vec4<u32>(plen, sg.pstart, 0u, 0u);
    }
  }
  let bi = workgroupUniformLoad(&info);
  if (bi.w == 0u) { return; }
  let ci = workgroupUniformLoad(&info2);
  let start = bi.x;
  let r0 = bi.y;
  let ntok = bi.z;
  let plen = ci.x;
  let pstart = ci.y;

  let tl = li / 32u;        // token within the block
  let hq = (li / 8u) % 4u;  // query head within the key/value head's group
  let lane = li % 8u;       // vec4s lane + 8 i of the row
  let h = kvh * 4u + hq;
  let live = tl < ntok;
  let t = start + r0 + min(tl, ntok - 1u);
  let nk = plen + r0 + tl + 1u;         // keys this row sees
  let kmax = plen + r0 + ntok;          // keys any row of the block sees
  var q: array<vec4<f32>, 8>;
  for (var i = 0u; i < 8u; i++) { q[i] = PROJ[t * PROW + h * 128u + lane + 8u * i] * 0.0625; }
  var o: array<vec4<f32>, 8>;
  var m = -3.0e38;
  var l = 0.0;

  for (var j0 = 0u; j0 < kmax; j0 += 16u) {
    workgroupBarrier();
    for (var e = li; e < 1024u; e += 256u) {
      let j = j0 + e / 64u;
      let dv = e % 64u;
      var kk = vec4<f32>(0.0);
      var vv = vec4<f32>(0.0);
      if (j < plen) {
        let row = (pstart + j) * KROW + kvh * 64u + dv;
        kk = KV[row];
        vv = KV[row + ATTN_K / 4u];
      } else if (j < kmax) {
        let row = (start + j - plen) * PROW + kvh * 64u + dv;
        kk = PROJ[row + KOFF];
        vv = PROJ[row + VOFF];
      }
      ks[e] = vec4<f16>(kk);
      vs[e] = vec4<f16>(vv);
    }
    workgroupBarrier();

    var s: array<f32, 16>;
    var top = m;
    for (var k = 0u; k < 16u; k++) {
      var part = 0.0;
      for (var i = 0u; i < 8u; i++) { part += dot(q[i], vec4<f32>(ks[k * 64u + lane + 8u * i])); }
      let full = row_sum(part);
      s[k] = select(-3.0e38, full, j0 + k < nk);
      top = max(top, s[k]);
    }
    let corr = exp(m - top);
    l *= corr;
    for (var i = 0u; i < 8u; i++) { o[i] *= corr; }
    for (var k = 0u; k < 16u; k++) {
      let pk = select(0.0, exp(s[k] - top), j0 + k < nk);
      l += pk;
      for (var i = 0u; i < 8u; i++) { o[i] += pk * vec4<f32>(vs[k * 64u + lane + 8u * i]); }
    }
    m = top;
  }

  if (live) {
    for (var i = 0u; i < 8u; i++) {
      let gate = PROJ[t * PROW + h * 128u + 64u + lane + 8u * i];
      OUT[t * (ATTN_Q / 4u) + h * 64u + lane + 8u * i] = o[i] / l / (vec4<f32>(1.0) + exp(-gate));
    }
  }
}
