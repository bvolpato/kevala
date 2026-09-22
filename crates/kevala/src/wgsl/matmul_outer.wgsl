// Y[T, N] = X[T, K] . W[N, K]^T (+ bias), W int8 in u32 words with one f32 scale per 32 weights.
//
// This is an experimental alternative to matmul.wgsl. It keeps the same 16 x 16
// workgroup and 16 x {{ROWS}} by 64 x {{GROUPS}} tile, but gives each thread four
// adjacent output columns. Each four-value input vector is then accumulated with
// four vector FMAs against a transposed weight tile. The accumulator remains f32;
// only the two workgroup tiles use {{TILE}}.
//
// The first shared array is used as a short-lived weight staging area. For one
// 64-column group, it holds W in its normal [column][k/4] layout while the second
// array receives the transposed [k][column/4] layout. With two groups, the extra
// half of xs keeps X resident while each 64-column group is transposed in turn.
// This avoids component writes by multiple invocations, which are not reliable on
// all WebGPU implementations.
//#if F16
enable f16;
//#endif
//#include common

struct P { N: u32, K: u32, mode: u32, bias: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> S: array<f32>;
@group(0) @binding(5) var<storage, read> B: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
@group(0) @binding(7) var<storage, read_write> PART: array<f32>;

// For one 64-column group, 64 * 32 f16/f32 weights occupy 512 vec4 values.
// GROUPS=2 uses the second half of xs for the raw W staging area while its first
// half retains X. The compiled GROUPS=1 variant uses all of xs for staging, then
// overwrites it with X after the transpose.
var<workgroup> xs: array<vec4<{{TILE}}>, {{WS_LEN}}>;
// One transposed 64-column group: [k scalar][four adjacent output columns].
var<workgroup> ws: array<vec4<{{TILE}}>, {{WS_LEN}}>;

// Four int8 weights as f32, exactly and without int-to-float conversions (Marlin's trick): the
// xor makes each byte b + 128, and a byte in the low mantissa bits of 2^23 reads as 2^23 + b + 128
fn sx(w: u32) -> vec4<f32> {
  let u = w ^ 0x80808080u;
  let bytes = vec4<u32>(u, u >> 8u, u >> 16u, u >> 24u) & vec4<u32>(0xFFu);
  return bitcast<vec4<f32>>(bytes | vec4<u32>(0x4B000000u)) - vec4<f32>(8388736.0);
}

//#include splits

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let T = g.T;
//#if SHAPE
  let N = {{N}}u;
  let K = {{K}}u;
//#else
  let N = p.N;
  let K = p.K;
//#endif
  let nb = K / 32u;
  let m0 = wg.y * {{BM}}u;
  let n0 = wg.x * {{BN}}u;
  let splits = mm_splits(T, N, K);
  let per = (nb + splits - 1u) / splits;
  let kb0 = wg.z * per;
  let kb1 = min(nb, kb0 + per);

  // One vec4 accumulator owns four adjacent output columns. The explicit vector
  // read before component selection is intentional: nested array/vector indexing
  // has produced lost components in NVIDIA/Naga before.
  var acc: array<vec4<f32>, {{ACC_LEN}}>;
  for (var i = 0u; i < {{ROWS}}u; i++) {
    for (var gg = 0u; gg < {{GROUPS}}u; gg++) {
      acc[i * {{GROUPS}}u + gg] = vec4<f32>(0.0);
    }
  }

  let lr = li / 4u;
  let lq = li % 4u;
  // GROUPS=1: raw W occupies xs[0..512). GROUPS=2: xs[0..512) is X and
  // xs[512..1024) is the raw W staging area for the current output group.
  let rawBase = ({{GROUPS}}u - 1u) * 512u;

  for (var kb = kb0; kb < kb1; kb++) {
    // Process one 64-column output group at a time. This lets GROUPS=2 retain
    // the X tile while reusing the same transposed W storage for both groups.
    for (var gg = 0u; gg < {{GROUPS}}u; gg++) {
      let wrow = n0 + 64u * gg + lr;
      let rawRow = rawBase + lr * 8u;

      // Load W in its packed row-major form. Even lq lanes load one u32 vector
      // and expand all four halves, so each global weight vector is read once.
      // The four complete vec4 stores have one owner each and therefore avoid
      // sub-vector workgroup races.
      if ((lq & 1u) == 0u) {
        let dst = rawRow + (lq / 2u) * 4u;
        if (wrow < N) {
          let v = W[(wrow * K + kb * 32u) / 16u + lq / 2u];
          let scale = S[wrow * nb + kb];
          xs[dst] = vec4<{{TILE}}>(sx(v.x) * scale);
          xs[dst + 1u] = vec4<{{TILE}}>(sx(v.y) * scale);
          xs[dst + 2u] = vec4<{{TILE}}>(sx(v.z) * scale);
          xs[dst + 3u] = vec4<{{TILE}}>(sx(v.w) * scale);
        } else {
          xs[dst] = vec4<{{TILE}}>(0.0);
          xs[dst + 1u] = vec4<{{TILE}}>(0.0);
          xs[dst + 2u] = vec4<{{TILE}}>(0.0);
          xs[dst + 3u] = vec4<{{TILE}}>(0.0);
        }
      }
      workgroupBarrier();

      // Transpose 64 columns x 32 values into 32 k rows x 16 vec4 columns.
      // li%16 selects the four-column vector. Each invocation owns two vectors
      // for the current 64-column group, covering all 512 destinations. The
      // GROUPS=2 variant repeats this transpose for its second group.
      let colGroup = li % 16u;
      let chunk = li / 16u;
      for (var q = 0u; q < 2u; q++) {
        let k = chunk + q * 16u;
        let kv = k / 4u;
        let lane = k & 3u;
        let r0 = vec4<f32>(xs[rawBase + (colGroup * 4u + 0u) * 8u + kv]);
        let r1 = vec4<f32>(xs[rawBase + (colGroup * 4u + 1u) * 8u + kv]);
        let r2 = vec4<f32>(xs[rawBase + (colGroup * 4u + 2u) * 8u + kv]);
        let r3 = vec4<f32>(xs[rawBase + (colGroup * 4u + 3u) * 8u + kv]);
        ws[k * 16u + colGroup] = vec4<{{TILE}}>(vec4<f32>(r0[lane], r1[lane], r2[lane], r3[lane]));
      }
      workgroupBarrier();

      // Load the X tile after the GROUPS=1 staging area has been consumed. For
      // GROUPS=2 it uses xs[0..512), leaves the raw W area untouched, and is
      // retained for the second output group.
      if (gg == 0u || {{GROUPS}}u == 1u) {
        let xrow = m0 + lr;
        var xv0 = vec4<f32>(0.0);
        var xv1 = vec4<f32>(0.0);
        if (lr < {{BM}}u && xrow < T) {
          let base = (xrow * K + kb * 32u) / 4u + lq * 2u;
          xv0 = X[base];
          xv1 = X[base + 1u];
        }
        if (lr < {{BM}}u) {
          xs[lr * 8u + lq * 2u] = vec4<{{TILE}}>(xv0);
          xs[lr * 8u + lq * 2u + 1u] = vec4<{{TILE}}>(xv1);
        }
      }
      workgroupBarrier();

      // Four scalar products from each input vec4 form four vector FMAs over
      // the adjacent output columns owned by this invocation.
      for (var i = 0u; i < {{ROWS}}u; i++) {
        let row = lid.y + 16u * i;
        let a = vec4<f32>(xs[row * 8u + 0u]);
        let b0 = vec4<f32>(ws[0u * 16u + lid.x]);
        let b1 = vec4<f32>(ws[1u * 16u + lid.x]);
        let b2 = vec4<f32>(ws[2u * 16u + lid.x]);
        let b3 = vec4<f32>(ws[3u * 16u + lid.x]);
        let slot = i * {{GROUPS}}u + gg;
        acc[slot] = fma(b0, vec4<f32>(a[0]), acc[slot]);
        acc[slot] = fma(b1, vec4<f32>(a[1]), acc[slot]);
        acc[slot] = fma(b2, vec4<f32>(a[2]), acc[slot]);
        acc[slot] = fma(b3, vec4<f32>(a[3]), acc[slot]);
        for (var kq = 1u; kq < 8u; kq++) {
          let av = vec4<f32>(xs[row * 8u + kq]);
          let o = kq * 4u;
          let c0 = vec4<f32>(ws[(o + 0u) * 16u + lid.x]);
          let c1 = vec4<f32>(ws[(o + 1u) * 16u + lid.x]);
          let c2 = vec4<f32>(ws[(o + 2u) * 16u + lid.x]);
          let c3 = vec4<f32>(ws[(o + 3u) * 16u + lid.x]);
          acc[slot] = fma(c0, vec4<f32>(av[0]), acc[slot]);
          acc[slot] = fma(c1, vec4<f32>(av[1]), acc[slot]);
          acc[slot] = fma(c2, vec4<f32>(av[2]), acc[slot]);
          acc[slot] = fma(c3, vec4<f32>(av[3]), acc[slot]);
        }
      }
      workgroupBarrier();
    }
  }

  // Each thread writes its four adjacent columns. Keep the vector read separate
  // from component selection for the same NVIDIA/Naga indexing issue above.
  for (var gg = 0u; gg < {{GROUPS}}u; gg++) {
    let valuesBase = gg;
    for (var j = 0u; j < 4u; j++) {
      let col = n0 + 64u * gg + lid.x * 4u + j;
      if (col >= N) { break; }
      var bias = 0.0;
      if (p.bias == 1u) { bias = B[col]; }
      for (var i = 0u; i < {{ROWS}}u; i++) {
        let row = m0 + lid.y + 16u * i;
        if (row >= T) { break; }
        let o = row * N + col;
        let values = acc[i * {{GROUPS}}u + valuesBase];
        let v0 = values[j];
        if (splits > 1u) {
          PART[wg.z * T * N + o] = v0;
          continue;
        }
        var v = v0 + bias;
        if (p.mode == 1u) { v += Y[o]; } else if (p.mode == 2u) { v = max(v, 0.0); }
        Y[o] = v;
      }
    }
  }
}
