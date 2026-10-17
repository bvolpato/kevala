// Q8/F32 paired prefix MLP: R2, WG16x8, physical BN32 and virtual BN64 splits.
// p.splitT preserves the original prefix partition; gate/up sums stay independent.
//#include common

struct P { N: u32, K: u32, splitT: u32, _pad: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> WG: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> SG: array<f32>;
@group(0) @binding(5) var<storage, read> WU: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> SU: array<f32>;
@group(0) @binding(7) var<storage, read_write> Y: array<f32>;
@group(0) @binding(8) var<storage, read_write> PART: array<f32>;

var<workgroup> xs: array<vec4<f32>, 256>;
var<workgroup> gs: array<vec4<f32>, 256>;
var<workgroup> us: array<vec4<f32>, 256>;

fn sx(w: u32) -> vec4<f32> {
  let bytes = vec4<i32>(bitcast<i32>(w << 24u), bitcast<i32>(w << 16u), bitcast<i32>(w << 8u), bitcast<i32>(w)) >> vec4<u32>(24u);
  return vec4<f32>(bytes);
}

fn gelu(x: f32) -> f32 {
  if (x >= 10.0) { return x; }
  if (x <= -10.0) { return 0.0; }
  let c = 0.7978845608028654;
  return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

fn accumulate(kq: u32, c: u32, y: u32,
  acc0: ptr<function, vec4<f32>>, acc1: ptr<function, vec4<f32>>,
  acc2: ptr<function, vec4<f32>>, acc3: ptr<function, vec4<f32>>,
) {
  let nx = c & 7u;
  let g0 = vec4<f32>(gs[c * 8u + (kq ^ nx)]);
  let g1 = vec4<f32>(gs[(c + 16u) * 8u + (kq ^ nx)]);
  let u0 = vec4<f32>(us[c * 8u + (kq ^ nx)]);
  let u1 = vec4<f32>(us[(c + 16u) * 8u + (kq ^ nx)]);
  let a0 = vec4<f32>(xs[y * 8u + kq]);
  *acc0 += vec4<f32>(dot(a0, g0), dot(a0, g1), dot(a0, u0), dot(a0, u1));
  let a1 = vec4<f32>(xs[(y + 8u) * 8u + kq]);
  *acc1 += vec4<f32>(dot(a1, g0), dot(a1, g1), dot(a1, u0), dot(a1, u1));
  let a2 = vec4<f32>(xs[(y + 16u) * 8u + kq]);
  *acc2 += vec4<f32>(dot(a2, g0), dot(a2, g1), dot(a2, u0), dot(a2, u1));
  let a3 = vec4<f32>(xs[(y + 24u) * 8u + kq]);
  *acc3 += vec4<f32>(dot(a3, g0), dot(a3, g1), dot(a3, u0), dot(a3, u1));
}

fn store_pair(value: vec2<f32>, row: u32, col: u32, T: u32, N: u32, splits: u32, z: u32) {
  if (row >= T || col >= N) { return; }
  let o = row * N + col;
  if (splits > 1u) {
    PART[z * T * N + o] = value.x;
    PART[(splits + z) * T * N + o] = value.y;
    return;
  }
  // The unfused zero-bias matmul also adds positive zero before its output store.
  let gate = value.x + 0.0;
  let up = value.y + 0.0;
  Y[o] = gelu(gate) * up;
}

// Split-K: narrow matmuls at short lengths launch too few tiles to fill a GPU (a 1024-wide output
// at 64 tokens is 16 workgroups), so K is split across up to 8 workgroups whose partial tiles a
// second pass sums. The host computes the same count (runtime `mm_splits` twin in gpu.js).
//#include splits

@compute @workgroup_size(16, 8)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let T = g.T;
  let N = p.N;
  let K = p.K;
  let nb = K / 32u;
  let m0 = wg.y * 32u;
  let n0 = wg.x * 32u;
  // splitT=0 uses the active row count; splitT=1 retains independent suffix partitions.
  let splitT = select(p.splitT, T, p.splitT == 0u);
  let splits = mm_splits(splitT, N, K);
  let per = (nb + splits - 1u) / splits;
  let kb0 = wg.z * per;
  let kb1 = min(nb, kb0 + per);
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);
  let lr = li / 4u;
  let lq = li % 4u;
  let xrow = m0 + lr;
  let wrow = n0 + lr;
  let sw = lr & 7u;
  for (var kb = kb0; kb < kb1; kb++) {
    var xv0 = vec4<f32>(0.0);
    var xv1 = vec4<f32>(0.0);
    if (lr < 32u && xrow < T) {
      let base = (xrow * K + kb * 32u) / 4u + lq * 2u;
      xv0 = X[base];
      xv1 = X[base + 1u];
    }
    var gv0 = vec4<f32>(0.0);
    var gv1 = vec4<f32>(0.0);
    var uv0 = vec4<f32>(0.0);
    var uv1 = vec4<f32>(0.0);
    if (wrow < N) {
      let at = (wrow * K + kb * 32u) / 16u + lq / 2u;
      let gate = WG[at];
      let up = WU[at];
      let scaleG = SG[wrow * nb + kb];
      let scaleU = SU[wrow * nb + kb];
      let gateBytes = select(gate.xy, gate.zw, (lq & 1u) == 1u);
      let upBytes = select(up.xy, up.zw, (lq & 1u) == 1u);
      gv0 = sx(gateBytes.x) * scaleG;
      gv1 = sx(gateBytes.y) * scaleG;
      uv0 = sx(upBytes.x) * scaleU;
      uv1 = sx(upBytes.y) * scaleU;
    }
    if (lr < 32u) {
      xs[lr * 8u + lq * 2u] = vec4<f32>(xv0);
      xs[lr * 8u + lq * 2u + 1u] = vec4<f32>(xv1);
    }
    gs[lr * 8u + ((lq * 2u) ^ sw)] = vec4<f32>(gv0);
    gs[lr * 8u + ((lq * 2u + 1u) ^ sw)] = vec4<f32>(gv1);
    us[lr * 8u + ((lq * 2u) ^ sw)] = vec4<f32>(uv0);
    us[lr * 8u + ((lq * 2u + 1u) ^ sw)] = vec4<f32>(uv1);
    workgroupBarrier();
    accumulate(0u, lid.x, lid.y, &acc0, &acc1,
      &acc2, &acc3,
    );
    accumulate(1u, lid.x, lid.y, &acc0, &acc1,
      &acc2, &acc3,
    );
    accumulate(2u, lid.x, lid.y, &acc0, &acc1,
      &acc2, &acc3,
    );
    accumulate(3u, lid.x, lid.y, &acc0, &acc1,
      &acc2, &acc3,
    );
    accumulate(4u, lid.x, lid.y, &acc0, &acc1,
      &acc2, &acc3,
    );
    accumulate(5u, lid.x, lid.y, &acc0, &acc1,
      &acc2, &acc3,
    );
    accumulate(6u, lid.x, lid.y, &acc0, &acc1,
      &acc2, &acc3,
    );
    accumulate(7u, lid.x, lid.y, &acc0, &acc1,
      &acc2, &acc3,
    );
    workgroupBarrier();
  }
  let row = m0 + lid.y;
  store_pair(acc0.xz, row + 0u, n0 + lid.x, T, N, splits, wg.z);
  store_pair(acc0.yw, row + 0u, n0 + lid.x + 16u, T, N, splits, wg.z);
  store_pair(acc1.xz, row + 8u, n0 + lid.x, T, N, splits, wg.z);
  store_pair(acc1.yw, row + 8u, n0 + lid.x + 16u, T, N, splits, wg.z);
  store_pair(acc2.xz, row + 16u, n0 + lid.x, T, N, splits, wg.z);
  store_pair(acc2.yw, row + 16u, n0 + lid.x + 16u, T, N, splits, wg.z);
  store_pair(acc3.xz, row + 24u, n0 + lid.x, T, N, splits, wg.z);
  store_pair(acc3.yw, row + 24u, n0 + lid.x + 16u, T, N, splits, wg.z);
}
