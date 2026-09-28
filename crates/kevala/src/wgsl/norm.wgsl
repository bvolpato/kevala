// Row layer norm: one workgroup per token, D = 1024 features.
//#if SUBGROUPS
//#include ordered_sum
//#endif
//#include common

struct P { D: u32, bias: u32, typed: u32, eps: f32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> Wt: array<f32>;
@group(0) @binding(4) var<storage, read> Bs: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;
@group(0) @binding(6) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read> TE: array<f32>;
var<workgroup> red: array<f32, 256>;

fn reduce(v: f32, l: u32
//#if SUBGROUPS
  , subgroupLane: u32
//#endif
) -> f32 {
  red[l] = v;
  workgroupBarrier();
//#if SUBGROUPS
  for (var s = 128u; s >= 32u; s >>= 1u) {
//#else
  for (var s = 128u; s > 0u; s >>= 1u) {
//#endif
    if (l < s) { red[l] += red[l + s]; }
    workgroupBarrier();
  }
//#if SUBGROUPS
  let r = ordered_sum32(red[subgroupLane], subgroupLane);
//#else
  let r = red[0];
//#endif
  workgroupBarrier();
  return r;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) l: u32
//#if SUBGROUPS
  , @builtin(subgroup_invocation_id) subgroupLane: u32
//#endif
) {
  let t = wg.x;
  if (t >= g.T) { return; }
  let D = p.D;
  let base = t * D;
  var v: array<f32, 4>;
  var s = 0.0;
  for (var i = 0u; i < 4u; i++) { v[i] = X[base + l + i * 256u]; s += v[i]; }
  let mean = reduce(s, l
//#if SUBGROUPS
    , subgroupLane
//#endif
  ) / f32(D);
  var q = 0.0;
  for (var i = 0u; i < 4u; i++) { let d = v[i] - mean; q += d * d; }
  let inv = inverseSqrt(reduce(q, l
//#if SUBGROUPS
    , subgroupLane
//#endif
  ) / f32(D) + p.eps);
  let qt = tok[t].w;
  for (var i = 0u; i < 4u; i++) {
    let c = l + i * 256u;
    var o = (v[i] - mean) * inv * Wt[c];
    if (p.bias == 1u) { o += Bs[c]; }
    if (p.typed == 1u) { o += TE[qt * D + c]; }
    Y[base + c] = o;
  }
}
