// Row layer norm: one workgroup per token, D = 1024 features.
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

fn reduce(v: f32, l: u32) -> f32 {
  red[l] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (l < s) { red[l] += red[l + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let t = wg.x;
  if (t >= g.T) { return; }
  let D = p.D;
  let base = t * D;
  var v: array<f32, 4>;
  var s = 0.0;
  for (var i = 0u; i < 4u; i++) { v[i] = X[base + l + i * 256u]; s += v[i]; }
  let mean = reduce(s, l) / f32(D);
  var q = 0.0;
  for (var i = 0u; i < 4u; i++) { let d = v[i] - mean; q += d * d; }
  let inv = inverseSqrt(reduce(q, l) / f32(D) + p.eps);
  let qt = tok[t].w;
  for (var i = 0u; i < 4u; i++) {
    let c = l + i * 256u;
    var o = (v[i] - mean) * inv * Wt[c];
    if (p.bias == 1u) { o += Bs[c]; }
    if (p.typed == 1u) { o += TE[qt * D + c]; }
    Y[base + c] = o;
  }
}
