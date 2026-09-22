// Causal depthwise conv + SiLU and q/k l2 normalization, continuing a parent's tail.
//#include kev_common

@group(0) @binding(1) var<storage, read> PROJ: array<f32>;
@group(0) @binding(2) var<storage, read> CW: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> segs: array<Seg>;
@group(0) @binding(5) var<storage, read> TAIL: array<f32>;
@group(0) @binding(6) var<storage, read_write> C: array<f32>;
const CD = LIN_DIM;
const PW = LIN_WIDTH;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let t = wg.x;
  if (t >= g.T) { return; }
  let c = wg.y * 256u + l;
  let sg = segs[tok[t].x];
  let r = i32(tok[t].y);
  let w = CW[c];
  var s = 0.0;
  for (var i = 0; i < 4; i++) {
    let rr = r - 3 + i;
    var x = 0.0;
    if (rr >= 0) {
      x = PROJ[(sg.start + u32(rr)) * PW + c];
    } else if (sg.parent != 0xffffffffu) {
      x = TAIL[(sg.parent * 3u + u32(3 + rr)) * CD + c];
    }
    s += w[i] * x;
  }
  var v = s / (1.0 + exp(-s));
  if (wg.y < LIN_KEY_HEADS) {
    // Each workgroup contains two 128-wide q/k heads.
    let j = l % 128u;
    red[l] = v * v;
    workgroupBarrier();
    for (var k = 64u; k > 0u; k >>= 1u) {
      if (j < k) { red[l] += red[l + k]; }
      workgroupBarrier();
    }
    var inv = inverseSqrt(red[l - j] + 1e-6);
    if (wg.y < LIN_KEY_HEADS / 2u) { inv *= 0.08838834764831845; }
    v *= inv;
  }
  C[t * CD + c] = v;
}
