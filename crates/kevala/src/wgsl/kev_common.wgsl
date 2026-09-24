// Kev trunk: the per-pass globals and the packed segment table.
struct Globals { T: u32, R: u32, S: u32, stage: u32 }
// a packed segment: tokens start..start+len continue carry slot parent (keys at KV rows
// pstart..pstart+plen) and, in stage 1, leave their carry in slot dst, own keys from KV row kvdst
struct Seg { start: u32, len: u32, parent: u32, pstart: u32, plen: u32, dst: u32, kvdst: u32, _b: u32 }
@group(0) @binding(0) var<uniform> g: Globals;
const HIDDEN = {{KEV_HIDDEN}}u;
const HEADS = {{KEV_HEADS}}u;
const KV_HEADS = {{KEV_KV_HEADS}}u;
const LIN_KEY_HEADS = {{KEV_LIN_KEY_HEADS}}u;
const LIN_HEADS = {{KEV_LIN_HEADS}}u;
const ROTARY = {{KEV_ROTARY}}u;
const LIN_QK = LIN_KEY_HEADS * 128u;
const LIN_OUT = LIN_HEADS * 128u;
const LIN_DIM = 2u * LIN_QK + LIN_OUT;
const LIN_WIDTH = LIN_DIM + LIN_OUT;
const ATTN_Q = HEADS * 256u;
const ATTN_K = KV_HEADS * 256u;
const ATTN_KV = 2u * ATTN_K;
const ATTN_WIDTH = 2u * ATTN_Q + ATTN_KV;

fn sigmoid(value: f32) -> f32 {
  let magnitude = exp(-abs(value));
  let denominator = 1.0 + magnitude;
  return select(magnitude / denominator, 1.0 / denominator, value >= 0.0);
}

fn sigmoid4(value: vec4<f32>) -> vec4<f32> {
  let magnitude = exp(-abs(value));
  let denominator = vec4<f32>(1.0) + magnitude;
  return select(magnitude / denominator, vec4<f32>(1.0) / denominator, value >= vec4<f32>(0.0));
}

fn silu(value: f32) -> f32 {
  if (value >= 20.0) { return value; }
  if (value <= -120.0) { return 0.0; }
  return value * sigmoid(value);
}

fn softplus(value: f32) -> f32 {
  if (value > 20.0) { return value; }
  let magnitude = exp(value);
  let shifted = 1.0 + magnitude;
  if (shifted == 1.0) { return magnitude; }
  return log(shifted) * (magnitude / (shifted - 1.0));
}
