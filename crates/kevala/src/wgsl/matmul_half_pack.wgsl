// Dequantize the int8 words once into f16 weights. The packed layout is
// [k block][64-row tile][row within tile][32 values], so one row's 32 values
// are contiguous and rows in the same tile are adjacent.
//
// PW stores eight vec4<f16> values per row and PS keeps the matching packed
// scales for bind-group compatibility. The consumer does not read PS: the
// scale has already been rounded into each f16 weight.
enable f16;

struct P { N: u32, K: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> S: array<f32>;
@group(0) @binding(3) var<storage, read_write> PW: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read_write> PS: array<f32>;

// four int8 weights as f32, exactly and without int-to-float conversions (Marlin's trick): the
// xor makes each byte b + 128, and a byte in the low mantissa bits of 2^23 reads as 2^23 + b + 128
fn sx(w: u32) -> vec4<f32> {
  let u = w ^ 0x80808080u;
  let bytes = vec4<u32>(u, u >> 8u, u >> 16u, u >> 24u) & vec4<u32>(0xFFu);
  return bitcast<vec4<f32>>(bytes | vec4<u32>(0x4B000000u)) - vec4<f32>(8388736.0);
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_index) li: u32,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let paddedN = ((p.N + 63u) / 64u) * 64u;
  let nb = p.K / 32u;

  // Flatten a one- or two-dimensional dispatch. The host keeps x within the
  // WebGPU per-dimension limit and uses y for the remaining workgroups.
  let packedVecIdx = (wg.y * nwg.x + wg.x) * 256u + li;
  let totalVectors = nb * paddedN * 8u;
  if (packedVecIdx >= totalVectors) {
    return;
  }

  let vecWithin32 = packedVecIdx & 7u;
  let rowBlock = packedVecIdx / 8u;
  let n = rowBlock % paddedN;
  let kb = rowBlock / paddedN;
  let packedScaleIdx = kb * paddedN + n;

  if (n < p.N) {
    let scale = S[n * nb + kb];
    let sourceWordIdx = n * (p.K / 4u) + kb * 8u + vecWithin32;
    // The f32 multiplication and the f16 store match the original matmul
    // path's sx(word) * scale followed by its f16 workgroup-tile store.
    PW[packedVecIdx] = vec4<f16>(sx(W[sourceWordIdx]) * scale);
    if (vecWithin32 == 0u) {
      PS[packedScaleIdx] = scale;
    }
  } else {
    PW[packedVecIdx] = vec4<f16>(0.0);
    if (vecWithin32 == 0u) {
      PS[packedScaleIdx] = 0.0;
    }
  }
}
