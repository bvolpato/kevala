// Repack the row-major int8 words and scales used by matmul.wgsl into a layout
// that makes the 64 adjacent output rows of one K block contiguous.
//
// PW is indexed as [k block][64-row tile][row within tile][word within 32],
// with zero-filled rows in the final tile. PS uses the matching
// [k block][padded row] layout. One invocation owns one PW u32; the invocation
// whose wordWithin32 is zero also owns the corresponding scale.

struct P { N: u32, K: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> S: array<f32>;
@group(0) @binding(3) var<storage, read_write> PW: array<u32>;
@group(0) @binding(4) var<storage, read_write> PS: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_index) li: u32,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let paddedN = ((p.N + 63u) / 64u) * 64u;
  let nb = p.K / 32u;

  // Flatten a one- or two-dimensional dispatch. Keeping x at or below the
  // WebGPU workgroup limit lets large model packs use y for the remaining
  // workgroups without changing the element mapping.
  let packedWordIdx = (wg.y * nwg.x + wg.x) * 256u + li;
  let totalWords = nb * paddedN * 8u;
  if (packedWordIdx >= totalWords) {
    return;
  }

  let wordWithin32 = packedWordIdx & 7u;
  let rowBlock = packedWordIdx / 8u;
  let n = rowBlock % paddedN;
  let kb = rowBlock / paddedN;
  let packedScaleIdx = kb * paddedN + n;

  if (n < p.N) {
    // The source has K/4 u32 words per row and eight words per 32-value
    // quantization block.
    let sourceWordIdx = n * (p.K / 4u) + kb * 8u + wordWithin32;
    PW[packedWordIdx] = W[sourceWordIdx];
    if (wordWithin32 == 0u) {
      PS[packedScaleIdx] = S[n * nb + kb];
    }
  } else {
    PW[packedWordIdx] = 0u;
    if (wordWithin32 == 0u) {
      PS[packedScaleIdx] = 0.0;
    }
  }
}
