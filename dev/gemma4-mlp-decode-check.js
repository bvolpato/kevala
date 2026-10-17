import { startMlpProbe } from "./gemma4-mlp-k16-check.js";

const SPEC = { weights: "q8", f16: true, rows: 4, row56: false, groups: 1, splitTarget: 256 };
const MARLIN = `fn sx(w: u32) -> vec4<f32> {
  let u = w ^ 0x80808080u;
  let bytes = vec4<u32>(u, u >> 8u, u >> 16u, u >> 24u) & vec4<u32>(0xFFu);
  return bitcast<vec4<f32>>(bytes | vec4<u32>(0x4B000000u)) - vec4<f32>(8388736.0);
}`;

export function marlinDecodeCode(original, registeredReference) {
  const sx = (code) => code.match(/fn sx\([\s\S]*?\n\}/)?.[0];
  const before = sx(original), replacement = sx(registeredReference);
  if (!before || replacement !== MARLIN || !before.includes("vec4<i32>")) throw new Error("GEMM decode probe requires current integer sx and the registered Marlin float-bit sx");
  if (original.includes("enable f16;")) throw new Error("GEMM decode candidate must retain F32 execution");
  const code = original.replace(before, replacement);
  if (code.replace(replacement, before) !== original || sx(code) !== replacement) throw new Error("GEMM decode probe changed more than sx");
  return code;
}

export function decodeProof() {
  const f = new Float32Array(1), u = new Uint32Array(f.buffer);
  const bits = (value) => { f[0] = value; return u[0]; };
  const scales = [0, -0, 0.0002, -0.0002, 1 / 3, -1 / 3, 1, -1, 17, -17, 2 ** -149, -(2 ** -149), 2 ** -126, -(2 ** -126), 2 ** 20, -(2 ** 20)].map(Math.fround);
  let products = 0;
  for (let lane = 0; lane < 4; lane++) for (let byte = 0; byte < 256; byte++) {
    const shift = lane * 8, packed = ((0x5a5a5a5a & ~(255 << shift)) | (byte << shift)) >>> 0;
    const unsigned = ((packed ^ 0x80808080) >>> shift) & 255;
    u[0] = 0x4b000000 | unsigned;
    const actual = Math.fround(f[0] - 8388736), expected = (byte << 24) >> 24;
    if (bits(actual) !== bits(expected)) throw new Error(`Marlin decode mismatch at byte ${byte}, lane ${lane}`);
    for (const scale of scales) {
      if (bits(Math.fround(actual * scale)) !== bits(Math.fround(expected * scale))) throw new Error(`Marlin signed-scale mismatch at byte ${byte}, lane ${lane}`);
      products++;
    }
  }
  return { ok: true, byteValues: 256, lanes: 4, decodedWords: 1024, signedScaleProducts: products, scaleBits: scales.map((value) => `0x${bits(value).toString(16).padStart(8, "0")}`), signedZeroChecked: true };
}

if (typeof window !== "undefined") {
  const reference = (source) => source("matmul_gemma_batch", SPEC);
  startMlpProbe({
    derive: (paired, ordinary, source) => marlinDecodeCode(paired, reference(source)),
    plainVariant: (ordinary, source) => marlinDecodeCode(ordinary, reference(source)),
    proof: decodeProof, candidateName: "Decode", checkName: "gemma4-mlp-decode", sharedBytes: 24576,
    metadata: { decoderProvenance: { kernel: "matmul_gemma_batch", spec: SPEC, extractedFunctionOnly: "sx", actualExecutionPrecision: "f32", currentF32Decoder: "integer sign extension" } },
  });
}
