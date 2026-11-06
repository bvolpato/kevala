export const UTILITY_KERNELS = ["weight_rows", "gemma4_vision_clip", "gemma4_vision_pool", "gemma4_vision_finite"];

export function isolatedUnavailable(kernel, code, capabilities) {
  const reasons = [];
  if (kernel === "attention_subgroup" && !capabilities.subgroup32) reasons.push("exact 32-lane subgroups");
  if (kernel === "kev_attention_tile" && (!capabilities.subgroups || capabilities.subgroupMinSize < 8)) reasons.push("subgroups of at least eight lanes");
  const directives = [...code.matchAll(/\b(?:enable|requires)\s+([a-z0-9_]+)\s*;/g)].map((match) => match[1]);
  for (const feature of directives) {
    if (feature === "f16" ? !capabilities.features.has("shader-f16") : feature === "subgroups" ? !capabilities.subgroups : !capabilities.languageFeatures.has(feature)) reasons.push(feature);
  }
  const storage = kernel === "attention_tile_f32" ? 30224 : kernel === "attention_tile" ? 25616 + (directives.includes("subgroups") ? 0 : 4352) : 0;
  if (storage > capabilities.workgroupStorageSize) reasons.push(`${storage} bytes of workgroup storage`);
  return [...new Set(reasons)];
}

export function recurrenceFixtureInput(input, tokens) {
  const heads = 16, width = 128, qk = heads * width, stride = 3 * qk;
  if (!(input instanceof Float32Array) || !Number.isSafeInteger(tokens) || tokens < 1 || input.length < tokens * stride) throw new Error("recurrence fixture requires complete default Kev Q/K/V rows");
  const output = input.slice(0, tokens * stride), reduction = new Float32Array(width), f32 = Math.fround;
  if (output.some((value) => !Number.isFinite(value))) throw new Error("recurrence fixture requires finite Q/K/V input");
  for (let token = 0; token < tokens; token++) for (let head = 0; head < 2 * heads; head++) {
    const base = token * stride + head * width;
    for (let lane = 0; lane < width; lane++) reduction[lane] = f32(output[base + lane] * output[base + lane]);
    for (let step = width / 2; step > 0; step >>= 1) for (let lane = 0; lane < step; lane++) reduction[lane] = f32(reduction[lane] + reduction[lane + step]);
    let inverse = f32(1 / Math.sqrt(f32(reduction[0] + f32(1e-6))));
    if (head < heads) inverse = f32(inverse * f32(0.08838834764831845));
    for (let lane = 0; lane < width; lane++) output[base + lane] = f32(output[base + lane] * inverse);
  }
  return output;
}

export function utilityFixture(kernel, tokens, width = 1024) {
  if (!UTILITY_KERNELS.includes(kernel)) return null;
  if (!Number.isSafeInteger(tokens) || tokens < 1 || !Number.isSafeInteger(width) || width < 4 || width % 4) {
    throw new Error("utility fixture needs positive rows and a width divisible by four");
  }
  const count = tokens * width;
  if (kernel === "gemma4_vision_finite") return { grid: [Math.ceil(count / 1024)], params: [count, 0, 0, 0], output: ["status", 1],
    preserveOutputInitialization: true, outputContract: "zero-initialized atomic status, all-finite scan", overrides: { status: new Uint32Array(4) } };
  if (kernel === "weight_rows") {
    if (width % 32) throw new Error("Q8 weight-row fixture width must be divisible by 32");
    return { grid: [tokens], params: [tokens, width, 32, 0], output: ["O", count],
      spec: { weights: "q8" }, overrides: { ids: Uint32Array.from({ length: tokens }, (_, row) => row % 4) } };
  }
  if (kernel === "gemma4_vision_clip") {
    return { grid: [Math.ceil(count / 1024)], params: [count, 0, 0, 0, { f: -0.75 }, { f: 0.75 }, 0, 0], output: ["Y", count], overrides: {} };
  }
  return { grid: [Math.ceil(count / 1024)], params: [tokens, width, 4, 1], output: ["Y", count],
    overrides: { I: Uint32Array.from({ length: tokens * 4 }, (_, index) => index) } };
}

// The fixture binds the same seeded F32 array to W/S or X/B/S. Reinterpret Q8
// bytes exactly, and round each pooling operation separately as the WGSL does.
export function utilityReference(kernel, input, tokens, width = 1024) {
  if (!utilityFixture(kernel, tokens, width)) return null;
  const count = tokens * width;
  const needed = kernel === "gemma4_vision_pool" ? count * 4 : kernel === "weight_rows" ? width : count;
  if (!(input instanceof Float32Array) || input.length < needed) throw new Error("utility reference input is too short");
  if (kernel === "gemma4_vision_finite") return Uint32Array.of(input.subarray(0, count).some((value) => !Number.isFinite(value)) ? 1 : 0);
  const output = new Float32Array(count);
  if (kernel === "weight_rows") {
    const weights = new Int8Array(input.buffer, input.byteOffset, input.byteLength);
    for (let row = 0; row < tokens; row++) for (let column = 0; column < width; column++) {
      const source = row % 4 * width + column;
      output[row * width + column] = Math.fround(weights[source] * input[Math.floor(source / 32)]);
    }
  } else if (kernel === "gemma4_vision_clip") {
    for (let index = 0; index < count; index++) output[index] = Math.min(0.75, Math.max(-0.75, input[index]));
  } else {
    for (let row = 0; row < tokens; row++) for (let column = 0; column < width; column++) {
      let value = 0;
      for (let patch = 0; patch < 4; patch++) value = Math.fround(value + Math.fround(input[(row * 4 + patch) * width + column] * 0.25));
      value = Math.fround(value * Math.fround(Math.sqrt(width)));
      output[row * width + column] = Math.fround(Math.fround(value - input[column]) * input[column]);
    }
  }
  return output;
}

export function utilityError(kernel, input, actual, tokens, width = 1024) {
  const expected = utilityReference(kernel, input, tokens, width);
  if (!expected) return null;
  if (actual.length !== expected.length) throw new Error(`${kernel}: output length differs from CPU reference`);
  if (kernel === "gemma4_vision_finite") {
    const status = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
    if (status[0] !== expected[0]) throw new Error(`${kernel}: atomic status ${status[0]} differs from CPU status ${expected[0]}`);
    return 0;
  }
  let maxAbs = 0;
  for (let index = 0; index < expected.length; index++) {
    if (!Number.isFinite(actual[index]) || !Number.isFinite(expected[index])) throw new Error(`${kernel}: nonfinite CPU comparison at ${index}`);
    maxAbs = Math.max(maxAbs, Math.abs(actual[index] - expected[index]));
  }
  if (maxAbs > 1e-5) throw new Error(`${kernel}: CPU mismatch ${maxAbs}`);
  return maxAbs;
}
