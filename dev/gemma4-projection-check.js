import { GpuGemma4Vision } from "../js/src/gpu-gemma4-vision.js";

const decode = (data) => Uint8Array.from(atob(data), (value) => value.charCodeAt(0));

async function tinyVision(runtime, width) {
  const response = await fetch("../tests/fixtures/gemma4-vision-tiny.json");
  if (!response.ok) throw new Error(`vision fixture HTTP ${response.status}`);
  const fixture = await response.json();
  let offset = 0;
  const values = [];
  const tensors = fixture.tensors.map((tensor) => {
    let value = decode(tensor.data);
    let shape = tensor.shape;
    if (tensor.name === "embed_vision.embedding_projection") {
      shape = [width, shape[1]];
      value = value.slice(0, width * shape[1] * 2);
    }
    values.push(value);
    const info = { name: tensor.name, dtype: tensor.dtype, shape, offset, size: value.byteLength };
    offset += value.byteLength;
    return info;
  });
  const header = { config: { ...fixture.config, text_hidden_size: width }, tensors };
  const json = new TextEncoder().encode(JSON.stringify(header));
  const prefix = new Uint8Array(16 + json.length);
  prefix.set([0x4b, 0x56, 0x4c, 0x41]);
  new DataView(prefix.buffer).setUint32(4, 1, true);
  new DataView(prefix.buffer).setUint32(8, json.length, true);
  prefix.set(json, 16);
  const vision = new GpuGemma4Vision(runtime, { prefix }, header);
  offset = 0;
  for (const value of values) { vision.write(offset, value); offset += value.byteLength; }
  await vision.init();
  return { vision, cases: fixture.cases };
}

function compare(actual, expected, tolerance, label) {
  if (actual.length !== expected.length) throw new Error(`${label} length mismatch`);
  let maxAbs = 0;
  for (let i = 0; i < actual.length; i++) {
    if (!Number.isFinite(actual[i])) throw new Error(`${label} nonfinite decoder row`);
    maxAbs = Math.max(maxAbs, Math.abs(actual[i] - expected[i]));
  }
  if (maxAbs > tolerance) throw new Error(`${label} maxAbs ${maxAbs} exceeds ${tolerance}`);
  return maxAbs;
}

/** Tiny same-device vision/decoder integration, including delayed nonfinite rejection. */
export async function checkGpuProjectionDecoder(runtime, target) {
  const { vision, cases } = await tinyVision(runtime, target.cfg.hidden);
  const owned = [];
  const profiler = target.profiler;
  target.profiler = null;
  try {
    const projected = [];
    for (const test of [cases[0], cases.at(-1)]) {
      const image = { patches: new Float32Array(decode(test.patches).buffer), positions: new Int32Array(decode(test.positions).buffer), softTokens: test.softTokens };
      const cpu = await vision.forward(image);
      const gpu = await vision.forward(image, { gpuOutput: true });
      owned.push(gpu);
      projected.push({ cpu, gpu, image });
    }
    const ids = Uint32Array.from([3, 258880, 258880, 4, 258880, 5, 7, 9]);
    const spans = (kind) => [{ start: 1, len: 2, rows: projected[0][kind] }, { start: 4, len: 1, rows: projected[1][kind] }];
    const expected = await target.forward(ids, null, spans("cpu"));
    const actual = await target.forward(ids, null, spans("gpu"));
    const ordinaryMaxAbs = compare(actual, expected, 0, "GPU image handoff");
    const inputs = [ids, Uint32Array.from([3, 258880, 258880, 4, 258880, 5, 11, 13, 15])];
    const dense = new Float32Array(2 * target.cfg.hidden);
    for (let i = 0; i < inputs.length; i++) dense.set(await target.forward(inputs[i], null, spans("cpu")), i * target.cfg.hidden);
    const shared = await target.forwardSharedPrefix(inputs.map((input) => ({ ids: input, imageSpans: spans("gpu") })));
    if (!shared || !target.lastPrefixReuse.enabled) throw new Error("GPU image handoff did not exercise shared-prefix decoding");
    const sharedMaxAbs = compare(shared, dense, 2e-5, "shared GPU image handoff");
    const mixed = [];
    for (const index of [0, 1]) {
      const parts = spans("cpu");
      parts[index].rows = projected[index].gpu;
      const ordinary = await target.forward(ids, null, parts);
      const joined = await target.forwardSharedPrefix(inputs.map((input) => ({ ids: input, imageSpans: parts })));
      if (!joined) throw new Error("mixed image rows skipped shared-prefix decoding");
      mixed.push({ gpuImage: index, ordinaryMaxAbs: compare(ordinary, expected, 0, "mixed image handoff"), sharedMaxAbs: compare(joined, dense, 2e-5, "shared mixed image handoff") });
    }

    // Overflow the actual projector, allowing the production finite kernel to flag its output.
    const matrix = vision.weights.get("embed_vision.embedding_projection");
    runtime.device.queue.writeBuffer(matrix.buf, 0, new Uint32Array([0x7f807f80]));
    const invalid = await vision.forward(projected[0].image, { gpuOutput: true });
    owned.push(invalid);
    const invalidSpans = [{ start: 1, len: 2, rows: invalid }];
    const invalidIds = Uint32Array.from([3, 258880, 258880, 4, 5, 7, 9]);
    const rejected = [];
    for (const [kind, run] of [
      ["ordinary", () => target.forward(invalidIds, null, invalidSpans)],
      ["shared", () => target.forwardSharedPrefix([invalidIds, Uint32Array.from([3, 258880, 258880, 4, 5, 11, 13])].map((input) => ({ ids: input, imageSpans: invalidSpans })))],
    ]) {
      let failed = false;
      try { await run(); }
      catch (error) { if (!/image projection contains non-finite values/.test(error.message)) throw error; failed = true; }
      if (!failed) throw new Error(`${kind} decoder accepted a nonfinite GPU projection`);
      rejected.push(kind);
    }
    return { images: 2, shapeReset: true, ordinaryMaxAbs, sharedMaxAbs, mixed, nonfiniteRejected: rejected };
  } finally {
    target.profiler = profiler;
    for (const rows of owned) rows.destroy();
    vision.destroy();
  }
}
