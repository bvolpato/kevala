// Gemma 4's GPU owns the embeddings and decoder; the coordinator tokenizes and scores labels.
import { GpuGemma4, gemma4Config } from "../gpu-gemma4.js";
import { gpuLayouts } from "../pack-layout.js";
import { loadGemma4Vision } from "../gemma4-vision-pack.js";
import { assertImageCount, imagePixels } from "../images.js";
import { prepareGemma4Image } from "../gemma4-image.js";

const enc = new TextEncoder();

export default {
  arch: "gemma4",
  about: "Gemma 4 dense text decoder with direct option scoring",
  maxShards: () => 1,
  gpuLayouts: (header, bytes) => gpuLayouts(header, bytes, (name) => name !== "readout.labels"),
  createGpu: (gpu, layout, header, engine) => new GpuGemma4(gpu, layout, { ...gemma4Config(header), hostPle: !!engine?.vision }),
  async initGpu(engine) {
    await engine.gpu.init();
    if (engine.vision) {
      engine.gpu.ensure(engine.imageTokens + 128);
      engine.gpu.ensureTail(1);
    }
  },
  loadVision: loadGemma4Vision,
  async run(engine, requests) {
    const coord = engine.coord;
    const start = performance.now();
    const images = [];
    if (engine.vision) engine.vision.lastProfile = null;
    const imageTiming = { decode: 0, preprocess: 0, forward: 0, images: 0, patches: 0, softTokens: 0 };
    const visionProfile = {};
    try {
      const normalized = [];
      for (const request of requests) {
        if (assertImageCount(request) && !engine.vision) throw new Error("load a Gemma 4 vision companion to classify images");
        const parts = [];
        for (const part of request.parts || []) {
          if (part.type !== "image") { parts.push(part); continue; }
          const decodeStart = performance.now();
          const pixels = await imagePixels(part.data, { signal: engine.signal });
          const preprocessStart = performance.now();
          imageTiming.decode += preprocessStart - decodeStart;
          let prepared;
          try { prepared = prepareGemma4Image(pixels, engine.imageTokens, engine.visionHeader.config.vision_config); }
          finally { pixels.data.fill(0); }
          const visionStart = performance.now();
          imageTiming.preprocess += visionStart - preprocessStart;
          const index = images.length;
          try { images.push(await engine.vision.forward(prepared)); }
          finally { prepared.patches.fill(0); }
          imageTiming.forward += performance.now() - visionStart;
          imageTiming.images += 1;
          imageTiming.patches += prepared.positions.length / 2;
          imageTiming.softTokens += prepared.softTokens;
          for (const [kernel, ms] of Object.entries(engine.vision.lastProfile || {})) visionProfile[kernel] = (visionProfile[kernel] || 0) + ms;
          parts.push({ type: "image", image_tokens: prepared.softTokens, image_index: index });
        }
        normalized.push({ ...request, ...(request.parts ? { parts } : {}) });
      }
      const hasImages = images.length > 0;
      const tokenizeStart = performance.now();
      coord.withInput(enc.encode(JSON.stringify({ requests: normalized })), (p, n) => coord.check(hasImages ? coord.x.kevala_gemma4_prepare_images(p, n) : coord.x.kevala_gemma4_prepare(p, n)));
      const sequences = hasImages ? JSON.parse(coord.outText()).sequences : null;
      const batch = hasImages ? null : new Uint32Array(coord.out().slice().buffer);
      const count = hasImages ? sequences.length : batch[0];
      const width = engine.header.config.hidden_size;
      const rows = new Float32Array(count * width);
      const ready = performance.now();
      const profile = { ...visionProfile };
      let offset = 1;
      let tokens = 0;
      for (let i = 0; i < count; i++) {
        const length = hasImages ? sequences[i].ids.length : batch[offset++];
        const ids = hasImages ? Uint32Array.from(sequences[i].ids) : batch.subarray(offset, offset + length);
        if (!hasImages) offset += length;
        tokens += length;
        const spans = hasImages ? sequences[i].image_spans.map((span) => ({ start: span.start, len: span.len, rows: images[span.image_index] })) : [];
        rows.set(await engine.gpu.forward(ids, null, spans), i * width);
        for (const [kernel, ms] of Object.entries(engine.gpu.lastProfile || {})) {
          profile[kernel] = (profile[kernel] || 0) + ms;
        }
      }
      engine.gpu.lastProfile = Object.keys(profile).length ? profile : null;
      const finished = performance.now();
      const [ptr, bytes] = coord.put(new Uint8Array(rows.buffer));
      try {
        coord.call(() => coord.check(coord.x.kevala_gemma4_finish(ptr, rows.length)));
      } finally {
        coord.x.kevala_free(ptr, bytes);
      }
      return {
        responses: JSON.parse(coord.outText()),
        timing: {
          prepare: ready - start, forward: finished - ready, total: performance.now() - start, tokens,
          ...(hasImages ? { image: imageTiming, tokenize: ready - tokenizeStart, readout: performance.now() - finished } : {}),
        },
      };
    } finally {
      for (const rows of images) rows.fill(0);
    }
  },
};
