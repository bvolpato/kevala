// Gemma 4's GPU owns the embeddings and decoder; the coordinator tokenizes and scores labels.
import { GpuGemma4, gemma4Config } from "../gpu-gemma4.js";
import { gpuLayouts, parsePackHeader } from "../pack-layout.js";
import { loadGemma4Vision } from "../gemma4-vision-pack.js";
import { assertImageCount, imagePixels } from "../images.js";
import { disposeGemma4VisionImage, prepareGemma4GpuVisionImage, prepareGemma4Image, prepareGemma4VisionImage } from "../gemma4-image.js";
import { createGemma4PleStore } from "../gemma4-ple-store.js";

const enc = new TextEncoder();
const GPU_IMAGE_ROWS_LIMIT = 64 * 1024 * 1024;

function imageTokenBudget(request, engine) {
  if (request.imageTokens === undefined) return engine.imageTokens;
  if (!engine.vision || !Number.isSafeInteger(engine.imageTokens) || engine.imageTokens < 1) {
    throw new Error("imageTokens requires a loaded Gemma 4 vision companion with an imageTokens capacity");
  }
  if (!Number.isSafeInteger(request.imageTokens) || request.imageTokens < 1 || request.imageTokens > engine.imageTokens) {
    throw new RangeError(`imageTokens must be an integer from 1 to ${engine.imageTokens}`);
  }
  return request.imageTokens;
}

export default {
  arch: "gemma4",
  about: "Gemma 4 dense text decoder with direct option scoring",
  maxShards: () => 1,
  gpuLayouts: (header, bytes) => gpuLayouts(header, bytes, (name) => name !== "readout.labels"),
  async createGpu(gpu, layout, header, engine) {
    const cfg = { ...gemma4Config(header), hostPle: !!engine?.vision };
    if (cfg.hostPle) {
      const tensors = parsePackHeader(layout.prefix).tensors;
      const word = tensors.find((t) => t.name === "embed");
      cfg.hostStore = await createGemma4PleStore(tensors.filter((t) => /^ple\.\d+\.embed$/.test(t.name) || (t === word && t.dtype === "q8")), engine.hostStoreResource);
      cfg.hostWord = !!cfg.hostStore && word?.dtype === "q8";
      if (cfg.hostStore) engine.registerResource(cfg.hostStore);
    }
    return new GpuGemma4(gpu, layout, cfg);
  },
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
    let gpuImageBytes = 0;
    if (engine.vision) engine.vision.lastProfile = null;
    const imageTiming = { decode: 0, preprocess: 0, forward: 0, images: 0, patches: 0, softTokens: 0 };
    const visionProfile = {};
    try {
      const normalized = [];
      for (const request of requests) {
        const imageTokens = imageTokenBudget(request, engine);
        if (assertImageCount(request) && !engine.vision) throw new Error("load a Gemma 4 vision companion to classify images");
        const parts = [];
        for (const part of request.parts || []) {
          if (part.type !== "image") { parts.push(part); continue; }
          const decodeStart = performance.now();
          // This plugin runs after engine-worker's message/MessagePort clone. Borrow raw RGBA
          // bytes from that request and let requestQueue wipe them after inference completes.
          const pixels = await imagePixels(part.data, { signal: engine.signal, borrowRGBA: true });
          const requestOwnsPixels = pixels.data === part.data?.data;
          const preprocessStart = performance.now();
          imageTiming.decode += preprocessStart - decodeStart;
          let prepared;
          try {
            if (engine.vision.prepareGpuPixels === true && engine.vision.imageGpu?.enabled === true && !engine.vision.profiler && engine.vision.prepareCompactPixels === true && engine.vision.compactPatches !== false) {
              prepared = prepareGemma4GpuVisionImage(pixels, imageTokens, engine.visionHeader.config.vision_config, { borrowInput: requestOwnsPixels });
            } else {
              const prepare = engine.vision.prepareCompactPixels === true && engine.vision.compactPatches !== false ? prepareGemma4VisionImage : prepareGemma4Image;
              prepared = prepare(pixels, imageTokens, engine.visionHeader.config.vision_config);
            }
          }
          finally { if (!requestOwnsPixels) pixels.data.fill(0); }
          const visionStart = performance.now();
          imageTiming.preprocess += visionStart - preprocessStart;
          const index = images.length;
          try {
            const bytes = prepared.softTokens * engine.vision.cfg.textHidden * 4;
            const projected = await engine.vision.forward(prepared, { gpuOutput: !engine.vision.profiler && gpuImageBytes + bytes <= GPU_IMAGE_ROWS_LIMIT });
            images.push(projected);
            const mode = projected instanceof Float32Array ? "encoder-wall" : "submission";
            imageTiming.forwardMode = imageTiming.forwardMode && imageTiming.forwardMode !== mode ? "mixed" : mode;
            const attention = engine.vision.attentionMode;
            imageTiming.attention = imageTiming.attention && imageTiming.attention !== attention ? "mixed" : attention;
            imageTiming.fusedPasses = (imageTiming.fusedPasses || 0) + (engine.vision.fusionStats?.removedPasses || 0);
            if (!(projected instanceof Float32Array)) gpuImageBytes += projected.byteLength;
          }
          finally {
            disposeGemma4VisionImage(prepared);
            prepared.patches?.fill(0);
          }
          imageTiming.forward += performance.now() - visionStart;
          imageTiming.images += 1;
          imageTiming.patches += prepared.paddedRows ?? prepared.positions.length / 2;
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
      const pleTiming = { ms: 0, cacheHits: 0, sequences: count, vectorCacheHits: 0, vectorCacheCaptures: 0, vectorCacheBytes: 0 };
      let offset = 1;
      let tokens = 0;
      const prefixReuse = { enabled: false, batchedSuffixes: false, compactSourceRows: false, cachedTextPrefix: false, staticPrefixRows: 0, retiledPrefixMatrices: 0, narrowPrefixMatrices: 0, rmsNextFusions: 0, prefixGeluFusions: 0, sharedMatvecMatrices: 0, splitScratchBytes: 0, reusedTokens: 0, totalTokens: 0, sharedPrefixTokens: 0 };
      const collectPleVectors = () => {
        const cache = engine.gpu.lastPleVectorCache;
        pleTiming.vectorCacheHits += cache?.hits || 0;
        pleTiming.vectorCacheCaptures += cache?.captures || 0;
        pleTiming.vectorCacheBytes = Math.max(pleTiming.vectorCacheBytes, cache?.bytes || 0);
      };
      const forwardOne = async (i) => {
        const length = hasImages ? sequences[i].ids.length : batch[offset++];
        const ids = hasImages ? Uint32Array.from(sequences[i].ids) : batch.subarray(offset, offset + length);
        if (!hasImages) offset += length;
        tokens += length;
        const spans = hasImages ? sequences[i].image_spans.map((span) => ({ start: span.start, len: span.len, rows: images[span.image_index] })) : [];
        rows.set(await engine.gpu.forward(ids, null, spans), i * width);
        collectPleVectors();
        if (engine.gpu.lastPle) {
          pleTiming.ms += engine.gpu.lastPle.ms;
          pleTiming.cacheHits += engine.gpu.lastPle.cacheHits ?? Number(engine.gpu.lastPle.cacheHit);
        }
        for (const [kernel, ms] of Object.entries(engine.gpu.lastProfile || {})) {
          profile[kernel] = (profile[kernel] || 0) + ms;
        }
      };
      let ranges = [];
      if (hasImages) {
        let cursor = 0;
        for (const request of requests) {
          const questions = request.questions;
          const n = Array.isArray(questions) ? questions.length : questions && typeof questions === "object" ? Object.keys(questions).length : 0;
          ranges.push([cursor, cursor + n]);
          cursor += n;
        }
        if (cursor !== count) ranges = [];
      }
      if (ranges.length) {
        for (let requestIndex = 0; requestIndex < ranges.length; requestIndex++) {
          const [startIndex, endIndex] = ranges[requestIndex];
          let sharedRows = null;
          if (endIndex - startIndex > 1 && requests[requestIndex].shareImagePrefix !== false) {
            const inputs = sequences.slice(startIndex, endIndex).map((sequence) => ({
              ids: Uint32Array.from(sequence.ids),
              imageSpans: sequence.image_spans.map((span) => ({ start: span.start, len: span.len, rows: images[span.image_index] })),
            }));
            sharedRows = await engine.gpu.forwardSharedPrefix(inputs);
          }
          if (sharedRows) {
            rows.set(sharedRows, startIndex * width);
            const reuse = engine.gpu.lastPrefixReuse;
            prefixReuse.enabled = true;
            prefixReuse.batchedSuffixes ||= reuse.batchedSuffixes;
            prefixReuse.compactSourceRows ||= reuse.compactSourceRows;
            prefixReuse.cachedTextPrefix ||= reuse.cachedTextPrefix;
            prefixReuse.staticPrefixRows += reuse.staticPrefixRows || 0;
            prefixReuse.retiledPrefixMatrices += reuse.retiledPrefixMatrices || 0;
            prefixReuse.narrowPrefixMatrices += reuse.narrowPrefixMatrices || 0;
            prefixReuse.rmsNextFusions += reuse.rmsNextFusions || 0;
            prefixReuse.prefixGeluFusions += reuse.prefixGeluFusions || 0;
            prefixReuse.sharedMatvecMatrices += reuse.sharedMatvecMatrices || 0;
            prefixReuse.splitScratchBytes = Math.max(prefixReuse.splitScratchBytes, reuse.splitScratchBytes || 0);
            prefixReuse.reusedTokens += reuse.reusedTokens;
            prefixReuse.sharedPrefixTokens += reuse.sharedPrefixTokens;
            tokens += reuse.totalTokens;
            collectPleVectors();
            if (engine.gpu.lastPle) {
              pleTiming.ms += engine.gpu.lastPle.ms;
              pleTiming.cacheHits += engine.gpu.lastPle.cacheHits ?? Number(engine.gpu.lastPle.cacheHit);
            }
          } else {
            for (let i = startIndex; i < endIndex; i++) await forwardOne(i);
          }
        }
      } else {
        for (let i = 0; i < count; i++) await forwardOne(i);
      }
      if (hasImages) prefixReuse.totalTokens = tokens;
      engine.gpu.lastProfile = Object.keys(profile).length ? profile : null;
      const finished = performance.now();
      let minFinalRms = Infinity;
      let maxFinalRms = 0;
      if (hasImages) for (let i = 0; i < count; i++) {
        let sum = 0;
        for (const value of rows.subarray(i * width, (i + 1) * width)) sum += value * value;
        const rms = Math.sqrt(sum / width);
        minFinalRms = Math.min(minFinalRms, rms);
        maxFinalRms = Math.max(maxFinalRms, rms);
      }
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
          ...(hasImages ? { image: imageTiming, ple: pleTiming, decoder: { minFinalRms, maxFinalRms }, prefixReuse, tokenize: ready - tokenizeStart, readout: performance.now() - finished } : {}),
        },
      };
    } finally {
      for (const rows of images) {
        if (rows instanceof Float32Array) rows.fill(0);
        else rows.destroy();
      }
    }
  },
};
