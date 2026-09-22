// Gemma 4's GPU owns the embeddings and decoder; the coordinator tokenizes and scores labels.
import { GpuGemma4, gemma4Config } from "../gpu-gemma4.js";
import { gpuLayouts } from "../kev-layout.js";

const enc = new TextEncoder();

export default {
  arch: "gemma4",
  about: "Gemma 4 dense text decoder with direct option scoring",
  maxShards: () => 1,
  gpuLayouts: (header, bytes) => gpuLayouts(header, bytes, (name) => name !== "readout.labels"),
  createGpu: (gpu, layout, header) => new GpuGemma4(gpu, layout, gemma4Config(header)),
  async initGpu(engine) {
    await engine.gpu.init();
  },
  async run(engine, requests) {
    const coord = engine.coord;
    const start = performance.now();
    coord.withInput(enc.encode(JSON.stringify({ requests })), (p, n) => coord.check(coord.x.kevala_gemma4_prepare(p, n)));
    const batch = new Uint32Array(coord.out().slice().buffer);
    const count = batch[0];
    const width = engine.header.config.hidden_size;
    const rows = new Float32Array(count * width);
    const ready = performance.now();
    const profile = {};
    let offset = 1;
    let tokens = 0;
    for (let i = 0; i < count; i++) {
      const length = batch[offset++];
      const ids = batch.subarray(offset, offset + length);
      offset += length;
      tokens += length;
      rows.set(await engine.gpu.forward(ids), i * width);
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
      timing: { prepare: ready - start, forward: finished - ready, total: performance.now() - start, tokens },
    };
  },
};
