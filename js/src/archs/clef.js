// Clef text records use every Qwen hidden row. The GPU holds both untied embedding tables;
// WebAssembly applies Qwen's final RMSNorm and the joint schema head.
import { GpuKev, kevConfig } from "../gpu-kev.js";
import { gpuLayouts } from "../pack-layout.js";

const enc = new TextEncoder();

export default {
  arch: "clef",
  about: "Qwen3.5 text decoder with a joint schema transformer head",
  maxShards: () => 1,
  gpuLayouts: (header, bytes) => gpuLayouts(header, bytes),
  createGpu(gpu, layout, header) {
    const trunk = new GpuKev(gpu, layout, kevConfig(header));
    trunk.stateCacheEnabled = false;
    return trunk;
  },
  async initGpu(engine) {
    await engine.gpu.init();
  },
  async run(engine, requests) {
    const coord = engine.coord;
    const width = engine.header.config.hidden_size;
    const start = performance.now();
    const responses = [];
    const profile = {};
    let prepare = 0;
    let forward = 0;
    let tokens = 0;
    // Keep all fields of a record together. Independent records can run sequentially without
    // multiplying the backbone's scratch memory or changing Clef's joint field attention.
    for (const request of requests) {
      const before = performance.now();
      if (request.images?.length || request.videos?.length) {
        throw new Error("Clef in Kevala supports text only; images and videos require the upstream multimodal processor and vision tower");
      }
      coord.withInput(enc.encode(JSON.stringify({ requests: [request] })), (p, n) => coord.check(coord.x.kevala_clef_prepare(p, n)));
      const plan = JSON.parse(coord.outText());
      const ids = Uint32Array.from(plan.ids);
      const count = ids.length;
      const batch = { R: 1, T1: 0, T2: count, states: [[0, 0]], branches: [[0, count, 0]], rows: Array.from({ length: count }, (_, i) => i) };
      const embeddings = await engine.gpu.weightRows("emb", ids);
      const ready = performance.now();
      const hidden = await engine.gpu.forward(new Float32Array(), embeddings, batch, new Uint32Array());
      for (const [kernel, ms] of Object.entries(engine.gpu.lastProfile || {})) profile[kernel] = (profile[kernel] || 0) + ms;
      const lexicalIds = [];
      for (const [a, b] of plan.option_spans) for (let i = a; i < b; i++) lexicalIds.push(ids[i]);
      const lexicalRows = await engine.gpu.weightRows("clef.lexical", Uint32Array.from(lexicalIds));
      const means = new Float32Array(plan.option_spans.length * width);
      let row = 0;
      plan.option_spans.forEach(([a, b], option) => {
        for (let j = 0; j < b - a; j++, row++) for (let k = 0; k < width; k++) means[option * width + k] += lexicalRows[row * width + k];
        for (let k = 0; k < width; k++) means[option * width + k] /= b - a;
      });
      const finished = performance.now();
      const [hp, hn] = coord.put(new Uint8Array(hidden.buffer, hidden.byteOffset, hidden.byteLength));
      const [lp, ln] = coord.put(new Uint8Array(means.buffer));
      try {
        coord.call(() => coord.check(coord.x.kevala_clef_finish(hp, hidden.length, lp, means.length)));
        responses.push(...JSON.parse(coord.outText()));
      } finally {
        coord.x.kevala_free(hp, hn);
        coord.x.kevala_free(lp, ln);
      }
      prepare += ready - before;
      forward += finished - ready;
      tokens += count;
    }
    engine.gpu.lastProfile = Object.keys(profile).length ? profile : null;
    return { responses, timing: { prepare, forward, total: performance.now() - start, tokens } };
  },
  async convert() {
    throw Object.assign(new Error("Clef's merged checkpoint exceeds the browser conversion memory limit. Convert locally with: kevala convert /path/to/clef-flash -o clef-flash-bf16.kevala --precision bf16. Use --precision q8 for optional quantization, then load the pack in Kevala."), { code: "NATIVE_CONVERSION_REQUIRED" });
  },
};
