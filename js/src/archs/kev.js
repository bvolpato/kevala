// Kev: Qwen3.5 hybrid decoder (Gated DeltaNet + gated attention) + pointer head
// (jaredpalmer/kev). The coordinator tokenizes, embeds and runs the pointer head; the GPU runs
// the transformer layers, states first and then question branches from their state's carry.

import { GpuKev, kevConfig, parseKevBatch } from "../gpu-kev.js";
import { Wasm } from "../wasm.js";
import { fetchBytes, fetchRange, hfFile } from "../source.js";
import { gpuLayouts } from "../pack-layout.js";

const enc = new TextEncoder();
const now = () => performance.now();

export default {
  arch: "kev",
  about: "Qwen3.5 hybrid decoder + pointer head",

  /** A state's recurrent carries and attention keys are kept for the next request about it. */
  stateCache: true,

  /** Not split across WebAssembly workers yet: on the CPU one instance runs every layer. */
  maxShards: () => 1,

  gpuLayouts: (header, headerBytes) => gpuLayouts(header, headerBytes),

  createGpu: (gpu, layout, header) => new GpuKev(gpu, layout, kevConfig(header)),

  async initGpu(e) {
    await e.gpu.init();
  },

  async run(e, requests) {
    const c = e.coord;
    const D = e.header.config.hidden_size;
    const t0 = now();
    c.withInput(enc.encode(JSON.stringify({ requests })), (p, l) => c.check(c.x.kevala_kev_prepare(p, l)));
    const batch = parseKevBatch(new Uint32Array(c.out().slice().buffer));
    const limit = Math.min(e.gpu.device.limits.maxStorageBufferBindingSize, e.gpu.device.limits.maxBufferSize);
    const width = Math.max(2 * e.header.config.intermediate_size, e.gpu.dims.linearProj, e.gpu.dims.attentionProj);
    // ensure() rounds capacities up to a power of two. Bound scratch and recurrent carries
    // before embeddings are allocated, keeping large candidate sets usable on smaller GPUs.
    const maxTokens = 2 ** Math.floor(Math.log2(Math.min(limit / (4 * width), D > 1024 ? 2048 : Infinity)));
    if (requests.length > 1 && (Math.max(batch.T1, batch.T2) > maxTokens || (D > 1024 && requests.length > 4))) {
      const mid = Math.ceil(requests.length / 2);
      const first = await this.run(e, requests.slice(0, mid));
      const firstProfile = e.gpu.lastProfile;
      const second = await this.run(e, requests.slice(mid));
      if (firstProfile && e.gpu.lastProfile) {
        const merged = { ...firstProfile };
        for (const [kernel, ms] of Object.entries(e.gpu.lastProfile)) merged[kernel] = (merged[kernel] || 0) + ms;
        e.gpu.lastProfile = merged;
      }
      return {
        responses: [...first.responses, ...second.responses],
        timing: {
          prepare: first.timing.prepare + second.timing.prepare,
          forward: first.timing.forward + second.timing.forward,
          total: now() - t0,
          tokens: first.timing.tokens + second.timing.tokens,
          cache: second.timing.cache,
        },
      };
    }
    const x1 = c.f32(c.call(() => c.x.kevala_kev_embed(1)), batch.T1 * D).slice();
    const x2 = c.f32(c.call(() => c.x.kevala_kev_embed(2)), batch.T2 * D).slice();
    const t1 = now();
    const ids = c.u32(c.x.kevala_kev_ids(1), batch.T1).slice();
    const rows = await e.gpu.forward(x1, x2, batch, ids);
    const t2 = now();
    const [rp, rl] = c.put(new Uint8Array(rows.buffer));
    try {
      c.call(() => c.check(c.x.kevala_kev_finish(rp, rows.length)));
    } finally {
      c.x.kevala_free(rp, rl);
    }
    return {
      responses: JSON.parse(c.outText()),
      timing: { prepare: t1 - t0, forward: t2 - t1, total: now() - t0, tokens: batch.T1 + batch.T2, cache: { ...e.gpu.stats } },
    };
  },

  /**
   * Converts from the upstream repos: Kev's adapter, head and tokenizer, and the Qwen3.5 base
   * weights, of which only the language model's byte range is downloaded (the vision tower and
   * the multi-token-prediction head are skipped). Tensors are merged and stored as they
   * arrive; the pack streams out of WebAssembly memory without a second full copy.
   */
  async convert(module, spec, opts) {
    const { signal, onProgress } = opts;
    const precision = opts.precision || "bf16";
    if (!["bf16", "f32", "q8"].includes(precision)) throw new Error('precision must be "bf16", "f32", or "q8"');
    const kev = (f) => hfFile(spec.repo, spec.revision, f);
    const base = (f) => hfFile(spec.base.repo, spec.base.revision, f);
    const text = async (url) => {
      const r = await fetch(url, { signal });
      if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
      return r.text();
    };
    onProgress?.({ phase: "download", file: "configs", loaded: 0, total: 0 });
    const [cfg, acfg, index] = await Promise.all([text(base("config.json")), text(kev("adapter_config.json")), text(base("model.safetensors.index.json"))]);
    const files = [...new Set(Object.values(JSON.parse(index).weight_map))];
    if (!files.length) throw new Error("base checkpoint index has no weight files");
    const tok = await fetchBytes(kev("tokenizer.json"), "tokenizer.json", opts);
    const adapter = await fetchBytes(kev("adapter_model.safetensors"), "adapter_model.safetensors", opts);
    const head = await fetchBytes(kev("head.pt"), "head.pt", opts);
    // An error page must not be read as the safetensors header length.
    const range = async (url, lo, hi) => {
      const r = await fetch(url, { signal, headers: { range: `bytes=${lo}-${hi}` } });
      if (r.status !== 206) throw new Error(`${url}: HTTP ${r.status} for a byte range`);
      const bytes = new Uint8Array(await r.arrayBuffer());
      if (bytes.byteLength !== hi - lo + 1) throw new Error(`${url}: got ${bytes.byteLength} of ${hi - lo + 1} bytes`);
      return bytes;
    };
    const heads = await Promise.all(files.map(async (file) => {
      const url = base(file);
      const n = Number(new DataView((await range(url, 0, 7)).buffer).getBigUint64(0, true));
      if (!Number.isSafeInteger(n) || n < 2 || n > 32 * 1024 * 1024) throw new Error(`${file}: invalid safetensors header length ${n}`);
      return range(url, 0, 8 + n - 1);
    }));
    const stHead = files.length === 1 ? heads[0] : new Uint8Array(heads.reduce((sum, h) => sum + 4 + h.byteLength, 0));
    if (files.length > 1) {
      const view = new DataView(stHead.buffer);
      let at = 0;
      for (const h of heads) {
        view.setUint32(at, h.byteLength, true);
        stHead.set(h, at + 4);
        at += 4 + h.byteLength;
      }
    }
    const model = JSON.stringify({
      name: spec.name || spec.repo.split("/").pop(),
      source: spec.source || (spec.repo ? `https://huggingface.co/${spec.repo}` : undefined),
      revision: spec.revision,
      base: `https://huggingface.co/${spec.base.repo}`,
      base_revision: spec.base.revision,
      author: spec.author,
      license: spec.license,
      converter: "kevala (in-browser)",
      precision,
      quantization: precision === "q8"
        ? `LoRA merged in f32, then int8 symmetric absmax, one f32 scale per ${spec.block || 32} weights; norms, gates, conv, pointer head in f32`
        : `LoRA merged in f32, then ${precision.toUpperCase()} matrices without quantization; norms, gates, conv, pointer head in f32`,
    });
    const w = await Wasm.create(module, 0);
    const args = [stHead, cfg, tok, adapter, acfg, head, model].map((d) => w.put(typeof d === "string" ? enc.encode(d) : d));
    const precisionArg = w.put(enc.encode(precision));
    try {
      const plan = files.length === 1 ? w.x.kevala_kev_convert_plan_precision : w.x.kevala_kev_convert_plan_sharded_precision;
      if (plan) w.call(() => w.check(plan(...args.flat(), spec.block || 32, ...precisionArg)));
      else if (precision === "q8" && files.length === 1) w.call(() => w.check(w.x.kevala_kev_convert_plan(...args.flat(), spec.block || 32)));
      else throw new Error(`This WebAssembly build does not support ${precision} Kev conversion for this checkpoint. Rebuild Kevala.`);
    } finally {
      w.x.kevala_free(...precisionArg);
    }
    for (const [p, l] of args) w.x.kevala_free(p, l);
    const plan = JSON.parse(w.outText());
    const sources = plan.sources.map((s) => (s.length === 2 ? [0, ...s] : s).map(Number));
    let si = 0;
    for (let shard = 0; shard < files.length; shard++) {
      const shardSources = sources.filter(([file]) => file === shard);
      if (!shardSources.length) continue;
      const lo = shardSources[0][1];
      const hi = shardSources.at(-1)[1] + shardSources.at(-1)[2];
      let ptr = 0;
      let fill = -1;
      let at = lo;
      for await (const chunk of await fetchRange(base(files[shard]), lo, hi, files[shard], opts)) {
        let p = 0;
        while (p < chunk.byteLength && si < sources.length && sources[si][0] === shard) {
          const [, o, l] = sources[si];
          if (at + p < o) {
            p += Math.min(chunk.byteLength - p, o - (at + p));
            continue;
          }
          if (fill < 0) {
            ptr = w.x.kevala_alloc_vec(l);
            fill = 0;
          }
          const k = Math.min(chunk.byteLength - p, l - fill);
          w.bytes(ptr + fill, k).set(chunk.subarray(p, p + k));
          fill += k;
          p += k;
          if (fill === l) {
            w.call(() => w.check(w.x.kevala_kev_convert_source(si, ptr, l)));
            fill = -1;
            si++;
            onProgress?.({ phase: "convert", loaded: si, total: sources.length });
          }
        }
        at += chunk.byteLength;
      }
      if (fill >= 0 || (si < sources.length && sources[si][0] === shard)) throw new Error(`conversion stopped before ${files[shard]} ended`);
    }
    if (w.outText() !== "1") throw new Error(`conversion stopped at source ${si} of ${sources.length}`);
    const total = Number(plan.total);
    const packPtr = Number(plan.pack);
    return {
      size: total,
      async *chunks() {
        for (let off = 0; off < total; off += 16 << 20) yield w.bytes(packPtr + off, Math.min(16 << 20, total - off)).slice();
        w.x.kevala_convert_drop();
      },
    };
  },
};
