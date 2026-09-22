// Laya: ModernBERT-large encoder + decision head (convaiinnovations/laya).
//
// An architecture plugin tells the generic engine worker how this family runs outside a single
// WebAssembly instance: on WebGPU, or split across tensor-parallel shard workers, and how to
// convert its upstream checkpoint in the browser.

import { Wasm, parseSegments } from "../wasm.js";
import { fetchUpstream } from "../source.js";
import { GpuTrunk } from "../gpu.js";

const enc = new TextEncoder();
const now = () => performance.now();

function trunkConfig(header) {
  const c = header.config;
  return {
    hidden: c.hidden_size,
    heads: c.num_attention_heads,
    layers: c.num_hidden_layers,
    intermediate: c.intermediate_size,
    global_every: c.global_attn_every_n_layers,
    window: c.local_attention / 2,
    rope_global: c.global_rope_theta,
    rope_local: c.local_rope_theta,
    norm_eps: c.norm_eps,
    head_layers: c.head_layers,
    head_ff: c.head_ff,
    head_heads: c.head_heads,
    head_norm_eps: c.head_norm_eps,
    steps: 2 * (c.num_hidden_layers + c.head_layers),
  };
}

export default {
  arch: "laya",
  about: "ModernBERT encoder + decision head",

  /** Tensor-parallel shards split attention heads, so at most one shard per head. */
  maxShards: (header) => header.config.num_attention_heads,

  createGpu: (gpu, layout, header) => new GpuTrunk(gpu, layout, trunkConfig(header)),

  /** The GPU also runs the bridge between encoder and head, which needs two coordinator tensors. */
  async initGpu(e) {
    const t = (n) => e.coordHeader.tensors.find((x) => x.name === n);
    const [fnorm, temb] = [t("final_norm"), t("type_emb")];
    // the coordinator keeps its sub-pack at coordPtr for its whole life
    const f = e.coord.f32(e.coordPtr + fnorm.offset, fnorm.size / 4).slice();
    const te = e.coord.f32(e.coordPtr + temb.offset, temb.size / 4).slice();
    await e.gpu.init(f, te);
  },

  /** One pass with the layers on the GPU or in shards; the coordinator embeds and scores. */
  async run(e, requests) {
    const c = e.coord;
    const cfg = trunkConfig(e.header);
    const body = enc.encode(JSON.stringify({ requests }));
    const t0 = now();
    c.withInput(body, (p, l) => c.check(c.x.kevala_prepare(p, l)));
    const table = new Uint32Array(c.out().slice().buffer);
    const { tokens, segs } = parseSegments(table);
    const D = cfg.hidden;
    c.call(() => c.x.kevala_embed());
    const t1 = now();
    if (e.gpu) {
      const x = c.f32(c.x.kevala_x_ptr(), tokens * D).slice();
      const rows = [];
      for (const s of segs) {
        rows.push(s.start);
        for (const m of s.markers) rows.push(s.start + m);
      }
      const uniq = [...new Set(rows)];
      const out = await e.gpu.forward(x, segs, uniq);
      const xv = c.f32(c.x.kevala_x_ptr(), tokens * D);
      uniq.forEach((r, i) => xv.set(out.subarray(i * D, (i + 1) * D), r * D));
    } else {
      await Promise.all(e.shards.map((r) => r.call({ type: "batch", table: table.slice() })));
      const w = e.local.w;
      const [tp] = w.put(new Uint8Array(table.buffer));
      const xptr = w.call(() => w.x.kevala_shard_batch(tp, table.length));
      w.x.kevala_free(tp, table.byteLength);
      for (let s = 0; s < cfg.steps; s++) {
        if (s === 2 * cfg.layers) c.check(c.x.kevala_bridge());
        const x = c.f32(c.x.kevala_x_ptr(), tokens * D);
        const pending = e.shards.map((r) => {
          const copy = x.slice();
          return r.call({ type: "step", s, x: copy }, [copy.buffer]);
        });
        w.f32(xptr, tokens * D).set(x);
        const pp = w.call(() => w.x.kevala_shard_step(s));
        const acc = new Float32Array(w.f32(pp, tokens * D));
        for (const m of await Promise.all(pending)) {
          const p = m.p;
          for (let i = 0; i < acc.length; i++) acc[i] += p[i];
        }
        const xv = c.f32(c.x.kevala_x_ptr(), tokens * D);
        for (let i = 0; i < acc.length; i++) xv[i] += acc[i];
      }
    }
    const t2 = now();
    c.call(() => c.check(c.x.kevala_finish()));
    return { responses: JSON.parse(c.outText()), timing: { prepare: t1 - t0, forward: t2 - t1, total: now() - t0, tokens } };
  },

  /** Converts the upstream checkpoint to a pack inside a throwaway instance. */
  async convert(module, up, { signal, onProgress }) {
    onProgress?.({ phase: "download", file: "tokenizer and configs", loaded: 0, total: 0 });
    const src = await fetchUpstream(up, { signal, onProgress });
    const w = await Wasm.create(module);
    const it = src.chunks();
    // gather the safetensors header
    let buf = new Uint8Array(0);
    const more = async () => {
      const { done, value } = await it.next();
      if (done) return false;
      const n = new Uint8Array(buf.byteLength + value.byteLength);
      n.set(buf);
      n.set(value, buf.byteLength);
      buf = n;
      return true;
    };
    while (buf.byteLength < 8) if (!(await more())) throw new Error("truncated safetensors");
    const hlen = Number(new DataView(buf.buffer).getBigUint64(0, true));
    while (buf.byteLength < 8 + hlen) if (!(await more())) throw new Error("truncated safetensors header");
    const model = JSON.stringify({
      name: "laya",
      source: `https://huggingface.co/${up.repo}`,
      revision: up.revision,
      author: "Nandakishor M, Convai Innovations",
      license: "apache-2.0",
      converter: "kevala (in-browser)",
      quantization: `int8 symmetric absmax, one f32 scale per ${up.block} weights; norms, biases, type embedding, scorer and act head in f32`,
    });
    const args = [buf.subarray(0, 8 + hlen), src.enc, src.agent, src.tok, model].map((d) => w.put(typeof d === "string" ? enc.encode(d) : d));
    w.call(() => w.check(w.x.kevala_convert_plan(...args.flat(), up.block)));
    const plan = JSON.parse(w.outText());
    const jobs = plan.jobs.map(([o, l]) => ({ o: Number(o), l: Number(l) }));
    let ji = 0;
    let jptr = 0;
    let jfill = -1;
    const feed = (chunk, off) => {
      let p = 0;
      while (p < chunk.byteLength && ji < jobs.length) {
        const j = jobs[ji];
        const pos = off + p;
        if (pos < j.o) {
          p += Math.min(chunk.byteLength - p, j.o - pos);
          continue;
        }
        if (jfill < 0) {
          jptr = w.alloc(j.l);
          jfill = 0;
        }
        const n = Math.min(chunk.byteLength - p, j.l - jfill);
        w.bytes(jptr + jfill, n).set(chunk.subarray(p, p + n));
        jfill += n;
        p += n;
        if (jfill === j.l) {
          w.call(() => w.check(w.x.kevala_convert_job(ji, jptr, j.l)));
          w.x.kevala_free(jptr, j.l);
          jfill = -1;
          ji++;
          onProgress?.({ phase: "convert", loaded: ji, total: jobs.length });
        }
      }
    };
    feed(buf, 0);
    let at = buf.byteLength;
    buf = null;
    for (;;) {
      const { done, value } = await it.next();
      if (done) break;
      feed(value, at);
      at += value.byteLength;
    }
    if (ji !== jobs.length) throw new Error(`conversion stopped at tensor ${ji} of ${jobs.length}`);
    const out = new Uint8Array(Number(plan.total));
    out.set(w.bytes(Number(plan.pack), out.byteLength));
    w.x.kevala_convert_drop();
    return out;
  },
};
