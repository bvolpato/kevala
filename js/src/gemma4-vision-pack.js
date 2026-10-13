import { openPack, PieceSink } from "./source.js";
import { assertPackPrecision, gpuLayouts, packSize, readPackHead } from "./pack-layout.js";
import { GpuGemma4Vision, gemma4VisionConfig } from "./gpu-gemma4-vision.js";

export function validateVisionCompanion(text, vision) {
  const c = vision.config;
  if (c?.arch !== "gemma4-vision") throw new Error("expected a gemma4-vision companion pack");
  assertPackPrecision(vision, "bf16");
  const cfg = gemma4VisionConfig(vision);
  if (cfg.textHidden !== text.config.hidden_size) throw new Error("vision companion hidden size does not match the decoder");
  if (!text.model?.source || !text.model?.revision || text.model.source !== vision.model?.source || text.model.revision !== vision.model?.revision) throw new Error("vision companion must use the same source and pinned revision as the decoder");
  if (c.image_token_id !== 258880 || c.boi_token_id !== 255999 || c.eoi_token_id !== 258882 || c.pad_token_id !== 0) throw new Error("vision companion has unsupported image token IDs");
  if (text.config.use_bidirectional_attention != null && text.config.use_bidirectional_attention !== false) throw new Error("this image path currently requires a causal Gemma 4 decoder");
  if (cfg.patch !== 16 || cfg.pool !== 3) throw new Error("vision companion requires an unsupported image processor");
  return cfg;
}

/** The companion shares the decoder device and is released with that device. */
export async function loadGemma4Vision(engine, options, gpu, signal, progress) {
  const pack = await openPack(options.vision, { cache: options.cache !== false, signal, onProgress: (p) => progress({ ...p, message: p.message || "loading image companion" }) });
  const iterator = pack.chunks();
  try {
    const { head, header, headerBytes } = await readPackHead(iterator);
    const cfg = validateVisionCompanion(engine.header, header);
    const size = packSize(header, headerBytes.byteLength);
    if (size > 1024 ** 3) throw new Error("vision companion exceeds the 1 GiB limit");
    const layout = gpuLayouts(header, headerBytes.byteLength, () => true)[1];
    const vision = new GpuGemma4Vision(gpu, layout, header, { submitMode: options.submit || "split" });
    const imageTokens = options.imageTokens ?? 70;
    const patches = imageTokens * cfg.pool * cfg.pool;
    try { vision.checkBuffer(patches * patches * cfg.heads * 4, "image attention scores"); }
    catch (error) { throw Object.assign(error, { code: "IMAGE_BUDGET" }); }
    const sink = new PieceSink(layout, (dst, bytes) => vision.write(dst, bytes));
    let at = 0;
    let pending = 0;
    const feed = (bytes) => { sink.push(bytes, at); at += bytes.byteLength; pending += bytes.byteLength; };
    feed(head);
    for (;;) {
      signal.throwIfAborted();
      const { value, done } = await iterator.next();
      if (done) break;
      feed(value);
      if (pending >= 64 * 1024 ** 2) { await gpu.device.queue.onSubmittedWorkDone(); pending = 0; }
    }
    if (at < size) throw new Error(`vision companion is truncated: ${at} of ${size} bytes`);
    await gpu.device.queue.onSubmittedWorkDone();
    if (pack.saving) await pack.saving;
    signal.throwIfAborted();
    await vision.init();
    engine.vision = vision;
    engine.visionHeader = header;
    engine.imageTokens = imageTokens;
    engine.visionPack = { bytes: size, cached: pack.cached, model: header.model, precision: "bf16" };
  } finally { await iterator.return?.(); }
}
