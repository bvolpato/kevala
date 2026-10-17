// Retains the causal text rows before an image, using the original prefix execution policy.
// Dynamic rows still refresh each layer's full image-prefix K/V before question branches run.

export const TEXT_PREFIX_CACHE_LIMIT = 16 * 1024 * 1024;

const positive = (value) => Number.isSafeInteger(value) && value > 0;

/** A cache plan is eligible only when every token-local storage view is aligned. */
export function textPrefixCachePlan({ device, cfg, policy, calibrationPolicy = policy, maxBytes = TEXT_PREFIX_CACHE_LIMIT }, sequences, logicalRows) {
  if (!device?.limits || !cfg || typeof policy !== "string" || !policy || typeof calibrationPolicy !== "string" || !calibrationPolicy || !positive(maxBytes) || maxBytes > TEXT_PREFIX_CACHE_LIMIT) return null;
  if (!Array.isArray(sequences) || sequences.length < 2 || sequences.length > 8 || !positive(logicalRows) || logicalRows > 256 || cfg.causal !== 1) return null;
  if (!positive(cfg.sharedStart) || !positive(cfg.kvSharedLayers) || !positive(cfg.layers) || cfg.sharedStart >= cfg.layers || !Array.isArray(cfg.full) || cfg.full.length !== cfg.layers) return null;
  const first = sequences[0];
  const staticRows = first?.imageSpans?.[0]?.start;
  if (!positive(staticRows) || staticRows >= logicalRows || !(first.ids instanceof Uint32Array) || first.ids.length <= logicalRows) return null;
  for (const sequence of sequences) {
    if (!(sequence?.ids instanceof Uint32Array) || sequence.ids.length <= logicalRows || !Array.isArray(sequence.imageSpans) || !sequence.imageSpans.length || sequence.imageSpans[0].start !== staticRows) return null;
    for (let row = 0; row < logicalRows; row++) if (sequence.ids[row] !== first.ids[row]) return null;
    let previous = staticRows;
    for (const span of sequence.imageSpans) {
      if (!Number.isSafeInteger(span.start) || !positive(span.len) || span.start < previous || span.start + span.len > logicalRows) return null;
      previous = span.start + span.len;
    }
  }
  const alignment = device.limits.minStorageBufferOffsetAlignment;
  const widths = [cfg.hidden, cfg.pleDim, cfg.intermediate, cfg.heads * cfg.localHeadDim, cfg.heads * cfg.globalHeadDim, cfg.kvHeads * cfg.localHeadDim, cfg.kvHeads * cfg.globalHeadDim];
  if (!positive(alignment) || widths.some((width) => !positive(width) || width % 4 || !Number.isSafeInteger(staticRows * width * 4) || staticRows * width * 4 % alignment)) return null;
  const layerWidths = cfg.full.slice(0, cfg.sharedStart).map((full) => cfg.kvHeads * (full ? cfg.globalHeadDim : cfg.localHeadDim));
  const layerBytes = layerWidths.map((width) => Math.max(16, logicalRows * width * 4));
  const bytes = layerBytes.reduce((sum, size) => sum + 2 * size, 0);
  const bufferLimit = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
  if (!positive(bufferLimit) || !Number.isSafeInteger(bytes) || bytes > maxBytes || layerBytes.some((size) => size > bufferLimit)) return null;
  const config = [cfg.hidden, cfg.layers, cfg.heads, cfg.kvHeads, cfg.localHeadDim, cfg.globalHeadDim, cfg.intermediate, cfg.pleDim, cfg.sharedStart, cfg.kvSharedLayers, cfg.window, cfg.eps, cfg.embeddingScale, cfg.pleEmbeddingScale, cfg.pleInputScale, cfg.pleProjectionScale, cfg.ropeLocalTheta, cfg.ropeGlobalTheta, cfg.ropeGlobalFraction, cfg.hostWord, cfg.hostPle, cfg.full];
  const ids = Array.from(first.ids.subarray(0, staticRows));
  const key = JSON.stringify([logicalRows, staticRows, ids, config, policy]);
  const calibrationKey = calibrationPolicy === policy ? key : JSON.stringify([logicalRows, staticRows, ids, config, calibrationPolicy]);
  return Object.freeze({
    device,
    key,
    calibrationKey,
    logicalRows,
    staticRows,
    activeRows: logicalRows - staticRows,
    layerWidths: Object.freeze(layerWidths),
    layerBytes: Object.freeze(layerBytes),
    bytes,
    alignment,
    bindingLimit: device.limits.maxStorageBufferBindingSize,
  });
}

/** Slice only token-local tensors. Attention must retain unsliced full-length K/V. */
export function prefixRowView(plan, buffer, width) {
  if (!plan || !buffer || !positive(width) || width % 4) return null;
  const offset = plan.staticRows * width * 4;
  const size = plan.activeRows * width * 4;
  if (!Number.isSafeInteger(offset) || !positive(size) || offset % plan.alignment || offset + size > buffer.size || size > plan.bindingLimit) return null;
  return { buffer, offset, size };
}

/** One model/device-local retained prefix, committed after every layer and GPU validation pass. */
export class Gemma4TextPrefixCache {
  constructor(device, { maxBytes = TEXT_PREFIX_CACHE_LIMIT } = {}) {
    if (!positive(maxBytes) || maxBytes > TEXT_PREFIX_CACHE_LIMIT) throw new RangeError("Gemma text prefix cache exceeds its memory limit");
    this.device = device;
    this.maxBytes = maxBytes;
    this.entry = null;
    this.active = null;
    this.disposed = false;
    device.lost?.then(() => this.dispose());
  }

  begin(plan) {
    if (this.active) throw new Error("Gemma text prefix cache already has an active request");
    if (this.disposed || !plan || plan.device !== this.device || plan.bytes > this.maxBytes) {
      this.invalidate();
      return null;
    }
    if (this.entry?.key !== plan.key || this.entry?.device !== this.device) {
      this.invalidate();
      const kv = new Map();
      try {
        for (let layer = 0; layer < plan.layerWidths.length; layer++) {
          const size = plan.layerBytes[layer];
          const k = this.device.createBuffer({ label: `Gemma 4 retained prefix K ${layer}`, size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
          let v;
          try { v = this.device.createBuffer({ label: `Gemma 4 retained prefix V ${layer}`, size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }); }
          catch (error) { k.destroy(); throw error; }
          kv.set(layer, { k, v, width: plan.layerWidths[layer] });
        }
      } catch (error) {
        for (const pair of kv.values()) { pair.k.destroy(); pair.v.destroy(); }
        throw error;
      }
      this.entry = { key: plan.key, device: this.device, kv, bytes: plan.bytes, ready: false };
    }
    const lease = { plan, entry: this.entry, kv: this.entry.kv, hit: this.entry.ready, captured: new Set(), invalidated: false };
    this.active = lease;
    return lease;
  }

  recordCapture(lease, layer) {
    if (this.active !== lease || lease.invalidated || !Number.isSafeInteger(layer) || !lease.kv.has(layer)) return false;
    lease.captured.add(layer);
    return true;
  }

  /** Call with success only after readback and the enclosing GPU error scopes complete. */
  finish(lease, success) {
    if (this.active !== lease) return false;
    this.active = null;
    const complete = success === true && !lease.invalidated && this.entry === lease.entry && lease.captured.size === lease.kv.size;
    if (complete) {
      this.entry.ready = true;
      return true;
    }
    if (this.entry === lease.entry) this.entry = null;
    this.destroyEntry(lease.entry);
    return false;
  }

  invalidate() {
    if (this.active) {
      this.active.invalidated = true;
      this.active.entry.ready = false;
    } else this.destroyEntry(this.entry);
    this.entry = null;
  }

  dispose() {
    this.disposed = true;
    this.invalidate();
  }

  destroyEntry(entry) {
    for (const pair of entry?.kv.values() || []) { pair.k.destroy(); pair.v.destroy(); }
  }
}
