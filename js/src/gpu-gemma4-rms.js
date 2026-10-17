// Load-time exactness gate for Gemma 4's two adjacent RMS fusions.
import { bufferUsage, inErrorScopes, scratchBuffer } from "./gpu.js";
import { mapBeforeDeadline } from "./gpu-matvec.js";

const TOKENS = 2;
const DEADLINE_MS = 750;
const MAX_SCRATCH_BYTES = 16 * 1024 * 1024;
const now = () => (typeof performance === "undefined" ? Date.now() : performance.now());

function failure(reason, startedAt, extra = {}) {
  return {
    enabled: false,
    diagnostics: {
      status: "disabled",
      ok: false,
      reason,
      elapsedMs: now() - startedAt,
      ...extra,
    },
  };
}

function errorDetails(error) {
  return {
    message: String(error?.message || error || "unknown WebGPU error"),
    ...(typeof error?.code === "string" ? { code: error.code } : {}),
  };
}

function tensorEntry(weights, name) {
  if (weights?.tensors instanceof Map) return weights.tensors.get(name);
  if (weights instanceof Map) return weights.get(name);
  if (typeof weights?.get === "function") return weights.get(name);
  return weights?.[name];
}

function validateWeight(weights, name, elements) {
  const entry = tensorEntry(weights, name);
  const info = entry?.info;
  if (!entry?.buf || !info) throw new Error(`loaded RMS weight is unavailable: ${name}`);
  if (info.dtype && info.dtype !== "f32") throw new Error(`RMS weight ${name} has unsupported dtype ${info.dtype}`);
  if (entry.filled !== undefined && entry.need !== undefined && entry.filled !== entry.need) {
    throw new Error(`RMS weight is incomplete: ${name}`);
  }
  const shape = Array.isArray(info.shape) ? info.shape.map(Number) : [];
  const count = shape.length && shape.every((value) => Number.isSafeInteger(value) && value > 0)
    ? shape.reduce((product, value) => product * value, 1)
    : 1;
  if (count !== elements || entry.buf.size < elements * 4) {
    throw new Error(`RMS weight ${name} has ${count} elements; expected ${elements}`);
  }
  return entry.buf;
}

function makeInput(hidden) {
  const branch = new Float32Array(TOKENS * hidden);
  const residual = new Float32Array(TOKENS * hidden);
  for (let i = 0; i < branch.length; i++) {
    branch[i] = Math.fround((((i * 37 + 11) % 101) - 50) * 0.00437);
    residual[i] = Math.fround((((i * 53 + 7) % 127) - 63) * 0.0713);
  }
  return { branch, residual };
}

function makeUniform(device, values, label, usage) {
  const buffer = scratchBuffer(device, 16, usage.UNIFORM | usage.COPY_DST, label);
  const bytes = new ArrayBuffer(16);
  const view = new DataView(bytes);
  view.setUint32(0, values.D, true);
  view.setFloat32(4, values.eps, true);
  view.setUint32(8, values.mode, true);
  try {
    device.queue.writeBuffer(buffer, 0, bytes);
    return buffer;
  } catch (error) {
    buffer.destroy();
    throw error;
  }
}

function makeUintUniform(device, values, label, usage) {
  const buffer = scratchBuffer(device, 16, usage.UNIFORM | usage.COPY_DST, label);
  try {
    device.queue.writeBuffer(buffer, 0, new Uint32Array(values));
    return buffer;
  } catch (error) {
    buffer.destroy();
    throw error;
  }
}

function makeGlobals(device, usage) {
  const buffer = scratchBuffer(device, 16, usage.UNIFORM | usage.COPY_DST, "Gemma RMS guard globals");
  try {
    device.queue.writeBuffer(buffer, 0, new Uint32Array([TOKENS, 0, 0, 0]));
    return buffer;
  } catch (error) {
    buffer.destroy();
    throw error;
  }
}

function makeBindGroup(device, pipeline, buffers, label) {
  return device.createBindGroup({
    label,
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
}

function encodeDispatch(encoder, pipeline, bindGroup) {
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(TOKENS);
  pass.end();
}

function bitString(bits) {
  return `0x${bits.toString(16).padStart(8, "0")}`;
}

function compare(mapped, offset, length, layer, comparison) {
  const floats = new Float32Array(mapped, offset, length * 2);
  const bits = new Uint32Array(mapped, offset, length * 2);
  let finite = true;
  for (let i = 0; i < length; i++) {
    if (!Number.isFinite(floats[i])) finite = false;
    if (!Number.isFinite(floats[length + i])) finite = false;
    if (bits[i] !== bits[length + i]) {
      return {
        ok: false,
        mismatch: {
          layer,
          comparison,
          index: i,
          expected: floats[i],
          actual: floats[length + i],
          expectedBits: bitString(bits[i]),
          actualBits: bitString(bits[length + i]),
          finite,
        },
      };
    }
  }
  return { ok: finite, mismatch: finite ? null : { layer, comparison, reason: "non-finite output" } };
}

/**
 * Compares the production RMS fusion pipelines against their unfused sequences using
 * every layer's loaded checkpoint weights. A mismatch or inability to check fails
 * closed for this device, without failing model load.
 */
async function validateExistingGemma4Rms(device, pipelines, weights, cfg, options = {}) {
  const startedAt = now();
  const enabled = options?.enabled ?? true;
  if (!enabled) return failure("disabled by caller", startedAt);

  let readback = null;
  let buffers = [];
  let checkedLayers = 0;
  let scratchBytes = 0;
  try {
    const hidden = Number(cfg?.hidden);
    const layers = Number(cfg?.layers);
    const eps = Math.fround(Number(cfg?.eps));
    if (!Number.isSafeInteger(hidden) || hidden < 1 || !Number.isSafeInteger(layers) || layers < 1 || !Number.isFinite(eps) || eps <= 0) {
      return failure("invalid Gemma RMS config", startedAt);
    }

    const rms = pipelines?.RMS || pipelines?.rms;
    const residualNorm = pipelines?.RMS_RESIDUAL_NORM || pipelines?.rmsResidualNorm;
    const rmsScale = pipelines?.RMS_SCALE || pipelines?.rmsScale;
    const residualPipeline = pipelines?.RESIDUAL || pipelines?.residual;
    if (![rms, residualNorm, rmsScale, residualPipeline].every((pipeline) => pipeline?.getBindGroupLayout)) {
      return failure("production RMS pipelines are unavailable", startedAt);
    }

    const layerWeights = [];
    for (let layer = 0; layer < layers; layer++) {
      layerWeights.push({
        post: validateWeight(weights, `l.${layer}.attn_post_norm`, hidden),
        next: validateWeight(weights, `l.${layer}.ffn_norm`, hidden),
        ple: validateWeight(weights, `l.${layer}.ple_norm`, hidden),
        scalar: validateWeight(weights, `l.${layer}.scalar`, 1),
      });
    }

    const outputBytes = TOKENS * hidden * Float32Array.BYTES_PER_ELEMENT;
    const readbackBytes = layers * outputBytes * 6;
    const bufferBytes = Math.max(16, outputBytes);
    scratchBytes = readbackBytes + bufferBytes * 6 + 80;
    const limits = device?.limits || {};
    if (!Number.isSafeInteger(readbackBytes) || scratchBytes > MAX_SCRATCH_BYTES) {
      return failure("RMS guard scratch limit exceeded", startedAt, { hidden, layers, scratchBytes, maxScratchBytes: MAX_SCRATCH_BYTES });
    }
    if (readbackBytes > (limits.maxBufferSize || Infinity) || outputBytes > (limits.maxStorageBufferBindingSize || Infinity)) {
      return failure("RMS guard exceeds device buffer limits", startedAt, { hidden, layers, scratchBytes });
    }

    let mapped = null;
    await inErrorScopes(device, async () => {
      const usage = bufferUsage();
      const own = (buffer) => (buffers.push(buffer), buffer);
      const values = makeInput(hidden);
      const storage = (data, flags, label, size = data?.byteLength || 16) => {
        const buffer = own(scratchBuffer(device, size, usage.STORAGE | flags | (data ? usage.COPY_DST : 0), label));
        if (data) device.queue.writeBuffer(buffer, 0, data);
        return buffer;
      };
      const globals = own(makeGlobals(device, usage));
      const mode1 = own(makeUniform(device, { D: hidden, eps, mode: 1 }, "Gemma RMS guard mode 1", usage));
      const mode2 = own(makeUniform(device, { D: hidden, eps, mode: 2 }, "Gemma RMS guard mode 2", usage));
      const scaleParams = own(makeUintUniform(device, [hidden, 1, 0, 0], "Gemma RMS guard residual scale", usage));
      const branch = storage(values.branch, 0, "Gemma RMS guard branch");
      const initialResidual = storage(values.residual, usage.COPY_SRC, "Gemma RMS guard initial residual");
      const oldY = storage(null, usage.COPY_SRC | usage.COPY_DST, "Gemma RMS guard baseline residual", outputBytes);
      const newY = storage(null, usage.COPY_SRC | usage.COPY_DST, "Gemma RMS guard fused residual", outputBytes);
      const oldH = storage(null, usage.COPY_SRC, "Gemma RMS guard baseline output", outputBytes);
      const newH = storage(null, usage.COPY_SRC, "Gemma RMS guard fused output", outputBytes);
      const zero = storage(new Uint32Array([0]), 0, "Gemma RMS guard zero");
      readback = own(scratchBuffer(device, readbackBytes, usage.MAP_READ | usage.COPY_DST, "Gemma RMS guard readback"));

      const encoder = device.createCommandEncoder({ label: "Gemma RMS exactness gate" });
      let offset = 0;
      const snapshot = (source) => {
        encoder.copyBufferToBuffer(source, 0, readback, offset, outputBytes);
        offset += outputBytes;
      };
      const copyInitial = (destination) => encoder.copyBufferToBuffer(initialResidual, 0, destination, 0, outputBytes);

      for (let layer = 0; layer < layers; layer++) {
        const weightsForLayer = layerWeights[layer];
        const rmsMode2 = makeBindGroup(device, rms, [globals, mode2, branch, weightsForLayer.post, oldY], `rms.guard.${layer}.baseline-post`);
        const rmsMode1 = makeBindGroup(device, rms, [globals, mode1, oldY, weightsForLayer.next, oldH], `rms.guard.${layer}.baseline-next`);
        const fusedNorm = makeBindGroup(device, residualNorm, [globals, mode2, branch, weightsForLayer.post, newY, weightsForLayer.next, newH], `rms.guard.${layer}.fused-next`);
        const scaleRms = makeBindGroup(device, rms, [globals, mode2, branch, weightsForLayer.ple, oldY], `rms.guard.${layer}.baseline-ple`);
        const scaleResidual = makeBindGroup(device, residualPipeline, [globals, scaleParams, oldY, zero, weightsForLayer.scalar], `rms.guard.${layer}.baseline-scale`);
        const fusedScale = makeBindGroup(device, rmsScale, [globals, mode2, branch, weightsForLayer.ple, newY, weightsForLayer.scalar], `rms.guard.${layer}.fused-scale`);

        copyInitial(oldY);
        copyInitial(newY);
        encodeDispatch(encoder, rms, rmsMode2);
        encodeDispatch(encoder, rms, rmsMode1);
        encodeDispatch(encoder, residualNorm, fusedNorm);
        snapshot(oldY);
        snapshot(newY);
        snapshot(oldH);
        snapshot(newH);

        copyInitial(oldY);
        copyInitial(newY);
        encodeDispatch(encoder, rms, scaleRms);
        encodeDispatch(encoder, residualPipeline, scaleResidual);
        encodeDispatch(encoder, rmsScale, fusedScale);
        snapshot(oldY);
        snapshot(newY);
        checkedLayers++;
      }

      if (offset !== readbackBytes) throw new Error("RMS guard readback layout mismatch");
      const deadline = startedAt + DEADLINE_MS;
      device.queue.submit([encoder.finish()]);
      if (!await mapBeforeDeadline(readback, readbackBytes, deadline)) throw new Error("RMS exactness gate deadline exceeded");
      mapped = readback.getMappedRange(0, readbackBytes);
    });

    try {
      let cursor = 0;
      for (let layer = 0; layer < layers; layer++) {
        for (const comparison of ["residual", "nextNorm", "pleScale"]) {
          const result = compare(mapped, cursor, outputBytes / 4, layer, comparison);
          if (!result.ok) {
            return failure("RMS fusion is not bit-exact with loaded weights", startedAt, {
              hidden, layers, checkedLayers, scratchBytes, mismatch: result.mismatch,
            });
          }
          cursor += outputBytes * 2;
        }
      }
    } finally {
      readback.unmap();
    }

    if (now() - startedAt > DEADLINE_MS) {
      return failure("RMS exactness gate deadline exceeded", startedAt, { hidden, layers, checkedLayers, scratchBytes });
    }

    return {
      enabled: true,
      diagnostics: {
        status: "passed",
        ok: true,
        reason: null,
        hidden,
        tokens: TOKENS,
        layers,
        layersChecked: checkedLayers,
        comparisons: layers * 3,
        valuesCompared: layers * 3 * TOKENS * hidden,
        bitExact: true,
        tolerance: 0,
        scratchBytes,
        elapsedMs: now() - startedAt,
      },
    };
  } catch (error) {
    return failure("RMS exactness gate could not complete", startedAt, {
      checkedLayers,
      scratchBytes,
      ...errorDetails(error),
    });
  } finally {
    if (readback?.mapState === "mapped") readback.unmap();
    for (const buffer of buffers) {
      try { buffer.destroy(); } catch { /* best-effort cleanup after device loss */ }
    }
  }
}

async function validateScaleNext(device, pipelines, weights, cfg, candidate, startedAt) {
  const deadline = startedAt + DEADLINE_MS;
  if (now() >= deadline) return failure("RMS scale/next guard deadline exceeded", startedAt);
  const hidden = Number(cfg.hidden), layers = Number(cfg.layers), eps = Math.fround(Number(cfg.eps));
  const outputValues = TOKENS * hidden, tail = new Float32Array([12345.25, -1731.5, 913.75, -0]);
  const outputBytes = (outputValues + tail.length) * 4, readbackBytes = layers * outputBytes * 4;
  const scratchBytes = readbackBytes + Math.max(16, outputValues * 4) + outputBytes * 6 + 48;
  const limits = device.limits || {};
  if (!Number.isSafeInteger(readbackBytes) || scratchBytes > MAX_SCRATCH_BYTES) {
    return failure("RMS scale/next guard scratch limit exceeded", startedAt, { hidden, layers, scratchBytes, maxScratchBytes: MAX_SCRATCH_BYTES });
  }
  if (readbackBytes > (limits.maxBufferSize || Infinity) || outputBytes > (limits.maxStorageBufferBindingSize || Infinity)) {
    return failure("RMS scale/next guard exceeds device buffer limits", startedAt, { hidden, layers, scratchBytes });
  }
  const owned = [];
  let readback, mapped = false, checkedLayers = 0;
  try {
    const rms = pipelines.RMS || pipelines.rms, rmsScale = pipelines.RMS_SCALE || pipelines.rmsScale;
    if (![rms, rmsScale, candidate].every((value) => value?.getBindGroupLayout)) return failure("RMS scale/next pipelines are unavailable", startedAt);
    const layerWeights = [];
    for (let layer = 0; layer < layers; layer++) {
      const nextName = layer + 1 < layers ? `l.${layer + 1}.attn_norm` : "norm";
      layerWeights.push({ ple: validateWeight(weights, `l.${layer}.ple_norm`, hidden), scalar: validateWeight(weights, `l.${layer}.scalar`, 1), next: validateWeight(weights, nextName, hidden), nextName });
    }
    await inErrorScopes(device, async () => {
      const usage = bufferUsage(), own = (buffer) => (owned.push(buffer), buffer);
      const storage = (data, flags, label, size = data?.byteLength || outputBytes) => {
        const buffer = own(scratchBuffer(device, size, usage.STORAGE | flags | (data ? usage.COPY_DST : 0), label));
        if (data) device.queue.writeBuffer(buffer, 0, data);
        return buffer;
      };
      const values = makeInput(hidden), initial = new Float32Array(outputValues + tail.length), poison = new Float32Array(initial.length).fill(NaN);
      // One signed, non-power-of-two row and one zero row exercise actual weights and scalar signs.
      for (let i = hidden; i < outputValues; i++) { values.branch[i] = i % 2 ? -0 : 0; values.residual[i] = i % 3 ? 0 : -0; }
      initial.set(values.residual); initial.set(tail, outputValues); poison.set(tail, outputValues);
      const globals = own(makeGlobals(device, usage));
      const mode1 = own(makeUniform(device, { D: hidden, eps, mode: 1 }, "Gemma RMS scale/next guard mode 1", usage));
      const mode2 = own(makeUniform(device, { D: hidden, eps, mode: 2 }, "Gemma RMS scale/next guard mode 2", usage));
      const branch = storage(values.branch, 0, "Gemma RMS scale/next guard branch");
      const seed = storage(initial, usage.COPY_SRC, "Gemma RMS scale/next guard initial residual");
      const poisonH = storage(poison, usage.COPY_SRC, "Gemma RMS scale/next guard poisoned output");
      const oldY = storage(null, usage.COPY_SRC | usage.COPY_DST, "Gemma RMS scale/next baseline residual");
      const newY = storage(null, usage.COPY_SRC | usage.COPY_DST, "Gemma RMS scale/next fused residual");
      const oldH = storage(null, usage.COPY_SRC | usage.COPY_DST, "Gemma RMS scale/next baseline norm");
      const newH = storage(null, usage.COPY_SRC | usage.COPY_DST, "Gemma RMS scale/next fused norm");
      readback = own(scratchBuffer(device, readbackBytes, usage.MAP_READ | usage.COPY_DST, "Gemma RMS scale/next guard readback"));
      const encoder = device.createCommandEncoder({ label: "Gemma RMS scale/next exactness gate" });
      let offset = 0;
      for (let layer = 0; layer < layers; layer++) {
        if (now() >= deadline) throw new Error("RMS scale/next guard deadline exceeded");
        const loaded = layerWeights[layer];
        const scaleGroup = makeBindGroup(device, rmsScale, [globals, mode2, branch, loaded.ple, oldY, loaded.scalar], `rms.next.guard.${layer}.baseline-scale`);
        const nextGroup = makeBindGroup(device, rms, [globals, mode1, oldY, loaded.next, oldH], `rms.next.guard.${layer}.baseline-next`);
        const fusedGroup = makeBindGroup(device, candidate, [globals, mode2, branch, loaded.ple, newY, loaded.scalar, loaded.next, newH], `rms.next.guard.${layer}.fused`);
        for (const output of [oldY, newY]) encoder.copyBufferToBuffer(seed, 0, output, 0, outputBytes);
        for (const output of [oldH, newH]) encoder.copyBufferToBuffer(poisonH, 0, output, 0, outputBytes);
        const pass = encoder.beginComputePass();
        pass.setPipeline(rmsScale); pass.setBindGroup(0, scaleGroup); pass.dispatchWorkgroups(TOKENS);
        pass.setPipeline(rms); pass.setBindGroup(0, nextGroup); pass.dispatchWorkgroups(TOKENS); pass.end();
        encodeDispatch(encoder, candidate, fusedGroup);
        for (const output of [oldY, newY, oldH, newH]) { encoder.copyBufferToBuffer(output, 0, readback, offset, outputBytes); offset += outputBytes; }
        checkedLayers++;
      }
      if (offset !== readbackBytes) throw new Error("RMS scale/next guard readback layout mismatch");
      device.queue.submit([encoder.finish()]);
      const withinDeadline = await mapBeforeDeadline(readback, readbackBytes, deadline);
      mapped = true;
      if (!withinDeadline) throw new Error("RMS scale/next guard deadline exceeded");
      const actual = readback.getMappedRange(0, readbackBytes), expectedTail = new Uint32Array(tail.buffer);
      let cursor = 0;
      for (let layer = 0; layer < layers; layer++) for (const comparison of ["scaledResidual", "nextNorm"]) {
        for (let side = 0; side < 2; side++) {
          const actualTail = new Uint32Array(actual, cursor + side * outputBytes + outputValues * 4, tail.length);
          for (let i = 0; i < tail.length; i++) if (actualTail[i] !== expectedTail[i]) {
            throw Object.assign(new Error("RMS scale/next guard overwrote an output tail"), { mismatch: { layer, comparison, side: side === 0 ? "baseline" : "candidate", index: outputValues + i, expectedBits: bitString(expectedTail[i]), actualBits: bitString(actualTail[i]) } });
          }
        }
        const check = compare(actual, cursor, outputBytes / 4, layer, comparison);
        if (!check.ok) throw Object.assign(new Error("RMS scale/next fusion is not bit-exact with loaded weights"), { mismatch: { ...check.mismatch, nextWeight: layerWeights[layer].nextName } });
        cursor += outputBytes * 2;
      }
      readback.unmap(); mapped = false;
    });
    if (now() >= deadline) return failure("RMS scale/next guard deadline exceeded", startedAt, { hidden, layers, checkedLayers, scratchBytes });
    return { enabled: true, diagnostics: { status: "passed", ok: true, reason: null, hidden, tokens: TOKENS, layers, layersChecked: checkedLayers,
      comparisons: layers * 2, valuesCompared: layers * 2 * outputValues, bitExact: true, tolerance: 0, outputTailsChecked: true, scratchBytes, elapsedMs: now() - startedAt } };
  } catch (error) {
    return failure("RMS scale/next exactness gate could not complete", startedAt, { checkedLayers, scratchBytes, ...errorDetails(error), ...(error?.mismatch ? { mismatch: error.mismatch } : {}) });
  } finally {
    if (mapped || readback?.mapState === "mapped") readback?.unmap();
    for (const buffer of owned) { try { buffer.destroy(); } catch { /* best-effort cleanup after device loss */ } }
  }
}

/** The optional cross-layer gate cannot disable the two existing, independently validated fusions. */
export async function validateGemma4Rms(device, pipelines, weights, cfg, options = {}) {
  const startedAt = now();
  const existing = await validateExistingGemma4Rms(device, pipelines, weights, cfg, options);
  const candidate = pipelines?.RMS_SCALE_NEXT || pipelines?.rmsScaleNext;
  if (!candidate) return existing;
  const next = existing.enabled
    ? await validateScaleNext(device, pipelines, weights, cfg, candidate, startedAt)
    : failure("existing RMS fusion gate is disabled", startedAt);
  return { ...existing, nextEnabled: next.enabled, nextDiagnostics: next.diagnostics };
}
