// Optional exact Q22 bicubic resize into the vision tower's existing F32 pixels.
import { pipeline, withGpuErrors } from "./gpu.js";
import { gemma4VisionPixelLut } from "./gemma4-image.js";

const MAX_RAW = 4 * 1024 * 1024, MAX_SCRATCH = 8 * 1024 * 1024;

export async function createGemma4ImageGpu(runtime, source) {
  const device = runtime.device;
  const helper = {
    enabled: false, diagnostics: { enabled: false, reason: "pipeline-unavailable" }, buffers: [], geometry: null,
    reset() { for (const buffer of this.buffers) buffer.destroy(); this.buffers = []; this.geometry = null; },
    clear(encoder) { if (this.geometry) { encoder.clearBuffer(this.geometry.rgba); encoder.clearBuffer(this.geometry.horizontal); } },
    encode(encoder, prepared) {
      for (const [kernel, group, work] of [[this.horizontal, prepared.horizontalGroup, prepared.horizontalWork], [this.vertical, prepared.verticalGroup, prepared.verticalWork]]) {
        const pass = encoder.beginComputePass(); pass.setPipeline(kernel); pass.setBindGroup(0, group); pass.dispatchWorkgroups(work); pass.end();
      }
    },
    async prepare(raw, output) {
      if (!this.enabled || !raw) return null;
      const horizontalBytes = Math.ceil(raw.width * raw.inputHeight * 3 / 4) * 4;
      const tables = [raw.xs.metadata, raw.xs.weights, raw.ys.metadata, raw.ys.weights];
      const sizes = [raw.rgba.byteLength, horizontalBytes, ...tables.map((table) => table.byteLength), 1024, 32].map((bytes) => Math.max(16, bytes));
      const totalBytes = sizes.reduce((sum, bytes) => sum + Math.max(16, bytes), 0), limit = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
      if (raw.rgba.byteLength > MAX_RAW || totalBytes > MAX_SCRATCH || sizes.some((bytes) => !Number.isSafeInteger(bytes) || bytes < 1 || bytes > limit) || raw.width * raw.height * 12 > output.size) {
        this.diagnostics.lastFallback = "geometry-unsupported"; return null;
      }
      const key = [raw.inputWidth, raw.inputHeight, raw.width, raw.height].join(":");
      const cached = this.geometry;
      if (cached?.key === key && cached.output === output && cached.verticalGroup) {
        device.queue.writeBuffer(cached.rgba, 0, raw.rgba);
        this.diagnostics = { enabled: true, reason: "ready", scratchBytes: totalBytes };
        return { geometry: cached, verticalGroup: cached.verticalGroup, horizontalGroup: cached.horizontalGroup, horizontalWork: cached.horizontalWork, verticalWork: cached.verticalWork, totalBytes };
      }
      const owned = [];
      try {
        return await withGpuErrors(runtime, () => {
          const U = GPUBufferUsage;
          const make = (size, usage, data, label) => {
            const buffer = device.createBuffer({ size: Math.max(16, size), usage, label: `Gemma image prep ${label}` }); owned.push(buffer);
            if (data) device.queue.writeBuffer(buffer, 0, data);
            return buffer;
          };
          const bind = (kernel, buffers) => device.createBindGroup({ layout: kernel.getBindGroupLayout(0), entries: buffers.map(([binding, buffer]) => ({ binding, resource: { buffer } })) });
          let geometry = this.geometry;
          if (!geometry || geometry.key !== key) {
            const rgba = make(sizes[0], U.STORAGE | U.COPY_DST, null, "RGBA");
            const horizontal = make(horizontalBytes, U.STORAGE | U.COPY_DST, null, "horizontal RGB");
            const [xm, xw, ym, yw] = tables.map((table, i) => make(table.byteLength, U.STORAGE | U.COPY_DST, table, `coefficients${i}`));
            const lut = make(1024, U.STORAGE | U.COPY_DST, gemma4VisionPixelLut(), "normalization LUT");
            const params = make(32, U.UNIFORM | U.COPY_DST, new Uint32Array([raw.inputWidth, raw.inputHeight, raw.width, raw.height, raw.width * raw.height * 3, raw.width / 16, raw.width * raw.inputHeight * 3, 0]), "parameters");
            geometry = { key, rgba, horizontal, params, ym, yw, lut, output: null, horizontalGroup: bind(this.horizontal, [[0, params], [1, rgba], [2, xm], [3, xw], [4, horizontal]]), horizontalWork: Math.ceil(horizontalBytes / 4 / 128), verticalWork: Math.ceil(raw.width * raw.height * 3 / 128) };
          }
          const verticalGroup = geometry.output === output ? geometry.verticalGroup : bind(this.vertical, [[0, geometry.params], [5, geometry.ym], [6, geometry.yw], [7, geometry.horizontal], [8, geometry.lut], [9, output]]);
          device.queue.writeBuffer(geometry.rgba, 0, raw.rgba);
          return { geometry, verticalGroup, horizontalGroup: geometry.horizontalGroup, horizontalWork: geometry.horizontalWork, verticalWork: geometry.verticalWork, totalBytes, owned };
        }, "preparing optional Gemma GPU image pixels").then((prepared) => {
          if (owned.length) { this.reset(); this.buffers = owned; this.geometry = prepared.geometry; }
          this.geometry.output = output; this.geometry.verticalGroup = prepared.verticalGroup;
          this.diagnostics = { enabled: true, reason: "ready", scratchBytes: prepared.totalBytes };
          return prepared;
        });
      } catch (error) {
        for (const buffer of owned) buffer.destroy();
        if (runtime.lost || /device lost/i.test(String(error?.message || error))) throw error;
        this.reset(); this.enabled = false;
        this.diagnostics = { enabled: false, reason: "setup-failed", error: String(error?.message || error) };
        return null;
      }
    },
  };
  try {
    const compiled = await withGpuErrors(runtime, async () => ({
      horizontal: await pipeline(device, source("gemma4_image_resize_horizontal"), "gemma4_image_resize_horizontal"),
      vertical: await pipeline(device, source("gemma4_image_resize_vertical"), "gemma4_image_resize_vertical"),
    }), "compiling optional Gemma GPU image preparation");
    Object.assign(helper, compiled, { enabled: true, diagnostics: { enabled: true, reason: "compiled" } });
  } catch (error) {
    if (runtime.lost || /device lost/i.test(String(error?.message || error))) throw error;
    helper.diagnostics = { enabled: false, reason: "pipeline-failed", error: String(error?.message || error) };
  }
  return helper;
}
