// The architecture plugin registry of the browser runtime.
//
// A plugin is a module whose default export is:
//
//   {
//     arch: "name",                      // matches the pack's config.arch
//     about: "one line",
//     stateCache: true,                  // optional: the family reuses state across requests (Kev, SemIf)
//     maxShards(header) -> n,            // optional: WebAssembly shard workers it can split layers across (1 = none)
//     cpuProbe(header) -> config,        // optional: { hidden_size, intermediate_size } for Q8 gated MLP calibration
//     createGpu(gpu, layout, header),    // optional: a GPU trunk with write(dst, bytes) for streamed weights
//     gpuLayouts(header, headerBytes),  // optional: safe whole-tensor placement for large GPU packs
//     initGpu(engine),                   // after the coordinator loads
//     run(engine, requests),             // one pass through the GPU or shard trunk -> { responses, timing }
//     convert(module, spec, opts),       // optional: build a .kevala pack from upstream files in the browser
//   }
//
// A pack family must have a registered plugin before the engine will load it. A plugin with only
// `arch` runs its family on the CPU in one WebAssembly instance. Load extra plugins with
// `Kevala.load({ plugins: [url] })` before loading packs that use them.

import laya from "./laya.js";
import kev from "./kev.js";
import gemma4 from "./gemma4.js";

const ARCHS = new Map();

export function registerArch(plugin) {
  if (!plugin?.arch) throw new Error("an architecture plugin needs an arch name");
  ARCHS.set(plugin.arch, plugin);
}

export function archPlugin(name) {
  return ARCHS.get(name) || null;
}

/** Validate the family before any coordinator, GPU, or pack-weight allocation happens. */
export function validatePackArchitecture(header, expectedArch = null) {
  const declared = header?.config?.arch;
  const arch = declared == null ? "laya" : declared;
  if (typeof arch !== "string" || !arch) {
    throw Object.assign(new Error(`invalid pack config.arch ${JSON.stringify(declared)}`), { code: "ARCH_UNSUPPORTED" });
  }
  if (expectedArch != null && arch !== expectedArch) {
    throw Object.assign(new Error(`model spec arch ${JSON.stringify(expectedArch)} does not match pack config.arch ${JSON.stringify(arch)}`), { code: "ARCH_MISMATCH" });
  }
  const plugin = archPlugin(arch);
  if (!plugin) {
    throw Object.assign(new Error(`unknown pack architecture ${JSON.stringify(arch)} in config.arch; register an architecture plugin before loading this pack`), { code: "ARCH_UNSUPPORTED" });
  }
  return { arch, plugin };
}

registerArch(laya);
registerArch(kev);
registerArch(gemma4);
