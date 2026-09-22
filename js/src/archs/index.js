// The architecture plugin registry of the browser runtime.
//
// A plugin is a module whose default export is:
//
//   {
//     arch: "name",                      // matches the pack's config.arch
//     about: "one line",
//     maxShards(header) -> n,            // WebAssembly shard workers it can split layers across (1 = none)
//     cpuProbe(header) -> config,        // optional: { hidden_size, intermediate_size } for Q8 gated MLP calibration
//     createGpu(gpu, layout, header),    // optional: a GPU trunk with write(dst, bytes) for streamed weights
//     gpuLayouts(header, headerBytes),  // optional: safe whole-tensor placement for large GPU packs
//     initGpu(engine),                   // after the coordinator loads
//     run(engine, requests),             // one pass through the GPU or shard trunk -> { responses, timing }
//     convert(module, spec, opts),       // optional: build a .kevala pack from upstream files in the browser
//   }
//
// A pack family must have a registered plugin before the engine will load it. Load extra plugins
// with `Kevala.load({ plugins: [url] })` before loading packs that use them.

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

export function archNames() {
  return [...ARCHS.keys()];
}

registerArch(laya);
registerArch(kev);
registerArch(gemma4);
