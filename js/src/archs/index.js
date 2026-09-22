// The architecture plugin registry of the browser runtime.
//
// A plugin is a module whose default export is:
//
//   {
//     arch: "name",                      // matches the pack's config.arch
//     about: "one line",
//     maxShards(header) -> n,            // WebAssembly shard workers it can split layers across (1 = none)
//     createGpu(gpu, layout, header),    // optional: a GPU trunk with write(dst, bytes) for streamed weights
//     initGpu(engine),                   // after the coordinator loads
//     run(engine, requests),             // one pass through the GPU or shard trunk -> { responses, timing }
//     convert(module, spec, opts),       // optional: build a .kevala pack from upstream files in the browser
//   }
//
// Families the WebAssembly build knows but no plugin describes still run, on the CPU, through
// the generic `kevala_decide` path. Load extra plugins with `Kevala.load({ plugins: [url] })`.

import laya from "./laya.js";
import kev from "./kev.js";

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
