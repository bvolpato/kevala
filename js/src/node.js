// kevala in Node.js (or Deno, Bun): the same WebAssembly engine, one instance, no workers, no GPU.
//
//   import { loadFile } from "kevala/node";
//   const kevala = await loadFile("laya-q8.kevala");
//   const r = kevala.decide("Refund me or I cancel.", { churn: { type: "noul", instructions: "Will they leave?" } });
//
// Packs come from the common `kevala convert` path (family aliases remain compatible), or from a
// browser's converted cache.

import { readFile, stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { CPU_KERNELS, cpuOptions } from "./cpu-policy.js";
import { Wasm, wasmFlavor } from "./wasm.js";
import { assertWasmPackSize } from "./pack-layout.js";

export async function loadFile(path, { flavor, cpuKernel = "auto", threads = "auto" } = {}) {
  cpuOptions({ cpuKernel, threads });
  if (threads !== "auto" && threads !== 1) throw new Error("kevala/node supports one CPU thread");
  assertWasmPackSize((await stat(path)).size);
  const f = wasmFlavor(flavor);
  const module = await WebAssembly.compile(await readFile(fileURLToPath(new URL(`kevala-${f}.wasm`, import.meta.url))));
  const w = await Wasm.create(module, cpuKernel === "auto" ? undefined : CPU_KERNELS.indexOf(cpuKernel));
  const tile = w.tile;
  // the engine takes ownership of the pack bytes: they are not freed here
  const [ptr, length] = w.put(await readFile(path));
  w.call(() => w.check(w.x.kevala_engine_load(ptr, length)));
  const meta = JSON.parse(w.outText());
  const run = (requests) => w.withInput(JSON.stringify({ requests }), (p, l) => (w.check(w.x.kevala_decide(p, l)), JSON.parse(w.outText())));
  return {
    info: { ...meta, flavor: f, threads: 1, cpuTiles: [tile], cpuTuning: { kernel: CPU_KERNELS[tile], threads: 1, kernelSource: cpuKernel === "auto" ? "measured" : "override", threadSource: "model-limit" } },
    decide: (state, questions, { parts } = {}) => run([{ state, questions, ...(parts ? { parts } : {}) }])[0],
    decideMany: (items) => run(items),
  };
}
