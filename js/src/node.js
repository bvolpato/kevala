// kevala in Node.js (or Deno, Bun): the same WebAssembly engine, one instance, no workers, no GPU.
//
//   import { loadFile } from "kevala/node";
//   const kevala = await loadFile("laya-q8.kevala");
//   const r = kevala.decide("Refund me or I cancel.", { churn: { type: "noul", instructions: "Will they leave?" } });
//
// Packs come from `kevala convert` / `kevala convert-kev`, or from a browser's converted cache.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { wasmFlavor } from "./wasm.js";

const enc = new TextEncoder();
const dec = new TextDecoder();

export async function loadFile(path, { flavor } = {}) {
  const f = wasmFlavor(flavor);
  const wasm = await readFile(fileURLToPath(new URL(`kevala-${f}.wasm`, import.meta.url)));
  const { instance } = await WebAssembly.instantiate(wasm, {});
  const x = instance.exports;
  x.kevala_init();
  // the 4x4 CPU tile needs 32 vector registers (ARM): time both and keep the faster
  const time = (t) => {
    x.kevala_set_tile(t);
    // warm up first: timings taken before the engine's optimizing tier kicks in mislead
    for (let i = 0; i < 4; i++) x.kevala_tile_probe();
    const t0 = performance.now();
    for (let i = 0; i < 2; i++) x.kevala_tile_probe();
    return performance.now() - t0;
  };
  time(0);
  x.kevala_set_tile(time(1) < time(0) ? 1 : 0);
  const mem = () => new Uint8Array(x.memory.buffer);
  const text = (p, n) => dec.decode(mem().subarray(p, p + n));
  const check = (rc) => {
    if (rc) throw new Error(text(x.kevala_error_ptr(), x.kevala_error_len()));
  };
  const bytes = await readFile(path);
  const ptr = x.kevala_alloc(bytes.length);
  mem().set(bytes, ptr);
  check(x.kevala_engine_load(ptr, bytes.length));
  const meta = JSON.parse(text(x.kevala_out_ptr(), x.kevala_out_len()));
  const run = (requests) => {
    const b = enc.encode(JSON.stringify({ requests }));
    const p = x.kevala_alloc(b.length);
    mem().set(b, p);
    try {
      check(x.kevala_decide(p, b.length));
    } finally {
      x.kevala_free(p, b.length);
    }
    return JSON.parse(text(x.kevala_out_ptr(), x.kevala_out_len()));
  };
  return {
    info: { ...meta, flavor: f },
    decide: (state, questions, { parts } = {}) => run([{ state, questions, ...(parts ? { parts } : {}) }])[0],
    decideMany: (items) => run(items),
  };
}
