// The dev pages build pipelines themselves; this gives them the same kernels the runtime uses,
// from the same WebAssembly binary (crates/kevala/src/gpu.rs specializes crates/kevala/src/wgsl).
import { Wasm } from "../js/src/wasm.js";

export async function kernelSource() {
  const module = await WebAssembly.compileStreaming(fetch(new URL("../js/src/kevala-base.wasm", import.meta.url)));
  const w = await Wasm.create(module);
  return (kernel, spec = {}) => w.withInput(JSON.stringify({ kernel, ...spec }), (p, l) => (w.check(w.x.kevala_wgsl(p, l)), w.outText()));
}
