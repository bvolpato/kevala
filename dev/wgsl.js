// The dev pages build pipelines themselves; this gives them the same kernels the runtime uses,
// from the same WebAssembly binary (crates/kevala/src/gpu.rs specializes crates/kevala/src/wgsl).
import { Wasm } from "../js/src/wasm.js";

const DEFAULT_WASM_URL = new URL("../js/src/kevala-base.wasm", import.meta.url);

export async function kernelSource(wasmUrl = DEFAULT_WASM_URL) {
  const url = new URL(wasmUrl, import.meta.url);
  const module = await WebAssembly.compileStreaming(fetch(url));
  const w = await Wasm.create(module);
  const source = (kernel, spec = {}) => w.withInput(JSON.stringify({ kernel, ...spec }), (p, l) => (w.check(w.x.kevala_wgsl(p, l)), w.outText()));
  source.wasmUrl = url.href;
  return source;
}
