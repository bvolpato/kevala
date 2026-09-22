// Loads the kevala WebAssembly module and wraps its C ABI.
//
// Three builds ship side by side and the best one the browser validates is used:
// relaxed (SIMD128 + relaxed-simd fused multiply-add), simd (SIMD128), base (no SIMD).

import { selectCpuTile } from "./cpu-tune.js";

const PROBES = {
  relaxed: [0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 15, 1, 13, 0, 65, 1, 253, 15, 65, 2, 253, 15, 253, 128, 2, 11],
  simd: [0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11],
};

export function wasmFlavor(prefer) {
  if (prefer) return prefer;
  for (const f of ["relaxed", "simd"]) {
    try {
      if (WebAssembly.validate(new Uint8Array(PROBES[f]))) return f;
    } catch {}
  }
  return "base";
}

const modules = new Map();

/** Compiles (once per flavor) the module at `${base}kevala-${flavor}.wasm`. */
export async function compile(base, flavor) {
  const url = new URL(`kevala-${flavor}.wasm`, base).href;
  if (!modules.has(url)) {
    modules.set(
      url,
      (async () => {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
        if (WebAssembly.compileStreaming && (res.headers.get("content-type") || "").includes("application/wasm")) {
          return WebAssembly.compileStreaming(res);
        }
        return WebAssembly.compile(await res.arrayBuffer());
      })(),
    );
  }
  return modules.get(url);
}

const enc = new TextEncoder();
const dec = new TextDecoder();

/** One instance of the module, in one of its roles (engine, coordinator, shard, converter). */
export class Wasm {
  static async create(module, tile) {
    const w = new Wasm();
    w.instance = await WebAssembly.instantiate(module, {});
    w.x = w.instance.exports;
    w.x.kevala_init();
    w.tile = tile ?? w.tune();
    w.x.kevala_set_tile(w.tile);
    return w;
  }

  /** Times both CPU register tiles and returns the selected tile. */
  tune() {
    return selectCpuTile(this.x);
  }

  get memory() {
    return this.x.memory.buffer;
  }

  bytes(ptr, len) {
    return new Uint8Array(this.memory, ptr, len);
  }

  f32(ptr, len) {
    return new Float32Array(this.memory, ptr, len);
  }

  u32(ptr, len) {
    return new Uint32Array(this.memory, ptr, len);
  }

  error() {
    return dec.decode(this.bytes(this.x.kevala_error_ptr(), this.x.kevala_error_len())) || "unknown kevala error";
  }

  check(rc) {
    if (rc !== 0) throw new Error(this.error());
  }

  /** Runs `fn` and turns a trap into an Error that carries the Rust panic message. */
  call(fn) {
    try {
      return fn();
    } catch (e) {
      if (e instanceof WebAssembly.RuntimeError) {
        const msg = this.error();
        const err = new Error(msg && msg !== "unknown kevala error" ? msg : `kevala trapped: ${e.message}`);
        err.trapped = true;
        throw err;
      }
      throw e;
    }
  }

  out() {
    return this.bytes(this.x.kevala_out_ptr(), this.x.kevala_out_len());
  }

  outText() {
    return dec.decode(this.out());
  }

  /** Copies `data` (bytes or string) into fresh module memory; returns [ptr, len]. */
  put(data) {
    const b = typeof data === "string" ? enc.encode(data) : data;
    const ptr = this.x.kevala_alloc(b.length);
    this.bytes(ptr, b.length).set(b);
    return [ptr, b.length];
  }

  withInput(data, fn) {
    const [ptr, len] = this.put(data);
    try {
      return this.call(() => fn(ptr, len));
    } finally {
      this.x.kevala_free(ptr, len);
    }
  }

  alloc(len) {
    return this.x.kevala_alloc(len);
  }
}

/** Parses the layout blob written by `kevala_layouts`. */
export function parseLayouts(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let at = 0;
  const u32 = () => {
    const v = dv.getUint32(at, true);
    at += 4;
    return v;
  };
  const layouts = [];
  while (at < bytes.byteLength) {
    const plen = u32();
    const prefix = bytes.slice(at, at + plen);
    at += plen;
    const total = u32();
    const n = u32();
    const pieces = new Uint32Array(n * 5);
    for (let i = 0; i < n * 5; i++) pieces[i] = u32();
    layouts.push({ prefix, total, pieces });
  }
  return layouts;
}

/** Parses the segment table written by `kevala_prepare`. */
export function parseSegments(u32) {
  const tokens = u32[0];
  const n = u32[1];
  const segs = [];
  let i = 2;
  for (let s = 0; s < n; s++) {
    const [start, len, qtype, k] = [u32[i], u32[i + 1], u32[i + 2], u32[i + 3]];
    segs.push({ start, len, qtype, markers: Array.from(u32.subarray(i + 4, i + 4 + k)) });
    i += 4 + k;
  }
  return { tokens, segs };
}
