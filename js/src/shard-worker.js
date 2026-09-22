// A tensor-parallel shard: one slice of every transformer layer. It talks to the engine worker
// over a MessagePort and never sees token ids, only the residual stream.

import { Wasm } from "./wasm.js";

let w = null;
let ptr = 0;
let total = 0;
let xptr = 0;
let tokens = 0;
let hidden = 0;

self.onmessage = (ev) => {
  if (ev.data?.type === "port") ev.data.port.onmessage = (e) => handle(ev.data.port, e.data);
};

async function handle(port, m) {
  try {
    switch (m.type) {
      case "init":
        w = await Wasm.create(m.module);
        total = m.total;
        ptr = w.alloc(total);
        port.postMessage({ seq: m.seq, type: "ok", tile: w.tile });
        break;
      case "data":
        w.bytes(ptr + m.dst, m.bytes.byteLength).set(m.bytes);
        break;
      case "load": {
        w.call(() => w.check(w.x.kevala_shard_load(ptr, total, m.primary ? 1 : 0)));
        const dv = new DataView(w.memory, ptr);
        const h = JSON.parse(new TextDecoder().decode(w.bytes(ptr + 16, dv.getUint32(8, true))));
        hidden = h.config.hidden_size;
        port.postMessage({ seq: m.seq, type: "ok" });
        break;
      }
      case "batch": {
        const t = m.table;
        const [tp] = w.put(new Uint8Array(t.buffer, t.byteOffset, t.byteLength));
        xptr = w.call(() => w.x.kevala_shard_batch(tp, t.length));
        w.x.kevala_free(tp, t.byteLength);
        tokens = t[0];
        port.postMessage({ seq: m.seq, type: "ok" });
        break;
      }
      case "step": {
        const expected = tokens * hidden;
        if (!(m.x instanceof Float32Array) || m.x.length !== expected) {
          throw new Error(`invalid Laya shard input: expected ${expected} f32 values`);
        }
        w.f32(xptr, expected).set(m.x);
        const pp = w.call(() => w.x.kevala_shard_step(m.s));
        m.x.set(w.f32(pp, expected));
        port.postMessage({ seq: m.seq, type: "partial", p: m.x }, [m.x.buffer]);
        break;
      }
    }
  } catch (e) {
    port.postMessage({ seq: m.seq, type: "error", message: String(e?.message || e) });
  }
}
