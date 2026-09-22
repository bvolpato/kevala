// Where packs come from: the Cache API, the network with progress, or an upstream Hugging Face
// checkpoint converted in the browser. Everything streams, so no stage holds a second copy of
// the weights in JavaScript memory.

import { clearTuning } from "./cpu-policy.js";

export const CACHE_NAME = "kevala-v1";

/** Pre-converted int8 packs of the models below, pinned to one commit of their Hugging Face repo. */
const PACKS = "https://huggingface.co/bvolpato/kevala-packs/resolve/e75a06d9329e19fe879f44cfcce33914fb96dade";

/**
 * Known models, by name. Each downloads its pre-converted pack (`hosted`); when that is
 * unreachable, models with browserConvert can convert their pinned upstream checkpoint. A model
 * whose pack is not uploaded yet has no `hosted` URL: load it from a .kevala file you have.
 */
export const MODELS = {
  laya: {
    arch: "laya",
    label: "Laya (English), ModernBERT-large encoder, 421M",
    repo: "convaiinnovations/laya",
    revision: "1c5edc17a7acd8701df6fc341c0d179f1c62c982",
    license: "apache-2.0",
    hosted: `${PACKS}/laya-q8.kevala`,
    // true when load() can convert it in the browser from upstream files (no pack to host)
    browserConvert: true,
    // bytes of the upstream checkpoint and of the int8 pack
    download: 842609210,
    pack: 478886720,
    block: 32,
  },
  "kev-0.8b": {
    arch: "kev",
    label: "Kev-0.8B, Qwen3.5-0.8B decoder with a pointer head",
    repo: "jaredpalmer/kev-0.8b",
    revision: "54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8",
    base: { repo: "Qwen/Qwen3.5-0.8B-Base", revision: "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68" },
    license: "apache-2.0",
    hosted: `${PACKS}/kev-0.8b-q8.kevala`,
    browserConvert: true,
    download: 1620000000,
    pack: 857259584,
    block: 32,
  },
  "kev-4b": {
    arch: "kev", label: "Kev-4B, Qwen3.5 decoder with a trained pointer head",
    repo: "jaredpalmer/kev-4b", revision: "485ace8703592fcf405488b262449990824cfed1",
    base: { repo: "Qwen/Qwen3.5-4B-Base", revision: "1001bb4d826a52d1f399e183466143f4da7b741b" },
    // hosted: `${PACKS}/kev-4b-q8.kevala` once uploaded (docs/packs.md)
    license: "apache-2.0", hosted: null,
    browserConvert: false, download: 9502534940, pack: 4756384192, block: 32,
    packSha256: "f75af1de41025a0c4d8656980521c6284031dc5b1c8480f183b2d92b5eb4319c",
  },
  "kev-9b": {
    arch: "kev", label: "Kev-9B, Qwen3.5 decoder with a trained pointer head",
    repo: "jaredpalmer/kev-9b", revision: "2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c",
    base: { repo: "Qwen/Qwen3.5-9B-Base", revision: "68c46c4b3498877f3ef123c856ecfde50c39f404" },
    // hosted: `${PACKS}/kev-9b-q8.kevala` once uploaded
    license: "apache-2.0", hosted: null,
    browserConvert: false, download: 19530970823, pack: 8963899968, block: 32,
    packSha256: "65ed43e644e895519f9c4987642ef52e1d97d25e7edcf93b0c01212fe17a5185",
  },
  ...Object.fromEntries([
    ["0.8b", "0.8B", "2fc06364715b967f1860aea9cf38778875588b17", 1769980465, 855225984, "f06789ea73ec84057cab9c9b12f560c904218d0b1506795727a988a2122add6a"],
    ["2b", "2B", "15852e8c16360a2fea060d615a32b45270f8a8fc", 4571274023, 2127742976, "5995ad5ed1c44301818185e3af7fd1490b937b36ff6bff7ada15bcd26025e8b2"],
    ["4b", "4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", 9342907469, 4751303104, "ad96f2bc64d357bed4c453af3b4af9d166b0f7cde70b509cb842350e83959b5e"],
  ].map(([size, upstream, revision, download, pack, packSha256]) => [`semif-qwen3.5-${size}`, {
    arch: "kev", label: `SemIf-style Qwen3.5-${upstream}, frozen model with direct option scoring`,
    repo: `Qwen/Qwen3.5-${upstream}`, revision, license: "apache-2.0",
    // hosted: `${PACKS}/semif-qwen3.5-${size}-q8.kevala` once uploaded
    hosted: null,
    browserConvert: false, download, pack, packSha256, block: 32,
  }])),
};

export const UPSTREAM = MODELS.laya;

/** Resolves a model option to `{ url }` for a pack or `{ spec }` for an upstream checkpoint. */
export function resolveModel(model) {
  if (model == null) return { spec: MODELS.laya };
  if (typeof model === "string") {
    if (MODELS[model]) return { spec: MODELS[model] };
    return { url: model };
  }
  if (model.url) return { url: model.url };
  if (model.name && MODELS[model.name]) return { spec: { ...MODELS[model.name], ...model } };
  return { spec: { ...MODELS.laya, ...model } };
}

/**
 * Pack storage. The Origin Private File System takes multi-hundred-megabyte files as streamed
 * writes (the Cache API rejects entries that large in some browsers); the Cache API is the
 * fallback where OPFS is missing. Both expose match(key) -> Response | null, put(key, Response),
 * keys() and remove(key).
 */
function fileName(key) {
  let h = 0x811c9dc5;
  for (let i = 0; i < key.length; i++) h = Math.imul(h ^ key.charCodeAt(i), 0x01000193) >>> 0;
  return `${h.toString(16).padStart(8, "0")}-${key.split("/").pop().replace(/[^A-Za-z0-9._-]/g, "_")}`;
}

async function opfsStore() {
  if (typeof navigator === "undefined" || !navigator.storage?.getDirectory) return null;
  const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle(CACHE_NAME, { create: true });
  const readIndex = async () => {
    try {
      const index = JSON.parse(await (await (await dir.getFileHandle("index.json")).getFile()).text());
      return index && typeof index === "object" && !Array.isArray(index) ? index : {};
    } catch {
      return {};
    }
  };
  const writeFile = async (name, chunks) => {
    const fh = await dir.getFileHandle(name, { create: true });
    let n = 0;
    if (fh.createWritable) {
      const out = await fh.createWritable();
      try {
        for await (const c of chunks) {
          await out.write(c);
          n += c.byteLength;
        }
        await out.close();
      } catch (e) {
        try { await out.abort?.(); } catch {}
        throw e;
      }
    } else {
      // Safari workers: synchronous access handles only
      const h = await fh.createSyncAccessHandle();
      try {
        h.truncate(0);
        for await (const c of chunks) n += h.write(c, { at: n });
        h.flush();
      } finally {
        try { h.close(); } catch {}
      }
    }
    return n;
  };
  // Entries are only listed in the index once their file is complete. Keep all index and file
  // mutations in one queue so stale-entry cleanup cannot race a replacement put.
  let lock = Promise.resolve();
  const withLock = (fn) => {
    const run = lock.then(fn);
    lock = run.catch(() => {});
    return run;
  };
  const writeIndex = (idx) => writeFile("index.json", [new TextEncoder().encode(JSON.stringify(idx))]);
  const validEntry = (e) => e && typeof e.name === "string" && Number.isSafeInteger(e.bytes) && e.bytes >= 0;
  return {
    kind: "opfs",
    async match(key) {
      return withLock(async () => {
        const idx = await readIndex();
        const e = idx[key];
        if (!e) return null;
        let f;
        try {
          if (!validEntry(e)) throw new Error("invalid cache index entry");
          f = await (await dir.getFileHandle(e.name)).getFile();
          if (f.size !== e.bytes) throw new Error("stale cache index entry");
          return new Response(f.stream(), { headers: { "content-length": String(f.size), "x-kevala-size": String(f.size) } });
        } catch {
          delete idx[key];
          await writeIndex(idx).catch(() => {});
          if (validEntry(e)) await dir.removeEntry(e.name).catch(() => {});
          return null;
        }
      });
    },
    async put(key, res) {
      return withLock(async () => {
        const name = fileName(key);
        // Files the index does not list are leftovers of interrupted writes: free their space first.
        const listed = new Set(Object.values(await readIndex()).filter(validEntry).map((e) => e.name));
        for await (const [file] of dir.entries()) {
          if (file !== "index.json" && file !== name && !listed.has(file)) await dir.removeEntry(file).catch(() => {});
        }
        const reader = res.body?.getReader();
        if (!reader) throw new Error("cache put requires a response body");
        const chunks = {
          async *[Symbol.asyncIterator]() {
            for (;;) {
              const { done, value } = await reader.read();
              if (done) return;
              yield value;
            }
          },
        };
        try {
          const bytes = await writeFile(name, chunks);
          const idx = await readIndex();
          idx[key] = { name, bytes };
          await writeIndex(idx);
        } catch (e) {
          await dir.removeEntry(name).catch(() => {}); // a partial file would only hold quota
          throw e;
        } finally {
          try { void reader.cancel().catch(() => {}); } catch {}
          try { reader.releaseLock(); } catch {}
        }
      });
    },
    async keys() {
      return withLock(async () => {
        const idx = await readIndex();
        const entries = [];
        let changed = false;
        for (const [key, e] of Object.entries(idx)) {
          let valid = validEntry(e);
          if (valid) {
            try {
              const f = await (await dir.getFileHandle(e.name)).getFile();
              valid = f.size === e.bytes;
            } catch {
              valid = false;
            }
          }
          if (!valid) {
            delete idx[key];
            changed = true;
            if (e?.name) await dir.removeEntry(e.name).catch(() => {});
          } else {
            entries.push({ key, bytes: e.bytes });
          }
        }
        if (changed) await writeIndex(idx).catch(() => {});
        return entries;
      });
    },
    async remove(key) {
      return withLock(async () => {
        const idx = await readIndex();
        const e = idx[key];
        if (e?.name) await dir.removeEntry(e.name).catch(() => {});
        if (key in idx) {
          delete idx[key];
          await writeIndex(idx);
        }
      });
    },
  };
}

async function cacheStore() {
  if (typeof caches === "undefined") return null;
  const c = await caches.open(CACHE_NAME);
  return {
    kind: "cache",
    match: (key) => c.match(key).then((r) => r || null),
    async put(key, res) {
      try {
        await c.put(key, res);
      } catch (e) {
        try { void res.body?.cancel(e).catch(() => {}); } catch {}
        throw e;
      }
    },
    async keys() {
      const out = [];
      for (const req of await c.keys()) {
        const r = await c.match(req);
        out.push({ key: req.url, bytes: Number(r?.headers.get("x-kevala-size")) || Number(r?.headers.get("content-length")) || 0 });
      }
      return out;
    },
    remove: (key) => c.delete(key),
  };
}

async function openCache(enabled) {
  if (!enabled) return null;
  for (const make of [opfsStore, cacheStore]) {
    try {
      const s = await make();
      if (s) return s;
    } catch {}
  }
  return null;
}

/** Cache key for a converted upstream checkpoint: revision and quantization both matter. */
export function upstreamKey(up) {
  const base = up.base ? `+${up.base.repo}@${up.base.revision}` : "";
  const artifact = up.packSha256 ? `-${up.packSha256}` : "";
  return `https://kevala.cache/${up.repo}/${up.revision}${base}/q8-b${up.block}${artifact}.kevala`;
}

function hfUrl(up, file) {
  return `https://huggingface.co/${up.repo}/resolve/${up.revision}/${file}`;
}

async function checked(res, url) {
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res;
}

/** Yields the body of `res` as chunks, reporting progress. */
async function* body(res, file, total, onProgress, signal) {
  const reader = res.body.getReader();
  let loaded = 0;
  try {
    for (;;) {
      if (signal?.aborted) throw signal.reason ?? new DOMException("aborted", "AbortError");
      const { done, value } = await reader.read();
      if (done) break;
      loaded += value.byteLength;
      onProgress?.({ phase: "download", file, loaded, total });
      yield value;
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Resolves a pack source to `{ size, chunks(), cached, key }`.
 * `model` is a URL string, an ArrayBuffer / Uint8Array / Blob, or `{ repo, revision, block }`
 * naming an upstream checkpoint (the default). For a known model, `from` picks where its weights
 * come from: "pack" (its hosted int8 pack, the default) or "checkpoint" (the original weights,
 * converted here).
 */
export async function openPack(model, { cache = true, from = "pack", signal, onProgress, convert } = {}) {
  if (from !== "pack" && from !== "checkpoint") throw new Error(`from must be "pack" or "checkpoint", not ${JSON.stringify(from)}`);
  if (model instanceof ArrayBuffer || ArrayBuffer.isView(model)) {
    const bytes = model instanceof ArrayBuffer ? new Uint8Array(model) : new Uint8Array(model.buffer, model.byteOffset, model.byteLength);
    return { size: bytes.byteLength, cached: false, key: null, async *chunks() { yield bytes; } };
  }
  if (typeof Blob !== "undefined" && model instanceof Blob) {
    return fromResponse(new Response(model), "pack", model.size, false, null, onProgress, signal);
  }
  const store = await openCache(cache);
  const which = resolveModel(model);
  const key = which.url ? new URL(which.url, self.location?.href).href : upstreamKey(which.spec);
  if (store) {
    const hit = await store.match(key).catch(() => null);
    if (hit) {
      const size = Number(hit.headers.get("content-length")) || Number(hit.headers.get("x-kevala-size")) || 0;
      onProgress?.({ phase: "cache", file: key, loaded: size, total: size });
      return fromResponse(hit, "cache", size, true, key, null, signal);
    }
  }
  if (which.url) {
    const size = await remoteSize(key, signal).catch(() => 0);
    if (size && (await acceptsRanges(key, signal))) {
      return streamIntoCache(store, key, size, fetchRange(key, 0, size, key.split("/").pop(), { signal, onProgress }), { signal, onProgress });
    }
    // a server without byte ranges (or without HEAD): one plain stream
    const res = await checked(await fetch(key, { signal }), key);
    const length = Number(res.headers.get("content-length")) || 0;
    return streamIntoCache(store, key, length, body(res, key.split("/").pop(), length, onProgress, signal), { signal, onProgress });
  }
  const up = which.spec;
  // a model with a hosted int8 pack downloads that (about half the bytes of the checkpoint and no
  // conversion); if it is missing or unreachable, convert from the upstream checkpoint instead
  if (up.hosted && from === "pack") {
    const size = await remoteSize(up.hosted, signal).catch((e) => (signal?.aborted ? Promise.reject(e) : 0));
    if (size) return streamIntoCache(store, key, size, fetchRange(up.hosted, 0, size, up.hosted.split("/").pop(), { signal, onProgress }), { signal, onProgress });
  }
  // upstream checkpoint: convert, cache, and hand back the finished bytes
  if (up.browserConvert === false) {
    if (!up.hosted) throw new Error(`${up.label || up.repo} has no published pack yet: pass the URL of its .kevala file.`);
    throw new Error(from === "checkpoint"
      ? `${up.label || up.repo} requires offline conversion; load its hosted .kevala pack with from: "pack".`
      : `The hosted pack for ${up.label || up.repo} is unavailable. Retry the download or pass a local .kevala URL.`);
  }
  const made = await convert(up, { signal, onProgress });
  if (!(made instanceof Uint8Array)) return streamIntoCache(store, key, made.size, made.chunks(), { signal, onProgress, file: "converted pack" });
  const bytes = made;
  if (store) {
    onProgress?.({ phase: "cache", file: key, loaded: 0, total: bytes.byteLength });
    try {
      await store.put(key, new Response(bytes, { headers: { "content-type": "application/octet-stream", "x-kevala-size": String(bytes.byteLength) } }));
    } catch (e) {
      onProgress?.({ phase: "cache-failed", message: String(e?.message || e) });
    }
  }
  return { size: bytes.byteLength, cached: false, key, async *chunks() { yield bytes; } };
}

/**
 * A pack arriving as chunks (an async iterator), copied into the cache under `key` while the
 * loader reads it. `saving` settles when the cached copy is complete.
 */
function streamIntoCache(store, key, size, chunks, { signal, onProgress, file = "pack" } = {}) {
  let stream = new ReadableStream({
    async pull(controller) {
      const { done, value } = await chunks.next();
      if (done) controller.close();
      else controller.enqueue(value);
    },
  });
  let saving = null;
  if (store) {
    const [toCache, toLoader] = stream.tee();
    stream = toLoader;
    const headers = { "content-type": "application/octet-stream", "x-kevala-size": String(size) };
    // a pack that cannot be stored still loads; say why, so a full disk does not look like a hang
    const cacheBody = new Response(toCache, { headers });
    saving = store.put(key, cacheBody).catch((e) => {
      // Do not await this: ReadableStream tee cancellation may wait for the loader branch, which
      // is still consuming the pack. Canceling releases the unread cache queue immediately.
      void toCache.cancel(e).catch(() => {});
      onProgress?.({ phase: "cache-failed", message: String(e?.message || e) });
    });
  }
  return { ...fromResponse(new Response(stream), file, size, false, key, null, signal), saving };
}

function fromResponse(res, file, size, cached, key, onProgress, signal) {
  return {
    size,
    cached,
    key,
    chunks: () => body(res, file, size, onProgress, signal),
  };
}

/** The size of a remote file, from a HEAD request (which follows Hugging Face's CDN redirect). */
async function remoteSize(url, signal) {
  const res = await checked(await fetch(url, { method: "HEAD", signal }), url);
  const size = Number(res.headers.get("content-length")) || Number(res.headers.get("x-linked-size"));
  if (!size) throw new Error(`${url}: the server did not report the file size`);
  return size;
}

/** Whether a server answers byte-range requests, probed with a one-byte range. */
async function acceptsRanges(url, signal) {
  try {
    const res = await fetch(url, { signal, headers: { range: "bytes=0-0" } });
    await res.body?.cancel();
    return res.status === 206;
  } catch (e) {
    if (signal?.aborted) throw e;
    return false;
  }
}

/** Fetches the upstream files a conversion needs, reporting progress. */
export async function fetchUpstream(up, { signal, onProgress }) {
  const text = async (file) => (await checked(await fetch(hfUrl(up, file), { signal }), file)).text();
  const [enc, agent, tok] = await Promise.all([text("encoder/config.json"), text("rl_agent_config.json"), text("tokenizer/tokenizer.json")]);
  const url = hfUrl(up, "model.safetensors");
  const total = await remoteSize(url, signal);
  return { enc, agent, tok, total, chunks: () => fetchRange(url, 0, total, "model.safetensors", { signal, onProgress }) };
}

/** Lists stored packs with their sizes. */
export async function cacheInfo() {
  const store = await openCache(true);
  if (!store) return { available: false, entries: [], bytes: 0 };
  const entries = await store.keys();
  return { available: true, storage: store.kind, entries, bytes: entries.reduce((a, e) => a + e.bytes, 0) };
}

/** Whether a model (a MODELS name, a spec, or a pack URL) is already stored locally. */
export async function isCached(model) {
  const store = await openCache(true);
  if (!store) return false;
  const which = resolveModel(model);
  const key = which.url ? new URL(which.url, self.location?.href).href : upstreamKey(which.spec);
  return (await store.keys()).some((e) => e.key === key);
}

/** Deletes stored packs and CPU tuning profiles. */
export async function clearCache() {
  const store = await openCache(true);
  for (const e of (await store?.keys()) || []) await store.remove(e.key);
  if (typeof caches !== "undefined") await caches.delete(CACHE_NAME).catch(() => {});
  await clearTuning();
  return true;
}

/**
 * Applies a layout (from `kevala_layouts`) to a pack streaming by. `write(dst, bytes)` receives
 * fragments in increasing destination order; the layout prefix is written first.
 */
export class PieceSink {
  constructor(layout, write) {
    this.write = write;
    this.total = layout.total;
    const p = layout.pieces;
    this.pieces = [];
    for (let i = 0; i < p.length; i += 5) {
      const [src, dst, len, rows, stride] = [p[i], p[i + 1], p[i + 2], p[i + 3], p[i + 4]];
      if (rows && len) this.pieces.push({ src, dst, len, rows, stride, row: 0 });
    }
    this.pieces.sort((a, b) => a.src - b.src);
    this.next = 0;
    write(0, layout.prefix);
  }

  /** Feeds bytes `[at, at + chunk.length)` of the source. */
  push(chunk, at) {
    const end = at + chunk.byteLength;
    for (let i = this.next; i < this.pieces.length; i++) {
      const pc = this.pieces[i];
      if (pc.src >= end) break;
      while (pc.row < pc.rows) {
        const s = pc.src + pc.row * pc.stride;
        if (s >= end) break;
        const lo = Math.max(s, at);
        const hi = Math.min(s + pc.len, end);
        if (hi > lo) this.write(pc.dst + pc.row * pc.len + (lo - s), chunk.subarray(lo - at, hi - at));
        if (s + pc.len <= end) pc.row++;
        else break;
      }
    }
    while (this.next < this.pieces.length && this.pieces[this.next].row === this.pieces[this.next].rows) this.next++;
  }

  get done() {
    return this.next >= this.pieces.length;
  }
}

/** Collects destination fragments into ~4 MB messages for a remote worker. */
export class Batcher {
  constructor(send, capacity = 4 << 20) {
    this.send = send;
    this.capacity = capacity;
    this.buf = null;
    this.base = 0;
    this.used = 0;
  }

  write(dst, bytes) {
    let off = 0;
    while (off < bytes.byteLength) {
      if (!this.buf || dst + off < this.base || dst + off >= this.base + this.capacity) {
        this.flush();
        this.buf = new Uint8Array(this.capacity);
        this.base = dst + off;
        this.used = 0;
      }
      const at = dst + off - this.base;
      const n = Math.min(bytes.byteLength - off, this.capacity - at);
      this.buf.set(bytes.subarray(off, off + n), at);
      this.used = Math.max(this.used, at + n);
      off += n;
    }
  }

  flush() {
    if (this.buf && this.used) this.send(this.base, this.buf.subarray(0, this.used));
    this.buf = null;
  }
}

/** A pinned Hugging Face file URL. */
export function hfFile(repo, revision, file) {
  return `https://huggingface.co/${repo}/resolve/${revision}/${file}`;
}

/** Fetches a whole file with download progress. */
export async function fetchBytes(url, file, { signal, onProgress } = {}) {
  const res = await checked(await fetch(url, { signal }), url);
  const total = Number(res.headers.get("content-length")) || 0;
  const parts = [];
  let n = 0;
  for await (const c of body(res, file, total, onProgress, signal)) {
    parts.push(c);
    n += c.byteLength;
  }
  const out = new Uint8Array(n);
  let at = 0;
  for (const c of parts) {
    out.set(c, at);
    at += c.byteLength;
  }
  return out;
}

/**
 * Streams bytes `[start, end)` of a file as chunks, in order, with progress. Several byte ranges
 * download at once: a single stream from a CDN often runs far below the connection's bandwidth.
 * At most `streams` pieces of `piece` bytes are held in memory while they wait their turn.
 */
export async function* fetchRange(url, start, end, file, { signal, onProgress, streams = 6, piece = 16 << 20 } = {}) {
  const total = end - start;
  let loaded = 0;
  const download = async (lo, hi) => {
    const res = await checked(await fetch(url, { signal, headers: { range: `bytes=${lo}-${hi - 1}` } }), url);
    if (res.status !== 206) throw new Error(`${url}: the server ignored the byte range`);
    const out = new Uint8Array(hi - lo);
    let at = 0;
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      out.set(value, at);
      at += value.byteLength;
      loaded += value.byteLength;
      onProgress?.({ phase: "download", file, loaded, total });
    }
    if (at !== out.byteLength) throw new Error(`${url}: got ${at} of ${out.byteLength} bytes`);
    return out;
  };
  const pending = [];
  let next = start;
  const startNext = () => {
    const lo = next;
    next = Math.min(end, lo + piece);
    const job = download(lo, next);
    job.catch(() => {}); // each job is awaited in order below; this only covers the ones never reached
    pending.push(job);
  };
  while (next < end && pending.length < streams) startNext();
  while (pending.length) {
    const bytes = await pending.shift();
    if (next < end) startNext();
    yield bytes;
  }
}
