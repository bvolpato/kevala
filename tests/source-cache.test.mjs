import assert from "node:assert/strict";
import test from "node:test";
import { cacheInfo, isCached, openPack } from "../js/src/source.js";

function replaceGlobals(t, values) {
  const previous = new Map();
  for (const [name, value] of Object.entries(values)) {
    previous.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
    Object.defineProperty(globalThis, name, { configurable: true, enumerable: true, writable: true, value });
  }
  t.after(() => {
    for (const [name, descriptor] of previous) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else delete globalThis[name];
    }
  });
}

test("a failed cache writer releases its tee branch without blocking the loader", async (t) => {
  let cachedResponse;
  const progress = [];
  const cache = {
    match: async () => null,
    put: async (_key, response) => {
      cachedResponse = response;
      throw new Error("quota exceeded");
    },
    keys: async () => [],
    delete: async () => true,
  };
  const sourceChunks = [Uint8Array.from([1, 2]), Uint8Array.from([3, 4]), Uint8Array.from([5])];
  let next = 0;
  const fetchStub = async (_url, options = {}) => {
    if (options.method === "HEAD") return new Response(null, { status: 405 });
    const body = new ReadableStream({
      pull(controller) {
        if (next === sourceChunks.length) controller.close();
        else controller.enqueue(sourceChunks[next++]);
      },
    });
    return new Response(body, { status: 200 });
  };
  replaceGlobals(t, {
    caches: { open: async () => cache },
    fetch: fetchStub,
    navigator: { storage: undefined },
    self: { location: { href: "https://example.test/" } },
  });

  const canceled = new WeakSet();
  const originalCancel = Object.getOwnPropertyDescriptor(ReadableStream.prototype, "cancel");
  Object.defineProperty(ReadableStream.prototype, "cancel", {
    configurable: true,
    value(reason) {
      canceled.add(this);
      return originalCancel.value.call(this, reason);
    },
  });
  t.after(() => Object.defineProperty(ReadableStream.prototype, "cancel", originalCancel));

  const pack = await openPack("https://example.test/model.kevala", {
    onProgress: (event) => progress.push(event),
  });
  await pack.saving;
  assert.ok(cachedResponse?.body);
  assert.equal(canceled.has(cachedResponse.body), true);

  const loaded = [];
  for await (const chunk of pack.chunks()) loaded.push(...chunk);
  assert.deepEqual(loaded, [1, 2, 3, 4, 5]);
  assert.ok(progress.some((event) => event.phase === "cache-failed"));
});

function opfsFixture(index) {
  const files = new Map([["index.json", new TextEncoder().encode(JSON.stringify(index))]]);
  const bytes = (value) => (value instanceof Uint8Array ? value : new Uint8Array(value));
  const fileHandle = (name, create) => ({
    async getFile() {
      if (!files.has(name)) throw new Error(`missing ${name}`);
      return new Blob([files.get(name)]);
    },
    async createWritable() {
      if (!create && !files.has(name)) throw new Error(`missing ${name}`);
      const chunks = [];
      return {
        write: async (chunk) => chunks.push(bytes(chunk).slice()),
        close: async () => {
          const total = chunks.reduce((n, chunk) => n + chunk.byteLength, 0);
          const out = new Uint8Array(total);
          let at = 0;
          for (const chunk of chunks) {
            out.set(chunk, at);
            at += chunk.byteLength;
          }
          files.set(name, out);
        },
        abort: async () => {},
      };
    },
  });
  const dir = {
    getFileHandle: async (name, options = {}) => {
      if (!files.has(name) && !options.create) throw new Error(`missing ${name}`);
      if (!files.has(name)) files.set(name, new Uint8Array());
      return fileHandle(name, !!options.create);
    },
    async *entries() {
      for (const name of files.keys()) yield [name, {}];
    },
    removeEntry: async (name) => {
      files.delete(name);
    },
  };
  const root = { getDirectoryHandle: async () => dir };
  return { files, root };
}

test("OPFS cache keys prune files evicted behind the index", async (t) => {
  const key = "https://example.test/evicted.kevala";
  const fixture = opfsFixture({ [key]: { name: "evicted-pack", bytes: 17 } });
  replaceGlobals(t, {
    navigator: { storage: { getDirectory: async () => fixture.root } },
    self: { location: { href: "https://example.test/" } },
    caches: undefined,
  });

  const info = await cacheInfo();
  assert.deepEqual(info, { available: true, storage: "opfs", entries: [], bytes: 0 });
  assert.equal(await isCached(key), false);
  assert.equal(new TextDecoder().decode(fixture.files.get("index.json")), "{}");
});
