import assert from "node:assert/strict";
import test from "node:test";
import { createGemma4PleResource, createGemma4PleStore } from "../js/src/gemma4-ple-store.js";
import { GpuWeights } from "../js/src/gpu.js";

function replaceGlobal(t, key, value) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, key);
  Object.defineProperty(globalThis, key, { configurable: true, value });
  t.after(() => previous ? Object.defineProperty(globalThis, key, previous) : delete globalThis[key]);
}

function dedicatedWorker(t, matches = true) {
  class Scope {
    static [Symbol.hasInstance](value) { return matches && value === globalThis; }
  }
  replaceGlobal(t, "DedicatedWorkerGlobalScope", Scope);
}

function fixture(t, options = {}) {
  const existing = Uint8Array.of(11, 22, 33);
  const files = new Map([["existing-model.kevala", existing]]);
  const writes = [];
  const reads = [];
  const removed = [];
  const syncWrites = [];
  const syncReads = [];
  const syncReadBuffers = new Set();
  let opens = 0, closes = 0, aborts = 0, snapshots = 0, active = 0;
  let syncOpens = 0, syncCloses = 0, syncFlushes = 0;
  const directory = {
    async getFileHandle(name, { create = false } = {}) {
      if (!files.has(name)) {
        if (!create) throw new DOMException("Missing file", "NotFoundError");
        files.set(name, new Uint8Array());
      }
      return {
        createSyncAccessHandle: !options.sync ? undefined : async () => {
          syncOpens++;
          if (options.syncOpenError) throw options.syncOpenError;
          let closed = false;
          return {
            truncate(size) {
              assert.equal(closed, false);
              if (options.syncTruncateError) throw options.syncTruncateError;
              files.set(name, new Uint8Array(size));
            },
            write(data, { at }) {
              assert.equal(closed, false);
              syncWrites.push([at, at + data.byteLength]);
              if (options.syncWriteError) throw options.syncWriteError;
              if (options.syncWriteCount !== undefined) return options.syncWriteCount;
              const count = Math.min(data.byteLength, options.partialWrite ?? data.byteLength);
              let bytes = files.get(name);
              if (bytes.byteLength < at + count) {
                const grown = new Uint8Array(at + count);
                grown.set(bytes);
                bytes = grown;
              }
              bytes.set(data.subarray(0, count), at);
              files.set(name, bytes);
              return count;
            },
            read(data, { at }) {
              assert.equal(closed, false);
              syncReads.push([at, at + data.byteLength]);
              syncReadBuffers.add(data.buffer);
              if (options.syncReadError) throw options.syncReadError;
              if (options.syncReadCount !== undefined) return options.syncReadCount;
              const bytes = files.get(name);
              const count = Math.min(data.byteLength, Math.max(0, bytes.byteLength - at), options.partialRead ?? data.byteLength);
              data.set(bytes.subarray(at, at + count));
              return count;
            },
            flush() {
              assert.equal(closed, false);
              syncFlushes++;
              if (options.syncFlushError) throw options.syncFlushError;
            },
            getSize() {
              assert.equal(closed, false);
              if (options.syncSizeError) throw options.syncSizeError;
              return files.get(name).byteLength - (options.syncShortFile ? 1 : 0);
            },
            close() {
              syncCloses++;
              if (options.syncCloseError) throw options.syncCloseError;
              closed = true;
            },
          };
        },
        async getFile() {
          snapshots++;
          if (options.getFileError) throw options.getFileError;
          const blob = new Blob([files.get(name)]);
          return {
            size: options.shortFile ? blob.size - 1 : blob.size,
            slice(start, end) {
              reads.push([start, end]);
              return {
                async arrayBuffer() {
                  if (options.readError) throw options.readError;
                  const bytes = await blob.slice(start, end).arrayBuffer();
                  return options.shortRead ? bytes.slice(0, -1) : bytes;
                },
              };
            },
          };
        },
        createWritable: options.noWritable ? undefined : async () => {
          opens++;
          if (options.openError) throw options.openError;
          let draft = new Uint8Array();
          return {
            async write({ type, position, data }) {
              assert.equal(type, "write");
              assert.equal(active++, 0, "OPFS writes must be serialized");
              writes.push({ position, data });
              try {
                await options.beforeWrite?.(writes.length);
                if (options.writeError) throw options.writeError;
                if (draft.length < position + data.byteLength) {
                  const grown = new Uint8Array(position + data.byteLength);
                  grown.set(draft);
                  draft = grown;
                }
                draft.set(data, position);
              } finally {
                active--;
              }
            },
            async close() {
              closes++;
              if (options.closeError) throw options.closeError;
              files.set(name, draft);
            },
            async abort() {
              aborts++;
              if (options.abortError) throw options.abortError;
            },
          };
        },
      };
    },
    async removeEntry(name) {
      removed.push(name);
      if (options.removeError) throw options.removeError;
      if (!files.delete(name)) throw new DOMException("Missing file", "NotFoundError");
    },
  };
  replaceGlobal(t, "navigator", { storage: { getDirectory: async () => directory } });
  return {
    directory, existing, files, writes, reads, removed, syncWrites, syncReads, syncReadBuffers,
    stats: () => ({ opens, closes, aborts, snapshots }),
    syncStats: () => ({ opens: syncOpens, closes: syncCloses, flushes: syncFlushes }),
  };
}

function tensor(name = "ple.0.embed", rows = 3, width = 64) {
  const info = { name, dtype: "q8", shape: [rows, width], size: rows * width, block: 32, scales_size: rows * width / 32 * 4 };
  const data = Int8Array.from({ length: info.size }, (_, i) => i % 256 - 128);
  const scales = Float32Array.from({ length: info.scales_size / 4 }, (_, i) => i < width / 32 ? 0 : (i % 2 ? -1 : 1) * (i + 1) / 7.123456789);
  return { info, data, scales };
}

function stream(store, table, part, size) {
  const source = part === "data" ? new Uint8Array(table.data.buffer) : new Uint8Array(table.scales.buffer);
  const scratch = new Uint8Array(size);
  for (let at = 0; at < source.length; at += size) {
    const count = Math.min(size, source.length - at);
    scratch.set(source.subarray(at, at + count));
    store.write(table.info.name, part, at, scratch.subarray(0, count));
    scratch.fill(255);
  }
}

function memoryWeights(tables) {
  const weights = Object.create(GpuWeights.prototype);
  weights.tensors = new Map(tables.map(({ info, data, scales }) => [info.name, {
    info, hostTable: true, filled: info.size + info.scales_size, need: info.size + info.scales_size, host: { data, scales },
  }]));
  return weights;
}

test("PLE store falls back only when OPFS is unavailable", async (t) => {
  replaceGlobal(t, "navigator", undefined);
  assert.equal(await createGemma4PleStore([]), null);
  replaceGlobal(t, "navigator", { storage: {} });
  assert.equal(await createGemma4PleStore([]), null);
  for (const name of ["SecurityError", "NotSupportedError", "QuotaExceededError"]) {
    const error = new DOMException("Cannot open OPFS", name);
    replaceGlobal(t, "navigator", { storage: { getDirectory: async () => { throw error; } } });
    if (name === "QuotaExceededError") await assert.rejects(createGemma4PleStore([]), (caught) => caught === error);
    else assert.equal(await createGemma4PleStore([]), null);
  }
});

test("late store setup cannot recreate a file removed by its owner", async (t) => {
  const f = fixture(t);
  const resource = await createGemma4PleResource();
  await resource.directory.removeEntry(resource.name);
  await assert.rejects(createGemma4PleStore([tensor().info], resource), { name: "NotFoundError" });
  assert.equal(f.files.has(resource.name), false);
  assert.equal(f.stats().opens, 0);
});

test("PLE files compact unordered tensors, copy split fragments, and gather exact unique rows", async (t) => {
  const f = fixture(t);
  const a = tensor("ple.0.embed");
  const b = tensor("ple.1.embed", 4, 32);
  Object.assign(a.info, { offset: 4096, scales_offset: 8192 });
  Object.assign(b.info, { offset: 16384, scales_offset: 32768 });
  const store = await createGemma4PleStore([b.info, a.info]);
  assert.equal(store.resource.directory, f.directory);
  assert.match(store.resource.name, /^kevala-ple-.+\.tmp$/);
  assert.equal(store.byteLength, a.info.size + a.info.scales_size + b.info.size + b.info.scales_size);
  await assert.rejects(store.gather(a.info.name, [0]), /is incomplete/);
  stream(store, a, "data", 7);
  stream(store, b, "scales", 3);
  await store.flush();
  stream(store, b, "data", 11);
  stream(store, a, "scales", 3);
  assert.equal(await store.finish(), store);
  assert.equal(await store.finish(), store);
  assert.deepEqual(f.stats(), { opens: 1, closes: 1, aborts: 0, snapshots: 1 });
  const expectedBytes = Buffer.concat([Buffer.from(b.data.buffer), Buffer.from(b.scales.buffer), Buffer.from(a.data.buffer), Buffer.from(a.scales.buffer)]);
  assert.deepEqual(Buffer.from(f.files.get(store.resource.name)), expectedBytes);

  const ids = new Int32Array([2, 0, 2, 1, 0]);
  const factor = 1.123456789;
  const target = new Float32Array(5 + ids.length * 64 + 3).fill(100);
  const expected = memoryWeights([a, b]).gatherHostRows(a.info.name, ids, factor, target.slice(), 5);
  assert.equal(await store.gather(a.info.name, ids, factor, target, 5), target);
  assert.deepEqual(target, expected);
  assert.equal(f.reads.length, 6, "duplicates and PAD row zero are each read once");
  const base = b.info.size + b.info.scales_size;
  assert.deepEqual(f.reads, [
    [base + 128, base + 192], [base + 192 + 16, base + 192 + 24],
    [base, base + 64], [base + 192, base + 192 + 8],
    [base + 64, base + 128], [base + 192 + 8, base + 192 + 16],
  ]);
  assert.deepEqual(await store.gather(b.info.name, [3, 1]), memoryWeights([b]).gatherHostRows(b.info.name, [3, 1]));
  assert.throws(() => store.write(a.info.name, "data", 0, Uint8Array.of(1)), /no longer writable/);
  await store.dispose();
  await store.dispose();
  assert.deepEqual(f.removed, [store.resource.name]);
  assert.deepEqual(f.files, new Map([["existing-model.kevala", f.existing]]));
});

test("PLE store rejects invalid Q8 metadata before creating files", async (t) => {
  const f = fixture(t);
  const { info } = tensor();
  for (const changed of [
    { dtype: "bf16" }, { block: 16 }, { shape: [3, 33] }, { shape: [0, 64] },
    { shape: [3, 64, 1] }, { size: 191 }, { scales_size: 23 }, { name: "" },
  ]) await assert.rejects(createGemma4PleStore([{ ...info, ...changed }]), /requires a block-32 Q8 matrix/);
  await assert.rejects(createGemma4PleStore([info, info]), /requires a block-32 Q8 matrix/);
  assert.deepEqual(f.stats(), { opens: 0, closes: 0, aborts: 0, snapshots: 0 });
  assert.equal(f.files.size, 1);
});

test("PLE store validates writes and removes incomplete files", async (t) => {
  const f = fixture(t);
  const a = tensor();
  const store = await createGemma4PleStore([a.info]);
  for (const args of [
    ["missing", "data", 0, Uint8Array.of(1)], [a.info.name, "unknown", 0, Uint8Array.of(1)],
    [a.info.name, "data", 1, Uint8Array.of(1)], [a.info.name, "data", -1, Uint8Array.of(1)],
    [a.info.name, "data", 0, new Uint8Array(a.info.size + 1)], [a.info.name, "data", 0, new Int8Array(1)],
  ]) assert.throws(() => store.write(...args), /invalid or out-of-order write/);
  store.write(a.info.name, "data", 0, Uint8Array.of(1));
  assert.throws(() => store.write(a.info.name, "data", 0, Uint8Array.of(2)), /out-of-order write/);
  await assert.rejects(store.finish(), /host row table ple\.0\.embed is incomplete/);
  assert.equal(f.stats().aborts, 1);
  assert.deepEqual(f.removed, [store.resource.name]);
  assert.equal(f.files.size, 1);
});

test("PLE gathers reject invalid selections and destinations without invalidating valid storage", async (t) => {
  const f = fixture(t);
  const a = tensor();
  const store = await createGemma4PleStore([a.info]);
  stream(store, a, "data", 32);
  stream(store, a, "scales", 4);
  await store.finish();
  for (const ids of [[-1], [3], [0.5], [NaN], null, "0"]) await assert.rejects(store.gather(a.info.name, ids), /invalid row selection/);
  for (const scale of [Infinity, NaN]) await assert.rejects(store.gather(a.info.name, [0], scale), /invalid row selection/);
  for (const offset of [-1, 0.5, Number.MAX_SAFE_INTEGER]) await assert.rejects(store.gather(a.info.name, [0], 1, null, offset), /invalid row selection/);
  await assert.rejects(store.gather(a.info.name, [0], 1, new Float32Array(63)), /undersized F32 destination/);
  await assert.rejects(store.gather(a.info.name, [0], 1, new Float64Array(64)), /undersized F32 destination/);
  await assert.rejects(store.gather("missing", [0]), /is incomplete/);
  assert.deepEqual(await store.gather(a.info.name, [], 1, null, 2), new Float32Array(2));
  assert.equal(f.reads.length, 0);
  assert.deepEqual(await store.gather(a.info.name, [1], 0), memoryWeights([a]).gatherHostRows(a.info.name, [1], 0));
  await store.dispose();
});

test("asynchronous PLE write failures are observed and remove the owned file", async (t) => {
  const error = new DOMException("PLE disk is full", "QuotaExceededError");
  const f = fixture(t, { writeError: error });
  const a = tensor();
  const store = await createGemma4PleStore([a.info]);
  stream(store, a, "data", 7);
  stream(store, a, "scales", 3);
  await new Promise((resolve) => setImmediate(resolve));
  await assert.rejects(store.flush(), (caught) => caught === error);
  assert.equal(f.writes.length, 1);
  assert.equal(f.stats().aborts, 1);
  assert.deepEqual(f.removed, [store.resource.name]);
  await store.dispose();
  assert.equal(f.files.size, 1);
});

test("PLE cancellation waits for the active write and skips queued copies", async (t) => {
  let release, started;
  const waiting = new Promise((resolve) => { release = resolve; });
  const writing = new Promise((resolve) => { started = resolve; });
  const f = fixture(t, { beforeWrite: async () => { started(); await waiting; } });
  const a = tensor();
  const store = await createGemma4PleStore([a.info]);
  store.write(a.info.name, "data", 0, Uint8Array.of(1));
  await writing;
  store.write(a.info.name, "data", 1, Uint8Array.of(2));
  const disposal = store.dispose();
  assert.equal(f.removed.length, 0);
  release();
  await disposal;
  assert.equal(f.writes.length, 1);
  assert.deepEqual(f.stats(), { opens: 1, closes: 0, aborts: 1, snapshots: 0 });
  assert.deepEqual(f.removed, [store.resource.name]);
  await assert.rejects(store.flush(), /is disposed/);
  await assert.rejects(store.finish(), /is disposed/);
  await assert.rejects(store.gather(a.info.name, [0]), /is disposed/);
  assert.throws(() => store.write(a.info.name, "data", 2, Uint8Array.of(3)), /is disposed/);
});

test("PLE setup, finalization, and read failures propagate with cleanup", async (t) => {
  for (const kind of ["openError", "closeError", "getFileError", "readError", "shortFile", "shortRead"]) {
    await t.test(kind, async (t) => {
      const error = new Error(`OPFS ${kind} failed`);
      const f = fixture(t, { [kind]: kind.startsWith("short") ? true : error });
      const a = tensor();
      let store;
      const operation = async () => {
        store = await createGemma4PleStore([a.info]);
        stream(store, a, "data", 32);
        stream(store, a, "scales", 4);
        await store.finish();
        await store.gather(a.info.name, [1]);
      };
      await assert.rejects(operation(), kind === "shortFile" ? /host PLE file has/ : kind === "shortRead" ? /incomplete stored rows/ : (caught) => caught === error);
      assert.equal(f.removed.length, 1);
      assert.equal(f.files.size, 1);
      await store?.dispose();
    });
  }
});

test("PLE cleanup still removes the tempfile after an abort failure", async (t) => {
  const error = new Error("OPFS abort failed");
  const f = fixture(t, { abortError: error });
  const store = await createGemma4PleStore([tensor().info]);
  await assert.rejects(store.dispose(), (caught) => caught === error);
  assert.deepEqual(f.removed, [store.resource.name]);
  assert.equal(f.files.size, 1);
});

test("PLE store falls back and removes its file when writable handles are unavailable", async (t) => {
  const f = fixture(t, { noWritable: true });
  assert.equal(await createGemma4PleStore([tensor().info]), null);
  assert.equal(f.removed.length, 1);
  assert.equal(f.files.size, 1);
});

test("PLE filenames preserve existing files and bound collision retries", async (t) => {
  const f = fixture(t);
  const existingName = "kevala-ple-collision.tmp";
  f.files.set(existingName, f.existing);
  let calls = 0;
  replaceGlobal(t, "crypto", { randomUUID: () => calls++ === 0 ? "collision" : "unique" });
  const store = await createGemma4PleStore([tensor().info]);
  assert.equal(store.resource.name, "kevala-ple-unique.tmp");
  await store.dispose();
  assert.deepEqual(f.files.get(existingName), f.existing);
  replaceGlobal(t, "crypto", { randomUUID: () => "collision" });
  await assert.rejects(createGemma4PleStore([tensor().info]), /could not allocate a unique file/);
  assert.deepEqual(f.removed, ["kevala-ple-unique.tmp"]);
});

test("PLE failure reports cleanup errors without hiding the original I/O error", async (t) => {
  const writeError = new Error("OPFS write failed");
  const removeError = new Error("OPFS remove failed");
  const f = fixture(t, { writeError, removeError });
  const a = tensor();
  const store = await createGemma4PleStore([a.info]);
  store.write(a.info.name, "data", 0, Uint8Array.of(1));
  await assert.rejects(store.flush(), (error) => {
    assert.ok(error instanceof AggregateError);
    assert.equal(error.cause, writeError);
    assert.deepEqual(error.errors, [writeError, removeError]);
    return true;
  });
  assert.deepEqual(f.removed, [store.resource.name]);
  assert.deepEqual(f.files.get("existing-model.kevala"), f.existing);
});

test("PLE store caps queued copies before allocating an oversized fragment", async (t) => {
  const f = fixture(t);
  const size = 129 * 1024 * 1024;
  const info = { ...tensor().info, shape: [size / 32, 32], size, scales_size: size / 8 };
  const store = await createGemma4PleStore([info]);
  assert.throws(() => store.write(info.name, "data", 0, new Uint8Array(size)), /must flush before queuing more than 128 MiB/);
  assert.equal(f.writes.length, 0);
  await store.dispose();
});

test("PLE gathers bound read windows even for a row wider than the read budget", async (t) => {
  const f = fixture(t);
  const a = tensor("ple.0.embed", 2, 1024 * 1024 + 32);
  const store = await createGemma4PleStore([a.info]);
  stream(store, a, "data", 128 * 1024);
  stream(store, a, "scales", 128 * 1024);
  await store.finish();
  assert.deepEqual(await store.gather(a.info.name, [1, 1]), memoryWeights([a]).gatherHostRows(a.info.name, [1, 1]));
  assert.equal(f.reads.length, 4);
  for (let i = 0; i < f.reads.length; i += 2) {
    const bytes = f.reads[i][1] - f.reads[i][0] + f.reads[i + 1][1] - f.reads[i + 1][0];
    assert.ok(bytes <= 1024 * 1024);
  }
  await store.dispose();
});

test("PLE synchronous access is restricted to dedicated workers and falls back when absent", async (t) => {
  for (const context of ["window", "other-worker", "dedicated-without-sync"]) {
    await t.test(context, async (t) => {
      if (context === "window") replaceGlobal(t, "DedicatedWorkerGlobalScope", undefined);
      else dedicatedWorker(t, context === "dedicated-without-sync");
      const f = fixture(t, { sync: context !== "dedicated-without-sync" });
      const a = tensor();
      const store = await createGemma4PleStore([a.info]);
      stream(store, a, "data", 32);
      stream(store, a, "scales", 4);
      await store.finish();
      assert.deepEqual(await store.gather(a.info.name, [1]), memoryWeights([a]).gatherHostRows(a.info.name, [1]));
      assert.deepEqual(f.syncStats(), { opens: 0, closes: 0, flushes: 0 });
      assert.equal(f.stats().opens, 1);
      await store.dispose();
    });
  }
});

test("dedicated-worker PLE storage loops over partial writes and reads with bit-exact duplicates", async (t) => {
  dedicatedWorker(t);
  const f = fixture(t, { sync: true, noWritable: true, partialWrite: 3, partialRead: 5 });
  const a = tensor();
  const b = tensor("ple.1.embed", 4, 32);
  const store = await createGemma4PleStore([b.info, a.info]);
  stream(store, a, "data", 7);
  stream(store, b, "scales", 3);
  await store.flush();
  stream(store, b, "data", 11);
  stream(store, a, "scales", 3);
  await store.finish();
  await store.finish();
  assert.deepEqual(f.stats(), { opens: 0, closes: 0, aborts: 0, snapshots: 0 });
  assert.deepEqual(f.syncStats(), { opens: 1, closes: 0, flushes: 2 });
  const base = b.info.size + b.info.scales_size;
  assert.deepEqual(f.syncWrites.slice(0, 3), [[base, base + 7], [base + 3, base + 7], [base + 6, base + 7]]);
  const expectedBytes = Buffer.concat([Buffer.from(b.data.buffer), Buffer.from(b.scales.buffer), Buffer.from(a.data.buffer), Buffer.from(a.scales.buffer)]);
  assert.deepEqual(Buffer.from(f.files.get(store.resource.name)), expectedBytes);
  const ids = new Int32Array([2, 0, 2, 1, 0]);
  const factor = 1.123456789;
  const target = new Float32Array(5 + ids.length * 64 + 3).fill(100);
  const expected = memoryWeights([a]).gatherHostRows(a.info.name, ids, factor, target.slice(), 5);
  const gathering = store.gather(a.info.name, ids, factor, target, 5);
  assert.ok(f.syncReads.length > 0, "the worker reads rows before yielding its gather promise");
  assert.equal(await gathering, target);
  assert.deepEqual(target, expected);
  const partialRanges = (at, length) => Array.from({ length: Math.ceil(length / 5) }, (_, i) => [at + i * 5, at + length]);
  assert.deepEqual(f.syncReads, [2, 0, 1].flatMap((id) => [
    ...partialRanges(base + id * 64, 64), ...partialRanges(base + a.info.size + id * 8, 8),
  ]));
  assert.equal(f.syncReadBuffers.size, 2, "all selected rows reuse the same data and scale buffers");
  assert.equal(f.reads.length, 0);
  await store.dispose();
  await store.dispose();
  assert.equal(f.syncStats().closes, 1);
  assert.deepEqual(f.removed, [store.resource.name]);
  assert.deepEqual(f.files, new Map([["existing-model.kevala", f.existing]]));
});

test("synchronous PLE storage failures close the descriptor and remove its exact tempfile", async (t) => {
  for (const kind of ["syncOpenError", "syncTruncateError", "syncWriteError", "syncFlushError", "syncSizeError", "syncShortFile", "syncReadError", "syncCloseError"]) {
    await t.test(kind, async (t) => {
      dedicatedWorker(t);
      const error = new Error(`OPFS ${kind} failed`);
      const f = fixture(t, { sync: true, [kind]: kind === "syncShortFile" ? true : error });
      const a = tensor();
      let store;
      const operation = async () => {
        store = await createGemma4PleStore([a.info]);
        stream(store, a, "data", 32);
        stream(store, a, "scales", 4);
        if (kind === "syncFlushError") await store.flush();
        await store.finish();
        await store.gather(a.info.name, [1]);
        await store.dispose();
      };
      await assert.rejects(operation(), kind === "syncShortFile" ? /host PLE file has/ : (caught) => caught === error);
      assert.equal(f.syncStats().closes, kind === "syncOpenError" ? 0 : 1);
      assert.equal(f.removed.length, 1);
      assert.equal(f.files.size, 1);
      assert.equal(f.stats().opens, 0);
    });
  }
});

test("synchronous PLE I/O rejects zero progress and impossible byte counts", async (t) => {
  for (const method of ["write", "read"]) {
    for (const count of [0, 1000]) {
      await t.test(`${method} returning ${count}`, async (t) => {
        dedicatedWorker(t);
        const f = fixture(t, { sync: true, [method === "write" ? "syncWriteCount" : "syncReadCount"]: count });
        const a = tensor();
        const store = await createGemma4PleStore([a.info]);
        stream(store, a, "data", 32);
        stream(store, a, "scales", 4);
        const operation = async () => {
          await store.finish();
          await store.gather(a.info.name, [1]);
        };
        await assert.rejects(operation(), new RegExp(`host PLE file ${method} returned ${count} bytes`));
        assert.equal(f.syncStats().closes, 1);
        assert.deepEqual(f.removed, [store.resource.name]);
        assert.equal(f.files.size, 1);
      });
    }
  }
});

test("synchronous PLE cancellation skips queued writes and closes its descriptor", async (t) => {
  dedicatedWorker(t);
  const f = fixture(t, { sync: true });
  const a = tensor();
  const store = await createGemma4PleStore([a.info]);
  stream(store, a, "data", 7);
  stream(store, a, "scales", 3);
  await store.dispose();
  assert.equal(f.syncWrites.length, 0);
  assert.equal(f.syncStats().closes, 1);
  assert.deepEqual(f.removed, [store.resource.name]);
  await assert.rejects(store.finish(), /is disposed/);
});

test("synchronous PLE gathers reuse bounded windows across a wide row", async (t) => {
  dedicatedWorker(t);
  const f = fixture(t, { sync: true });
  const a = tensor("ple.0.embed", 2, 1024 * 1024 + 32);
  const store = await createGemma4PleStore([a.info]);
  stream(store, a, "data", 128 * 1024);
  stream(store, a, "scales", 128 * 1024);
  await store.finish();
  assert.deepEqual(await store.gather(a.info.name, [1, 1]), memoryWeights([a]).gatherHostRows(a.info.name, [1, 1]));
  assert.equal(f.syncReads.length, 4);
  assert.equal(f.syncReadBuffers.size, 2);
  for (let i = 0; i < f.syncReads.length; i += 2) {
    const bytes = f.syncReads[i][1] - f.syncReads[i][0] + f.syncReads[i + 1][1] - f.syncReads[i + 1][0];
    assert.ok(bytes <= 1024 * 1024);
  }
  await store.dispose();
});
