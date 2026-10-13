const MAX_PENDING_BYTES = 128 * 1024 * 1024;
const READ_COLUMNS = Math.floor(1024 * 1024 / 36) * 32;

function transfer(handle, method, bytes, at) {
  let offset = 0;
  while (offset < bytes.byteLength) {
    const remaining = bytes.byteLength - offset;
    const count = handle[method](bytes.subarray(offset), { at: at + offset });
    if (!Number.isSafeInteger(count) || count <= 0 || count > remaining) throw new Error(`host PLE file ${method} returned ${count} bytes for a ${remaining}-byte range`);
    offset += count;
  }
}

function tableLayout(tensors) {
  const tables = new Map();
  let byteLength = 0;
  for (const t of tensors) {
    const [rows, width] = t.shape || [];
    if (typeof t.name !== "string" || !t.name || tables.has(t.name) || t.dtype !== "q8" || t.shape?.length !== 2 ||
        !Number.isSafeInteger(rows) || rows < 1 || !Number.isSafeInteger(width) || width < 32 || width % 32 ||
        Number(t.block) !== 32 || !Number.isSafeInteger(rows * width) || t.size !== rows * width ||
        t.scales_size !== rows * (width / 32) * 4 || !Number.isSafeInteger(byteLength + t.size + t.scales_size)) {
      throw new Error(`host row table ${t.name} requires a block-32 Q8 matrix with complete scales`);
    }
    const data = { offset: byteLength, size: t.size, written: 0 };
    byteLength += t.size;
    const scales = { offset: byteLength, size: t.scales_size, written: 0 };
    byteLength += t.scales_size;
    tables.set(t.name, { rows, width, data, scales });
  }
  return { tables, byteLength };
}

async function storageDirectory() {
  const storage = globalThis.navigator?.storage;
  if (typeof storage?.getDirectory !== "function") return null;
  try {
    return await storage.getDirectory();
  } catch (error) {
    if (error.name === "SecurityError" || error.name === "NotSupportedError") return null;
    throw error;
  }
}

/** Allocate on the owner page so worker termination cannot orphan an unknown file. */
export async function createGemma4PleResource() {
  const directory = await storageDirectory();
  if (!directory) return null;
  let name;
  for (let attempt = 0; attempt < 4; attempt++) {
    const candidate = `kevala-ple-${globalThis.crypto.randomUUID()}.tmp`;
    try {
      await directory.getFileHandle(candidate);
    } catch (error) {
      if (error.name === "NotFoundError") {
        name = candidate;
        break;
      }
      throw error;
    }
  }
  if (!name) throw new Error("host PLE store could not allocate a unique file");
  await directory.getFileHandle(name, { create: true });
  return { directory, name };
}

/** Ephemeral, exact Q8 PLE storage. The loader must flush its copied fragments regularly. */
export async function createGemma4PleStore(tensors, resource = undefined) {
  const { tables, byteLength } = tableLayout(tensors);
  resource = resource === undefined ? await createGemma4PleResource() : resource;
  if (!resource) return null;
  const { name } = resource;
  // File-system handles cannot cross worker boundaries in some Safari versions.
  const directory = resource.directory ?? await storageDirectory();
  if (!directory) return null;
  const handle = await directory.getFileHandle(name);
  let writer;
  let syncHandle = null;
  try {
    const worker = typeof globalThis.DedicatedWorkerGlobalScope === "function" && globalThis instanceof globalThis.DedicatedWorkerGlobalScope;
    if (worker && typeof handle.createSyncAccessHandle === "function") {
      syncHandle = await handle.createSyncAccessHandle();
      syncHandle.truncate(0);
    } else if (typeof handle.createWritable === "function") {
      writer = await handle.createWritable();
    } else {
      await directory.removeEntry(name);
      return null;
    }
  } catch (error) {
    const errors = [error];
    try {
      syncHandle?.close();
    } catch (cleanupError) {
      errors.push(cleanupError);
    }
    try {
      await directory.removeEntry(name);
    } catch (cleanupError) {
      errors.push(cleanupError);
    }
    if (errors.length > 1) throw new AggregateError(errors, "PLE file setup and cleanup failed", { cause: error });
    throw error;
  }

  let state = "writing";
  let queue = Promise.resolve();
  let failure = null;
  let pendingBytes = 0;
  let file = null;
  let finishing = null;
  let disposing = null;
  const unavailable = () => new Error("host PLE store is disposed");
  const fail = async () => {
    const error = failure;
    try {
      await store.dispose();
    } catch (cleanupError) {
      throw new AggregateError([error, cleanupError], "PLE storage and cleanup failed", { cause: error });
    }
    throw error;
  };
  const store = {
    resource: { name },
    byteLength,

    write(name, part, offset, bytes) {
      if (state === "disposed") throw unavailable();
      if (failure) throw failure;
      if (state !== "writing") throw new Error("host PLE store is no longer writable");
      const table = tables.get(name);
      const range = (part === "data" || part === "scales") && table?.[part];
      if (!range || !(bytes instanceof Uint8Array) || !Number.isSafeInteger(offset) || offset !== range.written ||
          offset + bytes.byteLength > range.size) throw new Error(`host row table ${name}.${part} has an invalid or out-of-order write`);
      if (pendingBytes + bytes.byteLength > MAX_PENDING_BYTES) throw new Error("host PLE store must flush before queuing more than 128 MiB");
      const copy = bytes.slice();
      range.written += copy.byteLength;
      pendingBytes += copy.byteLength;
      queue = queue.then(async () => {
        if (state === "disposed" || failure) return;
        if (syncHandle) transfer(syncHandle, "write", copy, range.offset + offset);
        else await writer.write({ type: "write", position: range.offset + offset, data: copy });
      }).catch((error) => { failure ||= error; }).finally(() => { pendingBytes -= copy.byteLength; });
    },

    async flush() {
      await queue;
      if (failure) await fail();
      if (state === "disposed") throw unavailable();
      try {
        syncHandle?.flush();
      } catch (error) {
        failure ||= error;
        await fail();
      }
    },

    async finish() {
      if (state === "disposed") throw unavailable();
      if (finishing) return finishing;
      state = "finishing";
      queue = queue.then(async () => {
        if (state === "disposed" || failure) return;
        for (const [name, table] of tables) {
          if (table.data.written !== table.data.size || table.scales.written !== table.scales.size) throw new Error(`host row table ${name} is incomplete`);
        }
        if (syncHandle) {
          syncHandle.flush();
          const size = syncHandle.getSize();
          if (size !== byteLength) throw new Error(`host PLE file has ${size} bytes; expected ${byteLength}`);
        } else {
          await writer.close();
          writer = null;
          const result = await handle.getFile();
          if (result.size !== byteLength) throw new Error(`host PLE file has ${result.size} bytes; expected ${byteLength}`);
          if (state !== "disposed") file = result;
        }
      }).catch((error) => { failure ||= error; });
      finishing = (async () => {
        await queue;
        if (failure) await fail();
        if (state === "disposed") throw unavailable();
        state = "finished";
        return store;
      })();
      return finishing;
    },

    async gather(name, ids, scale = 1, target = null, offset = 0) {
      if (state === "disposed") throw unavailable();
      if (failure) await fail();
      const table = tables.get(name);
      if (!table || state !== "finished") throw new Error(`host row table ${name} is incomplete`);
      if (ids == null || typeof ids[Symbol.iterator] !== "function") throw new Error(`host row table ${name} has an invalid row selection or scale`);
      const selection = Array.from(ids);
      const { rows, width } = table;
      const end = offset + selection.length * width;
      if (!Number.isFinite(scale) || !Number.isSafeInteger(offset) || offset < 0 || !Number.isSafeInteger(end) ||
          selection.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= rows)) throw new Error(`host row table ${name} has an invalid row selection or scale`);
      target ??= new Float32Array(end);
      if (!(target instanceof Float32Array) || end > target.length) throw new Error(`host row table ${name} has an undersized F32 destination`);
      const destinations = new Map();
      for (let row = 0; row < selection.length; row++) {
        const id = selection[row];
        if (!destinations.has(id)) destinations.set(id, []);
        destinations.get(id).push(offset + row * width);
      }
      const factor = Math.fround(scale);
      const syncData = syncHandle ? new Uint8Array(Math.min(width, READ_COLUMNS)) : null;
      const syncScales = syncHandle ? new Uint8Array(syncData.byteLength / 32 * 4) : null;
      try {
        for (const [id, starts] of destinations) {
          for (let col = 0; col < width; col += READ_COLUMNS) {
            if (state === "disposed") throw unavailable();
            const count = Math.min(READ_COLUMNS, width - col);
            const at = table.data.offset + id * width + col;
            const scalesAt = table.scales.offset + (id * width + col) / 32 * 4;
            let q8, factors;
            if (syncHandle) {
              transfer(syncHandle, "read", syncData.subarray(0, count), at);
              transfer(syncHandle, "read", syncScales.subarray(0, count / 32 * 4), scalesAt);
              q8 = new Int8Array(syncData.buffer, 0, count);
              factors = new Float32Array(syncScales.buffer, 0, count / 32);
            } else {
              const [data, scales] = await Promise.all([file.slice(at, at + count).arrayBuffer(), file.slice(scalesAt, scalesAt + count / 32 * 4).arrayBuffer()]);
              if (data.byteLength !== count || scales.byteLength !== count / 32 * 4) throw new Error(`host row table ${name} has incomplete stored rows`);
              q8 = new Int8Array(data);
              factors = new Float32Array(scales);
            }
            if (state === "disposed") throw unavailable();
            for (let i = 0; i < count; i++) {
              const value = Math.fround(Math.fround(q8[i] * factors[Math.floor(i / 32)]) * factor);
              for (const start of starts) target[start + col + i] = value;
            }
          }
        }
      } catch (error) {
        if (state === "disposed") throw error;
        failure ||= error;
        await fail();
      }
      return target;
    },

    async dispose() {
      if (disposing) return disposing;
      state = "disposed";
      disposing = (async () => {
        await queue;
        file = null;
        let error;
        try {
          if (syncHandle) syncHandle.close();
          else if (writer) await writer.abort();
        } catch (caught) {
          error = caught;
        }
        writer = null;
        syncHandle = null;
        try {
          await directory.removeEntry(name);
        } catch (caught) {
          if (caught.name !== "NotFoundError") error ||= caught;
        }
        if (error) throw error;
      })();
      return disposing;
    },
  };
  return store;
}
