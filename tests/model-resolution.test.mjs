import assert from "node:assert/strict";
import test from "node:test";
import { MODELS, openPack, resolveModel } from "../js/src/source.js";

function replaceFetch(t, fn) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, "fetch");
  Object.defineProperty(globalThis, "fetch", { configurable: true, writable: true, value: fn });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, "fetch", previous);
    else delete globalThis.fetch;
  });
}

test("custom specs keep their own architecture and source metadata", async (t) => {
  const calls = [];
  replaceFetch(t, async (url) => {
    throw new Error(`unexpected network request for ${url}`);
  });
  for (const spec of [
    { name: "custom-gemma", arch: "gemma4", repo: "example/gemma", revision: "gemma-rev", block: 32 },
    { name: "custom-qwen", arch: "kev", repo: "example/qwen", revision: "qwen-rev", block: 32 },
    { name: "custom-laya", repo: "example/laya", revision: "laya-rev", block: 32 },
  ]) {
    const resolved = resolveModel(spec);
    assert.deepEqual(resolved, { spec });
    const pack = await openPack(spec, {
      cache: false,
      convert: async (received) => {
        calls.push(received);
        return Uint8Array.of(1, 2, 3);
      },
    });
    assert.equal(pack.expectedArch, spec.arch);
  }
  assert.deepEqual(calls, [
    { name: "custom-gemma", arch: "gemma4", repo: "example/gemma", revision: "gemma-rev", block: 32 },
    { name: "custom-qwen", arch: "kev", repo: "example/qwen", revision: "qwen-rev", block: 32 },
    { name: "custom-laya", repo: "example/laya", revision: "laya-rev", block: 32 },
  ]);
});

test("known source overrides do not retain hosted pack provenance", () => {
  const resolved = resolveModel({
    name: "laya",
    arch: "laya",
    repo: "example/custom-laya",
    revision: "custom-revision",
  }).spec;
  assert.equal(resolved.arch, "laya");
  assert.equal(resolved.repo, "example/custom-laya");
  assert.equal(resolved.revision, "custom-revision");
  assert.equal(resolved.hosted, undefined);
  assert.equal(resolved.download, undefined);
  assert.equal(resolved.pack, undefined);
  assert.equal(resolved.packSha256, undefined);
  assert.equal(resolved.author, undefined);
  assert.equal(resolved.license, undefined);
  assert.equal(resolved.browserConvert, true);
});

test("an explicit empty hosted value falls back to conversion", async () => {
  let received;
  const pack = await openPack({ name: "laya", hosted: null }, {
    cache: false,
    from: "checkpoint",
    convert: async (spec) => {
      received = spec;
      return Uint8Array.of(1, 2, 3);
    },
  });
  assert.equal(received.hosted, null);
  assert.equal(pack.expectedArch, "laya");
});

test("known names reject an explicitly conflicting architecture", () => {
  assert.throws(
    () => resolveModel({ name: "gemma-4-e2b", arch: "laya" }),
    /model "gemma-4-e2b" uses architecture "gemma4", not "laya"/,
  );
});

test("distinct custom hosted sources use distinct cache entries", async (t) => {
  const previousCaches = Object.getOwnPropertyDescriptor(globalThis, "caches");
  const cacheKeys = [];
  const cache = {
    async match(key) {
      cacheKeys.push(["match", key]);
      return null;
    },
    async put(key, response) {
      cacheKeys.push(["put", key]);
      await response.arrayBuffer();
    },
    async keys() { return []; },
    async delete() { return true; },
  };
  Object.defineProperty(globalThis, "caches", { configurable: true, value: { open: async () => cache } });
  t.after(() => {
    if (previousCaches) Object.defineProperty(globalThis, "caches", previousCaches);
    else delete globalThis.caches;
  });
  replaceFetch(t, async (_url, options = {}) => {
    if (options.method === "HEAD") return new Response(null, { headers: { "content-length": "3" } });
    if (options.headers?.range) return new Response(Uint8Array.of(1, 2, 3), { status: 206 });
    throw new Error("unexpected non-range request");
  });

  for (const hosted of ["https://example.test/a.kevala", "https://example.test/b.kevala"]) {
    const pack = await openPack({ name: hosted, arch: "laya", hosted, block: 32 });
    for await (const _chunk of pack.chunks()) {}
    await pack.saving;
  }
  const matches = cacheKeys.filter(([kind]) => kind === "match").map(([, key]) => key);
  assert.deepEqual(matches, ["https://example.test/a.kevala", "https://example.test/b.kevala"]);
});

test("default and registered model names retain their existing specs", async () => {
  assert.equal(resolveModel(null).spec, MODELS.laya);
  assert.equal(resolveModel("laya").spec, MODELS.laya);
  assert.equal(resolveModel("gemma-4-e2b").spec, MODELS["gemma-4-e2b"]);

  let converted;
  const pack = await openPack(null, {
    cache: false,
    from: "checkpoint",
    convert: async (spec) => {
      converted = spec;
      return Uint8Array.of(1, 2, 3);
    },
  });
  assert.equal(converted, MODELS.laya);
  assert.equal(pack.expectedArch, "laya");
});

const previousSelf = Object.getOwnPropertyDescriptor(globalThis, "self");
Object.defineProperty(globalThis, "self", { configurable: true, writable: true, value: { postMessage() {} } });
const { validatePackArchitecture } = await import("../js/src/engine-worker.js?model-resolution-test");
if (previousSelf) Object.defineProperty(globalThis, "self", previousSelf);
else delete globalThis.self;

test("pack architecture is rejected before execution when unknown", () => {
  assert.throws(
    () => validatePackArchitecture({ config: { arch: "unknown-family" } }),
    (error) => error.code === "ARCH_UNSUPPORTED" && /unknown pack architecture "unknown-family" in config\.arch/.test(error.message),
  );
  assert.throws(
    () => validatePackArchitecture({ config: { arch: "laya" } }, "gemma4"),
    (error) => error.code === "ARCH_MISMATCH" && /model spec arch "gemma4" does not match pack config\.arch "laya"/.test(error.message),
  );
  assert.equal(validatePackArchitecture({ config: {} }).arch, "laya");
});
