import assert from "node:assert/strict";
import test from "node:test";
import { validatePackArchitecture } from "../js/src/archs/index.js";
import { assertPackPrecision, readPackHead } from "../js/src/pack-layout.js";
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

test("an explicit empty hosted value falls back to conversion", async (t) => {
  replaceFetch(t, async (url) => {
    throw new Error(`unexpected network request for ${url}`);
  });
  let received;
  const pack = await openPack({ name: "laya", hosted: null }, {
    cache: false,
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

test("refreshed and added models default to unquantized weights without fake hosted packs", async (t) => {
  for (const [name, revision, arch, readout] of [
    ["kev-0.8b", "9a45d25eb2ab761841196625383fa1dff0e56c1e", "kev", undefined],
    ["kev-4b", "139fdd94f1b6a6ad80cc15e08fcb99cac885a101", "kev", undefined],
    ["kev-9b", "b5d8c18e44c60888d138b65cb6507ff0a5a448a0", "kev", undefined],
    ["h2o-lightning-4b", "acaf0d4ea251e54de928c75ef4352670d33192d3", "kev", "h2o-lightning"],
    ["clef-flash", "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3", "clef", undefined],
  ]) {
    const spec = MODELS[name];
    assert.equal(spec.revision, revision);
    assert.equal(spec.arch, arch);
    assert.equal(spec.readout, readout);
    assert.equal(spec.defaultPrecision, "bf16");
    assert.equal(spec.hosted, undefined);
    assert.equal(spec.pack, undefined);
    assert.equal(spec.packSha256, undefined);
  }
  assert.equal(MODELS["kev-0.8b"].precisionPackBytes.bf16, 1514645952);
  assert.equal(MODELS["kev-0.8b"].precisionPackBytes.q8, 857259648);
  assert.equal(MODELS["kev-4b"].precisionPackBytes.bf16, 8432127168);
  assert.equal(MODELS["kev-4b"].precisionPackBytes.q8, 4756384256);
  assert.equal(MODELS["kev-9b"].precisionPackBytes.bf16, 15902057920);
  assert.equal(MODELS["kev-9b"].precisionPackBytes.q8, 8963900032);
  assert.equal(MODELS["h2o-lightning-4b"].precisionPackBytes.bf16, 8429496384);
  assert.equal(MODELS["h2o-lightning-4b"].precisionPackBytes.q8, 4753753472);
  assert.equal(MODELS["clef-flash"].precisionPackBytes.bf16, 18414969408);
  assert.equal(MODELS["clef-flash"].precisionPackBytes.q8, 10586832640);

  let fetched = false;
  replaceFetch(t, async (url) => {
    fetched = true;
    throw new Error(`unexpected fetch for ${url}`);
  });
  await assert.rejects(openPack("kev-4b", { cache: false }), /no published BF16 pack and cannot be converted in the browser/);
  assert.equal(fetched, false, "the unavailable large checkpoint must fail before network access");
});

test("browser conversion receives explicit precision and rejects oversized output before download", async (t) => {
  let receivedPrecision;
  const pack = await openPack({
    name: "small",
    repo: "example/small",
    revision: "revision",
    arch: "kev",
    block: 32,
    browserConvert: true,
    defaultPrecision: "bf16",
    precisions: ["bf16", "q8"],
  }, {
    cache: false,
    precision: "q8",
    convert: async (_spec, options) => {
      receivedPrecision = options.precision;
      return Uint8Array.of(1, 2, 3);
    },
  });
  assert.equal(receivedPrecision, "q8");
  assert.equal(pack.precision, "q8");

  let fetched = false;
  replaceFetch(t, async (url) => {
    fetched = true;
    throw new Error(`unexpected fetch for ${url}`);
  });
  await assert.rejects(openPack("kev-0.8b", { cache: false, precision: "f32" }), (error) =>
    error.code === "BROWSER_CONVERSION_TOO_LARGE" && /exceeds the browser conversion limit/.test(error.message));
  assert.equal(fetched, false, "an oversized browser conversion must fail before downloading weights");
});

test("known source metadata overrides retain BF16 defaults and optional quantization", async () => {
  const model = { name: "kev-0.8b", download: 123 };
  const spec = resolveModel(model).spec;
  assert.equal(spec.defaultPrecision, "bf16");
  assert.deepEqual(spec.precisions, ["bf16", "f32", "q8"]);
  assert.equal(spec.precisionPackBytes, undefined, "the old source's pack sizes must be cleared");
  for (const [requested, expected] of [[undefined, "bf16"], ["bf16", "bf16"], ["q8", "q8"]]) {
    let convertedPrecision;
    const pack = await openPack(model, {
      cache: false,
      precision: requested,
      convert: async (_spec, options) => {
        convertedPrecision = options.precision;
        return Uint8Array.of(1, 2, 3);
      },
    });
    assert.equal(convertedPrecision, expected);
    assert.equal(pack.precision, expected);
  }
});

test("byte and Blob packs enforce an explicit precision at header validation", async () => {
  const header = { config: { arch: "kev" }, model: { precision: "q8" }, tensors: [{ dtype: "q8" }] };
  const json = new TextEncoder().encode(JSON.stringify(header));
  const bytes = new Uint8Array(16 + json.byteLength);
  bytes.set(new TextEncoder().encode("KVLA"));
  const view = new DataView(bytes.buffer);
  view.setUint32(4, 1, true);
  view.setUint32(8, json.byteLength, true);
  bytes.set(json, 16);
  const shifted = new Uint8Array(bytes.length + 5);
  shifted.set(bytes, 5);
  for (const model of [bytes.buffer, bytes, shifted.subarray(5), new Blob([bytes])]) {
    const mismatched = await openPack(model, { cache: false, precision: "bf16" });
    const parsed = await readPackHead(mismatched.chunks());
    assert.throws(() => assertPackPrecision(parsed.header, mismatched.expectedPrecision), { code: "PRECISION_MISMATCH" });
    const matching = await openPack(model, { cache: false, precision: "q8" });
    assert.doesNotThrow(() => assertPackPrecision(parsed.header, matching.expectedPrecision));
    const implicit = await openPack(model, { cache: false });
    assert.equal(implicit.expectedPrecision, undefined, "an omitted precision uses the pack's own format");
  }
});

test("a named model URL retains its architecture and treats explicit precision as an expectation", async (t) => {
  const model = { name: "h2o-lightning-4b", url: "https://example.test/h2o-bf16.kevala" };
  assert.equal(resolveModel(model).spec.arch, "kev");
  replaceFetch(t, async (url, options = {}) => {
    if (options.method === "HEAD") return new Response(null, { headers: { "content-length": "3", "accept-ranges": "bytes" } });
    if (options.headers?.range) return new Response(Uint8Array.of(1, 2, 3), { status: 206 });
    throw new Error(`unexpected request ${url}`);
  });
  const pack = await openPack(model, { cache: false, precision: "bf16" });
  assert.equal(pack.expectedArch, "kev");
  assert.equal(pack.precision, undefined, "precision is embedded in the pack URL and must not be inferred from the selector");
  assert.equal(pack.expectedPrecision, "bf16", "an explicit precision request is validated against the loaded header");
  const chunks = [];
  for await (const chunk of pack.chunks()) chunks.push(...chunk);
  assert.deepEqual(chunks, [1, 2, 3]);
  const selfDescribing = await openPack(model, { cache: false });
  assert.equal(selfDescribing.expectedPrecision, undefined, "without a precision request, a pack URL may use its embedded format");
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
