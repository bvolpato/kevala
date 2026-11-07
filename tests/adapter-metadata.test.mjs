import assert from "node:assert/strict";
import { test } from "node:test";
import { adapterFallbackMetadata } from "../dev/adapter-metadata.js";
import { adapterKey } from "../scripts/lib/model-profile.mjs";

test("missing or non-boolean fallback flags remain unknown and fail hardware validation", () => {
  const metadata = adapterFallbackMetadata({ info: { isFallbackAdapter: "false" } });
  assert.deepEqual(metadata, {
    isFallbackAdapter: null,
    fallbackAdapterSource: "unavailable",
    fallbackAdapterFlags: { adapterInfo: null, adapter: null },
  });
  assert.throws(() => adapterKey({ name: "hardware", features: ["timestamp-query"], ...metadata }), /Hardware WebGPU adapter is required/);
});

test("an explicit false from adapter info is preserved when the adapter flag is absent", () => {
  assert.deepEqual(adapterFallbackMetadata({ info: { isFallbackAdapter: false } }), {
    isFallbackAdapter: false,
    fallbackAdapterSource: "adapter.info",
    fallbackAdapterFlags: { adapterInfo: false, adapter: null },
  });
});

test("an explicit true from the adapter is preserved when adapter info is absent", () => {
  assert.deepEqual(adapterFallbackMetadata({ isFallbackAdapter: true }), {
    isFallbackAdapter: true,
    fallbackAdapterSource: "adapter",
    fallbackAdapterFlags: { adapterInfo: null, adapter: true },
  });
});

test("conflicting boolean flags fail closed while preserving both observations", () => {
  assert.deepEqual(adapterFallbackMetadata({ info: { isFallbackAdapter: false }, isFallbackAdapter: true }), {
    isFallbackAdapter: null,
    fallbackAdapterSource: "conflict",
    fallbackAdapterFlags: { adapterInfo: false, adapter: true },
  });
});
