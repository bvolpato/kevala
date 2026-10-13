import assert from "node:assert/strict";
import { test } from "node:test";
import { discoverLocalPacks } from "../app/model-packs.js";

const base = "http://127.0.0.1:18086/app/session.js";
const pack = (id, precision, bytes) => ({ url: `/.kevala/packs/${id}-${precision}.kevala`, bytes });

test("the preview discovers both precisions without downloading weights", async () => {
  const calls = [];
  const models = {
    "kev-0.8b": { bf16: pack("kev-0.8b", "bf16", 1514645952), q8: pack("kev-0.8b", "q8", 857259648) },
    laya: { q8: pack("laya", "q8", 478886720) },
  };
  const result = await discoverLocalPacks(base, async (url) => {
    calls.push(url.href);
    return Response.json({ models });
  });
  assert.deepEqual(calls, ["http://127.0.0.1:18086/.kevala/models.json"]);
  assert.deepEqual(result["kev-0.8b"].bf16, {
    url: "http://127.0.0.1:18086/.kevala/packs/kev-0.8b-bf16.kevala", bytes: 1514645952,
  });
  assert.equal(result["kev-0.8b"].q8.bytes, 857259648);
  assert.equal(result.laya.q8.bytes, 478886720);
});

test("public pages retain catalog loading when a local manifest is absent or unavailable", async () => {
  assert.deepEqual(await discoverLocalPacks(base, async () => new Response("not found", { status: 404 })), {});
  assert.deepEqual(await discoverLocalPacks(base, async () => { throw new TypeError("Failed to fetch"); }), {});
});

test("discovery excludes unknown models, unsupported precisions and external or invalid packs", async () => {
  const models = {
    unknown: { bf16: pack("unknown", "bf16", 100) },
    laya: { bf16: pack("laya", "bf16", 100), q8: { url: "https://example.test/weights.kevala", bytes: 100 } },
    "kev-4b": { bf16: { url: "/tmp/other.kevala", bytes: 100 }, q8: { ...pack("kev-4b", "q8", 100), bytes: -1 } },
    "kev-9b": { bf16: pack("kev-9b", "bf16", 15902057920) },
    "clef-flash": { bf16: pack("clef-flash", "bf16", 18414969408) },
  };
  assert.deepEqual(await discoverLocalPacks(base, async () => Response.json({ models })), {});
});
