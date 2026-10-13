import assert from "node:assert/strict";
import { get as httpGet } from "node:http";
import { mkdtemp, mkdir, rm, truncate, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve, sep } from "node:path";
import { test } from "node:test";
import { createKevalaServer } from "../scripts/serve.mjs";
import { resolveRequestPath } from "../scripts/lib/static-path.mjs";

const root = resolve("/srv/kevala");
const at = (...parts) => ({ path: join(root, ...parts) });

test("request paths resolve inside the served directory", () => {
  assert.deepEqual(resolveRequestPath(root, "/"), { path: root + sep });
  assert.deepEqual(resolveRequestPath(root, "/js/src/index.js"), at("js", "src", "index.js"));
  assert.deepEqual(resolveRequestPath(root, "/docs/a%20b.md"), at("docs", "a b.md"));
  assert.deepEqual(resolveRequestPath(root, "/app/../index.html"), at("index.html"));
});

test("a path that leaves the served directory is refused", () => {
  for (const pathname of ["/../secret", "/%2e%2e/secret", "/app/..%2F..%2Fsecret", "/..%2F..%2Fetc%2Fpasswd"]) {
    assert.deepEqual(resolveRequestPath(root, pathname), { status: 403 }, pathname);
  }
  // A sibling directory whose name only starts with the root's name is outside it.
  assert.deepEqual(resolveRequestPath(root, "/..%2Fkevala-private/key.pem"), { status: 403 });
  assert.deepEqual(resolveRequestPath(root, "/..%2Fkevala/README.md"), at("README.md"), "the root itself, reached by its own name");
});

test("a path that cannot be decoded is a bad request, not a crash", () => {
  assert.deepEqual(resolveRequestPath(root, "/%zz"), { status: 400 });
  assert.deepEqual(resolveRequestPath(root, "/%E0%A4%A"), { status: 400 });
  assert.deepEqual(resolveRequestPath(root, "/a%00b"), { status: 400 });
});

async function startServer(directory, models) {
  const server = createKevalaServer({ root: directory, models });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  return { server, url: `http://127.0.0.1:${server.address().port}` };
}

async function closeServer(server) {
  await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
}

test("local pack manifest discovers exact catalog packs and serves ranges through aliases", async () => {
  const directory = await mkdtemp(join(tmpdir(), "kevala-serve-"));
  const packDirectory = join(directory, "tmp");
  await mkdir(packDirectory);
  await writeFile(join(directory, "index.html"), "local app");
  const models = {
    "fixture-model": {
      precisionPackBytes: { bf16: 12, q8: 4 },
      precisionPackApprox: { f32: true },
      vision: { arch: "gemma4-vision", precision: "bf16", pack: 8 },
    },
  };
  const { server, url } = await startServer(directory, models);

  try {
    const manifestUrl = `${url}/.kevala/models.json`;
    assert.equal((await fetch(manifestUrl)).status, 404, "an empty staged site has no local model manifest");

    await writeFile(join(packDirectory, "fixture-model-bf16.kevala"), "abcdefghijkl");
    await writeFile(join(packDirectory, "fixture-model-q8.kevala"), "old");
    await writeFile(join(packDirectory, "unknown-model-bf16.kevala"), "unregistered");

    const response = await fetch(manifestUrl);
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      models: {
        "fixture-model": {
          bf16: { url: "/.kevala/packs/fixture-model-bf16.kevala", bytes: 12 },
        },
      },
    }, "only exact-size registered packs are advertised");

    const alias = `${url}/.kevala/packs/fixture-model-bf16.kevala`;
    const head = await fetch(alias, { method: "HEAD" });
    assert.equal(head.status, 200);
    assert.equal(head.headers.get("content-length"), "12");
    assert.equal(head.headers.get("content-type"), "application/octet-stream");

    const clamped = await fetch(alias, { headers: { range: "bytes=2-999" } });
    assert.equal(clamped.status, 206);
    assert.equal(clamped.headers.get("content-range"), "bytes 2-11/12");
    assert.equal(clamped.headers.get("content-length"), "10");
    assert.equal(await clamped.text(), "cdefghijkl");

    const suffix = await fetch(alias, { headers: { range: "bytes=-3" } });
    assert.equal(suffix.status, 206);
    assert.equal(suffix.headers.get("content-range"), "bytes 9-11/12");
    assert.equal(await suffix.text(), "jkl");

    const largeEnd = await fetch(alias, { headers: { range: "bytes=0-999999999999999999999999" } });
    assert.equal(largeEnd.status, 206);
    assert.equal(largeEnd.headers.get("content-range"), "bytes 0-11/12");

    const largeSuffix = await fetch(alias, { headers: { range: "bytes=-999999999999999999999999" } });
    assert.equal(largeSuffix.status, 206);
    assert.equal(largeSuffix.headers.get("content-range"), "bytes 0-11/12");

    for (const range of ["bytes=12-", "bytes=8-4", "bytes=-0", "bytes=0-1,4-5"]) {
      const invalid = await fetch(alias, { headers: { range } });
      assert.equal(invalid.status, 416, range);
      assert.equal(invalid.headers.get("content-range"), "bytes */12", range);
      assert.equal(invalid.headers.get("content-length"), "0", range);
    }

    assert.equal((await fetch(`${url}/.kevala/packs/unknown-model-bf16.kevala`)).status, 404);
    assert.equal((await fetch(`${url}/index.html`)).status, 200, "normal static content remains available");

    // The inventory is rescanned for every manifest request, so a converted pack can appear
    // while this server is already running.
    await writeFile(join(packDirectory, "fixture-model-q8.kevala"), "1234");
    const refreshed = await (await fetch(manifestUrl)).json();
    assert.deepEqual(refreshed.models["fixture-model"].q8, {
      url: "/.kevala/packs/fixture-model-q8.kevala",
      bytes: 4,
    });
    await writeFile(join(packDirectory, "fixture-model-vision-bf16.kevala"), "partial");
    assert.equal((await (await fetch(manifestUrl)).json()).models["fixture-model"].vision, undefined);
    await writeFile(join(packDirectory, "fixture-model-vision-bf16.kevala"), "12345678");
    assert.deepEqual((await (await fetch(manifestUrl)).json()).models["fixture-model"].vision, {
      url: "/.kevala/packs/fixture-model-vision-bf16.kevala", bytes: 8, precision: "bf16",
    });
    const vision = await fetch(`${url}/.kevala/packs/fixture-model-vision-bf16.kevala`, { headers: { range: "bytes=2-4" } });
    assert.equal(vision.status, 206);
    assert.equal(await vision.text(), "345");
  } finally {
    await closeServer(server);
    await rm(directory, { recursive: true, force: true });
  }
});

test("cancelling a model pack response leaves the static server usable", async () => {
  const directory = await mkdtemp(join(tmpdir(), "kevala-serve-cancel-"));
  const packDirectory = join(directory, "tmp");
  const packBytes = 32 * 1024 * 1024;
  await mkdir(packDirectory);
  await writeFile(join(directory, "index.html"), "still serving");
  const packPath = join(packDirectory, "fixture-model-bf16.kevala");
  await writeFile(packPath, "");
  await truncate(packPath, packBytes);
  const models = { "fixture-model": { precisionPackBytes: { bf16: packBytes } } };
  const { server, url } = await startServer(directory, models);

  try {
    await new Promise((resolve, reject) => {
      const request = httpGet(`${url}/.kevala/packs/fixture-model-bf16.kevala`, {
        headers: { range: `bytes=0-${packBytes - 1}` },
      }, (response) => {
        response.once("error", () => {});
        response.once("data", () => {
          response.destroy();
          request.destroy();
          resolve();
        });
      });
      request.once("error", reject);
      request.setTimeout(5000, () => request.destroy(new Error("timed out waiting for pack data")));
    });
    const stillServing = await fetch(`${url}/index.html`);
    assert.equal(stillServing.status, 200);
    assert.equal(await stillServing.text(), "still serving");
  } finally {
    await closeServer(server);
    await rm(directory, { recursive: true, force: true });
  }
});
