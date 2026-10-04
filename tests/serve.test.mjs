import assert from "node:assert/strict";
import { join, resolve, sep } from "node:path";
import { test } from "node:test";
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
