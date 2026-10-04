// Release contract: the three places that state the version agree, and the package exposes the
// entry points the README documents.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { VERSION } from "../js/src/index.js";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("the runtime, the npm package, and the Rust workspace state the same version", async () => {
  const pkg = JSON.parse(await read("package.json"));
  const cargo = (await read("Cargo.toml")).match(/\[workspace\.package\]\s+version = "([^"]+)"/)?.[1];
  assert.equal(VERSION, pkg.version);
  assert.equal(cargo, pkg.version);
});

test("every package entry point resolves to a module with its documented exports", async () => {
  const pkg = JSON.parse(await read("package.json"));
  const entry = async (name) => import(new URL(`../${pkg.exports[name]}`, import.meta.url).href);
  const main = await entry(".");
  for (const name of ["Kevala", "load", "MODELS", "presets", "cacheInfo", "clearCache", "isCached", "VERSION"]) {
    assert.ok(name in main, `kevala does not export ${name}`);
  }
  assert.equal(typeof (await entry("./node")).loadFile, "function");
  assert.ok(Object.keys(await entry("./presets")).length > 0);
});
