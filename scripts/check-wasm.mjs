#!/usr/bin/env node
// Check the generated WebAssembly contract used by the browser and Node entrypoints.
import { readFile } from "node:fs/promises";

const requiredExports = ["memory", "kevala_init", "kevala_prepare", "kevala_decide", "kevala_free"];
const flavors = ["relaxed", "simd", "base"];

for (const flavor of flavors) {
  const file = new URL(`../js/src/kevala-${flavor}.wasm`, import.meta.url);
  let bytes;
  try {
    bytes = await readFile(file);
  } catch (error) {
    throw new Error(`missing generated WebAssembly: ${file.pathname}`, { cause: error });
  }
  if (!WebAssembly.validate(bytes)) throw new Error(`${file.pathname} is not a valid WebAssembly module`);

  const module = new WebAssembly.Module(bytes);
  const exports = new Set(WebAssembly.Module.exports(module).map(({ name }) => name));
  for (const name of requiredExports) {
    if (!exports.has(name)) throw new Error(`${file.pathname} is missing export ${name}`);
  }
  console.log(`${flavor.padEnd(7)} ${String(bytes.byteLength).padStart(8)} bytes  valid`);
}
