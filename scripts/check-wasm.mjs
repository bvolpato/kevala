#!/usr/bin/env node
// Check the generated WebAssembly contract used by the browser and Node entrypoints: every build
// is valid and exports each function the runtime in js/src calls.
import { readFile } from "node:fs/promises";
import { assertModule, flavorPath, FLAVORS, requiredExports } from "./lib/wasm-abi.mjs";

const required = await requiredExports();
for (const flavor of FLAVORS) {
  const path = flavorPath(flavor);
  let bytes;
  try {
    bytes = await readFile(path);
  } catch (error) {
    throw new Error(`missing generated WebAssembly: ${path} (run scripts/build-wasm.sh)`, { cause: error });
  }
  assertModule(bytes, required, path);
  console.log(`${flavor.padEnd(7)} ${String(bytes.byteLength).padStart(8)} bytes  valid, ${required.length} exports`);
}
