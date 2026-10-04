// The WebAssembly contract between the Rust build and the JavaScript runtime, for the checks
// that run on a checkout, on the packed npm package, and on the staged site.
import { readdir, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

/** The three builds that ship side by side (see scripts/build-wasm.sh and js/src/wasm.js). */
export const FLAVORS = ["relaxed", "simd", "base"];

const runtime = new URL("../../js/src/", import.meta.url);

/**
 * Every export the runtime uses: `memory`, and each `kevala_*` name that appears in js/src. The
 * list is read from the sources, so a call added to the runtime is checked without editing it.
 */
export async function requiredExports() {
  const names = new Set(["memory"]);
  for (const entry of await readdir(runtime, { recursive: true, withFileTypes: true })) {
    if (!entry.isFile() || !entry.name.endsWith(".js")) continue;
    const source = await readFile(new URL(entry.name, new URL(`${entry.parentPath ?? entry.path}/`, "file://")), "utf8");
    for (const [name] of source.matchAll(/\bkevala_[a-z0-9_]+\b/g)) names.add(name);
  }
  return [...names].sort();
}

/** Throws unless `bytes` is a valid module that exports every name in `required`. */
export function assertModule(bytes, required, label) {
  if (!WebAssembly.validate(bytes)) throw new Error(`${label} is not a valid WebAssembly module`);
  const exported = new Set(WebAssembly.Module.exports(new WebAssembly.Module(bytes)).map(({ name }) => name));
  const missing = required.filter((name) => !exported.has(name));
  if (missing.length) throw new Error(`${label} is missing ${missing.length} export(s) the runtime calls: ${missing.join(", ")}`);
}

/** The path of a build next to the runtime that loads it. */
export function flavorPath(flavor) {
  return fileURLToPath(new URL(`kevala-${flavor}.wasm`, runtime));
}
