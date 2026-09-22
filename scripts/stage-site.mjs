#!/usr/bin/env node
// Stage the public static site into the fixed dist/site destination.
// The allowlist keeps model packs, build output, and Rust sources out of Pages.
import { cp, lstat, mkdir, mkdtemp, readFile, readdir, rename, rm } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(root, "dist");
const destination = join(dist, "site");
const publicFiles = [".nojekyll", "LICENSE", "THIRD_PARTY_NOTICES", "README.md", "package.json", "bench.html", "index.html", "parity.html", "parity-kev.html"];
const publicDirectories = ["app", "assets", "docs", "examples", "js/src", "skills", "tests/fixtures"];
const publicHarnesses = [
  "dev/cache-test.html",
  "dev/convert-test.html",
  "dev/gpu-bench.html",
  "dev/gpu-bench.js",
  "dev/kernels.html",
  "dev/matmul-bench.html",
  "dev/recur-bench.html",
  "dev/tetris-eval.html",
  "dev/wgsl.js",
];
const flavors = ["relaxed", "simd", "base"];
const forbidden = /^(?:node_modules|target|tmp|crates|packs|.*\.kevala)$/;

async function existingStat(path) {
  try {
    return await lstat(path);
  } catch (error) {
    if (error.code === "ENOENT") return null;
    throw error;
  }
}

async function assertDirectory(path, label) {
  const stat = await existingStat(path);
  if (stat?.isSymbolicLink()) throw new Error(`${label} must not be a symlink: ${path}`);
  if (stat && !stat.isDirectory()) throw new Error(`${label} must be a directory: ${path}`);
}

async function copy(source, target) {
  const stat = await lstat(source);
  if (stat.isSymbolicLink()) throw new Error(`public source must not be a symlink: ${source}`);
  await mkdir(dirname(target), { recursive: true });
  await cp(source, target, { recursive: stat.isDirectory(), force: true, errorOnExist: false });
}

async function assertNoForbidden(path) {
  for (const entry of await readdir(path, { withFileTypes: true })) {
    if (forbidden.test(entry.name)) throw new Error(`site contains a private build path: ${join(path, entry.name)}`);
    if (entry.isDirectory()) await assertNoForbidden(join(path, entry.name));
  }
}

await assertDirectory(dist, "dist");
const destinationStat = await existingStat(destination);
if (destinationStat?.isSymbolicLink()) throw new Error(`site destination must not be a symlink: ${destination}`);
if (destinationStat && !destinationStat.isDirectory()) throw new Error(`site destination must be a directory: ${destination}`);
await mkdir(dist, { recursive: true });

const temporary = await mkdtemp(join(dist, ".site-"));
try {
  for (const file of [...publicFiles, ...publicHarnesses]) await copy(join(root, file), join(temporary, file));
  for (const directory of publicDirectories) await copy(join(root, directory), join(temporary, directory));
  await assertNoForbidden(temporary);

  for (const flavor of flavors) {
    const bytes = await readFile(join(temporary, "js/src", `kevala-${flavor}.wasm`));
    if (!WebAssembly.validate(bytes)) throw new Error(`staged ${flavor} module is invalid WebAssembly`);
  }

  const finalStat = await existingStat(destination);
  if (finalStat?.isSymbolicLink()) throw new Error(`site destination became a symlink: ${destination}`);
  if (finalStat && !finalStat.isDirectory()) throw new Error(`site destination must be a directory: ${destination}`);
  if (finalStat) await rm(destination, { recursive: true, force: true });
  await rename(temporary, destination);
} catch (error) {
  await rm(temporary, { recursive: true, force: true });
  throw error;
}

console.log(`staged site: ${destination}`);
