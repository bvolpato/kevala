#!/usr/bin/env node
// Validate the packed package without installing or publishing it.
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const run = promisify(execFile);
const tarball = process.argv[2];
if (!tarball) throw new Error("usage: node scripts/check-package.mjs <package.tgz>");

const { stdout } = await run("tar", ["-tzf", tarball], { maxBuffer: 16 * 1024 * 1024 });
const files = new Set(stdout.trim().split("\n").filter(Boolean));
const required = [
  "package/package.json",
  "package/README.md",
  "package/LICENSE",
  "package/js/src/index.js",
  "package/js/src/node.js",
  "package/js/src/kevala-relaxed.wasm",
  "package/js/src/kevala-simd.wasm",
  "package/js/src/kevala-base.wasm",
];
for (const file of required) {
  if (!files.has(file)) throw new Error(`packed package is missing ${file}`);
}
for (const file of files) {
  if (/^package\/(?:target|tmp|node_modules|crates)\//.test(file)) {
    throw new Error(`packed package contains a private build path: ${file}`);
  }
}

const packageJson = JSON.parse((await run("tar", ["-xOf", tarball, "package/package.json"])).stdout);
if (packageJson.exports?.["."] !== "./js/src/index.js") throw new Error("root package export changed");
if (packageJson.exports?.["./node"] !== "./js/src/node.js") throw new Error("Node package export changed");

for (const flavor of ["relaxed", "simd", "base"]) {
  const { stdout: bytes } = await run("tar", ["-xOf", tarball, `package/js/src/kevala-${flavor}.wasm`], {
    encoding: "buffer",
    maxBuffer: 16 * 1024 * 1024,
  });
  if (!WebAssembly.validate(bytes)) throw new Error(`packed ${flavor} module is invalid WebAssembly`);
}
console.log(`package ${packageJson.name}@${packageJson.version}: ${files.size} files, 3 WebAssembly modules`);
