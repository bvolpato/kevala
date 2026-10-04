#!/usr/bin/env node
// Syntax-check every JavaScript source in the repository: modules, scripts, tests, and the
// inline module scripts of the HTML harness pages. There is no bundler or linter to catch a typo.
import { globSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";

const MODULES = ["js/src/**/*.js", "app/**/*.js", "dev/*.js", "benchmarks/**/*.{js,mjs}", "scripts/**/*.mjs", "tests/**/*.mjs"];
const PAGES = ["*.html", "dev/*.html", "examples/*.html", "site/*.html"];
const INLINE_MODULE = /<script\b[^>]*\btype="module"[^>]*>([\s\S]*?)<\/script>/g;

function check(file, label = file) {
  const result = spawnSync(process.execPath, ["--check", file], { encoding: "utf8" });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    process.stderr.write(result.stderr.replaceAll(file, label));
    process.exit(result.status ?? 1);
  }
}

const modules = globSync(MODULES);
for (const file of modules) check(file);

// An inline script has no file of its own: check a copy, and report errors against the page.
const scratch = realpathSync(mkdtempSync(join(tmpdir(), "kevala-check-js-")));
let inline = 0;
try {
  for (const page of globSync(PAGES)) {
    for (const [tag, source] of readFileSync(page, "utf8").matchAll(INLINE_MODULE)) {
      if (/\bsrc=/.test(tag.slice(0, tag.indexOf(">")))) continue;
      const copy = join(scratch, `inline-${inline++}.mjs`);
      writeFileSync(copy, source);
      check(copy, `${page} (inline module script)`);
    }
  }
} finally {
  rmSync(scratch, { recursive: true, force: true });
}
console.log(`Checked ${modules.length} JavaScript files and ${inline} inline module scripts.`);
