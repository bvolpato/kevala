import { globSync } from "node:fs";
import { spawnSync } from "node:child_process";

const files = globSync(["js/src/**/*.js", "app/**/*.js", "dev/*.js", "scripts/*.mjs", "tests/*.mjs"]);
for (const file of files) {
  const result = spawnSync(process.execPath, ["--check", file], { stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}
console.log(`Checked ${files.length} JavaScript files.`);
