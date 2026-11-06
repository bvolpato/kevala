import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { once } from "node:events";
import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

test("profile writers share the OS lock across path aliases and release it when the holder exits", { skip: process.platform === "win32" }, async () => {
  const directory = await mkdtemp(join(tmpdir(), "kevala-profile-lock-"));
  let holder;
  try {
    const real = join(directory, "real");
    const alias = join(directory, "alias");
    await mkdir(real);
    await symlink(real, alias, "dir");
    const lock = join(real, "gpu.lock");
    const fixture = join(directory, "writer.mjs");
    const helper = new URL("../scripts/lib/profile-lock.mjs", import.meta.url).href;
    await writeFile(fixture, `import { withProfileLock } from ${JSON.stringify(helper)};\nawait withProfileLock(async () => { console.log("entered"); }, {path:process.argv[2]});\n`);
    holder = spawn("flock", ["--exclusive", lock, process.execPath, "-e", "console.log('ready'); process.stdin.resume();"], { stdio: ["pipe", "pipe", "pipe"] });
    await once(holder.stdout, "data");
    const blocked = spawnSync(process.execPath, [fixture, join(alias, "gpu.lock")], { encoding: "utf8", timeout: 5000 });
    assert.equal(blocked.status, 1);
    assert.match(blocked.stderr, /Another GPU profile holds/);
    assert.doesNotMatch(blocked.stdout, /entered/);
    const exited = once(holder, "exit");
    holder.stdin.end();
    await exited;
    holder = null;
    const accepted = spawnSync(process.execPath, [fixture, join(alias, "gpu.lock")], { encoding: "utf8", timeout: 5000 });
    assert.equal(accepted.status, 0, accepted.stderr);
    assert.equal(accepted.stdout.trim(), "entered");
  } finally {
    if (holder) { holder.stdin.end(); if (holder.exitCode === null) await once(holder, "exit"); }
    await rm(directory, { recursive: true });
  }
});
