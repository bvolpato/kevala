import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { KERNEL_HARNESSES, harnessUrl, kernelCoverage, kernelHarnessSkipReason, parseKernelRegistry, validateKernelResult } from "../scripts/lib/kernel-profile-catalog.mjs";
import { harnessIdentity, profileOptions } from "../scripts/profile-kernels.mjs";
import { isolatedUnavailable, recurrenceFixtureInput, utilityError, utilityFixture, utilityReference } from "../dev/kernel-profile-fixtures.js";
import { pleReference, reduceReference, sourceFixture, sourcePartials } from "../dev/kernel-profile-source-fixtures.js";

const harness = (id) => KERNEL_HARNESSES.find((item) => item.id === id);
const compile = (names, skipped = []) => ({ status: "done", done: true, backend: "webgpu", method: "GPU pipeline compilation wall clock", metricMs: 5,
  correctness: { ok: true, registeredKernels: names.length, compiled: names.filter((name) => !skipped.includes(name)).map((kernel) => ({ kernel, spec: {} })), skipped: skipped.map((kernel) => ({ kernel, unavailable: ["shader-f16"] })), failures: [] } });
const done = (extra = {}) => ({ status: "done", done: true, backend: "webgpu", correctness: { ok: true }, ...extra });
function isolated(names, { cpu = false, paired = false } = {}) {
  return done({ method: "WebGPU timestamp-query", fixtureSeed: 20261106, samples: 7, warmups: 3, tokens: [1, 47, 128, 512], exactComparisonRequired: paired,
    cases: names.flatMap((kernel) => [1, 47, 128, 512].map((tokens) => ({ kernel, tokens, finiteValues: tokens * 32, grid: [tokens], spec: {}, repetitions: 128,
      cpuMaxAbs: cpu ? 0 : null, bitDifferences: 0, maxAbs: 0,
      timings: (paired ? ["baseline", "current"] : ["current"]).map((label) => ({ label, samples: Array(7).fill(0.001), medianMs: 0.001, shaderSha256: "a".repeat(64) })) }))) });
}

test("registry rows remain accounted for when a kernel has no dispatched fixture", () => {
  const registered = parseKernelRegistry(readFileSync(new URL("../crates/kevala/src/gpu.rs", import.meta.url), "utf8"));
  const report = kernelCoverage([...registered, "future_kernel"], []);
  assert.deepEqual(report.kernels.map((row) => row.name), [...registered, "future_kernel"]);
  assert.equal(report.kernels.at(-1).status, "unmeasured");
  assert.equal(report.coverageComplete, false);
  assert.throws(() => parseKernelRegistry('pub const KERNELS: &[&str] = &["a", "a"];'), /duplicate/);
  assert.throws(() => parseKernelRegistry("no registry"), /empty/);
});

test("pipeline compilation and unsupported variants do not imply dispatch or timing coverage", () => {
  const names = ["matmul", "matmul_tall"];
  const report = kernelCoverage(names, [{ id: "compile", status: "passed", result: compile(names, ["matmul_tall"]) }]);
  assert.equal(report.kernels[0].status, "compile-only");
  assert.equal(report.kernels[0].numericalStatus, "not-run");
  assert.equal(report.kernels[0].timingStatus, "not-run");
  assert.equal(report.kernels[1].status, "unsupported");
  assert.deepEqual(report.kernels[1].compilation.missingCapabilities, ["shader-f16"]);
  const incomplete = compile(names);
  incomplete.correctness.compiled.pop();
  assert.throws(() => validateKernelResult(harness("compile"), incomplete, names), /match the registry/);
});

test("isolated finite smoke stays distinct from independent or exact numerical validation", () => {
  const names = ["matmul"];
  const runs = [{ id: "compile", status: "passed", result: compile(names) }];
  const finite = kernelCoverage(names, [...runs, { id: "isolated", status: "passed", result: isolated(names) }]);
  assert.equal(finite.kernels[0].timingStatus, "isolated");
  assert.equal(finite.kernels[0].numericalStatus, "finite-only");
  assert.equal(finite.kernels[0].status, "timed");
  const exact = kernelCoverage(names, [...runs, { id: "isolated", status: "passed", result: isolated(names, { paired: true }) }]);
  assert.equal(exact.kernels[0].status, "measured");
  assert.equal(exact.coverageComplete, true);
  const cpu = kernelCoverage(names, [...runs, { id: "isolated", status: "passed", result: isolated(names, { cpu: true }) }]);
  assert.equal(cpu.kernels[0].status, "measured");
});

test("isolated output rejects missing shapes, changed protocol, invalid timing and nonzero required parity", () => {
  const names = ["matmul"], check = (value) => validateKernelResult(harness("isolated"), value, names);
  const missing = isolated(names); missing.cases.pop();
  assert.throws(() => check(missing), /omitted or duplicated/);
  const protocol = isolated(names); protocol.warmups = 0;
  assert.throws(() => check(protocol), /sampling protocol/);
  const time = isolated(names); time.cases[0].timings[0].samples[1] = NaN;
  assert.throws(() => check(time), /invalid samples/);
  const paired = isolated(names, { paired: true }); paired.cases[0].bitDifferences = 1;
  assert.throws(() => check(paired), /bit-exact/);
  const software = isolated(names); software.adapter = { isFallbackAdapter: true };
  assert.throws(() => check(software), /software adapter/);
});

test("guard wall and constant metrics are not credited as GPU timestamps", () => {
  const names = ["matvec"], base = [{ id: "compile", status: "passed", result: compile(names) }];
  const wall = done({ metricMs: 42, cases: [{ timing: { method: "submit and readback wall time", directMs: 1, matmulMs: 2 } }] });
  const report = kernelCoverage(names, [...base, { id: "matvec", status: "passed", result: wall }]);
  assert.equal(report.kernels[0].numericalStatus, "passed");
  assert.equal(report.kernels[0].timingStatus, "not-run");
  const gpu = done({ metricMs: 42, cases: [{ timing: { method: "GPU timestamp", directMs: 0.1, matmulMs: 0.2 } }] });
  const timed = kernelCoverage(names, [...base, { id: "matvec", status: "passed", result: gpu }]);
  assert.equal(timed.kernels[0].timingStatus, "composite");
  assert.match(timed.kernels[0].reason, /not this kernel alone/);
});

test("failed and capability-skipped guards never become successful numerical evidence", () => {
  const names = ["matvec_two"];
  const base = [{ id: "compile", status: "passed", result: compile(names) }];
  const skipped = kernelCoverage(names, [...base, { id: "two-row-subgroups", status: "skipped", reason: "Missing capabilities: subgroup32" }]);
  assert.equal(skipped.kernels[0].status, "compile-only");
  assert.equal(skipped.kernels[0].skipped[0].reason, "Missing capabilities: subgroup32");
  const failed = kernelCoverage(names, [...base, { id: "two-row-subgroups", status: "failed", reason: "tail overwrite" }]);
  assert.equal(failed.kernels[0].status, "failed");
  assert.equal(failed.counts.failed, 1);
});

test("wide paired projections skip before dispatch when the adapter lacks shader-f16", () => {
  const selected = harness("paired-projections-wide"), names = selected.kernels;
  const reason = kernelHarnessSkipReason(selected, { shaderF16: false, timestampQuery: true, subgroup32: true });
  assert.equal(reason, "Missing capabilities: shaderF16");
  assert.equal(kernelHarnessSkipReason(selected, { shaderF16: true }), null);
  const report = kernelCoverage(names, [{ id: "compile", status: "passed", result: compile(names) }, { id: selected.id, status: "skipped", reason }]);
  assert.equal(report.counts.failed, 0);
  assert.ok(report.kernels.every((row) => row.status === "compile-only" && row.numericalStatus === "not-run" && row.timingStatus === "not-run"));
  assert.ok(report.kernels.every((row) => row.skipped[0].reason === reason));
});

test("honest all-skipped focused guards preserve their reason without numerical or timing credit", () => {
  const names = ["matmul_gelu_tall"], page = done({ skipped: true, correctness: { ok: true, bitExact: false,
    cases: [{ skipped: true, reason: "controlled-temporary-memory-limit", timing: { candidateMs: 0.1 } }] } });
  const outcome = validateKernelResult(harness("tall-paired"), page, names);
  assert.deepEqual(outcome, { status: "skipped", reason: "controlled-temporary-memory-limit" });
  const report = kernelCoverage(names, [{ id: "compile", status: "passed", result: compile(names) }, { id: "tall-paired", status: "passed", result: page }]);
  assert.equal(report.kernels[0].status, "compile-only");
  assert.equal(report.kernels[0].numericalStatus, "not-run");
  assert.equal(report.kernels[0].timingStatus, "not-run");
  assert.equal(report.kernels[0].skipped[0].reason, "controlled-temporary-memory-limit");
  delete page.skipped;
  assert.deepEqual(validateKernelResult(harness("tall-paired"), page, names), outcome);
  delete page.correctness.cases[0].reason;
  assert.throws(() => validateKernelResult(harness("tall-paired"), page, names), /skip without an explicit reason/);
  assert.throws(() => validateKernelResult(harness("tall-paired"), done({ correctness: { ok: true, cases: [{ ok: false }] } }), names), /failed executed case/);
});

test("mixed focused cases credit only the registered path that executed", () => {
  const names = ["matvec_two", "matvec_gelu_two"], page = done({ correctness: { ok: true, cases: [
    { kind: "plain", ok: true, timing: { method: "GPU timestamp", baselineMs: 0.1, candidateMs: 0.05 } },
    { kind: "gelu", skipped: true, reason: "controlled-temporary-memory-limit", timing: { candidateMs: 99 } },
  ] } });
  const report = kernelCoverage(names, [{ id: "compile", status: "passed", result: compile(names) }, { id: "two-row-subgroups", status: "passed", result: page }]);
  assert.equal(report.kernels[0].status, "measured");
  assert.equal(report.kernels[1].numericalStatus, "not-run");
  assert.equal(report.kernels[1].timingStatus, "not-run");
  assert.equal(report.kernels[1].skipped[0].reason, "controlled-temporary-memory-limit");
  assert.ok(report.kernels[0].evidence.at(-1).measurements.every((item) => !Object.values(item.values).includes(99)));
});

test("isolated capability skips retain the full selected inventory without manufacturing dispatch evidence", () => {
  const names = ["matmul", "attention_subgroup"], result = isolated(["matmul"], { cpu: true });
  result.skippedKernels = ["attention_subgroup"]; result.skipped = [{ kernel: "attention_subgroup", unavailable: ["exact 32-lane subgroups"] }];
  validateKernelResult(harness("isolated"), result, names);
  const report = kernelCoverage(names, [{ id: "compile", status: "passed", result: compile(names) }, { id: "isolated", status: "passed", result }]);
  assert.equal(report.kernels[0].status, "measured");
  assert.equal(report.kernels[1].timingStatus, "not-run");
  assert.equal(report.kernels[1].numericalStatus, "not-run");
  result.skipped[0].unavailable = [];
  assert.throws(() => validateKernelResult(harness("isolated"), result, names), /capability skips/);
  const capabilities = { features: new Set(["timestamp-query"]), languageFeatures: new Set(), subgroups: false, subgroup32: false, subgroupMinSize: 0, workgroupStorageSize: 32768 };
  assert.deepEqual(isolatedUnavailable("attention_subgroup", "enable subgroups;", capabilities), ["exact 32-lane subgroups", "subgroups"]);
  assert.deepEqual(isolatedUnavailable("attention_tile", "enable f16;", capabilities), ["f16"]);
  assert.deepEqual(isolatedUnavailable("attention_tile_f32", "", capabilities), []);
  assert.deepEqual(isolatedUnavailable("attention_tile_f32", "", { ...capabilities, workgroupStorageSize: 16384 }), ["30224 bytes of workgroup storage"]);
});

test("profile arguments validate bounded runs and same-origin fixture URLs", () => {
  assert.equal(profileOptions(["--base", "http://localhost:18171/demo", "--only", "compile,isolated"]).base, "http://localhost:18171/demo/");
  assert.throws(() => profileOptions(["--browser", "webkit"]), /browser/);
  assert.throws(() => profileOptions(["--only", "compile,compile"]), /duplicate/);
  assert.throws(() => profileOptions(["--timeout", "0"]), /timeout/);
  assert.throws(() => profileOptions(["--budget", "Infinity"]), /budget/);
  const url = new URL(harnessUrl(harness("isolated"), "http://localhost:18171/demo/", ["matmul"], "baseline.wasm"));
  assert.equal(url.pathname, "/demo/dev/kernel-bench.html");
  assert.equal(url.searchParams.get("exact"), "1");
  assert.throws(() => harnessUrl(harness("isolated"), "http://localhost/", ["matmul"], "https://other.test/control.wasm"), /same-origin/);
});

test("harness identity includes nested imported helpers independently of the root trailing slash", async () => {
  const folder = await mkdtemp(join(tmpdir(), "kevala-kernel-identity-")), root = join(folder, "checkout");
  const write = async (path, content) => { const absolute = join(root, path); await mkdir(join(absolute, ".."), { recursive: true }); await writeFile(absolute, content); };
  try {
    for (const path of ["scripts/profile-kernels.mjs", "scripts/lib/kernel-profile-catalog.mjs", "dev/wgsl.js", "dev/kernel-profile-fixtures.js"]) await write(path, "export {};\n");
    await write("dev/profile.html", '<script type="module">import "./profile.js";</script>');
    await write("dev/profile.js", 'import { fixture } from "./nested/fixture.js?v=1"; import "../shared.js"; import "../../outside.js";');
    await write("dev/nested/fixture.js", 'export { fixture } from "../nested/deeper/helper.mjs";');
    await write("dev/nested/deeper/helper.mjs", "export const fixture = 1;\n");
    await write("shared.js", "export {};\n");
    await writeFile(join(folder, "outside.js"), "throw new Error('outside');\n");
    const selected = [{ path: "/dev/profile.html" }], identity = await harnessIdentity(selected, root);
    assert.deepEqual(await harnessIdentity(selected, `${root}/`), identity);
    assert.ok(identity.files.some((file) => file.path === "dev/nested/fixture.js"));
    assert.ok(identity.files.some((file) => file.path === "dev/nested/deeper/helper.mjs"));
    assert.ok(identity.files.every((file) => !file.path.startsWith("/") && !file.path.includes("outside.js")));
    await write("dev/nested/deeper/helper.mjs", "export const fixture = 2;\n");
    assert.notEqual((await harnessIdentity(selected, root)).fingerprint, identity.fingerprint);
    await rm(join(root, "dev/nested/deeper/helper.mjs"));
    await assert.rejects(harnessIdentity(selected, root), /harness dependency is missing.*helper\.mjs/);
  } finally { await rm(folder, { recursive: true, force: true }); }
});

test("utility fixtures gather signed Q8 rows with duplicates and keep precision interpretation", () => {
  const input = new Float32Array(1024).fill(0.25);
  input.set([1, -2, 0.5, 1.25]);
  const reference = utilityReference("weight_rows", input, 6, 32);
  const raw = new Int8Array(input.buffer);
  for (let row = 0; row < 6; row++) for (let column = 0; column < 32; column++) assert.equal(reference[row * 32 + column], Math.fround(raw[(row % 4) * 32 + column] * input[row % 4]));
  assert.deepEqual(reference.slice(0, 32), reference.slice(128, 160));
  assert.deepEqual([...utilityFixture("weight_rows", 6, 32).overrides.ids], [0, 1, 2, 3, 0, 1]);
  assert.equal(utilityError("weight_rows", input, reference, 6, 32), 0);
  assert.throws(() => utilityFixture("weight_rows", 1, 12), /divisible by 32/);
});

test("utility clamp and pooling references cover full outputs and reject corruption", () => {
  const input = new Float32Array([-2, -0.75, -0, 0.25, 0.75, 1, 2, -1]);
  const clipped = utilityReference("gemma4_vision_clip", input, 1, 8);
  assert.deepEqual([...clipped], [-0.75, -0.75, -0, 0.25, 0.75, 0.75, 0.75, -0.75]);
  const pooledInput = new Float32Array(2 * 4 * 16).fill(1);
  const pooled = utilityReference("gemma4_vision_pool", pooledInput, 2, 16);
  assert.deepEqual([...pooled], Array(32).fill(3));
  assert.equal(utilityError("gemma4_vision_pool", pooledInput, pooled, 2, 16), 0);
  const changed = pooled.slice(); changed[31] += 0.01;
  assert.throws(() => utilityError("gemma4_vision_pool", pooledInput, changed, 2, 16), /CPU mismatch/);
  assert.throws(() => utilityError("gemma4_vision_pool", pooledInput, pooled.slice(1), 2, 16), /output length/);
  const nonfinite = pooled.slice(); nonfinite[0] = Infinity;
  assert.throws(() => utilityError("gemma4_vision_pool", pooledInput, nonfinite, 2, 16), /nonfinite/);
  assert.throws(() => utilityReference("gemma4_vision_pool", input, 2, 16), /too short/);
});

test("finite scanner fixture compares the exact atomic status bits for finite and nonfinite inputs", () => {
  const input = new Float32Array(16).fill(0.5), clear = new Float32Array(1), flagged = new Float32Array(Uint32Array.of(1).buffer);
  assert.deepEqual([...utilityReference("gemma4_vision_finite", input, 1, 16)], [0]);
  assert.equal(utilityError("gemma4_vision_finite", input, clear, 1, 16), 0);
  assert.throws(() => utilityError("gemma4_vision_finite", input, flagged, 1, 16), /atomic status 1 differs from CPU status 0/);
  for (const value of [NaN, Infinity, -Infinity]) {
    input[15] = value;
    assert.deepEqual([...utilityReference("gemma4_vision_finite", input, 1, 16)], [1]);
    assert.equal(utilityError("gemma4_vision_finite", input, flagged, 1, 16), 0);
    assert.throws(() => utilityError("gemma4_vision_finite", input, clear, 1, 16), /atomic status 0 differs from CPU status 1/);
  }
  const fixture = utilityFixture("gemma4_vision_finite", 2, 16);
  assert.equal(fixture.preserveOutputInitialization, true);
  assert.equal(fixture.overrides.status[0], 0);
  assert.deepEqual(fixture.output, ["status", 1]);
});

test("recurrence fixtures preserve V and normalize all default Kev Q/K heads through long sequences", () => {
  const tokens = 512, width = 128, qk = 16 * width, stride = 3 * qk;
  let seed = 20261106;
  const input = Float32Array.from({ length: tokens * stride }, () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return (seed / 0x100000000 - 0.5) * 4; });
  const output = recurrenceFixtureInput(input, tokens);
  assert.equal(output.length, input.length);
  for (let token = 0; token < tokens; token++) {
    for (let head = 0; head < 32; head++) {
      const base = token * stride + head * width;
      let norm = 0;
      for (let lane = 0; lane < width; lane++) norm += output[base + lane] ** 2;
      assert.ok(Math.abs(norm - (head < 16 ? 1 / 128 : 1)) < 3e-7);
    }
    assert.deepEqual(output.subarray(token * stride + 2 * qk, (token + 1) * stride), input.subarray(token * stride + 2 * qk, (token + 1) * stride));
  }
  const f32 = Math.fround;
  const recurColumn = (values) => {
    const state = new Float32Array(width);
    for (let token = 0; token < tokens; token++) {
      const base = token * stride, products = new Float32Array(4);
      for (let group = 0; group < 4; group++) for (let lane = 0; lane < 32; lane++) products[group] = f32(products[group] + f32(state[group * 32 + lane] * values[base + qk + group * 32 + lane]));
      const dot = f32(f32(products[0] + products[1]) + f32(products[2] + products[3]));
      const delta = f32(f32(values[base + 2 * qk] - f32(f32(0.99) * dot)) * 0.5);
      for (let lane = 0; lane < width; lane++) state[lane] = f32(f32(f32(0.99) * state[lane]) + f32(values[base + qk + lane] * delta));
      if (state.some((value) => !Number.isFinite(value))) return false;
    }
    return true;
  };
  assert.equal(recurColumn(input), false, "the old unnormalized fixture overflows this recurrence");
  assert.equal(recurColumn(output), true);
  assert.throws(() => recurrenceFixtureInput(input.subarray(1), tokens), /complete default Kev/);
  const invalid = new Float32Array(stride); invalid[0] = Infinity;
  assert.throws(() => recurrenceFixtureInput(invalid, 1), /finite Q\/K\/V/);
});

test("source matrix CPU references preserve signed Q8, partitioning, bias and residual modes", () => {
  const T = 2, N = 2, K = 256;
  const fixture = { T, N, K, weights: Int8Array.from({ length: N * K }, (_, index) => index < K ? 2 : -2), scales: new Float32Array(N * K / 32).fill(1 / 32), input: new Float32Array(T * K).fill(1 / 64) };
  const partials = sourcePartials(fixture, 2);
  assert.deepEqual([...partials], [0.125, -0.125, 0.125, -0.125, 0.125, -0.125, 0.125, -0.125]);
  assert.deepEqual([...reduceReference(partials, { T, N, splits: 2, mode: 0 })], [0.25, -0.25, 0.25, -0.25]);
  const bias = Float32Array.of(0.125, -0.125), residual = new Float32Array(T * N).fill(0.5);
  assert.deepEqual([...reduceReference(partials, { T, N, splits: 2, mode: 1, bias, residual })], [0.875, 0.125, 0.875, 0.125]);
  assert.deepEqual([...reduceReference(partials, { T, N, splits: 2, mode: 2, bias })], [0.375, 0, 0.375, 0]);
  assert.throws(() => sourcePartials(fixture, 0), /invalid source/);
  assert.throws(() => reduceReference(partials, { T, N, splits: 2, mode: 1 }), /invalid reducer/);
  assert.throws(() => sourceFixture({ T, N, K: 31 }), /dimensions/);
  assert.deepEqual(sourceFixture({ T, N, K: 32 }, 17), sourceFixture({ T, N, K: 32 }, 17));
});

test("generic PLE reference handles the exact positive and negative saturation branches", () => {
  assert.deepEqual([...pleReference(Float32Array.of(-20, 0, 20), Float32Array.of(0.5, 2, 0.25))], [0, 0, 5]);
  assert.throws(() => pleReference(Float32Array.of(1), Float32Array.of(1, 2)), /lengths differ/);
});
