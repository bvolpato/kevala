import { UTILITY_KERNELS } from "../../dev/kernel-profile-fixtures.js";

export const ISOLATED_KERNELS = [
  "matmul", "matmul_wide", "reduce", "norm", "rope", "attention", "attention_subgroup", "attention_tile", "attention_tile_f32", "geglu", "gather",
  "kev_rms", "kev_gather", "kev_gates", "kev_conv", "kev_save_tail", "kev_qknorm", "kev_recur_lanes", "kev_recur_lanes8", "kev_recur_lanes16", "kev_gnorm", "kev_aprep", "kev_save_kv", "kev_attention_keys", "kev_attention", "kev_attention_tile", "kev_silumul",
  "gemma4_embed", "gemma4_rms", "gemma4_qkv", "gemma4_attention", "gemma4_gelu", "gemma4_ple", "gemma4_residual", "gemma4_gather", ...UTILITY_KERNELS,
];

const guard = (id, path, kernels, required = []) => ({ id, path, kernels, required, kind: "gpu", numerical: "guard", timing: "composite" });

// These mappings describe dispatched checks in the named pages, not just source
// references. A successful compile never supplies numerical or timing evidence.
export const KERNEL_HARNESSES = [
  { id: "compile", path: "/dev/kernel-compile-check.html", kernels: "registry", kind: "gpu", numerical: "none", timing: "none", required: [] },
  { id: "isolated", path: "/dev/kernel-bench.html?tokens=1,47,128,512&seed=20261106&samples=7&warmups=3", kernels: ISOLATED_KERNELS, kind: "gpu", numerical: "isolated", timing: "isolated", required: ["timestampQuery"] },
  { id: "core-reference", path: "/dev/kernels.html", kernels: ["matmul", "reduce", "norm", "rope", "attention", "geglu"], kind: "kernels", numerical: "cpu", timing: "none", required: [] },
  guard("projections", "/dev/gpu-bench.html?cases=7x132x96,45x132x1024,140x3072x1024&kernels=matmul@256:4,matmul_wide@256:4,matmul_k64@256:4&seed=20261106&samples=7&warmups=3", ["matmul", "matmul_wide", "matmul_k64", "reduce"]),
  guard("clef-head", "/dev/clef-head-check.html", ["clef_head_projection", "clef_head_tail"]),
  guard("matvec", "/dev/gemma4-matvec-check.html", ["matvec"]),
  guard("matvec-gelu", "/dev/gemma4-matvec-gelu-check.html", ["matvec", "matvec_gelu", "gemma4_gelu"]),
  guard("two-row-subgroups", "/dev/gemma4-matvec-two-subgroup-check.html?strategy=vector&registered=1", ["matvec_two", "matvec_gelu_two"], ["subgroup32", "timestampQuery"]),
  guard("paired-projections", "/dev/gemma4-matmul-gelu-check.html?kernel=generic&precision=f32", ["matmul", "matmul_gemma_batch", "reduce", "reduce_gemma_batch", "matmul_gelu", "reduce_gelu", "gemma4_gelu"]),
  guard("paired-projections-wide", "/dev/gemma4-matmul-gelu-check.html?kernel=wide", ["matmul_wide", "matmul_wide_gemma_batch", "reduce", "reduce_gemma_batch", "matmul_gelu", "reduce_gelu", "gemma4_gelu"], ["shaderF16"]),
  guard("ple-reducers", "/dev/gemma4-ple-gelu-check.html?registered=1", ["reduce_gemma_batch", "reduce_gemma_source", "reduce_ple_gemma_batch", "reduce_ple_gemma_source", "gemma4_gelu"]),
  guard("tall-paired", "/dev/gemma4-suffix-tall-check.html?tile=104&registered=1&f16=1", ["matmul_gelu", "matmul_gelu_tall", "reduce_gelu"], ["shaderF16", "timestampQuery"]),
  guard("tall-single", "/dev/gemma4-matmul-f16-tall-check.html", ["matmul_gemma_batch", "matmul_tall", "reduce_gemma_batch"], ["shaderF16", "timestampQuery"]),
  guard("source-bn32", "/dev/gemma4-matmul-bn32-check.html?scope=fast&registered=1", ["matmul_gemma_source", "matmul_bn32_gemma_source", "reduce_gemma_source"], ["timestampQuery"]),
  guard("source-paired-bn32", "/dev/gemma4-prefix-gelu-bn32-check.html?registered=1&bounds=all", ["matmul_bn32_gemma_source", "matmul_gelu_bn32", "reduce_gemma_source", "reduce_gelu", "gemma4_gelu"], ["timestampQuery"]),
  guard("source-variants", "/dev/kernel-profile-source-check.html", ["matmul_gemma_source", "matmul_wide_gemma_source", "matmul_k64_gemma_source", "reduce_gemma_source", "reduce", "reduce_ple", "gemma4_gelu"], ["timestampQuery", "sourceProjection"]),
  guard("rms-fusions", "/dev/gemma4-rms-fusion-check.html", ["gemma4_rms", "gemma4_residual", "gemma4_rms_residual_norm", "gemma4_rms_scale"]),
  guard("rms-next", "/dev/gemma4-rms-scale-next-check.html?registered=1", ["gemma4_rms", "gemma4_rms_scale", "gemma4_rms_residual_norm", "gemma4_rms_scale_next"], ["timestampQuery"]),
  guard("laya-attention", "/dev/attn-bench.html?cases=1:0,47:0,128:64,512:0,150/40:64&kernels=attention,attention_subgroup,attention_tile,attention_tile_f32&samples=7&warmups=3&checks=3", ["attention", "attention_subgroup", "attention_tile", "attention_tile_f32"]),
  guard("qwen-attention", "/dev/kev-attn-bench.html?gateTails=1&samples=7&warmups=3", ["kev_attention", "kev_attention_tile"], ["subgroup32", "timestampQuery"]),
  ...[[4, "current"], [8, "recur8"], [16, "recur16"]].map(([lanes, candidate]) => guard(`recurrence-${lanes}`, `/dev/recur-check.html?candidate=${candidate}&subgroups=0&actual=1&samples=7&warmups=3`, [`kev_recur_lanes${lanes === 4 ? "" : lanes}`], ["timestampQuery"])),
  guard("gemma-attention", "/dev/gemma-attn-bench.html?samples=7&warmups=3", ["gemma4_attention"]),
  guard("gemma-attention-subgroups", "/dev/gemma4-attention-subgroup-check.html", ["gemma4_attention", "gemma4_attention_subgroup"], ["subgroup32", "timestampQuery"]),
  guard("vision-norm-rope", "/dev/gemma4-vision-norm-rope-check.html", ["gemma4_vision_norm", "gemma4_vision_rope", "gemma4_vision_norm_rope"]),
  guard("vision-pool-norm", "/dev/gemma4-vision-pool-norm-check.html", ["gemma4_vision_pool", "gemma4_vision_norm", "gemma4_vision_pool_norm"]),
  guard("vision-reduce-gelu", "/dev/gemma4-vision-reduce-gelu-check.html", ["reduce", "gemma4_vision_gelu", "gemma4_vision_reduce_gelu"]),
  guard("vision-reduce-residual", "/dev/gemma4-vision-reduce-residual-check.html", ["reduce", "gemma4_vision_residual_norm", "gemma4_vision_reduce_residual"]),
  guard("vision-reduce-norm-rope", "/dev/gemma4-vision-reduce-norm-rope-check.html", ["reduce", "gemma4_vision_norm_rope", "gemma4_vision_reduce_norm_rope"]),
  guard("vision-attention", "/dev/gemma4-vision-attention-check.html?fused=1", ["gemma4_vision_scores", "gemma4_vision_scores_tiled", "gemma4_vision_attention", "gemma4_vision_attention_fused"]),
  guard("vision-scores", "/dev/gemma4-vision-scores-check.html", ["gemma4_vision_scores", "gemma4_vision_scores_tiled"]),
  guard("vision-qkv", "/dev/gemma4-vision-qkv-bf16-check.html?registered=1&full=1", ["gemma4_vision_qkv", "gemma4_vision_norm_rope", "reduce"], ["timestampQuery", "visionQkv"]),
  guard("grid-mlp", "/dev/gemma4-mlp-grid-check.html?registered=1&guards=1", ["gemma4_mlp_grid", "matmul_gemma_batch", "reduce_gemma_batch", "reduce_gelu", "gemma4_gelu"], ["timestampQuery"]),
  guard("image-preparation", "/dev/gemma4-image-gpu-check.html?autorun=1&edges=1&registered=1", ["gemma4_image_resize_horizontal", "gemma4_image_resize_vertical"]),
  { ...guard("vision-profile", "/dev/gemma4-vision-check.html?profile=1&fused=0", ["gemma4_vision_position", "gemma4_vision_norm", "gemma4_vision_clip", "gemma4_vision_rope", "gemma4_vision_gelu", "gemma4_vision_pool"], ["timestampQuery"]), timing: "vision-profile" },
  guard("vision-finite-output", "/dev/gemma4-vision-check.html?gpu=1", ["gemma4_vision_finite"]),
];

const VISION_LABELS = {
  gemma4_vision_position: ["vision.position"], gemma4_vision_norm: ["vision.norm."], gemma4_vision_clip: ["vision.clip"],
  gemma4_vision_rope: ["vision.rope"], gemma4_vision_gelu: ["vision.gelu"], gemma4_vision_pool: ["vision.pool"],
};
const positive = (value) => typeof value === "number" && Number.isFinite(value) && value > 0;
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const isSkipped = (item) => item?.skipped === true || item?.status === "skipped";
const focusedCases = (result) => [result?.cases, result?.results, result?.correctness?.cases, result?.correctness?.cpu].filter(Array.isArray).flat();
const caseKernels = (harness, item) => Array.isArray(item?.kernels) ? item.kernels : harness.id === "two-row-subgroups" && ["plain", "gelu"].includes(item?.kind) ? [item.kind === "plain" ? "matvec_two" : "matvec_gelu_two"] : [];

function focusedSkipReason(harness, result) {
  if (["compile", "isolated"].includes(harness.id) || harness.kind === "kernels" || result?.backend !== "webgpu"
    || result.adapter?.isFallbackAdapter === true || result.adapter?.info?.isFallbackAdapter === true || !["done", "skipped"].includes(result.status)) return null;
  const cases = focusedCases(result);
  if (!isSkipped(result) && (!cases.length || !cases.every(isSkipped))) return null;
  const reasons = [...new Set([result.reason, result.skipReason, ...cases.filter(isSkipped).flatMap((item) => [item.reason, item.skipReason])].filter((reason) => typeof reason === "string" && reason.trim()))];
  if (!reasons.length) throw new Error("focused guard reported a skip without an explicit reason");
  return reasons.join("; ");
}

export function parseKernelRegistry(source) {
  const section = source.match(/pub const KERNELS:\s*&\[&str\]\s*=\s*&\[([\s\S]*?)\];/)?.[1];
  const kernels = [...(section || "").matchAll(/"([a-z][a-z0-9_]*)"/g)].map((match) => match[1]);
  if (!kernels.length || new Set(kernels).size !== kernels.length) throw new Error("kernel registry is empty or contains duplicate names");
  return kernels;
}

export function harnessKernels(harness, registry) {
  return harness.kernels === "registry" ? registry : harness.kernels.filter((name) => registry.includes(name));
}

export function kernelHarnessSkipReason(harness, capabilities) {
  if (harness.id === "compile") return null;
  if (!capabilities) return "Compilation probe failed; device capabilities are unknown.";
  const missing = harness.required.filter((name) => capabilities[name] !== true);
  return missing.length ? `Missing capabilities: ${missing.join(", ")}` : null;
}

export function harnessUrl(harness, base, registry, baselineWasm) {
  const directory = new URL(base);
  if (!directory.pathname.endsWith("/")) directory.pathname += "/";
  const url = new URL(harness.path.replace(/^\//, ""), directory);
  if (harness.id === "compile") url.searchParams.set("kernels", registry.join(","));
  if (harness.id === "isolated" && baselineWasm) {
    const baseline = new URL(baselineWasm, base);
    if (baseline.origin !== url.origin) throw new Error("baseline WASM must be same-origin");
    url.searchParams.set("baselineWasm", baseline.href);
    url.searchParams.set("exact", "1");
  }
  return url.href;
}

export function validateKernelResult(harness, result, registry) {
  const skipped = focusedSkipReason(harness, result);
  if (skipped) return { status: "skipped", reason: skipped };
  if (!result || result.error || result.status === "error") throw new Error(result?.error || "kernel page returned no successful result");
  if (harness.kind === "kernels") {
    const names = ["matmul_k96", "matmul_k1024", "norm", "attention_w0", "attention_w64", "geglu", "rope"];
    if (result.done !== true || names.some((key) => !finite(result[key]) || result[key] < 0 || result[key] >= 1e-4)) throw new Error("incomplete or failed independent core-reference checks");
    return;
  }
  if (result.backend !== "webgpu" || result.correctness?.ok !== true || result.status !== "done") throw new Error("kernel result lacks completed WebGPU correctness checks");
  if (result.adapter?.isFallbackAdapter === true || result.adapter?.info?.isFallbackAdapter === true) throw new Error("software adapter cannot establish GPU evidence");
  if (!["compile", "isolated"].includes(harness.id)) {
    const cases = focusedCases(result), executed = cases.filter((item) => item && typeof item === "object" && !isSkipped(item));
    if (executed.some((item) => item.ok === false || item.status === "error" || item.correctness?.ok === false)) throw new Error("focused guard contains a failed executed case");
    const counts = [result.correctness.cases, result.correctness.cpu, result.correctness.checkedPixels, result.correctness.checkedValues];
    if (cases.length ? !executed.length : !counts.some(positive)) throw new Error("focused guard returned no executed checked cases or values");
  }
  if (harness.id === "compile") {
    const { compiled, skipped, failures, registeredKernels } = result.correctness;
    if (registeredKernels !== registry.length || !Array.isArray(compiled) || !Array.isArray(skipped) || !Array.isArray(failures) || failures.length) throw new Error("incomplete kernel compilation ledger");
    const covered = new Set([...compiled, ...skipped].map((item) => item.kernel));
    if (registry.some((name) => !covered.has(name)) || [...covered].some((name) => !registry.includes(name))) throw new Error("compilation ledger does not match the registry");
    if (skipped.some((item) => !Array.isArray(item.unavailable) || !item.unavailable.length)) throw new Error("unsupported pipeline has no capability reason");
  }
  if (harness.id === "isolated") {
    if (result.method !== "WebGPU timestamp-query" || !Array.isArray(result.cases) || !result.cases.length) throw new Error("isolated dispatches require GPU timestamps and cases");
    const selected = harnessKernels(harness, registry), skipped = result.skipped || [];
    if (!Array.isArray(skipped) || skipped.some((item) => !selected.includes(item.kernel) || !Array.isArray(item.unavailable) || !item.unavailable.length)
      || new Set(skipped.map((item) => item.kernel)).size !== skipped.length
      || JSON.stringify(result.skippedKernels || []) !== JSON.stringify(skipped.map((item) => item.kernel))) throw new Error("isolated capability skips lack valid selected-kernel reasons");
    const expected = selected.filter((name) => !skipped.some((item) => item.kernel === name));
    const tokens = [1, 47, 128, 512];
    if (result.fixtureSeed !== 20261106 || result.samples !== 7 || result.warmups !== 3 || JSON.stringify(result.tokens) !== JSON.stringify(tokens)) throw new Error("isolated sampling protocol differs from the requested seeded fixtures");
    if (result.cases.length !== expected.length * tokens.length || expected.some((name) => tokens.some((length) => result.cases.filter((item) => item.kernel === name && item.tokens === length).length !== 1))) throw new Error("isolated inventory omitted or duplicated a selected kernel shape");
    for (const item of result.cases) {
      if (!expected.includes(item.kernel) || !Number.isSafeInteger(item.tokens) || item.tokens < 1 || !positive(item.finiteValues) || !Array.isArray(item.timings) || !item.timings.length) throw new Error("invalid isolated fixture identity or output coverage");
      for (const timing of item.timings) if (!Array.isArray(timing.samples) || timing.samples.length !== result.samples || !timing.samples.every(positive) || !positive(timing.medianMs) || !/^[a-f0-9]{64}$/.test(timing.shaderSha256)) throw new Error("isolated fixture has invalid samples or shader hash");
      if (item.cpuMaxAbs !== null && (!finite(item.cpuMaxAbs) || item.cpuMaxAbs < 0 || item.cpuMaxAbs > 1e-5)) throw new Error("isolated CPU-reference check failed");
      if (result.exactComparisonRequired && (item.timings.length !== 2 || item.bitDifferences !== 0 || item.maxAbs !== 0)) throw new Error("required paired bit-exact check failed");
    }
  }
  if (harness.timing === "vision-profile") {
    const cases = result.correctness.cases;
    if (result.profile !== true || !Array.isArray(cases) || !cases.length || cases.some((item) => !Array.isArray(item.profile?.samples) || item.profile.samples.length < 3 || item.profile.samples.some((sample) => !sample.gpu || !Object.keys(sample.gpu).length || Object.values(sample.gpu).some((value) => !finite(value) || value < 0)))) throw new Error("vision profile lacks actual GPU timestamp samples");
  }
}

function timingEvidence(result) {
  const records = [];
  const walk = (value, path, inside = false) => {
    if (!value || typeof value !== "object" || isSkipped(value)) return;
    if (inside && /wall/i.test(value.method || "") && !/timestamp/i.test(value.method || "")) return;
    if (inside && !Array.isArray(value)) {
      const fields = Object.entries(value).filter(([key, number]) => /(?:Ms|p50|p95)$/.test(key) && positive(number));
      const series = Object.entries(value).filter(([, item]) => Array.isArray(item) && item.length >= 3 && item.every(positive));
      if (fields.length || series.length) records.push({ path, values: Object.fromEntries(fields), method: value.method || null,
        rawSamples: series.length > 0, ...(series.length ? { samples: Object.fromEntries(series) } : {}) });
    }
    for (const [key, child] of Object.entries(value)) {
      // Only explicit timing containers are eligible. metricMs and guard elapsed
      // wall time are deliberately excluded even when their values are positive.
      if (key === "timing" || key === "timings") walk(child, `${path}.${key}`, true);
      else if (!inside || typeof child === "object") walk(child, `${path}.${key}`, inside);
    }
  };
  walk(result, "result");
  return records;
}

export function kernelCoverage(registry, runs) {
  const rows = registry.map((name) => ({ name, status: "unmeasured", compileStatus: "not-run", numericalStatus: "not-run", timingStatus: "not-run", reason: "No dispatched evidence recorded.", harnesses: [], evidence: [] }));
  const byName = new Map(rows.map((row) => [row.name, row]));
  for (const run of runs) {
    const harness = KERNEL_HARNESSES.find((item) => item.id === run.id);
    if (!harness) throw new Error(`unknown kernel harness ${run.id}`);
    const names = harnessKernels(harness, registry);
    for (const name of names) {
      const row = byName.get(name);
      row.harnesses.push(harness.id);
      row.evidence.push({ harness: harness.id, status: run.status, source: run.source || null, ...(run.reason ? { reason: run.reason } : {}) });
    }
    if (run.status !== "passed") continue;
    const outcome = validateKernelResult(harness, run.result, registry);
    if (outcome?.status === "skipped") {
      for (const name of names) Object.assign(byName.get(name).evidence.at(-1), outcome);
      continue;
    }
    const cases = focusedCases(run.result).filter((item) => !isSkipped(item));
    const explicitKernels = cases.flatMap((item) => caseKernels(harness, item)).filter((name) => registry.includes(name));
    const dispatched = names.filter((name) => !run.result.skippedKernels?.includes(name) && (!explicitKernels.length || explicitKernels.includes(name))
      && (harness.timing !== "vision-profile" || run.result.correctness.cases.some((item) => Object.entries(item.profile.kernelMedianMs || {}).some(([label, value]) => VISION_LABELS[name].some((prefix) => label.startsWith(prefix)) && positive(value)))));
    for (const name of names.filter((item) => !dispatched.includes(item))) {
      const skippedCases = focusedCases(run.result).filter((item) => isSkipped(item) && (!caseKernels(harness, item).length || caseKernels(harness, item).includes(name)));
      const reasons = [...new Set([...(run.result.skipped || []).filter((item) => item.kernel === name).flatMap((item) => item.unavailable || [item.reason]), ...skippedCases.map((item) => item.reason || item.skipReason)].filter(Boolean))];
      Object.assign(byName.get(name).evidence.at(-1), { status: "skipped", reason: reasons.join("; ") || "No matching dispatched kernel evidence was reported by this harness." });
    }
    if (harness.id === "compile") {
      for (const row of rows) {
        const compiled = run.result.correctness.compiled.filter((item) => item.kernel === row.name).length;
        const unsupported = run.result.correctness.skipped.filter((item) => item.kernel === row.name);
        row.compileStatus = compiled ? "passed" : "unsupported";
        row.compilation = { compatibleVariants: compiled, unsupportedVariants: unsupported.length, missingCapabilities: [...new Set(unsupported.flatMap((item) => item.unavailable))] };
      }
      continue;
    }
    if (harness.id === "isolated") {
      for (const name of dispatched) {
        const row = byName.get(name), cases = run.result.cases.filter((item) => item.kernel === name);
        const cpu = cases.every((item) => item.cpuMaxAbs !== null);
        const exact = run.result.exactComparisonRequired === true && cases.every((item) => item.bitDifferences === 0);
        if (cpu || exact) row.numericalStatus = "passed";
        else if (row.numericalStatus !== "passed") row.numericalStatus = "finite-only";
        row.timingStatus = "isolated";
        row.isolated = cases.map((item) => ({ tokens: item.tokens, grid: item.grid, spec: item.spec, repetitions: item.repetitions, cpuMaxAbs: item.cpuMaxAbs, bitDifferences: item.bitDifferences,
          timings: item.timings }));
        row.evidence.at(-1).numericalScope = cpu ? "full-output CPU reference" : exact ? "bit-exact output equivalence to the supplied baseline WASM" : "finite-output and overwrite smoke only";
      }
      continue;
    }
    for (const name of dispatched) {
      const row = byName.get(name);
      row.numericalStatus = "passed";
      row.evidence.at(-1).numericalScope = harness.numerical === "cpu" ? "independent CPU reference" : "focused registered-kernel numerical guard; see raw result for tolerance and scope";
    }
    if (harness.timing === "vision-profile") {
      for (const name of dispatched) {
        const prefixes = VISION_LABELS[name];
        const measurements = (run.result.correctness.cases || []).flatMap((item) => Object.entries(item.profile?.kernelMedianMs || {}).filter(([label, value]) => prefixes.some((prefix) => label.startsWith(prefix)) && positive(value)).map(([label, medianMs]) => ({ fixture: item.id, label, medianMs })));
        if (measurements.length && byName.get(name).timingStatus !== "isolated") {
          byName.get(name).timingStatus = "profiled";
          byName.get(name).evidence.at(-1).measurements = measurements;
        }
      }
    } else if (harness.timing !== "none") {
      const measurements = timingEvidence(run.result);
      const declaredWall = /wall clock|wall time/i.test(run.result.method || "") && !/timestamp/i.test(run.result.method || "");
      if (measurements.length && !declaredWall) for (const name of dispatched) {
        const row = byName.get(name);
        if (!["isolated", "profiled"].includes(row.timingStatus)) row.timingStatus = "composite";
        row.evidence.at(-1).timingScope = "paired or composite fixture; not attributable to this kernel alone";
        row.evidence.at(-1).measurements = measurements;
      }
    }
  }
  for (const row of rows) {
    const failure = row.evidence.find((item) => item.status === "failed");
    const unavailable = row.evidence.filter((item) => item.status === "skipped");
    if (failure) { row.status = "failed"; row.reason = failure.reason || "A kernel fixture failed."; }
    else if (row.compileStatus === "unsupported") { row.status = "unsupported"; row.reason = `No compatible compiled specialization: ${row.compilation.missingCapabilities.join(", ")}`; }
    else if (row.compileStatus === "passed" && row.numericalStatus === "passed" && row.timingStatus !== "not-run") {
      row.status = "measured";
      row.reason = row.timingStatus === "composite" ? "Numerical guard passed; timing covers a composite operation, not this kernel alone." : "Numerical guard and GPU timing passed for the recorded fixtures.";
    } else if (row.numericalStatus === "passed") { row.status = "validated"; row.reason = "Numerical guard passed; no attributable GPU timestamp result recorded."; }
    else if (row.timingStatus !== "not-run") { row.status = "timed"; row.reason = "GPU timestamps exist, but independent or baseline numerical equivalence is not established by this run."; }
    else if (row.compileStatus === "passed") { row.status = "compile-only"; row.reason = "Compatible pipeline compilation passed; dispatched numerical and timing coverage is incomplete."; }
    if (unavailable.length) row.skipped = unavailable.map((item) => ({ harness: item.harness, reason: item.reason }));
  }
  const counts = {
    registered: rows.length, compiled: rows.filter((row) => row.compileStatus === "passed").length,
    numericallyValidated: rows.filter((row) => row.numericalStatus === "passed").length,
    isolatedTimed: rows.filter((row) => row.timingStatus === "isolated").length,
    profiledTimed: rows.filter((row) => row.timingStatus === "profiled").length,
    compositeTimed: rows.filter((row) => row.timingStatus === "composite").length,
    failed: rows.filter((row) => row.status === "failed").length,
    unsupported: rows.filter((row) => row.status === "unsupported").length,
    unmeasured: rows.filter((row) => !["measured", "unsupported"].includes(row.status)).length,
  };
  return { registered: registry.length, counts, kernels: rows, coverageComplete: counts.unmeasured === 0 && counts.failed === 0 };
}
