# Profiling models and kernels

Use this workflow to compare the registered WebGPU kernels and the current model catalog on one measured browser and adapter. The [current report](benchmarks/kernel-profile-2026-11-06/summary.json) keeps hardware, precision, correctness, and missing coverage separate. The website shows the report under [How it works → Speed](https://bvolpato.github.io/kevala/#/how/bench).

See the [native and experimental performance reference](performance-reference.md) for the separate cuBLAS FP32 timing scope, Chrome feature gate, and candidate screens that were not promoted.

The [native Q8 AVX2 row report](native-q8-performance.md) records a separate CPU-only dequantization experiment. Its row timings do not measure WebGPU dispatch or model latency.

The [native AVX2 matrix report](native-linear-performance.md) separates projection timings from
native model inference calls after load. It retains the small-shape controls and labels
fresh-process request measurements separately from warm resident model latency.

The [Clef GPU head report](clef-head-performance.md) separates exact projection and full-logit
checks from paired complete-request wall timings. Compare `clefHead: "auto"` with `"cpu"` on
the same GPU backbone; keep load, warmup, and first-use validation outside warm-request timings.

These measurements identify the best tested candidates for the recorded shapes and feature set. They do not prove a universal hardware optimum or model accuracy.

## Prepare the checkout

Follow [the contributor setup](../CONTRIBUTING.md), then build with the pinned Rust toolchain. If another Rust installation precedes rustup on PATH, select the rustup shims for this command:

```sh
PATH="$HOME/.cargo/bin:$PATH" pnpm build
pnpm check:wasm
pnpm serve --port 18171
```

Keep model packs in `tmp/` as `<model>-<precision>.kevala`. The runners do not download weights. They check the pack header, architecture, model identity, revision, precision, and length. They hash each pack before and after measurement. A local conversion without a catalog-pinned digest is identified as a local conversion, not as a verified upstream artifact.

Do not stop unrelated applications to clear the workstation. Record host and GPU load, retain raw samples, and treat noisy comparisons as unresolved. Large-pack runs require at least 32 GiB available host memory and 4 GiB free swap. Use an OS memory scope when a browser run can consume substantial memory.

## Check the measurement plan

```sh
pnpm profile:models --list --precisions=q8,bf16
pnpm profile:kernels --list
```

The model plan starts from the current catalog. Missing packs, unsupported precisions, insufficient memory, and unavailable GPU capabilities remain named rows. The kernel plan starts from the Rust registry. Adding a kernel cannot silently remove it from the coverage denominator.

The runners share an OS-owned lock at `~/.cache/kevala/gpu-profile.lock`, including across worktrees and symlinked checkout paths. Do not remove a lock to bypass an active run. Do not nest an external lock around a runner that already owns this lock.

## Profile every available model

Start with a small plumbing check, then use the full sampling protocol:

```sh
pnpm profile:models --base=http://127.0.0.1:18171/ --models=laya --samples=3 --warmups=1 --blocks=2 --output=tmp/profile-smoke
pnpm profile:models --base=http://127.0.0.1:18171/ --precisions=q8,bf16 --samples=9 --warmups=3 --blocks=2 --output=tmp/profile-models
```

The default `--protocol=resident` loads each model once with profiling disabled, then toggles profiling between wall and GPU diagnostic lanes. Each block runs these lanes separately. The next block reverses their order. Requests use the same deterministic states, question definitions, and actual token counts. Pack and response/state caches are disabled. Model-internal resident buffers and guarded kernel optimizations remain enabled. Measurements exclude loading, tuning, and fixture hashing.

Resident lanes record a shared session ID, model-load ID, and exact canonical JSON and SHA-256 for the load-time GPU tuning snapshot. The snapshot is detached from runtime objects and transported as a string before parsing, so diagnostic numbers retain their original JSON representation. This identifies one load; per-shape caches may still evolve during measurement.

Use `--protocol=isolated` to retain the older protocol, which loads a fresh model for every lane. Independent calibration can select different kernels and change raw probabilities. Keep these failed comparisons and their raw samples. Do not increase the probability tolerance to make them pass, or combine resident and isolated captures in one report. `--timeout` is the per-lane budget; the resident browser budget covers all lanes.

The default cases are short, medium, and long text requests. Use `--shapes=0,1,2,3,4` for additional multi-question and batch cases when the model and device have enough memory. The default text cases do not establish image-inference or batch performance.

Each model/precision/browser row uses the on-page engine in a fresh isolated browser. Resident lanes share that browser and model load; isolated lanes start separate browsers. The report records that context. These wall timings are completed model-request latency, not camera sensor-to-screen or worker/UI latency.

GPU profiles change execution. Each dispatch has its own timed pass, optional paths can change, and profiling disables the normal long-request chunking in some trunks. The default unprofiled submission policy waits between long-request chunks. A profile can therefore have very different wall latency. Do not subtract profiled kernel totals from unprofiled wall time to estimate overhead, or use profiled wall latency as an application speedup.

Keep per-kernel medians and shares for attribution. Keep wall samples, p95, and block medians for latency. The runner checks backend and adapter identity, request hashes, token counts, model responses, normalized probabilities, and complete finite timestamp samples. Invalid measurements fail validation. Repeated same-runtime outputs must remain within the declared probability tolerance.

See the [2026-11-06 optimization comparison](model-optimization-performance.md) for scoped before/after results, complete mixed model timings, numerical guards and archived evidence.

## Profile the complete kernel registry

```sh
pnpm profile:kernels --base=http://127.0.0.1:18171/ --browser=firefox --output=tmp/profile-kernels --timeout=300 --budget=3600
```

Each registered kernel has separate compilation, numerical, and timing states. A compilation pass does not establish dispatched execution or numerical correctness. Finite outputs alone do not establish agreement with a reference. An unsupported specialization remains skipped with its required capability.

Isolated fixtures use deterministic inputs, warmups, repeated GPU timestamp samples, and shader hashes. Paired candidates take turns to reduce ordering and clock-ramp bias. Focused fixtures also exercise source, reduction, fusion, image, and loaded-model paths. A projection plus its reduction or epilogue is labeled composite timing. Its total must not be attributed independently to each component.

Use `--baseline-wasm=<same-origin URL>` to compare emitted kernels with a frozen control. Matching labels or filenames are insufficient: retain the actual generated shader hashes, checked outputs, shapes, precision, browser, and adapter.

## Compare a runtime change

Freeze both complete runtime directories, including the three WASM flavors, beneath the local server. Pass their entry points through `--runtime=` to the model runner. Use the same pack SHA-256, browser binary and flags, device features, requests, token counts, cache policy, and submission settings.

Interleave baseline and candidate blocks in an ABBA order for affected models. Retain a new kernel choice only after its numerical guard passes and paired timings show a repeatable gain. The runtime's existing matrix selector requires a stable gain of at least 5 percent. Recheck unprofiled model latency and decision outputs before promoting a kernel-level result to a model-level claim.

Load-time tuning diagnostics are evidence: retain selected shapes, token buckets, fallback reasons, and deadlines. A truncated calibration does not prove that every candidate was considered. The general selector keeps its default six-shape budget. Gemma can schedule up to twelve loaded shapes within the same calibration deadline, covering its larger set of projection shapes.

## Record browser capabilities

Firefox and Chromium can expose different precision and subgroup features on the same GPU. Keep those results separate. A software fallback cannot count as a hardware benchmark.

On this Linux workstation, Firefox used the NVIDIA Vulkan ICD. The Chromium hardware run used an isolated Chrome for Testing process with the flags below and without `VK_DRIVER_FILES`:

```sh
CHROME_FOR_BENCH=/path/to/chrome-for-testing
env -u VK_DRIVER_FILES pnpm profile:models --browser=chromium --binary="$CHROME_FOR_BENCH" \
  --chromium-arg=--enable-gpu --chromium-arg=--enable-unsafe-webgpu \
  --chromium-arg=--use-angle=vulkan --chromium-arg=--enable-features=Vulkan \
  --chromium-arg=--disable-vulkan-surface --output=tmp/profile-models-chromium
```

Use the same binary and flags with `profile:kernels`. These flags apply only to the isolated benchmark browser. See [Chromium's headless GPU guidance](https://chromium.googlesource.com/chromium/src/+/refs/heads/main/docs/gpu/using-gpu-hardware-in-headless-chrome.md) and [Chrome's Web AI testing guide](https://developer.chrome.com/blog/supercharge-web-ai-testing). Preserve the flags in the raw runner metadata.

## Publish a report

Combine only compatible captures of the current runtime. The report builder rejects mixed source maps, conflicting model rows, or incompatible sampling protocols. Baseline-control runs belong in comparison evidence, not in the current-model table.

```sh
pnpm exec node scripts/report-kernel-profile.mjs \
  --models=tmp/profile-models/summary.json \
  --models=tmp/profile-models-chromium/summary.json \
  --kernels=tmp/profile-kernels/summary.json \
  --evidence=tmp/controlled-comparison.json \
  --output=docs/benchmarks/kernel-profile-2026-11-06/summary.json
```

The public summary keeps measurements, coverage, and limitations readable. Deterministic gzip artifacts retain source manifests and raw samples. Use repeated `--evidence=` options for additional JSON comparison or numerical-guard artifacts; they do not affect coverage or promote an optimization claim. No model weights are published by this process. A capture made from an uncommitted worktree remains labeled that way and is identified by its source fingerprints. Host-specific build paths can change WASM bytes without changing emitted WGSL. For a deployment built elsewhere, verify the JavaScript source hashes and compare the WGSL emitter outputs for the captured specializations before treating the deployment as equivalent.

Before landing, run the relevant contributor checks and review the generated report. After landing, verify CI and Pages on the exact merged commit, then inspect the served report and runtime. A successful local benchmark does not prove deployment.
