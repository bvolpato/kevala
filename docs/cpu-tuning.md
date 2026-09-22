# Automatic CPU tuning

Kevala selects a CPU register tile and, for supported Laya packs, measures worker counts on the
current browser. Applications can override either choice. The demo has no additional controls.

```js
const model = await Kevala.load({
  model: "laya",
  backend: "wasm",
  cpuKernel: "auto", // default; also "2x4" or "4x4"
  threads: "auto",   // default; also an integer from 1 through 16
});
console.log(model.info.cpuTuning);
```

A numeric `threads` skips worker-count calibration. It is capped by the model's supported shard
count, preserving the existing behavior. Kev currently supports one CPU worker. Explicit counts
can exceed the browser's reported core count, so applications can measure their own workload.
Invalid values, including fractional counts and numeric strings, are rejected.

```js
// Reproducible settings chosen by the application; neither choice is benchmarked at load time.
const fixed = await Kevala.load({ model: "laya", backend: "wasm", threads: 8, cpuKernel: "2x4" });

// Measure again after changing power mode or the machine's competing workload.
const refreshed = await Kevala.load({ model: "laya", backend: "wasm", retune: true });
```

`kevala/node` accepts `cpuKernel` too. Its synchronous inference runtime has one compute instance;
`threads` accepts `"auto"` or `1` and rejects larger values. Browser WebGPU loads skip CPU tuning.

## What is measured

The register-tile selector warms both production tiles, alternates paired measurements, and requires
at least a 10% median improvement and two of three paired wins before choosing `4x4`. The chosen
tile is shared by the coordinator and every compute worker. Worker instances do not retune separately.

Laya's worker probe runs the production Q8 up projection, GELU gating, and down projection using
deterministic weights. It tests 32- and 128-row batches, including activation copies, round-trip
worker transfers, and ordered WASM SIMD reduction. The partition widths match the model's uneven
block-aligned partitions. A one-worker probe has no worker transfers or reduction.

Candidates are powers of two plus the available cap, bounded by the browser's reported cores,
16 workers, and the model's attention-head and MLP partition limits. Rust's pack-layout validator
also filters out counts that violate a custom pack's alignment constraints. Each shape gets three warmups
and three measured batches lasting at least 25 ms, capped at 32 passes. The score is the geometric
mean of the two shape medians. The selector prefers the smallest count within 10% of the fastest.
It finishes each candidate before checking a five-second calibration budget. This is a cooperative
budget, not a hard wall-clock deadline. Probe replies have a separate ten-second timeout.

The temporary probe workers are terminated before the selected production pool is created. Cancelling
a load terminates both kinds of workers. Probe allocation or timing failure falls back to one worker
and records the reason. A worker script startup failure rejects the load rather than hanging.

The probe represents matrix and coordination costs. It omits the final residual add, the whole model,
attention, and other input lengths. It is a practical starting point; application-specific measurements
and overrides remain useful. Custom architecture plugins without a compatible probe retain a core
and model limit. Unsupported Laya probe dimensions also use that limit.

## Cache and diagnostics

Successful profiles are stored in the origin's Cache API for seven days. The key includes the tuning
revision, browser user agent, reported cores, WASM flavor and base URL, model configuration, and API
overrides. `retune: true` replaces the profile. `cache: false` skips reading and writing it. If storage
is unavailable, calibration still works for the current load. `clearCache()` removes tuning profiles
alongside model packs.

Kernel or probe changes must bump the tuning revision. A different binary served at the same custom
`wasmBase` URL is not detected automatically; use `retune: true` when replacing it.

`info.cpuTiles` reports every actual compute instance's numeric tile (`0` is `2x4`, `1` is `4x4`).
`info.cpuTuning` includes:

- `kernel`, `threads`: the applied choices.
- `source`: `measured`, `override`, or `cache`.
- `kernelSource`, `threadSource`: whether each choice was measured, overridden, limited by the model
  or hardware, or selected as a fallback.
- `measurements`: the measured per-pass timings for each candidate and shape.
- `complete`: whether all candidates finished; incomplete or failed calibration is not cached.
- `tuningMs`: work spent on tuning or reading its cache during this load.
- `reason`: the error if calibration fell back to one worker.

The recorded timings are local diagnostics. No tuning data is sent to a server.

## Validation on Ubuntu

[Recorded samples and checks](benchmarks/cpu-tuning-linux-2026-09-22.json), September 22, 2026:
Ryzen 9 9950X3D, Firefox 152.0.3 and Chrome 149.0.7827.200, relaxed SIMD, local Laya Q8 pack.
Each latency below is the geometric mean of three shape medians (47, 140, and 512 tokens), with
two warmups and five samples per shape. Unique states prevent cache hits. Loading is excluded.

| Browser | Automatic choice | Automatic, ms | Manual 8, ms | Manual 16, ms |
| --- | --- | ---: | ---: | ---: |
| Firefox | 2x4, 8 workers | 476.63 | 471.34 | 618.25 |
| Chrome | 2x4, 16 workers | 258.21 | 293.98 | Not separately measured |

Firefox's automatic result was within 1.1% of manual eight-worker inference and 22.9% faster than
16 workers. Chrome's selected 16 workers were 12.2% faster than eight. The browser matters even
on the same CPU. These measurements do not establish an optimum for other machines or workloads.

An earlier probe timed individual calls as short as 2–3 ms. It selected 16 workers in Firefox,
where real inference was slower. That probe was rejected. The retained version batches timings
for at least 25 ms and selected eight workers in both subsequent Firefox runs.

Uncached calibration took 1.37 seconds in the final Firefox run and 1.40 seconds in Chrome. A
separate Chrome API check loaded from an existing profile in 0.54 seconds total, with 11.1 ms
spent validating candidates and reading the profile. These are individual local-pack loads.

Validation also covered:

- 37 JavaScript tests, including cache expiry/isolation, malformed options, timeout/abort cleanup,
  worker startup errors, and replacing the probe pool.
- Three native probe tests and an independent WASM numerical check of both tiles in all three
  flavors. Maximum output error was below 4.2e-9; the run did not grow memory.
- Real browser cache reuse, explicit `threads: 3` / `cpuKernel: "4x4"` on all workers, and cancellation
  that terminated every created worker. A custom-header check rejected an unaligned three-way split.
- CPU reference decisions: Laya 41/41 and Kev 13/13, with maximum probability differences 0.02400645
  and 0.00968519. Cache hit/extension probabilities matched fresh inference exactly.
- A fresh NVIDIA RTX 5070 Ti check: 48 GPU matrix cases, both model parity suites, and verified
  cache miss/hit/extension counters. CPU tuning is bypassed by this backend.
- A rebuilt package containing all three generated WASM files, an inference from its Node entry
  point with the tile override, and staged-site validation.

Reproduce the selector comparison with the setup from [cpu-benchmarks.md](cpu-benchmarks.md#reproduce):

```sh
pnpm exec node scripts/cpu-suite.mjs tmp/auto-firefox --models=laya --threads=auto --warmups=2 --runs=5
pnpm exec node scripts/cpu-suite.mjs tmp/manual-firefox --models=laya --threads=8 --warmups=2 --runs=5
pnpm exec node scripts/cpu-suite.mjs tmp/auto-chrome --browser=chromium --models=laya --threads=auto --warmups=2 --runs=5
pnpm exec node scripts/check-cpu-tuning.mjs
```
