# CPU kernel profiling and benchmarks

Measured on September 22, 2026, on Ubuntu with a Ryzen 9 9950X3D, Rust 1.95.0,
Node 24.21.0, Firefox 152.0.3, and Chrome 149.0.7827.200. The browser tests force
WebAssembly, so these measurements do not use a GPU.

[Recorded samples and validation results](benchmarks/cpu-linux-2026-09-22.json) include
the trial history, actual worker tiles, latency samples, and profile summaries.

This report records the kernel sweep with fixed worker counts. The subsequent
[automatic CPU selector](cpu-tuning.md) measures worker counts and exposes API overrides.

## Profile first

Node's inspector sampled the production WebAssembly implementation on two medium and two
long requests per model, after one warmup. This is a single-instance V8 profile, not a Firefox
profile or a browser latency measurement. Functions inlined into their caller appear under that
caller. The named WASM build retained debug names; the benchmark builds remain stripped.

| Sampled function | Laya | Kev 0.8B |
| --- | ---: | ---: |
| Matrix multiplication, including panel dequantization | 78.2% | 81.9% |
| Scalar exponential | 10.1% | 6.1% |
| Kev model loop, including inlined work | N/A | 11.6% |
| Laya attention orchestration | 8.6% | N/A |

Embedding and RMS normalization each accounted for less than 0.1% of the sampled time.
They were not worth additional tuning in this sweep. Work concentrated on matrix multiplication,
Laya softmax, and the cost of coordinating parallel matrix work. A recurrence-layout experiment
in Kev's model loop had already regressed browser latency and was reverted.

## Retained changes

- Reuse transferred Laya worker buffers across layer steps. Each worker writes its partial result
  into the received buffer and transfers it back, avoiding a new full residual allocation per step.
- Accumulate ordered worker partials and the residual in WASM with SIMD addition. The order and
  f32 rounding are unchanged. The coordinator reacquires views after calls that can grow memory.
- Increase the activation block target from 256 KiB to 1 MiB. Each dequantized weight panel is
  reused across more token rows before it is rebuilt. The numerical operations within each output
  dot product are unchanged.
- Stabilize startup tile selection with paired measurements. Choose the larger register tile only
  when its median is at least 10% faster and it wins at least two of three pairs. Node and the browser
  use the same selector. This corrects a problem discovered during final validation, after the sweep.

The CPU implementation already used 128-bit SIMD and Laya tensor parallelism. These changes
improve those paths. Laya supports multiple workers; Kev remains a single-worker CPU model.
Splitting Kev requires a cache-aware execution interface, rather than copying the full model into
independent workers. Native x86, ARM/NEON, and other machines were not performance-tested here.

## Experiment results

The tuning metric is the geometric mean of six browser wall-time medians: short, medium, and long
requests for each model. Laya uses four compute workers; Kev uses one. Every request has a distinct
state to avoid inference-cache hits. Model loading is excluded. Each trial uses one warmup and three
samples per shape. Lower is better.

| Iteration | Experiment | Combined ms | Decision |
| --- | --- | ---: | --- |
| 0 | Original CPU implementation | 1,329.97 | Baseline |
| 1 | Packed 4×8 output-lane matrix tile | 1,414.78 | Revert |
| 2 | Reuse worker transfer buffers | 1,299.58 | Keep |
| 3 | Tile Kev recurrence values | 1,478.39 | Revert |
| 4 | Packed 4×8 tile with 256-element K panels | 2,087.07 | Revert |
| 5 | 3×4 dot-product tile | 1,911.40 | Revert |
| 6 | Add sampled profiles and actual tile metadata | N/A | Refine |
| 7 | 3×3 dot-product tile | 1,366.03 | Revert |
| 8 | Switch from register layouts to reduction and math | N/A | Pivot |
| 9 | SIMD worker reduction | 1,260.26 | Keep |
| 10 | Four-lane softmax exponential | 1,272.03 | Revert |
| 11 | 64 KiB activation blocks | 1,784.49 | Revert |
| 12 | 1 MiB activation blocks | 1,192.28 | Keep |

All completed experiments passed model and cache probability checks. Matrix-layout experiments
also passed 1,400 numerical cases spanning F32/Q8 weights, bias, both register tiles, odd row and
column tails, and short inner dimensions. The rejected vector exponential passed 125,156 actual
WASM inputs with at most 1 ULP error, plus native attention tests; it still made Laya slower.
Accuracy alone was not a reason to retain it.

The bounded sweep retained three changes and reverted seven. Its best six-case metric improved
10.4% over its initial baseline. The two remaining iterations changed measurement strategy.
These are measurements on one workstation, not a claim of universally optimal kernels.

## Final browser validation

The independent comparison uses two warmups and five samples per input size. Baseline CPU kernels
are from `5f1a710`, with measurement tooling added in `fc8e1ee`. The retained kernel and reduction
changes are at `4343607`; the startup selector fix is at `90ea5ec`. All compute instances selected
tile 0 in the before/after comparisons reported here.

An earlier final Firefox run selected tile 1 on the local Laya worker and tile 0 on the other three.
Its Laya result was 706.61 ms, versus 591.06 ms in the tuning trial with identical WASM bytes and
all workers on tile 0. That exposed the short timing probe's sensitivity to noise. A fake-clock
regression reproduces the old wrong choice and passes with the new selector. The corrected Firefox
run selected tile 0 throughout and measured 601.09 ms.

| Browser | Model | Before, ms | After, ms | Reduction |
| --- | --- | ---: | ---: | ---: |
| Firefox | Laya | 679.35 | 601.09 | 11.5% |
| Firefox | Kev 0.8B | 2,602.37 | 2,407.09 | 7.5% |
| Chrome | Laya | 566.52 | 478.60 | 15.5% |
| Chrome | Kev 0.8B | 2,260.86 | 2,210.19 | 2.2% |

Across the six cases, Firefox improved from **1,329.63 to 1,202.87 ms (9.5%)** and Chrome from
**1,131.74 to 1,028.50 ms (9.1%)**. All six cases improved in both browsers. Laya inputs contain
47, 140, and 512 tokens; Kev inputs contain 31, 124, and 532 tokens.

Repeated startup probing adds loading work, which the inference metric excludes. The recorded
Firefox loads were 1.26 versus 1.76 seconds for Laya and 2.03 versus 2.82 seconds for Kev; these are
individual local-pack loads, not a controlled download or startup benchmark.

## Worker scaling

Laya's existing tensor parallelism scales the expensive matrix work across Web Workers. The final
Firefox implementation produced these geometric means across the same three input sizes:

| Compute workers | Geometric mean, ms | Short / medium / long, ms |
| ---: | ---: | --- |
| 1 | 1,824.95 | 595 / 1,614 / 6,329 |
| 2 | 1,008.06 | 330 / 905 / 3,430 |
| 4 | 601.09 | 202 / 532 / 2,021 |
| 8 | 478.32 | 162 / 423 / 1,597 |

Eight workers were 3.8× faster than one and 20.4% faster than four on this host. The 1/2/8-worker
runs use one warmup and three samples per shape; the four-worker row uses the final five-sample run.
At the time of this sweep, the default chose at most eight workers, bounded by available cores and
the model's shard limit. No hardware-specific worker count was hardcoded. Every final scaling worker
selected tile 0. The subsequent automatic selector replaces that fixed cap with measured choices.

The original eight-worker runtime measured 587.46 ms in a separate run, with one worker selecting
tile 1. The final result is 18.6% lower, but that comparison includes the selector fix as well as the
kernel and buffer improvements. It should not be attributed to matrix blocking alone.

```js
const model = await Kevala.load({ model: "laya", backend: "wasm", threads: 8 });
```

Measure worker counts in the target application: more workers consume more memory and add transfer
and reduction work. Kev's CPU shard limit remains one, regardless of the requested thread count.

## Correctness and continuous checks

- Firefox and Chrome match all 41 Laya and 13 Kev reference decisions. Maximum absolute probability
  differences remain 0.02400645 and 0.00968519, respectively, within the fixed 0.03 and 0.011 guards.
- CPU cache hits and extensions match fresh inference exactly in the browser checks. The native
  pack-backed integration test additionally verifies one miss, one hit, and one extension.
- Each WASM flavor passes 1,408 matrix cases, including model-sized dimensions that cross the new
  row block with odd row and column tails. CI builds and runs these synthetic numerical guards.
- All 26 JavaScript tests pass, including six tile-selection cases. The outlier case was also run
  against the actual previous selector: it chose the slower tile before the fix and the expected tile
  afterward.
- The native workspace suite exercised 28 tests successfully. One optional Qwen base-tokenizer test
  skipped because its separate fixture was absent. The Laya and Kev tokenizer fixtures and Kev model
  pack were present; their integration tests ran.
- A fresh NVIDIA WebGPU check passed 48 matrix tile cases, both model parity suites, and the cache
  guard with verified miss/hit/extension counters. This confirms that the CPU changes preserve the
  GPU path; the CPU performance tables above do not include GPU timings.
- `pnpm pack` rebuilt all three WASM flavors. Package validation, staged-site validation, and a Laya
  inference using the unpacked Node entry point passed.

The browser p50 calculation now averages the two middle samples for even run counts, matching the
runner's median definition. All reported tuning and comparison runs use odd sample counts and are
unaffected by that correction. Browser cache results explicitly report whether counters were available;
CPU probability checks do not claim to have verified browser cache counters.

## Reproduce

Use the packs and pinned revisions documented in [gpu-benchmarks.md](gpu-benchmarks.md#reproduce).
Then build and serve the checkout:

```sh
pnpm install --frozen-lockfile
pnpm build
pnpm serve --port=18086
```

Run measurements serially, with no other build, inference, or profiling job using the machine:

```sh
pnpm exec node scripts/cpu-suite.mjs tmp/cpu-firefox --runs=5 --warmups=2
pnpm exec node scripts/cpu-suite.mjs tmp/cpu-chromium --browser=chromium --runs=5 --warmups=2
pnpm exec node scripts/cpu-guard.mjs tmp/cpu-guard
scripts/build-cpu-bench.sh all
pnpm exec node scripts/bench-cpu-kernels.mjs guard relaxed
pnpm exec node scripts/bench-cpu-kernels.mjs guard simd
pnpm exec node scripts/bench-cpu-kernels.mjs guard base
pnpm exec node scripts/bench-cpu-kernels.mjs benchmark relaxed
```

`KEVALA_BENCH_URL` selects a different checkout's server. `CHROMIUM_BIN` selects the Chromium
executable. The suite accepts `--threads`, `--flavor`, `--models`, and `--shapes`; its JSON records
the actual backend, worker count, and register tile of every compute instance.

The browser CPU cache check compares probabilities for a hit and extension against fresh inference.
Its CPU ABI does not expose cache counters. Run the native integration test with the local Kev pack
to verify that it actually exercised one miss, one hit, and one extension:

```sh
KEVALA_KEV_PACK="$PWD/tmp/kev-0.8b-q8.kevala" cargo test --release -p kevala --test kev_cache -- --nocapture
```

To collect a named sampling profile without changing the shipping artifacts:

```sh
CARGO_PROFILE_RELEASE_WASM_STRIP=false \
RUSTFLAGS='-C target-feature=+simd128,+relaxed-simd' \
cargo build --locked -p kevala-wasm --target wasm32-unknown-unknown \
  --profile release-wasm --target-dir target/cpu-profile
mkdir -p tmp/cpu-profile-runtime
cp js/src/node.js js/src/wasm.js js/src/cpu-tune.js tmp/cpu-profile-runtime/
cp target/cpu-profile/wasm32-unknown-unknown/release-wasm/kevala_wasm.wasm \
  tmp/cpu-profile-runtime/kevala-relaxed.wasm
pnpm exec node scripts/profile-cpu.mjs --runtime=tmp/cpu-profile-runtime/node.js \
  --pack=tmp/laya-q8.kevala --output=tmp/laya-profile --runs=2
```

The profiler writes a standard `.cpuprofile` and a JSON summary. Its samples are evidence for
choosing experiments; browser wall time and numerical checks decide whether an experiment stays.
