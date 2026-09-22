# Firefox Linux GPU kernel sweep

Measured on September 22, 2026, using Firefox 152.0.3 on Ubuntu, an NVIDIA RTX 5070 Ti
(595.71.05), and the Ryzen 9950X3D integrated AMD GPU (RADV, Mesa 25.2.8).
Firefox uses Vulkan for WebGPU on this platform. These results do not use CUDA.

[Recorded measurements and validation evidence](benchmarks/firefox-linux-2026-09-22.json)
include the five GPU samples per case, per-kernel medians, all trial decisions, and numerical results.

## GPU results

Each value is the geometric mean of three input sizes for one model. Times are milliseconds
spent in GPU kernels, measured with WebGPU timestamp queries. Lower is better.

| GPU | Model | Before | After | Reduction |
| --- | --- | ---: | ---: | ---: |
| RTX 5070 Ti | Laya | 25.51 | 21.34 | 16.3% |
| RTX 5070 Ti | Kev 0.8B | 45.18 | 28.21 | 37.6% |
| AMD integrated | Laya | 622.99 | 561.65 | 9.8% |
| AMD integrated | Kev 0.8B | 1,066.65 | 731.48 | 31.4% |

Across all six model/input cases, GPU time fell **27.7% on NVIDIA** and **21.4% on AMD**.
Every case improved. The longest Kev case fell from 169.04 to 97.70 ms on NVIDIA and
from 5,964.30 to 3,009.74 ms on AMD.

The twelve tuning trials retained eight changes and rejected four. Their best aggregate
NVIDIA result was 24.61 ms versus the initial 34.41 ms baseline. The table above comes
from a fresh comparison after tuning: 33.95 versus 24.54 ms on NVIDIA.

## Browser latency and Firefox queue waits

GPU timestamps exclude browser scheduling, CPU work, IPC, and readback. Profiling also disables
the engine's normal chunk pacing. Separate runs with profiling disabled measured these p50
latencies with the default `submit: "await"` setting:

| GPU | Model | Tokens | Before, ms | After, ms |
| --- | --- | ---: | ---: | ---: |
| NVIDIA | Laya | 47 / 140 / 512 | 100 / 100 / 601 | 100 / 100 / 600 |
| NVIDIA | Kev | 31 / 124 / 532 | 101 / 100 / 701 | 100 / 100 / 701 |
| AMD | Laya | 47 / 140 / 512 | 200 / 701 / 2,902 | 200 / 601 / 2,404 |
| AMD | Kev | 31 / 124 / 532 | 300 / 1,001 / 6,410 | 200 / 800 / 3,304 |

The NVIDIA runs hit roughly 100 ms completion intervals. The default scheduling drains the GPU
queue between long-pass chunks to give the compositor opportunities to run, so several waits
hide the kernel gains. This is consistent with the polling behavior documented in
[Mozilla bug 1870699](https://bugzilla.mozilla.org/show_bug.cgi?id=1870699). We measured the
completion timing without instrumenting Firefox internals.

The existing `submit: "split"` option queues the separate command buffers without waiting for
each chunk. With the final kernels on NVIDIA, it reduced the 512-token Laya case from **600 to
100 ms** and the 532-token Kev case from **701 to 200 ms**. These are scheduling comparisons
with identical kernels. Short requests still encounter the browser's completion delay.

```js
const model = await Kevala.load({ model: "laya", backend: "webgpu", submit: "split" });
```

Use this opt-in when latency matters and measure responsiveness in the target application.
Command-buffer boundaries preserve queue ordering, but do not guarantee a compositor yield.
The default remains `"await"`. Headless frame callbacks stayed near 17 ms in these runs;
that is not a substitute for testing a rendered application under GPU load.

```sh
pnpm exec node scripts/gpu-suite.mjs tmp/gpu-wall-split --wall --submit=split
```

## What changed

- Materialize a matrix accumulator vector before indexing its component. On this Firefox/NVIDIA
  combination, the original nested array/vector expression returned zero for columns 48 through 63
  of a 64-column tile in the three-row variant, in both f32 and f16. A seeded `45x132x96` case
  reproduced it: row 0, column 48 returned zero instead of 1.8406123. The exact compiler or driver
  stage responsible has not been isolated.
- Pad Laya's shared query/key rows and vectorize its score accumulation.
- Vectorize Kev's portable recurrent state and dot accumulators. The recurrence stays f32.
- Transpose current Kev attention keys into existing scratch storage for contiguous reads across
  invocations. Cached keys keep their existing layout.
- Interleave Kev's gate weights once during loading for contiguous GPU reads.
- Fuse Kev's convolution, SiLU, and Q/K normalization, preserving the two independent
  128-element reduction trees in each workgroup.
- Increase matrix split-K occupancy target from 128 to 256 and scale its cutoff accordingly.

Narrower quantized weight loads, 128-column matrix tiles, and smaller workgroups for Laya norm
and Kev RMS did not improve the aggregate and were reverted.

## Measurement method

- Baseline: `0596888`. It already includes the matrix correctness repair. Incorrect output from
  the original kernel is not used as a performance baseline.
- Final runtime: `2919831`. Later commits add benchmark tooling and documentation.
- Both versions build all three WebAssembly flavors with the Rust 1.95.0 toolchain.
- Packs are pinned to Hugging Face revision `e75a06d9329e19fe879f44cfcce33914fb96dade`.
- Two warmups, then five measured requests per shape. Each request gets a distinct state to
  bypass inference caches. Pack download and model loading are excluded from inference times.
- Laya inputs contain 47, 140, and 512 tokens; Kev inputs contain 31, 124, and 532 tokens.
- For each sample, sum the GPU kernel timings. Take the median across five samples, then the
  geometric mean across cases. The suite runs both models serially.
- The runner uses a fresh headless Firefox profile and enables WebGPU and worker WebGPU.
  It requires the WebGPU backend and rejects incomplete or invalid measurements.
- Only one Vulkan ICD is exposed per run. Both adapters report `isFallbackAdapter: false`,
  support `shader-f16` and `timestamp-query`, and do not expose subgroups. Software adapters
  are rejected by the runtime. NVIDIA activity was also checked with `nvidia-smi`.

These are measurements on one workstation. GPU clocks, desktop activity, browser versions,
and drivers affect results. Subgroup execution, Chromium, and Apple GPU performance were not
measured in this sweep.

## Correctness checks

The model error limits for normal feature-enabled runs were fixed before tuning: maximum absolute
probability differences of 0.024 for Laya and 0.011 for Kev, with every argmax matching the
reference. The corrected NVIDIA baseline already measured 0.010123 for Kev, above the earlier
0.0097 result on other hardware.

| GPU | Laya agreement | Laya max difference | Kev agreement | Kev max difference |
| --- | ---: | ---: | ---: | ---: |
| NVIDIA | 41/41 | 0.023906 | 13/13 | 0.010030 |
| AMD | 41/41 | 0.023606 | 13/13 | 0.010022 |

- Both GPUs passed 48 seeded CPU-reference matrix checks covering all four row variants,
  f32/f16 tiles, split and unsplit K, partial tiles, bias, residual, and ReLU.
- The existing GPU kernel checks stayed below `1e-4` maximum error on both adapters.
- Kev cache hits and extensions matched fresh inference exactly on both GPUs. Each run
  exercised one miss, one hit, and one extension.
- A custom 128-column, target-512 matrix configuration passed with 4,194,304 scratch floats,
  covering the benchmark's dynamic scratch sizing beyond the runtime's fixed allocation.
- A visible Firefox window on NVIDIA loaded Laya and passed all 41 reference decisions.
- With optional GPU features disabled and default device limits, Laya still matches 41/41
  decisions. Its maximum probability difference is 0.024006 in both the corrected baseline and
  final runtime. Both runs exceed the stricter 0.024 tuning guard by 0.000006; this inherited f32
  result is within the existing pack criterion of `< 0.03` in [packs.md](packs.md#checking-a-pack).
  Kev with optional features disabled matches 13/13 decisions with maximum difference 0.009685.
  The feature-enabled guards remain unchanged.
- All 29 Rust tests passed locally, including the pack-backed Kev cache test, plus all 20
  JavaScript regression tests. JavaScript syntax and Rust formatting checks passed.

## Reproduce

Install Rust with the `wasm32-unknown-unknown` target, pnpm, Firefox, and `uv`. The Python
runner declares Selenium as a script dependency, which `uv` installs in isolation.

```sh
mkdir -p tmp
pnpm install --frozen-lockfile
rev=e75a06d9329e19fe879f44cfcce33914fb96dade
curl -fL "https://huggingface.co/bvolpato/kevala-packs/resolve/$rev/laya-q8.kevala" -o tmp/laya-q8.kevala
curl -fL "https://huggingface.co/bvolpato/kevala-packs/resolve/$rev/kev-0.8b-q8.kevala" -o tmp/kev-0.8b-q8.kevala
pnpm build
pnpm serve --port=18086
```

In another terminal, select the intended GPU. ICD paths vary by distribution.

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
# AMD: /usr/share/vulkan/icd.d/radeon_icd.json
pnpm exec node scripts/gpu-suite.mjs tmp/gpu-profile
pnpm exec node scripts/gpu-suite.mjs tmp/gpu-wall --wall
pnpm exec node scripts/gpu-guard.mjs tmp/gpu-guard
pnpm test
```

The suite writes both model results and `summary.json`, with a scalar metric on the last stdout
line. `KEVALA_BENCH_URL` selects a server for another checkout. Serve the corrected baseline and
candidate on different ports and run them serially with identical packs and build tools.

For individual matrix kernels, open `dev/gpu-bench.html` or use the runner:

```sh
uv run scripts/bench-gpu.py --result gpu --output tmp/tiles.json \
  --url 'http://127.0.0.1:18086/dev/gpu-bench.html?cases=45x132x96,45x132x1024&kernels=matmul@256:3,matmul_h@256:3&samples=3&warmups=1'
```

Shapes are `T x N x K`; `K` must be a multiple of 32. `matmul` uses f32 tiles and `matmul_h`
uses f16 tiles with f32 accumulation. `@256` selects the split target, `:3` the rows per thread,
and `/2` two groups of 64 columns. The default `runtime` follows the engine's selected settings.
Small cases compare every output to a CPU dequantized reference, including bias, residual, and
ReLU epilogues. Larger cases check sampled positions and scan all output values for finiteness.

# Apple M4 Max, Chrome

Measured on September 22, 2026, with Chrome for Testing 151 on macOS and an Apple M4 Max (Metal,
32-lane subgroups, `shader-f16`). Before is `5f1a710` (the Firefox sweep merged); after is the
`bvolpato/kernel-research` branch. Same packs and method as above: GPU time is the sum of
per-kernel timestamps, the median of five requests per shape, then the geometric mean.

| Model | Tokens | Before, ms | After, ms | Change |
| --- | ---: | ---: | ---: | ---: |
| Laya | 47 / 140 / 512 | 9.3 / 25.1 / 67.6 | 8.9 / 23.2 / 57.0 | -4% / -7% / -16% |
| Kev 0.8B | 31 / 124 / 532 | 9.4 / 30.6 / 105.6 | 8.7 / 28.8 / 97.4 | -7% / -6% / -8% |

All six cases improved: the suite fell from 27.95 to 25.68 ms of GPU time (-8.1%), and from
29.9 to 27.6 ms of wall-clock p50 (-7.8%). The Firefox sweep's own changes were neutral on this
GPU (28.5 vs 28.6 ms), since it takes the subgroup kernels those changes did not touch.

## What changed

- **Laya attention in register tiles** (`attention_tile.wgsl`), FlashAttention-2's work split
  without matrix units: 64 queries per workgroup, 32 keys per step, each thread 4 queries x 2 keys
  for scores and 4 queries x 4 dimensions for the output, so a query's statistics stay with the
  same 16 threads. Q, K and V tiles are f16 in workgroup memory; the math is f32. At 512 tokens a
  global layer went from 0.82 to 0.18 ms and a windowed one from 0.24 to 0.07 ms; attention's
  share of Laya at 512 tokens fell from 14.3 to 3.5 ms.
- **Kev attention in query tiles** (`kev_attention_tile.wgsl`): 8 tokens x the 4 query heads that
  share a key/value head per workgroup, 8 lanes per row with the row's query and output slices in
  registers, keys and values staged once per 16 keys. 11.7 to 4.9 ms at 532 tokens.
- **Kev gates** split the 1024-long dot product over 8 slices of 32 threads: 0.98 to 0.26 ms at 31
  tokens.
- **Int8 widening** without int-to-float conversions (Marlin's exponent trick), bit for bit the
  same: about 1% of matmul time.

## For GPUs without subgroups

Firefox exposes `shader-f16` but no subgroups on the GPUs above, where attention and the
recurrence are a quarter to a third of the time. Two kernels gained variants for such GPUs, and
were measured here by loading with only some features (`gpuFeatures`, `baseline=1`):

- `attention_tile` without subgroups trades row statistics through workgroup memory (30 KB). With
  only f16 and timestamps, Laya's attention at 512 tokens takes 4.1 ms instead of about 21 ms
  with the portable kernel; per layer, 1.28 to 0.21 ms (global) and 0.31 to 0.08 ms (windowed).
- The lane-split recurrence without subgroups combines its four lanes through workgroup memory,
  one more barrier per token. With no optional features: 37.7 to 11.2 ms at 532 tokens, and Kev's
  GPU time falls 14%.

These variants were not measured in Firefox. Laya's parity with f16 and no subgroups is 41/41 at
max |dp| 0.0235, close to the 0.024 tuning guard: check it on NVIDIA and AMD before relying on it.

## Tried and dropped

On this GPU the matmul did not get faster with any of: a square 8 x 4 subgroup footprint, 8 rows
x 4 columns per thread (128 threads), two quantization blocks per step, or swizzling the input
tile. Each was 1-15% slower or within noise. The attention agent's tuning of the tiled kernel (P
stored key-major, keys at a stride of 16, a deferred row sum, exp2) was also up to 9% slower. The
next step for the matmul is Chrome's `chromium-experimental-subgroup-matrix`, available only
behind a flag.

## Reproduce

Serve the checkout and use a Chrome that listens for DevTools; the runner opens its own context
there, raises and sizes its window (Chrome slows covered windows), and closes it.

```sh
export KEVALA_BENCH_CDP=http://127.0.0.1:9333
KEVALA_BENCH_URL=http://127.0.0.1:8123 node scripts/gpu-suite.mjs tmp/gpu-profile
KEVALA_BENCH_URL=http://127.0.0.1:8123 node scripts/gpu-suite.mjs tmp/gpu-portable --baseline
uv run scripts/bench-gpu.py --result gpu --url 'http://127.0.0.1:8123/dev/attn-bench.html?cases=512:0,512:64&kernels=attention,attention_tile_shared,attention_tile'
```

Build the WebAssembly first (`pnpm build`, with the Rust 1.95.0 that `rust-toolchain.toml` pins).
`dev/gpu-bench.html` now times its variants in turns and in batches of about 20 GFLOP: before,
identical settings could differ by 4x on this GPU.
