# M4 Max kernel safety audit

September 24, 2026. Audit baseline: `main` at `6e1754e`. None of the three
numerical defects below was fixed in that upstream revision.

## Fixes

| Path | Reproduction | Repair |
|---|---|---|
| Gemma tanh GELU | `GELU(20) * 1` produced NaN on Metal and Gemma E2B failed during load warmup. | Return the rounded FP32 saturated tails outside `[-10, 10]`, before evaluating the cubic and tanh. |
| Qwen decay gates | With input `-20` and decay scale `-1e8`, the GPU returned `1`; the CPU reference is `0.8137391805648804`. | Compute softplus with a corrected `log1p` formula, preserving small negative-input values. Use a branch, not an eagerly evaluated `select`, above 20. |
| Qwen gated RMS | A finite negative-max-FP32 gate and a normalizable core row of 37 produced NaN instead of zero. | Evaluate SiLU with saturated FP32 tails before the gate multiplication. |

Qwen sigmoid now evaluates `exp(-abs(x))`, so its exponential argument is never
positive. Scalar and vector helpers are shared by the decay gates, convolution,
gated RMS, feed-forward SiLU, and both attention implementations. SiLU returns
`x` above 20 and zero below -120. At those boundaries the omitted tail correction
is below FP32 rounding or representability, respectively.

These are mathematical saturation limits, not clamping model activations to an
arbitrary range. The healthy matmul, recurrence, RoPE, and softmax implementations
were not rewritten. Odd linear key-head counts were investigated but are already
rejected by specialization validation; no convolution indexing change was needed.

The precise Metal/compiler stage responsible for the NaNs has not been isolated.
WGSL does not guarantee IEEE infinity propagation after runtime overflow; its
[floating-point evaluation rules](https://www.w3.org/TR/WGSL/#floating-point-evaluation)
permit indeterminate results. Avoiding invalid intermediates is safer than relying
on infinity later turning into zero.

## Validation

Hardware: Apple M4 Max, 40 GPU cores, 64 GiB unified memory. Chrome for Testing
151.0.7922.34, Metal. Every browser target was hidden and isolated inside the
already-running agent Chrome; no browser process was launched.

- All **35 registered kernels** compiled across **122 distinct rendered
  specializations**, including FP32, f16, subgroup/shared paths, matrix shapes and
  row tiles, BM56, and grouped Qwen dimensions.
- A minimum-feature/default-limit device compiled **87** specializations. **35**
  unavailable feature/resource combinations are explicit skips, not passes.
- The production activation guard checks **17,312 output values** against CPU
  references and checks output sentinels. An isolated WASM build from a clean git
  archive of `6e1754e` fails both Qwen regressions; the repaired build passes.
  Maximum normalized error is `6.32e-6`, within the unchanged guard tolerance.
- The Gemma GELU guard checks **348 gated values**, including maximum finite FP32
  inputs and zero/negative gates. Maximum absolute error is `4.77e-7`.
- Matrix tails and split-K epilogues, Laya attention boundaries, 12 Qwen attention
  plans, seven Gemma attention shapes, 45 Gemma RMS cases, and selected-row/shared-KV
  Gemma tails passed their existing CPU-reference guards.
- All **24 generic/wide matmul row/group/precision variants** passed eight shapes
  covering row/column tails and split-K, for **480 CPU-reference comparisons**.
  BM56 also passed 28 CPU comparisons and 14 bit-identical paired epilogues at
  token lengths 45, 56, 112, and 224.
- All 4/8/16-lane recurrence kernels passed shared-memory and subgroup paths,
  including actual model dimensions, cached parents, and token-boundary cases.
  Maximum CPU-reference error was `3.73e-9`.
- Laya parity matched **41/41** reference decisions; Kev-0.8B matched **13/13**.
  Maximum probability differences were `0.02270645` and `0.01013104`, below the
  existing `0.024` and `0.011` limits. Request-cache checks passed.
- **128 JavaScript tests**, **64 Rust tests**, syntax/format checks, all three
  rebuilt WASM flavors, static-site staging, and both published benchmark summary
  checks passed. The README's generated M4 table matched its raw results.
  Local command-line tools were Node 26.0.0 and Rust 1.95.0. These counts describe
  local runs; PR CI is reported separately.

### Fresh model replay

All **11 current catalog models**, including both Bruv sizes and both Gemma sizes,
completed load/warmup and **108 decisions each** on the 36-case Kevala suite in
three option orders. All **1,188 decisions** returned valid normalized probabilities
without inference errors. Kev-9B's first download failed with `network error` during
setup; its retry succeeded. That failed setup attempt is retained in the artifact.

This is a one-pass regression replay, not a replacement for the full three-suite,
three-pass README evaluation. Its weight identities were checked against loaded
headers/catalog revisions; catalog SHA-256 values are expectations, not fresh
full-download hash verification.

Comparing the nine previously measured models with the September 23 replay gives
**971/972 identical selected options**. The changed row is SemIf-2B's
`kev-candidate-01::perm:rotate2`: `north` and `no_match` previously had probabilities
`0.40805754` and `0.40648839`; the new build gives `0.40714258` and `0.40770054`.
Its Kevala result changes from 92/108 to 91/108. The other models retain the same
selected options. This near-boundary change is disclosed, not removed from scoring.
It is not evidence that either choice is more accurate on real-world data.

Compilation does not establish numerical correctness for every possible input.
The guards cover deterministic synthetic cases, not every model activation or GPU
driver. Firefox/Linux was not rerun in this audit. No speedup is claimed from these
safety fixes; numerical/compilation guard wall times are not inference latency.

The README remains the separately pinned September 23-24 (UTC) three-pass evaluation,
not a new performance claim for this changed build. Its dataset and reliability
limitations remain documented in [the protocol](../BENCHMARK.md).

## Reproduce without foreground windows

Build first, and serve this checkout on port 8123 using the existing server.
Attach to the shared Chrome; do not launch another browser:

```sh
RUSTUP_TOOLCHAIN=1.95.0 scripts/build-wasm.sh
KEVALA_BENCH_CDP=http://127.0.0.1:9333 \
  KEVALA_BENCH_URL=http://127.0.0.1:8123 \
  node scripts/gpu-guard.mjs tmp/gpu-guard
```

The guard derives its kernel inventory from the Rust registry and fails on
compilation errors, numerical mismatches, non-finite results, or overwritten
sentinels. The portable guard also runs Gemma attention/RMS/tail and shared-memory
recurrence checks. To exercise optional 32-lane subgroup paths on this GPU:

```sh
uv run scripts/bench-gpu.py --cdp http://127.0.0.1:9333 \
  --url 'http://127.0.0.1:8123/dev/kev-attn-bench.html?samples=1&warmups=1&gateTails=1'
for candidate in current recur8 recur16; do
  uv run scripts/bench-gpu.py --cdp http://127.0.0.1:9333 \
    --url "http://127.0.0.1:8123/dev/recur-check.html?candidate=$candidate&subgroups=1&actual=1&samples=1&warmups=1"
done
```

Raw before/after values, specialization inventories, tolerances, and numerical
results are in [the audit artifact](benchmarks/kernel-safety-metal-2026-09-24.json).
