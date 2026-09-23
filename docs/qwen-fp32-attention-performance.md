# FP32 query-tiled Qwen attention

The FP32 causal grouped-query attention path uses an 8-token by 4-query-head
tile for one KV head and stages eight K/V rows in workgroup memory. It is
eligible only when the device does not expose `shader-f16` and has 32-lane
subgroups. Other cases retain the generic attention path; the existing FP16
path is unchanged.

## Chrome production measurements

Measured serially on an NVIDIA RTX 5070 Ti Blackwell (16 GB, driver 595.71.05),
Ryzen 9 9950X3D, Ubuntu, and Chrome 149.0.7827.200 using WebGPU timestamp
queries through the NVIDIA Vulkan ICD. The sequence was baseline, candidate,
baseline, candidate. Values are geometric means of per-case median summed GPU
kernel times; each reported condition geometrically combines its two complete
production runs. The production baseline is revision `ffe9dbe`; the candidate
is based on `d45755c` plus the measured Qwen-only changes. Initialization and
tuning are excluded.

| Model or aggregate | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Kev 0.8B | 15.094 ms | 14.971 ms | 0.81% faster |
| SemIf Qwen3.5 2B | 54.326 ms | 53.126 ms | 2.21% faster |
| Affected-model geometric mean | 28.635 ms | 28.202 ms | 1.51% faster |
| Laya control | 11.064 ms | 10.889 ms | 1.59% faster |
| Gemma 4 E2B control | 34.570 ms | 34.615 ms | 0.13% slower |
| Four-family geometric mean | 23.665 ms | 23.399 ms | 1.12% faster |

The measured runtime changes apply to the two Qwen model paths above. Laya and
Gemma are controls whose model source paths were unchanged, so their timing
differences are not attributed to this kernel. The small improvements on the
affected models are consistent across both production repeats, but do not imply
a speedup for every sequence length.

## Attention microbenchmark

The paired Chrome timestamp-query microbenchmark used three warmups and eleven
samples per case. It compared the generic `kev_attention` kernel with the FP32
tile and checked both against the CPU reference at the unchanged absolute and
relative tolerances of `0.0002`.

| Case | Tokens | Generic | Tiled | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Empty dispatch | 0 | 0.000020 ms | 0.000020 ms | 1.00× |
| Fresh boundaries | 1152 | 0.735584 ms | 0.196224 ms | 3.75× |
| Cached extensions | 8 | 0.068683 ms | 0.091339 ms | 0.75× |
| Cached tile boundaries | 6 | 0.013156 ms | 0.010218 ms | 1.29× |
| Fresh tile boundaries | 73 | 0.015060 ms | 0.010210 ms | 1.48× |
| Actual short | 31 | 0.011452 ms | 0.013686 ms | 0.84× |
| Actual | 124 | 0.060448 ms | 0.050688 ms | 1.19× |
| Actual | 532 | 0.699392 ms | 0.273024 ms | 2.56× |
| Actual | 105 | 0.045344 ms | 0.042528 ms | 1.07× |
| Actual | 198 | 0.119040 ms | 0.077280 ms | 1.54× |
| Actual | 606 | 0.917088 ms | 0.319200 ms | 2.87× |
| Mixed GQA cache | 3 | 0.016284 ms | 0.013080 ms | 1.24× |

Across the twelve plans, the geometric mean of case medians was 0.074470 ms
for the generic kernel and 0.049978 ms for the tile. Short cached inputs can
regress: the 8-token extension and 31-token cases were slower. The tile is most
useful on the longer actual-context cases, rather than a universal replacement
for generic attention. All microbenchmark numerical checks passed; the largest
paired output difference was `5.960464477539063e-8`, and the largest tiled
CPU-reference absolute difference was `7.275386237903803e-8`.

## Model guards and cache checks

All nine Chrome model guards completed and matched 132 of 132 expected decisions. The
225 recorded non-Laya probability values had no argmax changes and a maximum
baseline-to-candidate absolute difference of `3.2782554626464844e-6`. Gemma's
60 probability values were identical. The probabilities are not bit-exact for
all models. `maxDp` below is the maximum difference from each model's expected
reference, not the baseline-to-candidate difference.

| Model | Decisions | Candidate maxDp / unchanged limit | Result |
| --- | ---: | ---: | --- |
| Kev 0.8B | 13/13 | 0.0096847415 / 0.011 | Pass |
| Kev 4B | 13/13 | 0.0099622458 / 0.011 | Pass |
| Kev 9B | 5/5 | 0.0046327710 / 0.005 | Pass |
| SemIf Qwen3.5 0.8B | 12/12 | 0.0308191180 / 0.032 | Pass |
| SemIf Qwen3.5 2B | 12/12 | 0.0297582746 / 0.032 | Pass |
| SemIf Qwen3.5 4B | 12/12 | 0.0204846263 / 0.022 | Pass |
| Gemma 4 E2B | 12/12 | 0.0191190243 / 0.030 | Pass |
| Gemma 4 E4B | 12/12 | 0.0099277347 / 0.030 | Pass |
| Laya | 41/41 | 0.0240064442 / 0.024 | Existing strict-limit failure |

Laya's strict-limit failure is unchanged from baseline. Its saved guard contains
decisions and `maxDp` but no per-question probabilities, so it is not included in
the probability-delta aggregate. All six Qwen baseline and candidate guard runs
selected 16 recurrence lanes.

Production-source cache repeats confirmed the earlier probe results. Cache
hit-versus-miss probabilities matched exactly in all measured cases, and default
Kev also matched fresh inference exactly. Long Kev extension-versus-fresh
differences were `1.19e-7` at baseline and `2.16e-7` at candidate. SemIf 2B
differences were `2.80e-6` and `7.00e-7`, respectively; its warmup trace showed
two initial misses. Strict-zero comparisons for long Kev and SemIf were nonzero
in both versions. Attention has not been established as their cause; matmul
split or segmentation differences remain hypotheses. The earlier baseline
probes from the research worktree are preserved in the artifact as excluded
attempts, alongside the production-source repeats.

Firefox's four-family geometric mean was 31.137206 ms at baseline and
31.103778 ms at candidate, a 0.11% difference. The relevant runtime paths were
unchanged in Firefox, so no whole-model Firefox gain is claimed.

## Validation and reproduction data

The candidate passed 64 Rust tests, 103 JavaScript tests, 77 JavaScript syntax
checks, all three WASM validations, site staging, and independent source review.
CI and Pages verification after PR creation are not included in these results.
The [full benchmark artifact](benchmarks/qwen-fp32-attention-linux-2026-09-23.json)
contains both production repeats, raw microbenchmark samples and outputs, all
model guard artifacts, cache diagnostics, and Firefox profiles. Earlier
experimental Kev tile trials and unshipped paired-projection fusion are excluded
from the production comparison.
