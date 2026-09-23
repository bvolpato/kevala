# Explicit FP32 matrix output stores

The shared FP32 matrix kernel now writes its named accumulator components directly.
This replaces dynamically indexed column and row loops in the output stage. The
ordered dot products, K splits, bias addition, residual addition, and ReLU remain
unchanged. FP16 and two-column-group shaders are unchanged.

The change applies to matrix operations across Laya, Kev, SemIf, and Gemma. It adds
no precision mode, model-specific rule, tuning step, or weight format.

## Measurements

Measured serially on an NVIDIA RTX 5070 Ti, driver 595.71.05, Ubuntu, Chrome
149.0.7827.200, using hardware WebGPU and precise timestamps in a dedicated headless
browser. The baseline is main at `9ed1f1e4`, including the row-padding selection.
Both production sources exclude the experimental paired-projection fusion.

Each value is the geometric mean of median GPU time across three request sizes,
with two warmups and five measured requests per size. These are profiled GPU times,
not unprofiled round-trip latency. Full samples, profiles, outputs, and hashes are
in the [benchmark artifact](benchmarks/fp32-explicit-stores-linux-2026-09-23.json).

| Model | Before | After |
| --- | ---: | ---: |
| Laya | 11.322 ms | 11.168 ms |
| Kev 0.8B | 16.174 ms | 15.892 ms |
| SemIf 2B | 56.315 ms | 55.194 ms |
| Gemma E2B | 36.428 ms | 36.175 ms |
| Four-family geometric mean | 24.757 ms | 24.399 ms |

The measured aggregate improvement is 1.45%. This is a small optimization, and
individual model differences remain subject to timing variation. Two earlier
four-family runs on the experimental branch also improved, by 0.96% and 1.74%.
Those runs are excluded from this table because that branch contains unshipped fusion.

Seven representative matrix shapes are 1.2–8.5% faster in the paired production
check. Earlier paired runs in both orders measured 1.3–9.1% improvements. A tiny
synthetic tail case varied from 4.4% slower to 1.0% faster, so no universal speedup
is claimed for every shape.

Firefox 152.0.3 completes the four-family regression suite at 31.548 ms versus
31.387 ms before. It uses unchanged FP16 shaders; this is not a Firefox speedup
claim. Measurements cover one GPU.

## Correctness and validation

- All 21 paired matrix arrays have zero maximum absolute difference. All 86
  CPU-reference checks pass, including R1–R4 boundaries, ragged columns, bias,
  residual addition, and ReLU.
- All nine weights preserve 132/132 reference decisions at unchanged limits.
- All 225 recorded Kev, SemIf, and Gemma probabilities exactly match the baseline
  when recurrence selection matches. Initial automatic choices differed for Kev 4B
  and SemIf 4B, producing differences below 0.000003. Those runs also pass their
  unchanged guards. The artifact retains all attempts and selection diagnostics.
- Chrome Laya retains its existing maximum probability difference of
  `0.024006444195624588`, above its unchanged `0.024` threshold. That strict check
  is not green. Its runner does not record individual probabilities.
- The Kev cache hit and extension check has zero probability error.
- 64 Rust tests, 103 JavaScript tests, 75 JavaScript syntax checks, three WASM
  validations, and site staging pass. Independent review found no correctness issue.

The output ownership and arithmetic are unchanged, so existing numerical tests
cover the contract. Generated WASM files remain build artifacts and are not tracked.
