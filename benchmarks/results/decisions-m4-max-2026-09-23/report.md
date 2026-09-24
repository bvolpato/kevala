# Decision benchmark report

Generated: 2026-09-24T20:24:01.301Z

Full passes: 3; requested decisions per model: 2592.

These are synthetic fixture scores, not general model accuracy. Rotations, perturbations, and repeated passes are correlated; confidence intervals resample the same 72 source groups. The combined accuracy below weights suites by their row counts; use the separate suite scores for model comparison. Mean latency pools individual awaited calls, not kernel timings.

## Model results

| Model | Status | Rows | Accuracy | Coverage | Invalid | Mean ms | p50 ms | p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| laya | ok | 2592 | 0.6655 | 1.0000 | 0 | 23.82 | 18.50 | 48.80 |
| kev-0.8b | ok | 2592 | 0.7326 | 1.0000 | 0 | 30.39 | 26.20 | 62.20 |
| kev-4b | ok | 2592 | 0.8993 | 1.0000 | 0 | 183.70 | 154.10 | 324.10 |
| kev-9b | ok | 2592 | 0.9410 | 1.0000 | 0 | 360.89 | 315.50 | 669.90 |
| semif-qwen3.5-0.8b | ok | 2592 | 0.4861 | 1.0000 | 0 | 56.08 | 53.70 | 83.20 |
| semif-qwen3.5-2b | ok | 2592 | 0.6308 | 1.0000 | 0 | 106.59 | 107.70 | 148.00 |
| semif-qwen3.5-4b | ok | 2592 | 0.8252 | 1.0000 | 0 | 362.95 | 338.40 | 575.00 |
| gemma-4-e2b | ok | 2592 | 0.7766 | 1.0000 | 0 | 61.02 | 57.40 | 88.60 |
| gemma-4-e4b | ok | 2592 | 0.9086 | 1.0000 | 0 | 250.96 | 194.90 | 524.20 |

## Per-pass results

Repeated passes measure execution variability, not additional independent tasks. Option-order flips are computed within each pass. Dates are UTC.

| Model | Pass | Date | Kevala accuracy | SemIf authored accuracy | SemIf perturbation accuracy | Mean ms | p95 ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| laya | 1 | 2026-09-23 | 75.0% | 62.3% | 69.4% | 19.18 | 34.80 |
| laya | 2 | 2026-09-24 | 75.0% | 62.3% | 69.4% | 21.95 | 38.30 |
| laya | 3 | 2026-09-24 | 75.0% | 62.3% | 69.4% | 30.33 | 62.80 |
| kev-0.8b | 1 | 2026-09-23 | 89.8% | 71.1% | 70.7% | 25.80 | 44.40 |
| kev-0.8b | 2 | 2026-09-24 | 89.8% | 71.1% | 70.7% | 32.68 | 54.60 |
| kev-0.8b | 3 | 2026-09-24 | 89.8% | 71.1% | 70.7% | 32.68 | 71.10 |
| kev-4b | 1 | 2026-09-23 | 91.7% | 89.4% | 90.1% | 258.90 | 414.70 |
| kev-4b | 2 | 2026-09-24 | 91.7% | 89.4% | 90.1% | 144.60 | 239.00 |
| kev-4b | 3 | 2026-09-24 | 91.7% | 89.4% | 90.1% | 147.61 | 259.90 |
| kev-9b | 1 | 2026-09-23 | 96.3% | 91.9% | 96.3% | 365.91 | 654.70 |
| kev-9b | 2 | 2026-09-24 | 96.3% | 91.9% | 96.3% | 372.65 | 695.60 |
| kev-9b | 3 | 2026-09-24 | 96.3% | 91.9% | 96.3% | 344.11 | 660.50 |
| semif-qwen3.5-0.8b | 1 | 2026-09-23 | 60.2% | 50.9% | 41.7% | 56.25 | 82.70 |
| semif-qwen3.5-0.8b | 2 | 2026-09-24 | 60.2% | 50.9% | 41.7% | 53.64 | 75.90 |
| semif-qwen3.5-0.8b | 3 | 2026-09-24 | 60.2% | 50.9% | 41.7% | 58.34 | 88.80 |
| semif-qwen3.5-2b | 1 | 2026-09-23 | 85.2% | 63.4% | 55.2% | 105.81 | 140.60 |
| semif-qwen3.5-2b | 2 | 2026-09-24 | 85.2% | 63.4% | 55.2% | 104.06 | 152.80 |
| semif-qwen3.5-2b | 3 | 2026-09-24 | 85.2% | 63.4% | 55.2% | 109.90 | 150.40 |
| semif-qwen3.5-4b | 1 | 2026-09-23 | 100.0% | 84.0% | 74.7% | 353.43 | 565.60 |
| semif-qwen3.5-4b | 2 | 2026-09-24 | 100.0% | 84.0% | 74.7% | 353.94 | 549.30 |
| semif-qwen3.5-4b | 3 | 2026-09-24 | 100.0% | 84.0% | 74.7% | 381.49 | 586.80 |
| gemma-4-e2b | 1 | 2026-09-23 | 94.4% | 78.2% | 71.3% | 60.16 | 88.30 |
| gemma-4-e2b | 2 | 2026-09-24 | 94.4% | 78.2% | 71.3% | 60.44 | 90.50 |
| gemma-4-e2b | 3 | 2026-09-24 | 94.4% | 78.2% | 71.3% | 62.46 | 87.50 |
| gemma-4-e4b | 1 | 2026-09-23 | 100.0% | 89.1% | 90.1% | 257.52 | 559.70 |
| gemma-4-e4b | 2 | 2026-09-24 | 100.0% | 89.1% | 90.1% | 237.02 | 489.20 |
| gemma-4-e4b | 3 | 2026-09-24 | 100.0% | 89.1% | 90.1% | 258.34 | 541.20 |

## Per-dataset results

| Model | Dataset | Rows | Accuracy | Coverage | Invalid | CI 95% | Flips |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| laya | kevala-authored36 | 324 | 0.7500 | 1.0000 | 0 | 0.6204–0.8611 | 0.2778 |
| laya | semif-authored144 | 1296 | 0.6227 | 1.0000 | 0 | 0.5532–0.6829 | 0.1528 |
| laya | semif-perturbations108 | 972 | 0.6944 | 1.0000 | 0 | 0.5741–0.8056 | 0.1574 |
| kev-0.8b | kevala-authored36 | 324 | 0.8981 | 1.0000 | 0 | 0.7963–0.9722 | 0.0556 |
| kev-0.8b | semif-authored144 | 1296 | 0.7106 | 1.0000 | 0 | 0.6458–0.7731 | 0.1319 |
| kev-0.8b | semif-perturbations108 | 972 | 0.7068 | 1.0000 | 0 | 0.5741–0.8395 | 0.1019 |
| kev-4b | kevala-authored36 | 324 | 0.9167 | 1.0000 | 0 | 0.8241–0.9815 | 0.0556 |
| kev-4b | semif-authored144 | 1296 | 0.8935 | 1.0000 | 0 | 0.8588–0.9306 | 0.0417 |
| kev-4b | semif-perturbations108 | 972 | 0.9012 | 1.0000 | 0 | 0.8117–0.9877 | 0.0648 |
| kev-9b | kevala-authored36 | 324 | 0.9630 | 1.0000 | 0 | 0.8981–1.0000 | 0.0278 |
| kev-9b | semif-authored144 | 1296 | 0.9190 | 1.0000 | 0 | 0.8819–0.9560 | 0.0208 |
| kev-9b | semif-perturbations108 | 972 | 0.9630 | 1.0000 | 0 | 0.9074–1.0000 | 0.0000 |
| semif-qwen3.5-0.8b | kevala-authored36 | 324 | 0.6019 | 1.0000 | 0 | 0.4630–0.7315 | 0.2500 |
| semif-qwen3.5-0.8b | semif-authored144 | 1296 | 0.5093 | 1.0000 | 0 | 0.4722–0.5417 | 0.4861 |
| semif-qwen3.5-0.8b | semif-perturbations108 | 972 | 0.4167 | 1.0000 | 0 | 0.3025–0.5432 | 0.3796 |
| semif-qwen3.5-2b | kevala-authored36 | 324 | 0.8519 | 1.0000 | 0 | 0.7778–0.9167 | 0.3611 |
| semif-qwen3.5-2b | semif-authored144 | 1296 | 0.6343 | 1.0000 | 0 | 0.5741–0.6944 | 0.6458 |
| semif-qwen3.5-2b | semif-perturbations108 | 972 | 0.5525 | 1.0000 | 0 | 0.4630–0.6389 | 0.7778 |
| semif-qwen3.5-4b | kevala-authored36 | 324 | 1.0000 | 1.0000 | 0 | 1.0000–1.0000 | 0.0000 |
| semif-qwen3.5-4b | semif-authored144 | 1296 | 0.8403 | 1.0000 | 0 | 0.7917–0.8912 | 0.3125 |
| semif-qwen3.5-4b | semif-perturbations108 | 972 | 0.7469 | 1.0000 | 0 | 0.6543–0.8364 | 0.3704 |
| gemma-4-e2b | kevala-authored36 | 324 | 0.9444 | 1.0000 | 0 | 0.8889–0.9907 | 0.1389 |
| gemma-4-e2b | semif-authored144 | 1296 | 0.7824 | 1.0000 | 0 | 0.7176–0.8495 | 0.2639 |
| gemma-4-e2b | semif-perturbations108 | 972 | 0.7130 | 1.0000 | 0 | 0.6173–0.8086 | 0.4167 |
| gemma-4-e4b | kevala-authored36 | 324 | 1.0000 | 1.0000 | 0 | 1.0000–1.0000 | 0.0000 |
| gemma-4-e4b | semif-authored144 | 1296 | 0.8912 | 1.0000 | 0 | 0.8495–0.9306 | 0.1181 |
| gemma-4-e4b | semif-perturbations108 | 972 | 0.9012 | 1.0000 | 0 | 0.8302–0.9660 | 0.1296 |

## Per-family results

| Model | Dataset | Family | Rows | Accuracy | Coverage | Invalid |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| laya | kevala-authored36 | candidate_selection | 108 | 0.7778 | 1.0000 | 0 |
| laya | kevala-authored36 | evidence_interpretation | 108 | 0.7500 | 1.0000 | 0 |
| laya | kevala-authored36 | rule_application | 108 | 0.7222 | 1.0000 | 0 |
| laya | semif-authored144 | candidate_selection | 432 | 0.5625 | 1.0000 | 0 |
| laya | semif-authored144 | evidence_interpretation | 432 | 0.6806 | 1.0000 | 0 |
| laya | semif-authored144 | rule_application | 432 | 0.6250 | 1.0000 | 0 |
| laya | semif-perturbations108 | candidate_selection | 324 | 0.6296 | 1.0000 | 0 |
| laya | semif-perturbations108 | evidence_interpretation | 324 | 0.6944 | 1.0000 | 0 |
| laya | semif-perturbations108 | rule_application | 324 | 0.7593 | 1.0000 | 0 |
| kev-0.8b | kevala-authored36 | candidate_selection | 108 | 1.0000 | 1.0000 | 0 |
| kev-0.8b | kevala-authored36 | evidence_interpretation | 108 | 0.8611 | 1.0000 | 0 |
| kev-0.8b | kevala-authored36 | rule_application | 108 | 0.8333 | 1.0000 | 0 |
| kev-0.8b | semif-authored144 | candidate_selection | 432 | 0.7986 | 1.0000 | 0 |
| kev-0.8b | semif-authored144 | evidence_interpretation | 432 | 0.6597 | 1.0000 | 0 |
| kev-0.8b | semif-authored144 | rule_application | 432 | 0.6736 | 1.0000 | 0 |
| kev-0.8b | semif-perturbations108 | candidate_selection | 324 | 0.7500 | 1.0000 | 0 |
| kev-0.8b | semif-perturbations108 | evidence_interpretation | 324 | 0.6852 | 1.0000 | 0 |
| kev-0.8b | semif-perturbations108 | rule_application | 324 | 0.6852 | 1.0000 | 0 |
| kev-4b | kevala-authored36 | candidate_selection | 108 | 1.0000 | 1.0000 | 0 |
| kev-4b | kevala-authored36 | evidence_interpretation | 108 | 0.8889 | 1.0000 | 0 |
| kev-4b | kevala-authored36 | rule_application | 108 | 0.8611 | 1.0000 | 0 |
| kev-4b | semif-authored144 | candidate_selection | 432 | 0.8958 | 1.0000 | 0 |
| kev-4b | semif-authored144 | evidence_interpretation | 432 | 0.8958 | 1.0000 | 0 |
| kev-4b | semif-authored144 | rule_application | 432 | 0.8889 | 1.0000 | 0 |
| kev-4b | semif-perturbations108 | candidate_selection | 324 | 1.0000 | 1.0000 | 0 |
| kev-4b | semif-perturbations108 | evidence_interpretation | 324 | 0.9167 | 1.0000 | 0 |
| kev-4b | semif-perturbations108 | rule_application | 324 | 0.7870 | 1.0000 | 0 |
| kev-9b | kevala-authored36 | candidate_selection | 108 | 1.0000 | 1.0000 | 0 |
| kev-9b | kevala-authored36 | evidence_interpretation | 108 | 0.9167 | 1.0000 | 0 |
| kev-9b | kevala-authored36 | rule_application | 108 | 0.9722 | 1.0000 | 0 |
| kev-9b | semif-authored144 | candidate_selection | 432 | 0.9375 | 1.0000 | 0 |
| kev-9b | semif-authored144 | evidence_interpretation | 432 | 0.9653 | 1.0000 | 0 |
| kev-9b | semif-authored144 | rule_application | 432 | 0.8542 | 1.0000 | 0 |
| kev-9b | semif-perturbations108 | candidate_selection | 324 | 1.0000 | 1.0000 | 0 |
| kev-9b | semif-perturbations108 | evidence_interpretation | 324 | 1.0000 | 1.0000 | 0 |
| kev-9b | semif-perturbations108 | rule_application | 324 | 0.8889 | 1.0000 | 0 |
| semif-qwen3.5-0.8b | kevala-authored36 | candidate_selection | 108 | 0.7500 | 1.0000 | 0 |
| semif-qwen3.5-0.8b | kevala-authored36 | evidence_interpretation | 108 | 0.3889 | 1.0000 | 0 |
| semif-qwen3.5-0.8b | kevala-authored36 | rule_application | 108 | 0.6667 | 1.0000 | 0 |
| semif-qwen3.5-0.8b | semif-authored144 | candidate_selection | 432 | 0.5139 | 1.0000 | 0 |
| semif-qwen3.5-0.8b | semif-authored144 | evidence_interpretation | 432 | 0.4861 | 1.0000 | 0 |
| semif-qwen3.5-0.8b | semif-authored144 | rule_application | 432 | 0.5278 | 1.0000 | 0 |
| semif-qwen3.5-0.8b | semif-perturbations108 | candidate_selection | 324 | 0.6296 | 1.0000 | 0 |
| semif-qwen3.5-0.8b | semif-perturbations108 | evidence_interpretation | 324 | 0.2778 | 1.0000 | 0 |
| semif-qwen3.5-0.8b | semif-perturbations108 | rule_application | 324 | 0.3426 | 1.0000 | 0 |
| semif-qwen3.5-2b | kevala-authored36 | candidate_selection | 108 | 0.9722 | 1.0000 | 0 |
| semif-qwen3.5-2b | kevala-authored36 | evidence_interpretation | 108 | 0.7778 | 1.0000 | 0 |
| semif-qwen3.5-2b | kevala-authored36 | rule_application | 108 | 0.8056 | 1.0000 | 0 |
| semif-qwen3.5-2b | semif-authored144 | candidate_selection | 432 | 0.4931 | 1.0000 | 0 |
| semif-qwen3.5-2b | semif-authored144 | evidence_interpretation | 432 | 0.7500 | 1.0000 | 0 |
| semif-qwen3.5-2b | semif-authored144 | rule_application | 432 | 0.6597 | 1.0000 | 0 |
| semif-qwen3.5-2b | semif-perturbations108 | candidate_selection | 324 | 0.5278 | 1.0000 | 0 |
| semif-qwen3.5-2b | semif-perturbations108 | evidence_interpretation | 324 | 0.6574 | 1.0000 | 0 |
| semif-qwen3.5-2b | semif-perturbations108 | rule_application | 324 | 0.4722 | 1.0000 | 0 |
| semif-qwen3.5-4b | kevala-authored36 | candidate_selection | 108 | 1.0000 | 1.0000 | 0 |
| semif-qwen3.5-4b | kevala-authored36 | evidence_interpretation | 108 | 1.0000 | 1.0000 | 0 |
| semif-qwen3.5-4b | kevala-authored36 | rule_application | 108 | 1.0000 | 1.0000 | 0 |
| semif-qwen3.5-4b | semif-authored144 | candidate_selection | 432 | 0.7153 | 1.0000 | 0 |
| semif-qwen3.5-4b | semif-authored144 | evidence_interpretation | 432 | 0.9306 | 1.0000 | 0 |
| semif-qwen3.5-4b | semif-authored144 | rule_application | 432 | 0.8750 | 1.0000 | 0 |
| semif-qwen3.5-4b | semif-perturbations108 | candidate_selection | 324 | 0.7685 | 1.0000 | 0 |
| semif-qwen3.5-4b | semif-perturbations108 | evidence_interpretation | 324 | 0.9167 | 1.0000 | 0 |
| semif-qwen3.5-4b | semif-perturbations108 | rule_application | 324 | 0.5556 | 1.0000 | 0 |
| gemma-4-e2b | kevala-authored36 | candidate_selection | 108 | 1.0000 | 1.0000 | 0 |
| gemma-4-e2b | kevala-authored36 | evidence_interpretation | 108 | 0.8333 | 1.0000 | 0 |
| gemma-4-e2b | kevala-authored36 | rule_application | 108 | 1.0000 | 1.0000 | 0 |
| gemma-4-e2b | semif-authored144 | candidate_selection | 432 | 0.9097 | 1.0000 | 0 |
| gemma-4-e2b | semif-authored144 | evidence_interpretation | 432 | 0.8125 | 1.0000 | 0 |
| gemma-4-e2b | semif-authored144 | rule_application | 432 | 0.6250 | 1.0000 | 0 |
| gemma-4-e2b | semif-perturbations108 | candidate_selection | 324 | 0.8519 | 1.0000 | 0 |
| gemma-4-e2b | semif-perturbations108 | evidence_interpretation | 324 | 0.7963 | 1.0000 | 0 |
| gemma-4-e2b | semif-perturbations108 | rule_application | 324 | 0.4907 | 1.0000 | 0 |
| gemma-4-e4b | kevala-authored36 | candidate_selection | 108 | 1.0000 | 1.0000 | 0 |
| gemma-4-e4b | kevala-authored36 | evidence_interpretation | 108 | 1.0000 | 1.0000 | 0 |
| gemma-4-e4b | kevala-authored36 | rule_application | 108 | 1.0000 | 1.0000 | 0 |
| gemma-4-e4b | semif-authored144 | candidate_selection | 432 | 0.8819 | 1.0000 | 0 |
| gemma-4-e4b | semif-authored144 | evidence_interpretation | 432 | 0.9236 | 1.0000 | 0 |
| gemma-4-e4b | semif-authored144 | rule_application | 432 | 0.8681 | 1.0000 | 0 |
| gemma-4-e4b | semif-perturbations108 | candidate_selection | 324 | 0.9167 | 1.0000 | 0 |
| gemma-4-e4b | semif-perturbations108 | evidence_interpretation | 324 | 0.9907 | 1.0000 | 0 |
| gemma-4-e4b | semif-perturbations108 | rule_application | 324 | 0.7963 | 1.0000 | 0 |

## Missing inputs

- None

## Gemma Q8 versus BF16 references

| Model | Reference | Cases | Agreement | Max abs diff | Mean abs diff | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
