# Native AVX-512 performance

The CPUID-gated candidate passed audited executable gates and an independent model-data audit. For four requests, the median paired-ratio estimates imply 2.79% to 9.03% lower time. Every experimental paired range includes 1. The independent median for the short Kev-0.8B case is 1.13% slower for the candidate. These results come from one busy host. They do not establish a general speed gain or a model ranking.

## Production-candidate comparison

For each of eight quartets, the ratio is the baseline arm median divided by the candidate arm median. The table reports the median of those eight paired ratios. A value above 1 favors the candidate. The implied paired time reduction is `1 - 1 / ratio`. Each range is the observed minimum to maximum of eight ratios, not a confidence interval.

| Model and request | Median paired ratio | Implied paired time reduction | Quartets favoring candidate | Paired ratio range |
| --- | ---: | ---: | ---: | ---: |
| Laya Q8, short | 1.05306 | 5.04% | 6 of 8 | 0.94956 to 1.07816 |
| Laya Q8, long | 1.09921 | 9.03% | 7 of 8 | 0.94572 to 1.18327 |
| Kev-0.8B Q8, short | 1.02869 | 2.79% | 7 of 8 | 0.83168 to 1.11108 |
| Kev-0.8B Q8, long | 1.05412 | 5.13% | 7 of 8 | 0.95168 to 1.08999 |

The independent arm medians use all 16 recorded timings per arm. Parentheses show the observed minimum and maximum in milliseconds. These arm summaries are separate from the paired ratios.

| Model and request | Baseline median (observed range) | Candidate median (observed range) |
| --- | ---: | ---: |
| Laya Q8, short | 397.80 (332.9 to 545.3) | 390.05 (320.3 to 482.2) |
| Laya Q8, long | 1,770.15 (1,413.2 to 1,999.1) | 1,607.85 (1,394.1 to 1,803.0) |
| Kev-0.8B Q8, short | 390.15 (323.6 to 543.0) | 394.55 (321.8 to 485.4) |
| Kev-0.8B Q8, long | 2,082.60 (1,744.0 to 2,400.2) | 1,962.45 (1,667.6 to 2,491.4) |

The independent Kev-0.8B short-input median increased from 390.15 ms to 394.55 ms. This is 1.13% slower, although its paired-ratio estimate implies a 2.79% reduction. Report both measures because their summaries differ.

The identical-build A/A control also varied. Each value is the median of eight paired ratios, followed by the observed range.

| Model and request | A/A median paired ratio | A/A paired ratio range |
| --- | ---: | ---: |
| Laya Q8, short | 1.02179 | 0.94543 to 1.09498 |
| Laya Q8, long | 0.99281 | 0.90748 to 1.13782 |
| Kev-0.8B Q8, short | 0.97561 | 0.91840 to 1.28611 |
| Kev-0.8B Q8, long | 1.03383 | 0.87160 to 1.11442 |

All four experimental ranges and all four A/A ranges include 1. The capture ran on a busy host. The audit applied no timing correction and excluded no outliers. The measurements do not show that every request is faster.

The production audit covered 306 command records, 622 lease proofs, 304 exact responses, and 1,520 numeric values. The metric was `Model::decide` after model load and request parsing, before response serialization. The four fixtures had 35 or 629 characters. The run used logical CPU 7 on a Ryzen 9 9950X3D. Printed timing resolution was 0.1 ms. Model load and total request time were separate observations. These are native CLI CPU timings, not browser, GPU, or WebAssembly latency.

## Production source and executable gate

The production candidate source is commit `f4596f1d239ca84eb18255819f56684a5c5cf086`, based on `dae1c95`. Its kernel SHA-256 is `a5c68612e41dcdeccf5b3b3568480bd2b91e3651ea1ba673c0fcc18a35167c6c`; its build script SHA-256 is `eb358238d4dde5adee39d69928b6d04521abfd8cd7a54d456e6227e78b9d3b32`. The candidate CLI SHA-256 is `85f210f659dae97fcbd0bedb856d7e894e6b3cbd37aba69fca3d91ab2d5b5ece`. The baseline kernel came from `dae1c95` and has SHA-256 `a1d101ed08726cafb2d7a301a39edf789adfc203044852fd8c7b4cf7cf12b5da`. The baseline Rust tree was byte-equivalent from `8e4ae3a` through `dae1c95` and was reverified before the run. The baseline CLI SHA-256 is `464cd74a3188a93dd7c630b3afc46b1533b5365a649aa1cb1201c277564ae775`; it is the historical baseline executable, not a new build from `dae1c95`. The compiled binaries are not included in the evidence package.

The production gate passed its audited compiler, code-generation, source, parity, and CPU-guard checks. The tested host was AuthenticAMD family 26, model 68, stepping 0, with AVX512F and runtime operating-system state support. See the [dispatch design](native-avx512-dispatch.md) for the CPU guard and compiler/platform limits. The model-run receipt records `actualDefault: 2` and `productionPromotion: false`. That boolean means the measurement runner did not perform a promotion action. Browser and WebAssembly delivery were outside this validation scope.

## Earlier private prototype snapshot

The earlier private prototype run is separate from the production-candidate comparison. It used a different candidate source and has its own audited model receipt.

| Model and request | Median paired ratio | Implied paired time reduction | Candidate faster quartets | Paired ratio range |
| --- | ---: | ---: | ---: | ---: |
| Laya Q8, short | 1.02596 | 2.53% | 7 of 8 | 0.93046 to 1.04912 |
| Laya Q8, long | 1.03519 | 3.40% | 7 of 8 | 0.99343 to 1.09106 |
| Kev-0.8B Q8, short | 1.02605 | 2.54% | 6 of 8 | 0.97974 to 1.12623 |
| Kev-0.8B Q8, long | 1.05282 | 5.02% | 8 of 8 | 1.01169 to 1.16510 |

The A/A paired-ratio medians were 0.99853, 1.01073, 0.99644, and 1.00731. Three experimental ranges included 1. The run recorded 304 exact responses and audited 1,520 numeric values. It measured the same post-load CLI method on logical CPU 7 of a busy host, with 0.1 ms printed resolution and no correction or outlier exclusion.

The prototype baseline was commit `8e4ae3a3ff6592e9581696057f07fa90e036e10a` with kernel SHA-256 `a1d101ed08726cafb2d7a301a39edf789adfc203044852fd8c7b4cf7cf12b5da`. Its private candidate used kernel SHA-256 `d43c91d9bc1b90f05d6ec870f90c8b11c5ca5adc8d00bf598eb4145fc2958d0e` and build script SHA-256 `443856de805d5d967c2f730cd4f5149b84523adff4181e8868efef999f1f9734`.

A separate nine-frame prototype gate passed its recorded checks. Its code-generation summary counted a 20-instruction ZMM inner loop. That count excludes helper setup, panic handling, and `vzeroupper`. The later production gate is separately pinned below.

## Evidence identities

- Production gate receipt, inventory, and release proof: `a3992a331088756d65341b92cbc23ab18c053325601d6f843a39cb6bdca15990`, `313d37d403330964851671fb00d11e8b2e82d6373f0cb2967f79066951b50ad2`, and `9beb847e3e95caad1395e3364444211d49d874c8bea0395a6d523fbc61587a20`.
- Production model receipt, inventory, and release proof: `17df22ea0a0eaff88b33b04413103440ec93f60d9856aaffa6a66dd671f1fa0a`, `2a5280aed47f4bdf4489771ffecaf3a8ecd1ead9f10f5b684cd5c9d5c5fa995a`, and `6eef79bccb5c23d880b9396711f6e385f872c8845a540f49bdd5d9568e91d0bb`.
- Earlier prototype model receipt, inventory, and release proof: `bc4d5c34cf6bbd63ff8548e721254d065b17eacd3576b7baac9af84b1f1b7186`, `04b787a1561d24b9e72f1b64a3318c1a21043d7faa2dee23436f00102eb5018d`, and `69f3d68d8688ce593f7d1537584af97518dba946cf66f6d740c2322deeca83ce`.
- Earlier prototype gate receipt and release proof: `a76d7ca545d5d6426bda8c841d141b2dc578193f07e2d79e24df7af3fdef7af5` and `5501f437833bca5a1b8c7102dc010f127093ed7cd808bf0571089056d8eb5218`.
- The incomplete earlier model-capture receipt remains `e2211a8f33ad7cb3ac26fee0ce11a1d12bf9808b1a85b88fafc8a844ea848297`. The original compile-budget failure receipt remains `c2de5dbb543cfd65c28b47799ad52fc41eebbb01d3b0dc0d6f19518db8372143`. Both remain labeled as failures.

The [evidence package README](benchmarks/native-avx512-2026-11-07/README.md) lists the exact archive coverage and review commands.
