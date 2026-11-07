# RTX 5070 Ti host profile

This profile records current Q8 model latency, the BF16 packs available under the local memory policy, and the registered WebGPU kernel inventory on an NVIDIA RTX 5070 Ti with driver 595.71.05 and an AMD Ryzen 9 9950X3D host. The browser was Chrome 156. The runtime was CI run `42350139676` for commit `4e27753e6f385d6fabbb6040fdd82f058a9c48a8`. The [summary](summary.json) feeds the live profile on the Speed page; raw captures and run evidence are linked from that summary.

Model requests used the resident, uncached protocol with three warmups, nine samples, two blocks, and the short, medium, and long single-question cases. The report lists unprofiled wall latency separately from GPU timestamp diagnostics. Loading, calibration, fixture hashing, and warmup requests are excluded. Kernel timestamp shares describe GPU kernel medians, not whole-request wall time.

## Coverage and findings

All 11 Q8 catalog models completed with fixture outputs within the declared same-runtime tolerance. Three local BF16 packs completed: Kev-0.8B, Kev-4B, and H2O-Lightning-4B.

The current Q8 wall medians, in milliseconds:

| Model | Short, ms (tokens) | Medium, ms (tokens) | Long, ms (tokens) |
| --- | ---: | ---: | ---: |
| Laya | 10.40 (47) | 18.55 (140) | 45.85 (512) |
| Kev-0.8B | 11.55 (31–32) | 24.10 (124–125) | 74.05 (532–533) |
| Kev-4B | 28.95 (31–32) | 82.15 (124–125) | 300.50 (532–533) |
| Kev-9B | 48.05 (31–32) | 145.55 (124–125) | 538.20 (532–533) |
| H2O-Lightning-4B | 55.60 (93–94) | 95.40 (185–186) | 289.05 (593–594) |
| Clef-flash | 289.40 (158–159) | 366.40 (251–252) | 805.90 (659–660) |
| SemIf-Qwen3.5-0.8B | 19.40 (105–106) | 30.15 (198–199) | 70.90 (606–607) |
| SemIf-Qwen3.5-2B | 32.55 (105–106) | 55.80 (198–199) | 140.75 (606–607) |
| SemIf-Qwen3.5-4B | 69.60 (105–106) | 123.20 (198–199) | 312.05 (606–607) |
| Gemma 4 E2B | 23.65 (99–100) | 31.35 (193–194) | 75.30 (601–602) |
| Gemma 4 E4B | 50.65 (99–100) | 81.80 (193–194) | 203.15 (601–602) |

The token ranges show actual counts across the nine samples. Laya has an exact count in each case. These are uncached requests from the measured run, not cold-start or model-load latency.

BF16 wall medians and actual token ranges:

| Model | Short, ms (tokens) | Medium, ms (tokens) | Long, ms (tokens) |
| --- | ---: | ---: | ---: |
| Kev-0.8B | 10.15 (31–32) | 22.65 (124–125) | 69.70 (532–533) |
| Kev-4B | 30.80 (31–32) | 83.15 (124–125) | 293.40 (532–533) |
| H2O-Lightning-4B | 55.10 (93–94) | 94.95 (185–186) | 294.40 (593–594) |

Kev-9B BF16 was blocked by the explicit 9 GB pack policy because its pack is 15,902,057,920 bytes. This was a policy block, not an observed allocation failure. Clef-flash BF16 was blocked because its local pack was unavailable. No model weights were downloaded. Q8 and BF16 were separate captures and do not provide a paired precision comparison.

BF16 rows use local pack conversions without a catalog-pinned source digest. BF16 identifies the stored pack format; these captures do not prove native BF16 arithmetic throughout the model. Measurements ran on a shared host, not an idle-host baseline.

The first kernel inventory recorded one `requestDevice` startup failure: `A valid external Instance reference no longer exists.` It occurred before the core-reference harness performed any math and maps to six failed kernel rows. A separate controlled browser run passed compilation and core-reference. The initial failure remains in the inventory; the retry is archived separately and does not rewrite its status.

These are measured host results, not proof of a universal fastest kernel or model ranking. Numerical checks establish agreement on these fixtures, not model accuracy. Some tuner summaries were truncated. Legacy adapter metadata can leave fallback status unknown when browser properties are absent, so the profile does not upgrade those older `false` values to confirmed non-fallback evidence.

## Gemma 4 Vision checks

Separate CI73 checks validated the actual E2B vision encoder pack against the pinned Transformers F32 reference at commit `2c4914fb939fe9de0d8e7a798af4684d552f18b4` using fixture SHA-256 `ba35047fb3ecb45b8d975a7b8d302ef3a6fe9772af5095dc3af8727c42d1da59`. The encoder processed 630 patches, of which 576 were active, and produced 64 output soft tokens. Five warmups preceded five unprofiled samples; the median was 24.6 ms, with maximum absolute error `2.0653e-5` below the `1e-4` reference limit. The five GPU timestamp samples are diagnostic only; their 14.834 ms median is not request latency.

A generated red/blue image check returned the expected color choices, verified prefix reuse with maximum probability deltas of `1.34e-14` and `1.63e-15` against a `5e-4` limit, and disposed the vision phase. Its two initial wall samples were 329.5 ms and 66.1 ms, so no stable image latency estimate is reported. A separate 152.9 ms profiled request disabled prefix reuse and is not paired with the wall samples.

The [published summary](summary.json) links separate encoder and image/prefix evidence artifacts. They retain the audited before/after receipts, exact result files, and their shared input helper source. The independent audit matched all 61 captured files, full decoder and vision pack hashes before and after, the CI73 runtime bytes, clean source identity, the canonical profile lock, and the 32/40/4 GiB memory scope. Both checks use the legacy fallback field that can leave adapter fallback unknown. They do not measure natural-image, face, or expression accuracy, or establish image support for the other models.

## Rebuild the published report

Keep the captured runtime bytes in `js/src`, the four text capture directories under `/tmp/kevala-profile-4e27753-20261107/tmp`, and both supplemental vision evidence directories under `tmp/sota-20261106-kdDTGn`. The build script uses the existing profile report validator and archiver. It keeps the successful core-reference retry and the vision checks as separate evidence artifacts instead of merging them into text latency rows.

```sh
node scripts/build-host-profile-2026-11-07.mjs
```

Set `KEVALA_PROFILE_CAPTURE_ROOT` to another directory containing the same four named text capture directories, and `KEVALA_PROFILE_CAMPAIGN_ROOT` to the copied campaign containing the two vision evidence directories. Set `KEVALA_PROFILE_PUBLICATION_ROOT` to a fresh directory for a relocated rebuild because supplemental evidence records absolute source paths. An existing evidence file is reused only when its bytes match exactly. The resulting summary is deliberately marked failed when the initial kernel capture records its startup error; the model results remain available alongside it.
