# GPU epilogue follow-up evidence, 2026-11-07

Both completed candidates passed their numerical checks and failed their predeclared performance policies on the RTX 5070 Ti and Ryzen 9950X3D host. Laya BN32 and Kev Q8 v6r2 remain private and disabled by default. These results do not establish a speed improvement or authorize production promotion.

This package follows the [first GPU epilogue publication](../gpu-epilogues-2026-11-07/README.md). It preserves a new Laya BN32 experiment and the first completed Kev Q8 v6 experiment. The separately retained Q8 v6r1 record is a failed launch. It is not a numerical or performance result and is distinct from the older Q8 v4 and v5 failures.

## Laya BN32

The candidate pairs the two Q8 projections with Laya's exact ERF GeGLU activation. It uses BM64, BN32, K32 tiles and 32 by 8 threads. The dispatch is 82 by 8 by 1, or 656 workgroups. Shared storage is 16 KiB, and the source declares 16 FP32 accumulator scalars per thread. The previous BN64 candidate used 328 workgroups, 24 KiB of shared storage and 32 accumulator scalars per thread. Restoring those resource counts did not improve the measured result.

The actual default GPU selector chose generic FP32 without `shader-f16`. Both arms used one loaded model, automatic relaxed WASM selection, coordinator tile 0, disabled caches, and unchanged tuning. The pinned request had T512, K1024 and I2624 across 28 encoder layers.

| Recorded metric | Result |
| --- | ---: |
| Median paired kernel gain, `(baseline - candidate) / baseline` | -15.0922731504% |
| Median paired complete-request gain | -7.3649008118% |
| Faster candidate quartets, kernel phase | 0 of 9 |
| Faster candidate quartets, request phase | 0 of 9 |
| ABBA request median gain | -10.6813316837% |
| BAAB request median gain | -6.3549627692% |
| Maximum absolute request A/A relative spread | 8.7955739067% |

The separate request and timestamp phases each retain three warmups per arm, nine A/A pairs and nine alternating ABBA/BAAB quartets. The timestamp comparison sums all 28 baseline `mm.wi` plus `geglu` pairs against all 28 candidate `mm.wi.geglu.private` dispatches. No samples were excluded or corrected. The candidate failed both acceptance gates.

Before and after timing, every layer compared exactly 1,343,488 active elements and 753,664 inactive-tail elements. All bit-mismatch, baseline-nonfinite, candidate-nonfinite and tail-mismatch counters were zero. The existing default-off golden guard retained its 0.03 tolerance. The independently recalculated maximum difference was 0.024, with finite vectors and matching argmax for every question.

The completed harness also enforced complete-response equality, exact-request permits, missing-permit and wrong-context fallback, and MEDIUM fallback. Individual measured response bodies were not retained. The archive therefore preserves evidence that the frozen harness gates passed, but it cannot support a later byte comparison of each full response.

Earlier CPU-only Q8 checks may have overlapped BN32 startup. Their exact start and end times are unavailable. A later lock-held Q8 source preflight occurred after BN32, but it does not resolve the earlier uncertainty. This result makes no idle-host or exclusive-CPU claim. The rejection is retained without a rescue rerun.

## Kev Q8 v6r2

The completed experiment used one Kev 9B Q8 model instance and compared the default path with a private fused SwiGLU path. The pinned request used 512 prefill tokens and 16 suffix tokens. The bundle retains the actual selector, WASM, device and tuning metadata. The request phase was unprofiled. It produced no separate GPU kernel timing result.

| Recorded metric | Result |
| --- | ---: |
| Median paired ratio, `baseline wall / candidate wall` | 0.9805480112706042 |
| Implied paired time reduction, `1 - 1 / median ratio` | -1.983787484734134% |
| Faster candidate pairs | 1 of 9 |
| Pairs with at least 5% improvement | 0 of 9 |
| Maximum A/A ratio deviation from 1 | 1.8346741546696244% |

The independently calculated arm medians were 466.455 ms for the baseline and 476.815 ms for the candidate. Those medians are a separate statistic from the median paired ratio. They must not replace or be mixed with the predeclared paired estimator.

The result retains the shape probe, verification request, three warmups per arm, nine A/A pairs, nine alternating baseline/candidate pairs, and both MEDIUM fallback requests. All 44 retained LONG output records are exactly equal, including their raw probability vectors. Both MEDIUM outputs are exactly equal and use the original path. Tuning snapshots are unchanged. All 12 candidate warmup and timed requests consumed the expected 31-layer permits.

Verification checked prefill layers 0 through 30. Each layer compared 6,291,456 active elements and 128 canaries, for totals of 195,035,136 active and 3,968 canary comparisons. Every mismatch, nonfinite and canary-change counter was zero. Layer 31 was not executed in this prefill path. Its explicit unexecuted record and null comparison counters remain in the archive. This result does not claim that all 32 layers were numerically checked.

Q8 v6r1 failed before source and runtime admission with `external lock envelope is missing /home/linuxbrew/.linuxbrew/Cellar/node@22/22.22.2_2/bin/node`. The completed v6r2 capture has its own controller, result, lease proofs and release receipt. No timing from the failed launch is reused.

## Provenance, containment and limits

`provenance.json` records each complete raw input's path, byte count and SHA-256. Every JSON projection declares its selected field paths and includes the SHA-256 of the complete raw input. Exact copies retain their original bytes. `archive-plan.json` fixes the public inputs and archive inventory.

The Laya packet preserves all 67 runtime source identities, and its runtime manifest preserves all 70 resource identities. The Q8 source preflight and controller preserve all 63 source identities. The copied private producer files are the changed Laya helper, runner, protocol JSON and packet, plus the Q8 page, numeric helper, controller, browser runner and monitor helper. Unchanged Laya integration, executable request protocol, page, browser adapter, shared browser helper and session-bus helper are retained by exact identity in the first publication. The follow-up does not duplicate those source bytes.

The compact evidence retains six Laya own-FD lease events and eight completed-Q8 own-FD lease events. They bind the current controller's PID and start time to the canonical lock device 66306, inode 134897311, and its own inherited descriptor's FLOCK rows. Global lock-table text is omitted. The exact external-release receipts retain bounded known-process absence checks, inactive or collected scopes, and successful free-lock probes. Both completed runs released all 20 tracked processes. These are sampled containment and cleanup proofs, not continuous process or thread containment.

The archive excludes model weights, pack payloads, WASM binaries, build output, full browser/controller JSON captures, full PID/TID inventories and global `/proc/locks` dumps. Pack and WASM metadata retain only what their producers actually recorded. Q8 pack file identity is not a full pack-content hash. No broad model-quality score, cross-device ranking or state-of-the-art claim follows from these two rejected candidates.

## Archive and verification

The publisher reuses the reviewed fixed-inventory projection pattern and the unchanged `cf3ad882` and `6efc5046` lease helpers. It caps an individual source at 1.5 MiB, the decoded archive at 1 MiB and gzip at 256 KiB. It checks both archive caps before writing either output, fixes gzip time to zero, and refuses existing output paths. `manifest.json` records every archived path, size and SHA-256.

Run archive and verification operations only after the resource grant, with the canonical lock inherited by the Node process. Use fresh receipt paths for each operation. A lock conflict exits 75 and must defer the operation.

```sh
env -u KEVALA_PROFILE_LOCK_HELD -u VK_DRIVER_FILES \
  flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock \
  node docs/benchmarks/gpu-epilogues-followup-2026-11-07/archive-publication.mjs \
  --archive --allow-io --receipt /tmp/kevala-gpu-epilogue-followup-archive-UNIQUE.json

env -u KEVALA_PROFILE_LOCK_HELD -u VK_DRIVER_FILES \
  flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock \
  node docs/benchmarks/gpu-epilogues-followup-2026-11-07/archive-publication.mjs \
  --verify --allow-io --receipt /tmp/kevala-gpu-epilogue-followup-verify-UNIQUE.json
```

Archive and verification operations do not launch a browser, load a model, read model or WASM payloads, or execute a GPU workload. Verification checks the public archive, manifest and plan. It does not reread the private raw captures or independently validate their historical producer receipts.
