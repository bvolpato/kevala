# GPU epilogues evidence package, 2026-11-07

This package supports the [host follow-up report](../../host-followup-experiments.md). It preserves the completed Laya R5 GEGLU epilogue comparison, the two unqualified Q8 v4 attempts, and the v5-r1 setup failure. The Laya candidate failed its predeclared performance policy and remains private and default off. The Q8 attempts produced no numerical or performance result.

## Evidence contents

The fixed archive contains the report, this README, the plan, publisher, and lease helpers. It includes the Laya private source packet with its 67 source pins, the changed kernel source and numeric helper, the runner, protocol, private page, browser adapter, and the shared browser and session-bus helpers used by the run. A compact runtime manifest retains the 70 resource identities without including any binary resource. It also includes the Q8 v4 controller and result projections, both v4 external-release receipts, and the completed v5-r1 controller, timeout, and release projections.

The browser-result projection retains the complete wall and GPU timing samples, A/A pairs, acceptance quartets, all 35 golden cases and 41 questions, and the two valid 28-layer before/after prototype checks. The controller projection retains execution, adapter, pack and WASM identities, selected environment and cleanup summaries, and the controller packet hash. Six lease-event projections retain the PID/start-time identity, canonical lock inode, inherited FD binding, and the process's own fdinfo FLOCK rows. The external-release receipt and Chrome GPU log are copied exactly.

The archive plan records the source SHA-256 and byte count for each complete raw JSON input. The browser projection omits its 577,191-byte `runner.containment` process snapshot. The controller projection omits its embedded page result and raw final process arrays. The lease projections omit global `/proc/locks` text, canonical descriptor inventories, and other process or thread snapshots. The Q8 R2 result projection retains its terminal error and small browser metadata, while omitting the 514,438-byte `runner.containment` process snapshot from the 839,090-byte raw result. The Q8 v5-r1 result projection retains its error, timeout diagnostics, and small browser metadata, while omitting the 543,330-byte `runner.containment` snapshot from its 877,745-byte raw result. The v5 controller projection omits process arrays; the exact external-release receipt retains the all-process cleanup outcome. The hashes and selected field paths in `archive-plan.json` make those omissions explicit. These compact projections support recalculation of the recorded latency statistics, acceptance results, golden-vector comparisons, and selected lock proofs. They do not support a full rerun of the validators or independent comparison of full per-sample responses, which were not retained.

The archive excludes model weights, the 478,886,720-byte Laya R5 pack, compiled WASM binaries, build output, the full browser/controller JSON files, full PID/TID and lock-table dumps, and unrelated campaign files. The Laya R5 pack and WASM hashes remain in its selected metadata. The Q8 v4 timeout record retains pack file identity metadata, not a full pack hash. The exact source and artifact sizes and hashes are listed in `archive-plan.json`.

## Archive and manifest

`evidence.json.gz` contains the 37 fixed entries sorted by UTF-8 byte order of archive path. The publisher checks each input's byte count and SHA-256 before selecting JSON fields or copying it. Individual JSON sources may be up to 1.5 MiB; the decoded archive is capped at 1 MiB and the compressed archive at 256 KiB. Both archive caps are checked before either output is created. Gzip uses a zero timestamp, and outputs use exclusive-create semantics.

The frozen source-only estimate is 803,766 decoded bytes, including selected records, base64 lengths, the plan, and JSON framing. The publisher checks the actual compressed size against the 256 KiB cap before writing either output; the generated manifest records the archive size and hashes.

The embedded `archive-plan.json` fixes source pins, field selections, exclusions, and limits. `manifest.json` records every archived path, byte count, and SHA-256, plus compressed and decoded archive sizes and hashes. Each archive or verification operation writes a separate durable receipt and `.leases` sidecar with that operation's current-process FD ownership proofs. Verification checks the archive, manifest, and current public plan. It does not inspect private campaign inputs or claim to validate the producer's historical receipt.

## Review and publication commands

Run from the canonical checkout. `--plan` reads only the fixed public plan; it does not read private campaign inputs or the archive.

```sh
cd /home/bruno/githubworkspace/kevala
node docs/benchmarks/gpu-epilogues-2026-11-07/archive-publication.mjs --plan
```

Create and verify the bundle while the Node process inherits the canonical lock. Set `--campaign-root` to the primary checkout that contains `tmp/sota-20261106-kdDTGn`. Use a new receipt path for each operation. Keep both receipts and their `.leases` directories with the evidence.

```sh
archive_receipt="${TMPDIR:-/tmp}/kevala-gpu-epilogues-archive-$$.json"
verify_receipt="${TMPDIR:-/tmp}/kevala-gpu-epilogues-verify-$$.json"

env -u KEVALA_PROFILE_LOCK_HELD -u VK_DRIVER_FILES \
  flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock \
  node docs/benchmarks/gpu-epilogues-2026-11-07/archive-publication.mjs \
  --archive --allow-io --campaign-root /home/bruno/githubworkspace/kevala --receipt "$archive_receipt"

env -u KEVALA_PROFILE_LOCK_HELD -u VK_DRIVER_FILES \
  flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock \
  node docs/benchmarks/gpu-epilogues-2026-11-07/archive-publication.mjs \
  --verify --allow-io --receipt "$verify_receipt"
```

Archive creation reads the pinned JSON sources and source files. Verification reads the archive, manifest, and current plan only. Neither operation launches a browser, loads a model, reads weights or WASM bytes, or runs a GPU workload.
