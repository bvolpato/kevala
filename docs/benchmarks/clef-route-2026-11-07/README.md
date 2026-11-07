# Clef head route diagnostic evidence, 2026-11-07

This package preserves a Clef-flash Q8 GPU-route diagnostic and a separate WebGPU `shader-f16` capability check. The Clef request samples are diagnostic wall measurements, not a paired performance comparison. The F16 probe reports adapter and device features only. Neither result changes runtime defaults or establishes a speedup.

See [the report](../../clef-head-route-diagnostic.md) for the medians, original validator failure, corrected offline validation, capability results, and limits.

## Evidence and provenance

`archive-plan.json` pins this package's source files and documents the fixed inventory. `provenance.json` binds each projection to the complete raw input's byte count and SHA-256 and lists the selected fields. Archive mode writes `evidence.json.gz` and `manifest.json`; the manifest records the archive and each stored path, byte count, and SHA-256.

The Clef result projection retains each case's nine-value wall vector, all 27 request timing records and route decisions, all request hashes, per-sample head timing summaries, and the three final-sample timing objects. The raw capture contains no decision vectors, answer text, logits, or raw probabilities. The original failed controller is represented by its exact error and status. Corrected offline validation and its receipt are separate evidence.

The F16 records keep R1's failed adapter request separate from R2's completed capability-only baseline and diagnostic-toggle requests. The R2 source packet and recorded source identities are retained. The diagnostic toggle only demonstrates that the feature appears in adapter and device metadata.

Selected own-FD lease projections preserve the current process binding and the before/after descriptor proof. They omit full descriptor inventories, process and thread inventories, and global `/proc/locks` text. Release projections preserve pass/fail, scope collection, and lock/free-probe summaries without process ID lists.

The unchanged product sources are referenced at the immutable source commit in the report. The package copies the private benchmark and validator sources needed to interpret the captures.

## Limits and exclusions

The Clef timing spans overlap. The runtime-reported forward span includes lexical work and waits, so it is not pure GPU time. The stage medians are marginal medians, and wall measurements include instrumentation. The capture provides no paired comparison or historical speedup.

The F16 probe did not create shaders or pipelines, allocate buffers, dispatch work, load models or WASM, check numerical output, or measure latency. R1's null adapter cause is unknown. R2 used the preferred adapter request in both lanes and did not use fallback.

The archive excludes model weights, pack payloads, compiled WASM, build output, copied runtime trees, complete browser/controller captures, complete process or thread inventories, global lock dumps, and browser logs. Pack evidence records metadata and size, not a new whole-pack content hash.

## Archive limits and commands

The publisher rejects source files above 1.5 MiB, decoded archives above 1 MiB, and gzip archives above 256 KiB. Plan mode verifies the pinned package files and exact decoded-size estimate. It does not read or create an archive, manifest, or gzip stream. Archive creation and verification use separate modes, require `--allow-io`, and need a fresh receipt path. A lock conflict exits 75.

```sh
env -u KEVALA_PROFILE_LOCK_HELD -u VK_DRIVER_FILES \
  flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock \
  node docs/benchmarks/clef-route-2026-11-07/archive-publication.mjs \
  --plan

env -u KEVALA_PROFILE_LOCK_HELD -u VK_DRIVER_FILES \
  flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock \
  node docs/benchmarks/clef-route-2026-11-07/archive-publication.mjs \
  --archive --allow-io --receipt /tmp/kevala-clef-route-archive-UNIQUE.json
```

Verification checks the archive, manifest, plan, and source pins. Neither archive mode nor verification reruns a browser or reads model, pack, or WASM payloads.
