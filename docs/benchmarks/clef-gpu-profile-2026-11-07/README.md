# Clef GPU profile evidence, 2026-11-07

This package preserves a completed Clef GPU timestamp diagnostic, a subgroup-matrix capability probe, and an independently audited integer packing gate. The [report](../../clef-gpu-kernel-profile.md) summarizes the three separate evidence lanes and their limits.

## Package contents

`archive-plan.json` fixes the source inventory and archive limits. `provenance.json` records every raw JSON input's exact byte count and SHA-256, plus the source-copy inventory. The profile result projection preserves all 27 samples, all three case wall vectors and complete last-sample timing objects, each sample's GPU label map, routing decisions, request hash, call and wave counters, timing fields, and complete API outputs. The matrix result projection preserves adapter and device capability metadata and the compile-skip reason. The integer gate projection preserves all 256 actual and reference outputs, canaries, packing metadata, selected adapter/device metadata, compilation status, and WGSL.

The package copies eleven producer files: two Clef profile files, four subgroup-matrix probe files, and five integer-gate files. The Clef controller source identity contains 64 path/size/hash pins, including 50 `js/src/*.js` runtime files. Sixty pins map to byte-matched files or archived source copies at immutable commit `7a6de58b74d2ad9e452e8ca317d84f4b83abfe86`; the two profile producer pins are copied in this package. The remaining two pins are marked hash-only because no byte-identical public copy was found. The separate corrected offline route validator in the shared source archive has a different hash and is not substituted for the capture's original validator pin. This is not a complete portable reproduction closure.

The shared Clef route source archive is [available at the immutable merge commit](https://github.com/bvolpato/kevala/tree/7a6de58b74d2ad9e452e8ca317d84f4b83abfe86/docs/benchmarks/clef-route-2026-11-07/source). Source identity records include the exact public path and commit for each byte-matched source pin.

The CLEF and subgroup-matrix release projections use corrected observed-union v2 records. The integer gate uses its audited observed-union v1 release record. All three preserve only pass status, process/start-identity counts, absence and scope-collection summaries, and free-lock-probe status. They do not publish process identifiers. Lease projections retain selected own-FD ownership and stability checks without full descriptor inventories.

The package also includes two static browser smoke summaries with zero GPU adapter/device calls and zero forbidden payload requests. It excludes model weights, pack payloads, compiled WASM, full raw controllers, trace files, browser logs, process or thread inventories, and global `/proc/locks` text. The 10.586 GB model pack is represented by captured metadata and size only; the capture has no fresh whole-pack content hash.

## Bounded preparation and publication

Preparation reads only the pinned JSON captures, eleven pinned producer files, and the byte-matched text source references at the immutable merge commit. It verifies each input's byte count and SHA-256 before writing the whitelisted projections. It does not read pack payloads, WASM, browser binaries, or trace files. Preparation, plan verification, archive creation, and archive verification use the same nonblocking lock and inherited-FD proof.

The archive limits are 1.5 MiB per source input, 1 MiB decoded, and 256 KiB gzip. Plan mode checks source pins and the exact decoded-size estimate without creating an archive or gzip stream. Archive and verify are separate modes and require `--allow-io` and a fresh receipt path.

Run preparation and plan verification from the repository root:

```sh
timeout --foreground 120s env -u KEVALA_PROFILE_LOCK_HELD -u VK_DRIVER_FILES \
  flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock \
  node docs/benchmarks/clef-gpu-profile-2026-11-07/prepare-evidence.mjs --prepare

timeout --foreground 120s env -u KEVALA_PROFILE_LOCK_HELD -u VK_DRIVER_FILES \
  flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock \
  node docs/benchmarks/clef-gpu-profile-2026-11-07/archive-publication.mjs --plan
```

After review, archive creation and verification can be run separately with unique receipt paths:

```sh
timeout --foreground 120s env -u KEVALA_PROFILE_LOCK_HELD -u VK_DRIVER_FILES \
  flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock \
  node docs/benchmarks/clef-gpu-profile-2026-11-07/archive-publication.mjs \
  --archive --allow-io --receipt /tmp/kevala-clef-gpu-profile-archive-UNIQUE.json

timeout --foreground 120s env -u KEVALA_PROFILE_LOCK_HELD -u VK_DRIVER_FILES \
  flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock \
  node docs/benchmarks/clef-gpu-profile-2026-11-07/archive-publication.mjs \
  --verify --allow-io --receipt /tmp/kevala-clef-gpu-profile-verify-UNIQUE.json
```
