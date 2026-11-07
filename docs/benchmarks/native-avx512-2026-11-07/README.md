# Native AVX-512 evidence package

This package supports the [performance report](../../native-avx512-performance.md) and links to the [dispatch design](../../native-avx512-dispatch.md). It keeps the audited production-candidate run, the earlier private prototype, and the failed receipts as separate records.

## Included evidence

- All 3,067 prototype model-run files, totaling 48,953,846 bytes, and all 3,073 production model-run files, totaling 49,475,274 bytes. The two pinned inventories describe every relative path, size, and SHA-256 digest. The combined model-run evidence is 6,140 files and 98,429,120 bytes.
- The production receipt's finalized plan, `plan.json` (113,673 bytes, SHA-256 `334de3fdaa358d59c097aef65f57f000f548e21d43b345e5171d2cdcffe58e32`), and `finalized-index.json` (11,403 bytes, SHA-256 `278df2adfd94d20b0b3e728d952007d2c9b0273d983c1983e6276643b240557d`). The distinct `draft-plan.json` (16,357 bytes, SHA-256 `d22082c0735adea4281216d40ed9353f51f9d132681f400d04dc6f2bd2aced74`) is archived and labeled as a draft.
- Production executable-gate inventory metadata for all 768 files. The archive includes every gate JSON and stderr file, plus all 86 stdout files at or below 1 MiB. This selected gate text contains 741 files and 20,370,701 bytes.
- The source scripts, plans, validations, receipts, release proofs, and selected gate text that support both model runs and the production executable gate.
- Baseline, prototype, and production kernel source snapshots, plus both candidate build scripts. The package also includes the dispatch design page and exact tested CLI hashes, but it excludes compiled executables.
- The incomplete model-capture receipt and compile-budget failure receipt. Each keeps its original failed status.
- This report, this README, the How page source, and the publisher source.

The production gate inventory describes 27 stdout files larger than 1 MiB, totaling 202,246,608 bytes. This includes 15 full assembly outputs and 12 full numerical-parity outputs. The archive omits these files. It also omits model weights, compiled executables and libraries, build caches, and full runtime directories from the earlier failed captures. The inventory metadata retains the paths, sizes, and hashes for the omitted gate outputs.

## Review commands

From the repository root, run the plan command to check the pinned sources, receipts, and inventory metadata. It enumerates the two model-run trees and the gate tree to check path coverage, but does not read model-run payload files or gate output contents.

```sh
node docs/benchmarks/native-avx512-2026-11-07/archive-publication.mjs --plan
```

The archive command reads the two model inventories and selected gate text. It includes the full 98,429,120 bytes described by the model inventories and 20,370,701 bytes of selected gate text. The verification command reads and decompresses the archive. Run each only when the canonical profile lock is available. The publisher requires the current Node process to hold the full-range lock and refuses existing outputs.

```sh
flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock \
  node docs/benchmarks/native-avx512-2026-11-07/archive-publication.mjs --archive

flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock \
  node docs/benchmarks/native-avx512-2026-11-07/archive-publication.mjs --verify
```

`evidence.json.gz` contains the selected files. `manifest.json` records each entry's size and SHA-256 digest, plus compressed and decoded archive digests. The publisher records the stable lease identity before and after I/O. It keeps the raw `/proc/locks` line for diagnosis, but does not use its mutable sequence prefix for identity comparison. Both outputs use exclusive-create semantics.
