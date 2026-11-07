# Native AUTO4 evidence archive

This archive documents the native x86_64 automatic 4×4 AVX2 candidate and its measured results. The [performance report](../../native-4x4-performance.md) summarizes the change and its limits. Full production validation passed and was independently audited. This archive does not establish the exact bytes delivered by GitHub Pages.

## Source roles and identities

The paired timing campaign used frozen Git base `9d7ee8623aad5a1fdd65dde3484fa416fa3f14f4` and candidate kernel `crates/kevala/src/kernels.rs`, SHA-256 `a1d101ed08726cafb2d7a301a39edf789adfc203044852fd8c7b4cf7cf12b5da`. The exact same kernel bytes were transplanted onto release base `dd6c94a25cb707112ff7f146ceff756d1157cfd0`, which contains the accepted native AVX2 2×4 implementation. The archive records both identities so the timing result is not attributed to a different source tree.

Unchanged source is referenced through that base commit and the source inventories rather than copied into the archive. The inventories list the exact experiment, parity, and code-generation source paths, byte counts, and SHA-256 hashes. Each immutable release snapshot also embeds a patch generated from the stated base commit to the candidate kernel. It includes the frozen candidate patch used by the experiment as a separate protocol artifact.

The experiment directories under the campaign evidence root have distinct roles:

| Group | Role | Pinned receipt or inventory |
| --- | --- | --- |
| `source-and-protocol` | Frozen source inventories, parity runner, strict code-generation plan, automatic-selection plan, source patches, and the exact release kernel patch | `source-inventory-v2.json` SHA-256 `ac67f24366b2cf7d836d1980d3c95b590d42c5959c440589b7b4836c8bea2841` |
| `prototype-micro-screen` | Earlier explicit tile-one screen and identical-library controls; kept separate from the default-mode model run | `micro-summary-v1-r2.json` SHA-256 `5cac799c773b4cc9b71998d116158d067d1b4bed43dea4a22eb9aad1920609b2`; full run receipt `micro-v1-r2/receipt.json` SHA-256 `6c9af91866364626d16f230f7be61142c55796973d088bd4c6e69c6fe3eae839` |
| `strict-parity-and-codegen` | Both default and global-AVX2 parity outputs, actual CLI loop inspection, exact source and executable identities, and gate command records | `gates-v2-r1/receipt.json` SHA-256 `4be0fbb0c3ee59de075ee36fecfa16a987f9ea3c6cdbb1575174a601585b9fbe`; gate inventory `gates-inventory-v2-r1.json` |
| `actual-default-model-run` | The paired CLI plan and helpers, all 84 request records and response outputs, host/child evidence, and release receipt | `paired-native-auto2-v2-r1/runs-v1/receipt.json` SHA-256 `b3f25efcfaab04b78fc5f8bf0f40b6ab4597ae2c124e7dc62f3bae6e2211365c`; inventory `paired-native-auto2-v2-r1/run-inventory-v1.json` |
| `production-validation-protocol` | Frozen validation plan, runner, source index, source-indexed harness helpers, launch helper, and a historical pre-fix publisher snapshot | `production-auto4-release-v1-r1/plan.json` SHA-256 `fbb68dca7c23ab543eb81d353f2cdb5ec3aaaf57ce7886a955d9f3031adaf78f`; source index SHA-256 `04bcaa369e43869ae8e488ded342a425ddfbc065be1f636484b656e8fe2ee801` |
| `production-validation-results` | Frozen result inventory and all allowed text evidence; binary build products are excluded and their identities remain in validation records | `result-inventory-v1.json` SHA-256 `d30cb21e6d815c0fdd3f8afabeb633f0e606a8a56410c6299ecb45ddf299a1c0`; receipt SHA-256 `057a37da8fe56854ba569cb876ce7253a6dfc5dc4d842d1dc0e9a685ff2d8df6` |
| `production-validation-execution` | External launch envelope and initial/final scope audits | `execution-audit-inventory-v1.json` SHA-256 `9371bab71d9370b04e3d6ea41cb529b94f98363ae8bd0ca572dec0f73b716d67`; final audit SHA-256 `7c15a11bcf4a9af5a0bbb59962aac38642f086285740ded130d5ec58e3d2baf4` |
| `publication-source` | This report, this archive guide, and the publisher source | Captured as exact source bytes in each release snapshot |

The gate receipt records 179 configurations and 297,954 output values in each build mode, with zero mismatches and passing canaries. The model receipt records 84 CLI requests, including 72 measured observations. Its results compare the candidate's actual automatic default with the accepted two-row default; no tile override was used. The production receipt records all 16 validation stages as passed, six generated WebAssembly artifact identities and ABI checks, and three exact-artifact numerical guards with 1,408 cases each. An independent audit verified all 55 result-inventory entries and the execution-scope release evidence.

## Exclusions and retained omissions

The publisher excludes model packs, measured activation and hidden-state arrays, compiled executables and libraries, object files, build trees, caches, and credentials. The candidate and baseline binaries are identified by hash in the receipts but are not embedded. Unchanged tracked source is retrievable from the base Git commit and is identified by the frozen source inventories. The archived experiment does not copy public toy source fixtures.

The compact micro-screen archive omits the 3,052,142-byte broad prototype stderr/disassembly and two shared-library copies. It also omits a small set of redundant control metadata. Every omission is listed in the release manifest with its original path, byte count, SHA-256, and reason. The source and run inventories remain included. Scoped assembly inspection outputs from the strict gate are retained.

The frozen production packet is `/home/bruno/githubworkspace/kevala/tmp/sota-20261106-kdDTGn/native-4x4/production-auto4-release-v1-r1`. Its result index is `result-inventory-v1.json` with a `files` map of relative paths to byte counts and SHA-256 hashes. The publisher accepts only this packet, verifies its exact index and passing receipt hashes, and copies allowed result files only after checking each recorded size and hash. It also archives the execution-audit index and its referenced records. Source-indexed validation helpers are included using their frozen hashes and sizes. Binary outputs stay excluded while their inventory descriptors and validation receipts remain available for audit.

## Reproduce and verify

Run `--plan --validation /home/bruno/githubworkspace/kevala/tmp/sota-20261106-kdDTGn/native-4x4/production-auto4-release-v1-r1` to inspect the input inventory and exclusions. This verifies the pinned inventory and receipt, reads compact metadata, and checks file sizes without reading the benchmark payloads or starting hardware work.

When production validation is complete and the shared profile lock is free, the parent workflow can make a snapshot from the repository root:

```sh
flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock \
  node docs/benchmarks/native-4x4-2026-11-07/archive-publication.mjs \
  --archive --allow-io \
  --evidence /home/bruno/githubworkspace/kevala/tmp/sota-20261106-kdDTGn/native-4x4 \
  --validation /home/bruno/githubworkspace/kevala/tmp/sota-20261106-kdDTGn/native-4x4/production-auto4-release-v1-r1
```

The publisher writes a new directory under `docs/benchmarks/native-4x4-2026-11-07/releases/`. It uses an exclusive partial directory and an atomic rename; it refuses to overwrite an existing release. It does not execute validation.

Verify an archive from the repository root while holding the same lock:

```sh
# Set this to the release ID printed by --archive.
RELEASE_ID='auto4-<hash-prefix>-<timestamp-from-archive-output>'
flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock \
  node docs/benchmarks/native-4x4-2026-11-07/archive-publication.mjs \
  --verify --allow-io --release "$RELEASE_ID"
```

Verification checks every compressed envelope, decoded file, report hash, raw model response against the same-model control, and recomputed paired timing statistic. Both archive and verification commands require the current process to hold the canonical full-range profile lock. They should run only in an authorized archive window after other timed profiling has stopped.
