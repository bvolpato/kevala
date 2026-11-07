# Host controls evidence package, 2026-11-07

This package supports the [host follow-up report](../../host-followup-experiments.md). The archive keeps the native Q8 panel screens, the distinct AA250 baseline control, and the Clef startup-only affinity diagnostic as separate evidence.

## Evidence contents

The archive includes the fixed plans and source indexes, Q8 code-generation gate and data-audit receipts, both panel policy decisions, the predeclared 25 ms policy, and the panel external-release record. It retains the complete panel JSONL captures and sanitized summaries with all sample arrays in their original order. The AA250 run record includes `executionPassed`, `aaControlPassed`, top-level `passed`, and `summary.passed` so the reduction status is distinct from the failed control policy.

For AA250, `summary.passed: true` means the reducer produced a complete summary from retained output. The run-level `passed` and `aaControlPassed` fields are false because the predeclared quartet policy failed. The independent audit passed its calculation and retained-data checks; it did not change that control outcome or rescue the earlier A/B result.

The startup section includes the original plan and receipt, execution record, independent audit, source index, inventory, and validator sources. The independent audit is a selected report. The archive omits the 1,630,593-byte `raw-startup.json` and 272,518-byte `lease-evidence.jsonl`, with both SHA-256 values recorded in `archive-plan.json`. Those omissions prevent full reproduction of the startup validator and thread inventory from the published bundle. The audit records the selected checks and limitations; it is not a substitute for those raw inputs.

The archive excludes model-pack contents, weights, compiled shared libraries, driver binaries, build output, and unrelated campaign files. The embedded plan preserves the fixed source identities and exact byte counts without copying those payloads.

## Archive and manifest

`evidence.json.gz` contains the sorted fixed entries in `archive-plan.json`. JSON selections require every named field, retain source array order, and omit unselected child process status fields. The publisher verifies every included source size and SHA-256 before serialization. The 1 MiB decoded cap leaves room for base64 and JSON framing around the selected receipts and samples; the 256 KiB compressed cap remains the tighter output ceiling. Both caps are checked before creating either output. Gzip uses a zero timestamp and outputs use exclusive-create semantics.

The embedded `archive-plan.json` contains the fixed source pins and exclusions. `manifest.json` records each archived path, byte count, and SHA-256, plus compressed and decoded archive sizes and hashes and exclusion metadata. Each archive or verify operation writes its own receipt and `.leases` sidecar with that operation's file-descriptor ownership proofs. Verification checks the archive, manifest, and current public plan, then records fresh proofs for its own lock ownership; it does not verify the producer's historical receipt or sidecar and does not need the private campaign inputs.

## Review commands

Run from the canonical repository checkout. `--plan` checks the fixed plan metadata only. It does not read the private campaign inputs, verify their source hashes, or read or write the archive.

```sh
node docs/benchmarks/host-controls-2026-11-07/archive-publication.mjs --plan
```

Create and verify the package while the Node process inherits the canonical full-range lock. `--no-fork` keeps that lock on the Node process. The publisher records and fsyncs each operation's lease proofs before and after that operation. Each explicit receipt path also names its own `.leases` directory.

```sh
cd /home/bruno/githubworkspace/kevala
archive_receipt="${TMPDIR:-/tmp}/kevala-host-controls-archive-$$.json"
verify_receipt="${TMPDIR:-/tmp}/kevala-host-controls-verify-$$.json"

flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock \
  node docs/benchmarks/host-controls-2026-11-07/archive-publication.mjs \
  --archive --allow-io --campaign-root /home/bruno/githubworkspace/kevala --receipt "$archive_receipt"

flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock \
  node docs/benchmarks/host-controls-2026-11-07/archive-publication.mjs \
  --verify --allow-io --receipt "$verify_receipt"
```

Verification reads the published archive, manifest, and current plan metadata. It does not access the private campaign inputs. Keep both operation receipts and their `.leases` directories with the published evidence.
