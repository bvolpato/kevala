# Host follow-up experiment evidence, 2026-11-07

This package supports the [host follow-up report](../../host-followup-experiments.md). It preserves original failed captures and supplemental evidence as separate files. The fixed inventory in `archive-publication.mjs` contains 100 campaign files, three pinned publication files (the report, this README, and the How page source), and the publisher itself, for 104 archive entries. It pins the fixed inputs by byte count and SHA-256; the publisher hashes itself. `--plan` lists the 104 entries and validates the pins without creating or reading an archive.

## Fixed inventory coverage

- `matmul-shape-constants-v1` includes its original failed receipt, raw browser result, plan, input and admission metadata, source index, runtime identity, and harness sources.
- `matmul-shape-constants-diagnostic-v1` includes its separate passed receipt and raw browser result, harness sources and patches, source index, runtime identity, input and admission metadata, and small retained diagnostic logs. It omits the 19,520,808-byte `chromedriver.log`. The receipt records its size.
- `matmul-fp32-k64-v1` includes its passed receipt, Clef cell receipt and raw browser result, plan and harness sources, input, command, inventory and admission metadata, source index, final cleanup record, and the CI75/CI76 delivery records plus runtime binding. It omits weights, compiled WASM, the full runtime directory, and the 1,689,696-byte ChromeDriver log.
- `gemma-attention-cold-v1` includes the failed campaign receipt, the first E2 A/B browser results, the A receipt, both command records, admission metadata, the candidate patch and changed attention source, source index, config, plan, and harness sources. It does not include later loads because the run stopped after that pair.
- `gemma-attention-budget-v1` includes its separate warm receipt and raw browser result, source index, configuration, plan, and candidate source. This warm result is not combined with the cold comparison.
- `firefox-submit-threeway-v1` includes the original raw timing file, original failed runner receipt, separate corrected derived receipt, source hash list, runtime pin, browser harness, strict validator, and offline ratio-revalidation helper.

The package includes the fixed harnesses, receipt JSON, raw result JSON, source identity metadata, and selected small logs. The K64 raw result retains emitted shader sources for review. Its source identity uses commit `8d4e4fa` and CI75 WASM, with a separate metadata binding to the actual CI76 delivery. The binding proves file identity only. It does not represent a second hardware run. The cold Gemma run records base `4e27753` and baseline CI73 delivery run `42350139676`; its candidate is an unpublished one-file deadline overlay with the same three CI73 WASM files. The separate warm run records source commit `dd6c94a` and artifact run `42340795009`. These are experiment identities. They do not establish the current deployed runtime. The Firefox capture uses historical Firefox 152.0.3 and does not establish current browser behavior. The package excludes full runtime directories, model weights, scales, model packs, compiled WASM, build output, and credentials.

## Review the fixed inventory

Run this command from the repository root. It checks file paths, sizes, and SHA-256 values. It does not compress or copy evidence.

```sh
node docs/benchmarks/host-followup-experiments-2026-11-07/archive-publication.mjs --plan
```

## Create and verify the archive

The archive command writes `evidence.json.gz` and `manifest.json` beside this README. It requires the Node process to hold the canonical, full-range profile lock. The output files use exclusive creation and are never overwritten.

```sh
flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock node \
  docs/benchmarks/host-followup-experiments-2026-11-07/archive-publication.mjs --archive --allow-io

flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock node \
  docs/benchmarks/host-followup-experiments-2026-11-07/archive-publication.mjs --verify --allow-io
```

The wrapper uses `--no-fork` so the current Node PID owns the lock. Archive and verification modes check that the lock is the canonical regular file and that its `FLOCK ADVISORY WRITE` covers byte zero through EOF. These commands do not run a benchmark, build, browser, or GPU operation.
