# Firefox submission-mode evidence archive

This archive supports the [submission-mode report](../../firefox-submission-performance.md). It keeps the original browser result and runner receipt, their exact probe and runtime sources, and a derived statistics sidecar. `manifest.json` lists the byte count and SHA-256 for each payload file; it excludes itself.

`raw/result.json.gz` and `raw/runner.json.gz` use deterministic gzip. Decompressing them yields the original capture files byte for byte: the result is 687,244 bytes with SHA-256 `00b0951bb86a5e82519f8e0c37d7efa2c09332e24b8c7e83b503ecdd83991740`; the runner receipt is 60,136 bytes with SHA-256 `18c8be494da45e1758919880d4d8debcf2a516c666031285f2bbcd14c672106a`. The raw files were not rewritten. `derived-statistics.json` is a same-raw recomputation that retains both the stored and recomputed values and exact IEEE-754 ULP distances.

`sources/probe-v2/` holds the executed page, fixtures, preflight, runner, and validator. `sources/probe-v1-rejected/` preserves the earlier source set and its original manifest. That version was rejected before browser, model, or GPU execution because it read unavailable precision metadata, could miss the trailing animation-frame interval, and did not bind served runtime bytes to frozen local sources.

`runtime/js/` contains the exact JavaScript runtime snapshot. `runtime/identity.json` records its source commit, fingerprint, and WebAssembly hashes. `runtime/validation/binaries.json` preserves the separate validation binary descriptors. No WebAssembly binaries are included. `capture-runner/` retains the exact benchmark wrapper and source identity inputs. Its two historical JSON validation inputs are stored under capture-runner/reference/ with their original source paths retained in the manifest. They use deterministic gzip; their decoded bytes match the runner receipt hashes.

The archive contains no model pack weights or hidden activation values. The original result retains public responses and complete hidden-buffer hashes. The archived inputs are for inspection and analysis; they do not include everything required to rerun the capture on another host.

Check the preserved raw bytes with:

```sh
gzip -dc raw/result.json.gz | sha256sum
gzip -dc raw/runner.json.gz | sha256sum
```
