# Disabled state-cache carry memory

The [report](../../gpu-cache-memory.md) records a 341,164,032-byte reduction in the first stateful uncached Kev-0.8B Q8 carry allocation on November 7, 2026. Decision response fields excluding timing and all returned hidden values matched exactly between the original and changed runtimes. The legacy mixed-batch cached/uncached numerical difference remains a failed strict gate in both source arms.

The completed control loaded original CI73 commit `4e27753e6f385d6fabbb6040fdd82f058a9c48a8`, run `42350139676`, then a runtime differing only in `gpu-kev.js`. The controller checkout was `8e4ae3a3ff6592e9581696057f07fa90e036e10a`. The runtime identities were pinned independently of that newer checkout.

| Audited record | SHA-256 |
| --- | --- |
| Completed control source index | `1f2b5aabb568ceab80d1556659528003a846c7fcfb337cfbec398b4526a37c8a` |
| Completed control receipt | `0a88207c22cfad784d2c4b742082b71be7ae04fea508a1bdce934c8d81c5ca69` |
| Completed control raw result | `45b0f3e683e8a5e87c2d554a160dc23b88913f884fccd4038363b2400b75b788` |
| Original GPU host source | `37519ef8c06c537c5fc736a7badef762062d082a7c3c19e605e462912c935a57` |
| Changed GPU host source | `31f9e322832be04b0773e1866be724c49a5d7c9de7d523d5d07fdb8fafd6b614` |

The publisher selects only the completed control's source, plan, index, receipt, input, command and full raw result; the corresponding records and source from the two preceding failures; the original and changed GPU host source; the small controller dependencies; and these publication sources. The first failure was a harness startup-warmup defect. The second exposed the legacy mixed-mode numerical difference. Both remain unchanged. Their copied runtime trees and snapshots are not duplicated in the archive.

`evidence.json.gz` is a gzip JSON envelope with exact file bytes encoded as base64. `manifest.json` lists file sizes, SHA-256 values, gzip and decoded-envelope identities, runtime/model references and publication freshness bindings. Model weights, copied compiled runtime binaries, build trees, browser profiles and logs are excluded. Full returned hidden values remain included in the raw JSON results. The receipt and input identify all runtime assets, including the three excluded WASM binaries, by exact size and hash.

The original experiment requires the pinned 857,259,648-byte Kev-0.8B Q8 pack, SHA-256 `a121fd39a959423e5f6a07d75a4cb33fdb87c0534aa27b7acf4f5381788097e3`, eligible hardware and the referenced CI73 runtime artifact. Its preserved controller uses the original host's absolute paths. Reproduction should copy that source into a new experiment and update locations while retaining the exact recorded bytes and results. The two false-to-true LRU clears are explicitly labeled isolation controls; both were no-ops in the changed engine. Archive verification needs neither the model nor the private campaign directory and does not rerun inference or numerical validation.

Freeze the publication sources before creating the archive. From the repository root, after the campaign owner grants the I/O window:

```sh
node docs/benchmarks/gpu-cache-memory-2026-11-07/archive-publication.mjs --plan
flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock node \
  docs/benchmarks/gpu-cache-memory-2026-11-07/archive-publication.mjs --archive --allow-io
flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock node \
  docs/benchmarks/gpu-cache-memory-2026-11-07/archive-publication.mjs --verify --allow-io
```

`--plan` reads only small source and identity records and checks payload sizes. It does not read the large input/raw files, compress evidence or write output. Archive creation and verification require the current Node PID to own the canonical whole-file kernel flock. The publisher refuses existing outputs, checks pinned bytes before and after reading, verifies gzip/base64 roundtrips, and detects later changes to the report, README, publisher, How source or changed GPU source.
