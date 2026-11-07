# Matched Q8 and FP32 weight-cache evidence

This November 7, 2026 screen compared the real Kev-0.8B `L.0.gate_up` Q8 matrix with its exact FP32 expansion, using deterministic synthetic activations. The [report](../../performance-reference.md#november-7-matched-real-weight-fp32-cache-screen) records the rejected result, paired and identical-path controls, strict output checks, cache construction and memory limits. It establishes no complete-model or runtime-default improvement.

The fixed publication list includes the final source, plan, validator and runner; the raw browser result, receipts, command logs and result inventory; the earlier unexecuted source draft; and the small CI runtime identity record. It also records this publication source and report. `evidence.json.gz` is one gzip JSON envelope whose entries preserve exact source bytes as base64. `manifest.json` records the file and envelope hashes. The publisher refuses to overwrite either output.

The audited anchors are:

| Record | SHA-256 |
| --- | --- |
| Final source plan | `14b01b9a47b41e743943a5f789717e458bc126411e01e011e0fcaea9fc874e5a` |
| Passed run receipt | `8178d96770ba9e4b383f840a344451cbda9d4577115b4ef42114eb33b6f070e7` |
| Raw browser result | `6c7d68548a1ab49556d77a1f00a2d4de16405b2cdb8a3cd478d918eb66c08321` |

The original private source draft remains unchanged at `tmp/sota-20261106-kdDTGn/f32-weight-cache-v1/pre-review-v1/`; its exact bytes are included under `draft/`. It was not a failed timing attempt. The one executed screen passed its numerical checks and was rejected on performance evidence. The original [November 6 archive](../performance-reference-2026-11-06/summary.json) is unchanged.

Runtime JavaScript and compiled WASM payloads are not duplicated. The manifest and archived identity record name all 53 assets by exact path, size and SHA-256, with CI commit `dd6c94a25cb707112ff7f146ceff756d1157cfd0`, run `42340795009` and artifact-proof hash. The unchanged browser runner is referenced by Git commit, path and SHA-256. These are exact identity references, not included binary payloads. The referenced CI71 Pages deployment was unverified; that does not change the recorded local hardware result.

Model weights, scales, activation/output buffers, compiled binaries, build trees and credentials are excluded. The run freshly hashed only the header and selected tensor/scale ranges, matched them to bounded HTTP ranges, and checked file metadata and hashes before and after. Its whole-pack SHA-256 is historical. Repeating the GPU screen requires the original converted pack and eligible hardware; archive verification requires neither.

Generate the archive only after the publication sources are frozen and the campaign owner grants an I/O window. From the repository root:

```sh
node docs/benchmarks/f32-weight-cache-2026-11-07/archive-publication.mjs --plan
flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock node \
  docs/benchmarks/f32-weight-cache-2026-11-07/archive-publication.mjs --archive --allow-io
flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock node \
  docs/benchmarks/f32-weight-cache-2026-11-07/archive-publication.mjs --verify --allow-io
```

`--plan` reads small source and identity records and checks the fixed list's metadata. It does not compress or copy evidence. The other modes require this Node process to own the canonical whole-file kernel flock. Publication and verification preserve the audited records; they do not rerun the projection or extend its numerical or performance scope.
