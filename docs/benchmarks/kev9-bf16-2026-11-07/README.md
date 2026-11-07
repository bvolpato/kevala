# Kev-9B BF16 short and medium capture

This is a supplemental capture of Kev-9B BF16 on the RTX 5070 Ti / Ryzen 9 9950X3D host. It measures only resident, uncached, short and medium requests (31–32 and 124–125 tokens). The [report](../../kev9-bf16-profile.md) gives wall latency separately from instrumented GPU timestamps. It does not replace the earlier [catalog profile](../kernel-profile-2026-11-07/README.md), where the 15,902,057,920-byte pack was blocked by the explicit 9 GB policy.

The independent audit passed for the two shapes. Repeated profiled and unprofiled lanes had zero maximum probability and public Noul deltas. This is fixture agreement, not model accuracy. No long shape, paired precision comparison, or model-level memory estimate is available.

## Frozen inputs and source

The fixed allowlist includes the complete summary, its four referenced wall/GPU raw blocks, `js/src/gpu-kev.js`, and the two harness files named by the summary (`bench.html` and `dev/adapter-metadata.js`). It also includes this report, this README, the reproduction script, the How page source, and the publisher itself. The publisher verifies pinned input hashes and sizes, regular-file paths, and the runtime/harness identity embedded in the summary. Model weights, scales, activation storage, WASM binaries, and build trees are excluded.

The converted pack is identified by size and whole-pack SHA-256 in the summary. Its full hash was verified before and after the capture. The conversion is local and has no catalog-pinned digest. The runtime identity records commit `2883619ff46856be57760112ae250df272e82b82`, tree `c3069b7aace1bae00273de34b6e262593f831f68`, and 53 assets. Their hashes remain references in the summary; the archive does not copy the WASM payloads.

## Reproduce

Use the existing model profiler and the same runtime plus locally converted pack served from one origin. The measured pack must be present as `kev-9b-bf16.kevala`; allow enough explicit pack capacity for its 15.9 GB size.

```sh
pnpm profile:models --base=http://127.0.0.1:PORT/ \
  --runtime=js/src/index.js --pack-base=tmp/ --packs=tmp \
  --models=kev-9b --precisions=bf16 --browser=chromium \
  --protocol=resident --samples=9 --warmups=3 --blocks=2 --shapes=0,1 \
  --max-pack-bytes=16000000000 --output=tmp/kev9-bf16-reproduction
```

Replace `PORT` with the local static server port. This command requires compatible WebGPU hardware and the exact local pack. It records a new run; it does not reconstruct the archived capture.

## Archive and verify

`--plan` checks the fixed files and their hashes without compressing or copying evidence. Archive and verification modes require the current Node process to hold the canonical whole-file profile lock. Outputs are exclusive and immutable. The archive is gzip-compressed JSON with exact file bytes encoded as base64; verification checks gzip roundtrip, byte counts, and SHA-256 values.

```sh
node docs/benchmarks/kev9-bf16-2026-11-07/archive-publication.mjs --plan
flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock node \
  docs/benchmarks/kev9-bf16-2026-11-07/archive-publication.mjs --archive --allow-io
flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  /home/bruno/.cache/kevala/gpu-profile.lock node \
  docs/benchmarks/kev9-bf16-2026-11-07/archive-publication.mjs --verify --allow-io
```

The evidence directory is fixed in the publisher as `/tmp/kevala-no-cache-gpu-carries/tmp/kev9-bf16-short-medium-v1`. The archive and manifest are created beside this README and will not be overwritten. GPU measurements and archive generation are separate operations.
