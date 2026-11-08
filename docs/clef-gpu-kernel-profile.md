# Clef GPU kernel profile, 2026-11-07

This completed diagnostic records per-request Clef GPU timestamps on an NVIDIA RTX 5070 Ti with an AMD Ryzen 9 9950X3D. It measures one default-profile request shape at a time. It does not establish a speedup or a production latency ranking.

## Measured request latency

The capture completed with exit code 0 and no reported failure. Its controller records Chromium 156. It used profile 1, automatic Clef-head routing, cache and state cache disabled, three warmups, nine samples per shape, and no head, kernel, flavor, or submit overrides. For the three request shapes, the input-token ranges were 158-159, 251-252, and 659-660. The package preserves all 27 outputs, each with `answers`, `model`, and `usage`; the API returned no raw probabilities.

| Input tokens | p50 forward (ms) | p50 GPU-label sum (ms) | p50 wall (ms) |
| ---: | ---: | ---: | ---: |
| 158-159 | 152.545 | 148.100 | 189.035 |
| 251-252 | 215.075 | 209.772 | 258.555 |
| 659-660 | 577.885 | 571.013 | 642.700 |

For each request, the GPU-label sum adds that request's finite GPU label values before taking the p50 across nine samples. It is a diagnostic timestamp total, not a clean kernel-only latency. The p50 `gate_up`, `down`, and `qkvz` labels were 62.985/33.800/24.357 ms, 87.197/50.834/33.329 ms, and 244.387/132.668/93.211 ms for the three shapes. These per-label medians are marginal medians. Do not add them to infer a request time. Timestamp spans overlap, and the profiler disables chunk yielding above 256 tokens.

All 27 request records selected the GPU and validated the route. They used automatic routing with the relaxed flavor, coordinate tile 0, head diagnostics tile 0, and no route reason. Every request had a positive finite GPU-label sum. The report preserves call and wave counters, each complete per-sample GPU label map, the request hashes, the full case wall vectors, and each complete `cases[].last` object.

These are instrumented measurements from a busy, shared host with normal CPU affinity. The capture does not claim continuous idle containment, paired comparison, historical improvement, or numerical parity with a baseline. The preserved outputs are evidence of what the API returned, not a parity test. The 10.586 GB model pack is recorded by captured metadata and size only; this run did not compute a new whole-pack hash or publish pack contents.

## Separate subgroup-matrix metadata probe

The separate WebGPU probe confirmed that the NVIDIA Blackwell adapter exposed `chromium-experimental-subgroup-matrix`. It advertised four integer configurations:

- `u8` to `u32`, M16 N16 K32
- `i8` to `i32`, M16 N16 K32
- `u8` to `u32`, M16 N8 K32
- `i8` to `i32`, M16 N8 K32

The probe created and destroyed a device with subgroups and the experimental matrix feature. The device reported a 16 KiB workgroup storage limit, 256 compute invocations, 128 MiB maximum storage-buffer binding, and a 256 MiB maximum buffer. The reported subgroup minimum and maximum were both 32. The preferred adapter request succeeded. The fallback request was not used. `shader-f16` was not requested.

Compilation was skipped with `no-advertised-f32-f32-configuration`. This probe produced no WGSL or pipeline artifact, allocated no GPU buffers, dispatched no work, and measured no numerical result or latency. The integer metadata does not establish support for an f32-f32 matrix kernel.

## Separate integer packing gate

An independently audited gate exercised one signed i8 16x16x32 subgroup-matrix operation. It used input offsets 72 and 88 words with a 1024-byte backing row stride. The browser compiled the pipeline, dispatched once, and reported no shader diagnostics, validation errors, or out-of-memory errors. It destroyed all four buffers and the device.

The gate compared all 256 i32 outputs with its reference. The arrays matched exactly, with little-endian output SHA-256 `498117f912a5f1a4a3a2f6cd8bda7bacd741f38d389ea754e842b8fa463a58b3`. It counted 106 positive, 150 negative, and zero zero outputs. All 128 prefix and tail canaries were unchanged. A separate data audit passed. The evidence preserves the complete actual and expected arrays, canaries, packing metadata, and WGSL.

The observed Chrome 156.0.8078.4 run used five recorded flags. The preferred adapter request succeeded, fallback was not used, and `shader-f16` was not requested. The projection retains the complete adapter and device metadata and the selected runner fields without process identifiers.

This result covers only the captured packing and workgroup-store exercise. It does not establish the full operator, arbitrary shapes, quantization behavior, or latency. It is separate from the capability-only probe above.

## Evidence and limits

The [evidence package](benchmarks/clef-gpu-profile-2026-11-07/README.md) keeps the CLEF profile, matrix capability probe, and integer packing gate in separate lanes. It pins the raw JSON input byte counts and SHA-256 hashes, selected own-file-descriptor lease summaries, release-union receipts, and static browser smokes. The package copies two Clef profile producer files, four matrix probe files, and five integer-gate files. Other runtime and shared route source pins map to byte-matched files or archived source copies at [immutable commit `7a6de58`](https://github.com/bvolpato/kevala/tree/7a6de58b74d2ad9e452e8ca317d84f4b83abfe86/docs/benchmarks/clef-route-2026-11-07/source), except for two pins explicitly marked hash-only. This package is not a complete portable reproduction closure.

The publisher excludes model and pack payloads, compiled WASM, full controller captures, trace files, process and thread inventories, and global lock dumps. The matrix probe is metadata-only evidence. Neither result changes runtime defaults or proves a user-visible speedup.
