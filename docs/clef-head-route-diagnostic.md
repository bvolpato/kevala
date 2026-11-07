# Clef head route diagnostic, 2026-11-07

This diagnostic records the route and timing events for three Clef-flash Q8 request shapes on one Linux host. It does not compare implementations or establish a speedup. Public defaults and numerical behavior are unchanged.

## Clef route capture

The saved capture contains three cases with nine measured samples each, Runs 3–11 after three warmups. Every one of the 27 samples selected a validated GPU Clef head route with relaxed WASM, coordinator tile 0, and diagnostics tile 0. Each sample recorded 13 head events: 11 Rust calls, one memory wave, and one KV wave. The samples recorded no timing errors, dropped events, overflow, or CPU fallback.

The pack and state caches were disabled. The recorded cache object was `packEnabled: false` with `state.backend: null`, `state.enabled: false`, `state.requested: false`, and `state.supported: false`. The pack catalog labels Q8 approximate. The runner used an explicit same-origin pack URL. The capture records pack metadata and size, not a fresh whole-pack content hash.

| Diagnostic median | Short, 158–159 tokens | Medium, 251–252 tokens | Long, 659–660 tokens |
| --- | ---: | ---: | ---: |
| Request wall | 182.130 ms | 249.945 ms | 643.405 ms |
| Runtime-reported forward span | 146.525 ms | 205.525 ms | 580.350 ms |
| Post-forward residual | 34.825 ms | 43.915 ms | 65.405 ms |
| Clef head begin | 1.490 ms | 2.630 ms | 7.035 ms |
| Memory ready | 0.555 ms | 0.855 ms | 2.280 ms |
| Clef head finish | 25.045 ms | 26.170 ms | 31.310 ms |
| Memory wave | 3.595 ms | 5.765 ms | 10.795 ms |
| KV wave | 3.300 ms | 5.065 ms | 9.640 ms |

The runtime-reported forward span includes lexical work and waits, not just GPU compute. The diagnostic spans overlap and must not be added; the stage medians are marginal medians, not components that add to median wall time. Wall measurements include instrumentation. The capture is not a paired comparison and cannot support a historical speedup claim.

The three `cases[].last` values are the final sample's timing objects, not medians or complete responses. The raw route capture contains no decision vectors, answer text, logits, or raw probabilities. The evidence package preserves all 27 measured samples, route decisions, request hashes, and timing maps, plus the three final timing objects. Sticky `validated` and tile fields do not establish a new per-request numerical-parity check.

The original controller remains `failed`. Its post-capture validator expected the legacy cache shape `{ packEnabled: false, stateCache: false }`, while the saved result recorded the state-cache object above. The original validator error was:

```text
AssertionError [ERR_ASSERTION]: Expected values to be strictly deep-equal:
+ actual - expected

  {
    packEnabled: false,
+   state: {
+     backend: null,
+     enabled: false,
+     requested: false,
+     supported: false
+   }
-   stateCache: false
  }
```

A corrected offline validator accepted the already-saved result: all 27 recorded routes were GPU, with nine samples per case. The original controller status and raw files were not rewritten, and no hardware rerun occurred. The package retains both validator versions, the original failure, the corrected validation receipt, the source-preservation audit, and selected own-FD lease proofs.

A separate static browser smoke loaded three modules without loading a model or requesting a GPU device. Its forced missing-module, foreign-runtime, and idle paths recorded zero model loads, GPU device requests, or forbidden payload requests. This is static delivery evidence, not part of the measured Clef requests.

## Separate F16 capability metadata

The F16 probe is a capability-only check. It did not create a shader or pipeline, allocate buffers, dispatch work, load a model or WASM module, check numerical output, or measure latency.

R1 remains a separate failed capture: the baseline page reported `No high-performance WebGPU adapter`, and its diagnostic-toggle lane did not run. The cause of that null adapter result is not established.

In R2, each lane selected an NVIDIA Blackwell nonfallback adapter in Chrome 156.0.8078.4, with matching adapter identity fields. The baseline adapter did not advertise `shader-f16`; the diagnostic-toggle adapter and device did. Both device requests succeeded and both devices were destroyed. The only Chromium argument difference was `--enable-dawn-features=vulkan_enable_f16_on_nvidia`; the requested limits were unchanged. R2 succeeded through the preferred adapter request and did not use a fallback. Therefore, the R2 source change is not evidence that a fallback change caused the difference observed between attempts.

Chrome 156.0.8078.4 pins Dawn at [`0a2c7df818e285d6db0f085135139cda04cee8bb`](https://dawn.googlesource.com/dawn.git/+/0a2c7df818e285d6db0f085135139cda04cee8bb), as recorded in the [Chrome DEPS snapshot](https://chromium.googlesource.com/chromium/src/+/refs/tags/156.0.8078.4/DEPS). Dawn's [Vulkan `shader-f16` gate](https://dawn.googlesource.com/dawn.git/+/8f96ae09bbaac54d52ca6f8aff8606ffdbe9de3f/src/dawn/native/vulkan/PhysicalDeviceVk.cpp) keeps the feature unavailable by default on NVIDIA drivers below 615.71 unless the diagnostic toggle is enabled. Fresh read-only driver metadata recorded NVIDIA GeForce RTX 5070 Ti, driver 595.71.05. This source gate is context only; it does not prove the runtime cause of the Clef route or any performance effect. The adapter also reported `chromium-experimental-subgroup-matrix`, but this probe did not request or test it.

## Evidence limits and inventory

The [evidence package README](benchmarks/clef-route-2026-11-07/README.md) documents the compact projections, source identities, exclusions, size caps, and later archive commands. The evidence preserves all selected Clef timing vectors, route records, request hashes, final timing objects, the original failure, corrected offline validation, F16 R1 and R2 results, source receipts, and bounded release/lease summaries.

The package excludes model weights, pack payloads, compiled WASM, copied runtime trees, full browser/controller captures, full PID/TID inventories, global lock-table dumps, and browser diagnostic logs. The retained process evidence is limited to selected own-FD proof fields and release summaries. The 61 original Clef source identities are retained as path, byte-count, and SHA-256 metadata; unchanged public source is referenced at immutable repository commit [`81a27608f24e0b2252318515e272387c48dec7d8`](https://github.com/bvolpato/kevala/tree/81a27608f24e0b2252318515e272387c48dec7d8).
