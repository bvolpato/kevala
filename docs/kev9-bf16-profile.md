# Kev-9B BF16 short and medium profile

This later, limited capture measured resident WebGPU requests for the local Kev-9B BF16 pack on an RTX 5070 Ti with an AMD Ryzen 9 9950X3D host. Chrome 156 reported `adapter.info.isFallbackAdapter=false`; `GPUAdapter.isFallbackAdapter` was unavailable. It covers two single-question shapes only.

| Fixture | Actual input tokens | Wall median | Wall p95 | Diagnostic GPU timestamp median |
| --- | ---: | ---: | ---: | ---: |
| Short | 31–32 | 52.10 ms | 60.20 ms | 41.2185 ms |
| Medium | 124–125 | 149.80 ms | 156.50 ms | 134.756 ms |

The protocol used nine samples, three warmups, and two alternating blocks; the summary contains 18 wall observations and 18 GPU diagnostics per shape. Pack and state caches were disabled. Wall timings are awaited request times; loading, tuning, fixture hashing, and warmup requests are excluded. GPU timestamps come from separate instrumented blocks that change dispatch passes and can change optional paths. They are diagnostic kernel timing, not application latency. The request run was uncached and not an idle-host baseline.

An independent raw-block audit found identical decision fields across the repeated profiled and unprofiled lanes for both fixtures: maximum probability delta 0 and maximum public Noul delta 0, within the configured `0.0005` and `0.0011` tolerances. This verifies fixture agreement for these two inputs, not model accuracy.

The locally converted pack is 15,902,057,920 bytes with SHA-256 `32610758de1b2a192c668cc05cc6bbc97808bd74160f8034c1d11ea15070ff4e`. Its full before/after hash matched. This conversion has no catalog-pinned digest; BF16 describes the stored backbone format, while norms, recurrent gates, and decision heads are F32. No weights are included in the [evidence archive](benchmarks/kev9-bf16-2026-11-07/README.md).

The earlier [11-model catalog profile](benchmarks/kernel-profile-2026-11-07/README.md) correctly recorded Kev-9B BF16 as blocked by its explicit 9 GB pack policy. That was a policy block, not an observed allocation failure. This later capture raised the pack limit for a single model and two shapes; it does not replace the historical catalog row. It adds no long-shape result or paired precision comparison.

The measured runtime was clean commit `2883619ff46856be57760112ae250df272e82b82`, tree `c3069b7aace1bae00273de34b6e262593f831f68`. The captured runtime inventory records 53 assets, including three unchanged WASM builds by hash. The archive contains the raw capture and small source/docs files; it records the runtime inventory by hash without copying WASM binaries. It makes no peak-VRAM, model-attributed memory, or global-best latency claim.

See the [capture and reproduction notes](benchmarks/kev9-bf16-2026-11-07/README.md) for exact input hashes, the profiler command, and archive verification steps.
