# Firefox WebGPU submission-mode experiment

On one headless Firefox host and one loaded Laya Q8 engine, the 47-token request p50 was **88 ms with `await` and 87 ms with `split`**. The 512-token request p50 was **592.5 ms with `await` and 91.5 ms with `split`**. Each fixture favored `split` in all nine counterbalanced quartets. This two-fixture capture does not establish a general model speedup or visible UI improvement.

The [raw evidence and source archive](benchmarks/firefox-submit-2026-11-07/README.md) retains the original raw and runner JSON, source identities, probe sources, failed v1 source review, and a derived-statistics sidecar. It excludes model weights, WebAssembly binaries, and hidden activation arrays. Each call's public response and complete hidden forward-buffer hashes remain in the unchanged raw JSON.

## Method

The probe loaded one Laya Q8 model once into an on-page WebGPU engine, with GPU profiling disabled. It used Firefox 152.0.3 in a headless Selenium profile on one NVIDIA GeForce RTX 5070 Ti host. The runtime identity was `staged-runtime-v3`, source commit `be350ec53b774bbfff3733d690cc9689625a02e7`, fingerprint `eb46c3119c1befb08f1e63bcd3499583ed6684af00d9d6094caa845ca7754bc6`. The 478,886,720-byte Laya Q8 pack had SHA-256 `b9cab75c38f026236b8d736377aa40f889d842c219d37da45cc06c782395555c`.

The fixed execution plan was 1,084 bytes with SHA-256 `523e82268185c0b9aa689b04f33ee5fb165fab490d269ce047da771abf8b9b6c`. The plan hash matched for every request. The only changed variable was `SUBMIT.mode`, set to `await` or `split` before each request.

Each fixture had three warmup calls using runs 0, 1, and 2 in `await, split, await` order. These warmups were separate from nine measured quartets using runs 3 through 11. A quartet repeated the same request twice in each mode. The order alternated between `await, split, split, await` and `split, await, await, split`. The raw calls produce 27 comparisons, three per quartet across nine rounds. Each fixture has 36 measured calls, 18 per mode. Request token counts were exactly 47 and 512.

The wall timer surrounds the awaited `Kevala.decide` call. It includes request preparation, GPU work, and readback through the public response. Loading, warmups, output hashing, and post-response validation are outside that timer. The long request crosses the 256-token chunk threshold. `await` waits for five `queue.onSubmittedWorkDone` completions per call; `split` submits chunks back to back without those waits. The probe added no queue fence. Both modes had one readback `mapAsync` call per request. The wrappers measure time until those JavaScript promises resolve; that duration is not GPU execution time. The latency difference is observed, but this probe does not isolate its exact cause or prove a contribution for each wait. No callback duration is subtracted from request time.

For each quartet, the paired ratio is the mean `split` time divided by the mean `await` time. The table reports the median of nine paired ratios. The per-mode p50 is the median of 18 individual calls. Those values answer different questions and need not produce the same ratio.

| Fixture | Await p50 | Split p50 | Median paired split/await | Paired reduction | Quartets won by split |
|---|---:|---:|---:|---:|---:|
| Short, 47 tokens | 88 ms | 87 ms | 0.988571 | 1.14% | 9/9 |
| Long, 512 tokens | 592.5 ms | 91.5 ms | 0.155085 | 84.49% | 9/9 |

All four calls in each quartet had the same request hash, public-response hash, and hidden-buffer SHA-256. Every call used the same plan hash. The capture's correctness checks passed for strict within-quartet output equality, exact token counts, fixed plan, and expected wait counts.

## Scope and limits

This measures completed request time on one headless Firefox profile, one on-page Laya Q8 engine, one runtime, one pack, and one RTX 5070 Ti host. It does not measure visible page rendering, input-to-screen latency, normal worker-backed inference, model quality, or other browsers and models. The requestAnimationFrame observations came from a headless page and do not prove visible compositor responsiveness.

This experiment changed `SUBMIT.mode` within one JavaScript page realm. The controller is module-global in that realm, so this result does not cover multiple engines using different submission modes at the same time. Keep a single selected on-page engine per page while this remains true.

`await`, `split`, and `none` are API options for model load. `await` waits between long-pass chunks, `split` submits chunks without waiting, and `none` uses one command buffer. This experiment measured only `await` and `split`; it does not justify a performance claim for `none`. The profiler's `--submit` option only changes a benchmark load. When omitted, the runner leaves the option unset and records `runtime-default`; it does not force `await`. The current decoder default is `await`, while Gemma 4 Vision defaults to `split` unless the caller supplies an option.

The measured single-engine load configuration can be selected explicitly:

```js
const model = await Kevala.load({
  model: "laya",
  precision: "q8",
  backend: "webgpu",
  onPage: true,
  submit: "split",
  cache: false,
});
```

`cache: false` disables persistent pack caching for this load. The experiment used deterministic unique states and repeated question fixtures, which the archived probe source defines. This opt-in does not change the runtime default. The API example reflects one engine in one page realm; check visible behavior in the target application before adopting it because the measurement used headless Firefox.

The retained summary metric differs from a direct recomputation of the raw calls by one ULP in each of three short-fixture quartet ratios and four ULPs in the combined metric. The case p50s and per-quartet mode means match exactly. The archive's `derived-statistics.json` lists the exact ULP distance, recomputed values, and original stored values. The raw result remains byte-for-byte unchanged.

The rejected v1 source set stopped at pre-execution review for three issues: the frozen runtime did not expose `kevala.info.precision`, its requestAnimationFrame instrumentation included a trailing frame outside the measured call, and it did not bind served runtime bytes to the frozen local source bytes. It performed no browser, model, or GPU execution. The archive retains that rejected source set separately from the passing capture.

## Reproduction inputs

The original raw result SHA-256 is `00b0951bb86a5e82519f8e0c37d7efa2c09332e24b8c7e83b503ecdd83991740`. The runner record SHA-256 is `18c8be494da45e1758919880d4d8debcf2a516c666031285f2bbcd14c672106a`. The archive manifest lists hashes for each retained raw, source, and derived file.
