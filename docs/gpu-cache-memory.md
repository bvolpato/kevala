# Smaller GPU carries when the state cache is disabled

With `stateCache: false`, Kev-0.8B Q8 used **325.359375 MiB less carry storage** for the first stateful request in a November 7, 2026 WebGPU comparison on an RTX 5070 Ti. The original allocated 367,132,672 bytes; the change allocated 25,968,640 bytes, a 92.9266% reduction. These are the actual sizes of explicitly created carry buffers, not total VRAM usage or a latency result.

The default remains `stateCache: true`. Cache-enabled planning and its allocation budget are unchanged. Explicitly disabling the cache now omits the four retained cache slots and their 4,096 reserved KV rows. States still keep the carries their question branches need, and exact duplicate states within a pass still share work. An empty folded state still uses the existing minimal allocation.

```js
const model = await Kevala.load({
  model: "kev-0.8b",
  precision: "q8",
  backend: "webgpu",
  stateCache: false,
});
```

## What was measured

The fixture used a 40-token state and three questions. One normal engine ran the original CI73 runtime, followed by a second normal engine with only `gpu-kev.js` changed. The first device was explicitly destroyed before the second load. Both used the same observed matrix choices, recurrence selection, browser features and submission policy on a Ryzen 9 9950X3D host.

| Explicit carry allocation | Original | Change |
| --- | ---: | ---: |
| Startup warmup | 24,788,992 bytes | 24,788,992 bytes |
| First stateful uncached request | 367,132,672 bytes | 25,968,640 bytes |
| After enabling and growing the cache | 367,132,672 bytes | 367,132,672 bytes |

The first stateful request saved **341,164,032 bytes**. Storage grows when a later request needs it. Disabling caching after that growth reuses the existing capacity; it does not shrink buffers or add allocation churn. Consequently, this run did not reduce its later cached allocation or its whole-run peak. The original and changed engines' peak explicit GPU-buffer totals were 981,013,584 and 981,016,720 bytes respectively.

The measurement excludes browser, driver and pipeline allocations. It does not establish lower latency, a total VRAM reduction, or eligibility for a larger model or another precision. Clef's normal complete-sequence path already has no state-prefix pass and is not the measured memory improvement here.

## Exact source parity and an existing numerical limit

All **21 corresponding operations and 254,976 returned hidden floats** matched the original runtime exactly, together with every decision response field excluding timing. This includes startup replay, empty states, duplicate states, prefix extension, multiple questions and cache-mode changes. Fourteen operations also matched the preceding candidate capture exactly.

The diagnostic explicitly cleared the LRU at each false-to-true boundary to give both source arms the same cold-cache history. The original removed two entries at each boundary. The changed engine already had an empty LRU at both boundaries, so its two clears were no-ops. This isolation control did not substitute a different candidate execution path.

The original runtime itself does not produce bit-identical cached and uncached outputs for every batch geometry. In the mixed fixture, 26,114 of 30,720 hidden words differed between modes, with a maximum absolute difference of 0.00002288818359375. Thirteen raw probability components differed by at most 0.000000476837158203125; rounded answers were unchanged. The changed runtime reproduced those same values exactly in each corresponding mode. The strict cross-mode gate remains recorded as failed in both arms; no tolerance was relaxed. The uncached mixed pass computes 97 state tokens together, whereas cached carries came from separate 40-token and 17-token passes with different split-K geometry.

The [fixed evidence archive and reproduction notes](benchmarks/gpu-cache-memory-2026-11-07/README.md) preserve the completed control, both earlier failures, exact runtime and model identities, and the original numerical limitation. This is a scoped allocation improvement under the checked workload, not a claim that every kernel or model is optimal.
