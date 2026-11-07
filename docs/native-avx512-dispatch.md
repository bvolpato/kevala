# Scoped Ryzen AVX-512 linear dispatch

The native four-row specialization is limited to the measured AMD CPUID identity: vendor `AuthenticAMD`, display family `0x1a` (26), model `0x44` (68), stepping 0. The measured host is a Ryzen 9 9950X3D. This identifies a processor revision, rather than a unique SKU. Other CPUs with the same identity remain unmeasured.

Eligibility also requires an x86-64 build without globally enabled AVX2, a Rust compiler with stable AVX-512 intrinsics, runtime AVX512F support with operating-system state enabled, and the existing automatic four-row geometry: tile 2, `T >= 8`, `N >= 1024`, and `K >= 1024`. CPU identity and runtime ISA support are cached once per process. Other identities and ineligible shapes retain the accepted AVX2 behavior. Explicit tile modes keep their existing selection.

The build script enables the private compiler configuration only for stable Rust 1.89 or newer on an eligible target. Unrecognized compiler versions and prereleases leave it disabled. Rust 1.85.1, globally AVX2-enabled builds, ARM, and WASM retain their existing source paths. There is no new Cargo feature, environment setting, or public API.

The kernel preserves the measured precursor's arithmetic: four accumulators, separate multiply and add, four reduction residues, `(0 + 2) + (1 + 3)` reduction, and the same bias placement. The existing decoder, AVX2 tails, finite-output scan, and single shared whole-call generic recomputation remain intact.

The production build passed workspace release tests with Rust 1.85.1, 1.89.0, and 1.95.0. Actual public-function comparisons passed within each compiler in default and globally AVX2-enabled builds, preserving every output bit, signed NaN payload and sign, and canary. CPU 7's actual CPUID and runtime ISA checks selected the automatic specialization on the eligible compilers. Rust 1.85.1 and globally AVX2-enabled builds excluded it as intended.

The Rust 1.95 production CLI and public parity probe retained the measured precursor's exact normalized ZMM hot loop: four accumulators, separate multiply and add, and no FMA, spills, or calls in the core. Rust 1.89 also passed positive ZMM code-generation checks and its own strict numerical and isolation comparisons. The generic wrapper, decoders, accepted AVX2 cores, and shared nonfinite fallback passed their isolation checks.

The three WASM flag sets, with and without the numerical benchmark feature, passed compile checks with the specialization excluded. ARM coverage consists of source preservation and build-script target exclusion. Shipping WASM byte identity and any GPU evidence carry-forward require separate artifact validation.

The production request capture used the actual default without a tile override. All 304 fresh responses matched their accepted baseline controls exactly. Short and longer requests for Laya Q8 and Kev 0.8B Q8 each had eight experimental quartets and eight interleaved identical-binary control quartets on CPU 7. All control paired medians met the declared range `[0.95, 1.05]`, without correction or outlier exclusion.

See the [native AVX-512 performance report](native-avx512-performance.md) for the qualified paired measurements and retained evidence. Every experimental quartet-ratio range crossed 1.0. The Kev-small independent candidate median was 1.13% slower. The result applies to the declared paired metric, four requests, and this host; it does not establish that every request is faster.
