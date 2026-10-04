# Contributing to kevala

This page shows where the code lives and which checks to run before a change lands. CI runs the same checks on every pull request and on every push to `main`.

## Repository layout

| Path | Contents |
|---|---|
| `crates/kevala` | The Rust engine, with zero dependencies. Each model family has a directory: `laya/`, `kev/` (Kev and SemIf), and `gemma4/`. Shared code covers the pack format, the tokenizer, the CPU kernels, and the WGSL kernels in `src/wgsl/`. |
| `crates/kevala-wasm` | The C ABI that the JavaScript runtime calls. |
| `crates/kevala-cli` | The `kevala` command: `convert`, `decide`, `parity`, `bench`, `inspect`, and `wgsl`. |
| `js/src` | The browser and Node runtime. npm publishes this directory. |
| `app`, `assets`, `index.html` | The demo site. |
| `examples` | Standalone pages that show one integration each. |
| `dev` | Browser harness pages for GPU kernels and model checks. The site publishes every `.html` and `.js` file in this directory. |
| `bench.html`, `parity.html`, `parity-kev.html` | The model benchmark page and the model parity pages. |
| `scripts` | Build, check, and benchmark runners. `scripts/lib` holds the helpers that they share. |
| `tools` | Python helpers: reference fixture generators, batch conversion, evaluations, and recordings. Run them with `uv run`. |
| `tests` | JavaScript tests, the reference fixtures in `tests/fixtures`, and test helpers in `tests/support`. |
| `crates/kevala/tests` | Rust integration tests. `support/` builds small synthetic packs. |
| `benchmarks` | The decision benchmark datasets and metrics, and the raw results of each campaign. |
| `docs` | Guides and measurement reports. See the [index](docs/README.md). |
| `packs`, `skills` | The model card for the hosted packs, and the agent skill. |

`tmp/`, `target/`, and `dist/` are local directories. Git ignores them.

## Set up

1. Install [pnpm](https://pnpm.io/installation), then run `pnpm install --frozen-lockfile`. The command also installs the pinned Node.js version.
2. Install [rustup](https://rustup.rs). `rust-toolchain.toml` selects the Rust version, the `wasm32-unknown-unknown` target, `rustfmt`, and Clippy.
3. Run `pnpm build`. The command builds the three WebAssembly flavors into `js/src/`. Git does not track them.
4. Run `scripts/fetch-test-tokenizers.sh`. The script downloads the reference tokenizers that the Rust parity tests need.

## Checks to run before you push

```sh
pnpm lint:rust                             # rustfmt, then Clippy with warnings as errors
KEVALA_REQUIRE_FIXTURES=1 pnpm test:rust   # a missing reference tokenizer fails the tests
pnpm build && pnpm check:wasm              # each flavor exports every function the runtime calls
node scripts/check-cpu-tuning.mjs
scripts/build-cpu-bench.sh all
for flavor in relaxed simd base; do node scripts/bench-cpu-kernels.mjs guard "$flavor"; done
pnpm check                                 # JavaScript syntax, including inline scripts in HTML pages
pnpm test                                  # JavaScript tests
python3 -m py_compile scripts/*.py tools/*.py
pnpm stage:site                            # the site tree that GitHub Pages deploys
```

CI also runs Clippy for the WebAssembly SIMD build:

```sh
RUSTFLAGS='-C target-feature=+simd128,+relaxed-simd' \
  cargo clippy -p kevala-wasm --target wasm32-unknown-unknown --features cpu-bench --locked -- -D warnings
```

## GPU and model checks

CI has no GPU and no model weights. If a change touches a GPU trunk in `js/src/gpu*.js`, a WGSL kernel, a harness page, or the forward pass of a family, run these checks on a machine with WebGPU.

1. Put the packs under test in `tmp/` as `<model>-q8.kevala`. [docs/packs.md](docs/packs.md) shows how to download or convert them. The guards use Laya and Kev-0.8B.
2. Start the server: `pnpm serve --port 18086`.
3. Run `pnpm guard:gpu`. The guard compiles every kernel, compares the kernels with CPU references, and compares the models with their PyTorch fixtures.
4. Run `pnpm guard:cpu` for the WebAssembly backend in a browser.
5. Run `pnpm smoke tmp/laya-q8.kevala` for the Node entry point.
6. Run the native parity commands in [docs/packs.md](docs/packs.md) for each family that you changed.

The runners start Firefox by default. To use Chromium, set `KEVALA_BENCH_BROWSER=chromium`. If the runner cannot find the browser, set `CHROMIUM_BIN` to the browser binary.

## Tests

A test must protect behavior that a user or another module depends on. Do not add a test that only repeats a constant, checks that some text exists, or mocks away the code under test.

- Put engine invariants that hold for any weights in `crates/kevala/tests`. Use the synthetic packs from `support/`. Examples are shard sums, the Kev state cache, and request parts.
- Put byte-exact comparisons with upstream code in the parity tests. They use the fixtures in `tests/fixtures`.
- Put runtime logic that does not need a GPU in `tests/*.test.mjs`. Examples are batching, the page protocol, pack layouts, caching, and tuning decisions.
- When you fix a bug, add a test that fails before the fix.

## Documentation and published numbers

- A script generates the decision table in `README.md`. Do not edit the text between the `decision-benchmark` markers. [BENCHMARK.md](BENCHMARK.md) gives the command, and CI fails when the table and the raw results disagree.
- Put a new measurement report in `docs/` and its raw results in `docs/benchmarks/`. Add the report to [docs/README.md](docs/README.md).
- Report a speedup only with its raw samples, the hardware, the browser, and the numerical check that ran with it.
