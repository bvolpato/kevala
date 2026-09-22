#!/usr/bin/env bash
# Build the three WebAssembly flavors into js/src, next to the runtime that loads them.
#
# The source tree keeps these files during the migration to release-time artifact
# generation. That keeps the static site and package usable from a checkout while
# making `pnpm pack` and CI rebuild the exact same files from the pinned toolchain.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
out=js/src

printf 'rustc: '
rustc --version
build() {
  local name=$1 features=$2
  RUSTFLAGS="-C target-feature=$features" cargo build --locked -q -p kevala-wasm --target wasm32-unknown-unknown \
    --profile release-wasm --target-dir "target/wasm-$name"
  cp "target/wasm-$name/wasm32-unknown-unknown/release-wasm/kevala_wasm.wasm" "$out/kevala-$name.wasm"
  printf '%-8s %7.0f KB\n' "$name" "$(($(wc -c < "$out/kevala-$name.wasm") / 1024))"
}
build relaxed +simd128,+relaxed-simd
build simd +simd128,-relaxed-simd
build base -simd128,-relaxed-simd
