#!/usr/bin/env bash
# Builds the three WebAssembly flavors into js/src (next to the runtime that loads them).
set -euo pipefail
cd "$(dirname "$0")/.."
out=js/src
build() {
  local name=$1 features=$2
  RUSTFLAGS="-C target-feature=$features" cargo build -q -p kevala-wasm --target wasm32-unknown-unknown \
    --profile release-wasm --target-dir "target/wasm-$name"
  cp "target/wasm-$name/wasm32-unknown-unknown/release-wasm/kevala_wasm.wasm" "$out/kevala-$name.wasm"
  printf '%-8s %7.0f KB\n' "$name" "$(($(wc -c < "$out/kevala-$name.wasm") / 1024))"
}
build relaxed +simd128,+relaxed-simd
build simd +simd128
build base -simd128
