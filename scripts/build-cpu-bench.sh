#!/usr/bin/env bash
# Builds development-only CPU benchmark artifacts under tmp/cpu-bench.
set -euo pipefail
cd "$(dirname "$0")/.."

out=tmp/cpu-bench
mkdir -p "$out"

build() {
  local name=$1 features=$2
  RUSTFLAGS="-C target-feature=$features" cargo build --locked -q -p kevala-wasm --features cpu-bench \
    --target wasm32-unknown-unknown --profile release-wasm --target-dir "target/cpu-bench-$name"
  cp "target/cpu-bench-$name/wasm32-unknown-unknown/release-wasm/kevala_wasm.wasm" "$out/kevala-$name.wasm"
  printf '%-8s %7.0f KB\n' "$name" "$(($(wc -c < "$out/kevala-$name.wasm") / 1024))"
}

case "${1:-all}" in
  all)
    build relaxed +simd128,+relaxed-simd
    build simd +simd128,-relaxed-simd
    build base -simd128,-relaxed-simd
    ;;
  relaxed)
    build relaxed +simd128,+relaxed-simd
    ;;
  simd)
    build simd +simd128,-relaxed-simd
    ;;
  base)
    build base -simd128,-relaxed-simd
    ;;
  *)
    printf 'usage: %s [all|relaxed|simd|base]\n' "$0" >&2
    exit 2
    ;;
esac
