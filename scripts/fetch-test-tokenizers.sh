#!/usr/bin/env bash
# Download the reference tokenizers that the Rust parity tests compare against, at pinned
# revisions, into tmp/. The files are large and belong to their model authors, so the repository
# does not track them. A file that is already present is kept.
#
# Without these files the tests in crates/kevala/tests print a notice and pass. With
# KEVALA_REQUIRE_FIXTURES=1 (as in CI) a missing file fails them.
set -euo pipefail
cd "$(dirname "$0")/.."

fetch() {
  local path=$1 url=$2
  [[ -s $path ]] && return
  mkdir -p "$(dirname "$path")"
  curl -sfL --retry 3 "$url" -o "$path.part"
  mv "$path.part" "$path"
  printf 'fetched %s\n' "$path"
}

hf=https://huggingface.co
qwen=$hf/Qwen/Qwen3.5-0.8B-Base/resolve/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
fetch tmp/laya/tokenizer/tokenizer.json "$hf/convaiinnovations/laya/resolve/1c5edc17a7acd8701df6fc341c0d179f1c62c982/tokenizer/tokenizer.json"
fetch tmp/kev/tokenizer.json "$hf/jaredpalmer/kev-0.8b/resolve/54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8/tokenizer.json"
fetch tmp/qwen35/tokenizer.json "$qwen/tokenizer.json"
fetch tmp/qwen35/tokenizer_config.json "$qwen/tokenizer_config.json"
fetch tmp/gemma4/E2B/tokenizer.json "$hf/google/gemma-4-E2B-it/resolve/3e22461f65e89153144f8adb70e3b8c2cc9845a7/tokenizer.json"
fetch tmp/gemma4/E4B/tokenizer.json "$hf/google/gemma-4-E4B-it/resolve/ee0ef6023621cff504d758262d4e04895a5af4a2/tokenizer.json"
