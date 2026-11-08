# Experimental source archive

This archive stores exact source snapshots as content-addressed, inert `.txt` blobs. The archived experiments remain unpromoted and are not activated by production. `manifest.json` records each source's original repository path, base commit, and validation status.

The manifest covers 27 experimental worktrees and the source changes in the older interrupted stash. Original files, worktrees, and the stash remain intact. Generated binaries, model packs, and raw run captures are excluded. The manifest also records older branch heads backed up to GitHub without rewriting their history.

Some snapshots already have published copies. Others were unfinished or rejected, and their exact bytes have no established functional or performance result. Historical notes in the manifest apply only to the runs they describe. This archive does not establish that every candidate was tested or improves latency.

Verify manifest references, object paths, sizes, and SHA-256 digests with:

```sh
uv run --offline --no-project docs/benchmarks/experimental-sources-2026-11-08/verify.py
```

Append `--syntax` to run syntax-only checks for supported languages. Syntax parsing does not prove functional behavior, GPU compatibility, or performance.

JavaScript, Python, JSON, TOML, and Rust snapshots are parsed without executing their code. HTML, WGSL, and Markdown snapshots receive UTF-8 and integrity checks only. Rust parsing requires the installed Rust 1.95.0 `rustfmt`; JavaScript parsing requires Node.js. No models, browsers, native builds, or GPU workloads are started.
