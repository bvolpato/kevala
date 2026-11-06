# Clef head evidence format

This directory accompanies [the Clef head performance report](../../clef-head-performance.md). The two runtime-v2 trials and the final runtime-v3 trial/default proof retain separate identities. See [the summary](summary.json) and [the archive manifest](manifest.json).

The archiver source is `tmp/sota-20261106-kdDTGn/archive-clef-head.mjs`. Its planning mode only lists selected paths, sizes and missing inputs. Its archive mode requires explicit final runtime, validation, QA and proof inputs, an explicit I/O flag, and the canonical profiling lock. It refuses existing output files.

Each `raw/*.json.gz` artifact is deterministic gzip JSON. The decoded object contains sorted file entries with logical source path, byte count, SHA-256, base64 encoding and the exact original bytes. `manifest.json` records compressed and decoded hashes/lengths, every per-file hash/length, source-reference coverage and omitted binary descriptors. `summary.json` keeps raw counts, original classifications, runtime/source identities and the recomputed paired results. It does not pool the three wall trials.

The groups cover precision and wall captures, harness/controller source, frozen runtime source and manifests, Rust/QA source and validation receipts, rejected arithmetic and source variants, central guard attempts, startup diagnostics, and executed preceding projection probes. Identical historical Rust files are supplied by the final source base through exact content hashes; older changed files and QA overrides are retained separately. Public deterministic test fixtures remain source. Checkpoint weights, compiled binaries, target/cache trees, next-phase source-only profiling work and private raw capture arrays are excluded. WASM SHA descriptors remain available even though WASM bytes are omitted.

The final archive command is:

```sh
node tmp/sota-20261106-kdDTGn/archive-clef-head.mjs --plan
node tmp/sota-20261106-kdDTGn/archive-clef-head.mjs --archive --allow-io \
  --final-runtime staged-runtime-v3 \
  --final-validation staged-validation-v4 \
  --final-qa production-v3 \
  --final-proof clef-projection/staged-v3/clef-default-v3-1.json \
  --final-proof clef-projection/staged-v3/clef-final-parity-v3-1.json \
  --final-proof clef-projection/staged-v3/clef-final-wall-v3-1.json \
  --extra clef-projection/staged-v3
```

The output manifest, rather than an assumed filename list, is the authoritative inventory. To decode an archive without running any model or GPU work:

```js
import { readFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";
import { createHash } from "node:crypto";

const decoded = gunzipSync(await readFile("raw/precision-wall.json.gz"));
const archive = JSON.parse(decoded);
for (const file of archive.files) {
  const bytes = Buffer.from(file.data, "base64");
  const hash = createHash("sha256").update(bytes).digest("hex");
  if (bytes.length !== file.bytes || hash !== file.sha256) throw Error(file.path);
}
```

`sourceCoverage.requiredMissing` must be empty. Overwritten intermediate native/Node validation-only source revisions are listed separately in `historicalValidationOnlyMissing`; their hash descriptors and raw receipts are preserved without reconstructing bytes. Executed GPU experiments and final signed source/validation require full source closure.
