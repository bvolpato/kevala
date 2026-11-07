# Native linear evidence format

This directory accompanies [the native AVX2 linear report](../../native-linear-performance.md). [summary.json](summary.json) identifies the final common-base source, strict public-function comparisons, kernel screen, and paired native CLI protocol. [manifest.json](manifest.json) is the authoritative archive inventory.

The collector source is `tmp/sota-20261106-kdDTGn/native-linear/archive-publication.mjs`. Its planning mode reads directory metadata without reading payload bytes. Archive mode requires an explicit final revision, `--allow-io`, and the canonical profiling lock. It refuses existing output files. Compression and archive verification run only in a separately granted I/O window.

Each `raw/*.json.gz` file is deterministic gzip JSON. Its decoded object has sorted entries containing the logical path, byte count, SHA-256, base64 encoding, and exact original bytes. The manifest records compressed and decoded hashes/lengths and every per-file hash/length. It also binds the summary and corrected derived statistics and verifies the complete final source inventory. The collector recomputes paired inference statistics from the raw observations and checks every public response against its pilot control.

[derived-statistics.json](derived-statistics.json) recomputes all eight kernel cases using conventional medians, averaging the two middle values of each 18-observation arm. The original summary fields `baselineMedianNsPerCall` and `candidateMedianNsPerCall` use the upper-middle observation, at sorted index 9. Their original hashes and bytes are preserved. The nine-round paired median ratios are unchanged. The derived file records both definitions and the raw input identities.

The groups retain final and earlier source snapshots, collectors and controllers, validation commands and receipts, raw paired samples, public responses, complete relevant assembly functions and hot-loop windows, and rejected v1/v2/v3 and inline-oracle experiments. Default and globally AVX2-enabled opaque public-function comparisons remain distinct. Earlier failures keep their original classification; their timings are not pooled into the final results.

Model weights, real input/activation tensors, compiled binaries, targets, and Cargo caches are excluded. Deterministic strict-corpus output bits are retained with their generator source and provenance receipts. Public checked-in test fixtures remain source. Whole-CLI disassembly is retained locally; the archive contains its original execution SHA/size descriptors and focused complete-function extracts instead of unrelated library disassembly. Binary and model hash descriptors identify omitted payloads.

From the primary checkout, after the I/O window is granted:

```sh
node tmp/sota-20261106-kdDTGn/native-linear/archive-publication.mjs --plan --final final-v3
node tmp/sota-20261106-kdDTGn/native-linear/archive-publication.mjs --archive --allow-io --final final-v3
node tmp/sota-20261106-kdDTGn/native-linear/archive-publication.mjs --verify --allow-io
```

The paired protocol contains 84 fresh CLI processes: 12 discarded and 72 measured. The measured timer surrounds `Model::decide`, after model load and state/questions JSON parsing, and before output JSON serialization and stdout. It includes preparation, forward, and response construction inside that call. Some preserved raw labels say "complete CLI inference request after load"; that label does not imply end-to-end CLI/process time. Model load, process wall time, and RSS remain separate. It does not measure warm resident latency. The usage field `output_tokens` counts tokenized serialized answers, rather than generation. The earlier four-call pilot remains a control rather than a standalone speed claim.

To decode and check an archive without running a model:

```js
import { readFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";
import { createHash } from "node:crypto";

const archive = JSON.parse(gunzipSync(await readFile("raw/paired-native-requests.json.gz")));
for (const file of archive.files) {
  const bytes = Buffer.from(file.data, "base64");
  const digest = createHash("sha256").update(bytes).digest("hex");
  if (bytes.length !== file.bytes || digest !== file.sha256) throw Error(file.path);
}
```
