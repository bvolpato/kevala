import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { FIXTURES, MEASURED_RUNS, WARMUP_RUNS, requestFor } from "./fixtures.mjs";

const REFERENCE_PATH = "tmp/kernel-profile-2026-11-06-I6lJc0/firefox-current-models/firefox/laya-q8/0-wall.json";
const REFERENCE_SHA256 = "2f6316f374500475038c3dab3280223c25b568cb4b061cd676e79eccb37b7375";
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

export async function verifyFixtures(root = fileURLToPath(new URL("../../../", import.meta.url))) {
  const referenceBytes = await readFile(resolve(root, REFERENCE_PATH));
  const referenceSha256 = sha256(referenceBytes);
  if (referenceSha256 !== REFERENCE_SHA256) throw new Error(`Laya fixture reference SHA-256 changed: ${referenceSha256}`);
  const reference = JSON.parse(referenceBytes);
  if (reference.done !== true || reference.status === "error" || reference.runs !== 9 || reference.warmups !== 3 || reference.unique !== true) {
    throw new Error("Laya fixture reference does not match the validated 9-run, 3-warmup uncached capture");
  }
  const sampleIndexes = [0, 2];
  const fixtures = FIXTURES.map((fixture, fixtureIndex) => {
    const original = reference.cases[sampleIndexes[fixtureIndex]];
    if (original?.samples?.length !== MEASURED_RUNS.length || fixture.measuredHashes.length !== MEASURED_RUNS.length) {
      throw new Error(`${fixture.id}: historical fixture sample count changed`);
    }
    if (original.samples.some((sample) => sample.tokens !== fixture.expectedTokens)) throw new Error(`${fixture.id}: historical token count changed`);
    const hashes = [...WARMUP_RUNS, ...MEASURED_RUNS].map((run) => {
      const actualHash = sha256(Buffer.from(JSON.stringify([requestFor(fixture, run)])));
      const measuredIndex = run - MEASURED_RUNS[0];
      if (measuredIndex >= 0) {
        const expectedHash = fixture.measuredHashes[measuredIndex];
        const historicalHash = original.samples[measuredIndex].requestHash;
        if (actualHash !== expectedHash || expectedHash !== historicalHash) {
          throw new Error(`${fixture.id} request Run ${run} hash mismatch: expected ${historicalHash}, got ${actualHash}`);
        }
      }
      return { run, measured: measuredIndex >= 0, sha256: actualHash };
    });
    if (new Set(hashes.map((item) => item.sha256)).size !== hashes.length) throw new Error(`${fixture.id}: generated request hashes are not unique`);
    return { id: fixture.id, expectedTokens: fixture.expectedTokens, warmupAndMeasuredHashes: hashes };
  });
  return { reference: { path: REFERENCE_PATH, sha256: referenceSha256 }, fixtures };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    process.stdout.write(`${JSON.stringify({ status: "passed", ...await verifyFixtures() }, null, 2)}\n`);
  } catch (error) {
    process.stderr.write(`${String(error?.message || error)}\n`);
    process.exitCode = 1;
  }
}
