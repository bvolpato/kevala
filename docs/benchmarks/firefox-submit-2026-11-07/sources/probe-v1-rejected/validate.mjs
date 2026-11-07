import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { FIXTURES, MEASURED_RUNS, ORDERS, QUARTETS, canonicalJson, requestFor } from "./fixtures.mjs";

const sha256 = (value) => createHash("sha256").update(value).digest("hex");
const finitePositive = (value) => typeof value === "number" && Number.isFinite(value) && value > 0;

export function validateResult(result) {
  assert.equal(result?.done, true, "browser result is not done");
  assert.equal(result?.status, "done", result?.error || "browser result status is not done");
  assert.equal(result.backend, "webgpu");
  assert.equal(result.threads, 1);
  assert.equal(result.profile, false);
  assert.equal(result.engine?.singleResident, true);
  assert.equal(result.engine?.onPage, true);
  assert.equal(result.correctness?.ok, true);
  assert.equal(result.cases?.length, FIXTURES.length);
  assert.equal(result.protocol?.quartets, QUARTETS);
  assert.deepStrictEqual(result.protocol?.orders, ORDERS);
  assert.equal(typeof result.fixedPlan?.json, "string");
  assert.deepStrictEqual(result.fixedPlan.snapshot, JSON.parse(result.fixedPlan.json));
  assert.equal(result.fixedPlan.sha256, sha256(result.fixedPlan.json));

  const caseSummaries = [];
  for (let fixtureIndex = 0; fixtureIndex < FIXTURES.length; fixtureIndex++) {
    const fixture = FIXTURES[fixtureIndex];
    const actual = result.cases[fixtureIndex];
    assert.equal(actual.id, fixture.id);
    assert.equal(actual.expectedTokens, fixture.expectedTokens);
    assert.equal(actual.quartets?.length, QUARTETS);
    assert.equal(actual.warmups?.length, 3);
    const wallSamples = [];

    for (let quartetIndex = 0; quartetIndex < QUARTETS; quartetIndex++) {
      const quartet = actual.quartets[quartetIndex];
      const expectedOrder = ORDERS[quartetIndex % ORDERS.length];
      const run = MEASURED_RUNS[quartetIndex];
      const request = requestFor(fixture, run);
      const expectedRequestJson = canonicalJson(request);
      const expectedRawHash = sha256(JSON.stringify([request]));
      assert.equal(quartet.quartet, quartetIndex);
      assert.equal(quartet.run, run);
      assert.deepStrictEqual(quartet.order, expectedOrder);
      assert.equal(quartet.calls?.length, 4);
      assert.deepStrictEqual(quartet.calls.map((call) => call.mode), expectedOrder);

      let firstPublic;
      let firstHiddenHash;
      let hiddenBytes;
      for (const call of quartet.calls) {
        assert.equal(call.run, run);
        assert.equal(call.requestHash, fixture.measuredHashes[quartetIndex]);
        assert.equal(call.requestHash, expectedRawHash);
        assert.equal(call.requestCanonicalJson, expectedRequestJson);
        assert.deepStrictEqual(call.requestSnapshot, JSON.parse(expectedRequestJson));
        assert.equal(call.requestCanonicalHash, sha256(expectedRequestJson));
        assert.equal(call.planSha256, result.fixedPlan.sha256);
        assert.equal(call.publicSha256, sha256(call.publicJson));
        assert.deepStrictEqual(call.publicSnapshot, JSON.parse(call.publicJson));
        assert.ok(finitePositive(call.wallMs));
        assert.equal(call.timing?.tokens, fixture.expectedTokens);
        assert.ok(Number.isSafeInteger(call.hiddenBuffer?.bytes) && call.hiddenBuffer.bytes > 0);
        assert.ok(Number.isSafeInteger(call.hiddenBuffer?.floats) && call.hiddenBuffer.floats > 0);
        assert.match(call.hiddenBuffer?.sha256 || "", /^[a-f0-9]{64}$/);
        assert.equal(call.waits?.mapRead?.count, 1);
        const expectedQueueWaits = fixture.expectedTokens > 256 && call.mode === "await" ? 5 : 0;
        assert.equal(call.waits?.queueOnSubmittedWorkDone?.count, expectedQueueWaits);
        assert.ok(Number.isFinite(call.waits?.mapRead?.totalMs) && call.waits.mapRead.totalMs >= 0);
        assert.ok(Number.isFinite(call.waits?.queueOnSubmittedWorkDone?.totalMs) && call.waits.queueOnSubmittedWorkDone.totalMs >= 0);
        if (call.requestAnimationFrame.maxGapMs !== null) assert.ok(Number.isFinite(call.requestAnimationFrame.maxGapMs) && call.requestAnimationFrame.maxGapMs >= 0);
        if (firstPublic === undefined) {
          firstPublic = call.publicSnapshot;
          firstHiddenHash = call.hiddenBuffer.sha256;
          hiddenBytes = call.hiddenBuffer.bytes;
        } else {
          assert.deepStrictEqual(call.publicSnapshot, firstPublic, `${fixture.id} Run ${run} public output differs within quartet`);
          assert.equal(call.hiddenBuffer.sha256, firstHiddenHash, `${fixture.id} Run ${run} hidden buffer differs within quartet`);
          assert.equal(call.hiddenBuffer.bytes, hiddenBytes);
        }
        wallSamples.push(call.wallMs);
      }
    }

    assert.equal(actual.samples?.length, QUARTETS * ORDERS[0].length);
    assert.ok(finitePositive(actual.p50));
    caseSummaries.push({ id: fixture.id, expectedTokens: fixture.expectedTokens, measuredCalls: wallSamples.length, p50WallMs: actual.p50,
      awaitWallMedianMs: medianFor(actual.quartets.flatMap((q) => q.calls.filter((c) => c.mode === "await").map((c) => c.wallMs))),
      splitWallMedianMs: medianFor(actual.quartets.flatMap((q) => q.calls.filter((c) => c.mode === "split").map((c) => c.wallMs))),
      awaitFrameGapMedianMs: medianFor(actual.quartets.flatMap((q) => q.calls.filter((c) => c.mode === "await").map((c) => c.requestAnimationFrame.maxGapMs).filter(Number.isFinite)), true),
      splitFrameGapMedianMs: medianFor(actual.quartets.flatMap((q) => q.calls.filter((c) => c.mode === "split").map((c) => c.requestAnimationFrame.maxGapMs).filter(Number.isFinite)), true),
    });
  }
  assert.ok(finitePositive(result.metricMs));
  return { status: "passed", planSha256: result.fixedPlan.sha256, cases: caseSummaries };
}

function medianFor(values, allowEmpty = false) {
  if (!values.length) {
    if (allowEmpty) return null;
    throw new Error("cannot summarize an empty sample list");
  }
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
