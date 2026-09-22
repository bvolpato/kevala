import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  DEFAULT_PERMUTATIONS,
  decisionDataset,
  evaluateDecisions,
  expandPermutations,
  goldOptionId,
  validateDecisionRow,
} from "../benchmarks/decisions/metrics.js";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const decisionDir = resolve(root, "benchmarks/decisions");

async function readJsonl(name) {
  const text = await readFile(resolve(decisionDir, name), "utf8");
  return text.trim().split("\n").map((line) => JSON.parse(line));
}

const [semifAuthored, semifPerturbations, kevala, manifest] = await Promise.all([
  readJsonl("semif-authored144.jsonl"),
  readJsonl("semif-perturbations108.jsonl"),
  readJsonl("kevala-authored36.jsonl"),
  readFile(resolve(decisionDir, "manifest.json"), "utf8").then(JSON.parse),
]);

function assertThreeOptionRows(rows) {
  const ids = new Set();
  for (const row of rows) {
    assert.equal(typeof row.id, "string");
    assert.equal(ids.has(row.id), false, `duplicate row id ${row.id}`);
    ids.add(row.id);
    assert.equal(row.options.length, 3, row.id);
    assert.equal(new Set(row.options.map((option) => option.id)).size, 3, row.id);
    validateDecisionRow(row);
    assert.equal(row.options[row.label].id, goldOptionId(row), row.id);
  }
}

function familyCounts(rows) {
  return Object.fromEntries([...rows.reduce((counts, row) => counts.add(row.family), new Set())].sort().map((family) => [
    family,
    rows.filter((row) => row.family === family).length,
  ]));
}

test("the frozen fixture manifest and source counts are complete", () => {
  assert.deepEqual(manifest.counts, {
    kevala_authored: 36,
    semif_authored: 144,
    semif_perturbations: 108,
  });
  assert.deepEqual(manifest.expanded_counts, {
    kevala_authored: 108,
    semif_authored: 432,
    semif_perturbations: 324,
    total: 864,
  });
  assert.equal(manifest.semif.revision, "1f2dea3e25379f9dfc98cb83c324f00ab5deda37");
  assert.equal(manifest.semif.license, "MIT");
  assert.deepEqual(manifest.permutations, ["identity", "rotate1", "rotate2"]);
  assert.equal(manifest.kevala.sha256, "2b1c67080104f910412457ef77a6d37575be0b07d4e9d47c15a3da24346233c2");
  assert.equal(manifest.semif.fixtures[0].sha256, "8162d1c73f925af64453f1ec05ef36d583b3815bf698e60f0d454bd11537e079");
  assert.equal(manifest.semif.fixtures[1].sha256, "1dd7ccf80518d0e34886478ca23982aa726e9daccd343b9e95cedaf6b569bec4");
  assert.deepEqual(familyCounts(semifAuthored), {
    candidate_selection: 48,
    evidence_interpretation: 48,
    rule_application: 48,
  });
  assert.deepEqual(familyCounts(semifPerturbations), {
    candidate_selection: 36,
    evidence_interpretation: 36,
    rule_application: 36,
  });
  assertThreeOptionRows(semifAuthored);
  assertThreeOptionRows(semifPerturbations);
  assertThreeOptionRows(kevala);
});

test("SemIf rows retain provenance and source group IDs", () => {
  const authored = semifAuthored.find((row) => row.provenance?.source_key === "i_e01");
  assert.ok(authored);
  assert.equal(authored.provenance.kind, "authored_synthetic");
  assert.equal(authored.provenance.rights, "Project authored; no copied external text");
  assert.equal(authored.provenance.variant, "original");

  const perturbation = semifPerturbations.find((row) => row.provenance?.base_id === authored.id);
  assert.ok(perturbation);
  assert.equal(perturbation.provenance.source_group_id, authored.group_id);
  assert.equal(perturbation.provenance.rights, "Project authored");
  assert.match(perturbation.group_id, /\/stability$/);
});

test("canonical source groups join authored and perturbation rows while dataset scores stay separate", () => {
  const rows = [...semifAuthored, ...semifPerturbations, ...kevala];
  const predictions = rows.map((row) => ({ id: row.id, option_id: goldOptionId(row) }));
  assert.equal(decisionDataset(semifAuthored[0]), "semif-authored144");
  assert.equal(decisionDataset(semifPerturbations[0]), "semif-perturbations108");
  assert.equal(decisionDataset(kevala[0]), "kevala-authored36");

  const result = evaluateDecisions(rows, predictions, { bootstrapSamples: 20, bootstrapSeed: 19 });
  assert.equal(result.total, 288);
  assert.equal(result.correct, 288);
  assert.equal(result.groupBootstrap.groups, 72, "36 SemIf source groups plus 36 Kevala groups");
  assert.deepEqual(Object.keys(result.perDataset), [
    "kevala-authored36",
    "semif-authored144",
    "semif-perturbations108",
  ]);
  assert.equal(result.perDataset["kevala-authored36"].total, 36);
  assert.equal(result.perDataset["semif-authored144"].total, 144);
  assert.equal(result.perDataset["semif-perturbations108"].total, 108);
  for (const dataset of Object.values(result.perDataset)) {
    assert.equal(dataset.accuracy, 1);
    assert.equal(dataset.groupBootstrap.groups, 36);
  }
});

test("Kevala authored rows are balanced within each family and avoid positional cues", () => {
  assert.deepEqual(familyCounts(kevala), {
    candidate_selection: 12,
    evidence_interpretation: 12,
    rule_application: 12,
  });
  for (const family of Object.keys(familyCounts(kevala))) {
    const positions = kevala.filter((row) => row.family === family).map((row) => row.label);
    assert.deepEqual(
      Object.fromEntries([0, 1, 2].map((position) => [position, positions.filter((value) => value === position).length])),
      { 0: 4, 1: 4, 2: 4 },
      family,
    );
  }
  assert.doesNotMatch(JSON.stringify(kevala), /candidate [ab]/i);
  for (const row of kevala) {
    assert.equal(row.provenance.kind, "kevala_authored");
    assert.equal(row.provenance.rights, "Kevala project authored; no copied external text");
  }
});

test("permutation expansion preserves semantic answers and balances positions", () => {
  assert.deepEqual(DEFAULT_PERMUTATIONS, ["identity", "rotate1", "rotate2"]);
  const expanded = expandPermutations(kevala);
  assert.equal(expanded.length, 108);
  assert.equal(new Set(expanded.map((row) => row.id)).size, expanded.length);

  for (const row of kevala) {
    const variants = expanded.filter((candidate) => candidate.base_id === row.id);
    assert.equal(variants.length, 3, row.id);
    assert.deepEqual(new Set(variants.map((candidate) => candidate.options.findIndex((option) => option.id === row.gold_id))), new Set([0, 1, 2]));
    for (const variant of variants) {
      assert.equal(variant.options[variant.label].id, row.gold_id, variant.id);
      assert.deepEqual(new Set(variant.options.map((option) => option.id)), new Set(row.options.map((option) => option.id)));
    }
  }
  assert.equal(expandPermutations(semifAuthored).length, 432);
  assert.equal(expandPermutations(semifPerturbations).length, 324);

  const twoOption = [{
    id: "two-option",
    group_id: "two-option",
    options: [{ id: "yes", description: "Yes" }, { id: "no", description: "No" }],
    gold_id: "yes",
  }];
  assert.equal(expandPermutations(twoOption).length, 2, "duplicate two-option orders must collapse");
});

test("exact metrics keep invalid and missing outputs in the denominator", () => {
  const rows = [
    {
      id: "metric-1",
      group_id: "metric-group-1",
      family: "evidence_interpretation",
      options: [{ id: "yes" }, { id: "no" }, { id: "unknown" }],
      gold_id: "yes",
    },
    {
      id: "metric-2",
      group_id: "metric-group-2",
      family: "rule_application",
      options: [{ id: "permit" }, { id: "deny" }, { id: "unknown" }],
      gold_id: "permit",
    },
    {
      id: "metric-3",
      group_id: "metric-group-3",
      family: "candidate_selection",
      options: [{ id: "north" }, { id: "south" }, { id: "none" }],
      gold_id: "south",
    },
  ];
  const result = evaluateDecisions(rows, [
    { id: "metric-1", option_id: "yes", status: "native_decision" },
    { id: "metric-2", option_id: "not-an-option" },
    { id: "metric-3", status: "missing" },
  ], { bootstrapSamples: 40, bootstrapSeed: 7 });

  assert.equal(result.total, 3);
  assert.equal(result.correct, 1);
  assert.equal(result.valid, 1);
  assert.equal(result.accuracy, 1 / 3);
  assert.equal(result.coverage, 1 / 3);
  assert.equal(result.invalid, 1);
  assert.equal(result.missing, 1);
  assert.equal(result.perFamily.evidence_interpretation.accuracy, 1);
  assert.equal(result.perFamily.rule_application.accuracy, 0);
  assert.equal(result.perFamily.candidate_selection.missing, 1);
  assert.equal(result.semanticPermutationFlip.groups, 0);
  assert.deepEqual(result.selectedPositionBias.selectedPositionCounts, [1, 0, 0]);
  assert.ok(result.selectedPositionBias.expectedPositionCounts.every((value) => Math.abs(value - 1 / 3) < Number.EPSILON));
  assert.ok(Math.abs(result.selectedPositionBias.maxAbsoluteBias - 2 / 3) < Number.EPSILON);
  assert.equal(result.groupBootstrap.groups, 3);
  assert.equal(result.groupBootstrap.samples, 40);
  assert.equal(result.groupBootstrap.estimate, 1 / 3);
});

test("semantic permutation flips compare option IDs, with deterministic group bootstrap", () => {
  const expanded = expandPermutations(kevala);
  const predictions = expanded.map((row) => ({ id: row.id, optionId: goldOptionId(row) }));
  const clean = evaluateDecisions(expanded, predictions, { bootstrapSamples: 80, bootstrapSeed: 217 });
  assert.equal(clean.accuracy, 1);
  assert.equal(clean.coverage, 1);
  assert.equal(clean.semanticPermutationFlip.groups, 36);
  assert.equal(clean.semanticPermutationFlip.groupsWithPredictions, 36);
  assert.equal(clean.semanticPermutationFlip.groupsWithFlip, 0);
  assert.equal(clean.semanticPermutationFlip.rate, 0);
  assert.deepEqual(clean.selectedPositionBias.selectedPositionCounts, [36, 36, 36]);
  assert.ok(clean.selectedPositionBias.expectedPositionCounts.every((value) => Math.abs(value - 36) < 1e-10));
  assert.ok(clean.selectedPositionBias.bias.every((value) => Math.abs(value) < Number.EPSILON));
  assert.deepEqual(clean.groupBootstrap.confidence95, [1, 1]);
  assert.deepEqual(clean, evaluateDecisions(expanded, predictions, { bootstrapSamples: 80, bootstrapSeed: 217 }));

  const firstBase = expanded.find((row) => row.base_id === kevala[0].id);
  const flipped = expanded.map((row) => ({ id: row.id, optionId: goldOptionId(row) }));
  const changed = flipped.find((prediction) => prediction.id === `${firstBase.base_id}::perm:rotate1`);
  changed.optionId = firstBase.options.find((option) => option.id !== firstBase.gold_id).id;
  const flippedResult = evaluateDecisions(expanded, flipped, { bootstrapSamples: 20, bootstrapSeed: 1 });
  assert.equal(flippedResult.semanticPermutationFlip.groupsWithFlip, 1);
  assert.equal(flippedResult.semanticPermutationFlip.rate, 1 / 36);
  assert.equal(flippedResult.correct, 107);
});

test("prediction and row validation rejects ambiguous inputs", () => {
  const row = {
    id: "validation-row",
    options: [{ id: "one" }, { id: "two" }, { id: "three" }],
    gold_id: "one",
  };
  assert.throws(() => evaluateDecisions([row], [
    { id: "validation-row", option_id: "one" },
    { id: "validation-row", option_id: "two" },
  ]), /duplicate prediction/);
  assert.throws(() => evaluateDecisions([row], [{ id: "other", option_id: "one" }]), /not in the benchmark/);
  assert.throws(() => evaluateDecisions([row, { ...row }], []), /duplicate benchmark row/);
  assert.throws(() => validateDecisionRow({ id: "duplicate-options", options: [{ id: "x" }, { id: "x" }], gold_id: "x" }), /duplicate option ids/);
  assert.throws(() => validateDecisionRow({ id: "mismatched-label", label: 1, options: [{ id: "x" }, { id: "y" }], gold_id: "x" }), /label does not identify/);
});
