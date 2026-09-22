// Metrics for the frozen decision benchmark. The browser harness can import this module
// without pulling in the model runtime.

export const DEFAULT_PERMUTATIONS = Object.freeze(["identity", "rotate1", "rotate2"]);

const DATASET_ALIASES = Object.freeze({
  kevala: "kevala-authored36",
  "kevala-authored36": "kevala-authored36",
  "kevala-authored36.jsonl": "kevala-authored36",
  semif: "semif",
  "semif-authored144": "semif-authored144",
  "semif-authored144.jsonl": "semif-authored144",
  "semif-perturbations108": "semif-perturbations108",
  "semif-perturbations108.jsonl": "semif-perturbations108",
});

function datasetAlias(value) {
  if (typeof value !== "string" || !value.trim()) return null;
  const normalized = value.trim().toLowerCase();
  if (DATASET_ALIASES[normalized]) return DATASET_ALIASES[normalized];
  if (normalized.includes("kevala")) return "kevala-authored36";
  if (normalized.includes("semif") && normalized.includes("perturb")) return "semif-perturbations108";
  if (normalized.includes("semif") && (normalized.includes("authored") || normalized.includes("144"))) return "semif-authored144";
  if (normalized === "semif") return "semif";
  return value.trim();
}

/** Return the stable source fixture name used for per-dataset score tables. */
export function decisionDataset(row) {
  const kind = row?.provenance?.kind;
  if (kind === "kevala_authored") return "kevala-authored36";
  if (kind === "authored_synthetic") return "semif-authored144";
  if (kind === "project_owned_output_blind_perturbation") return "semif-perturbations108";
  for (const value of [
    row?.dataset_name,
    row?.datasetName,
    row?.dataset_type,
    row?.datasetType,
    row?.source_type,
    row?.sourceType,
    row?.decision_dataset,
    row?.decisionDataset,
    row?.source_file,
    row?.sourceFile,
    row?.provenance?.dataset,
    row?.provenance?.source_file,
    row?.provenance?.type,
    row?.dataset,
  ]) {
    const alias = datasetAlias(value);
    if (alias) return alias;
  }
  return "unlabeled";
}

function groupNamespace(dataset) {
  if (dataset.startsWith("semif")) return "semif";
  if (dataset.startsWith("kevala")) return "kevala";
  return dataset;
}

function permutationOrder(length, name) {
  const identity = Array.from({ length }, (_, index) => index);
  if (name === "identity") return identity;
  if (name === "reverse") return [...identity].reverse();
  if (name === "rotate1") return identity.length ? identity.slice(1).concat(identity[0]) : [];
  if (name === "rotate2") return identity.length ? identity.slice(2).concat(identity.slice(0, 2)) : [];
  throw new Error(`unknown permutation ${name}`);
}

function distinctOrders(length, names) {
  const seen = new Set();
  return names.flatMap((name) => {
    const order = permutationOrder(length, name);
    const key = order.join(",");
    if (seen.has(key)) return [];
    seen.add(key);
    return [{ id: name, order }];
  });
}

function optionId(option) {
  if (!option || typeof option !== "object" || typeof option.id !== "string" || !option.id) {
    throw new Error("every decision option needs a nonempty string id");
  }
  return option.id;
}

export function goldOptionId(row) {
  if (typeof row?.gold_id === "string" && row.gold_id) return row.gold_id;
  if (Number.isInteger(row?.label) && row.label >= 0 && row.label < row.options?.length) {
    return optionId(row.options[row.label]);
  }
  throw new Error(`row ${row?.id ?? "<unknown>"} has no valid gold option`);
}

export function validateDecisionRow(row) {
  if (!row || typeof row !== "object") throw new Error("decision row must be an object");
  if (typeof row.id !== "string" || !row.id) throw new Error("decision row needs a stable id");
  if (!Array.isArray(row.options) || row.options.length < 2) throw new Error(`row ${row.id} needs at least two options`);
  const ids = row.options.map(optionId);
  if (new Set(ids).size !== ids.length) throw new Error(`row ${row.id} has duplicate option ids`);
  const gold = goldOptionId(row);
  if (!ids.includes(gold)) throw new Error(`row ${row.id} gold option is not listed`);
  if (Object.prototype.hasOwnProperty.call(row, "label")) {
    if (!Number.isInteger(row.label) || row.label < 0 || row.label >= ids.length || ids[row.label] !== gold) {
      throw new Error(`row ${row.id} label does not identify the gold option`);
    }
  }
  return row;
}

/**
 * Expand canonical rows into deterministic option-order runs.
 *
 * The semantic gold id is preserved while the numeric label is recomputed for each order.
 * Rows with two options naturally collapse rotate2 onto identity; one-option rows are rejected
 * by validateDecisionRow and are not part of the decision suite.
 */
export function expandPermutations(rows, names = DEFAULT_PERMUTATIONS) {
  if (!Array.isArray(rows)) throw new Error("rows must be an array");
  return rows.flatMap((row) => {
    validateDecisionRow(row);
    const gold = goldOptionId(row);
    const baseId = row.base_id ?? row.id;
    return distinctOrders(row.options.length, names).map(({ id, order }) => {
      const options = order.map((index) => row.options[index]);
      const expanded = {
        ...row,
        id: `${baseId}::perm:${id}`,
        base_id: baseId,
        permutation: id,
        options,
        gold_id: gold,
      };
      if (Object.prototype.hasOwnProperty.call(row, "label")) {
        expanded.label = options.findIndex((option) => option.id === gold);
      }
      return expanded;
    });
  });
}

function predictionId(prediction) {
  return prediction?.option_id ?? prediction?.optionId ?? prediction?.predicted_id ?? prediction?.predictedId ?? null;
}

function predictionStatus(prediction) {
  if (prediction?.status === "invalid" || prediction?.status === "error" || prediction?.status === "missing") return prediction.status;
  if (prediction?.status && !["ok", "valid", "native_decision", "distribution"].includes(prediction.status)) return "invalid";
  return predictionId(prediction) == null ? "invalid" : "ok";
}

function align(rows, predictions) {
  if (!Array.isArray(rows) || !Array.isArray(predictions)) throw new Error("rows and predictions must be arrays");
  const known = new Set();
  for (const row of rows) {
    validateDecisionRow(row);
    if (known.has(row.id)) throw new Error(`duplicate benchmark row ${row.id}`);
    known.add(row.id);
  }
  const byId = new Map();
  for (const prediction of predictions) {
    if (!prediction || typeof prediction.id !== "string") throw new Error("every prediction needs a row id");
    if (!known.has(prediction.id)) throw new Error(`prediction ${prediction.id} is not in the benchmark`);
    if (byId.has(prediction.id)) throw new Error(`duplicate prediction ${prediction.id}`);
    byId.set(prediction.id, prediction);
  }
  return rows.map((row) => {
    const prediction = byId.get(row.id);
    const status = prediction ? predictionStatus(prediction) : "missing";
    const selected = predictionId(prediction);
    const position = status === "ok" ? row.options.findIndex((option) => option.id === selected) : -1;
    const valid = status === "ok" && position >= 0;
    const finalStatus = status === "ok" && !valid ? "invalid" : status;
    const gold = goldOptionId(row);
    const dataset = decisionDataset(row);
    const sourceGroupId = row.provenance?.source_group_id ?? row.group_id ?? row.id;
    return {
      row,
      id: row.id,
      groupId: `${groupNamespace(dataset)}::${sourceGroupId}`,
      dataset,
      family: row.family ?? "unlabeled",
      baseId: row.base_id == null ? null : `${groupNamespace(dataset)}::${row.base_id}`,
      permutation: row.permutation ?? null,
      gold,
      selected: valid ? selected : null,
      position: valid ? position : -1,
      status: finalStatus,
      correct: valid && selected === gold,
    };
  });
}

function summaryFor(aligned) {
  const total = aligned.length;
  const valid = aligned.filter((row) => row.status === "ok");
  const correct = aligned.filter((row) => row.correct).length;
  const invalid = aligned.filter((row) => row.status === "invalid" || row.status === "error").length;
  const missing = aligned.filter((row) => row.status === "missing").length;
  return {
    total,
    correct,
    valid: valid.length,
    accuracy: total ? correct / total : null,
    coverage: total ? valid.length / total : 0,
    invalid,
    missing,
  };
}

function familySummary(aligned) {
  const groups = new Map();
  for (const row of aligned) {
    if (!groups.has(row.family)) groups.set(row.family, []);
    groups.get(row.family).push(row);
  }
  return Object.fromEntries([...groups.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([family, rows]) => [family, summaryFor(rows)]));
}

function positionBias(aligned) {
  const valid = aligned.filter((row) => row.status === "ok");
  const width = Math.max(0, ...aligned.map((row) => row.row.options.length));
  const selected = Array(width).fill(0);
  const gold = Array(width).fill(0);
  const expected = Array(width).fill(0);
  for (const item of aligned) {
    const count = item.row.options.length;
    const goldPosition = item.row.options.findIndex((option) => option.id === item.gold);
    if (goldPosition >= 0) gold[goldPosition]++;
    if (item.status === "ok") {
      for (let position = 0; position < count; position++) expected[position] += 1 / count;
    }
    if (item.status === "ok") selected[item.position]++;
  }
  const selectedTotal = valid.length;
  const total = aligned.length;
  const selectedRate = selected.map((value) => (selectedTotal ? value / selectedTotal : 0));
  const goldRate = gold.map((value) => (total ? value / total : 0));
  const expectedRate = expected.map((value) => (selectedTotal ? value / selectedTotal : 0));
  const bias = selectedRate.map((value, index) => value - expectedRate[index]);
  return {
    selectedPositionCounts: selected,
    goldPositionCounts: gold,
    expectedPositionCounts: expected,
    selectedPositionRate: selectedRate,
    goldPositionRate: goldRate,
    expectedPositionRate: expectedRate,
    bias,
    maxAbsoluteBias: Math.max(0, ...bias.map((value) => Math.abs(value))),
  };
}

function permutationFlips(aligned) {
  const grouped = new Map();
  for (const item of aligned) {
    if (!item.baseId) continue;
    if (!grouped.has(item.baseId)) grouped.set(item.baseId, []);
    grouped.get(item.baseId).push(item);
  }
  let withPredictions = 0;
  let flips = 0;
  for (const rows of grouped.values()) {
    const selected = new Set(rows.filter((row) => row.status === "ok").map((row) => row.selected));
    if (selected.size < 1) continue;
    withPredictions++;
    if (selected.size > 1) flips++;
  }
  return {
    groups: grouped.size,
    groupsWithPredictions: withPredictions,
    groupsWithFlip: flips,
    rate: withPredictions ? flips / withPredictions : null,
  };
}

function seededRandom(seed) {
  let state = (seed >>> 0) || 1;
  return () => {
    state = Math.imul(state ^ (state >>> 15), 1 | state);
    state ^= state + Math.imul(state ^ (state >>> 7), 61 | state);
    return ((state ^ (state >>> 14)) >>> 0) / 4294967296;
  };
}

function percentile(sorted, fraction) {
  if (!sorted.length) return null;
  const index = Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * fraction)));
  return sorted[index];
}

function groupBootstrap(aligned, samples, seed) {
  if (!Number.isInteger(samples) || samples < 1) throw new Error("bootstrap samples must be a positive integer");
  const groups = new Map();
  for (const row of aligned) {
    if (!groups.has(row.groupId)) groups.set(row.groupId, []);
    groups.get(row.groupId).push(row);
  }
  const units = [...groups.values()];
  if (!units.length) {
    return {
      groups: 0,
      samples,
      seed,
      estimate: null,
      confidence95: [null, null],
    };
  }
  const random = seededRandom(seed);
  const draws = [];
  for (let sample = 0; sample < samples; sample++) {
    let total = 0;
    let correct = 0;
    for (let index = 0; index < units.length; index++) {
      const unit = units[Math.floor(random() * units.length)];
      total += unit.length;
      correct += unit.filter((row) => row.correct).length;
    }
    draws.push(total ? correct / total : 0);
  }
  draws.sort((a, b) => a - b);
  return {
    groups: units.length,
    samples,
    seed,
    estimate: summaryFor(aligned).accuracy,
    confidence95: [percentile(draws, 0.025), percentile(draws, 0.975)],
  };
}

function metricSummary(aligned, bootstrapSamples, bootstrapSeed) {
  return {
    ...summaryFor(aligned),
    perFamily: familySummary(aligned),
    semanticPermutationFlip: permutationFlips(aligned),
    selectedPositionBias: positionBias(aligned),
    groupBootstrap: groupBootstrap(aligned, bootstrapSamples, bootstrapSeed),
  };
}

/**
 * Evaluate exact semantic decisions. Invalid and missing outputs stay in the denominator.
 * Probabilities are intentionally outside this contract: option scores are conditional and are
 * not treated as calibrated confidence.
 */
export function evaluateDecisions(rows, predictions, { bootstrapSamples = 1000, bootstrapSeed = 217 } = {}) {
  const aligned = align(rows, predictions);
  const datasets = new Map();
  for (const row of aligned) {
    if (!datasets.has(row.dataset)) datasets.set(row.dataset, []);
    datasets.get(row.dataset).push(row);
  }
  return {
    ...metricSummary(aligned, bootstrapSamples, bootstrapSeed),
    perDataset: Object.fromEntries(
      [...datasets.entries()]
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([dataset, datasetRows]) => [dataset, metricSummary(datasetRows, bootstrapSamples, bootstrapSeed)]),
    ),
  };
}
