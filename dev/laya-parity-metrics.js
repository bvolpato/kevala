export const LAYA_Q8_GOLDEN_MAX_DP = 0.03;
export const LAYA_WASM_GPU_MAX_DP = 5e-4;
export const LAYA_API_PROBABILITY_DECIMALS = 4;

const typeIndex = { choice: 0, score: 1, noul: 2 };

function invalid(message) {
  throw new Error(`Laya parity: ${message}`);
}

function sameKeys(left, right) {
  return left.length === right.length && new Set(left).size === left.length && new Set(right).size === right.length && left.every((key) => right.includes(key));
}

function optionKeys(question, expected, caseId, questionId) {
  const fail = (message) => invalid(`${caseId}/${questionId}: ${message}`);
  if (!question || !Object.hasOwn(typeIndex, question.type)) fail(`unsupported question type ${question?.type}`);
  if (!Array.isArray(expected?.logits) || !expected.logits.length || expected.logits.some((value) => !Number.isFinite(value))) {
    fail("golden logits must be a nonempty finite vector");
  }
  if (!Array.isArray(expected.options) || expected.options.length !== expected.logits.length) {
    fail("golden option and logit counts differ");
  }
  if (expected.options.some((option) => typeof option !== "string")) fail("golden options must be strings");

  if (question.type === "noul") {
    if (expected.logits.length !== 2) fail("noul questions must have two options");
    return ["false", "true"];
  }

  if (question.type === "score") {
    const criteria = question.criteria;
    const count = Array.isArray(criteria) ? criteria.length
      : criteria && typeof criteria === "object" ? Object.keys(criteria).length
      : typeof criteria === "string" ? [...criteria].length
      : 0;
    if (count !== expected.logits.length) fail(`score has ${count} criteria for ${expected.logits.length} logits`);
    return Array.from({ length: count }, (_, index) => String(index));
  }

  const criteria = question.criteria;
  if (Array.isArray(criteria)) {
    if (!criteria.length || criteria.some((value) => typeof value !== "string")) {
      fail("choice fixture criteria lists must contain strings");
    }
    const ids = [...new Set(criteria)];
    if (ids.length !== expected.logits.length) fail(`choice has ${ids.length} options for ${expected.logits.length} logits`);
    if (expected.options.length !== ids.length || expected.options.some((option, index) => option !== ids[index])) {
      fail("choice fixture options do not match criteria keys and order");
    }
    return ids;
  }
  if (!criteria || typeof criteria !== "object") fail("choice criteria must be an object or list");
  const ids = Object.keys(criteria);
  if (ids.length !== expected.logits.length) fail(`choice has ${ids.length} options for ${expected.logits.length} logits`);
  // Recover object keys from rendered labels so integer-like keys retain fixture order.
  const ordered = expected.options.map((option) => {
    const matches = ids.filter((id) => option === id || option.startsWith(`${id}: `));
    const longest = Math.max(...matches.map((id) => id.length), -1);
    const best = matches.filter((id) => id.length === longest);
    if (best.length !== 1) fail(`cannot map rendered option ${JSON.stringify(option)} to one choice key`);
    return best[0];
  });
  if (new Set(ordered).size !== ids.length || !sameKeys(ordered, ids)) fail("choice fixture options do not match its criteria keys");
  return ordered;
}

function goldenVector(golden, question, expected, caseId, questionId) {
  const count = expected.logits.length;
  const kind = question.type;
  const size = count <= 2 ? "2" : count <= 5 ? "3-5" : count <= 10 ? "6-10" : "11+";
  const calibrated = golden.temperature_by_options?.[`${kind}:${size}`] ?? golden.temperature?.[typeIndex[kind]];
  if (!Number.isFinite(calibrated)) invalid(`${caseId}/${questionId}: golden temperature is invalid`);
  const temperature = Math.min(5, Math.max(0.5, calibrated));
  const max = Math.max(...expected.logits);
  const exp = expected.logits.map((value) => Math.exp((value - max) / temperature));
  const sum = exp.reduce((total, value) => total + value, 0);
  if (!Number.isFinite(sum) || sum <= 0) invalid(`${caseId}/${questionId}: golden softmax is invalid`);
  return exp.map((value) => value / sum);
}

function publicVector(answer, kind, keys, caseId, questionId, backend) {
  const fail = (message) => invalid(`${backend} ${caseId}/${questionId}: ${message}`);
  if (!answer || answer.type !== kind) fail(`expected answer type ${kind}`);
  let values;
  if (kind === "noul") {
    if (!Number.isFinite(answer.noul) || answer.noul < 0 || answer.noul > 1) fail("noul probability is non-finite or outside [0, 1]");
    values = [1 - answer.noul, answer.noul];
  } else {
    const probabilities = answer.probabilities;
    if (!probabilities || typeof probabilities !== "object" || Array.isArray(probabilities)) fail("probabilities must be an option map");
    const actualKeys = Object.keys(probabilities);
    if (!sameKeys(actualKeys, keys)) fail(`probability keys ${JSON.stringify(actualKeys)} do not match options ${JSON.stringify(keys)}`);
    values = keys.map((key) => probabilities[key]);
    if (kind === "choice") {
      if (typeof answer.choice !== "string" || !keys.includes(answer.choice)) fail("choice result names an unknown option");
      const max = Math.max(...values);
      if (values[keys.indexOf(answer.choice)] !== max) fail("choice result does not match the public probability vector");
    }
  }
  if (values.length !== keys.length) fail(`probability count ${values.length} does not match ${keys.length} options`);
  if (values.some((value) => !Number.isFinite(value) || value < 0 || value > 1)) fail("probability vector contains a non-finite or out-of-range value");
  const sum = values.reduce((total, value) => total + value, 0);
  const roundingBound = keys.length * 0.0000501 + 1e-7;
  if (Math.abs(sum - 1) > roundingBound) fail(`probability vector sums to ${sum}, outside API rounding bound ${roundingBound}`);
  return values;
}

function argmax(values) {
  let best = 0;
  for (let index = 1; index < values.length; index++) if (values[index] > values[best]) best = index;
  return best;
}

function maxDelta(left, right) {
  if (left.length !== right.length) invalid("probability vector cardinalities differ");
  return Math.max(...left.map((value, index) => Math.abs(value - right[index])));
}

function validateRecords(golden, records, backend) {
  if (!Array.isArray(records) || records.length !== golden.cases.length) {
    invalid(`${backend} returned ${records?.length ?? "no"} cases for ${golden.cases.length} fixtures`);
  }
  return records;
}

export function assessLayaParity(golden, wasmRecords, gpuRecords) {
  if (!golden || !Array.isArray(golden.cases) || !golden.cases.length) invalid("golden fixtures are missing");
  validateRecords(golden, wasmRecords, "WASM");
  validateRecords(golden, gpuRecords, "WebGPU");

  let maxBackendDp = 0;
  let maxWasmGoldenDp = 0;
  let maxGpuGoldenDp = 0;
  let backendArgmaxMatches = 0;
  let wasmGoldenArgmaxMatches = 0;
  let gpuGoldenArgmaxMatches = 0;
  let questionCount = 0;
  const cases = [];

  for (let caseIndex = 0; caseIndex < golden.cases.length; caseIndex++) {
    const fixture = golden.cases[caseIndex];
    const wasmRecord = wasmRecords[caseIndex];
    const gpuRecord = gpuRecords[caseIndex];
    if (wasmRecord?.caseId !== fixture.id || gpuRecord?.caseId !== fixture.id) {
      invalid(`case ${caseIndex} order or id mismatch: expected ${fixture.id}, got WASM ${wasmRecord?.caseId} and WebGPU ${gpuRecord?.caseId}`);
    }
    const questions = fixture.questions;
    const expected = fixture.expect;
    const questionIds = Object.keys(questions || {});
    const expectedIds = Object.keys(expected || {});
    if (!questionIds.length || !sameKeys(questionIds, expectedIds)) invalid(`${fixture.id}: fixture questions and golden outputs differ`);
    const wasmAnswers = wasmRecord.answers;
    const gpuAnswers = gpuRecord.answers;
    const wasmIds = Object.keys(wasmAnswers || {});
    const gpuIds = Object.keys(gpuAnswers || {});
    if (!sameKeys(wasmIds, questionIds)) invalid(`${fixture.id}: WASM answers do not match fixture questions`);
    if (!sameKeys(gpuIds, questionIds)) invalid(`${fixture.id}: WebGPU answers do not match fixture questions`);

    const entries = [];
    for (const questionId of questionIds) {
      const question = questions[questionId];
      const expectedAnswer = expected[questionId];
      const keys = optionKeys(question, expectedAnswer, fixture.id, questionId);
      const reference = goldenVector(golden, question, expectedAnswer, fixture.id, questionId);
      const wasm = publicVector(wasmAnswers[questionId], question.type, keys, fixture.id, questionId, "WASM");
      const gpu = publicVector(gpuAnswers[questionId], question.type, keys, fixture.id, questionId, "WebGPU");
      const backendDp = maxDelta(wasm, gpu);
      const wasmGoldenDp = maxDelta(wasm, reference);
      const gpuGoldenDp = maxDelta(gpu, reference);
      const backendArgmax = argmax(wasm) === argmax(gpu);
      const wasmGoldenArgmax = argmax(wasm) === argmax(reference);
      const gpuGoldenArgmax = argmax(gpu) === argmax(reference);
      maxBackendDp = Math.max(maxBackendDp, backendDp);
      maxWasmGoldenDp = Math.max(maxWasmGoldenDp, wasmGoldenDp);
      maxGpuGoldenDp = Math.max(maxGpuGoldenDp, gpuGoldenDp);
      backendArgmaxMatches += Number(backendArgmax);
      wasmGoldenArgmaxMatches += Number(wasmGoldenArgmax);
      gpuGoldenArgmaxMatches += Number(gpuGoldenArgmax);
      questionCount++;
      entries.push({
        id: questionId,
        type: question.type,
        options: keys,
        wasm,
        webgpu: gpu,
        golden: reference,
        maxDp: { wasmWebgpu: backendDp, wasmGolden: wasmGoldenDp, webgpuGolden: gpuGoldenDp },
        argmax: { wasmWebgpu: backendArgmax, wasmGolden: wasmGoldenArgmax, webgpuGolden: gpuGoldenArgmax },
      });
    }
    cases.push({ id: fixture.id, questions: entries });
  }

  const allArgmax = (count) => count === questionCount;
  const failures = [];
  if (maxBackendDp > LAYA_WASM_GPU_MAX_DP) failures.push(`WASM/WebGPU maxDp ${maxBackendDp} exceeds ${LAYA_WASM_GPU_MAX_DP}`);
  if (maxWasmGoldenDp > LAYA_Q8_GOLDEN_MAX_DP) failures.push(`WASM/Q8 golden maxDp ${maxWasmGoldenDp} exceeds ${LAYA_Q8_GOLDEN_MAX_DP}`);
  if (maxGpuGoldenDp > LAYA_Q8_GOLDEN_MAX_DP) failures.push(`WebGPU/Q8 golden maxDp ${maxGpuGoldenDp} exceeds ${LAYA_Q8_GOLDEN_MAX_DP}`);
  if (!allArgmax(backendArgmaxMatches)) failures.push(`WASM/WebGPU argmax agreement ${backendArgmaxMatches}/${questionCount}`);
  if (!allArgmax(wasmGoldenArgmaxMatches)) failures.push(`WASM/Q8 golden argmax agreement ${wasmGoldenArgmaxMatches}/${questionCount}`);
  if (!allArgmax(gpuGoldenArgmaxMatches)) failures.push(`WebGPU/Q8 golden argmax agreement ${gpuGoldenArgmaxMatches}/${questionCount}`);

  return {
    ok: failures.length === 0,
    precision: "q8",
    questionCount,
    tolerances: {
      wasmWebgpuMaxDp: LAYA_WASM_GPU_MAX_DP,
      q8GoldenMaxDp: LAYA_Q8_GOLDEN_MAX_DP,
      publicProbabilityDecimals: LAYA_API_PROBABILITY_DECIMALS,
    },
    maxDp: { wasmWebgpu: maxBackendDp, wasmGolden: maxWasmGoldenDp, webgpuGolden: maxGpuGoldenDp },
    argmaxAgreement: {
      wasmWebgpu: { matched: backendArgmaxMatches, total: questionCount },
      wasmGolden: { matched: wasmGoldenArgmaxMatches, total: questionCount },
      webgpuGolden: { matched: gpuGoldenArgmaxMatches, total: questionCount },
    },
    checks: {
      cases: golden.cases.length,
      options: "fixture keys and complete probability vectors validated",
      probabilities: "finite, within [0,1], and normalized within four-decimal API rounding",
      wasmBackendTolerance: LAYA_WASM_GPU_MAX_DP,
      q8GoldenAccuracyTolerance: LAYA_Q8_GOLDEN_MAX_DP,
    },
    failures,
    cases,
  };
}
