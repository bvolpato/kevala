// Private preparation only. Importing this file does not load a model or run a benchmark.
const WORDS = "the customer wrote in about a recurring charge they did not recognise and wants a refund today ";
const question = () => ({ type: "noul", instructions: "Is the customer asking for money back?" });
export const longRequests = () => [{ state: `Run 0. ${WORDS.repeat(30)}`, questions: { q: question() } }];
export const mediumRequests = () => [{ state: `Run 0. ${WORDS.repeat(6)}`, questions: { q: question() } }];
export const quartets = Object.freeze(["ABBA", "BAAB", "ABBA", "BAAB", "ABBA", "BAAB", "ABBA", "BAAB", "ABBA"]);
const check = (condition, message) => { if (!condition) throw new Error(message); };

function completeOutputs(responses) {
  check(Array.isArray(responses) && responses.length === 1, "Expected one complete request response");
  const encode = (value) => {
    if (typeof value === "number") check(Number.isFinite(value), "Non-finite complete request output");
    if (Array.isArray(value)) return value.map(encode);
    if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map((key) => [key, encode(value[key])]));
    return value;
  };
  return JSON.stringify(responses.map(({ timing, ...output }) => encode(output)));
}

/** `run(requests, { beforeDecide })` calls beforeDecide with the actual requests immediately
 * before the existing loaded engine's decide call. Load once with both caches false.
 * `profilePaired` uses the existing profiler on this same instance, outside request timing.
 * It must retain the individual baseline mm.wi/geglu and candidate paired-kernel samples.
 */
export async function runLayaGeGluProtocol({ trunk, run, profilePaired, loadOptions }) {
  check(loadOptions?.cache === false && loadOptions?.stateCache === false, "Both caches must be disabled at load");
  check(!trunk.profiler, "Unprofiled request timing requires profiling off");
  check(typeof profilePaired === "function", "A separate paired-kernel profiling phase is required");
  const deadline = performance.now() + 600000;
  const bounded = async (work, label, capMs = 30000) => {
    const remaining = Math.min(capMs, deadline - performance.now());
    check(remaining > 0, "Private Laya GeGLU campaign deadline exceeded");
    let timer;
    try {
      return await Promise.race([work(), new Promise((_, reject) => {
        timer = setTimeout(() => reject(Object.assign(new Error(`Private Laya GeGLU ${label} deadline exceeded`), { code: "LAYA_GEGLU_DEADLINE" })), remaining);
      })]);
    } finally { clearTimeout(timer); }
  };
  const configure = (mode) => trunk.configureLayaGeGluPrototype({ mode });
  const requestsLong = longRequests(), requestsMedium = mediumRequests();
  const result = { protocol: "private-laya-geglu-v2", status: "incomplete", warmups: [], aa: [], quartets: [], validation: [] };
  const runWithPermit = async (requests, { role = "long:first", supplyPermit = true } = {}) => {
    trunk.layaGeGluPermit = null;
    let called = false;
    try {
      const response = await run(requests, { beforeDecide(actualRequests) {
        check(!called, "Request adapter invoked beforeDecide more than once");
        called = true;
        check(JSON.stringify(actualRequests) === JSON.stringify(requests), "Request adapter changed the actual requests");
        if (supplyPermit) trunk.armLayaGeGluPrototypeRequest({ role, requests: actualRequests });
      } });
      check(called, "Request adapter did not invoke beforeDecide with its actual requests");
      check(trunk.layaGeGluPermit === null, "Forward did not consume the one-shot request permit");
      return response;
    } finally { trunk.layaGeGluPermit = null; }
  };
  const execute = async (mode, request = requestsLong, context) => {
    await bounded(() => configure(mode), "configuration");
    const response = await bounded(() => runWithPermit(request, context), "request");
    return { response, output: completeOutputs(response), eligibility: { ...trunk.layaGeGlu?.lastEligibility } };
  };
  try {
    const baseline = await execute("off");
    result.referenceOutputs = JSON.parse(baseline.output);
    check(baseline.response[0].timing?.tokens === 512, "Pinned LONG request did not produce active T512");
    const diagnostic = await execute("validate");
    check(diagnostic.output === baseline.output, "Diagnostic complete request output differs");
    const counters = trunk.layaGeGlu.lastValidation;
    check(counters?.ok && counters.layers.length === 28, "Missing exact per-layer diagnostic coverage");
    result.validation.push({ request: "long:first", ...counters });
    const candidate = await execute("on");
    check(candidate.eligibility.enabled && candidate.output === baseline.output, "Candidate complete request output differs");
    for (const context of [{ supplyPermit: false }, { role: "wrong-role" }]) {
      const fallback = await execute("on", requestsLong, context);
      check(fallback.response[0].timing?.tokens === 512 && fallback.eligibility.enabled === false && fallback.output === baseline.output, "Missing or wrong-role permit did not fall back at T512");
      result.validation.push({ request: "long:first:permit-fallback", context, completeOutputExact: true, eligibility: fallback.eligibility });
    }
    const wrongRequests = longRequests();
    wrongRequests[0].state = wrongRequests[0].state.replace("Run 0. ", "Run 1. ");
    const wrongA = await execute("off", wrongRequests), wrongB = await execute("on", wrongRequests);
    check(wrongB.response[0].timing?.tokens === 512 && wrongB.eligibility.enabled === false && wrongA.output === wrongB.output, "Wrong-request permit did not fall back at T512");
    result.validation.push({ request: "long:wrong-request", completeOutputExact: true, eligibility: wrongB.eligibility });
    const mediumContext = { role: "medium:fallback" };
    const mediumA = await execute("off", requestsMedium, mediumContext), mediumB = await execute("on", requestsMedium, mediumContext);
    check(mediumB.response[0].timing?.tokens !== 512 && mediumB.eligibility.enabled === false, "MEDIUM did not retain the original path");
    check(mediumA.output === mediumB.output, "MEDIUM fallback complete request output differs");
    result.validation.push({ request: "medium:fallback", completeOutputExact: true, eligibility: mediumB.eligibility });
    result.tuningKey = trunk.layaGeGlu.tuningKey;
    const measure = async (lane) => {
      await bounded(() => configure(lane === "A" ? "off" : "on"), "configuration");
      check(!trunk.profiler, "Profiler must remain off during request timing");
      const begin = performance.now();
      const response = await bounded(() => runWithPermit(requestsLong), "request");
      const wallMs = performance.now() - begin;
      // Comparison and recording are outside the measured request interval.
      check(completeOutputs(response) === baseline.output, "Measured complete request output differs");
      check(response[0].timing?.tokens === 512, "Measured request active T changed");
      if (lane === "B") check(trunk.layaGeGlu.lastEligibility?.enabled === true, "Measured candidate did not consume a valid request permit");
      return { lane, wallMs, timing: response[0].timing };
    };
    for (const lane of "ABABAB") result.warmups.push(await measure(lane));
    for (const [round, order] of quartets.entries()) {
      const aa = [];
      for (const lane of "AA") aa.push(await measure(lane));
      result.aa.push({ round, samples: aa });
      const samples = [];
      for (const lane of order) samples.push(await measure(lane));
      result.quartets.push({ round, order, samples });
    }
    // Profiler timing is a separate phase, with the same fixed tuning and exact input.
    result.pairedKernel = await bounded(() => profilePaired({ trunk, requests: requestsLong, configure, run: () => runWithPermit(requestsLong), warmups: 3, quartets, aaPairs: 9 }), "paired-kernel phase", 600000);
    check(!trunk.profiler, "Paired-kernel phase must restore profiling off");
    const after = await execute("validate");
    check(after.output === baseline.output && trunk.layaGeGlu.lastValidation?.ok, "Post-timing exact validation failed");
    result.validation.push({ request: "long:first:after", ...trunk.layaGeGlu.lastValidation });
    check(trunk.layaGeGlu.tuningKey === result.tuningKey, "Tuning state changed during campaign");
    result.status = "complete";
    return result;
  } catch (error) {
    result.status = "failed";
    result.error = { message: error.message, code: error.code || null };
    throw Object.assign(error, { partialResult: result });
  } finally { await configure("off"); }
}
