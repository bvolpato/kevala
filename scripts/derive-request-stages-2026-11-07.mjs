#!/usr/bin/env node
import { createHash } from "node:crypto";
import { mkdir, open, readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { summarizeProfileRuns } from "./lib/model-profile.mjs";

const ROOT = resolve(fileURLToPath(new URL("..", import.meta.url)));
const CAPTURE_ID = "profile-ci73-all11-q8-chromium";
const CAPTURE_DIR = "/tmp/kevala-profile-4e27753-20261107/tmp/" + CAPTURE_ID;
const CAPTURE_SUMMARY = resolve(CAPTURE_DIR, "summary.json");
const REPORT_PATH = resolve(ROOT, "docs/benchmarks/kernel-profile-2026-11-07/summary.json");
const OUTPUT_PATH = resolve(ROOT, "docs/benchmarks/kernel-profile-2026-11-07/request-stages.json");
const REDUCER_PATH = resolve(ROOT, "scripts/lib/model-profile.mjs");
const CAPTURE_SUMMARY_SHA256 = "611af8053081b347142f7e21d8c5667b9cc8db5679b0ced62d402efa7d5ae751";
const SOURCE_REPORT_SHA256 = "ad27f87f7d006c748b67908d6516878ef05064de695f53fb09365bdac67088ae";
const RUNTIME_COMMIT = "4e27753e6f385d6fabbb6040fdd82f058a9c48a8";
const REDUCER_SHA256 = "cdd9c9b986adbabdaa17a01d1dfcf1b245dc57e220c161ce4d474ff6bacfefff";
const RAW_SUFFIXES = ["0-wall.json", "0-gpu.json", "1-gpu.json", "1-wall.json"];
const RAW_SHA256_BY_MODEL = {
  "laya": ["c572c56a44461b35dbfc78c6e879de985542eedc8b5c9b02d42eb82cff284176", "1826eba0e2dbe09ab94d8e1cc50591804b03f909b346bf924f26e40c2931b85c", "f6347d5d39fe6f9936e3d39d4ee46d4de34b037689004f25c080bc986687d0fa", "01e600287f3f542bd78d39940954ee2032c1fb952d7267685c269cb709cea0df"],
  "kev-0.8b": ["3940eb6685caef10372ba52745f7c33db016e6af2a85766377caf554fca3fea2", "1f76a1c32ff347de0b222ed23722b05c4986b65817142a9e3e94570b061c3423", "6561c7cba8616b2862dfdf86a42846a104592a391725e5f12165449165dc4bd3", "ef486b5ae9653f25234efaa8285fe84a2be7264cf0a395f83fe218c98cefc082"],
  "kev-4b": ["f511215add4fdcb598684979dabae6a18fbe6bd6dfc3b5db0b31695f8b5655dc", "7657937eda65133d1e3f63591b99580f89f3ff965c3b8b3e936360108caa6871", "f2cf37d0d121f151b2dcc188599c64a5fb6896b86b66645d0c351416313ec52e", "90023c7c3cb1c952a0100c14bbca42622a890896e79a4d4f5676e96f5fc49834"],
  "kev-9b": ["dd00140857279ab587813d23b55090cf3fd57db1b9fb0d617cf5c25bf4a7ad3b", "85553b44e7a92efacfb9ade86708968f33fec989e958c472f08790e39b330da3", "acad07d13e4e42c268281637cf39ff849ef4fb72dae00976c554fa2a64dd4c8b", "8f8cc868306c1eec9e56f338839f7af28110d69e9a4ae8c6bd441aa18058e021"],
  "h2o-lightning-4b": ["8275058dae06a2dfcd20d99fa0b752e496501862b40afb0e5062964efd805683", "c4a588d255e44a201e76a5a04c5b28b92bcc7614bbceb6e6f2c05e89d0e1accc", "77460632cc621f501be12c252255a7c1de92840ee7ededc89131343033645463", "c0992473b60d6331a9817104c72a22373eb519a6bb2b93761fa3ba0f3dfe0378"],
  "clef-flash": ["a93bea3846e4c9dd3e8c54a14e5609a803221c75228d77007f306a44b04def8d", "5e2f8668b097649742f6196ef8255cbd5d76559ab80ef467d079dd6e900ba8c2", "6acc2bc6a68f69a99f6d2cd66b579baaaf796113a41ba3790a1b5cdbaf5a8382", "b6a539290bc6fed6a338561c4b3dd108cb7834d8bd1b64c3003c0ce5c6466141"],
  "semif-qwen3.5-0.8b": ["1ed69a5c588b68ef806778990f2887a654753562b9195386c6e9d346aee90222", "5b4d60384181e2dadf569ba50bea7dbb0715d3496c317bd3b9bb6715c21b86d1", "209ee2d87ce15465a30cbb5612c7f15ef624f3a797dfb4482ab9b312b1a1be4e", "a7719f4d07275fb27e75b4e4f95be744bc7163da25e8008fb1c43250912386c9"],
  "semif-qwen3.5-2b": ["d68a5e832a25da19034e848663d545fe803b47ef343f8f44dddec03b45950d43", "f1990fe0bd8f778844b5864b4ccecd597fe9ff49f62a8236ec0e065323e2be86", "79fbd969f2204234b0189fb5d5087c37c8cb0f803c007e9f1f5b3bfd9fc3c27a", "a3de5179375e10e4a4b171f6b4fe8a01fa60ae93862a278afca00c7931af0e1b"],
  "semif-qwen3.5-4b": ["66caea931f2f535267fe198d2f99969a3eaafc57ae1afa43a7174988beee5a70", "16f3582f432a1d020c7cb19a50961cf3244652c651fd63e33c760f99f79d2476", "336fcc10adfbfeec94cc85c02b586ee2c0ab2c8d46f59d63f717d0a1f49926f6", "328ce8f41141cb6001db746f8178c43c72c82fdd20561f7925f6ad97a2ae44ec"],
  "gemma-4-e2b": ["a3a38407e1f4553ee4baa3949debedabf1dd7bcbe4dd18fb3990ab2866e62e2a", "17205d875f9692ab1b5067965e117ee821bb71504bd00c4a3ceafa17dfc10044", "03d02781edfc9b4a7798905935a810c974a811077feb5fb709084a723bf8f9c9", "1efc28878a023a3dc9a6db515bfe101ad4190ae98c45ffa8ae2fdc01e7e644da"],
  "gemma-4-e4b": ["54e7974164b8342e93e419bec7ec31894a8dc827170d67e00b0694515c8a341c", "90e87233187eac338449982b1ccccaca2ab4106f7f599f7bad2a858cbfbb2d92", "e6ac861b263c3fc2f9d42d5549203f53a310742c4d10e78e46b4cc2a08884cfb", "5feaed9e33c75e5448d0900e5772c72764363f4f0ca42a22951d66a750687ca7"],
};

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const check = (condition, message) => { if (!condition) throw new Error(message); };
const key = (row) => JSON.stringify([row.id, row.precision, row.browser]);

async function immutableWrite(path, bytes) {
  await mkdir(dirname(path), { recursive: true });
  let handle;
  try {
    handle = await open(path, "wx");
    await handle.writeFile(bytes);
    await handle.sync();
  } catch (error) {
    if (error.code !== "EEXIST") throw error;
    const existing = await readFile(path);
    check(existing.equals(bytes), "request-stage supplement already exists with different bytes");
  } finally {
    await handle?.close();
  }
}

const [captureBytes, reportBytes, reducerBytes] = await Promise.all([
  readFile(CAPTURE_SUMMARY), readFile(REPORT_PATH), readFile(REDUCER_PATH),
]);
check(sha256(captureBytes) === CAPTURE_SUMMARY_SHA256, "Q8 capture summary differs from the pinned input");
check(sha256(reportBytes) === SOURCE_REPORT_SHA256, "published model/kernel summary differs from the pinned input");
check(sha256(reducerBytes) === REDUCER_SHA256, "request-stage reducer differs from the audited source");
const capture = JSON.parse(captureBytes), report = JSON.parse(reportBytes);
check(capture.runtime?.commit === RUNTIME_COMMIT, "Q8 capture runtime commit differs from the pinned input");
check(capture.protocol?.precisions?.length === 1 && capture.protocol.precisions[0] === "q8"
  && capture.protocol.browsers?.length === 1 && capture.protocol.browsers[0] === "chromium"
  && capture.protocol.blocks === 2 && capture.protocol.samples === 9, "Q8 capture protocol differs from the pinned input");
check(Array.isArray(capture.models) && capture.models.length === Object.keys(RAW_SHA256_BY_MODEL).length, "Q8 capture model count differs from the pinned set");
check(report.runtime?.commit === RUNTIME_COMMIT && Array.isArray(report.models), "published report runtime identity differs from the pinned capture");

const models = [];
for (const input of capture.models) {
  const rawHashes = RAW_SHA256_BY_MODEL[input.id];
  check(rawHashes && input.precision === "q8" && input.browser === "chromium" && input.status === "measured", "unexpected Q8 capture model row");
  const expectedRaw = RAW_SUFFIXES.map((suffix) => "chromium/" + input.id + "-q8/" + suffix);
  check(JSON.stringify(input.raw) === JSON.stringify(expectedRaw), input.id + ": raw source list differs from the pinned order");
  const published = report.models.find((row) => key(row) === key(input));
  check(published && published.status === "measured" && Array.isArray(published.cases)
    && Array.isArray(published.evidence) && published.evidence.length === expectedRaw.length, input.id + ": published row or raw evidence references are missing");
  const runs = [];
  const rawEvidence = [];
  for (const [index, source] of expectedRaw.entries()) {
    const expectedHash = rawHashes[index], evidence = published.evidence[index];
    check(evidence?.sha256 === expectedHash && typeof evidence.url === "string" && Number.isSafeInteger(evidence.bytes)
      && typeof evidence.gzipSha256 === "string" && Number.isSafeInteger(evidence.gzipBytes), input.id + ": published raw identity differs from its pinned hash");
    const bytes = await readFile(resolve(CAPTURE_DIR, source));
    check(sha256(bytes) === expectedHash, input.id + ": raw JSON differs from its pinned SHA-256 for " + source);
    const result = JSON.parse(bytes);
    const match = /\/([01])-(wall|gpu)\.json$/.exec(source);
    check(match && result.model === input.id && result.precision === "q8" && result.profile === (match[2] === "gpu"),
      input.id + ": raw run identity differs for " + source);
    runs.push({ block: Number(match[1]), profile: match[2] === "gpu", result });
    rawEvidence.push({ source, url: evidence.url, sha256: evidence.sha256, bytes: evidence.bytes,
      gzipSha256: evidence.gzipSha256, gzipBytes: evidence.gzipBytes });
  }
  const derived = summarizeProfileRuns(runs, { shapes: runs[0].result.cases.map((_, index) => index) });
  check(derived.cases.length === published.cases.length, input.id + ": derived case count differs from the published row");
  const cases = derived.cases.map((testCase, index) => {
    check(testCase.name === published.cases[index].name && testCase.requestStages, input.id + ": derived case identity or stage data is missing");
    return { name: testCase.name, requestStages: testCase.requestStages };
  });
  models.push({ id: input.id, precision: input.precision, browser: input.browser, rawEvidence, cases });
}

const supplement = {
  schemaVersion: 1,
  sourceReportSha256: SOURCE_REPORT_SHA256,
  sourceCapture: { id: CAPTURE_ID, runtimeCommit: RUNTIME_COMMIT, summarySha256: CAPTURE_SUMMARY_SHA256 },
  reducerSha256: REDUCER_SHA256,
  models,
};
const output = Buffer.from(JSON.stringify(supplement, null, 2) + "\n");
await immutableWrite(OUTPUT_PATH, output);
process.stdout.write(JSON.stringify({ output: OUTPUT_PATH, sourceReportSha256: SOURCE_REPORT_SHA256,
  models: models.length, rawJsonFiles: models.reduce((total, model) => total + model.rawEvidence.length, 0),
  bytes: output.length, sha256: sha256(output) }) + "\n");
