#!/usr/bin/env node
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { lstat, readdir, readFile, realpath, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync, gunzipSync } from "node:zlib";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../../..");
const campaign = "/home/bruno/githubworkspace/kevala/tmp/sota-20261106-kdDTGn/campaign";
const modelRun = join(campaign, "avx512-auto4-model-followup-r3-phase-r1");
const modelData = join(modelRun, "models-v1");
const gateRun = join(campaign, "avx512-auto4-model-cache-r2/final-identity-r1");
const budgetFailure = join(campaign, "avx512-auto4-model-v1/gate-budget-r1");
const candidateRoot = "/tmp/kevala-private-avx512-auto4-model-v1";
const productionRoot = join(campaign, "avx512-production-zen5-v1");
const productionModelRun = join(productionRoot, "models-launch-r2");
const productionModelData = join(productionModelRun, "models-v1");
const productionGateRun = join(productionRoot, "gates-launch-r2");
const productionGateData = join(productionGateRun, "gates-v1");
const baselineCommit = "8e4ae3a3ff6592e9581696057f07fa90e036e10a";
const productionBaselineCommit = "dae1c95ff27d6dc3176f1d64dd424b2fb3cadac4";
const lockPath = "/home/bruno/.cache/kevala/gpu-profile.lock";
const archivePath = join(here, "evidence.json.gz");
const manifestPath = join(here, "manifest.json");
const mode = process.argv[2];
const hash = data => createHash("sha256").update(data).digest("hex");
const sortPaths = items => items.sort((a, b) => a < b ? -1 : a > b ? 1 : 0);

const inventoryPin = {
  relativePath: "models-inventory-v1.json",
  bytes: 498671,
  sha256: "04b787a1561d24b9e72f1b64a3318c1a21043d7faa2dee23436f00102eb5018d",
};
const inventoryTotals = { files: 3067, bytes: 48953846, json: 2457, stdout: 305, stderr: 305 };
const productionInventoryPin = {
  relativePath: "models-inventory-v1.json",
  bytes: 499669,
  sha256: "2a5280aed47f4bdf4489771ffecaf3a8ecd1ead9f10f5b684cd5c9d5c5fa995a",
};
const productionInventoryTotals = { files: 3073, bytes: 49475274, json: 2461, stdout: 306, stderr: 306 };
const productionGateInventoryPin = {
  relativePath: "gates-inventory-v1.json",
  bytes: 109066,
  sha256: "313d37d403330964851671fb00d11e8b2e82d6373f0cb2967f79066951b50ad2",
};
const productionGateInventoryTotals = { files: 768, bytes: 222617309, json: 542, stdout: 113, stderr: 113 };
const productionGateSelectionTotals = { files: 741, bytes: 20370701, stdout: 86, omittedStdoutFiles: 27, omittedStdoutBytes: 202246608 };
const maxGateStdoutBytes = 1024 * 1024;

const modelInventories = [
  { name: "prototype", run: modelRun, dataRoot: modelData, archivePrefix: "evidence/prototype-model-run/models-v1", inventoryArchivePath: "evidence/prototype-model-run/models-inventory-v1.json", pin: inventoryPin, totals: inventoryTotals },
  { name: "production", run: productionModelRun, dataRoot: productionModelData, archivePrefix: "evidence/production-model-run/models-v1", inventoryArchivePath: "evidence/production-model-run/models-inventory-v1.json", pin: productionInventoryPin, totals: productionInventoryTotals },
];

function pin(path, root, relativePath, bytes, sha256) {
  const absoluteRoot = resolve(root);
  const absolutePath = resolve(absoluteRoot, relativePath);
  const rel = relative(absoluteRoot, absolutePath);
  assert(rel && !rel.startsWith("..") && !isAbsolute(rel), `pin escapes its source root: ${path}`);
  return { path, root: absoluteRoot, relativePath: rel, absolutePath, bytes, sha256 };
}

const fixedPins = [
  pin("evidence/prototype-model-run/plan.json", modelRun, "plan.json", 108571, "70347014438f329765951a1b56c9eef08bebe538dd78195da78fc60dbed4c4b6"),
  pin("evidence/prototype-model-run/models-inventory-v1.json", modelRun, inventoryPin.relativePath, inventoryPin.bytes, inventoryPin.sha256),
  pin("evidence/prototype-model-run/models-release-proof-v1.json", modelRun, "models-release-proof-v1.json", 738, "69f3d68d8688ce593f7d1537584af97518dba946cf66f6d740c2322deeca83ce"),
  pin("evidence/prototype-model-run/source-index.json", modelRun, "source-index.json", 3079, "1a963e05810a010862a2820ecbef550320a38b16d6bc68e154421c114e3fa5a3"),
  pin("evidence/prototype-model-run/source-validation.json", modelRun, "source-validation.json", 6598, "2983e9d2ccdf65165235683b038b4690abf3aaa292d79ad089ff86f706bfee81"),
  pin("evidence/prototype-model-run/run-models.mjs", modelRun, "run-models.mjs", 10030, "ff67bd1f430ca1be3842438c712bc054e0701b40a41a2f2e1b431e3fbbaabd76"),
  pin("evidence/prototype-model-run/common.mjs", modelRun, "common.mjs", 7062, "c0c46fef835d5ed1d46faf05db07cb9b198917589075c1b07cc44f6ee29bc4e4"),
  pin("evidence/prototype-model-run/control.mjs", modelRun, "control.mjs", 5529, "1f8cacbbfc307a387744e00c0dae90d5d645d34d2364e0b9e6cfb0a90649846e"),
  pin("evidence/prototype-model-run/exec-cli.mjs", modelRun, "exec-cli.mjs", 461, "8a76fb66347633dfa506a7204b3670a02f5a3684a69ff845f581a1a1ed72bc13"),
  pin("evidence/prototype-model-run/freeze.mjs", modelRun, "freeze.mjs", 12020, "2f384cd69c9048d4457456cf775f5716421cd56afe66e8c022f7c22d6ade4881"),
  pin("evidence/prototype-model-run/schedule.json", modelRun, "schedule.json", 56141, "a5e24d5831988ee2be67a026b4f154713e7e8d14ce375ab1348b72d44f316ba9"),
  pin("evidence/prototype-model-run/schedule.mjs", modelRun, "schedule.mjs", 3129, "afa7ea7da8332c2495f17b0d327804350b02d42991ef9aeb80ee08c72c8f7707"),
  pin("evidence/prototype-model-run/verify-release.mjs", modelRun, "verify-release.mjs", 2484, "db9f97dce5f4686f2ab73e49307bd9f0131a2307c894fb956980c46aed91998a"),
  pin("evidence/prototype-model-run/metadata/model.mjs", modelRun, "metadata/model.mjs", 3891, "5539b06d1ec2bb3fe5dae4077ab54574d015b35d6ae8e9439d7cd42dbe61274b"),

  pin("evidence/production-model-run/models-inventory-v1.json", productionModelRun, "models-inventory-v1.json", 499669, "2a5280aed47f4bdf4489771ffecaf3a8ecd1ead9f10f5b684cd5c9d5c5fa995a"),
  pin("evidence/production-model-run/models-release-proof-v1.json", productionModelRun, "models-release-proof-v1.json", 735, "6eef79bccb5c23d880b9396711f6e385f872c8845a540f49bdd5d9568e91d0bb"),
  pin("evidence/production-model-run/source-index.json", productionModelRun, "source-index.json", 3657, "e3f9e4196e82270931e0640a0f0ecc3023475966734c6ccfa1327add6ebad20a"),
  pin("evidence/production-model-run/source-validation.json", productionModelRun, "source-validation.json", 6882, "354088d999355ee01ba4eba9dfad245cf55f350ab14bb7b3e93bacf80ad231fe"),
  pin("evidence/production-model-run/plan.json", productionModelRun, "plan.json", 113673, "334de3fdaa358d59c097aef65f57f000f548e21d43b345e5171d2cdcffe58e32"),
  pin("evidence/production-model-run/finalized-index.json", productionModelRun, "finalized-index.json", 11403, "278df2adfd94d20b0b3e728d952007d2c9b0273d983c1983e6276643b240557d"),
  pin("evidence/production-model-run/run-models.mjs", productionModelRun, "run-models.mjs", 11449, "7274d64dadc5110778bc9dba0732769f0278bf3b1df38509798bc0f925ee3a87"),
  pin("evidence/production-model-run/common.mjs", productionModelRun, "common.mjs", 7099, "171f37fa8a7d45d4da243a1efe783dc8a9e82b8ba6b72d99a1e3f873b584d181"),
  pin("evidence/production-model-run/control.mjs", productionModelRun, "control.mjs", 5529, "1f8cacbbfc307a387744e00c0dae90d5d645d34d2364e0b9e6cfb0a90649846e"),
  pin("evidence/production-model-run/draft-plan.json", productionModelRun, "draft-plan.json", 16357, "d22082c0735adea4281216d40ed9353f51f9d132681f400d04dc6f2bd2aced74"),
  pin("evidence/production-model-run/exec-cli.mjs", productionModelRun, "exec-cli.mjs", 461, "8a76fb66347633dfa506a7204b3670a02f5a3684a69ff845f581a1a1ed72bc13"),
  pin("evidence/production-model-run/finalize.mjs", productionModelRun, "finalize.mjs", 4454, "696a20fdb0b3368dbcc97db088bafd9c6c4ae5484cbb5a1bebf269c99c94d67a"),
  pin("evidence/production-model-run/gate-admission.mjs", productionModelRun, "gate-admission.mjs", 4351, "3ed89f028273c3fe6abd3573ad74d17e26114a8974727df665cb0bd6d5761f31"),
  pin("evidence/production-model-run/gate-admission.test.mjs", productionModelRun, "gate-admission.test.mjs", 4185, "60298b57626118ebd41d62489ad81cfded0d884a162932aa70f3a20960de7bdf"),
  pin("evidence/production-model-run/metadata/model.mjs", productionModelRun, "metadata/model.mjs", 3891, "5539b06d1ec2bb3fe5dae4077ab54574d015b35d6ae8e9439d7cd42dbe61274b"),
  pin("evidence/production-model-run/schedule.json", productionModelRun, "schedule.json", 56141, "a5e24d5831988ee2be67a026b4f154713e7e8d14ce375ab1348b72d44f316ba9"),
  pin("evidence/production-model-run/schedule.mjs", productionModelRun, "schedule.mjs", 3129, "afa7ea7da8332c2495f17b0d327804350b02d42991ef9aeb80ee08c72c8f7707"),
  pin("evidence/production-model-run/verify-release.mjs", productionModelRun, "verify-release.mjs", 2484, "db9f97dce5f4686f2ab73e49307bd9f0131a2307c894fb956980c46aed91998a"),

  pin("evidence/prototype-gate/gates-inventory-v1.json", gateRun, "gates-inventory-v1.json", 41034, "70a5b07e991cb5e4d8cf2e5959c2177f8ca88e8e326193d31ac2e42d1aa7bc1b"),
  pin("evidence/prototype-gate/gates-release-proof-v1.json", gateRun, "gates-release-proof-v1.json", 702, "5501f437833bca5a1b8c7102dc010f127093ed7cd808bf0571089056d8eb5218"),
  pin("evidence/prototype-gate/gates-v1/receipt.json", gateRun, "gates-v1/receipt.json", 340975, "a76d7ca545d5d6426bda8c841d141b2dc578193f07e2d79e24df7af3fdef7af5"),
  pin("evidence/prototype-gate/gates-v1/codegen.json", gateRun, "gates-v1/codegen.json", 36989, "c8abf6753d1c112da5107e4d711fcbb221358d0bec92e08efb1cfd6303ecd40b"),
  pin("evidence/prototype-gate/source-delta-proof.json", gateRun, "source-delta-proof.json", 1772, "11407368935c2a619c6f3bd6dd69b1993d6794c923c717be4e3ae591bde1e531"),
  pin("evidence/prototype-gate/source-index.json", gateRun, "source-index.json", 59729, "9dad9198e56cc932363698f91a0e0e55b7cb74536c52104655a13f1e69df06da"),
  pin("evidence/prototype-gate/source-receipt.json", gateRun, "source-receipt.json", 52499, "dae99ad6c59ea9e7ad75a79a942546967d6d72ef0619c0256b8f4cb3051c3126"),
  pin("evidence/prototype-gate/final-identity-revision.json", gateRun, "final-identity-revision.json", 1096, "9d4abaa6c6fd6b935481c66a0b533375bb6a48e4c4a507fb85169a9b7284bf4f"),
  pin("evidence/prototype-gate/scoped.patch", gateRun, "scoped.patch", 6685, "4f6db097ce034a667e00f809d9e0f89c36f50f685879cc7cec10a5347863509d"),

  pin("evidence/production-gate/gates-inventory-v1.json", productionGateRun, "gates-inventory-v1.json", 109066, "313d37d403330964851671fb00d11e8b2e82d6373f0cb2967f79066951b50ad2"),
  pin("evidence/production-gate/gates-release-proof-v1.json", productionGateRun, "gates-release-proof-v1.json", 734, "9beb847e3e95caad1395e3364444211d49d874c8bea0395a6d523fbc61587a20"),
  pin("evidence/production-gate/baseline-source-equivalence.json", productionGateRun, "baseline-source-equivalence.json", 268, "8a0ca647c2da3def15268706628e011194cf7a45b3e729fecd01e7e65a90ee60"),
  pin("evidence/production-gate/build-root.mjs", productionGateRun, "build-root.mjs", 5885, "916a4a8e714fd368a59f51f57ce5554122c014db2470e69660d6ea3a9cc71734"),
  pin("evidence/production-gate/cargo-home-guard.patch", productionGateRun, "cargo-home-guard.patch", 2410, "847c6ff32fd21594092ebe093b0828d925a1126bfb944199bfd82da19d02b601"),
  pin("evidence/production-gate/cargo-home.test.mjs", productionGateRun, "cargo-home.test.mjs", 2229, "15870c72102d44c0b6339d47e962dee26e2c71a4224206e248416268dae967df"),
  pin("evidence/production-gate/codegen.mjs", productionGateRun, "codegen.mjs", 7514, "61cdfef70d6024abea80669bd3730a9066a78abd17f87d38cb937a4a3adddfa3"),
  pin("evidence/production-gate/command-record.test.mjs", productionGateRun, "command-record.test.mjs", 4105, "328172944aff744553c49acd24d41fa743ccf726e7fb85ab9684d1a1af99a202"),
  pin("evidence/production-gate/common.mjs", productionGateRun, "common.mjs", 7492, "049d46de6cba51f54d057d5b947834d3bafe193df52800082a47c1c0df5f28bb"),
  pin("evidence/production-gate/control.mjs", productionGateRun, "control.mjs", 5529, "1f8cacbbfc307a387744e00c0dae90d5d645d34d2364e0b9e6cfb0a90649846e"),
  pin("evidence/production-gate/mock-rustc.sh", productionGateRun, "mock-rustc.sh", 53, "34d758cdda7a7ac983f3263b5de7d95a4ded32359921c8dca2a5d44e92a6fccb"),
  pin("evidence/production-gate/parse-cargo.mjs", productionGateRun, "parse-cargo.mjs", 737, "f627e9acf39b3bcceda39df091fbcbd7c96d3edd266aeb25d4b9abc8324b7084"),
  pin("evidence/production-gate/plan.json", productionGateRun, "plan.json", 162531, "4cc97041322d27e4530ff0e727d8a199f6faa7b05c82c1b4f302e086e340c01f"),
  pin("evidence/production-gate/run-gates.mjs", productionGateRun, "run-gates.mjs", 15318, "15b70b1ac024e8a3ab65666a878353525703a078bfbe85b697e2779a160a4231"),
  pin("evidence/production-gate/rust189-installed.json", productionGateRun, "rust189-installed.json", 1325, "bea87303c477666c4dc8be4aa026451e65a8d99861b4d6acdaac42933722a734"),
  pin("evidence/production-gate/source-contract.test.mjs", productionGateRun, "source-contract.test.mjs", 2020, "18085042331a7701983fe61a3f989350bcc6437eab4c518ac4c58b88c21a26b0"),
  pin("evidence/production-gate/source-index.json", productionGateRun, "source-index.json", 59825, "06abef2ea8846d719f7eb5b48cfc143ccf8ac09fd0bdd67325fa8a712319c919"),
  pin("evidence/production-gate/source-validation.json", productionGateRun, "source-validation.json", 6109, "b5a2680738eedd3d5a96ca6e71a82acd25b7cd6ac0eaff74e783e56b1f702d34"),
  pin("evidence/production-gate/verify-release.mjs", productionGateRun, "verify-release.mjs", 2480, "8592d2f9c3be80232f276316c1d669bd2cfa000c74b6945fbeaf3f01278f234e"),
  pin("evidence/production-gate/guard/adapter.rs", productionGateRun, "guard/adapter.rs", 2049, "41d66bfba0c5cd508726f0b91e94fd28e2c6c5abba126ec5ad9b9a708e755111"),
  pin("evidence/production-gate/guard/main.rs", productionGateRun, "guard/main.rs", 633, "56d8723fa1772067edae533ef6004f70654686fc2b6a9db9910de286099634ee"),
  pin("evidence/production-gate/metadata/child_metadata.rs", productionGateRun, "metadata/child_metadata.rs", 9294, "c4836d697048b419443eebddcc2117bde49c193673e43b576d7a5e3ee8982a29"),
  pin("evidence/production-gate/frozen/baseline/Cargo.lock", productionRoot, "gates-launch-r2/frozen/baseline/Cargo.lock", 313, "79c2cf708cd77b01490375152c37d488c4bfb100ad18c72f2d6bebd852c7a2d7"),
  pin("evidence/production-gate/frozen/baseline/Cargo.toml", productionRoot, "gates-launch-r2/frozen/baseline/Cargo.toml", 598, "e788484b57ae48aab33749a061a1e41cde15b0bbcb519855a8d2a1faf8eb4fcd"),
  pin("evidence/production-gate/frozen/baseline/rustfmt.toml", productionRoot, "gates-launch-r2/frozen/baseline/rustfmt.toml", 45, "93ed1668bb2ca659e88711456b72320bf2f53369ecf278c4a61af0cb63cc3a1e"),
  pin("evidence/production-gate/frozen/candidate/Cargo.lock", productionRoot, "gates-launch-r2/frozen/candidate/Cargo.lock", 313, "79c2cf708cd77b01490375152c37d488c4bfb100ad18c72f2d6bebd852c7a2d7"),
  pin("evidence/production-gate/frozen/candidate/Cargo.toml", productionRoot, "gates-launch-r2/frozen/candidate/Cargo.toml", 598, "e788484b57ae48aab33749a061a1e41cde15b0bbcb519855a8d2a1faf8eb4fcd"),
  pin("evidence/production-gate/frozen/candidate/rustfmt.toml", productionRoot, "gates-launch-r2/frozen/candidate/rustfmt.toml", 45, "93ed1668bb2ca659e88711456b72320bf2f53369ecf278c4a61af0cb63cc3a1e"),

  pin("evidence/prior-failures/incomplete-model-capture-receipt.json", gateRun, "models-v1/receipt.json", 527548, "e2211a8f33ad7cb3ac26fee0ce11a1d12bf9808b1a85b88fafc8a844ea848297"),
  pin("evidence/prior-failures/compile-budget-failure-receipt.json", budgetFailure, "gates-v1/receipt.json", 191410, "c2de5dbb543cfd65c28b47799ad52fc41eebbb01d3b0dc0d6f19518db8372143"),

  pin("publication/docs/native-avx512-performance.md", repo, "docs/native-avx512-performance.md", 7720, "2d445ca597e2277efd243d51061c5248369ecafca09409c674ee548a9e8ff012"),
  pin("publication/docs/native-avx512-dispatch.md", repo, "docs/native-avx512-dispatch.md", 3480, "d8ea7e96bb7ff2d66607a06f2b0eab4ab54bda4f03729c947e51b840ccf6576b"),
  pin("publication/docs/benchmarks/native-avx512-2026-11-07/README.md", repo, "docs/benchmarks/native-avx512-2026-11-07/README.md", 3858, "4fce0d45b53367a4e3b64b994d24ec0499ed115ed01a5bd8fef29020bd18946c"),
  pin("publication/app/views/how.js", repo, "app/views/how.js", 17166, "76a620ef98250e955f8c1b09f1ca15a8b759cb3aba4b472a32292b4ce35f6658"),
];

const sourcePins = [
  {
    path: "evidence/source/prototype-baseline/crates/kevala/src/kernels.rs",
    commit: baselineCommit,
    relativePath: "crates/kevala/src/kernels.rs",
    bytes: 33589,
    sha256: "a1d101ed08726cafb2d7a301a39edf789adfc203044852fd8c7b4cf7cf12b5da",
  },
  pin("evidence/source/prototype-candidate/crates/kevala/src/kernels.rs", candidateRoot, "crates/kevala/src/kernels.rs", 37184, "d43c91d9bc1b90f05d6ec870f90c8b11c5ca5adc8d00bf598eb4145fc2958d0e"),
  pin("evidence/source/prototype-candidate/crates/kevala/build.rs", candidateRoot, "crates/kevala/build.rs", 1540, "443856de805d5d967c2f730cd4f5149b84523adff4181e8868efef999f1f9734"),
  {
    path: "evidence/source/production-baseline/crates/kevala/src/kernels.rs",
    commit: productionBaselineCommit,
    relativePath: "crates/kevala/src/kernels.rs",
    bytes: 33589,
    sha256: "a1d101ed08726cafb2d7a301a39edf789adfc203044852fd8c7b4cf7cf12b5da",
  },
  {
    path: "evidence/source/production-candidate/crates/kevala/src/kernels.rs",
    commit: "f4596f1d239ca84eb18255819f56684a5c5cf086",
    relativePath: "crates/kevala/src/kernels.rs",
    bytes: 39363,
    sha256: "a5c68612e41dcdeccf5b3b3568480bd2b91e3651ea1ba673c0fcc18a35167c6c",
  },
  {
    path: "evidence/source/production-candidate/crates/kevala/build.rs",
    commit: "f4596f1d239ca84eb18255819f56684a5c5cf086",
    relativePath: "crates/kevala/build.rs",
    bytes: 1408,
    sha256: "eb358238d4dde5adee39d69928b6d04521abfd8cd7a54d456e6227e78b9d3b32",
  },
];

const exclusions = [
  "model weights and checkpoint payloads",
  "compiled executables, libraries, and build caches",
  "27 production gate stdout files larger than 1 MiB: 15 full assembly outputs and 12 full numerical-parity outputs",
  "full runtime directories from earlier failed captures",
];

function validateRelativePath(path) {
  assert(typeof path === "string" && path.length > 0, "inventory path must be a nonempty string");
  assert(!isAbsolute(path) && !path.includes("\\"), `inventory path is not a strict relative POSIX path: ${path}`);
  const parts = path.split("/");
  assert(parts.every(part => part && part !== "." && part !== ".."), `inventory path has an unsafe segment: ${path}`);
}

async function checkedFile(location, root, relativePath, bytes, digest, archivePath, includeData) {
  validateRelativePath(relativePath);
  const absoluteRoot = resolve(root);
  const absolutePath = resolve(absoluteRoot, relativePath);
  const rel = relative(absoluteRoot, absolutePath);
  assert.equal(rel, relativePath, `path does not resolve under its fixed root: ${archivePath}`);
  assert.equal(resolve(location), absolutePath, `unexpected source path for ${archivePath}`);
  assert.equal(await realpath(absoluteRoot), absoluteRoot, `source root is not canonical: ${absoluteRoot}`);
  const before = await lstat(absolutePath, { bigint: true });
  assert(before.isFile() && !before.isSymbolicLink(), `source is not a regular file: ${absolutePath}`);
  assert.equal(await realpath(absolutePath), absolutePath, `source path resolves through a symlink: ${absolutePath}`);
  const data = await readFile(absolutePath);
  const after = await lstat(absolutePath, { bigint: true });
  assert.equal(after.dev, before.dev, `source device changed while reading: ${absolutePath}`);
  assert.equal(after.ino, before.ino, `source inode changed while reading: ${absolutePath}`);
  assert.equal(after.size, before.size, `source size changed while reading: ${absolutePath}`);
  assert.equal(after.mtimeNs, before.mtimeNs, `source modified while reading: ${absolutePath}`);
  assert.equal(data.length, bytes, `byte count mismatch for ${archivePath}`);
  assert.equal(hash(data), digest, `SHA-256 mismatch for ${archivePath}`);
  return { path: archivePath, bytes: data.length, sha256: digest, ...(includeData ? { encoding: "base64", data: data.toString("base64") } : {}) };
}

async function checkedPin(spec, includeData) {
  return checkedFile(spec.absolutePath, spec.root, spec.relativePath, spec.bytes, spec.sha256, spec.path, includeData);
}

function checkedGitSource(spec, includeData) {
  const data = execFileSync("git", ["-C", repo, "show", `${spec.commit}:${spec.relativePath}`], { maxBuffer: 2 * 1024 * 1024 });
  assert.equal(data.length, spec.bytes, `byte count mismatch for ${spec.path}`);
  assert.equal(hash(data), spec.sha256, `SHA-256 mismatch for ${spec.path}`);
  return { path: spec.path, bytes: data.length, sha256: spec.sha256, ...(includeData ? { encoding: "base64", data: data.toString("base64") } : {}) };
}

async function loadInventory(spec, label) {
  const data = await readFile(join(spec.run, spec.pin.relativePath));
  assert.equal(data.length, spec.pin.bytes, `${label} inventory byte count changed`);
  assert.equal(hash(data), spec.pin.sha256, `${label} inventory SHA-256 changed`);
  const inventory = JSON.parse(data);
  assert(inventory && typeof inventory === "object" && !Array.isArray(inventory), `${label} inventory must be a path map`);
  const entries = Object.entries(inventory).map(([path, record]) => {
    validateRelativePath(path);
    assert.deepEqual(Object.keys(record).sort(), ["bytes", "sha256"], `unexpected inventory descriptor: ${path}`);
    assert(Number.isSafeInteger(record.bytes) && record.bytes >= 0, `invalid byte count: ${path}`);
    assert(/^[a-f0-9]{64}$/.test(record.sha256), `invalid SHA-256: ${path}`);
    return { path, bytes: record.bytes, sha256: record.sha256 };
  }).sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);
  const counts = { files: entries.length, bytes: entries.reduce((sum, file) => sum + file.bytes, 0), json: 0, stdout: 0, stderr: 0 };
  for (const entry of entries) {
    const extension = entry.path.split(".").at(-1);
    assert(Object.hasOwn(counts, extension), `unexpected model inventory extension: ${entry.path}`);
    counts[extension]++;
  }
  assert.deepEqual(counts, spec.totals, `${label} inventory summary changed`);
  return entries;
}

async function enumerateEvidenceTree(rootPath, label) {
  const root = resolve(rootPath);
  assert.equal(await realpath(root), root, `${label} evidence root must be canonical`);
  const files = [];
  async function visit(directory, prefix) {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const relativePath = prefix ? `${prefix}/${entry.name}` : entry.name;
      validateRelativePath(relativePath);
      const absolutePath = join(directory, entry.name);
      assert(!entry.isSymbolicLink(), `symbolic link in ${label} evidence tree: ${relativePath}`);
      if (entry.isDirectory()) {
        assert.equal(await realpath(absolutePath), absolutePath, `directory resolves through a symlink: ${relativePath}`);
        await visit(absolutePath, relativePath);
      } else {
        assert(entry.isFile(), `unsupported entry in model evidence tree: ${relativePath}`);
        assert.equal(await realpath(absolutePath), absolutePath, `file resolves through a symlink: ${relativePath}`);
        files.push(relativePath);
      }
    }
  }
  const info = await lstat(root, { bigint: true });
  assert(info.isDirectory() && !info.isSymbolicLink(), `${label} evidence root must be a real directory`);
  await visit(root, "");
  return sortPaths(files);
}

async function loadProductionGateInventory() {
  const data = await readFile(join(productionGateRun, productionGateInventoryPin.relativePath));
  assert.equal(data.length, productionGateInventoryPin.bytes, "production gate inventory byte count changed");
  assert.equal(hash(data), productionGateInventoryPin.sha256, "production gate inventory SHA-256 changed");
  const inventory = JSON.parse(data);
  assert(inventory && typeof inventory === "object" && !Array.isArray(inventory), "production gate inventory must be a path map");
  const entries = Object.entries(inventory).map(([path, record]) => {
    validateRelativePath(path);
    assert.deepEqual(Object.keys(record).sort(), ["bytes", "sha256"], `unexpected gate inventory descriptor: ${path}`);
    assert(Number.isSafeInteger(record.bytes) && record.bytes >= 0, `invalid gate byte count: ${path}`);
    assert(/^[a-f0-9]{64}$/.test(record.sha256), `invalid gate SHA-256: ${path}`);
    return { path, bytes: record.bytes, sha256: record.sha256 };
  }).sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);
  const counts = { files: entries.length, bytes: entries.reduce((sum, file) => sum + file.bytes, 0), json: 0, stdout: 0, stderr: 0 };
  for (const entry of entries) {
    const extension = entry.path.split(".").at(-1);
    assert(Object.hasOwn(counts, extension), `unexpected production gate inventory extension: ${entry.path}`);
    counts[extension]++;
  }
  assert.deepEqual(counts, productionGateInventoryTotals, "production gate inventory summary changed");
  const selected = [], omitted = [];
  for (const entry of entries) {
    const extension = entry.path.split(".").at(-1);
    if (extension === "stdout" && entry.bytes > maxGateStdoutBytes) omitted.push(entry);
    else selected.push(entry);
  }
  const selectedSummary = {
    files: selected.length,
    bytes: selected.reduce((sum, file) => sum + file.bytes, 0),
    stdout: selected.filter(file => file.path.endsWith(".stdout")).length,
    omittedStdoutFiles: omitted.length,
    omittedStdoutBytes: omitted.reduce((sum, file) => sum + file.bytes, 0),
  };
  assert.deepEqual(selectedSummary, productionGateSelectionTotals, "production gate text selection changed");
  return { entries, selected, omitted, counts, selectedSummary };
}

async function lease() {
  assert.equal(await realpath(lockPath), lockPath, "canonical profile lock path must resolve to itself");
  const info = await lstat(lockPath, { bigint: true });
  assert(info.isFile() && !info.isSymbolicLink(), "canonical profile lock must be a regular file");
  const major = info.dev >> 8n & 0xfffn | info.dev >> 32n & 0xfffff000n;
  const minor = info.dev & 0xffn | info.dev >> 12n & 0xffffff00n;
  const matches = (await readFile("/proc/locks", "utf8")).split("\n").filter(line => {
    const fields = line.trim().split(/\s+/), device = fields[5]?.split(":");
    const validDevice = device?.length === 3 && /^[a-f0-9]+$/i.test(device[0]) &&
      /^[a-f0-9]+$/i.test(device[1]) && /^\d+$/.test(device[2]);
    return fields[1] === "FLOCK" && fields[2] === "ADVISORY" && fields[3] === "WRITE" &&
      fields[4] === String(process.pid) && validDevice &&
      BigInt(`0x${device[0]}`) === major && BigInt(`0x${device[1]}`) === minor &&
      BigInt(device[2]) === info.ino && fields[6] === "0" && fields[7] === "EOF";
  });
  assert.equal(matches.length, 1, "current Node PID must hold the canonical FLOCK ADVISORY WRITE 0..EOF");
  return {
    identity: {
      canonicalPath: lockPath,
      pid: process.pid,
      deviceMajor: String(major),
      deviceMinor: String(minor),
      inode: String(info.ino),
      type: "FLOCK",
      advisory: "ADVISORY",
      mode: "WRITE",
      start: "0",
      end: "EOF",
    },
    rawLine: matches[0],
  };
}

function addFile(files, seen, record) {
  assert(!seen.has(record.path), `duplicate archive path: ${record.path}`);
  seen.add(record.path);
  files.push(record);
}

async function collect(includeData) {
  const files = [];
  const seen = new Set();
  for (const spec of fixedPins) addFile(files, seen, await checkedPin(spec, includeData));
  for (const spec of sourcePins) {
    const record = spec.commit ? checkedGitSource(spec, includeData) : await checkedPin(spec, includeData);
    addFile(files, seen, record);
  }
  const inventorySummary = {};
  for (const spec of modelInventories) {
    const inventory = await loadInventory(spec, `${spec.name} model`);
    const actualPaths = await enumerateEvidenceTree(spec.dataRoot, `${spec.name} model`);
    assert.deepEqual(actualPaths, inventory.map(entry => entry.path), `${spec.name} model evidence tree does not match its fixed inventory`);
    for (const entry of inventory) {
      const location = resolve(spec.dataRoot, entry.path);
      const archivePath = `${spec.archivePrefix}/${entry.path}`;
      const record = includeData
        ? await checkedFile(location, spec.dataRoot, entry.path, entry.bytes, entry.sha256, archivePath, true)
        : { path: archivePath, bytes: entry.bytes, sha256: entry.sha256 };
      addFile(files, seen, record);
    }
    inventorySummary[spec.name] = { sha256: spec.pin.sha256, files: spec.totals.files, bytes: spec.totals.bytes };
  }

  const productionGateInventory = await loadProductionGateInventory();
  const actualGatePaths = await enumerateEvidenceTree(productionGateData, "production gate");
  assert.deepEqual(actualGatePaths, productionGateInventory.entries.map(entry => entry.path), "production gate evidence tree does not match its fixed inventory");
  for (const entry of productionGateInventory.selected) {
    const archivePath = `evidence/production-gate/gates-v1/${entry.path}`;
    const location = resolve(productionGateData, entry.path);
    const record = includeData
      ? await checkedFile(location, productionGateData, entry.path, entry.bytes, entry.sha256, archivePath, true)
      : { path: archivePath, bytes: entry.bytes, sha256: entry.sha256 };
    addFile(files, seen, record);
  }
  inventorySummary.productionGate = {
    sha256: productionGateInventoryPin.sha256,
    files: productionGateInventory.counts.files,
    bytes: productionGateInventory.counts.bytes,
    selectedFiles: productionGateInventory.selectedSummary.files,
    selectedBytes: productionGateInventory.selectedSummary.bytes,
    omittedStdoutFiles: productionGateInventory.selectedSummary.omittedStdoutFiles,
    omittedStdoutBytes: productionGateInventory.selectedSummary.omittedStdoutBytes,
  };
  const publisherPath = "publication/docs/benchmarks/native-avx512-2026-11-07/archive-publication.mjs";
  const publisherAbsolute = fileURLToPath(import.meta.url);
  const publisherStat = await lstat(publisherAbsolute, { bigint: true });
  assert(publisherStat.isFile() && !publisherStat.isSymbolicLink(), "publisher must be a regular file");
  assert.equal(await realpath(publisherAbsolute), publisherAbsolute, "publisher path must be canonical");
  const publisherData = await readFile(publisherAbsolute);
  addFile(files, seen, { path: publisherPath, bytes: publisherData.length, sha256: hash(publisherData), ...(includeData ? { encoding: "base64", data: publisherData.toString("base64") } : {}) });
  files.sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);

  const byPath = new Map(files.map(file => [file.path, { path: file.path, bytes: file.bytes, sha256: file.sha256 }]));
  const references = {
    prototypeModelRun: {
      plan: byPath.get("evidence/prototype-model-run/plan.json"),
      receipt: byPath.get("evidence/prototype-model-run/models-v1/receipt.json"),
      inventory: byPath.get("evidence/prototype-model-run/models-inventory-v1.json"),
      releaseProof: byPath.get("evidence/prototype-model-run/models-release-proof-v1.json"),
      sourceIndex: byPath.get("evidence/prototype-model-run/source-index.json"),
      sourceValidation: byPath.get("evidence/prototype-model-run/source-validation.json"),
    },
    productionModelRun: {
      plan: byPath.get("evidence/production-model-run/plan.json"),
      draftPlan: byPath.get("evidence/production-model-run/draft-plan.json"),
      finalizedIndex: byPath.get("evidence/production-model-run/finalized-index.json"),
      receipt: byPath.get("evidence/production-model-run/models-v1/receipt.json"),
      inventory: byPath.get("evidence/production-model-run/models-inventory-v1.json"),
      releaseProof: byPath.get("evidence/production-model-run/models-release-proof-v1.json"),
      sourceIndex: byPath.get("evidence/production-model-run/source-index.json"),
      sourceValidation: byPath.get("evidence/production-model-run/source-validation.json"),
    },
    prototypeGate: {
      receipt: byPath.get("evidence/prototype-gate/gates-v1/receipt.json"),
      releaseProof: byPath.get("evidence/prototype-gate/gates-release-proof-v1.json"),
      inventoryMetadata: byPath.get("evidence/prototype-gate/gates-inventory-v1.json"),
      codegen: byPath.get("evidence/prototype-gate/gates-v1/codegen.json"),
      sourceDeltaProof: byPath.get("evidence/prototype-gate/source-delta-proof.json"),
      sourceIndex: byPath.get("evidence/prototype-gate/source-index.json"),
      sourceReceipt: byPath.get("evidence/prototype-gate/source-receipt.json"),
    },
    productionGate: {
      receipt: byPath.get("evidence/production-gate/gates-v1/receipt.json"),
      releaseProof: byPath.get("evidence/production-gate/gates-release-proof-v1.json"),
      inventoryMetadata: byPath.get("evidence/production-gate/gates-inventory-v1.json"),
      sourceIndex: byPath.get("evidence/production-gate/source-index.json"),
      sourceValidation: byPath.get("evidence/production-gate/source-validation.json"),
      plan: byPath.get("evidence/production-gate/plan.json"),
    },
    sourceSnapshots: {
      prototypeBaselineKernel: byPath.get("evidence/source/prototype-baseline/crates/kevala/src/kernels.rs"),
      prototypeCandidateKernel: byPath.get("evidence/source/prototype-candidate/crates/kevala/src/kernels.rs"),
      prototypeCandidateBuildScript: byPath.get("evidence/source/prototype-candidate/crates/kevala/build.rs"),
      productionBaselineKernel: byPath.get("evidence/source/production-baseline/crates/kevala/src/kernels.rs"),
      productionCandidateKernel: byPath.get("evidence/source/production-candidate/crates/kevala/src/kernels.rs"),
      productionCandidateBuildScript: byPath.get("evidence/source/production-candidate/crates/kevala/build.rs"),
    },
    priorFailures: {
      incompleteModelCapture: byPath.get("evidence/prior-failures/incomplete-model-capture-receipt.json"),
      compileBudget: byPath.get("evidence/prior-failures/compile-budget-failure-receipt.json"),
    },
    publication: {
      report: byPath.get("publication/docs/native-avx512-performance.md"),
      readme: byPath.get("publication/docs/benchmarks/native-avx512-2026-11-07/README.md"),
      howPageSource: byPath.get("publication/app/views/how.js"),
      dispatchDesign: byPath.get("publication/docs/native-avx512-dispatch.md"),
      publisher: byPath.get(publisherPath),
    },
  };
  for (const group of Object.values(references)) {
    for (const value of Object.values(group)) assert(value, "missing archive reference");
  }
  return { files, references, exclusions, inventorySummary };
}

async function outputExists(path) {
  try { await lstat(path); return true; }
  catch (error) { if (error.code === "ENOENT") return false; throw error; }
}

async function readRegularOutput(path) {
  const absolute = resolve(path);
  const info = await lstat(absolute, { bigint: true });
  assert(info.isFile() && !info.isSymbolicLink(), `output must be a regular file: ${absolute}`);
  assert.equal(await realpath(absolute), absolute, `output resolves through a symlink: ${absolute}`);
  return readFile(absolute);
}

async function verify() {
  const before = await lease();
  const plan = await collect(false);
  const manifestBytes = await readRegularOutput(manifestPath);
  const manifest = JSON.parse(manifestBytes);
  assert.equal(manifest.schemaVersion, 1);
  assert.equal(manifest.archive.path, "evidence.json.gz");
  assert.deepEqual(manifest.references, plan.references);
  assert.deepEqual(manifest.exclusions, exclusions);
  assert.deepEqual(manifest.inventory, plan.inventorySummary);
  const compressed = await readRegularOutput(archivePath);
  assert.equal(compressed.length, manifest.archive.bytes);
  assert.equal(hash(compressed), manifest.archive.sha256);
  const decoded = gunzipSync(compressed);
  assert.equal(decoded.length, manifest.archive.decodedBytes);
  assert.equal(hash(decoded), manifest.archive.decodedSha256);
  const payload = JSON.parse(decoded);
  assert.equal(payload.schemaVersion, 1);
  assert.deepEqual(payload.references, plan.references);
  assert.deepEqual(payload.exclusions, exclusions);
  assert.deepEqual(payload.inventory, plan.inventorySummary);
  assert.equal(payload.files.length, plan.files.length);
  const expected = new Map(plan.files.map(file => [file.path, { bytes: file.bytes, sha256: file.sha256 }]));
  const seen = new Set();
  const descriptors = [];
  for (const file of payload.files) {
    const { data, encoding, ...descriptor } = file;
    assert.equal(encoding, "base64");
    assert(expected.has(file.path), `unexpected archived path: ${file.path}`);
    assert(!seen.has(file.path), `duplicate archived path: ${file.path}`);
    seen.add(file.path);
    assert.deepEqual({ bytes: file.bytes, sha256: file.sha256 }, expected.get(file.path));
    const original = Buffer.from(data, "base64");
    assert.equal(original.length, file.bytes);
    assert.equal(hash(original), file.sha256);
    descriptors.push(descriptor);
  }
  assert.deepEqual(descriptors, plan.files);
  assert.deepEqual(manifest.files, descriptors);
  const after = await lease();
  assert.deepEqual(after.identity, before.identity);
  console.log(JSON.stringify({ verified: true, files: payload.files.length, inventory: plan.inventorySummary, archiveSha256: manifest.archive.sha256, manifestSha256: hash(manifestBytes) }));
}

async function main() {
  assert(["--plan", "--archive", "--verify"].includes(mode), "use --plan, --archive, or --verify");
  assert.equal(process.argv.length, 3, "unexpected arguments");
  if (mode === "--plan") {
    const result = await collect(false);
    console.log(JSON.stringify({
      plan: true,
      selectedFiles: result.files.length,
      inventories: result.inventorySummary,
      fixedEvidenceFiles: result.files.length - inventoryTotals.files - productionInventoryTotals.files - productionGateSelectionTotals.files,
      exclusions,
    }));
    return;
  }
  if (mode === "--verify") {
    await verify();
    return;
  }

  const before = await lease();
  assert.equal(await outputExists(archivePath), false, `refusing existing output: ${archivePath}`);
  assert.equal(await outputExists(manifestPath), false, `refusing existing output: ${manifestPath}`);
  const result = await collect(true);
  const decoded = Buffer.from(JSON.stringify({ schemaVersion: 1, files: result.files, references: result.references, exclusions, inventory: result.inventorySummary }) + "\n");
  const compressed = gzipSync(decoded, { level: 9, mtime: 0 });
  assert.equal(hash(gunzipSync(compressed)), hash(decoded), "gzip byte roundtrip failed");
  const after = await lease();
  assert.deepEqual(after.identity, before.identity);
  const manifest = {
    schemaVersion: 1,
    recordedAt: new Date().toISOString(),
    archive: { path: "evidence.json.gz", bytes: compressed.length, sha256: hash(compressed), decodedBytes: decoded.length, decodedSha256: hash(decoded) },
    files: result.files.map(({ data, encoding, ...descriptor }) => descriptor),
    references: result.references,
    exclusions,
    inventory: result.inventorySummary,
    lease: { before, after },
  };
  await writeFile(archivePath, compressed, { flag: "wx" });
  await writeFile(manifestPath, JSON.stringify(manifest, null, 2) + "\n", { flag: "wx" });
  console.log(JSON.stringify({ archived: true, files: result.files.length, inventories: result.inventorySummary, archiveSha256: hash(compressed), manifestSha256: hash(Buffer.from(JSON.stringify(manifest, null, 2) + "\n")) }));
}

main().catch(error => {
  console.error(error.stack ?? error);
  process.exitCode = 1;
});
