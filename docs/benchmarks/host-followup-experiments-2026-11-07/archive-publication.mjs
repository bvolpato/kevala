#!/usr/bin/env node
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { lstat, readFile, realpath, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync, gunzipSync } from "node:zlib";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../../..");
const campaign = "/home/bruno/githubworkspace/kevala/tmp/sota-20261106-kdDTGn";
const lock = "/home/bruno/.cache/kevala/gpu-profile.lock";
const archivePath = join(here, "evidence.json.gz");
const manifestPath = join(here, "manifest.json");
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const mode = process.argv[2];
const allowIo = process.argv[3] === "--allow-io";

const evidencePins = [
  ["matmul-shape-constants-v1/plan.json", 17943, "b2692355aa572a0b7c3e71c151f73b04bae21dbae2bd4d62e608890a3e2a7a26"],
  ["matmul-shape-constants-v1/prepare.mjs", 8755, "6f3b94a02e86c840d91445662336284be2990f0b2d781b2c833856a159dc0f58"],
  ["matmul-shape-constants-v1/run.mjs", 21211, "487d1fe12d06d43a92e2f83da00039faba4411c7418194f252ed3e58039bcb6e"],
  ["matmul-shape-constants-v1/validate.mjs", 10296, "6ec66b3422013395712bb95bd7cfbbdbb07b09f23f4a8a23600f5f2004174b23"],
  ["matmul-shape-constants-v1/probe.html", 177, "301f8fb31ccb49c1cf134e3fe84184f2f9b5af45c46585c84ddd19ff06f33569"],
  ["matmul-shape-constants-v1/probe.js", 20910, "28fd7c9b493369a5d32c4899df82150103fdb0725c44e3a4115d33cdfc6384a5"],
  ["matmul-shape-constants-v1/source-index.json", 1056, "3fd2e476ec944110f3ebd8b5c547088afaca3b481e6cad0855b9f7b6025bcf99"],
  ["matmul-shape-constants-v1/runtime-identity.json", 9060, "3380f0dc26a5aba9b3357617a340af3c94d86ff4912492bd01b37352bc885f3e"],
  ["matmul-shape-constants-v1/runs-v1/receipt.json", 71147, "b5e6178059913bce99d921fb7ebcfcad74b44821be7595186019e2574b803af6"],
  ["matmul-shape-constants-v1/runs-v1/raw-browser.json", 32701, "c6cfc7d7b790fa33b7c5b4a6d173e4526a2abf730865afc1310f2e161d274c2d"],
  ["matmul-shape-constants-v1/runs-v1/inventory.json", 1076, "2c31408a3f5ef3adc8c4139df3272fa8d1bb5621dcd585074999db93a445b769"],
  ["matmul-shape-constants-v1/runs-v1/command.json", 1224, "e9c6c00a961b55b48b9fcbf7e5b8bd610982026b4b5b4a56398545cadd5a700c"],
  ["matmul-shape-constants-v1/runs-v1/input.json", 14813, "cc64748753727335d89aedce02f1d576009ce53894ad96e4bc10cf54885f2423"],
  ["matmul-shape-constants-v1/runs-v1/admission.json", 2741, "e7028dfc0d3c5023622bff57bf49afd4eb37ae2d6cea6a599ffd7499330b3fb3"],

  ["matmul-shape-constants-diagnostic-v1/plan.json", 26035, "620bfafa04b303e6d9aa2535feebf2c01f4b0ef4fef0e579f473de6642333260"],
  ["matmul-shape-constants-diagnostic-v1/prepare.mjs", 7035, "89de3d3e8a789a45b6dd27bb4cc8c33c7b9243f7c6db76d9ad8fd165311f5858"],
  ["matmul-shape-constants-diagnostic-v1/run.mjs", 24068, "025db5a9e715e9dad9855826aaf9e88589f4bad68407820952b1c7816b3037ee"],
  ["matmul-shape-constants-diagnostic-v1/validate.mjs", 10296, "6ec66b3422013395712bb95bd7cfbbdbb07b09f23f4a8a23600f5f2004174b23"],
  ["matmul-shape-constants-diagnostic-v1/probe.html", 177, "301f8fb31ccb49c1cf134e3fe84184f2f9b5af45c46585c84ddd19ff06f33569"],
  ["matmul-shape-constants-diagnostic-v1/probe.js", 20910, "28fd7c9b493369a5d32c4899df82150103fdb0725c44e3a4115d33cdfc6384a5"],
  ["matmul-shape-constants-diagnostic-v1/source-index.json", 6521, "cb575439660ce4a3f059bd1b9ec7e8d30cddff00314e3f4b3bf384b088209e92"],
  ["matmul-shape-constants-diagnostic-v1/runtime-identity.json", 9060, "3380f0dc26a5aba9b3357617a340af3c94d86ff4912492bd01b37352bc885f3e"],
  ["matmul-shape-constants-diagnostic-v1/bench-gpu-diagnostic.py", 39082, "7d6e87ef23add1bfac7d4ac6941a0e42945b87cc19b7315db5c29dc32ca2ae6f"],
  ["matmul-shape-constants-diagnostic-v1/browser-runner.patch", 4408, "5dc119df9d45468d56d9e66874387fc355327153c3b2974986cef8ee44dd52e8"],
  ["matmul-shape-constants-diagnostic-v1/controller-launch.patch", 11510, "e7eb125df20480a0cd850e8402e83eabb48fe373c15da4f34d0663cc72c5db97"],
  ["matmul-shape-constants-diagnostic-v1/runs-v1/receipt.json", 90968, "74967220149fdfdbb9bf37ae9f4da76924f4eb54da680929a3950cb0a1e0d59c"],
  ["matmul-shape-constants-diagnostic-v1/runs-v1/raw-browser.json", 405299, "797c47ae2049a7ce629052776f033b95bdf0470e4804f4b5a0073b21cb3be3fc"],
  ["matmul-shape-constants-diagnostic-v1/runs-v1/inventory.json", 1571, "24f02e09011a4d13de0a1bd661257c9186e71c55e48fd3e0dd231baa4878e1ab"],
  ["matmul-shape-constants-diagnostic-v1/runs-v1/command.json", 1758, "cfeb08dbe3c6f7a16bf9d1f86b8730179f4f90aa5bf887134990ecca29740261"],
  ["matmul-shape-constants-diagnostic-v1/runs-v1/input.json", 14837, "ce6d0b0febd7a636e6bfbffd06a98df48072fe3f51248ca8bb7a2b8b54b3099e"],
  ["matmul-shape-constants-diagnostic-v1/runs-v1/admission.json", 10264, "40fec17f18afb34162082ee986ea6d3ebbd0212db6f8b6577b5885fe96ef9125"],
  ["matmul-shape-constants-diagnostic-v1/runs-v1/runner.stdout", 12, "40be9b3f9ac184bafe2c1918e0011197d56091c09129065b406bdf50e92a1eca"],
  ["matmul-shape-constants-diagnostic-v1/runs-v1/runner.stderr", 199, "992d0889f01955407f80eb0c0ffb74e5855bdd11fc1edf6990c04517b4fc7a88"],
  ["matmul-shape-constants-diagnostic-v1/runs-v1/browser-logs/lifecycle.jsonl", 884, "ad776a1a1cf2a8c743df8a155bdc89a48d5250e8ccfb05cb4e42110547ad3f40"],
  ["matmul-shape-constants-diagnostic-v1/runs-v1/browser-logs/chrome-gpu.log", 1303, "dfc6c307236c25996a2eaa6645f071a7dd0baeea329f37d3d04ce0427785a6e9"],

  ["matmul-fp32-k64-v1/plan.json", 30663, "e6b5b7a372536ff112f6496a7f0ff6f1bf1c99d9944f01d5bf7c4551b1eab67b"],
  ["matmul-fp32-k64-v1/prepare.mjs", 9312, "f35d4e5341c92ad090b9be9bbd4c549cb307128ae18ed83aec4c981232bda50e"],
  ["matmul-fp32-k64-v1/run.mjs", 25232, "2d87c93df1e16ca609a1e962e034d5bd97c72eb6a676d66cbd8b8cb50901450b"],
  ["matmul-fp32-k64-v1/validate.mjs", 11206, "9cc399c499eb3eb92f3433537a1802f5981b327b9a8eaaf37d28865b4b2f7a6f"],
  ["matmul-fp32-k64-v1/probe.html", 174, "a3c1f12bfeae953458aae62f68555ac6dd3a6cc1ec30092745c05ed880845d92"],
  ["matmul-fp32-k64-v1/probe.js", 22403, "bd62b1ce781179f31968e9f6d5f1ee19e34f310bc1f79e1f535ebd6a5635b056"],
  ["matmul-fp32-k64-v1/bench-gpu-diagnostic.py", 39080, "42866bc173ac424ca14db935e60cbe436bc412b996cec22025f88fbed10314f1"],
  ["matmul-fp32-k64-v1/source-index.json", 12307, "910823588072c6a4aa2fc4370ed1f7843196d94fd36975ff700491535257eb5f"],
  ["matmul-fp32-k64-v1/runs-v1/receipt.json", 93377, "17f6d8e9805ca48dd3816340be3b384629577f7d285090b1ec2f1424330161da"],
  ["matmul-fp32-k64-v1/runs-v1/inventory.json", 1838, "d523d1b5084e9da77b5ea995151145bcb878bf39f2e8df2f7e2317ce0efb8fa3"],
  ["matmul-fp32-k64-v1/runs-v1/admission.json", 3143, "8b5d813130bab99ca2dfd2a9c80e6f141e854fe906870094c4036426a296c9f3"],
  ["matmul-fp32-k64-v1/runs-v1/0-clef-flash/input.json", 12906, "5e29e6e45a52fb0d55dca572288d8005523cf207f23b19ed855b5f45d9faf242"],
  ["matmul-fp32-k64-v1/runs-v1/0-clef-flash/command.json", 1465, "0bb3d46b41abd7e68d83be4528f9abcb26217127edc2a5f7530f495d72c63559"],
  ["matmul-fp32-k64-v1/runs-v1/0-clef-flash/cell-receipt.json", 6892, "8d80fd724d32ee3bbc0f1001560dbc310e8f6a3e93caa5e98fa4099ee95f41a0"],
  ["matmul-fp32-k64-v1/runs-v1/0-clef-flash/raw-browser.json", 130280, "9ad75bb15e54ee8c787f63fa0d7a78eda06cad6dfa5081f2260d734d2d9f4918"],
  ["matmul-fp32-k64-v1/execution-final-v1.json", 836, "519ce73678cfc1b0405d22fc1b1463701c85fdc8d40b5ca77739f9a797817a77"],
  ["matmul-fp32-k64-v1/ci76-runtime-binding-v1.json", 12115, "a057048d9123d9ac1d0c848972311d74c7053e6d98cc1f26354cf42ddca50687"],
  ["no-cache-carries-delivery/delivery-06c58a1679e91ca2abcdbb2523440dc87099a27b-run42358893911-v1.json", 1673608, "9983ef96a3f0e54ea89e81d136c840ed516fc79780243255e821fcb17928b1d4"],
  ["kev9-bf16-delivery/delivery-8d4e4fa458c8faf724d624670de721bd51cd7175-run42361138898-v1.json", 3201746, "64690c4cee23f33bef564276dbbdd8fe9eecba45442a4b5dcab2333cadf80938"],

  ["gemma-attention-cold-v1/plan.json", 8155, "dabe699f4ebb4eb50078e31d522480c1fa31f3665b228f89d6d419d19395da4b"],
  ["gemma-attention-cold-v1/config.json", 29926, "d5c7a0537e6ecc4b9987eb35a0541d1ae389038ab3dafbca6026ff919454f786"],
  ["gemma-attention-cold-v1/fixtures.mjs", 1398, "52aeec28892dc9d0ea07332da8d42e4bdd626149929886401dab35274f77d95b"],
  ["gemma-attention-cold-v1/run.mjs", 25654, "e0607c84c935417911f1a8b8c34feb8cb0fa7cafa0d484a64479dc67bc713999"],
  ["gemma-attention-cold-v1/probe.js", 14154, "45e82f1c312517a41bc9622a9fac8c4b7c1f056a5e1b5f21a999d3e937e5fec7"],
  ["gemma-attention-cold-v1/run-browser.py", 3964, "d2e33ceb1d918e8cbb76c53e0f6ff0fa8028209b95d4ec4ed164081234ad4157"],
  ["gemma-attention-cold-v1/controller-lib.mjs", 9708, "ae7d50c678380871bd369dee8ff01fe31f1c71170ed9f3a69281f976257a1d50"],
  ["gemma-attention-cold-v1/production-candidate.patch", 479, "e7d55ff43686e956ecdd67a66b34dd0c6795506e8c95e927786fa4626f86d3fc"],
  ["gemma-attention-cold-v1/source-index.json", 1886, "e22294f1e688b5916669e10c51e8f204a647839fa7ea3f65c79176c112e54d5b"],
  ["gemma-attention-cold-v1/runtime-candidate/gpu-gemma4-attention.js", 24397, "00b24d908fbad27ad2b5630db5faa2633c4f342ab0485bc5e29e5640e44ccca6"],
  ["gemma-attention-cold-v1/index.html", 249, "40d545b5de61a14be96fc87c91e3f5d7d2ab5a92aecf56e231abed90e2996ade"],
  ["gemma-attention-cold-v1/reuse.json", 2276, "84707800fa59eddf09e16b84f9b9727f32b69ec285c5dfb62ecb170d95361abe"],
  ["gemma-attention-cold-v1/results-v1/receipt.json", 185621, "5dbb653a77dbaee7e98731c252aae3ec3b13cbf3f0233586ff1a9bdfff5f223d"],
  ["gemma-attention-cold-v1/results-v1/admission.json", 1960, "97bfd19093477ef39651c99102c344ce4329340a9629df0aa3c22b45da8543b2"],
  ["gemma-attention-cold-v1/results-v1/gemma-4-e2b-p0-0-A/receipt.json", 19198, "83a0504ec8f116395b6dea1ee949723022b771c6fecc3695912cd9159baec832"],
  ["gemma-attention-cold-v1/results-v1/gemma-4-e2b-p0-0-A/browser.json", 242281, "dc65259905dd1279c746dbf7e842b9d559621fa0fe332916e3ae0f653cd98690"],
  ["gemma-attention-cold-v1/results-v1/gemma-4-e2b-p0-0-A/browser-command.json", 1338, "90d853dc0db378f4ae6489ee00af59ec2c0c6ecee4153763145846220dd41cf3"],
  ["gemma-attention-cold-v1/results-v1/gemma-4-e2b-p0-1-B/browser.json", 234018, "0b17c3262c1e42ed336f423fc6fefe2bfad64c7ce6c59a2002d08488e965d42e"],
  ["gemma-attention-cold-v1/results-v1/gemma-4-e2b-p0-1-B/browser-command.json", 1306, "181fedeff2b5b7352f06500eaf92e0e898ca1d3d5895c391791dbbbf4fd77f60"],

  ["gemma-attention-budget-v1/plan.json", 6026, "100e9c9ae766f78e198dd46c599e957318d90ef87efd4595bd8913eb5791174b"],
  ["gemma-attention-budget-v1/config.json", 12418, "1ab63ea843a67afdcbaeef2b17b9ff028654fc8d515a2402e9c5cd14f09fa631"],
  ["gemma-attention-budget-v1/fixtures.mjs", 1488, "eaa037966036fba70a3ac579dad2f1c2394bb4120008369cb69dc1aa2fd7b442"],
  ["gemma-attention-budget-v1/run.mjs", 25218, "3594d1d9a82424713792fd6dbf771da2cb0bf2c999397c16c10a9fb042404193"],
  ["gemma-attention-budget-v1/probe.js", 17574, "dea8a1c99937cd3c4865ce706c064e2229ff789679906033bfbed1e086a1bbff"],
  ["gemma-attention-budget-v1/run-browser.py", 3970, "b4b2635b53085c4a000dfd29485601599f8720395696cf637277d693be9eeb6c"],
  ["gemma-attention-budget-v1/index.html", 238, "a5d572d2185118411e688913aaa87cfeb8d8a67f8ec652cb478f9caef23ddf4f"],
  ["gemma-attention-budget-v1/source-index.json", 1612, "f1979962f6fa2fec603e9908271a5848607cb65786917ad1c6d2846ecaeab70c"],
  ["gemma-attention-budget-v1/attention-budget-1500.js", 24805, "1df518dc1ae972ed0e3c180677bb6d389fa663d653160ef52798d8249bc3c40d"],
  ["gemma-attention-budget-v1/calibration-helper.patch", 1565, "40940914854351cc55cc0fdad854b6c19d1d52d03e055396a32160e13b471e56"],
  ["gemma-attention-budget-v1/results-v1/receipt.json", 47244, "df1adf9f9d36721f4b9890042b715e2d60801997020099cb50db960eb3423353"],
  ["gemma-attention-budget-v1/results-v1/browser.json", 452598, "867ec3d3205e259e7ab66ea2ae17c495546ce1a5582750f0546dc4030a87f18a"],
  ["gemma-attention-budget-v1/results-v1/admission.json", 1809, "fa534f314c69bf2831fa50d2e53a2cc09927a731eeac85bf56808d0623956d72"],
  ["gemma-attention-budget-v1/results-v1/browser-command.json", 1157, "9473ed6155759089e46bba0f4ff345b9fd8ab478fc704edadf28897a7a7c2dd8"],

  ["firefox-submit-threeway-v1/SOURCE-SHA256SUMS", 1050, "c06f0323958bf1e0beb6e68ab339bf195526b52a51952daad182856aafc851a6"],
  ["firefox-submit-threeway-v1/runtime-pin.json", 2894, "c19aecb64c763cc511118a01ea11559ca3368e307ce443d059928f57996684ef"],
  ["firefox-submit-threeway-v1/preflight.mjs", 3057, "26db706591401cd0bdc786abf7d8ab73f035a6289700c6f00ee90cb7c10182de"],
  ["firefox-submit-threeway-v1/fixtures.mjs", 3384, "154289413ef334f266a8037ac6dc418df72cf2070fbda3908e23b115f218d41f"],
  ["firefox-submit-threeway-v1/index.html", 1273, "55fbcc50c044e81dde7875ff466120e3036b3216adde5464ed8a166aee51fd93"],
  ["firefox-submit-threeway-v1/headed-probe.js", 47500, "200bb66128e5b1ed785fd05f4220d804dbb3e4505475f5fe2b657447f3e35150"],
  ["firefox-submit-threeway-v1/run-headed.py", 8540, "c0fa886f4224cffd2c2d54cb238db86709a63962c0c745361be63c759f901798"],
  ["firefox-submit-threeway-v1/run.mjs", 47459, "7c350cb2a49af684c12c39a610929e09daac355b082b8beb91058e58a7fe55b5"],
  ["firefox-submit-threeway-v1/validate.mjs", 20261, "d96cbd1cf41d88dd3d839ae14a9625defbef96b7d44822f4aa5746281229a2f9"],
  ["firefox-submit-threeway-v1/revalidate-ratio-drift.mjs", 9850, "d875a3ed678b64af348222bc65d39bdc060164ddbbd9eff106c40341755a16d5"],
  ["firefox-submit-threeway-v1/results/raw-timing-2026-11-07T08-09-37.171Z.json", 1378324, "3c5c9affc1d8953f3b6805608ecbe530abe939564c77bdae3dfa09b352c7fee9"],
  ["firefox-submit-threeway-v1/results/runner-timing-2026-11-07T08-09-37.171Z.json", 84632, "b884452a76312645ee8d329ebedfc95ae614eb2e26bad665481aa23a8b610d37"],
  ["firefox-submit-threeway-v1/results/derived-revalidation-ratio-canonicalized-2026-11-07-v1.json", 10390, "f72f6465c9a1b965b75e59c6d711948a28b4351262c2b6d14cf8a23d56130ed6"],
];

const publicationPins = [
  ["publication/docs/host-followup-experiments.md", "docs/host-followup-experiments.md", 7952, "d98abe0cf34e4a762c255e70f3cb6b92a356f927ee3da4c59301f47ab05421e6"],
  ["publication/docs/benchmarks/host-followup-experiments-2026-11-07/README.md", "docs/benchmarks/host-followup-experiments-2026-11-07/README.md", 4500, "d7178f6f255c3ddd45af12508d7e8f91d542f294c8505d44b692f26eb87c27d4"],
  ["publication/app/views/how.js", "app/views/how.js", 16800, "90b11ee20e86a4d4a7a34e8a25646f4a77715c0a2960a29cc1d5ac73f0283807"],
];
const exclusions = [
  "model weights, scales, tensor payloads, and full model packs",
  "compiled WASM and other binary runtime payloads",
  "full runtime-v1 and runtime-candidate directories",
  "matmul-shape-constants-diagnostic-v1/runs-v1/browser-logs/chromedriver.log (19,520,808 bytes)",
  "matmul-fp32-k64-v1/runs-v1/0-clef-flash/browser-logs/chromedriver.log (1,689,696 bytes)",
  "build output, credentials, and generated archives from earlier campaigns",
];

assert(["--plan", "--archive", "--verify"].includes(mode), "use --plan or --archive/--verify --allow-io");
assert(mode === "--plan" || allowIo, `${mode} requires --allow-io and the owner's I/O window`);
assert.deepEqual(process.argv.slice(2), mode === "--plan" ? ["--plan"] : [mode, "--allow-io"], "unexpected arguments");

async function lease() {
  assert.equal(await realpath(lock), lock, "canonical profile lock path must resolve to itself");
  const info = await lstat(lock, { bigint: true });
  assert(info.isFile() && !info.isSymbolicLink(), "canonical profile lock must be a regular file");
  const major = info.dev >> 8n & 0xfffn | info.dev >> 32n & 0xfffff000n;
  const minor = info.dev & 0xffn | info.dev >> 12n & 0xffffff00n;
  const matches = (await readFile("/proc/locks", "utf8")).split("\n").filter(line => {
    const fields = line.trim().split(/\s+/), device = fields[5]?.split(":");
    return fields[1] === "FLOCK" && fields[2] === "ADVISORY" && fields[3] === "WRITE" &&
      fields[4] === String(process.pid) && device?.length === 3 &&
      BigInt(`0x${device[0]}`) === major && BigInt(`0x${device[1]}`) === minor &&
      BigInt(device[2]) === info.ino && fields[6] === "0" && fields[7] === "EOF";
  });
  assert.equal(matches.length, 1, "current Node PID must hold the canonical FLOCK ADVISORY WRITE 0..EOF");
  return {
    identity: {
      canonicalPath: lock,
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

async function pinnedFile(path, location, bytes, digest) {
  const absolute = resolve(location);
  const root = path.startsWith("evidence/") ? campaign : repo;
  const rel = relative(root, absolute);
  assert(rel && !rel.startsWith("..") && !isAbsolute(rel), `path escapes its fixed root: ${path}`);
  const info = await lstat(absolute);
  assert(info.isFile() && !info.isSymbolicLink(), `not a regular file: ${path}`);
  assert.equal(await realpath(absolute), absolute, `symlink path rejected: ${path}`);
  assert.equal(info.size, bytes, `byte count differs: ${path}`);
  const data = await readFile(absolute);
  assert.equal(hash(data), digest, `SHA-256 differs: ${path}`);
  return { descriptor: { path, bytes: data.length, sha256: digest }, data };
}

async function publisherFile(includeData) {
  const path = "publication/docs/benchmarks/host-followup-experiments-2026-11-07/archive-publication.mjs";
  const location = fileURLToPath(import.meta.url);
  const info = await lstat(location);
  assert(info.isFile() && !info.isSymbolicLink() && info.size < 1024 * 1024, "publisher must be a small regular file");
  assert.equal(await realpath(location), resolve(location), "publisher path must not traverse a symlink");
  const data = await readFile(location);
  return { path, bytes: data.length, sha256: hash(data), ...(includeData ? { encoding: "base64", data: data.toString("base64") } : {}) };
}

function verifyEvidence(byPath, recordsByPath) {
  const json = path => JSON.parse(byPath.get(path).toString("utf8"));
  const descriptor = path => {
    const item = recordsByPath.get(path);
    assert(item, `missing pinned evidence: ${path}`);
    return item;
  };

  const originalMatmulPath = "evidence/matmul-shape-constants-v1/runs-v1/receipt.json";
  const originalMatmul = json(originalMatmulPath);
  assert.equal(originalMatmul.status, "failed");
  assert.match(originalMatmul.error, /A valid external Instance reference no longer exists/);
  assert.equal(json("evidence/matmul-shape-constants-v1/runs-v1/raw-browser.json").status, "error");

  const diagnosticPath = "evidence/matmul-shape-constants-diagnostic-v1/runs-v1/receipt.json";
  const diagnostic = json(diagnosticPath);
  assert.equal(diagnostic.status, "passed");
  assert.equal(diagnostic.validation.status, "passed");
  assert.equal(diagnostic.rawBrowser.sha256, descriptor("evidence/matmul-shape-constants-diagnostic-v1/runs-v1/raw-browser.json").sha256);
  assert.equal(diagnostic.diagnosticLogs.files.find(file => file.path === "browser-logs/chromedriver.log").bytes, 19520808);
  assert.equal(json("evidence/matmul-shape-constants-diagnostic-v1/runs-v1/raw-browser.json").status, "done");

  const k64Path = "evidence/matmul-fp32-k64-v1/runs-v1/receipt.json";
  const k64 = json(k64Path);
  const k64Cell = json("evidence/matmul-fp32-k64-v1/runs-v1/0-clef-flash/cell-receipt.json");
  const k64Execution = json("evidence/matmul-fp32-k64-v1/execution-final-v1.json");
  const k64Binding = json("evidence/matmul-fp32-k64-v1/ci76-runtime-binding-v1.json");
  const k64Plan = json("evidence/matmul-fp32-k64-v1/plan.json");
  const ci75DeliveryPath = "evidence/no-cache-carries-delivery/delivery-06c58a1679e91ca2abcdbb2523440dc87099a27b-run42358893911-v1.json";
  const ci76DeliveryPath = "evidence/kev9-bf16-delivery/delivery-8d4e4fa458c8faf724d624670de721bd51cd7175-run42361138898-v1.json";
  assert.equal(k64.status, "passed");
  assert.equal(k64.productionOrTunerChanged, false);
  assert.equal(k64.continuation.allowed, false);
  assert.equal(k64.cells.find(cell => cell.id === "clef-flash").status, "passed");
  assert.equal(k64.cells.find(cell => cell.id === "kev-0.8b").status, "skipped-predeclared-first-cell-rule");
  assert.equal(k64Cell.status, "passed");
  assert.equal(k64Cell.rawBrowser.sha256, descriptor("evidence/matmul-fp32-k64-v1/runs-v1/0-clef-flash/raw-browser.json").sha256);
  assert.equal(k64Execution.status, "passed");
  assert.equal(k64Execution.ownerPidGone, true);
  assert.equal(k64Execution.canonicalLeaseFreeAtProbe, true);
  assert.deepEqual(k64Execution.ownedChildrenRemaining, []);
  assert.equal(k64Execution.candidatePromoted, false);
  assert.equal(k64Execution.secondCellSkipped, true);
  assert.equal(k64Binding.status, "passed");
  assert.equal(k64Binding.source.sha256, descriptor(k64Path).sha256);
  assert.equal(k64Binding.ci76Sha, "8d4e4fa458c8faf724d624670de721bd51cd7175");
  assert.equal(k64Binding.ci76Run, 42361138898);
  assert.equal(k64Binding.assets.length, 53);
  assert.equal(k64Binding.modelLatencyClaimed, false);
  assert.equal(k64Binding.newHardwareRun, false);
  assert.equal(k64Plan.runtime.deliveryReference.sha256, descriptor(ci75DeliveryPath).sha256);
  assert.equal(k64Binding.actualCi76Delivery.sha256, descriptor(ci76DeliveryPath).sha256);

  const coldPath = "evidence/gemma-attention-cold-v1/results-v1/receipt.json";
  const cold = json(coldPath);
  assert.equal(cold.status, "failed");
  assert.match(cold.error, /load warmup hidden differs/);
  const coldA = json("evidence/gemma-attention-cold-v1/results-v1/gemma-4-e2b-p0-0-A/browser.json");
  const coldB = json("evidence/gemma-attention-cold-v1/results-v1/gemma-4-e2b-p0-1-B/browser.json");
  assert.equal(coldA.status, "complete");
  assert.equal(coldB.status, "complete");
  assert.notEqual(coldA.loadWarmupHidden.sha256, coldB.loadWarmupHidden.sha256);
  assert.equal(json("evidence/gemma-attention-budget-v1/results-v1/receipt.json").status, "passed");

  const firefoxRawPath = "evidence/firefox-submit-threeway-v1/results/raw-timing-2026-11-07T08-09-37.171Z.json";
  const firefoxRunnerPath = "evidence/firefox-submit-threeway-v1/results/runner-timing-2026-11-07T08-09-37.171Z.json";
  const firefoxDerivedPath = "evidence/firefox-submit-threeway-v1/results/derived-revalidation-ratio-canonicalized-2026-11-07-v1.json";
  const firefoxRaw = json(firefoxRawPath);
  const firefoxRunner = json(firefoxRunnerPath);
  const firefoxDerived = json(firefoxDerivedPath);
  const firefoxHelper = descriptor("evidence/firefox-submit-threeway-v1/revalidate-ratio-drift.mjs");
  assert.equal(firefoxRaw.status, "done");
  assert.equal(firefoxRunner.status, "failed");
  assert.equal(firefoxRunner.rawResult.sha256, descriptor(firefoxRawPath).sha256);
  assert.equal(firefoxDerived.status, "derived-revalidation-passed");
  assert.equal(firefoxDerived.originalCapture.rawSha256, descriptor(firefoxRawPath).sha256);
  assert.equal(firefoxDerived.originalCapture.runnerSha256, descriptor(firefoxRunnerPath).sha256);
  assert.equal(firefoxDerived.sourceIdentity.reconstructionHelperSha256, firefoxHelper.sha256);
  assert.equal(firefoxDerived.reconstruction.browserRerun, false);
  assert.equal(firefoxDerived.reconstruction.buildOrModelExecution, false);
  assert.equal(firefoxDerived.reconstruction.publicAndHiddenValuesUnmodified, true);
}

async function collect(includeData) {
  const files = [];
  const byPath = new Map();
  const recordsByPath = new Map();
  const add = found => {
    const { descriptor, data } = found;
    files.push({ ...descriptor, ...(includeData ? { encoding: "base64", data: data.toString("base64") } : {}) });
    byPath.set(descriptor.path, data);
    recordsByPath.set(descriptor.path, descriptor);
  };

  for (const [relativePath, bytes, digest] of evidencePins) {
    add(await pinnedFile(`evidence/${relativePath}`, resolve(campaign, relativePath), bytes, digest));
  }
  for (const [archivePath, relativePath, bytes, digest] of publicationPins) {
    add(await pinnedFile(archivePath, resolve(repo, relativePath), bytes, digest));
  }
  files.push(await publisherFile(includeData));
  files.sort((a, b) => a.path.localeCompare(b.path));
  verifyEvidence(byPath, recordsByPath);

  const ref = path => recordsByPath.get(path);
  const references = {
    matmulConstants: {
      originalFailedReceipt: ref("evidence/matmul-shape-constants-v1/runs-v1/receipt.json"),
      diagnosticReceipt: ref("evidence/matmul-shape-constants-diagnostic-v1/runs-v1/receipt.json"),
      diagnosticRawBrowser: ref("evidence/matmul-shape-constants-diagnostic-v1/runs-v1/raw-browser.json"),
    },
    fp32K64: {
      receipt: ref("evidence/matmul-fp32-k64-v1/runs-v1/receipt.json"),
      rawBrowser: ref("evidence/matmul-fp32-k64-v1/runs-v1/0-clef-flash/raw-browser.json"),
      cleanup: ref("evidence/matmul-fp32-k64-v1/execution-final-v1.json"),
      ci76RuntimeBinding: ref("evidence/matmul-fp32-k64-v1/ci76-runtime-binding-v1.json"),
      ci75DeliveryRecord: ref("evidence/no-cache-carries-delivery/delivery-06c58a1679e91ca2abcdbb2523440dc87099a27b-run42358893911-v1.json"),
      ci76DeliveryRecord: ref("evidence/kev9-bf16-delivery/delivery-8d4e4fa458c8faf724d624670de721bd51cd7175-run42361138898-v1.json"),
    },
    gemmaCold: {
      failedReceipt: ref("evidence/gemma-attention-cold-v1/results-v1/receipt.json"),
      baselineRawBrowser: ref("evidence/gemma-attention-cold-v1/results-v1/gemma-4-e2b-p0-0-A/browser.json"),
      candidateRawBrowser: ref("evidence/gemma-attention-cold-v1/results-v1/gemma-4-e2b-p0-1-B/browser.json"),
      candidateAttentionSource: ref("evidence/gemma-attention-cold-v1/runtime-candidate/gpu-gemma4-attention.js"),
    },
    gemmaWarm: { receipt: ref("evidence/gemma-attention-budget-v1/results-v1/receipt.json"), rawBrowser: ref("evidence/gemma-attention-budget-v1/results-v1/browser.json") },
    firefoxSubmit: {
      rawTiming: ref("evidence/firefox-submit-threeway-v1/results/raw-timing-2026-11-07T08-09-37.171Z.json"),
      originalFailedReceipt: ref("evidence/firefox-submit-threeway-v1/results/runner-timing-2026-11-07T08-09-37.171Z.json"),
      correctedDerivedReceipt: ref("evidence/firefox-submit-threeway-v1/results/derived-revalidation-ratio-canonicalized-2026-11-07-v1.json"),
      revalidationSource: ref("evidence/firefox-submit-threeway-v1/revalidate-ratio-drift.mjs"),
    },
  };
  return { files, references, exclusions };
}

async function outputExists(path) {
  try { await lstat(path); return true; }
  catch (error) { if (error.code === "ENOENT") return false; throw error; }
}

async function verify() {
  const before = await lease();
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  assert.equal(manifest.schemaVersion, 1);
  assert.equal(manifest.archive.path, "evidence.json.gz");
  const compressed = await readFile(archivePath);
  assert.equal(compressed.length, manifest.archive.bytes);
  assert.equal(hash(compressed), manifest.archive.sha256);
  const decoded = gunzipSync(compressed);
  assert.equal(decoded.length, manifest.archive.decodedBytes);
  assert.equal(hash(decoded), manifest.archive.decodedSha256);
  const payload = JSON.parse(decoded);
  assert.equal(payload.schemaVersion, 1);
  assert.deepEqual(payload.references, manifest.references);
  assert.deepEqual(payload.exclusions, manifest.exclusions);
  assert.equal(payload.files.length, manifest.files.length);

  const expected = new Map(evidencePins.map(([path, bytes, sha256]) => [`evidence/${path}`, { bytes, sha256 }]));
  for (const [path, relativePath, bytes, sha256] of publicationPins) expected.set(path, { bytes, sha256 });
  expected.set("publication/docs/benchmarks/host-followup-experiments-2026-11-07/archive-publication.mjs", null);
  const seen = new Set();
  for (const [index, file] of payload.files.entries()) {
    const { data, encoding, ...descriptor } = file;
    assert.equal(encoding, "base64");
    assert.deepEqual(descriptor, manifest.files[index]);
    assert(expected.has(file.path), `unexpected archived path: ${file.path}`);
    assert(!seen.has(file.path), `duplicate archived path: ${file.path}`);
    seen.add(file.path);
    const pin = expected.get(file.path);
    if (pin) assert.deepEqual({ bytes: file.bytes, sha256: file.sha256 }, pin);
    const original = Buffer.from(data, "base64");
    assert.equal(original.length, file.bytes);
    assert.equal(hash(original), file.sha256);
  }
  assert.equal(payload.files.length, expected.size);
  const after = await lease();
  assert.deepEqual(after.identity, before.identity);
  console.log(JSON.stringify({ verified: true, files: payload.files.length, archiveSha256: manifest.archive.sha256, manifestSha256: hash(Buffer.from(JSON.stringify(manifest, null, 2) + "\n")) }));
}

async function main() {
  if (mode === "--verify") {
    await verify();
    return;
  }
  if (mode === "--plan") {
    const result = await collect(false);
    console.log(JSON.stringify({ plan: true, files: result.files, references: result.references, exclusions: result.exclusions }, null, 2));
    return;
  }

  const before = await lease();
  assert.equal(await outputExists(archivePath), false, `refusing existing output: ${archivePath}`);
  assert.equal(await outputExists(manifestPath), false, `refusing existing output: ${manifestPath}`);
  const result = await collect(true);
  const decoded = Buffer.from(JSON.stringify({ schemaVersion: 1, files: result.files, references: result.references, exclusions: result.exclusions }) + "\n");
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
    exclusions: result.exclusions,
    lease: { before, after },
  };
  await writeFile(archivePath, compressed, { flag: "wx" });
  await writeFile(manifestPath, JSON.stringify(manifest, null, 2) + "\n", { flag: "wx" });
  console.log(JSON.stringify({ archived: true, files: result.files.length, archiveSha256: hash(compressed), manifestSha256: hash(Buffer.from(JSON.stringify(manifest, null, 2) + "\n")) }));
}

await main();
