const WORDS = "the customer wrote in about a recurring charge they did not recognise and wants a refund today ";
export const QUESTION = Object.freeze({ type: "noul", instructions: "Is the customer asking for money back?" });
export const WARMUP_RUNS = Object.freeze([0, 1, 2]);
export const MEASURED_RUNS = Object.freeze([3, 4, 5, 6, 7, 8, 9, 10, 11]);
export const QUARTETS = 9;
export const ORDERS = Object.freeze([
  Object.freeze(["await", "split", "split", "await"]),
  Object.freeze(["split", "await", "await", "split"]),
]);

export const FIXTURES = Object.freeze([
  Object.freeze({
    id: "short-47",
    name: "short, 47 tokens",
    expectedTokens: 47,
    state: "Refund the duplicate charge today or we cancel.",
    measuredHashes: Object.freeze([
      "b17490533a833d19b85a141891c14b4adc1c0afa819c37aaf15e04fe9e54e998",
      "6fbef706185090b11265a936cfd0f81c8d2d1c04bcfdbad63f863d2770f0888a",
      "6b8fb032bdd35b6a53f14f18b3cec6570d3f9e98032537fc76b89b2247b7c6b4",
      "d0075bf6c7e3ea4ba1e2499cdc94479195e79969806d842ea3ebe65b3b6fa713",
      "346cf27aedbc22eb91aacecc987318fa89f68d0142acad7c4816e1428d092619",
      "31bef6681aa685bd06404fba2c0a4ae706eef428ab481420e9f33e2789cb025c",
      "d2500d93757164370649a342d7267e191b77b54cc527aed111b037a3259871b5",
      "cc00d828652e08a69f9a47f3436fcd52bcff793414a083a3cbc3d97e71e37f76",
      "da4958c1f42e048d17f5b8173f8e480ddda107d2acf81a9aed533fd4b3ed989c",
    ]),
  }),
  Object.freeze({
    id: "long-512",
    name: "long, 512 tokens",
    expectedTokens: 512,
    state: WORDS.repeat(30),
    measuredHashes: Object.freeze([
      "aefa250182c6163f4c20a16e94bd1daac5b57492cf7dc8ec3e09fa046a08382d",
      "6d36df73b91e44d0b6c86c636deabcb814a7bf3cb608b8d8cef6437f29dc65a8",
      "9a0fa4667c47d4d11897f013d271db5808071818aec44a8f79d08e7371dc053b",
      "c3ba9301cd12a9cf1cd44f322c7f11c202ddb47f636e0acb4bd7f4d10d5fc81b",
      "c0916a5e43a87458d8d8534fc60c718507c54f0deb4683abd073b5780a0af7df",
      "b3bd4d7c432cc99f8fbda85ac78a50b9a3dfe8f32370cfc53fb92f87e41b16fd",
      "414bee9d8f5ccc0418724c977e60d9a361b912bc4bb6b9d5cd981af7ec8aa522",
      "57f897e5e2053d2a3aa5ce85ab26aa9295fb25c020d4e7d6927a8f1730f18957",
      "7d15f059814579e734c6127cd0d88127e8983291c20997d7ac0dfec7caf7c106",
    ]),
  }),
]);

export function requestFor(fixture, run) {
  return { state: `Run ${run}. ${fixture.state}`, questions: { q: QUESTION } };
}

export function canonicalJson(value) {
  const normalize = (item) => {
    if (item === undefined || typeof item === "function" || typeof item === "symbol") return null;
    if (typeof item === "number") return Number.isFinite(item) ? item : null;
    if (Array.isArray(item)) return item.map(normalize);
    if (item && typeof item === "object") return Object.fromEntries(Object.keys(item).sort().map((key) => [key, normalize(item[key])]));
    return item;
  };
  return JSON.stringify(normalize(value));
}

export function jsonSnapshot(value) {
  const json = canonicalJson(value);
  return { json, snapshot: JSON.parse(json) };
}
