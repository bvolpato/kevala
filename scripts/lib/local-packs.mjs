import { lstatSync } from "node:fs";
import { join, resolve } from "node:path";

const PRECISIONS = new Set(["bf16", "q8", "f32"]);
const MODEL_ID = /^[a-z0-9][a-z0-9.-]*$/;

function expectedPacks(spec) {
  if (spec?.precisionPackBytes && typeof spec.precisionPackBytes === "object") {
    return Object.entries(spec.precisionPackBytes).filter(([precision, bytes]) =>
      PRECISIONS.has(precision)
      && spec.precisionPackApprox?.[precision] !== true
      && Number.isSafeInteger(bytes)
      && bytes > 0,
    );
  }

  if (!Number.isSafeInteger(spec?.pack) || spec.pack <= 0) return [];
  const hostedPrecision = /-(bf16|q8|f32)\.kevala(?:$|[?#])/.exec(spec.hosted || "")?.[1];
  const precision = hostedPrecision || spec.defaultPrecision || "q8";
  return PRECISIONS.has(precision) && spec.precisionPackApprox?.[precision] !== true
    ? [[precision, spec.pack]]
    : [];
}

/** Find catalog packs that are present locally and whose exact sizes match the catalog. */
export function discoverLocalPacks(root, models) {
  const directory = join(resolve(root), "tmp");
  const manifest = { models: {} };
  const files = new Map();
  let tmpDirectory;
  try {
    tmpDirectory = lstatSync(directory);
  } catch {
    return { manifest, files };
  }
  if (!tmpDirectory.isDirectory()) return { manifest, files };

  for (const [modelId, spec] of Object.entries(models)) {
    if (!MODEL_ID.test(modelId)) continue;
    for (const [precision, bytes] of expectedPacks(spec)) {
      const filename = `${modelId}-${precision}.kevala`;
      const path = join(directory, filename);
      let file;
      try {
        file = lstatSync(path);
      } catch {
        continue;
      }
      // Ignore symlinks and incomplete or stale conversions. No pack contents are read or hashed.
      if (!file.isFile() || file.size !== bytes) continue;

      const url = `/.kevala/packs/${filename}`;
      (manifest.models[modelId] ||= {})[precision] = { url, bytes };
      files.set(url, { path, bytes });
    }
    const vision = spec.vision;
    if (vision?.arch === "gemma4-vision" && vision.precision === "bf16" && Number.isSafeInteger(vision.pack) && vision.pack > 0 && vision.pack <= 1024 ** 3) {
      const path = join(directory, `${modelId}-vision-bf16.kevala`);
      let file;
      try { file = lstatSync(path); } catch { continue; }
      if (!file.isFile() || file.size !== vision.pack) continue;
      const url = `/.kevala/packs/${modelId}-vision-bf16.kevala`;
      (manifest.models[modelId] ||= {}).vision = { url, bytes: vision.pack, precision: "bf16" };
      files.set(url, { path, bytes: vision.pack });
    }
  }

  return { manifest, files };
}
