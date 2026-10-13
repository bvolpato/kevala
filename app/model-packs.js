import { MODELS, precisionOptions } from "../js/src/source.js";
import { PREVIEW_MODEL_IDS } from "./model-policy.js";

const IMAGE_MODEL = "gemma-4-e2b";
const MAX_VISION_PACK_BYTES = 1024 ** 3;
const VISION_PACK_PATH = "/.kevala/packs/gemma-4-e2b-vision-bf16.kevala";

/** Discover packs advertised by the preview server, without fetching model weights. */
export async function discoverLocalPacks(base, fetcher = fetch) {
  const endpoint = new URL("../.kevala/models.json", base);
  try {
    const response = await fetcher(endpoint, { signal: AbortSignal.timeout(2000) });
    if (!response.ok) return {};
    const { models } = await response.json();
    const available = {};
    for (const id of Object.keys(MODELS)) {
      if (!PREVIEW_MODEL_IDS.has(id)) continue;
      for (const precision of precisionOptions(id)) {
        const pack = models?.[id]?.[precision];
        if (!pack || !Number.isSafeInteger(pack.bytes) || pack.bytes <= 0 || typeof pack.url !== "string") continue;
        const url = new URL(pack.url, endpoint);
        if (url.origin !== endpoint.origin || !url.pathname.startsWith("/.kevala/packs/")) continue;
        (available[id] ||= {})[precision] = { url: url.href, bytes: pack.bytes };
      }
      if (id === IMAGE_MODEL && MODELS[id]?.vision?.precision === "bf16") {
        const pack = models?.[id]?.vision;
        if (!pack || !Number.isSafeInteger(pack.bytes) || pack.bytes <= 0 || pack.bytes > MAX_VISION_PACK_BYTES
          || pack.bytes !== MODELS[id].vision.pack || typeof pack.url !== "string") continue;
        let url;
        try {
          url = new URL(pack.url, endpoint);
        } catch {
          continue;
        }
        if (url.origin !== endpoint.origin || url.pathname !== VISION_PACK_PATH || url.search || url.hash) continue;
        (available[id] ||= {}).vision = { url: url.href, bytes: pack.bytes, precision: "bf16" };
      }
    }
    return available;
  } catch {
    return {};
  }
}
