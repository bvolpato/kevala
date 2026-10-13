import { MODELS, precisionOptions } from "../js/src/source.js";
import { PREVIEW_MODEL_IDS } from "./model-policy.js";

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
    }
    return available;
  } catch {
    return {};
  }
}
