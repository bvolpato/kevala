import { runPleStoreBenchmark } from "./gemma4-ple-store-check.js";

self.onmessage = async ({ data }) => {
  if (data.type !== "run") return;
  self.onmessage = null;
  try {
    const result = await runPleStoreBenchmark("worker", (resource) => self.postMessage({ type: "resource", resource }), (value) => self.postMessage({ type: "progress", value }), data.resource);
    self.postMessage({ type: "result", result });
  } catch (error) {
    self.postMessage({ type: "error", error: String(error.stack || error) });
  }
};
