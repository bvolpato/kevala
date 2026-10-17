// The PLE gate consumes a completed split reduction, then the unchanged Gemma GELU.
import { inErrorScopes, mmSplits, pipeline } from "./gpu.js";

export async function createGemma4PleReduce(device, source, { enabled = true } = {}) {
  const result = { layout: null, variants: {}, diagnostics: { enabled: false, reason: enabled ? "pipeline-unavailable" : "disabled" } };
  if (!enabled) return result;
  try {
    return await inErrorScopes(device, async () => {
      const types = ["uniform", "uniform", "read-only-storage", "read-only-storage", "storage", "read-only-storage", "storage"];
      const layout = device.createBindGroupLayout({ label: "Gemma PLE reducer", entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] }), compiled = new Map(), variants = {};
      for (const [policy, kernel] of [["plain", "reduce_ple"], ["batch", "reduce_ple_gemma_batch"], ["source", "reduce_ple_gemma_source"]]) {
        variants[policy] = [];
        for (let rows = 1; rows <= 4; rows++) {
          const code = source(kernel, { weights: "q8", f16: false, rows, groups: 1, splitTarget: 256 });
          if (!compiled.has(code)) compiled.set(code, await pipeline(device, code, `${kernel}_r${rows}`, pipelineLayout));
          variants[policy][rows] = compiled.get(code);
        }
      }
      return { layout, variants, diagnostics: { enabled: true, reason: "compiled", pipelines: compiled.size, sharedBytes: 1024 } };
    });
  } catch (error) {
    return { ...result, diagnostics: { enabled: false, reason: "pipeline-failed", error: String(error?.message || error) } };
  }
}

export function gemma4PleReduceGroup(device, fusion, { globals, params, partials, bias, raw, ple, output }) {
  if (!fusion?.layout) return null;
  const resource = (buffer) => buffer.buffer ? buffer : { buffer };
  return device.createBindGroup({ layout: fusion.layout, entries: [globals, params, partials, bias, raw, ple, output].map((buffer, binding) => ({ binding, resource: resource(buffer) })) });
}

export function gemma4PleReduceAdjacent(op, next, tokens, enabled) {
  return enabled === true && Number.isSafeInteger(tokens) && tokens > 8 && tokens <= 256 && op?.kind === "mm" && op.label === "mm.ple.gate" &&
    !!op.pleReduce && op.pleReduce.partner === next && next?.kind === "GELU" && next.label === "gelu.ple" && op.pleReduce.N === op.N && op.pleReduce.K === op.K;
}

export function gemma4PleReduceChoice(fusion, op, next, tokens, pipes, rows, splitTokens, policy, retiled = false) {
  if (!gemma4PleReduceAdjacent(op, next, tokens, true) || !Number.isSafeInteger(rows) || rows < 1 || rows > 4 ||
    !Number.isSafeInteger(splitTokens) || splitTokens < 1 || pipes?.groups !== 1 || pipes.splitTarget !== 256 || !pipes.reduce?.[rows]) return null;
  const pipeline = fusion?.variants?.[policy]?.[rows], group = retiled ? op.pleReduce.retiledGroup : op.pleReduce.group;
  if (!pipeline || !group || !Number.isSafeInteger(op.N) || op.N < 1 || !Number.isSafeInteger(op.K) || op.K < 128) return null;
  const splits = mmSplits(splitTokens, op.N, op.K, pipes.splitTarget, 16 * rows, 64 * pipes.groups);
  if (splits <= 1) return null;
  const key = retiled ? "retiledOverrides" : "overrides";
  const overrides = op.pleReduce[key] ||= new Map();
  let choice = overrides.get(pipeline);
  if (!choice || choice.group !== group) { choice = { pipeline, group }; overrides.set(pipeline, choice); }
  return choice;
}
