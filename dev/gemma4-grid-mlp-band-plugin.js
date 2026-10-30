// Development-only architecture wrapper. The public runtime loads this only when requested.
import gemma4 from '../js/src/archs/gemma4.js';

const encoder = new TextEncoder();
const selection = (value) => value instanceof Map ? [...value].sort(([a], [b]) => String(a).localeCompare(String(b))) : value instanceof Set ? [...value].sort() : [];

function suffixTokens(engine, request, questions) {
  const parts = request.parts.map((part) => part.type === 'image' ? { type: 'image', image_tokens: 1, image_index: 0 } : part);
  engine.coord.withInput(encoder.encode(JSON.stringify({ requests: [{ ...request, questions, parts }] })), (p, n) => engine.coord.check(engine.coord.x.kevala_gemma4_prepare_images(p, n)));
  const sequences = JSON.parse(engine.coord.outText()).sequences.map((sequence) => sequence.ids);
  if (sequences.length !== 2) throw new Error('Grid band check requires two question sequences');
  let shared = 0;
  while (shared < Math.min(...sequences.map((ids) => ids.length)) && sequences.every((ids) => ids[shared] === sequences[0][shared])) shared++;
  return sequences.reduce((sum, ids) => sum + ids.length - shared, 0);
}

function questionsFor(engine, request, target) {
  const cache = engine.gridBandQuestions ||= new Map();
  const key = JSON.stringify([target, request.state, request.questions, request.parts.filter((part) => part.type !== 'image')]);
  if (cache.has(key)) return cache.get(key);
  if (suffixTokens(engine, request, request.questions) === target) { cache.set(key, request.questions); return request.questions; }
  for (let extra = 0; extra <= 48; extra++) {
    const questions = { ...request.questions, person_present: { ...request.questions.person_present,
      instructions: `Does the image show a person?${' carefully'.repeat(extra)}` } };
    if (suffixTokens(engine, request, questions) === target) { cache.set(key, questions); return questions; }
  }
  throw new Error(`No question wording produced suffix T${target}`);
}

export default {
  ...gemma4,
  async createGpu(...args) {
    const gpu = await gemma4.createGpu(...args);
    const prepare = gpu.prepareGridMlp;
    gpu.prepareGridMlp = async function(tokens, ...rest) {
      this.gridBandTokens = tokens;
      return prepare.call(this, tokens, ...rest);
    };
    const forward = gpu.forwardSharedPrefix;
    gpu.forwardSharedPrefix = async function(...inputs) {
      const result = await forward.apply(this, inputs);
      this.gridBandTokens = this.lastSourceCompaction?.originalRows ?? this.gridBandTokens;
      this.gridBandHidden = result instanceof Float32Array ? new Uint32Array(result.buffer, result.byteOffset, result.length).slice() : null;
      return result;
    };
    return gpu;
  },
  async run(engine, requests) {
    if (requests.every((request) => request.state?.gridMlpCheck === undefined)) return gemma4.run(engine, requests);
    const directive = requests[0]?.state?.gridMlpCheck;
    if (requests.length !== 1 || !directive || !['off', 'on', 'off-repeat'].includes(directive.mode) ||
        !Number.isSafeInteger(directive.tokens) || directive.tokens < 96 || directive.tokens > 113) {
      throw new Error('Grid band check requires one explicit off/on request for T96 to T113');
    }
    const gpu = engine.gpu, previous = gpu.fuseGridMlp;
    const request = { ...requests[0], state: requests[0].state.text };
    request.questions = questionsFor(engine, request, directive.tokens);
    gpu.fuseGridMlp = directive.mode === 'on';
    gpu.gridBandHidden = null;
    gpu.gridBandTokens = null;
    try {
      const result = await gemma4.run(engine, [request]);
      if (gpu.gridBandTokens !== directive.tokens) throw new Error(`Expected suffix T${directive.tokens}, got ${gpu.gridBandTokens}`);
      const context = gpu.gridMlp;
      const adapter = gpu.runtime.adapter, info = adapter.info || {};
      result.timing.gridMlpCheck = {
        mode: directive.mode, tokens: gpu.gridBandTokens, hiddenBits: gpu.gridBandHidden,
        adapter: { name: gpu.name, vendor: info.vendor || null, description: info.description || null,
          isFallbackAdapter: info.isFallbackAdapter === true || adapter.isFallbackAdapter === true, features: [...gpu.device.features] },
        dispatches: gpu.lastGridMlpDispatches, selected: context?.tokens === directive.tokens && context.selection.has('6144:1536'),
        calibration: context?.tokens === directive.tokens ? context.diagnostics : null,
        policy: {
          matmul: selection(gpu.tuning.selection), matvec: selection(gpu.matvec.selection),
          matvecTwo: [selection(gpu.matvecTwo.selection.plain), selection(gpu.matvecTwo.selection.gelu)],
          prefixRetile: selection(gpu.prefixRetile?.selection), batch: selection(gpu.batchMlp?.selection),
          visionQkv: engine.vision?.qkv?.selected, imageTokens: request.imageTokens,
        },
        questions: request.questions,
      };
      return result;
    } finally {
      gpu.gridBandHidden = null;
      gpu.fuseGridMlp = previous;
    }
  },
};
