import assert from 'node:assert/strict';
import test from 'node:test';
import { calibrateGemma4GridMlp, encodeGemma4GridMlp, gemma4GridMlpBindGroups, gemma4GridMlpEligible, gemma4GridMlpMemory, selectGemma4GridMlp } from '../js/src/gpu-gemma4-grid-mlp.js';
import { GpuGemma4 } from '../js/src/gpu-gemma4.js';

const baseline = { kernel: 'matmul', f16: false, groups: 1, splitTarget: 256, mm56: 'old-mm', layout: {}, reduceLayout: {}, reduce: { 4: 'old-reduce' } };
const weight = (name) => ({ info: { name, dtype: 'q8', shape: [6144, 1536] }, buf: { name }, sbuf: { name: `${name}.scales` } });
function graph() {
  const up = { kind: 'mm', label: 'mm.up', dtype: 'q8', N: 6144, K: 1536, batched: true, mode: 0, bias: 0, group: {}, reduce: {} };
  const gelu = { kind: 'GELU', label: 'gelu', pipeline: 'old-gelu', group: {}, dispatch() {} };
  const gate = { ...up, label: 'mm.gate', params: {}, mlpPartners: [up, gelu], gridMlpSource: { globals: {}, input: {}, output: {}, gate: weight('l.0.gate'), up: weight('l.0.up') } };
  return { gate, up, gelu, ops: [gate, up, gelu] };
}
const guard = () => ({ ok: true, bitMismatches: 0, nonfinite: 0, tailMismatches: 0, outputWords: 595968, partialWords: 3575808, rawBits: true });

test('grid selection requires full exact guards and all three finite five-percent wins', () => {
  const point = { before: guard(), after: guard(), baseline: [1, 1.01, 1.02], candidate: [0.94, 0.95, 0.96] };
  assert.equal(selectGemma4GridMlp(point).selected, true);
  for (const candidate of [[0.94, 0.95, 1], [0.94, 0.95], [0.94, NaN, 0.94], [0.94, 0, 0.94]]) assert.equal(selectGemma4GridMlp({ ...point, candidate }).selected, false);
  for (const side of ['before', 'after']) for (const change of [{ bitMismatches: 1 }, { nonfinite: 1 }, { tailMismatches: 1 }, { outputWords: 1 }, { partialWords: 1 }, { rawBits: false }]) {
    assert.equal(selectGemma4GridMlp({ ...point, [side]: { ...guard(), ...change } }).reason, 'numerical-guard-failed');
  }
});

test('grid eligibility preserves actual adjacency, suffix mode, weights, precision and split policy', () => {
  const f = graph();
  assert.equal(gemma4GridMlpEligible(97, f.gate, f.up, f.gelu, baseline), true);
  for (const T of [2, 96, 98, 104, 241]) assert.equal(gemma4GridMlpEligible(T, f.gate, f.up, f.gelu, baseline), false);
  for (const change of [{ mode: 1 }, { bias: 1 }, { dtype: 'bf16' }, { sourcePolicy: true }, { batched: false }, { N: 12288 }, { K: 6144 }]) {
    assert.equal(gemma4GridMlpEligible(97, { ...f.gate, ...change }, f.up, f.gelu, baseline), false);
    const modified = graph(); Object.assign(modified.up, change);
    assert.equal(gemma4GridMlpEligible(97, modified.gate, modified.up, modified.gelu, baseline), false);
  }
  for (const change of [{ kernel: 'matmul_wide' }, { f16: true }, { groups: 2 }, { splitTarget: 128 }, { mm56: null }]) assert.equal(gemma4GridMlpEligible(97, f.gate, f.up, f.gelu, { ...baseline, ...change }), false);
  assert.equal(gemma4GridMlpEligible(97, f.gate, f.up, { ...f.gelu }, baseline), false);
  f.gate.gridMlpSource.up = weight('bad'); f.gate.gridMlpSource.up.info.shape = [12288, 1536];
  assert.equal(gemma4GridMlpEligible(97, f.gate, f.up, f.gelu, baseline), false);
});

test('binding and dispatch reuse one private paired scratch without copying inputs or weights', () => {
  const { gate } = graph(), groups = [], calls = [];
  const device = { createBindGroup(value) { groups.push(value); return value; } };
  const context = { selection: new Set(['6144:1536']), pipeline: 'grid', reduce: { getBindGroupLayout() { return {}; } }, layout: {}, part: {}, unused: {}, params: {}, splits: 3, groups: new WeakMap() };
  const bindings = gemma4GridMlpBindGroups(device, context, gate);
  const r = gate.gridMlpSource;
  assert.deepEqual(bindings.matmul.entries.map((e) => e.resource.buffer), [r.globals, gate.params, r.input, r.gate.buf, r.gate.sbuf, r.up.buf, r.up.sbuf, context.unused, context.part]);
  assert.deepEqual(bindings.reduce.entries.map((e) => e.resource.buffer), [r.globals, context.params, context.part, r.output]);
  context.groups.set(gate, bindings);
  const pass = { setPipeline(p) { calls.push(p); }, setBindGroup() {}, dispatchWorkgroups(...v) { calls.push(v); } };
  for (let i = 0; i < 2; i++) encodeGemma4GridMlp(pass, context, gate, 97, device, baseline);
  assert.equal(groups.length, 2);
  assert.deepEqual(calls.slice(0, 4), ['grid', [96, 2, 6], context.reduce, [2328, 1]]);
  assert.throws(() => encodeGemma4GridMlp(pass, context, gate, 98, device, baseline), /calibrated T97/);
  const m = gemma4GridMlpMemory();
  assert.equal(m.pair, 2 * 3 * 97 * 6144 * 4 + 16);
  assert.equal(m.totalBytes, m.retainedBytes + m.gpuBytes + m.hostBytes);
  assert.ok(m.totalBytes < 64 * 1024 * 1024);
});

test('core toggles only grid dispatch and never consumes separated or unsupported partners', () => {
  const run = (changes = {}, split = false) => {
    const f = graph(), calls = [], pass = { setPipeline(p) { calls.push(p); }, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
    const ctx = { tokens: 97, selection: new Set(['6144:1536']), pipeline: 'grid', reduce: 'grid-reduce', splits: 3, groups: new WeakMap([[f.gate, { matmul: {}, reduce: {} }]]) };
    const value = { gpuKernel: 'auto', fuseGridMlp: true, gridMlp: ctx, lastGridMlpDispatches: 0, tuning: { selection: new Map(), pipelines: { generic: baseline } }, batchMatmul: { generic: baseline }, ...changes };
    GpuGemma4.prototype.encode.call(value, { beginComputePass() { return pass; } }, split ? f.ops.slice(0, 2) : f.ops, 97, null);
    return { calls, value, ctx };
  };
  assert.deepEqual(run().calls, ['grid', 'grid-reduce']);
  assert.equal(run().value.lastGridMlpDispatches, 1);
  for (const changes of [{ fuseGridMlp: false }, { gpuKernel: 'generic' }, { batchMlp: { selection: new Set(['6144:1536']) } }]) {
    const result = run(changes); assert.deepEqual(result.calls, ['old-mm', 'old-reduce', 'old-mm', 'old-reduce', 'old-gelu']); assert.strictEqual(result.value.gridMlp, result.ctx);
  }
  assert.deepEqual(run({}, true).calls, ['old-mm', 'old-reduce', 'old-mm', 'old-reduce']);
});

test('calibrated context survives off/on/off and other token counts without rebuilding caches', async () => {
  const f = graph(), ctx = { selection: new Set(['6144:1536']), diagnostics: { enabled: true } }, prefix = {};
  const key = JSON.stringify([97, 'auto', 'generic', 'matmul', false, 1, 256, true, false, 'l.0.gate', 'l.0.up']);
  const gpu = { gpuKernel: 'auto', batchMatmul: { generic: baseline }, tuning: { selection: new Map(), diagnostics: {} }, batchPrefixSuffixOps: f.ops, gridMlp: ctx, gridMlpKey: key, prefixRetile: prefix };
  for (const enabled of [false, true, false]) {
    gpu.fuseGridMlp = enabled;
    assert.strictEqual(await GpuGemma4.prototype.prepareGridMlp.call(gpu, 97), ctx);
    assert.strictEqual(gpu.prefixRetile, prefix);
    assert.equal(gpu.tuning.diagnostics.gridMlp.enabled, enabled);
    assert.deepEqual(structuredClone(gpu.tuning.diagnostics.gridMlp), gpu.tuning.diagnostics.gridMlp);
  }
  assert.strictEqual(await GpuGemma4.prototype.prepareGridMlp.call(gpu, 241), ctx);
  assert.equal(gpu.tuning.diagnostics.gridMlp.enabled, false);
});

test('unsupported calibration exits before GPU work and optional compile failure preserves baseline', async () => {
  const shape = graph().gate.gridMlpSource, unavailable = () => { throw new Error('must not compile'); };
  const device = { features: new Set() };
  for (const opts of [{ automatic: false }, { pairedSelected: true }, { tokens: 96 }, { baseline: { ...baseline, f16: true } }, {}]) {
    const result = await calibrateGemma4GridMlp(device, unavailable, shape, { tokens: 97, baseline, ...opts });
    assert.equal(result.selection.size, 0); assert.deepEqual(structuredClone(result.diagnostics), result.diagnostics);
  }
});

test('compile errors and device loss balance scopes and release all allocated context buffers', async () => {
  const savedStage = globalThis.GPUShaderStage, savedUsage = globalThis.GPUBufferUsage;
  globalThis.GPUShaderStage = { COMPUTE: 4 };
  globalThis.GPUBufferUsage = { STORAGE: 1, UNIFORM: 2, COPY_DST: 4, COPY_SRC: 8, MAP_READ: 16, QUERY_RESOLVE: 32 };
  try {
    for (const failureAt of [1, 2]) for (const loss of [null, 'runtime', 'message', 'scheduled', 'healthy']) {
      let writes = 0, notifyLoss, notification, timeout;
      const scopes = [], buffers = [], runtime = {}, device = {
        features: new Set(['timestamp-query']), limits: { maxBufferSize: 1e9, maxStorageBufferBindingSize: 1e9, maxComputeWorkgroupStorageSize: 32768 },
        pushErrorScope(v) { scopes.push(v); }, async popErrorScope() { scopes.pop(); return null; },
        createBindGroupLayout(v) { return v; }, createPipelineLayout(v) { return v; }, createShaderModule(v) { return v; },
        async createComputePipelineAsync() { return {}; },
        createBuffer({ size }) { const b = { size, destroyed: 0, destroy() { this.destroyed++; } }; buffers.push(b); return b; },
        queue: { writeBuffer() {
          if (++writes !== failureAt) return;
          if (loss === 'runtime') runtime.lost = { message: 'adapter removed' };
          if (loss === 'scheduled') notification = setTimeout(() => notifyLoss({ message: 'adapter removed after upload failure' }), 0);
          throw new Error(loss === 'message' ? 'device-lost during upload' : 'upload unavailable');
        } },
      };
      if (loss === 'scheduled' || loss === 'healthy') device.lost = new Promise((resolve) => { notifyLoss = resolve; });
      try {
        const attempt = calibrateGemma4GridMlp(device, () => '', graph().gate.gridMlpSource, { tokens: 97, baseline, gelu: {}, runtime });
        if (loss && loss !== 'healthy') await assert.rejects(attempt, /device lost/i);
        else {
          const result = await Promise.race([attempt, new Promise((resolve) => { timeout = setTimeout(() => resolve(null), 100); })]);
          assert.notEqual(result, null, 'healthy device with unresolved lost promise must fall back promptly');
          assert.equal(result.selection.size, 0); assert.equal(result.diagnostics.reason, 'calibration-failed'); assert.deepEqual(structuredClone(result.diagnostics), result.diagnostics);
        }
        assert.equal(buffers.length, failureAt === 1 ? 3 : 4); assert.ok(buffers.every((b) => b.destroyed === 1)); assert.equal(scopes.length, 0);
      } finally { clearTimeout(notification); clearTimeout(timeout); }
    }
  } finally { if (savedStage === undefined) delete globalThis.GPUShaderStage; else globalThis.GPUShaderStage = savedStage; if (savedUsage === undefined) delete globalThis.GPUBufferUsage; else globalThis.GPUBufferUsage = savedUsage; }
});

test('weight replacement and model disposal release the retained calibration context', () => {
  for (const method of ['write', 'dispose']) {
    let destroyed = 0, written = 0;
    const value = { gridMlp: { destroy() { destroyed++; } }, gridMlpKey: 'old-policy',
      batchSingleCache: new Map(), batchMlpCache: new Map(), prefixRetileCache: new Map(), pleCache: new Map(),
      clearPleVectorCache() {}, clearTextPrefixCache() {}, textPrefixCache: { dispose() {} }, weights: { write() { written++; } } };
    GpuGemma4.prototype[method].call(value, 0, new Uint8Array());
    assert.equal(destroyed, 1); assert.equal(value.gridMlp, null); assert.equal(value.gridMlpKey, null);
    assert.equal(written, Number(method === 'write'));
  }
});
