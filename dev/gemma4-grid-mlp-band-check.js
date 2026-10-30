import { Kevala } from '../js/src/index.js';

const query = new URLSearchParams(location.search), button = document.getElementById('run');
const log = (value) => { document.getElementById('log').textContent += `${JSON.stringify(value)}\n`; };
const questions = {
  person_present: { type: 'noul', instructions: 'Does the frame visibly contain at least one person?' },
  smiling: { type: 'noul', instructions: 'Does a visible person have a smiling or visibly happy facial expression? Judge only the visible expression, not inner emotion.' },
};
const modes = ['off', 'on', 'off-repeat'];
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

function frame(index) {
  const data = new Uint8ClampedArray(320 * 240 * 4);
  for (let pixel = 0; pixel < data.length / 4; pixel++) {
    data[pixel * 4] = (pixel + index * 37) % 256;
    data[pixel * 4 + 1] = (Math.floor(pixel / 320) + index * 71) % 256;
    data[pixel * 4 + 2] = (pixel * 3 + index * 19) % 256;
    data[pixel * 4 + 3] = 255;
  }
  return { width: 320, height: 240, data };
}

function compare(a, b) {
  const before = a.timing.gridMlpCheck, after = b.timing.gridMlpCheck;
  if (!(before.hiddenBits instanceof Uint32Array) || before.hiddenBits.length !== 3072 || after.hiddenBits?.length !== before.hiddenBits.length) throw new Error('Expected two complete E2B hidden rows');
  const values = new Float32Array(before.hiddenBits.buffer, before.hiddenBits.byteOffset, before.hiddenBits.length);
  if (!values.every(Number.isFinite)) throw new Error('Nonfinite baseline hidden row');
  for (const result of [a, b]) for (const key of Object.keys(questions)) {
    const probabilities = result.raw_probabilities?.[key];
    if (!Array.isArray(probabilities) || probabilities.length !== 2 || probabilities.some((value) => !Number.isFinite(value) || value < 0 || value > 1) || Math.abs(probabilities[0] + probabilities[1] - 1) > 1e-4) throw new Error(`Invalid raw probabilities for ${key}`);
  }
  let hiddenDifferences = 0;
  for (let i = 0; i < before.hiddenBits.length; i++) if (before.hiddenBits[i] !== after.hiddenBits[i]) hiddenDifferences++;
  const probabilityDifferences = JSON.stringify(a.raw_probabilities) !== JSON.stringify(b.raw_probabilities);
  const policyDifferences = JSON.stringify(before.policy) !== JSON.stringify(after.policy) || JSON.stringify(before.questions) !== JSON.stringify(after.questions);
  if (hiddenDifferences || probabilityDifferences || policyDifferences) throw new Error(`Grid band parity failed: ${JSON.stringify({ hiddenDifferences, probabilityDifferences, policyDifferences })}`);
  return { ok: true, hiddenWords: before.hiddenBits.length, hiddenDifferences, probabilityDifferences, policyDifferences };
}

async function main() {
  if (button.disabled) return;
  button.disabled = true;
  const tokens = (query.get('tokens') || '96,97,98,105,112,113').split(',').map(Number);
  if (tokens.some((value) => !Number.isSafeInteger(value) || value < 96 || value > 113)) throw new Error('tokens must be comma-separated integers from 96 to 113');
  const samples = Math.min(8, Math.max(1, Number(query.get('samples')) || 2));
  const state = window.gpuBench = { status: 'running', done: false, backend: 'webgpu', phase: 'load', cases: [], correctness: { ok: false } };
  let model;
  try {
    const response = await fetch('/.kevala/models.json');
    if (!response.ok) throw new Error(`Local model inventory HTTP ${response.status}`);
    const entry = (await response.json()).models?.['gemma-4-e2b'];
    if (!entry?.q8?.url || !entry.vision?.url) throw new Error('Local Gemma E2B Q8 decoder and vision packs are required');
    const gpuFeatures = query.has('features') ? query.get('features').split(',').filter(Boolean) : undefined;
    model = await Kevala.load({ model: new URL(entry.q8.url, location.href).href, vision: new URL(entry.vision.url, location.href).href,
      backend: 'webgpu', gpuKernel: query.get('kernel') || 'auto', imageTokens: 70, cache: false,
      ...(gpuFeatures ? { gpuFeatures } : {}),
      plugins: [new URL('./gemma4-grid-mlp-band-plugin.js', import.meta.url).href], signal: AbortSignal.timeout(240000),
      onProgress(progress) { state.phase = progress.phase; },
    });
    state.info = model.info;
    const decision = (tokens, mode, image) => model.decide({ gridMlpCheck: { tokens, mode }, text: 'Classify this camera frame.' }, questions,
      { parts: [{ type: 'image', data: frame(image) }], imageTokens: 16 });
    for (const count of tokens) {
      state.phase = `T${count}`;
      // Prime calibration and caches before the measured comparison phases.
      const prime = await decision(count, 'off', 0);
      state.adapter = prime.timing.gridMlpCheck.adapter;
      const results = new Map();
      for (const mode of modes) {
        await decision(count, mode, 0);
        const rows = [];
        for (let image = 0; image < samples; image++) {
          const start = performance.now(), result = await decision(count, mode, image);
          result.wallMs = performance.now() - start;
          const diagnostic = result.timing.gridMlpCheck;
          if (diagnostic.tokens !== count || mode !== 'on' && diagnostic.dispatches !== 0 || mode === 'on' && !diagnostic.selected && diagnostic.dispatches !== 0) throw new Error('Unexpected grid dispatch metadata');
          if ([96, 113].includes(count) && (diagnostic.selected || diagnostic.dispatches !== 0)) throw new Error('Excluded boundary selected the grid fusion');
          if (mode === 'on' && diagnostic.selected && diagnostic.dispatches < 1) throw new Error('Selected grid fusion did not dispatch');
          rows.push(result);
        }
        results.set(mode, rows);
      }
      const checks = [];
      for (let i = 0; i < samples; i++) checks.push(compare(results.get('off')[i], results.get('on')[i]), compare(results.get('off-repeat')[i], results.get('on')[i]));
      const row = { tokens: count, checks, selected: results.get('on')[0].timing.gridMlpCheck.selected,
        dispatches: results.get('on').map((result) => result.timing.gridMlpCheck.dispatches),
        calibration: results.get('on')[0].timing.gridMlpCheck.calibration,
        wallMedianMs: Object.fromEntries(modes.map((mode) => [mode, median(results.get(mode).map((result) => result.wallMs))])) };
      state.cases.push(row); log(row);
    }
    Object.assign(state, { status: 'done', done: true, phase: 'complete', correctness: { ok: true, bitExact: true, rawProbabilityExact: true },
      metricMs: median(state.cases.map((row) => row.wallMedianMs.on)), method: 'one loaded model; exact full hidden/probability off/on/off comparison; completed decision wall clock' });
  } catch (error) {
    Object.assign(state, { status: 'error', done: true, error: String(error?.message || error), errorStack: error?.stack || null }); log(error.stack || String(error));
  } finally { model?.dispose(); }
}

window.gpuBench = { status: 'ready', done: false, backend: 'webgpu' };
button.addEventListener('click', () => main().catch((error) => { window.gpuBench = { status: 'error', done: true, backend: 'webgpu', error: String(error?.message || error), errorStack: error?.stack || null }; log(error.stack || String(error)); }));
if (query.get('autorun') === '1') button.click();
