import { requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { createGemma4BatchMlpValidator, gemma4BatchMlpValidationGroups } from "../js/src/gpu-gemma4-batch-mlp.js";

const log = document.getElementById("log");
const FLOATS = 64;
const BYTES = FLOATS * 4;
const POISON = 0x7fc12345;
const SENTINEL = 0x42f6e979;
const STATUS_BYTES = 16;
const GPU_SCRATCH_BYTES = 16 + STATUS_BYTES + BYTES * 2 + 2 * BYTES + STATUS_BYTES;
const metadata = {};
const cases = [];

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function makeFiniteBits(length, salt = 0) {
  const values = new Float32Array(length);
  for (let i = 0; i < length; i++) values[i] = Math.fround(0.125 + ((i * 17 + salt * 13) % 97) / 31);
  return new Uint32Array(values.buffer);
}

function buffer(device, size, usage, label) {
  return device.createBuffer({ label, size, usage });
}

function bindValidationGroups(device, validator, params, baseline, candidate, status) {
  return gemma4BatchMlpValidationGroups(device, validator, { params, baseline, candidate, status });
}

async function mapWords(readback, bytes) {
  await readback.mapAsync(GPUMapMode.READ, 0, bytes);
  try {
    return new Uint32Array(readback.getMappedRange(0, bytes).slice(0));
  } finally {
    readback.unmap();
  }
}

function dispatch(pass, pipeline, group, lengthFloats) {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  const workgroups = Math.ceil(lengthFloats / 256);
  pass.dispatchWorkgroups(Math.min(workgroups, 65535), Math.ceil(workgroups / 65535));
}

async function main() {
  assert(GPU_SCRATCH_BYTES < 1024 * 1024, "validator scratch exceeds 1 MiB");
  const started = performance.now();
  const runtime = await requestDevice({ features: [] });
  const { device } = runtime;
  const U = GPUBufferUsage;
  const owned = [];
  const make = (size, usage, label) => {
    const value = buffer(device, size, usage, label);
    owned.push(value);
    return value;
  };

  metadata.adapter = {
    name: runtime.name,
    vendor: runtime.adapter.info?.vendor || null,
    architecture: runtime.adapter.info?.architecture || null,
    device: runtime.adapter.info?.device || null,
    description: runtime.adapter.info?.description || null,
    backend: runtime.adapter.info?.backend || null,
    driver: runtime.adapter.info?.driver || null,
    type: runtime.adapter.info?.type || null,
    isFallbackAdapter: Boolean(runtime.adapter.info?.isFallbackAdapter ?? runtime.adapter.isFallbackAdapter),
    features: [...device.features].map(String).sort(),
    maxBufferSize: device.limits.maxBufferSize,
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
  };
  assert(!metadata.adapter.isFallbackAdapter, "validator requires a hardware WebGPU adapter");

  try {
    await withGpuErrors(runtime, async () => {
      const validator = await createGemma4BatchMlpValidator(device);
      const params = make(16, U.UNIFORM | U.COPY_DST, "batch-validator-params");
      const status = make(STATUS_BYTES, U.STORAGE | U.COPY_DST | U.COPY_SRC, "batch-validator-status");
      const baseline = make(BYTES, U.STORAGE | U.COPY_DST | U.COPY_SRC, "batch-validator-baseline");
      const candidate = make(BYTES, U.STORAGE | U.COPY_DST | U.COPY_SRC, "batch-validator-candidate");
      const readback = make(2 * BYTES + STATUS_BYTES, U.MAP_READ | U.COPY_DST, "batch-validator-readback");

      const checkFill = async () => {
        const activeFloats = 37;
        device.queue.writeBuffer(params, 0, new Uint32Array([FLOATS, activeFloats, SENTINEL, POISON]));
        const groups = bindValidationGroups(device, validator, params, baseline, candidate, status);
        const encoder = device.createCommandEncoder({ label: "fill poison and tail sentinels" });
        encoder.clearBuffer(status);
        const pass = encoder.beginComputePass({ label: "batch validator fill contract" });
        dispatch(pass, validator.fill, groups.fill[0], FLOATS);
        dispatch(pass, validator.fill, groups.fill[1], FLOATS);
        pass.end();
        encoder.copyBufferToBuffer(baseline, 0, readback, 0, BYTES);
        encoder.copyBufferToBuffer(candidate, 0, readback, BYTES, BYTES);
        encoder.copyBufferToBuffer(status, 0, readback, 2 * BYTES, STATUS_BYTES);
        device.queue.submit([encoder.finish()]);
        const words = await mapWords(readback, 2 * BYTES + STATUS_BYTES);
        const expected = new Uint32Array(FLOATS);
        expected.fill(SENTINEL);
        expected.fill(POISON, 0, activeFloats);
        const expectedStatus = [0, 0, 0, 0];
        const first = words.subarray(0, FLOATS), second = words.subarray(FLOATS, 2 * FLOATS);
        const statusWords = [...words.subarray(2 * FLOATS, 2 * FLOATS + 4)];
        const ok = first.every((word, index) => word === expected[index])
          && second.every((word, index) => word === expected[index])
          && statusWords.every((word, index) => word === expectedStatus[index]);
        return { name: "fill-poison-active-and-sentinel-tail", ok, lengthFloats: FLOATS, activeFloats, poisonBits: POISON, sentinelBits: SENTINEL, status: statusWords, scratchBytes: GPU_SCRATCH_BYTES };
      };

      const checkCompare = async ({ name, activeFloats, baselineBits, candidateBits, expected }) => {
        const lengthFloats = baselineBits.length;
        assert(lengthFloats <= FLOATS && candidateBits.length === lengthFloats, `${name}: invalid case length`);
        device.queue.writeBuffer(params, 0, new Uint32Array([lengthFloats, activeFloats, SENTINEL, POISON]));
        device.queue.writeBuffer(baseline, 0, baselineBits);
        device.queue.writeBuffer(candidate, 0, candidateBits);
        const groups = bindValidationGroups(device, validator, params, baseline, candidate, status);
        const encoder = device.createCommandEncoder({ label: `compare ${name}` });
        encoder.clearBuffer(status);
        const pass = encoder.beginComputePass({ label: `batch validator compare ${name}` });
        dispatch(pass, validator.compare, groups.compare, lengthFloats);
        pass.end();
        encoder.copyBufferToBuffer(status, 0, readback, 0, STATUS_BYTES);
        device.queue.submit([encoder.finish()]);
        const statusWords = [...(await mapWords(readback, STATUS_BYTES))];
        const result = { name, status: statusWords, lengthFloats, activeFloats, scratchBytes: GPU_SCRATCH_BYTES, ...expected };
        result.ok = expected.detected
          ? (expected.bitMismatches === undefined || statusWords[0] === expected.bitMismatches)
            && (expected.activeNonfiniteValues === undefined || statusWords[1] === expected.activeNonfiniteValues)
            && (expected.tailMismatches === undefined || statusWords[2] === expected.tailMismatches)
          : statusWords.every((value) => value === 0);
        delete result.detected;
        return result;
      };

      const runRound = async (round) => {
        const roundCases = [];
        roundCases.push(await checkFill());

        // PART is plane-major. This deliberately targets the final value in the
        // fourth plane, where truncated range checks can miss stale data.
        const planeFloats = 16, planeCount = 4, lastPartIndex = planeFloats * planeCount - 1;
        const partBase = makeFiniteBits(planeFloats * planeCount, 1);
        roundCases.push(await checkCompare({ name: "four-plane-PART-identical-finite", activeFloats: partBase.length, baselineBits: partBase, candidateBits: partBase.slice(),
          expected: { detected: false, planeCount, planeFloats } }));
        const partCandidate = partBase.slice();
        partCandidate[lastPartIndex] ^= 1;
        roundCases.push(await checkCompare({ name: "four-plane-PART-last-plane-last-value", activeFloats: partBase.length, baselineBits: partBase, candidateBits: partCandidate,
          expected: { detected: true, bitMismatches: 1, activeNonfiniteValues: 0, tailMismatches: 0, planeCount, planeFloats, corruptedIndex: lastPartIndex } }));

        const zeroBase = new Uint32Array([0x00000000, 0x3f800000]);
        const zeroCandidate = zeroBase.slice(); zeroCandidate[0] = 0x80000000;
        roundCases.push(await checkCompare({ name: "signed-zero-bit-mismatch", activeFloats: 2, baselineBits: zeroBase, candidateBits: zeroCandidate,
          expected: { detected: true, bitMismatches: 1, activeNonfiniteValues: 0, tailMismatches: 0 } }));

        const nanBase = new Uint32Array([0x7fc12345, 0x3f800000]);
        roundCases.push(await checkCompare({ name: "active-NaN-in-baseline", activeFloats: 2, baselineBits: nanBase, candidateBits: new Uint32Array([0x3f800000, 0x3f800000]),
          expected: { detected: true, bitMismatches: 1, activeNonfiniteValues: 1, tailMismatches: 0 } }));

        const infCandidate = new Uint32Array([0x3f800000, 0x7f800000]);
        roundCases.push(await checkCompare({ name: "active-infinity-in-candidate", activeFloats: 2, baselineBits: new Uint32Array([0x3f800000, 0x3f800000]), candidateBits: infCandidate,
          expected: { detected: true, bitMismatches: 1, activeNonfiniteValues: 1, tailMismatches: 0 } }));

        const equalNonfinite = new Uint32Array([0x7fc12345, 0x7f800000]);
        roundCases.push(await checkCompare({ name: "reject-equal-active-NaN-and-infinity", activeFloats: 2, baselineBits: equalNonfinite, candidateBits: equalNonfinite.slice(),
          expected: { detected: true, bitMismatches: 0, activeNonfiniteValues: 4, tailMismatches: 0 } }));

        const tailLength = 16, activeFloats = 11;
        const tailBase = makeFiniteBits(tailLength, 4);
        tailBase.fill(SENTINEL, activeFloats);
        roundCases.push(await checkCompare({ name: "active-values-and-sentinel-tail-match", activeFloats, baselineBits: tailBase, candidateBits: tailBase.slice(),
          expected: { detected: false } }));
        const sameTailCorruption = tailBase.slice(); sameTailCorruption[12] ^= 1;
        roundCases.push(await checkCompare({ name: "matching-but-corrupt-tail-sentinel", activeFloats, baselineBits: sameTailCorruption, candidateBits: sameTailCorruption.slice(),
          expected: { detected: true, bitMismatches: 0, activeNonfiniteValues: 0, tailMismatches: 2, corruptedIndex: 12 } }));

        const divergentTail = tailBase.slice(); divergentTail[12] ^= 1;
        roundCases.push(await checkCompare({ name: "divergent-tail-sentinel", activeFloats, baselineBits: tailBase, candidateBits: divergentTail,
          expected: { detected: true, bitMismatches: 1, activeNonfiniteValues: 0, tailMismatches: 1, corruptedIndex: 12 } }));

        for (const result of roundCases) result.round = round;
        return roundCases;
      };

      const first = await runRound("independent-guard-round-1");
      const second = await runRound("independent-guard-round-2");
      cases.push(...first, ...second);
      const failed = cases.find((value) => !value.ok);
      if (failed) throw new Error(`${failed.name}: validator returned unexpected counters ${failed.status?.join(",") || "fill-contract mismatch"}`);
    }, "running Gemma batch MLP validator checks");

    const elapsedMs = performance.now() - started;
    window.gpuBench = {
      status: "done",
      done: true,
      backend: "webgpu",
      ...metadata,
      elapsedMs,
      method: "validator correctness only; two independent rounds; no selected-kernel timing",
      metricMs: 1,
      metricNote: "Positive contract sentinel for bench-gpu.py; this validator page makes no performance claim.",
      scratchBytes: GPU_SCRATCH_BYTES,
      scratchLimitBytes: 1024 * 1024,
      correctness: {
        ok: true,
        bitExact: true,
        tolerance: 0,
        signedZeroChecked: true,
        expectedFailureCasesDetected: true,
        gpuDispatches: 2 * (2 + 9),
        selectedTiming: "not measured",
        cases,
      },
    };
    log.textContent = JSON.stringify(window.gpuBench, null, 2);
  } finally {
    for (const value of owned) value.destroy();
    device.destroy();
  }
}

window.gpuBench = { status: "running", done: false, backend: "webgpu", method: "validator correctness only" };
main().catch((error) => {
  const message = String(error?.message || error);
  window.gpuBench = {
    status: "error",
    done: true,
    backend: "webgpu",
    ...metadata,
    method: "validator correctness only; no selected-kernel timing",
    error: message,
    correctness: { ok: false, bitExact: true, tolerance: 0, cases },
  };
  log.textContent = String(error?.stack || message);
});
