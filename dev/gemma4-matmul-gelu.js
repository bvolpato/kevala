// Candidate-only helpers. Runtime integration requires calibration and model parity checks.
import { mmSplits, pipeline } from "../js/src/gpu.js";

export async function pairedMatmulGeluPipelines(device, source, { f16 = false, rows = 4, row56 = false, splitTarget = 256 } = {}) {
  if (![1, 2, 3, 4].includes(rows)) throw new Error("paired matmul rows must be 1..4");
  if (row56 && (f16 || rows !== 4)) throw new Error("paired BM56 requires F32 R4");
  const sharedBytes = f16 ? 12288 : 24576;
  if (device.limits.maxComputeWorkgroupStorageSize < sharedBytes) throw new Error(`paired matmul requires ${sharedBytes} shared bytes`);
  const layoutFor = (label, types) => device.createBindGroupLayout({ label, entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
  const layout = layoutFor("paired matmul GELU", ["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage", "storage"]);
  const reduceLayout = layoutFor("paired matmul GELU reduce", ["uniform", "uniform", "read-only-storage", "storage"]);
  const spec = { weights: "q8", f16, rows, row56, groups: 1, splitTarget };
  const mmCode = source("matmul_gelu", spec);
  const reduceCode = source("reduce_gelu", spec);
  const [mm, reduce] = await Promise.all([
    pipeline(device, mmCode, "paired_matmul_gelu", device.createPipelineLayout({ bindGroupLayouts: [layout] })),
    pipeline(device, reduceCode, "paired_reduce_gelu", device.createPipelineLayout({ bindGroupLayouts: [reduceLayout] })),
  ]);
  return { mm, reduce, layout, reduceLayout, ...spec, sharedBytes, mmCode, reduceCode };
}

export function pairedMatmulGeluBindGroup(device, pipes, { globals, params, input, gate, gateScales, up, upScales, output, partials }) {
  const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  return {
    group: group(pipes.layout, [globals, params, input, gate, gateScales, up, upScales, output, partials]),
    reduce: group(pipes.reduceLayout, [globals, params, partials, output]),
  };
}

export function encodePairedMatmulGelu(pass, pipes, op, T, splitT = T) {
  if (pipes.row56 && Math.ceil(T / 56) !== Math.ceil(T / 64)) throw new Error("paired BM56 must retain the R4 workgroup grid");
  const splits = mmSplits(splitT, op.N, op.K, pipes.splitTarget, 16 * pipes.rows, 64);
  pass.setPipeline(pipes.mm);
  pass.setBindGroup(0, op.group);
  pass.dispatchWorkgroups(Math.ceil(op.N / 64), Math.ceil(T / (16 * pipes.rows)), splits);
  if (splits > 1) {
    const work = Math.ceil(T * op.N / 256);
    pass.setPipeline(pipes.reduce);
    pass.setBindGroup(0, op.reduce);
    pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
  }
}
