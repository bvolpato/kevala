#!/usr/bin/env node
// CPU-only camera preprocessing comparison. Prints one JSON receipt to stdout.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { disposeGemma4VisionImage, prepareGemma4Image, prepareGemma4VisionImage } from "../js/src/gemma4-image.js";
import { visionImagePlan } from "../js/src/gpu-gemma4-vision.js";

const cfg = { patch: 16, pool: 3, positionSize: 64, headDim: 64, theta: 10000 };
const width = 320, height = 240;
const frames = Array.from({ length: 4 }, (_, frame) => ({ width, height, data: Uint8Array.from({ length: width * height * 4 }, (_, i) => (i * 37 + Math.floor(i / 11) * 19 + frame * 71) & 255) }));
const points = [];

function cleanup(prepared, plan) {
  plan?.pixels.fill(0);
  disposeGemma4VisionImage(prepared);
  prepared?.patches?.fill(0);
}

function run(prepare, frame, budget) {
  let prepared, plan;
  const start = performance.now();
  try {
    prepared = prepare(frame, budget);
    const packed = performance.now();
    plan = visionImagePlan(prepared, cfg);
    const planned = performance.now();
    cleanup(prepared, plan);
    const finished = performance.now();
    return { prepare: packed - start, planning: planned - packed, cleanup: finished - planned, combined: finished - start };
  } catch (error) { cleanup(prepared, plan); throw error; }
}

function summary(samples) {
  return Object.fromEntries(["prepare", "planning", "cleanup", "combined"].map((key) => {
    const values = samples.map((value) => value[key]).sort((a, b) => a - b);
    return [key, { p50Ms: values[Math.floor(values.length * 0.5)], p95Ms: values[Math.ceil(values.length * 0.95) - 1] }];
  }));
}

for (const budget of [16, 40, 70]) {
  let activeRows, outWidth, outHeight;
  for (const frame of frames) {
    let a, b, pa, pb;
    try {
      a = prepareGemma4Image(frame, budget);
      b = prepareGemma4VisionImage(frame, budget);
      pa = visionImagePlan(a, cfg);
      pb = visionImagePlan(b, cfg);
      assert.deepEqual(new Uint32Array(pb.pixels.buffer, pb.pixels.byteOffset, pb.pixels.length), new Uint32Array(pa.pixels.buffer, pa.pixels.byteOffset, pa.pixels.length));
      assert.deepEqual(pb.positions, pa.positions);
      assert.deepEqual(pb.rope, pa.rope);
      assert.deepEqual(pb.pools, pa.pools);
      assert.equal(pb.rows, pa.rows);
      assert.equal(pb.softRows, pa.softRows);
      activeRows = pa.rows; outWidth = a.width; outHeight = a.height;
    } finally { cleanup(a, pa); cleanup(b, pb); }
  }
  for (let i = 0; i < 20; i++) {
    const order = i % 2 ? [prepareGemma4VisionImage, prepareGemma4Image] : [prepareGemma4Image, prepareGemma4VisionImage];
    for (const prepare of order) run(prepare, frames[i % frames.length], budget);
  }
  const baseline = [], candidate = [];
  for (let i = 0; i < 60; i++) {
    const pair = [[prepareGemma4Image, baseline], [prepareGemma4VisionImage, candidate]];
    if (i % 2) pair.reverse();
    for (const [prepare, samples] of pair) samples.push(run(prepare, frames[i % frames.length], budget));
  }
  const horizontal = outWidth * height * 3, paddedPatches = budget * 9 * 768 * 4, normalized = activeRows * 768 * 4, positions = budget * 9 * 2 * 4;
  const geometryCopies = activeRows * (2 * 4 + cfg.headDim * 4 + 4);
  const a = summary(baseline), b = summary(candidate);
  points.push({ budget, width: outWidth, height: outHeight, activeRows, completeBitGuards: frames.length, baseline: a, candidate: b,
    combinedP50Speedup: a.combined.p50Ms / b.combined.p50Ms,
    knownPerFrameArrayBytes: { baseline: horizontal + paddedPatches + normalized + positions + geometryCopies, candidate: horizontal + normalized + positions + geometryCopies, saved: paddedPatches, excludes: "cached geometry, input frames and JS object headers" } });
}
const sources = {};
for (const path of ["scripts/bench-gemma4-image-prep.mjs", "js/src/gemma4-image.js", "js/src/gpu-gemma4-vision.js"]) sources[path] = createHash("sha256").update(readFileSync(new URL("../" + path, import.meta.url))).digest("hex");
process.stdout.write(JSON.stringify({ benchmarkVersion: 1, generatedUtc: new Date().toISOString(), node: process.version, comparison: { baseline: "public prepareGemma4Image + visionImagePlan", candidate: "private prepareGemma4VisionImage + visionImagePlan" }, cfg, input: { width, height, frames: frames.length }, warmupPairs: 20, samplePairs: 60, alternatedOrder: true, includesCleanup: true, sources, points }, null, 2) + "\n");
