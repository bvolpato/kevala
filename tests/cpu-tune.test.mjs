import assert from "node:assert/strict";
import test from "node:test";
import { selectCpuTile } from "../js/src/cpu-tune.js";

function timedExports(samples) {
  const x = {
    tile: 0,
    kevala_set_tile(tile) {
      this.tile = tile;
    },
    kevala_tile_probe() {},
  };
  const offsets = [0, 0];
  let clock = 0;
  let started = true;
  const now = () => {
    if (started) {
      started = false;
      return clock;
    }
    started = true;
    clock += samples[x.tile][offsets[x.tile]++];
    return clock;
  };
  return { x, now };
}

test("selects a clearly faster baseline tile", () => {
  const { x, now } = timedExports([[100, 100, 100], [200, 200, 200]]);
  assert.equal(selectCpuTile(x, now), 0);
  assert.equal(x.tile, 0);
});

test("selects tile one when its median is materially faster and pair wins are reliable", () => {
  const { x, now } = timedExports([[200, 200, 200], [100, 100, 100]]);
  assert.equal(selectCpuTile(x, now), 1);
  assert.equal(x.tile, 1);
});

test("keeps tile zero for ties and near ties", () => {
  for (const samples of [
    [[100, 100, 100], [100, 100, 100]],
    [[100, 100, 100], [95, 95, 95]],
  ]) {
    const { x, now } = timedExports(samples);
    assert.equal(selectCpuTile(x, now), 0);
    assert.equal(x.tile, 0);
  }
});

test("an isolated slow timing sample does not choose the wrong tile", () => {
  const { x, now } = timedExports([[100, 500, 100], [110, 110, 110]]);
  assert.equal(selectCpuTile(x, now), 0);
  assert.equal(x.tile, 0);
});

test("keeps tile zero when the median advantage lacks paired wins", () => {
  const { x, now } = timedExports([[100, 200, 300], [101, 100, 301]]);
  assert.equal(selectCpuTile(x, now), 0);
});

test("keeps tile zero when clock resolution hides all measurements", () => {
  const { x, now } = timedExports([[0, 0, 0], [0, 0, 0]]);
  assert.equal(selectCpuTile(x, now), 0);
});
