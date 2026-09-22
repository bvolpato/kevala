const WARMUP_PROBES = 4;
const MEASURED_PROBES = 4;
const ROUNDS = 3;

const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
};

/** Selects the faster CPU register tile and leaves the export state on that tile. */
export function selectCpuTile(exports, now = () => performance.now()) {
  for (const tile of [0, 1]) {
    exports.kevala_set_tile(tile);
    for (let i = 0; i < WARMUP_PROBES; i++) exports.kevala_tile_probe();
  }

  const samples = [[], []];
  let tile1Wins = 0;
  for (let round = 0; round < ROUNDS; round++) {
    const first = round % 2;
    const pair = [null, null];
    for (const tile of [first, 1 - first]) {
      exports.kevala_set_tile(tile);
      const started = now();
      for (let i = 0; i < MEASURED_PROBES; i++) exports.kevala_tile_probe();
      const elapsed = now() - started;
      samples[tile].push(elapsed);
      pair[tile] = elapsed;
    }
    if (pair[1] < pair[0]) tile1Wins++;
  }

  const tile0Median = median(samples[0]);
  const tile1Median = median(samples[1]);
  const materialAdvantage = tile1Median <= tile0Median * 0.9;
  const usable = Number.isFinite(tile0Median) && Number.isFinite(tile1Median) && tile0Median > 0 && tile1Median > 0;
  const tile = usable && materialAdvantage && tile1Wins >= 2 ? 1 : 0;
  exports.kevala_set_tile(tile);
  return tile;
}
