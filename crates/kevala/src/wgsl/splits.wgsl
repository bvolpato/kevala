// Split-K: narrow matmuls at short lengths launch too few tiles to fill a GPU (a 1024-wide output
// at 64 tokens is 16 workgroups), so K is split across up to 8 workgroups whose partial tiles a
// second pass sums. The host computes the same count (runtime `mm_splits` twin in gpu.js).
fn mm_splits(T: u32, N: u32, K: u32) -> u32 {
  let tiles = ((N + {{BN_MINUS_1}}u) / {{BN}}u) * ((T + {{BM_MINUS_1}}u) / {{BM}}u);
  if (tiles >= 96u) { return 1u; }
  return max(1u, min(min(8u, ({{SPLIT_TARGET}}u + tiles - 1u) / tiles), K / 128u));
}
