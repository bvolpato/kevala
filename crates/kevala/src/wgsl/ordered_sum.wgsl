enable subgroups;

fn ordered_sum32(prefix: f32, subgroupLane: u32) -> f32 {
  var aggregate = prefix;
  for (var step = 16u; step > 0u; step >>= 1u) {
    let other = subgroupShuffleDown(aggregate, step);
    if (subgroupLane < step) { aggregate += other; }
  }
  return subgroupBroadcastFirst(aggregate);
}
