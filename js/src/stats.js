// Statistics the load-time calibrations share.

/** The middle value, or the upper of the two middle values. Not defined for an empty list. */
export function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}
