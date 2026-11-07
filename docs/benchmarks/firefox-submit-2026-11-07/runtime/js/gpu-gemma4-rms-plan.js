// Pair only norms in the same physical graph and storage views.
function sameView(a, b) {
  if (!a || !b) return false;
  const left = a.buffer || a, right = b.buffer || b;
  const leftOffset = a.buffer ? a.offset || 0 : 0, rightOffset = b.buffer ? b.offset || 0 : 0;
  const leftSize = a.buffer ? a.size ?? left.size - leftOffset : left.size;
  const rightSize = b.buffer ? b.size ?? right.size - rightOffset : right.size;
  return Number.isSafeInteger(leftSize) && leftSize > 0 && Number.isSafeInteger(leftOffset) && leftOffset >= 0 &&
    left === right && leftOffset === rightOffset && leftSize === rightSize;
}

// Call once per physical graph, including its final norm when present.
export function gemma4RmsScaleNextPlan(ops) {
  const links = new Map();
  for (let i = 0; i < ops.length; i++) {
    const candidate = ops[i].rmsScaleNext;
    if (!candidate?.globals || !candidate.fusion?.pipeline || !candidate.fusion.group || typeof candidate.fusion.dispatch !== "function" ||
        typeof candidate.nextName !== "string" || !candidate.nextName ||
        !Number.isSafeInteger(candidate.width) || candidate.width < 1 || !Number.isFinite(candidate.eps) || candidate.eps <= 0 ||
        ops[i + 1]?.label !== "scale") continue;
    let j = i + 2;
    if (ops[j]?.label === "mm.ple.proj") {
      if (!["ple.embed", "ple.embed.host"].includes(ops[j + 1]?.label) || ops[j + 2]?.label !== "ple.combine") continue;
      j += 3;
      if (ops[j]?.kind === "copy-ple-vector") j++;
    }
    const next = ops[j], norm = next?.rmsInput;
    if (!norm || !["rms.attn", "rms.final"].includes(next.label) || norm.name !== candidate.nextName || norm.mode !== 1 ||
        norm.width !== candidate.width || norm.eps !== candidate.eps || norm.globals !== candidate.globals ||
        !sameView(norm.input, candidate.residual) || !sameView(norm.output, candidate.output)) continue;
    links.set(ops[i], { fusion: candidate.fusion, next });
  }
  return links;
}
