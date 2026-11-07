function observedFlag(value) {
  return typeof value === "boolean" ? value : null;
}

export function adapterFallbackMetadata(adapter) {
  const adapterInfo = observedFlag(adapter?.info?.isFallbackAdapter);
  const adapterFlag = observedFlag(adapter?.isFallbackAdapter);
  const observations = [
    ["adapter.info", adapterInfo],
    ["adapter", adapterFlag],
  ].filter(([, value]) => value !== null);

  let isFallbackAdapter = null;
  let fallbackAdapterSource = "unavailable";
  if (observations.length === 2 && observations[0][1] !== observations[1][1]) {
    fallbackAdapterSource = "conflict";
  } else if (observations.length) {
    isFallbackAdapter = observations[0][1];
    fallbackAdapterSource = observations.map(([source]) => source).join("+");
  }

  return {
    isFallbackAdapter,
    fallbackAdapterSource,
    fallbackAdapterFlags: { adapterInfo, adapter: adapterFlag },
  };
}
