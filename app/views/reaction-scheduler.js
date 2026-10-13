/** True only when the loaded model explicitly advertises image input. */
export function supportsImage(info) {
  return Array.isArray(info?.modalities) && info.modalities.includes("image");
}

/**
 * Process the newest available frame without building a queue behind a slow model.
 * `capture` runs only when the previous inference has finished, so it reads the current video
 * frame instead of retaining stale camera frames while the model is busy.
 */
export function createLatestFrameScheduler({
  capture,
  infer,
  onResult = () => {},
  onError = () => {},
  release = (frame) => frame?.close?.(),
  intervalMs = 200,
  now = () => performance.now(),
  setTimer = setTimeout,
  clearTimer = clearTimeout,
}) {
  let timer = null;
  let active = false;
  let stopped = false;
  let running = false;
  let epoch = 0;
  let completedAt = [];

  function schedule(delay) {
    if (!active || stopped || running || timer !== null) return;
    timer = setTimer(() => {
      timer = null;
      void processFrame();
    }, delay);
  }

  async function processFrame() {
    if (!active || stopped || running) return;
    running = true;
    const requestEpoch = epoch;
    const startedAt = now();
    let frame = null;
    try {
      frame = capture();
      if (frame == null) return;
      const result = await infer(frame);
      const endedAt = now();
      if (active && !stopped && requestEpoch === epoch) {
        completedAt.push(endedAt);
        completedAt = completedAt.filter((time) => endedAt - time <= 5000);
        const seconds = completedAt.length > 1 ? (completedAt.at(-1) - completedAt[0]) / 1000 : 0;
        const fps = seconds > 0 ? (completedAt.length - 1) / seconds : 0;
        onResult(result, { latencyMs: endedAt - startedAt, fps });
      }
    } catch (error) {
      if (active && !stopped && requestEpoch === epoch) onError(error);
    } finally {
      if (frame != null) {
        try {
          release(frame);
        } catch {}
      }
      running = false;
      if (active && !stopped) schedule(intervalMs);
    }
  }

  function start() {
    if (stopped || active) return;
    active = true;
    epoch += 1;
    completedAt = [];
    schedule(0);
  }

  function pause() {
    if (!active || stopped) return;
    active = false;
    epoch += 1;
    if (timer !== null) clearTimer(timer);
    timer = null;
  }

  function stop() {
    if (stopped) return;
    stopped = true;
    active = false;
    epoch += 1;
    if (timer !== null) clearTimer(timer);
    timer = null;
  }

  return {
    start,
    resume: start,
    pause,
    stop,
    get running() {
      return running;
    },
  };
}
