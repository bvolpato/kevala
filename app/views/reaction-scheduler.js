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
  createMessageChannel = () => typeof globalThis.MessageChannel === "function" ? new globalThis.MessageChannel() : null,
}) {
  let task = null;
  let channel = null;
  let nextTaskId = 0;
  let active = false;
  let stopped = false;
  let running = false;
  let epoch = 0;
  let completedAt = [];
  const useMessageChannel = setTimer === globalThis.setTimeout && clearTimer === globalThis.clearTimeout;

  function closeChannel() {
    if (!channel) return;
    channel.port1.onmessage = null;
    channel.port1.close?.();
    channel.port2.close?.();
    channel = null;
  }

  function messageChannel() {
    if (channel) return channel;
    let candidate;
    try {
      candidate = createMessageChannel();
      if (!candidate) return null;
      candidate.port1.onmessage = ({ data }) => {
        if (task?.kind !== "channel" || task.id !== data) return;
        task = null;
        void processFrame();
      };
      candidate.port1.start?.();
      channel = candidate;
      return channel;
    } catch {
      candidate?.port1.close?.();
      candidate?.port2.close?.();
      return null;
    }
  }

  function cancelTask() {
    const pending = task;
    task = null;
    if (pending?.kind === "timer") clearTimer(pending.handle);
  }

  function schedule(delay) {
    if (!active || stopped || running || task !== null) return;
    const id = ++nextTaskId;
    if (delay === 0 && useMessageChannel) {
      const messages = messageChannel();
      if (messages) {
        task = { kind: "channel", id };
        try {
          messages.port2.postMessage(id);
          return;
        } catch {
          task = null;
          closeChannel();
        }
      }
    }
    const pending = { kind: "timer", id, handle: null };
    task = pending;
    pending.handle = setTimer(() => {
      if (task !== pending) return;
      task = null;
      void processFrame();
    }, delay);
  }

  async function processFrame() {
    if (!active || stopped || running) return;
    running = true;
    const requestEpoch = epoch;
    const startedAt = now();
    let frame = null;
    let emptyCapture = false;
    try {
      frame = capture();
      if (frame == null) {
        emptyCapture = true;
        return;
      }
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
      if (active && !stopped) {
        const delay = emptyCapture && intervalMs === 0 ? 16 : Math.max(0, intervalMs - (now() - startedAt));
        schedule(delay);
      }
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
    cancelTask();
  }

  function stop() {
    if (stopped) return;
    stopped = true;
    active = false;
    epoch += 1;
    cancelTask();
    closeChannel();
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
