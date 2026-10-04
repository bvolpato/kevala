// Packs the requests that arrive while a forward pass runs into the next pass.

/** Requests in one pass: enough to fill a pass, small enough to keep its buffers bounded. */
export const MAX_BATCH_REQUESTS = 64;

/**
 * A queue of `{ id, requests }` messages. `run(requests)` answers one pass with
 * `{ responses, timing }`; `post` receives one `result` or `error` message per queued message.
 * Messages are answered in arrival order.
 */
export function requestQueue(run, post, limit = MAX_BATCH_REQUESTS) {
  let queue = [];
  let busy = false;

  const fail = (m, e) => post({ type: "error", id: m.id, message: String(e?.message || e), code: e?.code });

  async function pass(batch) {
    const requests = batch.flatMap((m) => m.requests);
    const { responses, timing } = await run(requests);
    let at = 0;
    for (const m of batch) {
      post({ type: "result", id: m.id, responses: responses.slice(at, at + m.requests.length), timing: { ...timing, batched: requests.length } });
      at += m.requests.length;
    }
  }

  async function drain() {
    busy = true;
    while (queue.length) {
      const batch = [];
      let n = 0;
      // A message is never split: the first one always goes, however many requests it holds.
      while (queue.length && (batch.length === 0 || n + queue[0].requests.length <= limit)) {
        const m = queue.shift();
        batch.push(m);
        n += m.requests.length;
      }
      try {
        await pass(batch);
      } catch (e) {
        if (batch.length === 1) {
          fail(batch[0], e);
          continue;
        }
        // One bad request must not fail the others packed with it: answer each message alone.
        for (const m of batch) {
          try {
            await pass([m]);
          } catch (alone) {
            fail(m, alone);
          }
        }
      }
    }
    busy = false;
  }

  return {
    push(m) {
      queue.push(m);
      if (!busy) drain();
    },
    /** Drops every message that has not started; a pass that is running finishes. */
    clear() {
      queue = [];
    },
  };
}
