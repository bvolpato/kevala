// A stand-in for the engine worker: the page-side API (js/src/index.js) under test talks to it
// exactly as it talks to a real Worker.
import assert from "node:assert/strict";

/**
 * Installs a fake `Worker` (and the globals `Kevala.load` reads) for the test `t`. `outcomes[i]`
 * answers load attempt `i`: a message to reply with, or a function `(worker, options)`.
 * `onRequest(worker, message)` answers every message after the load (`decide`, `profile`).
 */
export function workers(t, outcomes, { probe = true, onRequest = null } = {}) {
  const instances = [];
  const loads = [];
  const events = [];
  const posted = [];
  class Worker {
    constructor() {
      this.index = instances.length;
      this.terminated = false;
      instances.push(this);
    }
    postMessage(message) {
      if (message.type === "probe") {
        if (probe !== null) queueMicrotask(() => {
          if (typeof probe === "function") probe(this);
          else this.reply({ type: "probe", gpu: probe });
        });
        return;
      }
      if (message.type !== "load") {
        posted.push(message);
        queueMicrotask(() => onRequest?.(this, message));
        return;
      }
      loads.push(message.options);
      events.push(`load:${this.index}`);
      const outcome = outcomes[loads.length - 1];
      assert.ok(outcome, "unexpected load attempt");
      queueMicrotask(() => {
        if (typeof outcome === "function") outcome(this, message.options);
        else this.reply(outcome);
      });
    }
    reply(data) {
      this.onmessage?.({ data });
    }
    terminate() {
      this.terminated = true;
      events.push(`terminate:${this.index}`);
    }
  }
  for (const [key, value] of Object.entries({ Worker, navigator: { gpu: {} }, location: { origin: "null" } })) {
    const previous = Object.getOwnPropertyDescriptor(globalThis, key);
    Object.defineProperty(globalThis, key, { value, configurable: true });
    t.after(() => {
      if (previous) Object.defineProperty(globalThis, key, previous);
      else delete globalThis[key];
    });
  }
  return { instances, loads, events, posted };
}
