/** A request/reply channel to one shard; calibration requests are cancellable and bounded. */
export class RemoteShard {
  constructor(port) {
    this.port = port;
    this.waits = new Map();
    this.seq = 0;
    port.onmessage = (ev) => {
      const m = ev.data;
      const wait = this.waits.get(m.seq);
      if (!wait) return;
      if (m.type === "error") wait.reject(new Error(m.message));
      else wait.resolve(m);
    };
  }
  call(msg, transfer = [], { signal, timeout } = {}) {
    const seq = ++this.seq;
    return new Promise((resolve, reject) => {
      let timer;
      const finish = (fn, value) => {
        if (!this.waits.delete(seq)) return;
        clearTimeout(timer);
        signal?.removeEventListener("abort", abort);
        fn(value);
      };
      const abort = () => finish(reject, signal.reason);
      this.waits.set(seq, { resolve: (m) => finish(resolve, m), reject: (e) => finish(reject, e) });
      signal?.addEventListener("abort", abort, { once: true });
      if (signal?.aborted) return abort();
      if (timeout) timer = setTimeout(() => finish(reject, new Error("CPU shard probe timed out")), timeout);
      try {
        this.port.postMessage({ ...msg, seq }, transfer);
      } catch (e) {
        finish(reject, e);
      }
    });
  }
  send(msg, transfer) {
    this.port.postMessage(msg, transfer || []);
  }
  close() {
    for (const wait of this.waits.values()) wait.reject(new Error("CPU shard closed"));
    this.port.close();
  }
}
