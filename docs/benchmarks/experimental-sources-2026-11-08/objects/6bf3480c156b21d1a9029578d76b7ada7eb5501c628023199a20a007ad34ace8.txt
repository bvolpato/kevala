import assert from "node:assert/strict";
import { test } from "node:test";
import { Kevala } from "../js/src/index.js";

function workers(t, outcomes, { probe = true } = {}) {
  const instances = [];
  const loads = [];
  const events = [];
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
      if (message.type !== "load") return;
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
  return { instances, loads, events };
}

const gpuError = (message) => ({ type: "error", code: "WEBGPU_INIT", message });
const ready = (backend = "webgpu") => ({ type: "ready", info: { backend, gpu: backend === "webgpu" ? "Test GPU" : null } });

test("GPU kernel overrides reject invalid values before starting a worker", async () => {
  for (const gpuKernel of [null, "cuda", 1]) {
    await assert.rejects(Kevala.load({ gpuKernel }), /gpuKernel must be/);
  }
});

test("GPU kernel overrides reach the engine unchanged", async (t) => {
  const state = workers(t, [ready()]);
  const model = await Kevala.load({ backend: "webgpu", gpuKernel: "wide", cache: false });
  assert.equal(state.loads[0].gpuKernel, "wide");
  model.dispose();
});

test("Auto retries a GPU load failure once on the low-power adapter", async (t) => {
  const state = workers(t, [gpuError("WebGPU: uploading model weights: Not enough memory left."), ready()]);
  const model = await Kevala.load({ backend: "auto", cache: false });
  t.after(() => model.dispose());
  assert.equal(model.info.backend, "webgpu");
  assert.deepEqual(state.loads.map((o) => o.gpuPowerPreference), ["high-performance", "low-power"]);
  assert.deepEqual(state.loads.map((o) => o.backend), ["auto", "auto"]);
  assert.deepEqual(state.events, ["load:0", "terminate:0", "load:1"]);
});

test("Auto preserves both GPU failures when it falls back to a fresh CPU worker", async (t) => {
  const state = workers(t, [gpuError("WebGPU: Not enough memory left."), gpuError("WebGPU: device lost"), ready("wasm-simd")]);
  const model = await Kevala.load({ cache: false });
  t.after(() => model.dispose());
  assert.equal(model.info.backend, "wasm-simd");
  assert.equal(model.info.gpuUnavailable, "high-performance: Not enough memory left.; low-power: device lost");
  assert.equal(state.loads[2].backend, "wasm");
  assert.deepEqual(state.events, ["load:0", "terminate:0", "load:1", "terminate:1", "load:2"]);
});

test("explicit WebGPU retries but rejects with both reasons instead of using the CPU", async (t) => {
  const state = workers(t, [gpuError("out of memory"), gpuError("adapter unavailable")]);
  await assert.rejects(Kevala.load({ backend: "webgpu", cache: false }), (error) => {
    assert.equal(error.code, "WEBGPU_INIT");
    assert.match(error.message, /high-performance: out of memory; low-power: adapter unavailable/);
    return true;
  });
  assert.equal(state.loads.length, 2);
  assert.ok(state.instances.every((w) => w.terminated));
});

for (const message of ["HTTP 503", "not a .kevala pack", "GPU trunk is missing 2 tensors (enc.0.wqkv...)", "WebGPU: application plugin rejected its input"]) {
  test(`does not retry an untagged load error: ${message}`, async (t) => {
    const state = workers(t, [{ type: "error", message }]);
    await assert.rejects(Kevala.load({ cache: false }), { message });
    assert.equal(state.loads.length, 1);
    assert.ok(state.instances[0].terminated);
  });
}

test("an explicit CPU load never retries a GPU-tagged error", async (t) => {
  const state = workers(t, [gpuError("plugin failure")]);
  await assert.rejects(Kevala.load({ backend: "wasm", cache: false }), { message: "plugin failure" });
  assert.equal(state.loads.length, 1);
  assert.ok(state.instances[0].terminated);
});

test("cancelling an active load terminates its worker without another attempt", async (t) => {
  const controller = new AbortController();
  const reason = new Error("cancelled by caller");
  const state = workers(t, [(worker) => {
    controller.abort(reason);
    worker.reply(gpuError("late GPU failure"));
  }]);
  await assert.rejects(Kevala.load({ signal: controller.signal, cache: false }), (error) => error === reason);
  assert.equal(state.loads.length, 1);
  assert.ok(state.instances[0].terminated);
});

test("cancelling from the retry progress callback prevents the low-power attempt", async (t) => {
  const controller = new AbortController();
  const state = workers(t, [gpuError("out of memory")]);
  await assert.rejects(Kevala.load({
    signal: controller.signal,
    cache: false,
    onProgress: () => controller.abort(),
  }), { name: "AbortError" });
  assert.equal(state.loads.length, 1);
  assert.ok(state.instances[0].terminated);
});

test("cancelling while probing worker WebGPU also settles the load", async (t) => {
  const controller = new AbortController();
  const state = workers(t, [], { probe: null });
  const loading = Kevala.load({ signal: controller.signal, cache: false });
  controller.abort();
  await assert.rejects(loading, { name: "AbortError" });
  assert.equal(state.loads.length, 0);
  assert.ok(state.instances[0].terminated);
});

test("worker startup errors during the GPU probe reject without loading or retrying", async (t) => {
  const message = "Failed to load module script: HTTP 404";
  const state = workers(t, [], { probe: (worker) => worker.onerror({ message }) });
  await assert.rejects(Kevala.load({ cache: false }), (error) => {
    assert.equal(error.message, message);
    assert.equal(error.code, undefined);
    return true;
  });
  assert.equal(state.loads.length, 0);
  assert.equal(state.instances.length, 1);
  assert.ok(state.instances[0].terminated);
});

test("cancelling as the CPU fallback becomes ready rejects and frees every worker", async (t) => {
  const controller = new AbortController();
  const reason = new Error("cancelled as CPU became ready");
  const state = workers(t, [gpuError("out of memory"), gpuError("device lost"), (worker) => {
    worker.reply(ready("wasm-simd"));
    controller.abort(reason);
  }]);
  await assert.rejects(Kevala.load({ signal: controller.signal, cache: false }), (error) => error === reason);
  assert.equal(state.loads.length, 3);
  assert.ok(state.instances.every((w) => w.terminated));
});

test("CPU probe workers are released before replacement workers load", async (t) => {
  const state = workers(t, [(worker) => {
    worker.reply({ type: "need-shards", n: 2, generation: 0 });
    assert.equal(state.instances.length, 3);
    worker.reply({ type: "release-shards" });
    assert.ok(state.instances.slice(1).every((w) => w.terminated));
    worker.reply({ type: "need-shards", n: 1, generation: 1 });
    worker.reply(ready("wasm-relaxed x2"));
  }]);
  const model = await Kevala.load({ backend: "wasm", cache: false });
  assert.equal(state.instances.length, 4);
  assert.equal(state.instances[3].terminated, false);
  model.dispose();
  assert.ok(state.instances.every((w) => w.terminated));
});

test("a shard startup error rejects loading and terminates its whole pool", async (t) => {
  const state = workers(t, [(worker) => {
    worker.reply({ type: "need-shards", n: 2, generation: 0 });
    state.instances[1].onerror({ message: "shard import failed" });
  }]);
  await assert.rejects(Kevala.load({ backend: "wasm", cache: false }), /shard import failed/);
  assert.ok(state.instances.every((w) => w.terminated));
});

test("cancelling during CPU calibration terminates every created worker", async (t) => {
  const controller = new AbortController();
  const state = workers(t, [(worker) => {
    worker.reply({ type: "need-shards", n: 3, generation: 0 });
    controller.abort();
  }]);
  await assert.rejects(Kevala.load({ backend: "wasm", cache: false, signal: controller.signal }), { name: "AbortError" });
  assert.equal(state.instances.length, 4);
  assert.ok(state.instances.every((w) => w.terminated));
});
