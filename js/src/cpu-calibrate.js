import { selectThreadCount } from "./cpu-policy.js";

/** Time the production Q8 up/down projections, worker transfers, and ordered SIMD reduction. */
export async function calibrateThreads(w, remotes, module, config, counts, signal) {
  const width = config.hidden_size;
  const units = config.intermediate_size / 32;
  const started = performance.now();
  const measurements = [];
  try {
    for (const threads of counts) {
      const active = remotes.slice(0, threads - 1);
      const inner = (i) => (Math.floor(units * (i + 1) / threads) - Math.floor(units * i / threads)) * 32;
      const samples = [];
      for (const rows of [32, 128]) {
        signal.throwIfAborted();
        await Promise.all(active.map((r, i) => r.call({ type: "tune-prepare", module, tile: w.tile, rows, width, inner: inner(i + 1) }, [], { signal, timeout: 10000 })));
        w.check(w.x.kevala_cpu_tune_prepare(rows, width, inner(0)));
        const len = rows * width;
        const input = new Float32Array(len).fill(0.5);
        const buffers = active.map(() => new Float32Array(len));
        const acc = threads > 1 ? w.x.kevala_reduce_prepare(len) : 0;
        const partial = w.x.kevala_reduce_partial_ptr();
        const times = [];
        const pass = async () => {
          signal.throwIfAborted();
          const pending = active.map((r, i) => {
            const x = buffers[i];
            x.set(input);
            return r.call({ type: "tune-run", x }, [x.buffer], { signal, timeout: 10000 });
          });
          w.f32(w.x.kevala_cpu_tune_input_ptr(), len).set(input);
          w.check(w.x.kevala_cpu_tune_run());
          if (threads > 1) w.f32(acc, len).set(w.f32(w.x.kevala_cpu_tune_output_ptr(), len));
          for (const [i, result] of (await Promise.all(pending)).entries()) {
            if (!(result.p instanceof Float32Array) || result.p.length !== len) throw new Error("invalid CPU probe partial");
            buffers[i] = result.p;
            w.f32(partial, len).set(result.p);
            w.check(w.x.kevala_reduce_add_partial());
          }
          const output = w.f32(threads > 1 ? acc : w.x.kevala_cpu_tune_output_ptr(), len);
          if (!Number.isFinite(output[0]) || !Number.isFinite(output[len - 1])) throw new Error("non-finite CPU probe output");
        };
        for (let warmup = 0; warmup < 3; warmup++) await pass();
        for (let sample = 0; sample < 3; sample++) {
          const t0 = performance.now();
          let passes = 0;
          do {
            await pass();
            passes++;
          } while (performance.now() - t0 < 25 && passes < 32);
          times.push((performance.now() - t0) / passes);
        }
        samples.push(times);
      }
      measurements.push({ threads, samples });
      // Finish each candidate, then bound work on slower devices. Retain the measured choices.
      if (performance.now() - started > 5000) break;
    }
    return { threads: selectThreadCount(measurements), measurements, complete: measurements.length === counts.length };
  } finally {
    w.x.kevala_cpu_tune_drop();
  }
}
