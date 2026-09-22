//! Development-only entry points for timing the production CPU linear kernel.
//!
//! This module is feature gated so the shipping WebAssembly ABI does not include the synthetic
//! benchmark state or entry points. The host creates the inputs once, then times only `run`.

use kevala::kernels::{self, Mat};

const SEED: u32 = 0x4b45_5641;
const X_TAG: u32 = 1;
const WEIGHT_TAG: u32 = 2;
const SCALE_TAG: u32 = 3;
const BIAS_TAG: u32 = 4;

#[derive(Default)]
struct BenchState {
    t: usize,
    n: usize,
    k: usize,
    q8: bool,
    bias: bool,
    tile: u8,
    x: Vec<f32>,
    weights: Vec<f32>,
    q: Vec<i8>,
    scales: Vec<f32>,
    biases: Vec<f32>,
    out: Vec<f32>,
    panel: Vec<f32>,
}

static mut STATE: Option<BenchState> = None;

#[allow(static_mut_refs)]
fn state() -> &'static mut BenchState {
    // SAFETY: a WebAssembly instance runs on one thread, and the benchmark API is synchronous.
    unsafe { STATE.get_or_insert_with(BenchState::default) }
}

fn next(rng: &mut u32) -> u32 {
    *rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *rng
}

fn f32_values(len: usize, tag: u32, divisor: f32) -> Vec<f32> {
    let mut rng = SEED ^ tag;
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        let centered = (next(&mut rng) % 2_001) as i32 - 1_000;
        values.push(centered as f32 / divisor);
    }
    values
}

fn q8_values(len: usize) -> Vec<i8> {
    let mut rng = SEED ^ WEIGHT_TAG;
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(((next(&mut rng) % 255) as i32 - 127) as i8);
    }
    values
}

fn scales(len: usize) -> Vec<f32> {
    let mut rng = SEED ^ SCALE_TAG;
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        let fraction = (next(&mut rng) % 251) as f32 / 1_000_000.0;
        values.push(0.0005 + fraction);
    }
    values
}

fn output_checksum(out: &[f32]) -> f64 {
    out.iter().enumerate().fold(0.0, |sum, (i, value)| sum + f64::from(value.abs()) * f64::from((i % 17 + 1) as u32))
}

fn checked_shape(t: u32, n: u32, k: u32, q8: bool) -> Option<(usize, usize, usize, usize, usize)> {
    let (t, n, k) = (t as usize, n as usize, k as usize);
    if t == 0 || n == 0 || k == 0 || k % 4 != 0 || (q8 && k % 32 != 0) {
        return None;
    }
    let x_len = t.checked_mul(k)?;
    let weight_len = n.checked_mul(k)?;
    let scale_len = n.checked_mul(k / 32)?;
    let out_len = t.checked_mul(n)?;
    k.checked_mul(16)?;
    Some((x_len, weight_len, scale_len, out_len, k))
}

/// Allocates deterministic inputs for one matrix shape. `q8` and `bias` are boolean flags, and
/// `tile` selects the production kernel's 2x4 (0) or 4x4 (1) register tile.
#[no_mangle]
pub extern "C" fn kevala_cpu_bench_prepare(t: u32, n: u32, k: u32, q8: u32, bias: u32, tile: u32) -> u32 {
    let q8 = q8 != 0;
    let bias = bias != 0;
    let Some((x_len, weight_len, scale_len, out_len, k)) = checked_shape(t, n, k, q8) else {
        return 1;
    };
    let (t, n) = (t as usize, n as usize);
    let x = f32_values(x_len, X_TAG, 4_096.0);
    let (weights, q, scales) = if q8 {
        (Vec::new(), q8_values(weight_len), scales(scale_len))
    } else {
        (f32_values(weight_len, WEIGHT_TAG, 8_192.0), Vec::new(), Vec::new())
    };
    let biases = if bias { f32_values(n, BIAS_TAG, 4_096.0) } else { Vec::new() };

    *state() = BenchState {
        t,
        n,
        k,
        q8,
        bias,
        tile: tile.min(1) as u8,
        x,
        weights,
        q,
        scales,
        biases,
        out: vec![0.0; out_len],
        // `linear` currently needs 4 * k values. Keep room for wider packed panels so timing
        // `run` never includes setup or allocation as the development kernel changes.
        panel: vec![0.0; 16 * k],
    };
    kernels::set_tile(tile.min(1) as u8);
    0
}

/// Runs the production `kevala::kernels::linear` implementation over the prepared inputs.
#[no_mangle]
pub extern "C" fn kevala_cpu_bench_run() -> u32 {
    let s = state();
    if s.out.is_empty() {
        return 1;
    }
    kernels::set_tile(s.tile);
    let mat = if s.q8 {
        Mat::Q8 { n: s.n, k: s.k, block: 32, q: &s.q, scales: &s.scales }
    } else {
        Mat::F32 { n: s.n, k: s.k, w: &s.weights }
    };
    let bias = s.bias.then_some(s.biases.as_slice());
    kernels::linear(&s.x, s.t, mat, bias, &mut s.out, &mut s.panel);
    0
}

/// Returns the output matrix (`t * n` f32 values) from the last run.
#[no_mangle]
pub extern "C" fn kevala_cpu_bench_out_ptr() -> *const f32 {
    state().out.as_ptr()
}

#[no_mangle]
pub extern "C" fn kevala_cpu_bench_out_len() -> usize {
    state().out.len()
}

/// Computes a checksum over the actual output. The host reads the output for numerical guards and
/// benchmark checksums outside the timed `run` call.
#[no_mangle]
pub extern "C" fn kevala_cpu_bench_checksum() -> f64 {
    output_checksum(&state().out)
}
