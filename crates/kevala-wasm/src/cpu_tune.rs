//! A bounded production-shaped CPU calibration probe.
//!
//! The host fills `input`, calls `run`, and reads `output`. The probe exercises the same Q8
//! linear kernels and gated MLP shape used by Laya's encoder, while keeping deterministic weights
//! so a worker-count comparison does not depend on a model pack. The selected global CPU tile is
//! read by `kevala::kernels::linear`; this module deliberately does not change it.

use kevala::kernels::{self, Mat};

const Q8_BLOCK: usize = 32;
const MAX_ROWS: usize = 128;
const MAX_WIDTH: usize = 4_096;
const MAX_INNER: usize = 8_192;
// The up matrix has two inner halves, and the down matrix has one. Keep the product bounded
// before allocating those three weight buffers: at the limit they total 12 MiB of i8 data.
const MAX_WEIGHT_ELEMENTS: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    rows: usize,
    width: usize,
    inner: usize,
    up_len: usize,
    down_len: usize,
    up_scales: usize,
    down_scales: usize,
    input_len: usize,
    middle_len: usize,
    output_len: usize,
    panel_len: usize,
}

fn checked_shape(rows: u32, width: u32, inner: u32) -> Result<Shape, String> {
    let (rows, width, inner) = (rows as usize, width as usize, inner as usize);
    if rows == 0 || width == 0 || inner == 0 {
        return Err("CPU tune dimensions must be nonzero".into());
    }
    if rows > MAX_ROWS {
        return Err(format!("CPU tune rows must be <= {MAX_ROWS}"));
    }
    if width > MAX_WIDTH {
        return Err(format!("CPU tune width must be <= {MAX_WIDTH}"));
    }
    if inner > MAX_INNER {
        return Err(format!("CPU tune inner must be <= {MAX_INNER}"));
    }
    if width % Q8_BLOCK != 0 || inner % Q8_BLOCK != 0 {
        return Err(format!("CPU tune width and inner must be multiples of {Q8_BLOCK}"));
    }

    let weight_elements = width.checked_mul(inner).ok_or_else(|| "CPU tune weight dimensions overflow".to_string())?;
    if weight_elements > MAX_WEIGHT_ELEMENTS {
        return Err(format!("CPU tune width*inner must be <= {MAX_WEIGHT_ELEMENTS} elements"));
    }
    let up_len = weight_elements.checked_mul(2).ok_or_else(|| "CPU tune up weight dimensions overflow".to_string())?;
    let down_len = weight_elements;
    let up_scales = up_len / Q8_BLOCK;
    let down_scales = down_len / Q8_BLOCK;
    let input_len = rows.checked_mul(width).ok_or_else(|| "CPU tune input dimensions overflow".to_string())?;
    let middle_len = rows
        .checked_mul(inner)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| "CPU tune middle dimensions overflow".to_string())?;
    let output_len = input_len;
    let panel_len = width.max(inner).checked_mul(4).ok_or_else(|| "CPU tune panel dimensions overflow".to_string())?;

    Ok(Shape {
        rows,
        width,
        inner,
        up_len,
        down_len,
        up_scales,
        down_scales,
        input_len,
        middle_len,
        output_len,
        panel_len,
    })
}

fn deterministic_input(len: usize) -> Vec<f32> {
    (0..len).map(|i| ((i.wrapping_mul(17) % 257) as f32 - 128.0) / 128.0).collect()
}

fn deterministic_q8(rows: usize, cols: usize, tag: usize) -> Vec<i8> {
    let mut q = vec![0i8; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let v = (r * 31 + c * 17 + tag * 13) % 255;
            q[r * cols + c] = (v as i32 - 127) as i8;
        }
    }
    q
}

fn deterministic_scales(rows: usize, cols: usize, tag: usize) -> Vec<f32> {
    let blocks = cols / Q8_BLOCK;
    let mut scales = vec![0.0f32; rows * blocks];
    for r in 0..rows {
        for b in 0..blocks {
            let v = (r * 7 + b * 11 + tag * 5) % 23;
            scales[r * blocks + b] = 0.001 + v as f32 * 0.00001;
        }
    }
    scales
}

/// Applies Laya's exact GELU gate and compacts each row from `[2 * inner]` to `[inner]` in place.
/// The destination row is always before its source row, so writing it cannot overwrite values
/// that a later row still needs.
fn compact_geglu(middle: &mut [f32], rows: usize, inner: usize) {
    for r in 0..rows {
        let src = r * 2 * inner;
        let dst = r * inner;
        for i in 0..inner {
            let gate = middle[src + i];
            let value = middle[src + inner + i];
            middle[dst + i] = kernels::gelu(gate) * value;
        }
    }
}

struct CpuTuneState {
    shape: Shape,
    input: Vec<f32>,
    // The up projection has two inner halves. Its result is gated in place so the first half can
    // be passed directly to the down projection without allocating during `run`.
    middle: Vec<f32>,
    output: Vec<f32>,
    up: Vec<i8>,
    down: Vec<i8>,
    scales: Vec<f32>,
    panel: Vec<f32>,
}

static mut STATE: Option<CpuTuneState> = None;

#[allow(static_mut_refs)]
fn state() -> &'static mut Option<CpuTuneState> {
    // SAFETY: a WebAssembly instance runs on one thread, and this ABI is synchronous.
    unsafe { &mut STATE }
}

fn new_state(shape: Shape) -> CpuTuneState {
    let up = deterministic_q8(2 * shape.inner, shape.width, 1);
    let down = deterministic_q8(shape.width, shape.inner, 2);
    let mut scales = Vec::with_capacity(shape.up_scales + shape.down_scales);
    scales.extend(deterministic_scales(2 * shape.inner, shape.width, 1));
    scales.extend(deterministic_scales(shape.width, shape.inner, 2));
    CpuTuneState {
        shape,
        input: deterministic_input(shape.input_len),
        middle: vec![0.0; shape.middle_len],
        output: vec![0.0; shape.output_len],
        up,
        down,
        scales,
        // `linear` needs four floats per input feature for its Q8 panel. Both projections fit.
        panel: vec![0.0; shape.panel_len],
    }
}

/// Allocates one production-shaped probe. `inner` is the size of one gated half; the up matrix
/// therefore has `2 * inner` output rows, as does Laya's `wi` tensor.
#[no_mangle]
pub extern "C" fn kevala_cpu_tune_prepare(rows: u32, width: u32, inner: u32) -> u32 {
    super::done((|| {
        let shape = checked_shape(rows, width, inner)?;
        *state() = Some(new_state(shape));
        Ok(())
    })())
}

/// Returns the input activation matrix (`rows * width`) for the host to fill before each run.
#[no_mangle]
pub extern "C" fn kevala_cpu_tune_input_ptr() -> *mut f32 {
    state().as_mut().map_or(std::ptr::null_mut(), |s| s.input.as_mut_ptr())
}

#[no_mangle]
pub extern "C" fn kevala_cpu_tune_input_len() -> usize {
    state().as_ref().map_or(0, |s| s.shape.input_len)
}

/// Returns the output matrix (`rows * width`) after a successful run.
#[no_mangle]
pub extern "C" fn kevala_cpu_tune_output_ptr() -> *const f32 {
    state().as_ref().map_or(std::ptr::null(), |s| s.output.as_ptr())
}

#[no_mangle]
pub extern "C" fn kevala_cpu_tune_output_len() -> usize {
    state().as_ref().map_or(0, |s| s.shape.output_len)
}

/// Runs the production Q8 up projection, gated MLP, and down projection without allocation.
#[no_mangle]
pub extern "C" fn kevala_cpu_tune_run() -> u32 {
    super::done((|| {
        let s = state().as_mut().ok_or("CPU tune probe is not prepared")?;
        let shape = s.shape;
        let up_scales = &s.scales[..shape.up_scales];
        kernels::linear(
            &s.input,
            shape.rows,
            Mat::Q8 { n: 2 * shape.inner, k: shape.width, block: Q8_BLOCK, q: &s.up, scales: up_scales },
            None,
            &mut s.middle,
            &mut s.panel,
        );

        // This is the same exact GELU gating used by Laya's production MLP. Compact the gated
        // rows because the up result is interleaved as two halves per row.
        compact_geglu(&mut s.middle, shape.rows, shape.inner);

        let down_scales = &s.scales[shape.up_scales..shape.up_scales + shape.down_scales];
        kernels::linear(
            &s.middle[..shape.rows * shape.inner],
            shape.rows,
            Mat::Q8 { n: shape.width, k: shape.inner, block: Q8_BLOCK, q: &s.down, scales: down_scales },
            None,
            &mut s.output,
            &mut s.panel,
        );
        Ok(())
    })())
}

/// Releases all probe buffers before model loading. WebAssembly memory pages remain available for
/// reuse by the allocator, but the probe no longer retains its allocations.
#[no_mangle]
pub extern "C" fn kevala_cpu_tune_drop() {
    *state() = None;
}

#[cfg(test)]
mod tests {
    use super::{checked_shape, compact_geglu};
    use kevala::kernels;

    #[test]
    fn rejects_invalid_probe_shapes() {
        assert!(checked_shape(0, 1024, 1024).is_err());
        assert!(checked_shape(1, 1025, 1024).is_err());
        assert!(checked_shape(129, 1024, 1024).is_err());
        assert!(checked_shape(1, 4096, 8192).is_err());
    }

    #[test]
    fn accepts_laya_shape_and_accounts_for_gated_up_projection() {
        let shape = checked_shape(32, 1024, 2624).unwrap();
        assert_eq!(shape.up_len, 2 * 2624 * 1024);
        assert_eq!(shape.down_len, 2624 * 1024);
        assert_eq!(shape.middle_len, 32 * 2 * 2624);
        assert_eq!(shape.input_len, shape.output_len);
    }

    #[test]
    fn compacts_gated_rows_without_cross_row_mixing() {
        let mut middle = vec![1.0, -1.0, 2.0, 3.0, 4.0, -4.0, 5.0, 6.0];
        compact_geglu(&mut middle, 2, 2);
        let expected =
            [kernels::gelu(1.0) * 2.0, kernels::gelu(-1.0) * 3.0, kernels::gelu(4.0) * 5.0, kernels::gelu(-4.0) * 6.0];
        assert_eq!(&middle[..expected.len()], &expected);
    }
}
