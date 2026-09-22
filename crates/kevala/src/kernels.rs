//! CPU kernels: int8-weight matrix multiply, layer norm, GELU, rotary embeddings, attention.
//!
//! Activations stay f32 end to end. Laya's encoder has outlier activation channels that
//! per-tensor activation quantization destroys (laya-web measured 65% argmax agreement with
//! dynamic int8), so only the weights are 8-bit and they are widened to f32 in cache-sized
//! panels right before use.

use crate::simd::{dequant16, dot16, transpose4x4, F4};

/// A weight matrix `[n, k]` (out features by in features), borrowed from a store.
#[derive(Clone, Copy)]
pub enum Mat<'a> {
    Q8 { n: usize, k: usize, block: usize, q: &'a [i8], scales: &'a [f32] },
    F32 { n: usize, k: usize, w: &'a [f32] },
}

impl Mat<'_> {
    pub fn n(&self) -> usize {
        match *self {
            Mat::Q8 { n, .. } | Mat::F32 { n, .. } => n,
        }
    }
    pub fn k(&self) -> usize {
        match *self {
            Mat::Q8 { k, .. } | Mat::F32 { k, .. } => k,
        }
    }
}

const NR: usize = 4;
const Q8_NR: usize = 8;

/// Widens `rows` rows starting at `n0` of a q8 matrix into `panel` (`rows * k` floats).
fn dequant_panel(q: &[i8], scales: &[f32], k: usize, block: usize, n0: usize, rows: usize, panel: &mut [f32]) {
    let nb = k / block;
    for r in 0..rows {
        let src = &q[(n0 + r) * k..(n0 + r + 1) * k];
        let sc = &scales[(n0 + r) * nb..(n0 + r + 1) * nb];
        let dst = &mut panel[r * k..(r + 1) * k];
        for b in 0..nb {
            let s = sc[b];
            let mut c = b * block;
            while c < (b + 1) * block {
                unsafe { dequant16(src.as_ptr().add(c), s, dst.as_mut_ptr().add(c)) };
                c += 16;
            }
        }
    }
}

/// Packs eight dequantized rows into `[k][8]` output lanes after the first eight rows in `panel`.
/// Missing rows are zeroed so the same path can be used for a short output tail.
fn dequant_panel8(q: &[i8], scales: &[f32], k: usize, block: usize, n0: usize, rows: usize, panel: &mut [f32]) {
    dequant_panel(q, scales, k, block, n0, rows, &mut panel[..Q8_NR * k]);
    for r in rows..Q8_NR {
        panel[r * k..(r + 1) * k].fill(0.0);
    }
}

/// Transposes the dequantized eight-row panel into output-lane vectors. Each pair of calls writes
/// four columns at a time, with a stride of eight floats between successive k positions.
#[inline(always)]
unsafe fn pack_panel8(panel: *const f32, packed: *mut f32, k: usize) {
    let mut i = 0;
    while i < k {
        let dst = packed.add(i * Q8_NR);
        transpose4x4(panel.add(i), panel.add(k + i), panel.add(2 * k + i), panel.add(3 * k + i), dst, Q8_NR);
        transpose4x4(
            panel.add(4 * k + i),
            panel.add(5 * k + i),
            panel.add(6 * k + i),
            panel.add(7 * k + i),
            dst.add(4),
            Q8_NR,
        );
        i += 4;
    }
}

/// Two rows of x against four weight rows: lanes run along k, eight accumulators.
#[inline(always)]
unsafe fn micro_2x4(a0: *const f32, a1: *const f32, w: *const f32, k: usize) -> ([f32; 4], [f32; 4]) {
    let (w0, w1, w2, w3) = (w, w.add(k), w.add(2 * k), w.add(3 * k));
    let mut c = [F4::zero(); 8];
    let mut i = 0;
    while i < k {
        let x0 = F4::load(a0.add(i));
        let x1 = F4::load(a1.add(i));
        let v0 = F4::load(w0.add(i));
        c[0] = c[0].fma(x0, v0);
        c[4] = c[4].fma(x1, v0);
        let v1 = F4::load(w1.add(i));
        c[1] = c[1].fma(x0, v1);
        c[5] = c[5].fma(x1, v1);
        let v2 = F4::load(w2.add(i));
        c[2] = c[2].fma(x0, v2);
        c[6] = c[6].fma(x1, v2);
        let v3 = F4::load(w3.add(i));
        c[3] = c[3].fma(x0, v3);
        c[7] = c[7].fma(x1, v3);
        i += 4;
    }
    ([c[0].hsum(), c[1].hsum(), c[2].hsum(), c[3].hsum()], [c[4].hsum(), c[5].hsum(), c[6].hsum(), c[7].hsum()])
}

/// Four rows of x against four weight rows: sixteen accumulators, half the loads per
/// multiply-add of `micro_2x4`. Needs 32 vector registers (ARM); on x86 it spills.
#[inline(always)]
unsafe fn micro_4x4(a: *const f32, w: *const f32, k: usize) -> [[f32; 4]; 4] {
    let (a0, a1, a2, a3) = (a, a.add(k), a.add(2 * k), a.add(3 * k));
    let (w0, w1, w2, w3) = (w, w.add(k), w.add(2 * k), w.add(3 * k));
    let mut c = [F4::zero(); 16];
    let mut i = 0;
    while i < k {
        let x = [F4::load(a0.add(i)), F4::load(a1.add(i)), F4::load(a2.add(i)), F4::load(a3.add(i))];
        for (j, wp) in [w0, w1, w2, w3].into_iter().enumerate() {
            let v = F4::load(wp.add(i));
            for r in 0..4 {
                c[r * 4 + j] = c[r * 4 + j].fma(x[r], v);
            }
        }
        i += 4;
    }
    let mut o = [[0.0; 4]; 4];
    for r in 0..4 {
        for j in 0..4 {
            o[r][j] = c[r * 4 + j].hsum();
        }
    }
    o
}

/// Four rows of x against eight output lanes. The packed weights are `[k][8]`, so each scalar
/// activation is broadcast across the two four-column vectors and the tile needs eight accumulators.
#[inline(always)]
unsafe fn micro_4x8(a: *const f32, w: *const f32, k: usize) -> [[f32; 8]; 4] {
    let (a0, a1, a2, a3) = (a, a.add(k), a.add(2 * k), a.add(3 * k));
    let mut c = [F4::zero(); 8];
    let mut i = 0;
    while i < k {
        let wp = w.add(i * Q8_NR);
        let (v0, v1) = (F4::load(wp), F4::load(wp.add(4)));
        let (x0, x1) = (F4::splat(*a0.add(i)), F4::splat(*a1.add(i)));
        let (x2, x3) = (F4::splat(*a2.add(i)), F4::splat(*a3.add(i)));
        c[0] = c[0].fma(x0, v0);
        c[1] = c[1].fma(x0, v1);
        c[2] = c[2].fma(x1, v0);
        c[3] = c[3].fma(x1, v1);
        c[4] = c[4].fma(x2, v0);
        c[5] = c[5].fma(x2, v1);
        c[6] = c[6].fma(x3, v0);
        c[7] = c[7].fma(x3, v1);
        i += 1;
    }
    let mut o = [[0.0; Q8_NR]; 4];
    for r in 0..4 {
        c[2 * r].store(o[r].as_mut_ptr());
        c[2 * r + 1].store(o[r].as_mut_ptr().add(4));
    }
    o
}

/// Which register tile `linear` uses: tile 0 is packed 4x8 for Q8 and 2x4 otherwise; tile 1 is
/// 4x4. Native ARM defaults to 4x4; WebAssembly starts at tile 0 and the host can switch after
/// timing both (`tune`).
static TILE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(if cfg!(target_arch = "aarch64") { 1 } else { 0 });

pub fn set_tile(t: u8) {
    TILE.store(t.min(1), std::sync::atomic::Ordering::Relaxed);
}

pub fn tile() -> u8 {
    TILE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Times both register tiles on a synthetic matmul and keeps the faster one. Returns the choice.
pub fn tune(now: &dyn Fn() -> f64) -> u8 {
    let (t, n, k) = (32, 256, 1024);
    let x = vec![0.5f32; t * k];
    let q = vec![1i8; n * k];
    let s = vec![0.01f32; n * k / 32];
    let mut out = vec![0.0; t * n];
    let mut panel = Vec::new();
    let mut best = (f64::INFINITY, 0);
    for tile in [0u8, 1] {
        set_tile(tile);
        linear(&x, t, Mat::Q8 { n, k, block: 32, q: &q, scales: &s }, None, &mut out, &mut panel);
        let t0 = now();
        for _ in 0..3 {
            linear(&x, t, Mat::Q8 { n, k, block: 32, q: &q, scales: &s }, None, &mut out, &mut panel);
        }
        let dt = now() - t0;
        if dt < best.0 {
            best = (dt, tile);
        }
    }
    set_tile(best.1);
    best.1
}

#[inline(always)]
unsafe fn dot4(a: *const f32, w: *const f32, k: usize) -> f32 {
    let (mut s0, mut s1) = (F4::zero(), F4::zero());
    let mut i = 0;
    while i + 8 <= k {
        s0 = s0.fma(F4::load(a.add(i)), F4::load(w.add(i)));
        s1 = s1.fma(F4::load(a.add(i + 4)), F4::load(w.add(i + 4)));
        i += 8;
    }
    if i < k {
        s0 = s0.fma(F4::load(a.add(i)), F4::load(w.add(i)));
    }
    s0.add(s1).hsum()
}

/// Q8 path for tile 0. Eight output rows are widened once, transposed to `[k][8]`, and reused
/// across four input rows at a time. A short N tail uses zero lanes in the same packed panel.
fn linear_q8_packed(
    x: &[f32],
    t: usize,
    n: usize,
    k: usize,
    block: usize,
    q: &[i8],
    scales: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    panel: &mut Vec<f32>,
) {
    let tb = ((256 * 1024) / (k * 4)).clamp(8, 512) & !1;
    panel.resize(2 * Q8_NR * k, 0.0);
    let mut t0 = 0;
    while t0 < t {
        let t1 = (t0 + tb).min(t);
        let mut n0 = 0;
        while n0 < n {
            let rows = Q8_NR.min(n - n0);
            dequant_panel8(q, scales, k, block, n0, rows, panel);
            unsafe { pack_panel8(panel.as_ptr(), panel.as_mut_ptr().add(Q8_NR * k), k) };
            let wp = unsafe { panel.as_ptr().add(Q8_NR * k) };
            let b = |j: usize| bias.map_or(0.0, |b| b[n0 + j]);
            let mut r = t0;
            unsafe {
                while r + 4 <= t1 {
                    let o = micro_4x8(x.as_ptr().add(r * k), wp, k);
                    for (i, row) in o.iter().enumerate() {
                        for j in 0..rows {
                            out[(r + i) * n + n0 + j] = row[j] + b(j);
                        }
                    }
                    r += 4;
                }
                while r < t1 {
                    for j in 0..rows {
                        out[r * n + n0 + j] = dot4(x.as_ptr().add(r * k), panel.as_ptr().add(j * k), k) + b(j);
                    }
                    r += 1;
                }
            }
            n0 += rows;
        }
        t0 = t1;
    }
}

/// out[t][n] = sum_k x[t][k] * w[n][k] (+ bias[n]) for t rows. `k % 4 == 0`, and q8 blocks are
/// multiples of 16. `panel` is scratch, grown as needed.
pub fn linear(x: &[f32], t: usize, m: Mat, bias: Option<&[f32]>, out: &mut [f32], panel: &mut Vec<f32>) {
    let (n, k) = (m.n(), m.k());
    debug_assert!(k % 4 == 0 && x.len() >= t * k && out.len() >= t * n);
    if tile() == 0 {
        if let Mat::Q8 { block, q, scales, .. } = m {
            linear_q8_packed(x, t, n, k, block, q, scales, bias, out, panel);
            return;
        }
    }
    // rows of x per pass, so a pass stays cache resident while every weight panel streams by once
    let tb = ((256 * 1024) / (k * 4)).clamp(8, 512) & !1;
    if let Mat::Q8 { .. } = m {
        if panel.len() < NR * k {
            panel.resize(NR * k, 0.0);
        }
    }
    let mut t0 = 0;
    while t0 < t {
        let t1 = (t0 + tb).min(t);
        let mut n0 = 0;
        while n0 < n {
            let rows = NR.min(n - n0);
            let wp: *const f32 = match m {
                Mat::Q8 { q, scales, block, .. } => {
                    dequant_panel(q, scales, k, block, n0, rows, panel);
                    panel.as_ptr()
                }
                Mat::F32 { w, .. } => w[n0 * k..].as_ptr(),
            };
            let b = |j: usize| bias.map_or(0.0, |b| b[n0 + j]);
            let mut r = t0;
            unsafe {
                if rows == NR && tile() == 1 {
                    while r + 4 <= t1 {
                        let o = micro_4x4(x.as_ptr().add(r * k), wp, k);
                        for (i, row) in o.iter().enumerate() {
                            for j in 0..NR {
                                out[(r + i) * n + n0 + j] = row[j] + b(j);
                            }
                        }
                        r += 4;
                    }
                }
                if rows == NR {
                    while r + 2 <= t1 {
                        let (o0, o1) = micro_2x4(x.as_ptr().add(r * k), x.as_ptr().add((r + 1) * k), wp, k);
                        for j in 0..NR {
                            out[r * n + n0 + j] = o0[j] + b(j);
                            out[(r + 1) * n + n0 + j] = o1[j] + b(j);
                        }
                        r += 2;
                    }
                }
                while r < t1 {
                    for j in 0..rows {
                        out[r * n + n0 + j] = dot4(x.as_ptr().add(r * k), wp.add(j * k), k) + b(j);
                    }
                    r += 1;
                }
            }
            n0 += NR;
        }
        t0 = t1;
    }
}

/// Row-wise layer norm over `d` features: `(x - mean) / sqrt(var + eps) * w + b`.
pub fn layer_norm(x: &[f32], t: usize, d: usize, w: &[f32], b: Option<&[f32]>, eps: f32, out: &mut [f32]) {
    debug_assert!(d % 16 == 0);
    for r in 0..t {
        let row = &x[r * d..(r + 1) * d];
        let mut s = F4::zero();
        for c in (0..d).step_by(4) {
            s = s.add(unsafe { F4::load(row.as_ptr().add(c)) });
        }
        let mean = s.hsum() / d as f32;
        let mv = F4::splat(mean);
        let mut v = F4::zero();
        for c in (0..d).step_by(4) {
            let e = unsafe { F4::load(row.as_ptr().add(c)) }.sub(mv);
            v = v.fma(e, e);
        }
        let inv = 1.0 / (v.hsum() / d as f32 + eps).sqrt();
        let iv = F4::splat(inv);
        let o = &mut out[r * d..(r + 1) * d];
        for c in (0..d).step_by(4) {
            unsafe {
                let e = F4::load(row.as_ptr().add(c)).sub(mv).mul(iv).mul(F4::load(w.as_ptr().add(c)));
                let e = match b {
                    Some(b) => e.add(F4::load(b.as_ptr().add(c))),
                    None => e,
                };
                e.store(o.as_mut_ptr().add(c));
            }
        }
    }
}

/// erf for f32, ported from musl's erff (itself from FreeBSD), accurate to about 1 ulp.
pub fn erf(x: f32) -> f32 {
    const ERX: f32 = 8.4506291151e-01;
    const EFX8: f32 = 1.0270333290e+00;
    const PP: [f32; 5] = [1.2837916613e-01, -3.2504209876e-01, -2.8481749818e-02, -5.7702702470e-03, -2.3763017452e-05];
    const QQ: [f32; 5] = [3.9791721106e-01, 6.5022252500e-02, 5.0813062117e-03, 1.3249473704e-04, -3.9602282413e-06];
    const PA: [f32; 7] = [
        -2.3621185683e-03,
        4.1485610604e-01,
        -3.7220788002e-01,
        3.1834661961e-01,
        -1.1089469492e-01,
        3.5478305072e-02,
        -2.1663755178e-03,
    ];
    const QA: [f32; 6] =
        [1.0642088205e-01, 5.4039794207e-01, 7.1828655899e-02, 1.2617121637e-01, 1.3637083583e-02, 1.1984500103e-02];
    const RA: [f32; 8] = [
        -9.8649440333e-03,
        -6.9385856390e-01,
        -1.0558626175e+01,
        -6.2375331879e+01,
        -1.6239666748e+02,
        -1.8460508728e+02,
        -8.1287437439e+01,
        -9.8143291473e+00,
    ];
    const SA: [f32; 8] = [
        1.9651271820e+01,
        1.3765776062e+02,
        4.3456588745e+02,
        6.4538726807e+02,
        4.2900814819e+02,
        1.0863500214e+02,
        6.5702495575e+00,
        -6.0424413532e-02,
    ];
    const RB: [f32; 7] = [
        -9.8649431020e-03,
        -7.9928326607e-01,
        -1.7757955551e+01,
        -1.6063638306e+02,
        -6.3756646729e+02,
        -1.0250950928e+03,
        -4.8351919556e+02,
    ];
    const SB: [f32; 7] = [
        3.0338060379e+01,
        3.2579251099e+02,
        1.5367296143e+03,
        3.1998581543e+03,
        2.5530502930e+03,
        4.7452853394e+02,
        -2.2440952301e+01,
    ];
    let ix = x.to_bits() & 0x7fffffff;
    let sign = x.is_sign_negative();
    if ix >= 0x7f800000 {
        return if x.is_nan() {
            x
        } else if sign {
            -1.0
        } else {
            1.0
        };
    }
    if ix < 0x3f580000 {
        // |x| < 0.84375
        if ix < 0x31800000 {
            return 0.125 * (8.0 * x + EFX8 * x);
        }
        let z = x * x;
        let r = PP[0] + z * (PP[1] + z * (PP[2] + z * (PP[3] + z * PP[4])));
        let s = 1.0 + z * (QQ[0] + z * (QQ[1] + z * (QQ[2] + z * (QQ[3] + z * QQ[4]))));
        return x + x * (r / s);
    }
    let ax = f32::from_bits(ix);
    if ix < 0x3fa00000 {
        // 0.84375 <= |x| < 1.25
        let s = ax - 1.0;
        let p = PA[0] + s * (PA[1] + s * (PA[2] + s * (PA[3] + s * (PA[4] + s * (PA[5] + s * PA[6])))));
        let q = 1.0 + s * (QA[0] + s * (QA[1] + s * (QA[2] + s * (QA[3] + s * (QA[4] + s * QA[5])))));
        return if sign { -ERX - p / q } else { ERX + p / q };
    }
    if ix >= 0x40c00000 {
        // |x| >= 6
        return if sign { -1.0 } else { 1.0 };
    }
    let s = 1.0 / (ax * ax);
    let (r, big_s) = if ix < 0x4036db6d {
        // |x| < 1/0.35
        (
            RA[0] + s * (RA[1] + s * (RA[2] + s * (RA[3] + s * (RA[4] + s * (RA[5] + s * (RA[6] + s * RA[7])))))),
            1.0 + s
                * (SA[0]
                    + s * (SA[1] + s * (SA[2] + s * (SA[3] + s * (SA[4] + s * (SA[5] + s * (SA[6] + s * SA[7]))))))),
        )
    } else {
        (
            RB[0] + s * (RB[1] + s * (RB[2] + s * (RB[3] + s * (RB[4] + s * (RB[5] + s * RB[6]))))),
            1.0 + s * (SB[0] + s * (SB[1] + s * (SB[2] + s * (SB[3] + s * (SB[4] + s * (SB[5] + s * SB[6])))))),
        )
    };
    let z = f32::from_bits(ix & 0xffffe000);
    let rr = (-z * z - 0.5625).exp() * ((z - ax) * (z + ax) + r / big_s).exp();
    let e = 1.0 - rr / ax;
    if sign {
        -e
    } else {
        e
    }
}

/// Exact (erf) GELU, what `nn.GELU()` and `ACT2FN["gelu"]` compute.
#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// ModernBERT's gated MLP: `out[t][i] = gelu(u[t][i]) * u[t][inter + i]`.
pub fn geglu(u: &[f32], t: usize, inter: usize, out: &mut [f32]) {
    for r in 0..t {
        let row = &u[r * 2 * inter..(r + 1) * 2 * inter];
        let o = &mut out[r * inter..(r + 1) * inter];
        for i in 0..inter {
            o[i] = gelu(row[i]) * row[inter + i];
        }
    }
}

/// Rotary tables for one theta: `cos[p][i]`, `sin[p][i]` for i < head_dim / 2, computed the way
/// PyTorch does (f32 inverse frequencies, f32 angle, accurate cos/sin of that angle).
pub struct Rope {
    pub half: usize,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

impl Rope {
    pub fn new(theta: f32, head_dim: usize, max_pos: usize) -> Rope {
        let half = head_dim / 2;
        let inv: Vec<f32> = (0..half).map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32)).collect();
        let mut cos = vec![0.0; max_pos * half];
        let mut sin = vec![0.0; max_pos * half];
        for p in 0..max_pos {
            for i in 0..half {
                let a = (p as f32 * inv[i]) as f64;
                cos[p * half + i] = a.cos() as f32;
                sin[p * half + i] = a.sin() as f32;
            }
        }
        Rope { half, cos, sin }
    }

    /// Rotates one head vector in place (rotate_half convention).
    #[inline]
    pub fn apply(&self, v: &mut [f32], pos: usize) {
        let h = self.half;
        let (c, s) = (&self.cos[pos * h..(pos + 1) * h], &self.sin[pos * h..(pos + 1) * h]);
        for i in 0..h {
            let (x1, x2) = (v[i], v[i + h]);
            v[i] = x1 * c[i] - x2 * s[i];
            v[i + h] = x2 * c[i] + x1 * s[i];
        }
    }
}

/// Multi-head attention over one packed segment.
///
/// `qkv` rows are `[q heads | k heads | v heads]` with `heads * hd` floats each part and row
/// stride `3 * heads * hd`; `ctx` receives `heads * hd` floats per row. `window` limits keys to
/// `|i - j| <= window` (ModernBERT's sliding layers), `None` attends everywhere.
pub fn attention(
    qkv: &[f32],
    seg_start: usize,
    len: usize,
    heads: usize,
    hd: usize,
    window: Option<usize>,
    ctx: &mut [f32],
    kbuf: &mut Vec<f32>,
    vbuf: &mut Vec<f32>,
    scores: &mut Vec<f32>,
) {
    let w = heads * hd;
    let stride = 3 * w;
    let scale = 1.0 / (hd as f32).sqrt();
    kbuf.resize(len * hd, 0.0);
    vbuf.resize(len * hd, 0.0);
    scores.resize(len, 0.0);
    for h in 0..heads {
        for j in 0..len {
            let row = &qkv[(seg_start + j) * stride..];
            kbuf[j * hd..(j + 1) * hd].copy_from_slice(&row[w + h * hd..w + (h + 1) * hd]);
            vbuf[j * hd..(j + 1) * hd].copy_from_slice(&row[2 * w + h * hd..2 * w + (h + 1) * hd]);
        }
        for i in 0..len {
            let q = &qkv[(seg_start + i) * stride + h * hd..(seg_start + i) * stride + (h + 1) * hd];
            let (lo, hi) = match window {
                Some(win) => (i.saturating_sub(win), (i + win + 1).min(len)),
                None => (0, len),
            };
            let mut m = f32::NEG_INFINITY;
            for j in lo..hi {
                let s = dot16(q, &kbuf[j * hd..(j + 1) * hd]) * scale;
                scores[j] = s;
                if s > m {
                    m = s;
                }
            }
            let mut sum = 0.0;
            for j in lo..hi {
                let e = (scores[j] - m).exp();
                scores[j] = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            let o = &mut ctx[(seg_start + i) * w + h * hd..(seg_start + i) * w + (h + 1) * hd];
            let mut acc = [F4::zero(); 16];
            debug_assert!(hd <= 64 && hd % 4 == 0);
            for j in lo..hi {
                let p = F4::splat(scores[j] * inv);
                let vrow = vbuf[j * hd..].as_ptr();
                for (c, a) in acc.iter_mut().enumerate().take(hd / 4) {
                    *a = a.fma(p, unsafe { F4::load(vrow.add(c * 4)) });
                }
            }
            for (c, a) in acc.iter().enumerate().take(hd / 4) {
                unsafe { a.store(o.as_mut_ptr().add(c * 4)) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_f64() {
        let mut worst = 0.0f64;
        let mut x = -7.0f32;
        while x < 7.0 {
            let want = libm_erf(x as f64);
            worst = worst.max((erf(x) as f64 - want).abs());
            x += 0.00137;
        }
        assert!(worst < 2e-7, "worst {worst}");
    }

    // high precision reference via the series / continued fraction split
    fn libm_erf(x: f64) -> f64 {
        if x.abs() < 3.0 {
            let mut sum = x;
            let mut term = x;
            let mut n = 0.0;
            loop {
                n += 1.0;
                term *= -x * x / n;
                let add = term / (2.0 * n + 1.0);
                sum += add;
                if add.abs() < 1e-17 {
                    break;
                }
            }
            sum * 2.0 / std::f64::consts::PI.sqrt()
        } else {
            // erfc continued fraction
            let ax = x.abs();
            let mut f = 0.0;
            for k in (1..200).rev() {
                f = (k as f64 / 2.0) / (ax + f);
            }
            let erfc = (-ax * ax).exp() / std::f64::consts::PI.sqrt() / (ax + f);
            (1.0 - erfc) * x.signum()
        }
    }

    #[test]
    fn linear_q8_matches_reference() {
        let (t, n, k, block) = (5, 7, 64, 32);
        let x: Vec<f32> = (0..t * k).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
        let q: Vec<i8> = (0..n * k).map(|i| ((i * 53 % 255) as i32 - 127) as i8).collect();
        let scales: Vec<f32> = (0..n * k / block).map(|i| 0.01 + i as f32 * 0.001).collect();
        let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let mut out = vec![0.0; t * n];
        let mut panel = Vec::new();
        let m = Mat::Q8 { n, k, block, q: &q, scales: &scales };
        linear(&x, t, m, Some(&bias), &mut out, &mut panel);
        for r in 0..t {
            for c in 0..n {
                let want: f32 = (0..k)
                    .map(|i| x[r * k + i] * q[c * k + i] as f32 * scales[c * (k / block) + i / block])
                    .sum::<f32>()
                    + bias[c];
                assert!((out[r * n + c] - want).abs() < 1e-3 * want.abs().max(1.0), "{r},{c}");
            }
        }
    }
}
