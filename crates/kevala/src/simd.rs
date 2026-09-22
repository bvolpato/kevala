//! Four-lane f32 vectors over WebAssembly SIMD128, NEON, or plain arrays.
//!
//! Every kernel is written once against `F4`. The wasm build picks up `simd128` (and
//! `relaxed-simd` for fused multiply-add when compiled with it), native aarch64 uses NEON, and
//! anything else gets arrays that LLVM vectorizes on its own.

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod imp {
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    pub struct F4(v128);

    impl F4 {
        #[inline(always)]
        pub fn zero() -> F4 {
            F4(f32x4_splat(0.0))
        }
        #[inline(always)]
        pub fn splat(v: f32) -> F4 {
            F4(f32x4_splat(v))
        }
        #[inline(always)]
        pub unsafe fn load(p: *const f32) -> F4 {
            F4(core::ptr::read_unaligned(p as *const v128))
        }
        #[inline(always)]
        pub unsafe fn store(self, p: *mut f32) {
            core::ptr::write_unaligned(p as *mut v128, self.0)
        }
        #[inline(always)]
        pub fn add(self, o: F4) -> F4 {
            F4(f32x4_add(self.0, o.0))
        }
        #[inline(always)]
        pub fn sub(self, o: F4) -> F4 {
            F4(f32x4_sub(self.0, o.0))
        }
        #[inline(always)]
        pub fn mul(self, o: F4) -> F4 {
            F4(f32x4_mul(self.0, o.0))
        }
        #[inline(always)]
        pub fn max(self, o: F4) -> F4 {
            F4(f32x4_pmax(self.0, o.0))
        }
        /// self + a * b
        #[inline(always)]
        pub fn fma(self, a: F4, b: F4) -> F4 {
            #[cfg(target_feature = "relaxed-simd")]
            {
                F4(f32x4_relaxed_madd(a.0, b.0, self.0))
            }
            #[cfg(not(target_feature = "relaxed-simd"))]
            {
                F4(f32x4_add(self.0, f32x4_mul(a.0, b.0)))
            }
        }
        #[inline(always)]
        pub fn hsum(self) -> f32 {
            let v = self.0;
            let s = f32x4_add(v, i32x4_shuffle::<2, 3, 0, 1>(v, v));
            let s = f32x4_add(s, i32x4_shuffle::<1, 0, 3, 2>(s, s));
            f32x4_extract_lane::<0>(s)
        }
        #[inline(always)]
        pub fn hmax(self) -> f32 {
            let v = self.0;
            let s = f32x4_pmax(v, i32x4_shuffle::<2, 3, 0, 1>(v, v));
            let s = f32x4_pmax(s, i32x4_shuffle::<1, 0, 3, 2>(s, s));
            f32x4_extract_lane::<0>(s)
        }
        #[inline(always)]
        pub(super) fn exp_reduction(self) -> Option<(F4, F4)> {
            if !i32x4_all_true(f32x4_le(f32x4_abs(self.0), f32x4_splat(80.0))) {
                return None;
            }
            let k = f32x4_nearest(f32x4_mul(self.0, f32x4_splat(core::f32::consts::LOG2_E)));
            let exponent = i32x4_shl(i32x4_add(i32x4_trunc_sat_f32x4(k), i32x4_splat(127)), 23);
            Some((F4(k), F4(exponent)))
        }
    }

    /// dst[0..16] = src[0..16] as f32 * scale
    #[inline(always)]
    pub unsafe fn dequant16(src: *const i8, scale: f32, dst: *mut f32) {
        let v = core::ptr::read_unaligned(src as *const v128);
        let lo = i16x8_extend_low_i8x16(v);
        let hi = i16x8_extend_high_i8x16(v);
        let s = f32x4_splat(scale);
        let parts = [
            i32x4_extend_low_i16x8(lo),
            i32x4_extend_high_i16x8(lo),
            i32x4_extend_low_i16x8(hi),
            i32x4_extend_high_i16x8(hi),
        ];
        for (i, p) in parts.into_iter().enumerate() {
            core::ptr::write_unaligned(dst.add(i * 4) as *mut v128, f32x4_mul(f32x4_convert_i32x4(p), s));
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod imp {
    use core::arch::aarch64::*;

    #[derive(Clone, Copy)]
    pub struct F4(float32x4_t);

    impl F4 {
        #[inline(always)]
        pub fn zero() -> F4 {
            unsafe { F4(vdupq_n_f32(0.0)) }
        }
        #[inline(always)]
        pub fn splat(v: f32) -> F4 {
            unsafe { F4(vdupq_n_f32(v)) }
        }
        #[inline(always)]
        pub unsafe fn load(p: *const f32) -> F4 {
            F4(vld1q_f32(p))
        }
        #[inline(always)]
        pub unsafe fn store(self, p: *mut f32) {
            vst1q_f32(p, self.0)
        }
        #[inline(always)]
        pub fn add(self, o: F4) -> F4 {
            unsafe { F4(vaddq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn sub(self, o: F4) -> F4 {
            unsafe { F4(vsubq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn mul(self, o: F4) -> F4 {
            unsafe { F4(vmulq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn max(self, o: F4) -> F4 {
            unsafe { F4(vmaxq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn fma(self, a: F4, b: F4) -> F4 {
            unsafe { F4(vfmaq_f32(self.0, a.0, b.0)) }
        }
        #[inline(always)]
        pub fn hsum(self) -> f32 {
            unsafe { vaddvq_f32(self.0) }
        }
        #[inline(always)]
        pub fn hmax(self) -> f32 {
            unsafe { vmaxvq_f32(self.0) }
        }
        #[inline(always)]
        pub(super) fn exp_reduction(self) -> Option<(F4, F4)> {
            unsafe {
                if vminvq_u32(vcleq_f32(vabsq_f32(self.0), vdupq_n_f32(80.0))) != u32::MAX {
                    return None;
                }
                let k = vrndnq_f32(vmulq_f32(self.0, vdupq_n_f32(core::f32::consts::LOG2_E)));
                let exponent = vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(k), vdupq_n_s32(127)));
                Some((F4(k), F4(vreinterpretq_f32_s32(exponent))))
            }
        }
    }

    #[inline(always)]
    pub unsafe fn dequant16(src: *const i8, scale: f32, dst: *mut f32) {
        let v = vld1q_s8(src);
        let lo = vmovl_s8(vget_low_s8(v));
        let hi = vmovl_s8(vget_high_s8(v));
        let s = vdupq_n_f32(scale);
        vst1q_f32(dst, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), s));
        vst1q_f32(dst.add(4), vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), s));
        vst1q_f32(dst.add(8), vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), s));
        vst1q_f32(dst.add(12), vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), s));
    }
}

#[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"), target_arch = "aarch64")))]
mod imp {
    #[derive(Clone, Copy)]
    pub struct F4([f32; 4]);

    impl F4 {
        #[inline(always)]
        pub fn zero() -> F4 {
            F4([0.0; 4])
        }
        #[inline(always)]
        pub fn splat(v: f32) -> F4 {
            F4([v; 4])
        }
        #[inline(always)]
        pub unsafe fn load(p: *const f32) -> F4 {
            F4(core::ptr::read_unaligned(p as *const [f32; 4]))
        }
        #[inline(always)]
        pub unsafe fn store(self, p: *mut f32) {
            core::ptr::write_unaligned(p as *mut [f32; 4], self.0)
        }
        #[inline(always)]
        fn map2(self, o: F4, f: impl Fn(f32, f32) -> f32) -> F4 {
            F4([f(self.0[0], o.0[0]), f(self.0[1], o.0[1]), f(self.0[2], o.0[2]), f(self.0[3], o.0[3])])
        }
        #[inline(always)]
        pub fn add(self, o: F4) -> F4 {
            self.map2(o, |a, b| a + b)
        }
        #[inline(always)]
        pub fn sub(self, o: F4) -> F4 {
            self.map2(o, |a, b| a - b)
        }
        #[inline(always)]
        pub fn mul(self, o: F4) -> F4 {
            self.map2(o, |a, b| a * b)
        }
        #[inline(always)]
        pub fn max(self, o: F4) -> F4 {
            self.map2(o, |a, b| if b > a { b } else { a })
        }
        #[inline(always)]
        pub fn fma(self, a: F4, b: F4) -> F4 {
            self.add(a.mul(b))
        }
        #[inline(always)]
        pub fn hsum(self) -> f32 {
            (self.0[0] + self.0[2]) + (self.0[1] + self.0[3])
        }
        #[inline(always)]
        pub fn hmax(self) -> f32 {
            self.0.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        }
        #[inline(always)]
        pub(super) fn exp_reduction(self) -> Option<(F4, F4)> {
            if !self.0.iter().all(|x| x.abs() <= 80.0) {
                return None;
            }
            let k = self.0.map(|x| (x * core::f32::consts::LOG2_E).round_ties_even());
            let scale = k.map(|k| f32::from_bits(((k as i32 + 127) as u32) << 23));
            Some((F4(k), F4(scale)))
        }
    }

    #[inline(always)]
    pub unsafe fn dequant16(src: *const i8, scale: f32, dst: *mut f32) {
        for i in 0..16 {
            *dst.add(i) = *src.add(i) as f32 * scale;
        }
    }
}

pub use imp::{dequant16, F4};

impl F4 {
    /// Exponentials of four lanes, with scalar handling outside [-80, 80].
    #[inline(always)]
    pub fn exp(self) -> F4 {
        let Some((k, scale)) = self.exp_reduction() else {
            let mut lanes = [0.0; 4];
            unsafe { self.store(lanes.as_mut_ptr()) };
            for x in &mut lanes {
                *x = x.exp();
            }
            return unsafe { F4::load(lanes.as_ptr()) };
        };
        // The first ln(2) part makes k * hi exact here. For |r| < 0.3466,
        // the degree-7 Taylor truncation error is < 7.4e-9 before f32 rounding.
        let r = self.sub(k.mul(F4::splat(0.693359375))).sub(k.mul(F4::splat(-2.1219444e-4)));
        let mut p = F4::splat(1.0 / 5040.0);
        for c in [1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0] {
            p = p.mul(r).add(F4::splat(c));
        }
        p.mul(scale)
    }
}

/// Dot product of two equal-length slices whose length is a multiple of 16.
#[inline(always)]
pub fn dot16(a: &[f32], b: &[f32]) -> f32 {
    debug_assert!(a.len() == b.len() && a.len() % 16 == 0);
    let (mut s0, mut s1, mut s2, mut s3) = (F4::zero(), F4::zero(), F4::zero(), F4::zero());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut i = 0;
    unsafe {
        while i < a.len() {
            s0 = s0.fma(F4::load(pa.add(i)), F4::load(pb.add(i)));
            s1 = s1.fma(F4::load(pa.add(i + 4)), F4::load(pb.add(i + 4)));
            s2 = s2.fma(F4::load(pa.add(i + 8)), F4::load(pb.add(i + 8)));
            s3 = s3.fma(F4::load(pa.add(i + 12)), F4::load(pb.add(i + 12)));
            i += 16;
        }
    }
    s0.add(s1).add(s2.add(s3)).hsum()
}

/// y += a * x over slices whose length is a multiple of 16.
#[inline(always)]
pub fn axpy16(y: &mut [f32], a: f32, x: &[f32]) {
    debug_assert!(y.len() == x.len() && y.len() % 16 == 0);
    let av = F4::splat(a);
    let (py, px) = (y.as_mut_ptr(), x.as_ptr());
    let mut i = 0;
    unsafe {
        while i < y.len() {
            for o in [0, 4, 8, 12] {
                F4::load(py.add(i + o)).fma(av, F4::load(px.add(i + o))).store(py.add(i + o));
            }
            i += 16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::F4;

    fn exp_lanes(input: [f32; 4]) -> [f32; 4] {
        let mut output = [0.0; 4];
        unsafe { F4::load(input.as_ptr()).exp().store(output.as_mut_ptr()) };
        output
    }

    #[test]
    fn exp_matches_scalar_across_range_and_reduction_boundaries() {
        let mut inputs: Vec<f32> = (0..=80_000).map(|i| -80.0 + i as f32 / 500.0).collect();
        for k in -115..=115 {
            for center in [k as f64, k as f64 + 0.5] {
                let bits = ((center * core::f64::consts::LN_2) as f32).to_bits();
                for offset in -8i32..=8 {
                    let x = f32::from_bits(bits.wrapping_add_signed(offset));
                    if x.abs() <= 80.0 {
                        inputs.push(x);
                    }
                }
            }
        }
        for chunk in inputs.chunks(4) {
            let mut lanes = [0.0; 4];
            lanes[..chunk.len()].copy_from_slice(chunk);
            for (x, actual) in lanes.into_iter().zip(exp_lanes(lanes)) {
                let expected = x.exp();
                let ulps = actual.to_bits().abs_diff(expected.to_bits());
                let relative = ((actual - expected) / expected).abs();
                assert!(ulps <= 4 && relative <= 5e-7, "exp({x}): {actual} vs {expected}, {ulps} ulp");
            }
        }
    }

    #[test]
    fn exp_preserves_extreme_and_special_values() {
        for lanes in [
            [f32::NEG_INFINITY, f32::INFINITY, f32::NAN, 0.0],
            [-104.0, -90.0, 88.0, 90.0],
            [-f32::from_bits(80.0f32.to_bits() + 1), f32::from_bits(80.0f32.to_bits() + 1), -0.0, 0.125],
        ] {
            for (x, actual) in lanes.into_iter().zip(exp_lanes(lanes)) {
                let expected = x.exp();
                if expected.is_nan() {
                    assert!(actual.is_nan());
                } else {
                    assert_eq!(actual.to_bits(), expected.to_bits(), "exp({x})");
                }
            }
        }
    }
}
