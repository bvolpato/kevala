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

    /// Transpose four four-lane rows into four output vectors. `dst_stride` is in f32s.
    #[inline(always)]
    pub unsafe fn transpose4x4(
        a0: *const f32,
        a1: *const f32,
        a2: *const f32,
        a3: *const f32,
        dst: *mut f32,
        dst_stride: usize,
    ) {
        let r0 = core::ptr::read_unaligned(a0 as *const v128);
        let r1 = core::ptr::read_unaligned(a1 as *const v128);
        let r2 = core::ptr::read_unaligned(a2 as *const v128);
        let r3 = core::ptr::read_unaligned(a3 as *const v128);
        let t0 = i32x4_shuffle::<0, 4, 1, 5>(r0, r1);
        let t1 = i32x4_shuffle::<2, 6, 3, 7>(r0, r1);
        let t2 = i32x4_shuffle::<0, 4, 1, 5>(r2, r3);
        let t3 = i32x4_shuffle::<2, 6, 3, 7>(r2, r3);
        let o0 = i32x4_shuffle::<0, 1, 4, 5>(t0, t2);
        let o1 = i32x4_shuffle::<2, 3, 6, 7>(t0, t2);
        let o2 = i32x4_shuffle::<0, 1, 4, 5>(t1, t3);
        let o3 = i32x4_shuffle::<2, 3, 6, 7>(t1, t3);
        core::ptr::write_unaligned(dst as *mut v128, o0);
        core::ptr::write_unaligned(dst.add(dst_stride) as *mut v128, o1);
        core::ptr::write_unaligned(dst.add(2 * dst_stride) as *mut v128, o2);
        core::ptr::write_unaligned(dst.add(3 * dst_stride) as *mut v128, o3);
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

    /// Transpose four four-lane rows into four output vectors. `dst_stride` is in f32s.
    #[inline(always)]
    pub unsafe fn transpose4x4(
        a0: *const f32,
        a1: *const f32,
        a2: *const f32,
        a3: *const f32,
        dst: *mut f32,
        dst_stride: usize,
    ) {
        let r0 = vld1q_f32(a0);
        let r1 = vld1q_f32(a1);
        let r2 = vld1q_f32(a2);
        let r3 = vld1q_f32(a3);
        let t0 = vtrn1q_f32(r0, r1);
        let t1 = vtrn2q_f32(r0, r1);
        let t2 = vtrn1q_f32(r2, r3);
        let t3 = vtrn2q_f32(r2, r3);
        let o0 = vcombine_f32(vget_low_f32(t0), vget_low_f32(t2));
        let o1 = vcombine_f32(vget_low_f32(t1), vget_low_f32(t3));
        let o2 = vcombine_f32(vget_high_f32(t0), vget_high_f32(t2));
        let o3 = vcombine_f32(vget_high_f32(t1), vget_high_f32(t3));
        vst1q_f32(dst, o0);
        vst1q_f32(dst.add(dst_stride), o1);
        vst1q_f32(dst.add(2 * dst_stride), o2);
        vst1q_f32(dst.add(3 * dst_stride), o3);
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
    }

    #[inline(always)]
    pub unsafe fn dequant16(src: *const i8, scale: f32, dst: *mut f32) {
        for i in 0..16 {
            *dst.add(i) = *src.add(i) as f32 * scale;
        }
    }

    /// Transpose four four-lane rows into four output vectors. `dst_stride` is in f32s.
    #[inline(always)]
    pub unsafe fn transpose4x4(
        a0: *const f32,
        a1: *const f32,
        a2: *const f32,
        a3: *const f32,
        dst: *mut f32,
        dst_stride: usize,
    ) {
        let r0 = core::ptr::read_unaligned(a0 as *const [f32; 4]);
        let r1 = core::ptr::read_unaligned(a1 as *const [f32; 4]);
        let r2 = core::ptr::read_unaligned(a2 as *const [f32; 4]);
        let r3 = core::ptr::read_unaligned(a3 as *const [f32; 4]);
        for i in 0..4 {
            let row = [r0[i], r1[i], r2[i], r3[i]];
            core::ptr::write_unaligned(dst.add(i * dst_stride) as *mut [f32; 4], row);
        }
    }
}

pub use imp::{dequant16, transpose4x4, F4};

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
