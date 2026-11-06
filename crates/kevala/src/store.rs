//! Where a loaded pack's bytes live: one aligned buffer, and typed views of the tensors in it.

use crate::kernels::Mat;
use crate::pack::{self, DType, Header, TensorInfo};
use crate::simd::dequant16;

/// A 64-byte aligned heap buffer that WebAssembly callers can fill in place.
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    fn layout(len: usize) -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(len.max(1), pack::ALIGN).expect("buffer too large")
    }
    pub fn new(len: usize) -> AlignedBuf {
        let ptr = unsafe { std::alloc::alloc_zeroed(Self::layout(len)) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(Self::layout(len));
        }
        AlignedBuf { ptr, len }
    }
    pub fn from_slice(b: &[u8]) -> AlignedBuf {
        let mut a = AlignedBuf::new(b.len());
        a.as_mut_slice().copy_from_slice(b);
        a
    }
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Takes back a buffer handed out with `into_raw`.
    ///
    /// # Safety
    /// `ptr` and `len` must come from `into_raw` and be used once.
    pub unsafe fn from_raw(ptr: *mut u8, len: usize) -> AlignedBuf {
        AlignedBuf { ptr, len }
    }
    pub fn into_raw(self) -> (*mut u8, usize) {
        let r = (self.ptr, self.len);
        std::mem::forget(self);
        r
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, Self::layout(self.len)) }
    }
}

/// Tensors living in one aligned buffer.
pub struct Store {
    buf: AlignedBuf,
    tensors: Vec<TensorInfo>,
}

impl Store {
    pub fn new(buf: AlignedBuf, tensors: Vec<TensorInfo>) -> Result<Store, String> {
        for t in &tensors {
            if t.dtype == DType::Q8 && (t.block == 0 || t.block % 16 != 0) {
                return Err(format!("tensor {}: q8 block must be a positive multiple of 16", t.name));
            }
            let end = t.offset.checked_add(t.size);
            let send = t.scales_offset.checked_add(t.scales_size);
            if end.is_none_or(|end| end > buf.len()) || (t.dtype == DType::Q8 && send.is_none_or(|end| end > buf.len()))
            {
                return Err(format!("tensor {} lies outside the pack ({} bytes)", t.name, buf.len()));
            }
            let element = match t.dtype {
                DType::F32 => 4,
                DType::Bf16 => 2,
                DType::Q8 => 1,
            };
            if t.offset % element != 0
                || t.size % element != 0
                || (t.dtype == DType::Q8 && (t.scales_offset % 4 != 0 || t.scales_size % 4 != 0))
            {
                return Err(format!("tensor {} has an unaligned typed view", t.name));
            }
        }
        Ok(Store { buf, tensors })
    }

    pub fn bytes(&self) -> &[u8] {
        self.buf.as_slice()
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo, String> {
        self.tensors.iter().find(|t| t.name == name).ok_or_else(|| format!("pack has no tensor {name}"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.iter().any(|t| t.name == name)
    }

    pub fn f32s(&self, t: &TensorInfo) -> &[f32] {
        debug_assert!(t.dtype == DType::F32);
        unsafe { std::slice::from_raw_parts(self.buf.ptr.add(t.offset) as *const f32, t.size / 4) }
    }

    pub fn mat(&self, t: &TensorInfo) -> Mat<'_> {
        let (n, k) = (t.rows(), t.cols());
        match t.dtype {
            DType::F32 => Mat::F32 { n, k, w: self.f32s(t) },
            DType::Bf16 => Mat::Bf16 {
                n,
                k,
                w: unsafe { std::slice::from_raw_parts(self.buf.ptr.add(t.offset) as *const u16, t.size / 2) },
            },
            DType::Q8 => unsafe {
                Mat::Q8 {
                    n,
                    k,
                    block: t.block,
                    q: std::slice::from_raw_parts(self.buf.ptr.add(t.offset) as *const i8, t.size),
                    scales: std::slice::from_raw_parts(
                        self.buf.ptr.add(t.scales_offset) as *const f32,
                        t.scales_size / 4,
                    ),
                }
            },
        }
    }

    /// Row `r` of a matrix as f32 (widened for bf16, dequantized for q8).
    pub fn row(&self, t: &TensorInfo, r: usize, out: &mut [f32]) {
        match self.mat(t) {
            Mat::F32 { k, w, .. } => out.copy_from_slice(&w[r * k..(r + 1) * k]),
            Mat::Bf16 { k, w, .. } => {
                for (value, dst) in w[r * k..(r + 1) * k].iter().zip(out) {
                    *dst = crate::convert::bf16_to_f32(*value);
                }
            }
            Mat::Q8 { k, block, q, scales, .. } => {
                let nb = k / block;
                #[cfg(target_arch = "x86_64")]
                if out.len() >= 64 && std::arch::is_x86_feature_detected!("avx2") {
                    // Amortize runtime dispatch over at least four sixteen-value chunks.
                    unsafe { crate::simd::dequant_q8_row_avx2(&q[r * k..], &scales[r * nb..], block, out) };
                    return;
                }
                for (b, dst) in out.chunks_mut(block).enumerate() {
                    let scale = scales[r * nb + b];
                    let start = r * k + b * block;
                    let src = &q[start..start + dst.len()];
                    let mut src = src.chunks_exact(16);
                    let mut dst = dst.chunks_exact_mut(16);
                    for (src, dst) in src.by_ref().zip(dst.by_ref()) {
                        // Both slices contain sixteen elements within one scale block.
                        unsafe { dequant16(src.as_ptr(), scale, dst.as_mut_ptr()) };
                    }
                    for (src, dst) in src.remainder().iter().zip(dst.into_remainder()) {
                        *dst = *src as f32 * scale;
                    }
                }
            }
        }
    }
}

/// Loads a pack (whole or a sub-pack) into a store.
pub fn load_store(buf: AlignedBuf) -> Result<(Store, Header), String> {
    let h = pack::parse_header(buf.as_slice())?;
    if buf.len() < h.total_size {
        return Err(format!("pack is truncated: {} of {} bytes", buf.len(), h.total_size));
    }
    Ok((Store::new(buf, h.tensors.clone())?, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_q8_row_matches_scalar_with_unaligned_inputs_and_tails() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let scales = [
            0.,
            -0.,
            0.125,
            -0.375,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::MAX / 128.,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0xffc0_5678),
            f32::from_bits(0x7f80_1234),
            f32::from_bits(0xff80_5678),
        ];
        for block in [16, 32, 64] {
            let len = scales.len() * block;
            let q: Vec<i8> = (0..len + 1).map(|i| i.wrapping_mul(73).wrapping_add(19) as i8).collect();
            let q = &q[1..];
            for width in 0..=len {
                let mut out = vec![f32::from_bits(0x7fc0_1234); width + 2];
                // Feature detection above permits the ISA-specific call; slice offsets test unaligned loads/stores.
                unsafe { crate::simd::dequant_q8_row_avx2(q, &scales, block, &mut out[1..width + 1]) };
                assert_eq!(out[0].to_bits(), 0x7fc0_1234);
                assert_eq!(out[width + 1].to_bits(), 0x7fc0_1234);
                for (c, value) in out[1..width + 1].iter().enumerate() {
                    let expected = q[c] as f32 * scales[c / block];
                    assert_eq!(value.to_bits(), expected.to_bits(), "block={block}, width={width}, c={c}");
                }
            }
        }
    }

    #[test]
    fn q8_rows_match_scalar_dequantization_bitwise() {
        for block in [16, 32, 64, 128] {
            for k in [block, block * 3, 256, 384] {
                let rows = 5;
                let nb = k / block;
                let scales = [
                    0.,
                    -0.,
                    0.125,
                    -0.375,
                    f32::MIN_POSITIVE,
                    f32::from_bits(1),
                    f32::MAX / 128.,
                    f32::MAX,
                    -f32::MAX,
                ];
                let mut writer = pack::Writer::new(crate::json::Value::Null, crate::json::Value::Null, vec![]);
                writer.add_q8("weight", rows, k, block);
                let (_, tensors, total) = writer.layout();
                let tensor = &tensors[0];
                let mut buf = AlignedBuf::new(total);
                for i in 0..tensor.size {
                    buf.as_mut_slice()[tensor.offset + i] = i.wrapping_mul(73).wrapping_add(19) as u8;
                }
                for i in 0..tensor.scales_size / 4 {
                    let at = tensor.scales_offset + i * 4;
                    buf.as_mut_slice()[at..at + 4].copy_from_slice(&scales[i % scales.len()].to_le_bytes());
                }
                let store = Store::new(buf, tensors.clone()).unwrap();
                let Mat::Q8 { q, scales, .. } = store.mat(tensor) else { unreachable!() };
                for r in 0..rows {
                    for len in (0..=k).chain([k + 5].into_iter().filter(|_| r + 1 < rows)) {
                        let mut storage = vec![f32::from_bits(0x7fc0_1234); len + 2];
                        store.row(tensor, r, &mut storage[1..len + 1]);
                        assert_eq!(storage[0].to_bits(), 0x7fc0_1234);
                        assert_eq!(storage[len + 1].to_bits(), 0x7fc0_1234);
                        for (c, value) in storage[1..len + 1].iter().enumerate() {
                            let expected = q[r * k + c] as f32 * scales[r * nb + c / block];
                            assert_eq!(
                                value.to_bits(),
                                expected.to_bits(),
                                "block={block}, k={k}, r={r}, len={len}, c={c}"
                            );
                        }
                    }
                }
                store.row(tensor, rows, &mut []);
            }
        }
    }

    #[test]
    fn rejects_invalid_bf16_typed_view_ranges() {
        let tensor = |offset, size| TensorInfo {
            name: "weight".into(),
            dtype: DType::Bf16,
            shape: vec![64],
            offset,
            size,
            block: 0,
            scales_offset: 0,
            scales_size: 0,
        };
        assert!(Store::new(AlignedBuf::new(128), vec![tensor(usize::MAX - 63, 128)]).is_err());
        assert!(Store::new(AlignedBuf::new(128), vec![tensor(1, 2)]).is_err());
    }
}
