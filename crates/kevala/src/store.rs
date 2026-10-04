//! Where a loaded pack's bytes live: one aligned buffer, and typed views of the tensors in it.

use crate::kernels::Mat;
use crate::pack::{self, DType, Header, TensorInfo};

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
            let end = t.offset + t.size;
            let send = t.scales_offset + t.scales_size;
            if end > buf.len() || (t.dtype == DType::Q8 && send > buf.len()) {
                return Err(format!("tensor {} lies outside the pack ({} bytes)", t.name, buf.len()));
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

    /// Row `r` of a matrix as f32 (dequantized for q8).
    pub fn row(&self, t: &TensorInfo, r: usize, out: &mut [f32]) {
        match self.mat(t) {
            Mat::F32 { k, w, .. } => out.copy_from_slice(&w[r * k..(r + 1) * k]),
            Mat::Q8 { k, block, q, scales, .. } => {
                let nb = k / block;
                for (c, o) in out.iter_mut().enumerate() {
                    *o = q[r * k + c] as f32 * scales[r * nb + c / block];
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
