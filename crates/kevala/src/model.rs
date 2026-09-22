//! The Laya decision model: ModernBERT-large encoder, a two-layer transformer decision head,
//! an option-marker scorer, and an act head.
//!
//! The work splits in two so it can run across Web Workers without shared memory:
//!
//! - `Coord` owns the tokenizer-facing ends: embeddings, the final norm and type embedding, the
//!   scorer and the act head. It is cheap.
//! - `Trunk` owns the 30 transformer layers, or a tensor-parallel shard of them: a contiguous
//!   range of attention heads and MLP columns. A shard turns the replicated residual stream into
//!   a partial update, and the partial updates of all shards sum to the full layer update
//!   (Megatron-style), so each layer costs two all-reduces and nothing else crosses workers.
//!
//! Sequences are packed: every question of a request is one segment of a single token stream,
//! with attention confined to its segment and positions restarting at zero, which is exactly
//! what the padded PyTorch batch computes.

use crate::json::Value;
use crate::kernels::{attention, geglu, layer_norm, linear, Mat, Rope};
use crate::pack::{self, DType, Header, Layout, Slice, TensorInfo};
use std::sync::Arc;

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

#[derive(Clone, Debug)]
pub struct Config {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub global_every: usize,
    /// Half window of the sliding layers: a token sees keys with |i - j| <= window.
    pub window: usize,
    pub rope_global: f32,
    pub rope_local: f32,
    pub norm_eps: f32,
    pub head_layers: usize,
    pub head_ff: usize,
    pub head_heads: usize,
    pub head_norm_eps: f32,
    pub max_len: usize,
    pub head_max_len: usize,
    pub act_hidden: usize,
    /// Quantization block; shard boundaries fall on multiples of it.
    pub block: usize,
}

impl Config {
    pub fn from_json(c: &crate::json::Value) -> Result<Config, String> {
        let u = |k: &str| pack::get_usize(c, k);
        let f = |k: &str| pack::get_f64(c, k).map(|v| v as f32);
        let hidden = u("hidden_size")?;
        let heads = u("num_attention_heads")?;
        Ok(Config {
            hidden,
            heads,
            head_dim: hidden / heads,
            layers: u("num_hidden_layers")?,
            intermediate: u("intermediate_size")?,
            global_every: u("global_attn_every_n_layers")?,
            window: u("local_attention")? / 2,
            rope_global: f("global_rope_theta")?,
            rope_local: f("local_rope_theta")?,
            norm_eps: f("norm_eps")?,
            head_layers: u("head_layers")?,
            head_ff: u("head_ff")?,
            head_heads: u("head_heads")?,
            head_norm_eps: f("head_norm_eps")?,
            max_len: u("max_len")?,
            head_max_len: u("head_max_len")?,
            act_hidden: u("act_hidden")?,
            block: u("block").unwrap_or(32),
        })
    }

    pub fn is_global(&self, layer: usize) -> bool {
        layer % self.global_every == 0
    }

    /// Steps in one trunk pass: attention and MLP for every encoder and head layer.
    pub fn steps(&self) -> usize {
        2 * (self.layers + self.head_layers)
    }
}

/// One packed sequence.
#[derive(Clone, Debug)]
pub struct Seg {
    pub start: usize,
    pub len: usize,
    pub qtype: usize,
    /// Marker positions relative to `start`.
    pub markers: Vec<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct Batch {
    pub ids: Vec<u32>,
    pub segs: Vec<Seg>,
}

impl Batch {
    pub fn tokens(&self) -> usize {
        self.ids.len()
    }
    pub fn push(&mut self, ids: &[u32], qtype: usize, markers: &[usize]) {
        self.segs.push(Seg { start: self.ids.len(), len: ids.len(), qtype, markers: markers.to_vec() });
        self.ids.extend_from_slice(ids);
    }
}

/// Reusable per-forward buffers.
#[derive(Default)]
pub struct Scratch {
    ln: Vec<f32>,
    qkv: Vec<f32>,
    ctx: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    panel: Vec<f32>,
    kbuf: Vec<f32>,
    vbuf: Vec<f32>,
    scores: Vec<f32>,
}

fn grow(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

struct EncLayer {
    attn_norm: Option<TensorInfo>,
    wqkv: TensorInfo,
    wo: TensorInfo,
    mlp_norm: TensorInfo,
    wi: TensorInfo,
    wo2: TensorInfo,
}

struct HeadLayer {
    norm1_w: TensorInfo,
    norm1_b: TensorInfo,
    in_proj: TensorInfo,
    in_proj_b: TensorInfo,
    out_proj: TensorInfo,
    out_proj_b: TensorInfo,
    norm2_w: TensorInfo,
    norm2_b: TensorInfo,
    lin1: TensorInfo,
    lin1_b: TensorInfo,
    lin2: TensorInfo,
    lin2_b: TensorInfo,
}

/// All transformer layers, or one tensor-parallel shard of them.
pub struct Trunk {
    pub cfg: Config,
    store: Arc<Store>,
    enc: Vec<EncLayer>,
    head: Vec<HeadLayer>,
    /// The shard that adds the biases of row-parallel projections, so they count once.
    primary: bool,
    rope_global: Rope,
    rope_local: Rope,
}

impl Trunk {
    pub fn new(cfg: Config, store: Arc<Store>, primary: bool) -> Result<Trunk, String> {
        let t = |n: String| store.info(&n).cloned();
        let mut enc = Vec::new();
        for i in 0..cfg.layers {
            enc.push(EncLayer {
                attn_norm: if store.has(&format!("enc.{i}.attn_norm")) {
                    Some(t(format!("enc.{i}.attn_norm"))?)
                } else {
                    None
                },
                wqkv: t(format!("enc.{i}.wqkv"))?,
                wo: t(format!("enc.{i}.wo"))?,
                mlp_norm: t(format!("enc.{i}.mlp_norm"))?,
                wi: t(format!("enc.{i}.wi"))?,
                wo2: t(format!("enc.{i}.wo2"))?,
            });
        }
        let mut head = Vec::new();
        for i in 0..cfg.head_layers {
            let h = |s: &str| t(format!("head.{i}.{s}"));
            head.push(HeadLayer {
                norm1_w: h("norm1.w")?,
                norm1_b: h("norm1.b")?,
                in_proj: h("in_proj")?,
                in_proj_b: h("in_proj.b")?,
                out_proj: h("out_proj")?,
                out_proj_b: h("out_proj.b")?,
                norm2_w: h("norm2.w")?,
                norm2_b: h("norm2.b")?,
                lin1: h("lin1")?,
                lin1_b: h("lin1.b")?,
                lin2: h("lin2")?,
                lin2_b: h("lin2.b")?,
            });
        }
        let rope_global = Rope::new(cfg.rope_global, cfg.head_dim, cfg.max_len);
        let rope_local = Rope::new(cfg.rope_local, cfg.head_dim, cfg.max_len);
        Ok(Trunk { cfg, store, enc, head, primary, rope_global, rope_local })
    }

    /// Runs step `s` of the pass on the full residual stream `x` and writes this shard's partial
    /// update to `out` (`tokens * hidden`). Summing `out` over all shards and adding it to `x`
    /// completes the step.
    pub fn step(&self, s: usize, x: &[f32], batch: &Batch, out: &mut [f32], sc: &mut Scratch) {
        let layer = s / 2;
        let (d, t) = (self.cfg.hidden, batch.tokens());
        let st = &self.store;
        grow(&mut sc.ln, t * d);
        if layer < self.cfg.layers {
            let l = &self.enc[layer];
            if s % 2 == 0 {
                let mut ln = std::mem::take(&mut sc.ln);
                let h: &[f32] = match &l.attn_norm {
                    Some(w) => {
                        layer_norm(x, t, d, st.f32s(w), None, self.cfg.norm_eps, &mut ln);
                        &ln
                    }
                    None => x,
                };
                let global = self.cfg.is_global(layer);
                let rope = if global { &self.rope_global } else { &self.rope_local };
                let window = if global { None } else { Some(self.cfg.window) };
                self.attend(h, batch, st.mat(&l.wqkv), None, Some(rope), window, sc);
                sc.ln = ln;
                linear(&sc.ctx, t, st.mat(&l.wo), None, out, &mut sc.panel);
            } else {
                layer_norm(x, t, d, st.f32s(&l.mlp_norm), None, self.cfg.norm_eps, &mut sc.ln);
                let wi = st.mat(&l.wi);
                let inter = wi.n() / 2;
                grow(&mut sc.up, t * wi.n());
                grow(&mut sc.act, t * inter);
                linear(&sc.ln, t, wi, None, &mut sc.up, &mut sc.panel);
                geglu(&sc.up, t, inter, &mut sc.act);
                linear(&sc.act, t, st.mat(&l.wo2), None, out, &mut sc.panel);
            }
        } else {
            let l = &self.head[layer - self.cfg.layers];
            let eps = self.cfg.head_norm_eps;
            if s % 2 == 0 {
                layer_norm(x, t, d, st.f32s(&l.norm1_w), Some(st.f32s(&l.norm1_b)), eps, &mut sc.ln);
                let ln = std::mem::take(&mut sc.ln);
                self.attend(&ln, batch, st.mat(&l.in_proj), Some(st.f32s(&l.in_proj_b)), None, None, sc);
                sc.ln = ln;
                let b = self.primary.then(|| st.f32s(&l.out_proj_b));
                linear(&sc.ctx, t, st.mat(&l.out_proj), b, out, &mut sc.panel);
            } else {
                layer_norm(x, t, d, st.f32s(&l.norm2_w), Some(st.f32s(&l.norm2_b)), eps, &mut sc.ln);
                let lin1 = st.mat(&l.lin1);
                grow(&mut sc.act, t * lin1.n());
                linear(&sc.ln, t, lin1, Some(st.f32s(&l.lin1_b)), &mut sc.act, &mut sc.panel);
                for v in sc.act[..t * lin1.n()].iter_mut() {
                    *v = v.max(0.0);
                }
                let b = self.primary.then(|| st.f32s(&l.lin2_b));
                linear(&sc.act, t, st.mat(&l.lin2), b, out, &mut sc.panel);
            }
        }
    }

    /// qkv projection, rotary embedding, and per-segment attention into `sc.ctx`.
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        h: &[f32],
        batch: &Batch,
        wqkv: Mat,
        bias: Option<&[f32]>,
        rope: Option<&Rope>,
        window: Option<usize>,
        sc: &mut Scratch,
    ) {
        let t = batch.tokens();
        let hd = self.cfg.head_dim;
        let width = wqkv.n() / 3;
        let heads = width / hd;
        grow(&mut sc.qkv, t * 3 * width);
        grow(&mut sc.ctx, t * width);
        linear(h, t, wqkv, bias, &mut sc.qkv, &mut sc.panel);
        if let Some(rope) = rope {
            for seg in &batch.segs {
                for p in 0..seg.len {
                    let row = &mut sc.qkv[(seg.start + p) * 3 * width..(seg.start + p + 1) * 3 * width];
                    for hh in 0..2 * heads {
                        rope.apply(&mut row[hh * hd..(hh + 1) * hd], p);
                    }
                }
            }
        }
        for seg in &batch.segs {
            attention(
                &sc.qkv,
                seg.start,
                seg.len,
                heads,
                hd,
                window,
                &mut sc.ctx,
                &mut sc.kbuf,
                &mut sc.vbuf,
                &mut sc.scores,
            );
        }
    }
}

/// Raw model outputs for one segment.
#[derive(Clone, Debug)]
pub struct SegOut {
    pub logits: Vec<f32>,
    pub act_logits: [f32; 2],
}

/// Embeddings in, decisions out.
pub struct Coord {
    pub cfg: Config,
    store: Arc<Store>,
    emb: TensorInfo,
    emb_norm: TensorInfo,
    final_norm: TensorInfo,
    type_emb: TensorInfo,
    sc_norm_w: TensorInfo,
    sc_norm_b: TensorInfo,
    sc_l1: TensorInfo,
    sc_l1_b: TensorInfo,
    sc_l2: TensorInfo,
    sc_l2_b: TensorInfo,
    act_l1: TensorInfo,
    act_l1_b: TensorInfo,
    act_l2: TensorInfo,
    act_l2_b: TensorInfo,
}

impl Coord {
    pub fn new(cfg: Config, store: Arc<Store>) -> Result<Coord, String> {
        let t = |n: &str| store.info(n).cloned();
        Ok(Coord {
            emb: t("emb")?,
            emb_norm: t("emb_norm")?,
            final_norm: t("final_norm")?,
            type_emb: t("type_emb")?,
            sc_norm_w: t("scorer.norm.w")?,
            sc_norm_b: t("scorer.norm.b")?,
            sc_l1: t("scorer.l1")?,
            sc_l1_b: t("scorer.l1.b")?,
            sc_l2: t("scorer.l2")?,
            sc_l2_b: t("scorer.l2.b")?,
            act_l1: t("act.l1")?,
            act_l1_b: t("act.l1.b")?,
            act_l2: t("act.l2")?,
            act_l2_b: t("act.l2.b")?,
            cfg,
            store,
        })
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    /// Token embeddings followed by the embedding layer norm.
    pub fn embed(&self, batch: &Batch) -> Vec<f32> {
        let d = self.cfg.hidden;
        let t = batch.tokens();
        let vocab = self.emb.rows();
        let mut raw = vec![0.0; t * d];
        for (i, &id) in batch.ids.iter().enumerate() {
            self.store.row(&self.emb, (id as usize).min(vocab - 1), &mut raw[i * d..(i + 1) * d]);
        }
        let mut x = vec![0.0; t * d];
        layer_norm(&raw, t, d, self.store.f32s(&self.emb_norm), None, self.cfg.norm_eps, &mut x);
        x
    }

    /// Between the encoder and the decision head: final norm, then the question-type embedding.
    pub fn bridge(&self, x: &mut [f32], batch: &Batch) {
        let d = self.cfg.hidden;
        let t = batch.tokens();
        let src = x.to_vec();
        layer_norm(&src, t, d, self.store.f32s(&self.final_norm), None, self.cfg.norm_eps, x);
        let te = self.store.f32s(&self.type_emb);
        for seg in &batch.segs {
            let e = &te[seg.qtype * d..(seg.qtype + 1) * d];
            for r in seg.start..seg.start + seg.len {
                for (v, a) in x[r * d..(r + 1) * d].iter_mut().zip(e) {
                    *v += a;
                }
            }
        }
    }

    /// Scorer on every marker, act head on every segment's first token.
    pub fn score(&self, x: &[f32], batch: &Batch) -> Vec<SegOut> {
        let d = self.cfg.hidden;
        let st = &self.store;
        let mut panel = Vec::new();
        let rows: Vec<usize> = batch.segs.iter().flat_map(|s| s.markers.iter().map(move |m| s.start + m)).collect();
        let m = rows.len();
        let mut gathered = vec![0.0; m * d];
        for (i, &r) in rows.iter().enumerate() {
            gathered[i * d..(i + 1) * d].copy_from_slice(&x[r * d..(r + 1) * d]);
        }
        let mut ln = vec![0.0; m * d];
        let eps = self.cfg.head_norm_eps;
        layer_norm(&gathered, m, d, st.f32s(&self.sc_norm_w), Some(st.f32s(&self.sc_norm_b)), eps, &mut ln);
        let mut hid = vec![0.0; m * d];
        linear(&ln, m, st.mat(&self.sc_l1), Some(st.f32s(&self.sc_l1_b)), &mut hid, &mut panel);
        for v in hid.iter_mut() {
            *v = crate::kernels::gelu(*v);
        }
        let mut logits = vec![0.0; m];
        linear(&hid, m, st.mat(&self.sc_l2), Some(st.f32s(&self.sc_l2_b)), &mut logits, &mut panel);

        let ah = self.cfg.act_hidden;
        let mut outs = Vec::with_capacity(batch.segs.len());
        let mut at = 0;
        let mut inp = vec![0.0; d + 4];
        let mut hid = vec![0.0; ah];
        for seg in &batch.segs {
            let k = seg.markers.len();
            let lg = logits[at..at + k].to_vec();
            at += k;
            // act features from the raw (untempered) distribution, as the PyTorch head computes them
            let mx = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = lg.iter().map(|v| (v - mx).exp()).collect();
            let s: f32 = e.iter().sum();
            let mut p: Vec<f32> = e.iter().map(|v| v / s).collect();
            let kf = k.max(2) as f32;
            let ent = -p.iter().map(|&v| v * v.max(1e-9).ln()).sum::<f32>() / kf.ln();
            p.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let top1 = p.first().copied().unwrap_or(0.0);
            let top2 = p.get(1).copied().unwrap_or(0.0);
            inp[..d].copy_from_slice(&x[seg.start * d..(seg.start + 1) * d]);
            inp[d..].copy_from_slice(&[top1, top1 - top2, ent, kf / 255.0]);
            linear(&inp, 1, st.mat(&self.act_l1), Some(st.f32s(&self.act_l1_b)), &mut hid, &mut panel);
            for v in hid.iter_mut() {
                *v = crate::kernels::gelu(*v);
            }
            let mut act = [0.0f32; 2];
            linear(&hid, 1, st.mat(&self.act_l2), Some(st.f32s(&self.act_l2_b)), &mut act, &mut panel);
            outs.push(SegOut { logits: lg, act_logits: act });
        }
        outs
    }
}

/// How shard `index` of `count` splits heads and MLP columns. Ranges are in units of whole heads
/// and whole quantization blocks so every shard slices q8 tensors on block boundaries.
#[derive(Clone, Copy, Debug)]
pub struct ShardPlan {
    pub index: usize,
    pub count: usize,
}

impl ShardPlan {
    pub fn split(&self, units: usize) -> (usize, usize) {
        (self.index * units / self.count, (self.index + 1) * units / self.count)
    }
}

/// The sub-pack a trunk shard needs: its heads' rows of every qkv projection, the matching
/// columns of every output projection, and its share of every MLP.
pub fn shard_layout(cfg: &Config, h: &Header, plan: ShardPlan) -> Result<Layout, String> {
    if plan.count == 0 || plan.index >= plan.count || plan.count > cfg.heads.min(cfg.head_heads) {
        return Err(format!("cannot split {} heads into shard {} of {}", cfg.heads, plan.index, plan.count));
    }
    let (hd, d, unit) = (cfg.head_dim, cfg.hidden, cfg.block);
    let qkv = |heads: usize| {
        let (h0, h1) = plan.split(heads);
        (vec![(h0 * hd, h1 * hd), (d + h0 * hd, d + h1 * hd), (2 * d + h0 * hd, 2 * d + h1 * hd)], h0 * hd, h1 * hd)
    };
    let (enc_qkv, enc_c0, enc_c1) = qkv(cfg.heads);
    let (head_qkv, head_c0, head_c1) = qkv(cfg.head_heads);
    let (u0, u1) = plan.split(cfg.intermediate / unit);
    let (i0, i1, inter) = (u0 * unit, u1 * unit, cfg.intermediate);
    let (f0, f1) = plan.split(cfg.head_ff / unit);
    let (f0, f1) = (f0 * unit, f1 * unit);
    let select = |t: &TensorInfo| -> Option<Slice> {
        let (scope, rest) = t.name.split_once('.')?;
        let (_, what) = rest.split_once('.')?;
        match (scope, what) {
            ("enc", "attn_norm" | "mlp_norm") => Some(Slice::Whole),
            ("enc", "wqkv") => Some(Slice::Rows(enc_qkv.clone())),
            ("enc", "wo") => Some(Slice::Cols(enc_c0, enc_c1)),
            ("enc", "wi") => Some(Slice::Rows(vec![(i0, i1), (inter + i0, inter + i1)])),
            ("enc", "wo2") => Some(Slice::Cols(i0, i1)),
            ("head", "in_proj" | "in_proj.b") => Some(Slice::Rows(head_qkv.clone())),
            ("head", "out_proj") => Some(Slice::Cols(head_c0, head_c1)),
            ("head", "lin1" | "lin1.b") => Some(Slice::Rows(vec![(f0, f1)])),
            ("head", "lin2") => Some(Slice::Cols(f0, f1)),
            ("head", _) => Some(Slice::Whole),
            _ => None,
        }
    };
    let n = |v: usize| Value::Int(v.to_string());
    let meta = vec![(
        "shard".to_string(),
        Value::Object(vec![("index".into(), n(plan.index)), ("count".into(), n(plan.count))]),
    )];
    pack::subset(h, &select, false, meta)
}

/// The sub-pack the coordinator needs: tokenizer, embeddings, scorer, act head.
pub fn coord_layout(h: &Header) -> Result<Layout, String> {
    let select = |t: &TensorInfo| (!is_trunk(&t.name)).then_some(Slice::Whole);
    pack::subset(h, &select, true, Vec::new())
}

/// Tensors of the transformer layers, which run in shards or on the GPU.
pub fn is_trunk(name: &str) -> bool {
    name.starts_with("enc.") || name.starts_with("head.") || name.starts_with("L.")
}

/// Every trunk tensor, whole: what a GPU uploads.
pub fn trunk_layout(h: &Header) -> Result<Layout, String> {
    let select = |t: &TensorInfo| is_trunk(&t.name).then_some(Slice::Whole);
    pack::subset(h, &select, false, Vec::new())
}

/// Loads a pack (whole or a sub-pack) into a store.
pub fn load_store(buf: AlignedBuf) -> Result<(Store, Header), String> {
    let h = pack::parse_header(buf.as_slice())?;
    if buf.len() < h.total_size {
        return Err(format!("pack is truncated: {} of {} bytes", buf.len(), h.total_size));
    }
    Ok((Store::new(buf, h.tensors.clone())?, h))
}

/// Builds shard `plan` natively from a whole pack held in memory.
pub fn build_shard(cfg: &Config, full: &[u8], plan: ShardPlan) -> Result<Store, String> {
    let h = pack::parse_header(full)?;
    let bytes = shard_layout(cfg, &h, plan)?.apply(full);
    Ok(load_store(AlignedBuf::from_slice(&bytes))?.0)
}
