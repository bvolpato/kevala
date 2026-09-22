//! Kev: a Qwen3.5 decoder (Gated DeltaNet + gated attention) with a pointer head.
//!
//! Port of jaredpalmer/kev (kev/model.py, kev/api.py). A request becomes one causal row per
//! question, `<state> state <q> instructions (<opt> option </opt>)* <decide>`, and the pointer head
//! scores every `</opt>` hidden state against `<decide>`. Rows share the state, so the state runs
//! once: its recurrent DeltaNet states, conv tails and attention keys/values seed every question
//! branch, which is exactly Kev's own state-prefix cache.

use crate::json::{self, Value};
use crate::kernels::linear;
use crate::model::{AlignedBuf, Store};
use crate::pack::{self, TensorInfo};
use crate::simd::{axpy16, dot16, F4};
use crate::tokenizer::Tokenizer;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct KevConfig {
    pub hidden: usize,
    pub layers: usize,
    /// true = full attention, false = Gated DeltaNet
    pub full: Vec<bool>,
    pub intermediate: usize,
    pub eps: f32,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary: usize,
    pub rope_theta: f32,
    pub lin_heads: usize,
    pub lin_k: usize,
    pub lin_v: usize,
    pub conv: usize,
    pub ptr_dim: usize,
    pub temperature: f32,
    pub max_state: usize,
    pub max_branch: usize,
    /// `<state> <q> <opt> </opt> <decide>`
    pub tokens: [u32; 5],
}

impl KevConfig {
    pub fn from_json(c: &Value) -> Result<KevConfig, String> {
        let u = |k: &str| pack::get_usize(c, k);
        let f = |k: &str| pack::get_f64(c, k).map(|v| v as f32);
        let full = c
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or("kev config: no layer_types")?
            .iter()
            .map(|v| v.as_str() == Some("full_attention"))
            .collect::<Vec<_>>();
        let t = c
            .get("template")
            .and_then(|t| t.get("token_ids"))
            .and_then(Value::as_array)
            .ok_or("kev config: no template token ids")?;
        let mut tokens = [0u32; 5];
        for (i, v) in t.iter().take(5).enumerate() {
            tokens[i] = v.as_usize().ok_or("kev config: bad token id")? as u32;
        }
        Ok(KevConfig {
            hidden: u("hidden_size")?,
            layers: full.len(),
            full,
            intermediate: u("intermediate_size")?,
            eps: f("rms_norm_eps")?,
            heads: u("num_attention_heads")?,
            kv_heads: u("num_key_value_heads")?,
            head_dim: u("head_dim")?,
            rotary: u("rotary_dim")?,
            rope_theta: f("rope_theta")?,
            lin_heads: u("linear_num_value_heads")?,
            lin_k: u("linear_key_head_dim")?,
            lin_v: u("linear_value_head_dim")?,
            conv: u("linear_conv_kernel_dim")?,
            ptr_dim: u("pointer_dim")?,
            temperature: f("temperature")?,
            max_state: u("max_state").unwrap_or(8192),
            max_branch: u("max_branch").unwrap_or(8192),
            tokens,
        })
    }
    fn lin_dim(&self) -> usize {
        self.lin_heads * self.lin_k * 2 + self.lin_heads * self.lin_v
    }
}

// ------------------------------------------------------------------ request template (kev/api.py)

fn py_str(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => (if *b { "True" } else { "False" }).into(),
        Value::Int(s) => s.clone(),
        Value::Float(f) => {
            if f.is_nan() {
                "nan".into()
            } else if f.is_infinite() {
                (if *f > 0.0 { "inf" } else { "-inf" }).into()
            } else {
                let mut s = String::new();
                json::write_py_float(&mut s, *f);
                s
            }
        }
        Value::Str(s) => s.clone(),
        _ => String::new(),
    }
}

/// `render`: objects and arrays become labelled text, field names kept as labels.
pub fn render(v: &Value, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match v {
        Value::Null => String::new(),
        Value::Array(a) => {
            a.iter().map(|x| format!("{pad}- {}", render(x, indent + 1).trim_start())).collect::<Vec<_>>().join("\n")
        }
        Value::Object(m) => m
            .iter()
            .map(|(k, x)| match x {
                Value::Object(_) | Value::Array(_) => format!("{pad}{k}:\n{}", render(x, indent + 1)),
                _ => format!("{pad}{k}: {}", render(x, 0)),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        other => py_str(other),
    }
}

fn option_text(name: &str, desc: Option<&Value>) -> String {
    match desc {
        None | Some(Value::Null) => name.to_string(),
        Some(Value::Str(s)) if s.is_empty() => name.to_string(),
        Some(d) => format!("{name}: {}", render(d, 0)),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Noul,
    Choice,
    Score,
}

#[derive(Clone, Debug)]
pub struct KevQuestion {
    pub id: String,
    pub kind: Kind,
    pub keys: Vec<String>,
    pub instructions: String,
    pub options: Vec<String>,
}

pub fn parse_questions(qs: &Value) -> Result<Vec<KevQuestion>, String> {
    let m = qs.as_object().ok_or("questions must be an object of {id: question}")?;
    if m.is_empty() {
        return Err("questions must not be empty".into());
    }
    let mut out = Vec::new();
    for (id, q) in m {
        let t = q.get("type").and_then(Value::as_str).ok_or_else(|| format!("question {id:?}: missing type"))?;
        let instructions = render(q.get("instructions").unwrap_or(&Value::Null), 0);
        let crit = q.get("criteria");
        let (kind, keys, options) = match t {
            "noul" => {
                let c = crit.filter(|c| !c.is_null());
                let get = |k: &str| c.and_then(|c| c.get(k));
                (
                    Kind::Noul,
                    vec!["false".into(), "true".into()],
                    vec![option_text("no", get("false")), option_text("yes", get("true"))],
                )
            }
            "choice" => {
                let mut keys = Vec::new();
                let mut opts = Vec::new();
                match crit {
                    Some(Value::Object(c)) => {
                        for (k, v) in c {
                            keys.push(k.clone());
                            opts.push(option_text(k, Some(v)));
                        }
                    }
                    // a list of names, as Laya accepts, means names without descriptions
                    Some(Value::Array(a)) => {
                        for v in a {
                            let k = py_str(v);
                            if !keys.contains(&k) {
                                opts.push(k.clone());
                                keys.push(k);
                            }
                        }
                    }
                    _ => return Err(format!("question {id:?}: choice needs criteria")),
                }
                if keys.is_empty() || keys.len() > 255 {
                    return Err(format!("question {id:?}: choice needs 1..255 options"));
                }
                (Kind::Choice, keys, opts)
            }
            "score" => {
                let c = crit
                    .and_then(Value::as_array)
                    .ok_or_else(|| format!("question {id:?}: score needs a criteria list"))?;
                if c.len() < 2 || c.len() > 255 {
                    return Err(format!("question {id:?}: score needs 2..255 levels"));
                }
                (Kind::Score, (0..c.len()).map(|i| i.to_string()).collect(), c.iter().map(|x| render(x, 0)).collect())
            }
            other => return Err(format!("question {id:?}: unknown type {other:?}")),
        };
        out.push(KevQuestion { id: id.clone(), kind, keys, instructions, options });
    }
    Ok(out)
}

/// Caller text can never forge a delimiter: `<|name|>` is rewritten to `<¦name¦>` first.
pub fn escape_specials(text: &str) -> String {
    let b = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    let mut last = 0;
    while i + 1 < b.len() {
        if b[i] == b'<' && b[i + 1] == b'|' {
            let mut j = i + 2;
            while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
                j += 1;
            }
            if j > i + 2 && j + 1 < b.len() && b[j] == b'|' && b[j + 1] == b'>' {
                out.push_str(&text[last..i]);
                out.push_str("<¦");
                out.push_str(&text[i + 2..j]);
                out.push_str("¦>");
                i = j + 2;
                last = i;
                continue;
            }
        }
        i += 1;
    }
    out.push_str(&text[last..]);
    out
}

/// One question row, relative to its state.
#[derive(Clone, Debug)]
pub struct Branch {
    pub ids: Vec<u32>,
    pub decide: usize,
    pub opts: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Encoded {
    pub state: Vec<u32>,
    pub branches: Vec<Branch>,
}

impl Encoded {
    pub fn tokens(&self) -> usize {
        self.state.len() + self.branches.iter().map(|b| b.ids.len()).sum::<usize>()
    }
}

pub fn encode(tok: &Tokenizer, cfg: &KevConfig, state: &str, qs: &[KevQuestion]) -> Result<Encoded, String> {
    let [st, q_id, o_id, c_id, d_id] = cfg.tokens;
    let user = |t: &str| tok.encode(&escape_specials(t));
    let mut s = vec![st];
    let body = user(state);
    s.extend_from_slice(&body[..body.len().min(cfg.max_state - 1)]);
    let mut branches = Vec::new();
    for q in qs {
        let mut ids = vec![q_id];
        ids.extend(user(&q.instructions));
        let mut opts = Vec::new();
        for o in &q.options {
            ids.push(o_id);
            ids.extend(user(o));
            ids.push(c_id);
            opts.push(ids.len() - 1);
        }
        ids.push(d_id);
        if ids.len() > cfg.max_branch.saturating_sub(s.len()) {
            return Err(format!("branch too long: {}", ids.len()));
        }
        branches.push(Branch { decide: ids.len() - 1, ids, opts });
    }
    Ok(Encoded { state: s, branches })
}

// ------------------------------------------------------------------ kernels

fn rms_norm(x: &[f32], d: usize, w: &[f32], eps: f32, out: &mut [f32]) {
    for (row, o) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let mut s = F4::zero();
        for c in (0..d).step_by(4) {
            let v = unsafe { F4::load(row.as_ptr().add(c)) };
            s = s.fma(v, v);
        }
        let inv = 1.0 / (s.hsum() / d as f32 + eps).sqrt();
        for c in 0..d {
            o[c] = row[c] * inv * w[c];
        }
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

// ------------------------------------------------------------------ weights

struct LinLayer {
    in_norm: TensorInfo,
    post_norm: TensorInfo,
    qkvz: TensorInfo,
    a: TensorInfo,
    b: TensorInfo,
    conv: TensorInfo,
    dt_bias: TensorInfo,
    neg_a: TensorInfo,
    gnorm: TensorInfo,
    out: TensorInfo,
    gate_up: TensorInfo,
    down: TensorInfo,
}

struct AttnLayer {
    in_norm: TensorInfo,
    post_norm: TensorInfo,
    qkv: TensorInfo,
    q_norm: TensorInfo,
    k_norm: TensorInfo,
    o: TensorInfo,
    gate_up: TensorInfo,
    down: TensorInfo,
}

enum Layer {
    Lin(LinLayer),
    Attn(AttnLayer),
}

/// What a state leaves behind for its question branches (and for later requests that repeat or
/// extend it), per layer: Kev's KV cache.
pub enum Carry {
    /// DeltaNet: recurrent state `[heads][k][v]` and the last `conv - 1` pre-conv inputs.
    Lin { state: Vec<f32>, tail: Vec<f32> },
    /// Attention: keys (rotated) and values of every state token, `[len][kv_heads * head_dim]`.
    Attn { k: Vec<f32>, v: Vec<f32> },
}

struct Seg {
    start: usize,
    len: usize,
    pos0: usize,
    parent: Option<usize>,
}

pub struct KevModel {
    pub cfg: KevConfig,
    store: Arc<Store>,
    emb: TensorInfo,
    norm: TensorInfo,
    ptr_q: TensorInfo,
    ptr_qb: TensorInfo,
    ptr_k: TensorInfo,
    ptr_kb: TensorInfo,
    layers: Vec<Layer>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    panel: Vec<f32>,
    /// Carries of recent states, most recent last: a repeated state skips its pass, and a state
    /// that extends a cached one only runs its new tokens.
    cache: Vec<(Vec<u32>, Arc<Vec<Carry>>)>,
    pub cache_states: usize,
    pub stats: CacheStats,
}

/// How often requests reused a cached state.
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheStats {
    pub hits: usize,
    pub extensions: usize,
    pub misses: usize,
    /// state tokens that did not have to run
    pub tokens_saved: usize,
}

/// States shorter than this are cheaper to recompute than to cache.
pub const CACHE_MIN_TOKENS: usize = 32;
/// A cached state reused as the prefix of a longer one must share at least this many tokens.
pub const EXTEND_MIN_TOKENS: usize = 16;

impl KevModel {
    pub fn new(cfg: KevConfig, store: Arc<Store>) -> Result<KevModel, String> {
        let t = |n: String| store.info(&n).cloned();
        let mut layers = Vec::new();
        // a coordinator sub-pack has no layers: they run on the GPU
        let trunk = if store.has("L.0.in_norm") { cfg.layers } else { 0 };
        for i in 0..trunk {
            let n = |s: &str| format!("L.{i}.{s}");
            layers.push(if cfg.full[i] {
                Layer::Attn(AttnLayer {
                    in_norm: t(n("in_norm"))?,
                    post_norm: t(n("post_norm"))?,
                    qkv: t(n("qkv"))?,
                    q_norm: t(n("q_norm"))?,
                    k_norm: t(n("k_norm"))?,
                    o: t(n("o"))?,
                    gate_up: t(n("gate_up"))?,
                    down: t(n("down"))?,
                })
            } else {
                Layer::Lin(LinLayer {
                    in_norm: t(n("in_norm"))?,
                    post_norm: t(n("post_norm"))?,
                    qkvz: t(n("qkvz"))?,
                    a: t(n("a"))?,
                    b: t(n("b"))?,
                    conv: t(n("conv"))?,
                    dt_bias: t(n("dt_bias"))?,
                    neg_a: t(n("neg_a"))?,
                    gnorm: t(n("gnorm"))?,
                    out: t(n("out"))?,
                    gate_up: t(n("gate_up"))?,
                    down: t(n("down"))?,
                })
            });
        }
        let half = cfg.rotary / 2;
        let max_pos = cfg.max_state + cfg.max_branch;
        let inv: Vec<f32> = (0..half).map(|i| 1.0 / cfg.rope_theta.powf((2 * i) as f32 / cfg.rotary as f32)).collect();
        let mut cos = vec![0.0; max_pos * half];
        let mut sin = vec![0.0; max_pos * half];
        for p in 0..max_pos {
            for i in 0..half {
                let a = (p as f32 * inv[i]) as f64;
                cos[p * half + i] = a.cos() as f32;
                sin[p * half + i] = a.sin() as f32;
            }
        }
        Ok(KevModel {
            emb: t("emb".into())?,
            norm: t("norm".into())?,
            ptr_q: t("ptr.q".into())?,
            ptr_qb: t("ptr.q.b".into())?,
            ptr_k: t("ptr.k".into())?,
            ptr_kb: t("ptr.k.b".into())?,
            layers,
            cos,
            sin,
            panel: Vec::new(),
            cache: Vec::new(),
            cache_states: 4,
            stats: CacheStats::default(),
            cfg,
            store,
        })
    }

    pub fn embed(&self, ids: &[u32]) -> Vec<f32> {
        let d = self.cfg.hidden;
        let mut x = vec![0.0; ids.len() * d];
        let vocab = self.emb.rows();
        for (i, &id) in ids.iter().enumerate() {
            self.store.row(&self.emb, (id as usize).min(vocab - 1), &mut x[i * d..(i + 1) * d]);
        }
        x
    }

    fn rope(&self, v: &mut [f32], pos: usize) {
        let h = self.cfg.rotary / 2;
        let (c, s) = (&self.cos[pos * h..(pos + 1) * h], &self.sin[pos * h..(pos + 1) * h]);
        for i in 0..h {
            let (x1, x2) = (v[i], v[i + h]);
            v[i] = x1 * c[i] - x2 * s[i];
            v[i + h] = x2 * c[i] + x1 * s[i];
        }
    }

    /// Runs every layer over `x` (all segments packed). Prefix segments (`parent == None`) fill
    /// `carry`; branch segments start from their parent's carry.
    /// Runs every layer over `x` (all segments packed). A segment with `parent: Some(p)` continues
    /// from `parents[p]`; with `fill`, each segment's carry (its parent's plus its own tokens) is
    /// written to `fill[segment]`.
    fn run(
        &mut self,
        x: &mut [f32],
        segs: &[Seg],
        parents: &[Arc<Vec<Carry>>],
        mut fill: Option<&mut Vec<Vec<Carry>>>,
    ) {
        let cfg = self.cfg.clone();
        let d = cfg.hidden;
        let t = x.len() / d;
        let st = self.store.clone();
        let mut h = vec![0.0; t * d];
        let mut delta_out = vec![0.0; t * d];
        let mut panel = std::mem::take(&mut self.panel);
        for (li, layer) in self.layers.iter().enumerate() {
            match layer {
                Layer::Lin(l) => {
                    rms_norm(x, d, st.f32s(&l.in_norm), cfg.eps, &mut h);
                    let (nh, dk, dv) = (cfg.lin_heads, cfg.lin_k, cfg.lin_v);
                    let (kd, vd) = (nh * dk, nh * dv);
                    let cd = cfg.lin_dim();
                    let wz = st.mat(&l.qkvz);
                    let pw = wz.n();
                    let mut proj = vec![0.0; t * pw];
                    linear(&h, t, wz, None, &mut proj, &mut panel);
                    let mut ab = vec![0.0; t * nh * 2];
                    {
                        let mut a = vec![0.0; t * nh];
                        let mut b = vec![0.0; t * nh];
                        linear(&h, t, st.mat(&l.a), None, &mut a, &mut panel);
                        linear(&h, t, st.mat(&l.b), None, &mut b, &mut panel);
                        let (dt, na) = (st.f32s(&l.dt_bias), st.f32s(&l.neg_a));
                        for r in 0..t {
                            for hh in 0..nh {
                                // g = -exp(A_log) * softplus(a + dt_bias); the decay is exp(g)
                                ab[r * 2 * nh + hh] = (na[hh] * softplus(a[r * nh + hh] + dt[hh])).exp();
                                ab[r * 2 * nh + nh + hh] = sigmoid(b[r * nh + hh]);
                            }
                        }
                    }
                    let conv = st.f32s(&l.conv);
                    let kw = cfg.conv;
                    let mut core = vec![0.0; t * vd];
                    let mut qkv = vec![0.0; cd];
                    for (si, seg) in segs.iter().enumerate() {
                        // causal depthwise conv, continuing from the parent's tail
                        let tail: Vec<f32> = match seg.parent {
                            Some(p) => match &parents[p][li] {
                                Carry::Lin { tail, .. } => tail.clone(),
                                _ => unreachable!(),
                            },
                            None => vec![0.0; (kw - 1) * cd],
                        };
                        let mut state = match seg.parent {
                            Some(p) => match &parents[p][li] {
                                Carry::Lin { state, .. } => state.clone(),
                                _ => unreachable!(),
                            },
                            None => vec![0.0; nh * dk * dv],
                        };
                        let input = |r: usize, c: usize| -> f32 {
                            // r counts from the segment start; negative rows come from the tail
                            proj[(seg.start + r) * pw + c]
                        };
                        let mut kv = vec![0.0; dv];
                        let mut delta = vec![0.0; dv];
                        for r in 0..seg.len {
                            for c in 0..cd {
                                let mut s = 0.0;
                                for i in 0..kw {
                                    let back = kw - 1 - i;
                                    let v = if r >= back {
                                        input(r - back, c)
                                    } else {
                                        tail[(kw - 1 - (back - r)) * cd + c]
                                    };
                                    s += conv[c * kw + i] * v;
                                }
                                qkv[c] = silu(s);
                            }
                            let row = seg.start + r;
                            for hh in 0..nh {
                                let (qs, rest) = qkv.split_at_mut(kd);
                                let (ks, vs) = rest.split_at_mut(kd);
                                let q = &mut qs[hh * dk..(hh + 1) * dk];
                                let k = &mut ks[hh * dk..(hh + 1) * dk];
                                let v = &vs[hh * dv..(hh + 1) * dv];
                                // l2-normalized q and k, q also scaled by 1/sqrt(dk)
                                let nq =
                                    1.0 / (q.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt() / (dk as f32).sqrt();
                                let nk = 1.0 / (k.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                                q.iter_mut().for_each(|x| *x *= nq);
                                k.iter_mut().for_each(|x| *x *= nk);
                                let decay = ab[row * 2 * nh + hh];
                                let beta = ab[row * 2 * nh + nh + hh];
                                let s = &mut state[hh * dk * dv..(hh + 1) * dk * dv];
                                // kv = (decay * S)^T k
                                kv.iter_mut().for_each(|x| *x = 0.0);
                                for i in 0..dk {
                                    axpy16(&mut kv, k[i], &s[i * dv..(i + 1) * dv]);
                                }
                                for j in 0..dv {
                                    delta[j] = (v[j] - decay * kv[j]) * beta;
                                }
                                // S = decay * S + k delta^T, and o = S^T q in the same sweep
                                let o = &mut core[row * vd + hh * dv..row * vd + (hh + 1) * dv];
                                o.iter_mut().for_each(|x| *x = 0.0);
                                let dvec = F4::splat(decay);
                                for i in 0..dk {
                                    let srow = &mut s[i * dv..(i + 1) * dv];
                                    let ki = F4::splat(k[i]);
                                    let qi = F4::splat(q[i]);
                                    for jj in (0..dv).step_by(4) {
                                        unsafe {
                                            let sv = F4::load(srow.as_ptr().add(jj))
                                                .mul(dvec)
                                                .fma(ki, F4::load(delta.as_ptr().add(jj)));
                                            sv.store(srow.as_mut_ptr().add(jj));
                                            F4::load(o.as_ptr().add(jj)).fma(qi, sv).store(o.as_mut_ptr().add(jj));
                                        }
                                    }
                                }
                            }
                        }
                        if let Some(f) = fill.as_deref_mut() {
                            let mut nt = vec![0.0; (kw - 1) * cd];
                            for i in 0..kw - 1 {
                                let back = kw - 1 - i;
                                for c in 0..cd {
                                    nt[i * cd + c] = if seg.len >= back {
                                        input(seg.len - back, c)
                                    } else {
                                        tail[(kw - 1 - (back - seg.len)) * cd + c]
                                    };
                                }
                            }
                            f[si][li] = Carry::Lin { state, tail: nt };
                        }
                    }
                    // gated RMS norm per head, then the output projection
                    let gw = st.f32s(&l.gnorm);
                    for r in 0..t {
                        for hh in 0..nh {
                            let o = &mut core[r * vd + hh * dv..r * vd + (hh + 1) * dv];
                            let ms = o.iter().map(|v| v * v).sum::<f32>() / dv as f32;
                            let inv = 1.0 / (ms + cfg.eps).sqrt();
                            for j in 0..dv {
                                let z = proj[r * pw + cd + hh * dv + j];
                                o[j] = o[j] * inv * gw[j] * silu(z);
                            }
                        }
                    }
                    linear(&core, t, st.mat(&l.out), None, &mut delta_out, &mut panel);
                }
                Layer::Attn(l) => {
                    rms_norm(x, d, st.f32s(&l.in_norm), cfg.eps, &mut h);
                    let (nq, nkv, hd) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
                    let w = st.mat(&l.qkv);
                    let pw = w.n();
                    let mut proj = vec![0.0; t * pw];
                    linear(&h, t, w, None, &mut proj, &mut panel);
                    let (qn, kn) = (st.f32s(&l.q_norm), st.f32s(&l.k_norm));
                    let qoff = 0;
                    let koff = nq * 2 * hd;
                    let voff = koff + nkv * hd;
                    let norm_head = |v: &mut [f32], w: &[f32]| {
                        let ms = v.iter().map(|x| x * x).sum::<f32>() / hd as f32;
                        let inv = 1.0 / (ms + cfg.eps).sqrt();
                        v.iter_mut().zip(w).for_each(|(x, w)| *x = *x * inv * w);
                    };
                    for seg in segs {
                        for r in 0..seg.len {
                            let row = &mut proj[(seg.start + r) * pw..(seg.start + r + 1) * pw];
                            for hh in 0..nq {
                                let q = &mut row[qoff + hh * 2 * hd..qoff + hh * 2 * hd + hd];
                                norm_head(q, qn);
                                self.rope(q, seg.pos0 + r);
                            }
                            for hh in 0..nkv {
                                let k = &mut row[koff + hh * hd..koff + (hh + 1) * hd];
                                norm_head(k, kn);
                                self.rope(k, seg.pos0 + r);
                            }
                        }
                    }
                    let mut att = vec![0.0; t * nq * hd];
                    let scale = 1.0 / (hd as f32).sqrt();
                    let kvw = nkv * hd;
                    for (si, seg) in segs.iter().enumerate() {
                        let (pk, pv): (&[f32], &[f32]) = match seg.parent {
                            Some(p) => match &parents[p][li] {
                                Carry::Attn { k, v } => (k, v),
                                _ => unreachable!(),
                            },
                            None => (&[], &[]),
                        };
                        let plen = pk.len() / kvw;
                        let mut scores = vec![0.0; plen + seg.len];
                        let key = |j: usize, g: usize| -> &[f32] {
                            if j < plen {
                                &pk[j * kvw + g * hd..j * kvw + (g + 1) * hd]
                            } else {
                                let r = seg.start + j - plen;
                                &proj[r * pw + koff + g * hd..r * pw + koff + (g + 1) * hd]
                            }
                        };
                        let val = |j: usize, g: usize| -> &[f32] {
                            if j < plen {
                                &pv[j * kvw + g * hd..j * kvw + (g + 1) * hd]
                            } else {
                                let r = seg.start + j - plen;
                                &proj[r * pw + voff + g * hd..r * pw + voff + (g + 1) * hd]
                            }
                        };
                        for r in 0..seg.len {
                            let n = plen + r + 1;
                            for hh in 0..nq {
                                let g = hh / (nq / nkv);
                                let q = &proj[(seg.start + r) * pw + qoff + hh * 2 * hd
                                    ..(seg.start + r) * pw + qoff + hh * 2 * hd + hd];
                                let mut m = f32::NEG_INFINITY;
                                for j in 0..n {
                                    let s = dot16(q, key(j, g)) * scale;
                                    scores[j] = s;
                                    m = m.max(s);
                                }
                                let mut z = 0.0;
                                for s in scores[..n].iter_mut() {
                                    *s = (*s - m).exp();
                                    z += *s;
                                }
                                let o = &mut att
                                    [(seg.start + r) * nq * hd + hh * hd..(seg.start + r) * nq * hd + (hh + 1) * hd];
                                o.iter_mut().for_each(|v| *v = 0.0);
                                for j in 0..n {
                                    axpy16(o, scores[j] / z, val(j, g));
                                }
                                let gate = &proj[(seg.start + r) * pw + qoff + hh * 2 * hd + hd
                                    ..(seg.start + r) * pw + qoff + (hh + 1) * 2 * hd];
                                o.iter_mut().zip(gate).for_each(|(v, g)| *v *= sigmoid(*g));
                            }
                        }
                        if let Some(f) = fill.as_deref_mut() {
                            // the parent's keys and values, then this segment's
                            let mut k = pk.to_vec();
                            let mut v = pv.to_vec();
                            for r in 0..seg.len {
                                let row = &proj[(seg.start + r) * pw..(seg.start + r + 1) * pw];
                                k.extend_from_slice(&row[koff..koff + kvw]);
                                v.extend_from_slice(&row[voff..voff + kvw]);
                            }
                            f[si][li] = Carry::Attn { k, v };
                        }
                    }
                    linear(&att, t, st.mat(&l.o), None, &mut delta_out, &mut panel);
                }
            }
            for (a, b) in x.iter_mut().zip(&delta_out) {
                *a += b;
            }
            // SwiGLU MLP
            let (norm_w, gate_up, down) = match layer {
                Layer::Lin(l) => (&l.post_norm, &l.gate_up, &l.down),
                Layer::Attn(l) => (&l.post_norm, &l.gate_up, &l.down),
            };
            rms_norm(x, d, st.f32s(norm_w), cfg.eps, &mut h);
            let gu = st.mat(gate_up);
            let inter = gu.n() / 2;
            let mut u = vec![0.0; t * 2 * inter];
            linear(&h, t, gu, None, &mut u, &mut panel);
            let mut a = vec![0.0; t * inter];
            for r in 0..t {
                for i in 0..inter {
                    a[r * inter + i] = silu(u[r * 2 * inter + i]) * u[r * 2 * inter + inter + i];
                }
            }
            linear(&a, t, st.mat(down), None, &mut delta_out, &mut panel);
            for (a, b) in x.iter_mut().zip(&delta_out) {
                *a += b;
            }
        }
        self.panel = panel;
    }

    /// Pointer logits (temperature applied) for every branch of every request.
    pub fn forward(&mut self, reqs: &[Encoded]) -> Vec<Vec<Vec<f32>>> {
        let d = self.cfg.hidden;
        // stage 1: every state that is not cached; a state extending a cached one runs only
        // its new tokens, continuing that carry
        let mut carries: Vec<Option<Arc<Vec<Carry>>>> = vec![None; reqs.len()];
        let mut ids = Vec::new();
        let mut segs = Vec::new();
        let mut parents: Vec<Arc<Vec<Carry>>> = Vec::new();
        let mut owner = Vec::new();
        for (ri, e) in reqs.iter().enumerate() {
            if e.state.is_empty() {
                continue;
            }
            if let Some(i) = self.cache.iter().position(|(k, _)| *k == e.state) {
                let hit = self.cache.remove(i);
                carries[ri] = Some(hit.1.clone());
                self.cache.push(hit);
                self.stats.hits += 1;
                self.stats.tokens_saved += e.state.len();
                continue;
            }
            let prefix = self
                .cache
                .iter()
                .enumerate()
                .filter(|(_, (k, _))| k.len() >= EXTEND_MIN_TOKENS && k.len() < e.state.len() && e.state.starts_with(k))
                .max_by_key(|(_, (k, _))| k.len())
                .map(|(i, _)| i);
            let (from, parent) = match prefix {
                Some(i) => {
                    let (k, c) = &self.cache[i];
                    parents.push(c.clone());
                    self.stats.extensions += 1;
                    self.stats.tokens_saved += k.len();
                    (k.len(), Some(parents.len() - 1))
                }
                None => {
                    self.stats.misses += 1;
                    (0, None)
                }
            };
            segs.push(Seg { start: ids.len(), len: e.state.len() - from, pos0: from, parent });
            ids.extend_from_slice(&e.state[from..]);
            owner.push(ri);
        }
        if !segs.is_empty() {
            let mut fresh: Vec<Vec<Carry>> = segs
                .iter()
                .map(|_| (0..self.cfg.layers).map(|_| Carry::Lin { state: Vec::new(), tail: Vec::new() }).collect())
                .collect();
            let mut x = self.embed(&ids);
            self.run(&mut x, &segs, &parents, Some(&mut fresh));
            for (c, ri) in fresh.into_iter().zip(owner) {
                let c = Arc::new(c);
                let state = &reqs[ri].state;
                if state.len() >= CACHE_MIN_TOKENS && self.cache_states > 0 {
                    if self.cache.len() >= self.cache_states {
                        self.cache.remove(0);
                    }
                    self.cache.push((state.clone(), c.clone()));
                }
                carries[ri] = Some(c);
            }
        }
        // stage 2: every question branch, continuing from its state
        let mut ids = Vec::new();
        let mut segs = Vec::new();
        let mut parents = Vec::new();
        for (ri, e) in reqs.iter().enumerate() {
            let parent = carries[ri].as_ref().map(|c| {
                parents.push(c.clone());
                parents.len() - 1
            });
            for b in &e.branches {
                segs.push(Seg { start: ids.len(), len: b.ids.len(), pos0: e.state.len(), parent });
                ids.extend_from_slice(&b.ids);
            }
        }
        let mut x = self.embed(&ids);
        self.run(&mut x, &segs, &parents, None);
        let mut rows = Vec::new();
        for (seg, b) in segs.iter().zip(reqs.iter().flat_map(|e| e.branches.iter())) {
            for r in std::iter::once(b.decide).chain(b.opts.iter().copied()) {
                rows.extend_from_slice(&x[(seg.start + r) * d..(seg.start + r + 1) * d]);
            }
        }
        self.readout(&rows, reqs)
    }

    /// Rows whose last-layer states the pointer head reads, in order: for every branch of every
    /// request, `<decide>` then each `</opt>`.
    pub fn readout_rows(reqs: &[Encoded]) -> Vec<usize> {
        let mut rows = Vec::new();
        let mut start = 0;
        for e in reqs {
            for b in &e.branches {
                rows.extend(std::iter::once(b.decide).chain(b.opts.iter().copied()).map(|r| start + r));
                start += b.ids.len();
            }
        }
        rows
    }

    /// Final norm and the pointer head over `rows` (see `readout_rows`); logits with the
    /// temperature applied, per request and question.
    pub fn readout(&mut self, rows: &[f32], reqs: &[Encoded]) -> Vec<Vec<Vec<f32>>> {
        let d = self.cfg.hidden;
        let st = self.store.clone();
        let dp = self.cfg.ptr_dim;
        let scale = 1.0 / (dp as f32).sqrt() / self.cfg.temperature;
        let mut out = Vec::new();
        let mut at = 0;
        for e in reqs {
            let mut qs = Vec::new();
            for b in &e.branches {
                let n = 1 + b.opts.len();
                let mut hn = vec![0.0; n * d];
                rms_norm(&rows[at * d..(at + n) * d], d, st.f32s(&self.norm), self.cfg.eps, &mut hn);
                at += n;
                let mut qv = vec![0.0; dp];
                linear(&hn[..d], 1, st.mat(&self.ptr_q), Some(st.f32s(&self.ptr_qb)), &mut qv, &mut self.panel);
                let k = b.opts.len();
                let mut kv = vec![0.0; k * dp];
                linear(&hn[d..], k, st.mat(&self.ptr_k), Some(st.f32s(&self.ptr_kb)), &mut kv, &mut self.panel);
                qs.push(
                    (0..k)
                        .map(|j| kv[j * dp..(j + 1) * dp].iter().zip(&qv).map(|(a, b)| a * b).sum::<f32>() * scale)
                        .collect(),
                );
            }
            out.push(qs);
        }
        out
    }
}

// ------------------------------------------------------------------ engine

pub struct KevEngine {
    pub model: KevModel,
    pub tok: Tokenizer,
    pub info: Value,
    pub modalities: Vec<crate::content::Modality>,
}

fn r2(x: f64) -> Value {
    Value::Float(format!("{x:.2}").parse().unwrap_or(x))
}

fn softmax(z: &[f32]) -> Vec<f32> {
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

impl KevEngine {
    pub fn load(buf: AlignedBuf) -> Result<KevEngine, String> {
        let h = pack::parse_header(buf.as_slice())?;
        if buf.len() < h.total_size {
            return Err(format!("pack is truncated: {} of {} bytes", buf.len(), h.total_size));
        }
        let cfg = KevConfig::from_json(h.config())?;
        let tok = Tokenizer::from_bytes(&buf.as_slice()[h.tokenizer_offset..h.tokenizer_offset + h.tokenizer_size])?;
        let info = h.json.get("model").cloned().unwrap_or(Value::Null);
        let modalities = crate::content::Modality::from_config(h.config());
        let store = Arc::new(Store::new(buf, h.tensors.clone())?);
        Ok(KevEngine { model: KevModel::new(cfg, store)?, tok, info, modalities })
    }

    pub fn prepare(&self, state: &Value, questions: &Value) -> Result<(Encoded, Vec<KevQuestion>), String> {
        let qs = parse_questions(questions)?;
        let mut e = encode(&self.tok, &self.model.cfg, &render(state, 0), &qs)?;
        // one question about a short state: its row is state + branch, so run it as one segment
        // and skip the state pass (the same causal row, half the dispatches). Longer states keep
        // their own pass so the next request about them can reuse its cache.
        if e.branches.len() == 1 && e.state.len() < CACHE_MIN_TOKENS {
            let n = e.state.len();
            let b = &mut e.branches[0];
            let mut ids = std::mem::take(&mut e.state);
            ids.extend_from_slice(&b.ids);
            b.ids = ids;
            b.decide += n;
            b.opts.iter_mut().for_each(|o| *o += n);
        }
        Ok((e, qs))
    }

    /// `POST /v1/systemone` for each request, answered in one pass. Answers match kev.api to_answers.
    pub fn decide(&mut self, requests: &[(Value, Value)]) -> Result<Vec<Value>, String> {
        let mut enc = Vec::new();
        let mut questions = Vec::new();
        for (s, q) in requests {
            let (e, qs) = self.prepare(s, q)?;
            enc.push(e);
            questions.push(qs);
        }
        let logits = self.model.forward(&enc);
        Ok(self.respond(&enc, &questions, &logits))
    }

    /// Tokenizes every request; the batch an external trunk (the GPU) runs.
    pub fn prepare_all(&self, requests: &[(Value, Value)]) -> Result<(Vec<Encoded>, Vec<Vec<KevQuestion>>), String> {
        let mut enc = Vec::new();
        let mut questions = Vec::new();
        for (s, q) in requests {
            let (e, qs) = self.prepare(s, q)?;
            enc.push(e);
            questions.push(qs);
        }
        Ok((enc, questions))
    }

    /// Kev-shaped responses from pointer logits.
    pub fn respond(&self, enc: &[Encoded], questions: &[Vec<KevQuestion>], logits: &[Vec<Vec<f32>>]) -> Vec<Value> {
        let mut out = Vec::new();
        for ((e, qs), lg) in enc.iter().zip(questions).zip(logits) {
            let mut answers = Vec::new();
            let mut raw = Vec::new();
            for (q, z) in qs.iter().zip(lg) {
                let p = softmax(z);
                raw.push(Value::Array(p.iter().map(|&v| Value::Float(v as f64)).collect()));
                let pd: Vec<f64> = p.iter().map(|&v| v as f64).collect();
                let best = pd.iter().enumerate().fold(0, |b, (i, &v)| if v > pd[b] { i } else { b });
                let k = pd.len();
                let a = match q.kind {
                    Kind::Noul => {
                        Value::Object(vec![("type".into(), Value::Str("noul".into())), ("noul".into(), r2(pd[1]))])
                    }
                    Kind::Choice => {
                        let conf = if k == 1 { 1.0 } else { (pd[best] - 1.0 / k as f64) / (1.0 - 1.0 / k as f64) };
                        Value::Object(vec![
                            ("type".into(), Value::Str("choice".into())),
                            ("choice".into(), Value::Str(q.keys[best].clone())),
                            ("confidence".into(), r2(conf)),
                            (
                                "probabilities".into(),
                                Value::Object(q.keys.iter().zip(&pd).map(|(k, &v)| (k.clone(), r2(v))).collect()),
                            ),
                        ])
                    }
                    Kind::Score => {
                        let score: f64 = pd.iter().enumerate().map(|(i, v)| i as f64 * v).sum();
                        let spread: f64 = pd.iter().enumerate().map(|(i, v)| v * (i as f64 - best as f64).abs()).sum();
                        Value::Object(vec![
                            ("type".into(), Value::Str("score".into())),
                            ("score".into(), r2(score)),
                            (
                                "legend".into(),
                                Value::Object(
                                    q.keys
                                        .iter()
                                        .zip(&q.options)
                                        .map(|(k, o)| (k.clone(), Value::Str(o.clone())))
                                        .collect(),
                                ),
                            ),
                            (
                                "probabilities".into(),
                                Value::Object(q.keys.iter().zip(&pd).map(|(k, &v)| (k.clone(), r2(v))).collect()),
                            ),
                            ("confidence".into(), r2(1.0 - spread / (k as f64 - 1.0))),
                        ])
                    }
                };
                answers.push((q.id.clone(), a));
            }
            let answers = Value::Object(answers);
            let output_tokens = self.tok.encode(&answers.py_dumps(true)).len();
            out.push(Value::Object(vec![
                ("model".into(), Value::Str("kev-latest".into())),
                ("answers".into(), answers),
                (
                    "usage".into(),
                    Value::Object(vec![
                        ("input_tokens".into(), Value::Int(e.tokens().to_string())),
                        ("output_tokens".into(), Value::Int(output_tokens.to_string())),
                    ]),
                ),
                // full-precision probabilities per question, next to the rounded reference answers
                ("raw_probabilities".into(), Value::Object(qs.iter().map(|q| q.id.clone()).zip(raw).collect())),
            ]));
        }
        out
    }
}

impl crate::runtime::Model for KevEngine {
    fn arch(&self) -> &'static str {
        "kev"
    }
    fn info(&self) -> &Value {
        &self.info
    }
    fn modalities(&self) -> &[crate::content::Modality] {
        &self.modalities
    }
    fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }
    fn decide(&mut self, requests: &[crate::content::Request]) -> Result<Vec<Value>, String> {
        let rs = crate::runtime::text_requests("kev", &self.modalities, requests, |v| render(v, 0))?;
        KevEngine::decide(self, &rs)
    }
    fn as_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
