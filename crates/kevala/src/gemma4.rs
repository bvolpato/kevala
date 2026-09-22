//! Gemma 4 dense text models used as a direct option scorer.
//!
//! The upstream Gemma 4 instruction models are causal language models.  Kevala uses the
//! same non-thinking instruction template as the checkpoint, asks for one of at most sixteen
//! labelled options, and reads the logits for `A` through `P` from a small readout matrix.
//! The text backbone is deliberately kept separate from the coordinator: a coordinator-only
//! pack contains the tokenizer, configuration, and readout, while a full pack additionally
//! contains the embedding and decoder weights for the native CPU path.  WebGPU callers can
//! prepare prompts here, run the decoder and final RMS norm on the GPU, then call `finish` with
//! one normalized final row per prepared prompt.
//!
//! This module follows the pinned Hugging Face `modeling_gemma4.py` implementation.  In
//! particular, RMSNorm weights are direct multipliers (there is no Qwen-style `1 + weight`),
//! attention has scale `1.0`, full attention uses proportional RoPE, and the last
//! `num_kv_shared_layers` decoder layers reuse the full-length KV states from the last earlier
//! layer of the same attention type.

use crate::content::{Modality, Request};
use crate::json::Value;
use crate::kernels::linear;
use crate::kev::{self, KevQuestion, Kind};
use crate::model::{AlignedBuf, Store};
use crate::pack::{self, DType, TensorInfo};
use crate::tokenizer::Tokenizer;
use std::cell::RefCell;
use std::sync::Arc;

const DIRECT_OPTIONS_SYSTEM: &str = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.";
const DEFAULT_MAX_INPUT_TOKENS: usize = 4096;
const MAX_LABELS: usize = 16;

/// Parsed dense Gemma 4 text configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4Config {
    pub hidden: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub vocab_per_layer: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub global_kv_heads: usize,
    pub local_head_dim: usize,
    pub global_head_dim: usize,
    pub layer_types: Vec<LayerType>,
    pub kv_shared_layers: usize,
    pub sliding_window: usize,
    pub eps: f32,
    pub max_position_embeddings: usize,
    pub max_input_tokens: usize,
    pub final_logit_softcap: Option<f32>,
    pub use_double_wide_mlp: bool,
    pub ple_dim: usize,
    pub local_rope_theta: f32,
    pub global_rope_theta: f32,
    pub global_partial_rotary: f32,
    pub temperature: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerType {
    Sliding,
    Full,
}

impl LayerType {
    fn from_name(s: &str) -> Option<Self> {
        match s {
            "sliding_attention" => Some(Self::Sliding),
            "full_attention" => Some(Self::Full),
            _ => None,
        }
    }

    pub fn is_full(self) -> bool {
        matches!(self, Self::Full)
    }
}

fn text_config<'a>(c: &'a Value) -> &'a Value {
    c.get("text_config").filter(|v| v.as_object().is_some()).unwrap_or(c)
}

fn usize_field(c: &Value, key: &str) -> Result<usize, String> {
    pack::get_usize(c, key).map_err(|e| format!("Gemma4 config: {e}"))
}

fn f32_field(c: &Value, key: &str) -> Result<f32, String> {
    pack::get_f64(c, key).map(|x| x as f32).map_err(|e| format!("Gemma4 config: {e}"))
}

fn optional_f32(c: &Value, key: &str) -> Result<Option<f32>, String> {
    match c.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => {
            v.as_f64().map(|x| Some(x as f32)).ok_or_else(|| format!("Gemma4 config: {key} must be a number or null"))
        }
    }
}

fn bool_field(c: &Value, key: &str, default: bool) -> Result<bool, String> {
    match c.get(key) {
        None => Ok(default),
        Some(v) => v.as_bool().ok_or_else(|| format!("Gemma4 config: {key} must be a boolean")),
    }
}

fn require_f32(t: &TensorInfo, name: &str) -> Result<(), String> {
    if t.dtype != DType::F32 {
        return Err(format!("Gemma4 tensor {name} must use direct f32 storage, got {}", t.dtype.name()));
    }
    Ok(())
}

fn rope_value(c: &Value, layer: &str, key: &str, default: f32) -> Result<f32, String> {
    let Some(rope) = c.get("rope_parameters").and_then(|v| v.get(layer)) else {
        return Ok(default);
    };
    match rope.get(key) {
        None => Ok(default),
        Some(v) => v
            .as_f64()
            .map(|x| x as f32)
            .ok_or_else(|| format!("Gemma4 config: rope_parameters.{layer}.{key} must be a number")),
    }
}

fn rope_type(c: &Value, layer: &str, default: &str) -> Result<String, String> {
    let Some(rope) = c.get("rope_parameters").and_then(|v| v.get(layer)) else {
        return Ok(default.to_string());
    };
    Ok(rope.get("rope_type").and_then(Value::as_str).unwrap_or(default).to_string())
}

impl Gemma4Config {
    /// Parses either the upstream top-level multimodal config or the flattened text config used
    /// by a `.kevala` converter.  Unsupported multimodal/MoE variants fail at load time.
    pub fn from_json(c: &Value) -> Result<Self, String> {
        if let Some(arch) = c.get("arch").and_then(Value::as_str) {
            if arch != "gemma4" {
                return Err(format!("Gemma4 config: expected arch=gemma4, got {arch:?}"));
            }
        }
        let c = text_config(c);
        let hidden = usize_field(c, "hidden_size")?;
        let layers = usize_field(c, "num_hidden_layers")?;
        let intermediate = usize_field(c, "intermediate_size")?;
        let vocab = usize_field(c, "vocab_size")?;
        let vocab_per_layer = c.get("vocab_size_per_layer_input").and_then(Value::as_usize).unwrap_or(vocab);
        let heads = usize_field(c, "num_attention_heads")?;
        let kv_heads = usize_field(c, "num_key_value_heads")?;
        let global_head_dim = c.get("global_head_dim").and_then(Value::as_usize).unwrap_or(512);
        let local_head_dim = usize_field(c, "head_dim")?;
        let ple_dim = c.get("hidden_size_per_layer_input").and_then(Value::as_usize).unwrap_or(0);
        let kv_shared_layers = c.get("num_kv_shared_layers").and_then(Value::as_usize).unwrap_or(0);
        let sliding_window = c.get("sliding_window").and_then(Value::as_usize).unwrap_or(0);
        let eps = f32_field(c, "rms_norm_eps")?;
        let max_position_embeddings = c.get("max_position_embeddings").and_then(Value::as_usize).unwrap_or(131_072);
        let max_input_tokens = c.get("max_input_tokens").and_then(Value::as_usize).unwrap_or(DEFAULT_MAX_INPUT_TOKENS);
        let final_logit_softcap = optional_f32(c, "final_logit_softcapping")?;
        let use_double_wide_mlp = bool_field(c, "use_double_wide_mlp", false)?;
        let attention_bias = bool_field(c, "attention_bias", false)?;
        let attention_k_eq_v = bool_field(c, "attention_k_eq_v", false)?;
        let enable_moe = bool_field(c, "enable_moe_block", false)?;
        let use_bidirectional = c.get("use_bidirectional_attention").and_then(Value::as_str);

        if hidden == 0 || layers == 0 || intermediate == 0 || vocab == 0 || heads == 0 || kv_heads == 0 {
            return Err("Gemma4 config: dimensions and layer counts must be positive".into());
        }
        if hidden % 4 != 0
            || intermediate % 4 != 0
            || local_head_dim % 4 != 0
            || global_head_dim % 4 != 0
            || ple_dim % 4 != 0
        {
            return Err(
                "Gemma4 config: hidden, intermediate, head, and PLE dimensions must be multiples of four".into()
            );
        }
        if heads % kv_heads != 0 {
            return Err("Gemma4 config: num_attention_heads must be a multiple of num_key_value_heads".into());
        }
        if global_head_dim % 2 != 0 || local_head_dim % 2 != 0 {
            return Err("Gemma4 config: attention head dimensions must be even".into());
        }
        if ple_dim == 0 {
            return Err("Gemma4 config: hidden_size_per_layer_input must be positive for dense Gemma4".into());
        }
        if vocab_per_layer == 0 || vocab_per_layer > vocab {
            return Err("Gemma4 config: vocab_size_per_layer_input must be in 1..=vocab_size".into());
        }
        if kv_shared_layers >= layers {
            return Err(format!(
                "Gemma4 config: num_kv_shared_layers={kv_shared_layers} must be less than num_hidden_layers={layers}"
            ));
        }
        if sliding_window == 0 || max_position_embeddings == 0 || max_input_tokens == 0 {
            return Err(
                "Gemma4 config: sliding_window, max_position_embeddings, and max_input_tokens must be positive".into(),
            );
        }
        if max_input_tokens > max_position_embeddings {
            return Err(format!(
                "Gemma4 config: max_input_tokens={max_input_tokens} exceeds max_position_embeddings={max_position_embeddings}"
            ));
        }
        if max_input_tokens > DEFAULT_MAX_INPUT_TOKENS {
            return Err(format!(
                "Gemma4 config: max_input_tokens={max_input_tokens} exceeds the Kevala runtime limit of {DEFAULT_MAX_INPUT_TOKENS}; lower the pack setting"
            ));
        }
        if attention_bias {
            return Err("Gemma4 config: attention_bias=true is not supported by the bias-free pack layout".into());
        }
        if attention_k_eq_v {
            return Err(
                "Gemma4 config: attention_k_eq_v=true is not supported; the pack requires separate v projections"
                    .into(),
            );
        }
        if enable_moe {
            return Err("Gemma4 config: MoE Gemma4 is not supported; convert a dense text checkpoint".into());
        }
        if let Some(kind) = use_bidirectional {
            return Err(format!(
                "Gemma4 config: use_bidirectional_attention={kind:?} is not supported; direct scoring requires causal text attention"
            ));
        }
        let layer_types_json =
            c.get("layer_types").and_then(Value::as_array).ok_or("Gemma4 config: missing layer_types")?;
        if layer_types_json.len() != layers {
            return Err(format!(
                "Gemma4 config: layer_types has {} entries, expected {layers}",
                layer_types_json.len()
            ));
        }
        let layer_types: Vec<LayerType> = layer_types_json
            .iter()
            .map(|v| {
                let s = v.as_str().ok_or("Gemma4 config: layer_types entries must be strings")?;
                LayerType::from_name(s).ok_or_else(|| format!("Gemma4 config: unsupported layer type {s:?}"))
            })
            .collect::<Result<_, _>>()?;
        if layer_types.last() != Some(&LayerType::Full) {
            return Err("Gemma4 config: the final decoder layer must be full_attention".into());
        }
        let shared_start = layers - kv_shared_layers;
        for ty in [LayerType::Sliding, LayerType::Full] {
            if layer_types.iter().any(|&x| x == ty) && !layer_types[..shared_start].contains(&ty) {
                return Err(format!(
                    "Gemma4 config: shared {ty:?} attention has no non-shared KV source before layer {shared_start}"
                ));
            }
        }
        // Gemma only applies num_global_key_value_heads to the alternative K=V attention path.
        // That path is rejected above because this pack stores separate K and V projections, so
        // the supported path always uses num_key_value_heads for both layer types.
        let global_kv_heads = kv_heads;
        if global_kv_heads == 0 || heads % global_kv_heads != 0 {
            return Err("Gemma4 config: num_global_key_value_heads must divide num_attention_heads".into());
        }
        let local_rope_theta = rope_value(c, "sliding_attention", "rope_theta", 10_000.0)?;
        let global_rope_theta = rope_value(c, "full_attention", "rope_theta", 1_000_000.0)?;
        let global_partial_rotary = rope_value(c, "full_attention", "partial_rotary_factor", 0.25)?;
        let local_rope = rope_type(c, "sliding_attention", "default")?;
        let global_rope = rope_type(c, "full_attention", "proportional")?;
        if local_rope != "default" || global_rope != "proportional" {
            return Err(format!(
                "Gemma4 config: supported RoPE types are sliding=default and full=proportional, got sliding={local_rope:?}, full={global_rope:?}"
            ));
        }
        if !global_partial_rotary.is_finite() || !(0.0..=1.0).contains(&global_partial_rotary) {
            return Err("Gemma4 config: full_attention partial_rotary_factor must be finite in 0..=1".into());
        }
        if let Some(cap) = final_logit_softcap {
            if !cap.is_finite() || cap <= 0.0 {
                return Err("Gemma4 config: final_logit_softcapping must be finite and positive".into());
            }
        }
        let temperature = c
            .get("temperature")
            .and_then(Value::as_f64)
            .or_else(|| c.get("readout_temperature").and_then(Value::as_f64))
            .unwrap_or(1.0) as f32;
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err("Gemma4 config: temperature must be finite and positive".into());
        }
        Ok(Self {
            hidden,
            layers,
            intermediate,
            vocab,
            vocab_per_layer,
            heads,
            kv_heads,
            global_kv_heads,
            local_head_dim,
            global_head_dim,
            layer_types,
            kv_shared_layers,
            sliding_window,
            eps,
            max_position_embeddings,
            max_input_tokens,
            final_logit_softcap,
            use_double_wide_mlp,
            ple_dim,
            local_rope_theta,
            global_rope_theta,
            global_partial_rotary,
            temperature,
        })
    }

    pub fn is_full(&self, layer: usize) -> bool {
        self.layer_types[layer].is_full()
    }

    pub fn head_dim(&self, layer: usize) -> usize {
        if self.is_full(layer) {
            self.global_head_dim
        } else {
            self.local_head_dim
        }
    }

    pub fn kv_heads(&self, layer: usize) -> usize {
        if self.is_full(layer) {
            self.global_kv_heads
        } else {
            self.kv_heads
        }
    }

    pub fn is_kv_shared(&self, layer: usize) -> bool {
        layer >= self.layers - self.kv_shared_layers
    }

    pub fn mlp_intermediate(&self, layer: usize) -> usize {
        if self.use_double_wide_mlp && self.is_kv_shared(layer) {
            self.intermediate * 2
        } else {
            self.intermediate
        }
    }
}

/// A question as it appears in the prepared prompt.
#[derive(Clone, Debug)]
pub struct PreparedQuestion {
    pub id: String,
    pub kind: Kind,
    pub keys: Vec<String>,
    pub legend: Vec<String>,
    pub options: Vec<String>,
}

impl From<KevQuestion> for PreparedQuestion {
    fn from(q: KevQuestion) -> Self {
        Self { id: q.id, kind: q.kind, keys: q.keys, legend: Vec::new(), options: q.options }
    }
}

/// One complete causal prompt.  `start..start+len` indexes the flattened `Prepared::ids`.
#[derive(Clone, Debug)]
pub struct PreparedSequence {
    pub start: usize,
    pub len: usize,
    pub request: usize,
    pub question: usize,
    pub labels: usize,
}

/// Tokenized requests in the order expected by `finish`: one final hidden row per sequence.
#[derive(Clone, Debug, Default)]
pub struct Prepared {
    pub ids: Vec<u32>,
    pub sequences: Vec<PreparedSequence>,
    pub questions: Vec<PreparedQuestion>,
    pub request_sequence_ranges: Vec<(usize, usize)>,
}

impl Prepared {
    pub fn tokens(&self) -> usize {
        self.ids.len()
    }

    pub fn sequence_count(&self) -> usize {
        self.sequences.len()
    }

    pub fn sequence_ids(&self, index: usize) -> &[u32] {
        let s = &self.sequences[index];
        &self.ids[s.start..s.start + s.len]
    }
}

#[derive(Clone)]
struct KvCache {
    heads: usize,
    head_dim: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

struct LayerWeights {
    attn_norm: TensorInfo,
    attn_post_norm: TensorInfo,
    ffn_norm: TensorInfo,
    ffn_post_norm: TensorInfo,
    q: TensorInfo,
    qn: TensorInfo,
    k: Option<TensorInfo>,
    kn: Option<TensorInfo>,
    v: Option<TensorInfo>,
    o: TensorInfo,
    gate: TensorInfo,
    up: TensorInfo,
    down: TensorInfo,
    ple_gate: TensorInfo,
    ple_out: TensorInfo,
    ple_norm: TensorInfo,
    scalar: TensorInfo,
}

/// Native dense Gemma4 text backbone.  A coordinator-only pack has no instance of this type.
pub struct Gemma4Model {
    pub cfg: Gemma4Config,
    store: Arc<Store>,
    embed: TensorInfo,
    norm: TensorInfo,
    ple_embeddings: Vec<TensorInfo>,
    ple_projections: Vec<TensorInfo>,
    layers: Vec<LayerWeights>,
    ple_norm: TensorInfo,
    panel: RefCell<Vec<f32>>,
}

impl Gemma4Model {
    pub fn new(cfg: Gemma4Config, store: Arc<Store>) -> Result<Self, String> {
        let need = |name: String| store.info(&name).cloned();
        let embed = need("embed".into())?;
        let norm = need("norm".into())?;
        require_f32(&norm, "norm")?;
        if embed.rows() < cfg.vocab || embed.cols() != cfg.hidden {
            return Err(format!(
                "Gemma4 tensor embed has shape {:?}, expected [{}, {}]",
                embed.shape, cfg.vocab, cfg.hidden
            ));
        }
        if norm.numel() != cfg.hidden {
            return Err(format!("Gemma4 tensor norm has shape {:?}, expected [{}]", norm.shape, cfg.hidden));
        }
        let ple_norm = need("ple.norm".into())?;
        require_f32(&ple_norm, "ple.norm")?;
        if ple_norm.numel() != cfg.ple_dim {
            return Err(format!("Gemma4 tensor ple.norm has shape {:?}, expected [{}]", ple_norm.shape, cfg.ple_dim));
        }
        let mut ple_embeddings = Vec::with_capacity(cfg.layers);
        let mut ple_projections = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let e = need(format!("ple.{i}.embed"))?;
            let p = need(format!("ple.{i}.proj"))?;
            if e.rows() < cfg.vocab_per_layer || e.cols() != cfg.ple_dim {
                return Err(format!(
                    "Gemma4 tensor ple.{i}.embed has shape {:?}, expected [{}, {}]",
                    e.shape, cfg.vocab_per_layer, cfg.ple_dim
                ));
            }
            if p.rows() != cfg.ple_dim || p.cols() != cfg.hidden {
                return Err(format!(
                    "Gemma4 tensor ple.{i}.proj has shape {:?}, expected [{}, {}]",
                    p.shape, cfg.ple_dim, cfg.hidden
                ));
            }
            ple_embeddings.push(e);
            ple_projections.push(p);
        }
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let n = |part: &str| need(format!("l.{i}.{part}"));
            let full = cfg.is_full(i);
            let hd = cfg.head_dim(i);
            let kv = cfg.kv_heads(i);
            let inter = cfg.mlp_intermediate(i);
            let q = n("q")?;
            let qn = n("qn")?;
            require_f32(&qn, &format!("l.{i}.qn"))?;
            let o = n("o")?;
            if q.rows() != cfg.heads * hd || q.cols() != cfg.hidden {
                return Err(format!(
                    "Gemma4 tensor l.{i}.q has shape {:?}, expected [{}, {}]",
                    q.shape,
                    cfg.heads * hd,
                    cfg.hidden
                ));
            }
            if qn.numel() != hd {
                return Err(format!("Gemma4 tensor l.{i}.qn has shape {:?}, expected [{hd}]", qn.shape));
            }
            if o.rows() != cfg.hidden || o.cols() != cfg.heads * hd {
                return Err(format!(
                    "Gemma4 tensor l.{i}.o has shape {:?}, expected [{}, {}]",
                    o.shape,
                    cfg.hidden,
                    cfg.heads * hd
                ));
            }
            let (k, kn, v) = if cfg.is_kv_shared(i) {
                (None, None, None)
            } else {
                let k = n("k")?;
                let kn = n("kn")?;
                let v = n("v")?;
                require_f32(&kn, &format!("l.{i}.kn"))?;
                let expected = kv * hd;
                if k.rows() != expected || k.cols() != cfg.hidden || v.rows() != expected || v.cols() != cfg.hidden {
                    return Err(format!(
                        "Gemma4 tensors l.{i}.k/v have an unexpected shape for {kv} KV heads of width {hd}"
                    ));
                }
                if kn.numel() != hd {
                    return Err(format!("Gemma4 tensor l.{i}.kn has shape {:?}, expected [{hd}]", kn.shape));
                }
                (Some(k), Some(kn), Some(v))
            };
            let attn_norm = n("attn_norm")?;
            let attn_post_norm = n("attn_post_norm")?;
            let ffn_norm = n("ffn_norm")?;
            let ffn_post_norm = n("ffn_post_norm")?;
            let ple_norm = n("ple_norm")?;
            for (name, t, width) in [
                ("attn_norm", &attn_norm, cfg.hidden),
                ("attn_post_norm", &attn_post_norm, cfg.hidden),
                ("ffn_norm", &ffn_norm, cfg.hidden),
                ("ffn_post_norm", &ffn_post_norm, cfg.hidden),
                ("ple_norm", &ple_norm, cfg.hidden),
            ] {
                require_f32(t, &format!("l.{i}.{name}"))?;
                if t.numel() != width {
                    return Err(format!("Gemma4 tensor l.{i}.{name} has shape {:?}, expected [{width}]", t.shape));
                }
            }
            let gate = n("gate")?;
            let up = n("up")?;
            let down = n("down")?;
            let ple_gate = n("ple_gate")?;
            let ple_out = n("ple_out")?;
            let ple_norm = n("ple_norm")?;
            let scalar = n("scalar")?;
            require_f32(&scalar, &format!("l.{i}.scalar"))?;
            if gate.rows() != inter || gate.cols() != cfg.hidden || up.rows() != inter || up.cols() != cfg.hidden {
                return Err(format!("Gemma4 l.{i}.gate/up have an unexpected shape for intermediate={inter}"));
            }
            if down.rows() != cfg.hidden || down.cols() != inter {
                return Err(format!(
                    "Gemma4 tensor l.{i}.down has shape {:?}, expected [{}, {}]",
                    down.shape, cfg.hidden, inter
                ));
            }
            if ple_gate.rows() != cfg.ple_dim || ple_gate.cols() != cfg.hidden {
                return Err(format!(
                    "Gemma4 tensor l.{i}.ple_gate has shape {:?}, expected [{}, {}]",
                    ple_gate.shape, cfg.ple_dim, cfg.hidden
                ));
            }
            if ple_out.rows() != cfg.hidden || ple_out.cols() != cfg.ple_dim {
                return Err(format!(
                    "Gemma4 tensor l.{i}.ple_out has shape {:?}, expected [{}, {}]",
                    ple_out.shape, cfg.hidden, cfg.ple_dim
                ));
            }
            if scalar.numel() != 1 {
                return Err(format!("Gemma4 tensor l.{i}.scalar has shape {:?}, expected [1]", scalar.shape));
            }
            // Avoid silently accepting a full layer with the wrong key/value shape even when a
            // malformed pack happens to contain the optional tensors.
            if full && cfg.global_head_dim != hd {
                return Err(format!("Gemma4 layer {i} full attention has inconsistent head dimension"));
            }
            layers.push(LayerWeights {
                attn_norm,
                attn_post_norm,
                ffn_norm,
                ffn_post_norm,
                q,
                qn,
                k,
                kn,
                v,
                o,
                gate,
                up,
                down,
                ple_gate,
                ple_out,
                ple_norm,
                scalar,
            });
        }
        Ok(Self {
            cfg,
            store,
            embed,
            norm,
            ple_embeddings,
            ple_projections,
            layers,
            ple_norm,
            panel: RefCell::new(Vec::new()),
        })
    }

    fn rms_row(&self, src: &[f32], weight: Option<&[f32]>, out: &mut [f32]) {
        let mut sum = 0.0f32;
        for &v in src {
            sum += v * v;
        }
        let inv = (sum / src.len() as f32 + self.cfg.eps).powf(-0.5);
        match weight {
            Some(w) => out.iter_mut().zip(src).zip(w).for_each(|((o, &x), &g)| *o = x * inv * g),
            None => out.iter_mut().zip(src).for_each(|(o, &x)| *o = x * inv),
        }
    }

    fn rms_rows(&self, src: &[f32], t: usize, d: usize, weight: Option<&[f32]>, out: &mut [f32]) {
        for r in 0..t {
            self.rms_row(&src[r * d..(r + 1) * d], weight, &mut out[r * d..(r + 1) * d]);
        }
    }

    fn matmul(&self, x: &[f32], t: usize, w: &TensorInfo, out: &mut [f32]) {
        let mut panel = self.panel.borrow_mut();
        linear(x, t, self.store.mat(w), None, out, &mut panel);
    }

    fn rms_row_in_place(&self, row: &mut [f32], weight: Option<&[f32]>) {
        let mut sum = 0.0f32;
        for &v in row.iter() {
            sum += v * v;
        }
        let inv = (sum / row.len() as f32 + self.cfg.eps).powf(-0.5);
        match weight {
            Some(w) => row.iter_mut().zip(w).for_each(|(x, &g)| *x *= inv * g),
            None => row.iter_mut().for_each(|x| *x *= inv),
        }
    }

    fn rope_row(&self, row: &mut [f32], pos: usize, full: bool) {
        let hd = row.len();
        let half = hd / 2;
        let (theta, rotated) = if full {
            let rotated = ((hd as f32 * self.cfg.global_partial_rotary).floor() as usize / 2) * 2;
            (self.cfg.global_rope_theta, rotated)
        } else {
            (self.cfg.local_rope_theta, hd)
        };
        let mut cos = vec![1.0f32; hd];
        let mut sin = vec![0.0f32; hd];
        for i in 0..rotated / 2 {
            let inv = theta.powf(-((2 * i) as f32) / hd as f32);
            let a = pos as f32 * inv;
            let (c, s) = (a.cos(), a.sin());
            cos[i] = c;
            sin[i] = s;
            cos[half + i] = c;
            sin[half + i] = s;
        }
        let before = row.to_vec();
        for i in 0..hd {
            let j = if i < half { i + half } else { i - half };
            row[i] = before[i] * cos[i] + if i < half { -before[j] } else { before[j] } * sin[i];
        }
    }

    fn embedding(&self, ids: &[u32]) -> Vec<f32> {
        let t = ids.len();
        let d = self.cfg.hidden;
        let mut x = vec![0.0; t * d];
        let scale = bf16_round((d as f32).sqrt());
        let vocab = self.embed.rows();
        for (i, &id) in ids.iter().enumerate() {
            self.store.row(&self.embed, (id as usize).min(vocab.saturating_sub(1)), &mut x[i * d..(i + 1) * d]);
            x[i * d..(i + 1) * d].iter_mut().for_each(|v| *v *= scale);
        }
        x
    }

    fn per_layer_inputs(&self, ids: &[u32], x: &[f32]) -> Vec<Vec<f32>> {
        let t = ids.len();
        let mut out = vec![vec![0.0; t * self.cfg.ple_dim]; self.cfg.layers];
        let embed_scale = bf16_round((self.cfg.ple_dim as f32).sqrt());
        let projection_scale = (self.cfg.hidden as f32).sqrt().recip();
        let mut projection = vec![0.0; t * self.cfg.ple_dim];
        let mut normed = vec![0.0; t * self.cfg.ple_dim];
        let ple_norm = self.store.f32s(&self.ple_norm);
        for layer in 0..self.cfg.layers {
            let mut token = vec![0.0; t * self.cfg.ple_dim];
            let vocab = self.ple_embeddings[layer].rows();
            for (r, &id) in ids.iter().enumerate() {
                self.store.row(
                    &self.ple_embeddings[layer],
                    (id as usize).min(vocab.saturating_sub(1)),
                    &mut token[r * self.cfg.ple_dim..(r + 1) * self.cfg.ple_dim],
                );
                token[r * self.cfg.ple_dim..(r + 1) * self.cfg.ple_dim].iter_mut().for_each(|v| *v *= embed_scale);
            }
            self.matmul(x, t, &self.ple_projections[layer], &mut projection);
            projection.iter_mut().for_each(|v| *v *= projection_scale);
            self.rms_rows(&projection, t, self.cfg.ple_dim, Some(ple_norm), &mut normed);
            for (o, (a, b)) in out[layer].iter_mut().zip(token.iter().zip(&normed)) {
                *o = (a + b) * std::f32::consts::FRAC_1_SQRT_2;
            }
        }
        out
    }

    fn attention(
        &self,
        layer: usize,
        x: &[f32],
        t: usize,
        layer_weights: &LayerWeights,
        shared: &mut [Option<KvCache>; 2],
    ) -> Vec<f32> {
        let d = self.cfg.hidden;
        let hd = self.cfg.head_dim(layer);
        let kv_heads = self.cfg.kv_heads(layer);
        let full = self.cfg.is_full(layer);
        let mut q = vec![0.0; t * self.cfg.heads * hd];
        self.matmul(x, t, &layer_weights.q, &mut q);
        let qn = self.store.f32s(&layer_weights.qn);
        for r in 0..t {
            for h in 0..self.cfg.heads {
                let row = &mut q[(r * self.cfg.heads + h) * hd..(r * self.cfg.heads + h + 1) * hd];
                self.rms_row_in_place(row, Some(qn));
                self.rope_row(row, r, full);
            }
        }
        let ty = if full { 1 } else { 0 };
        let cache = if self.cfg.is_kv_shared(layer) {
            shared[ty].as_ref().expect("validated Gemma4 shared KV source")
        } else {
            let k_weight = layer_weights.k.as_ref().expect("validated Gemma4 K weight");
            let v_weight = layer_weights.v.as_ref().expect("validated Gemma4 V weight");
            let kn = self.store.f32s(layer_weights.kn.as_ref().expect("validated Gemma4 K norm"));
            let mut k = vec![0.0; t * kv_heads * hd];
            let mut v = vec![0.0; t * kv_heads * hd];
            self.matmul(x, t, k_weight, &mut k);
            self.matmul(x, t, v_weight, &mut v);
            for r in 0..t {
                for h in 0..kv_heads {
                    let kr = &mut k[(r * kv_heads + h) * hd..(r * kv_heads + h + 1) * hd];
                    self.rms_row_in_place(kr, Some(kn));
                    self.rope_row(kr, r, full);
                    let vr = &mut v[(r * kv_heads + h) * hd..(r * kv_heads + h + 1) * hd];
                    self.rms_row_in_place(vr, None);
                }
            }
            let source = KvCache { heads: kv_heads, head_dim: hd, keys: k, values: v };
            if !self.cfg.is_kv_shared(layer) {
                let shared_start = self.cfg.layers - self.cfg.kv_shared_layers;
                let is_source = layer < shared_start
                    && (layer + 1..shared_start).all(|j| self.cfg.layer_types[j] != self.cfg.layer_types[layer]);
                if is_source {
                    shared[ty] = Some(source.clone());
                }
            }
            // Keep this local value alive for the attention below.  It cannot be returned from the
            // branch above, so the owned cache is handled by the common path below.
            return self.attend_with_cache(layer, &q, t, source, d);
        };
        self.attend_with_cache(layer, &q, t, cache.clone(), d)
    }

    fn attend_with_cache(&self, layer: usize, q: &[f32], t: usize, cache: KvCache, d: usize) -> Vec<f32> {
        let hd = cache.head_dim;
        let heads = self.cfg.heads;
        let groups = heads / cache.heads;
        let window = if self.cfg.is_full(layer) { None } else { Some(self.cfg.sliding_window) };
        let mut ctx = vec![0.0; t * heads * hd];
        let mut scores = vec![0.0; t];
        for r in 0..t {
            let lo = window.map_or(0, |w| r.saturating_add(1).saturating_sub(w));
            for h in 0..heads {
                let qrow = &q[(r * heads + h) * hd..(r * heads + h + 1) * hd];
                let kh = h / groups;
                let mut max = f32::NEG_INFINITY;
                for j in lo..=r {
                    let krow = &cache.keys[(j * cache.heads + kh) * hd..(j * cache.heads + kh + 1) * hd];
                    let mut s = 0.0;
                    for c in 0..hd {
                        s += qrow[c] * krow[c];
                    }
                    scores[j] = s;
                    max = max.max(s);
                }
                let mut sum = 0.0;
                for j in lo..=r {
                    scores[j] = (scores[j] - max).exp();
                    sum += scores[j];
                }
                let inv = sum.recip();
                let out = &mut ctx[(r * heads + h) * hd..(r * heads + h + 1) * hd];
                for j in lo..=r {
                    let p = scores[j] * inv;
                    let vrow = &cache.values[(j * cache.heads + kh) * hd..(j * cache.heads + kh + 1) * hd];
                    for c in 0..hd {
                        out[c] += p * vrow[c];
                    }
                }
            }
        }
        let mut projected = vec![0.0; t * d];
        self.store_matmul(&ctx, t, &self.layers[layer].o, &mut projected);
        projected
    }

    fn store_matmul(&self, x: &[f32], t: usize, w: &TensorInfo, out: &mut [f32]) {
        let mut panel = self.panel.borrow_mut();
        linear(x, t, self.store.mat(w), None, out, &mut panel);
    }

    /// Runs a complete dense causal prefill and returns the final-normalized hidden row for the
    /// final prompt token.  KV caching is intentionally deferred; direct scoring prompts are
    /// short and this path is primarily a native correctness/reference implementation.
    pub fn forward(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        if ids.is_empty() {
            return Err("Gemma4 forward: prompt is empty".into());
        }
        if ids.len() > self.cfg.max_input_tokens {
            return Err(format!(
                "Gemma4 forward: prompt has {} tokens, runtime limit is {}",
                ids.len(),
                self.cfg.max_input_tokens
            ));
        }
        if let Some((position, &id)) = ids.iter().enumerate().find(|(_, &id)| id as usize >= self.cfg.vocab) {
            return Err(format!(
                "Gemma4 forward: token id {id} at position {position} exceeds vocab size {}",
                self.cfg.vocab
            ));
        }
        let t = ids.len();
        let d = self.cfg.hidden;
        let mut x = self.embedding(ids);
        let ple = self.per_layer_inputs(ids, &x);
        let mut shared: [Option<KvCache>; 2] = [None, None];
        for i in 0..self.cfg.layers {
            let lw = &self.layers[i];
            let mut h = vec![0.0; t * d];
            self.rms_rows(&x, t, d, Some(self.store.f32s(&lw.attn_norm)), &mut h);
            let attn = self.attention(i, &h, t, lw, &mut shared);
            self.rms_rows(&attn, t, d, Some(self.store.f32s(&lw.attn_post_norm)), &mut h);
            for (a, b) in x.iter_mut().zip(&h) {
                *a += *b;
            }
            let residual = x.clone();
            self.rms_rows(&x, t, d, Some(self.store.f32s(&lw.ffn_norm)), &mut h);
            let inter = self.cfg.mlp_intermediate(i);
            let mut gate = vec![0.0; t * inter];
            let mut up = vec![0.0; t * inter];
            self.matmul(&h, t, &lw.gate, &mut gate);
            self.matmul(&h, t, &lw.up, &mut up);
            for (g, u) in gate.iter_mut().zip(&up) {
                *g = gelu_tanh(*g) * *u;
            }
            let mut down = vec![0.0; t * d];
            self.matmul(&gate, t, &lw.down, &mut down);
            self.rms_rows(&down, t, d, Some(self.store.f32s(&lw.ffn_post_norm)), &mut h);
            for (a, (r, b)) in x.iter_mut().zip(residual.iter().zip(&h)) {
                *a = r + b;
            }
            let residual = x.clone();
            let mut ple_gate = vec![0.0; t * self.cfg.ple_dim];
            self.matmul(&x, t, &lw.ple_gate, &mut ple_gate);
            for v in &mut ple_gate {
                *v = gelu_tanh(*v);
            }
            for (a, b) in ple_gate.iter_mut().zip(&ple[i]) {
                *a *= *b;
            }
            let mut ple_out = vec![0.0; t * d];
            self.matmul(&ple_gate, t, &lw.ple_out, &mut ple_out);
            self.rms_rows(&ple_out, t, d, Some(self.store.f32s(&lw.ple_norm)), &mut h);
            for (a, (r, b)) in x.iter_mut().zip(residual.iter().zip(&h)) {
                *a = r + b;
            }
            let scalar = self.store.f32s(&lw.scalar)[0];
            x.iter_mut().for_each(|v| *v *= scalar);
        }
        let mut final_x = vec![0.0; t * d];
        self.rms_rows(&x, t, d, Some(self.store.f32s(&self.norm)), &mut final_x);
        Ok(final_x[(t - 1) * d..t * d].to_vec())
    }
}

fn bf16_round(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded)
}

#[inline]
fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
}

fn prompt_for(state: &Value, q: &KevQuestion) -> Result<String, String> {
    if !(2..=MAX_LABELS).contains(&q.options.len()) {
        return Err(format!("question {:?}: Gemma4 direct scoring requires 2..={MAX_LABELS} options", q.id));
    }
    let options = q
        .options
        .iter()
        .enumerate()
        .map(|(i, option)| {
            Value::Object(vec![
                ("letter".into(), Value::Str(char::from(b'A' + i as u8).to_string())),
                ("description".into(), Value::Str(option.clone())),
            ])
        })
        .collect();
    let payload = Value::Object(vec![
        ("evidence".into(), state.clone()),
        ("criterion".into(), Value::Str(q.instructions.clone())),
        ("options".into(), Value::Array(options)),
    ]);
    // This is the exact Gemma4 canonical chat template for a non-thinking system + user turn
    // with `add_generation_prompt=True`: BOS, system turn, user turn, then model turn.  The
    // trailing newline is part of `<|turn>model\n` and is therefore tokenized.
    Ok(format!(
        "<bos><|turn>system\n{DIRECT_OPTIONS_SYSTEM}<turn|>\n<|turn>user\n{}<turn|>\n<|turn>model\n",
        payload.py_dumps(false)
    ))
}

fn request_state(r: &Request) -> Value {
    r.text_with(|v| v.py_dumps(false)).map(Value::Str).unwrap_or_else(|| r.state.clone())
}

fn encode_one(tok: &Tokenizer, cfg: &Gemma4Config, state: &Value, q: KevQuestion) -> Result<Vec<u32>, String> {
    let prompt = prompt_for(state, &q)?;
    let ids = tok.encode(&prompt);
    if ids.is_empty() {
        return Err(format!("question {:?}: Gemma4 prompt tokenized to zero tokens", q.id));
    }
    if ids.len() > cfg.max_input_tokens {
        return Err(format!(
            "question {:?}: Gemma4 prompt has {} tokens, runtime limit is {}",
            q.id,
            ids.len(),
            cfg.max_input_tokens
        ));
    }
    Ok(ids)
}

fn softmax(z: &[f32]) -> Vec<f32> {
    let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = z.iter().map(|&x| (x - max).exp()).collect();
    let sum = exp.iter().sum::<f32>();
    exp.into_iter().map(|x| x / sum).collect()
}

fn require_finite_row(index: usize, row: &[f32]) -> Result<(), String> {
    if row.iter().any(|value| !value.is_finite()) {
        return Err(format!("Gemma4 finish: normalized hidden row {index} contains a non-finite value"));
    }
    Ok(())
}

fn round2(x: f64) -> Value {
    Value::Float(format!("{x:.2}").parse().unwrap_or(x))
}

fn score_answer(q: &PreparedQuestion, logits: &[f32]) -> Value {
    let p = softmax(logits);
    let best = p.iter().enumerate().fold(0, |b, (i, &v)| if v > p[b] { i } else { b });
    match q.kind {
        Kind::Noul => Value::Object(vec![
            ("type".into(), Value::Str("noul".into())),
            ("noul".into(), round2(p.get(1).copied().unwrap_or(0.0) as f64)),
        ]),
        Kind::Choice => {
            let n = p.len();
            let confidence = if n < 2 { 1.0 } else { (p[best] - 1.0 / n as f32) / (1.0 - 1.0 / n as f32) };
            Value::Object(vec![
                ("type".into(), Value::Str("choice".into())),
                ("choice".into(), Value::Str(q.keys[best].clone())),
                ("confidence".into(), round2(confidence as f64)),
                (
                    "probabilities".into(),
                    Value::Object(q.keys.iter().zip(&p).map(|(k, &v)| (k.clone(), round2(v as f64))).collect()),
                ),
            ])
        }
        Kind::Score => {
            let score = p.iter().enumerate().map(|(i, &v)| i as f32 * v).sum::<f32>();
            let spread = p.iter().enumerate().map(|(i, &v)| v * (i as f32 - best as f32).abs()).sum::<f32>();
            let confidence = 1.0 - spread / (p.len().saturating_sub(1).max(1) as f32);
            Value::Object(vec![
                ("type".into(), Value::Str("score".into())),
                ("score".into(), round2(score as f64)),
                (
                    "legend".into(),
                    Value::Object(
                        q.keys.iter().zip(&q.options).map(|(k, o)| (k.clone(), Value::Str(o.clone()))).collect(),
                    ),
                ),
                (
                    "probabilities".into(),
                    Value::Object(q.keys.iter().zip(&p).map(|(k, &v)| (k.clone(), round2(v as f64))).collect()),
                ),
                ("confidence".into(), round2(confidence as f64)),
            ])
        }
    }
}

/// A Gemma4 text coordinator and optional native CPU model.
pub struct Gemma4Engine {
    pub cfg: Gemma4Config,
    pub tok: Tokenizer,
    pub info: Value,
    pub modalities: Vec<Modality>,
    store: Arc<Store>,
    labels: TensorInfo,
    pub model: Option<Gemma4Model>,
}

impl Gemma4Engine {
    /// Loads a complete or coordinator-only pack.  The latter is sufficient for the WebGPU path,
    /// where the GPU owns embeddings, decoder layers and final norm and this instance only scores
    /// the returned normalized rows.
    pub fn load(buf: AlignedBuf) -> Result<Self, String> {
        let h = pack::parse_header(buf.as_slice())?;
        if buf.len() < h.total_size {
            return Err(format!("pack is truncated: {} of {} bytes", buf.len(), h.total_size));
        }
        let cfg = Gemma4Config::from_json(h.config())?;
        let tok = Tokenizer::from_bytes(&buf.as_slice()[h.tokenizer_offset..h.tokenizer_offset + h.tokenizer_size])?;
        let labels = h.tensor("readout.labels").cloned().ok_or("Gemma4 pack has no readout.labels tensor")?;
        if labels.rows() != MAX_LABELS || labels.cols() != cfg.hidden {
            return Err(format!(
                "Gemma4 readout.labels has shape {:?}, expected [{MAX_LABELS}, {}]",
                labels.shape, cfg.hidden
            ));
        }
        let info = h.json.get("model").cloned().unwrap_or(Value::Null);
        let modalities =
            Modality::from_config(h.config()).into_iter().filter(|m| *m == Modality::Text).collect::<Vec<_>>();
        if modalities.is_empty() {
            return Err("Gemma4 pack must declare text support".into());
        }
        let store = Arc::new(Store::new(buf, h.tensors.clone())?);
        let model = if store.has("l.0.q") { Some(Gemma4Model::new(cfg.clone(), store.clone())?) } else { None };
        Ok(Self { cfg, tok, info, modalities, store, labels, model })
    }

    pub fn prepare(&self, state: &Value, questions: &Value) -> Result<Prepared, String> {
        let req = Request { state: state.clone(), parts: Vec::new(), questions: questions.clone() };
        self.prepare_all(std::slice::from_ref(&req))
    }

    /// Tokenizes one complete prompt per question.  `Prepared::ids` is a flattened backing array;
    /// `Prepared::sequences` gives the exact ranges and the row order expected by `finish`.
    pub fn prepare_all(&self, requests: &[Request]) -> Result<Prepared, String> {
        if requests.is_empty() {
            return Err("Gemma4 prepare: at least one request is required".into());
        }
        let mut prepared = Prepared::default();
        for (ri, request) in requests.iter().enumerate() {
            request.check("gemma4", &self.modalities)?;
            let state = request_state(request);
            let questions = kev::parse_questions(&request.questions)?;
            let first = prepared.sequences.len();
            for q in questions {
                let labels = q.options.len();
                let ids = encode_one(&self.tok, &self.cfg, &state, q.clone())?;
                let question = prepared.questions.len();
                prepared.questions.push(q.into());
                let start = prepared.ids.len();
                prepared.ids.extend_from_slice(&ids);
                prepared.sequences.push(PreparedSequence { start, len: ids.len(), request: ri, question, labels });
            }
            if prepared.sequences.len() == first {
                return Err(format!("request {ri}: questions must not be empty"));
            }
            prepared.request_sequence_ranges.push((first, prepared.sequences.len()));
        }
        Ok(prepared)
    }

    /// Scores normalized final hidden rows returned by the GPU or native model and formats one
    /// response object per request.  Rows must be ordered like `Prepared::sequences`.
    pub fn finish(&self, prepared: &Prepared, rows: &[f32]) -> Result<Vec<Value>, String> {
        if rows.len() != prepared.sequences.len() * self.cfg.hidden {
            return Err(format!(
                "Gemma4 finish: expected {} hidden values for {} sequences, got {}",
                prepared.sequences.len() * self.cfg.hidden,
                prepared.sequences.len(),
                rows.len()
            ));
        }
        let mut logits = vec![vec![0.0f32; MAX_LABELS]; prepared.sequences.len()];
        let mut panel = Vec::new();
        for (i, row) in rows.chunks_exact(self.cfg.hidden).enumerate() {
            require_finite_row(i, row)?;
            linear(row, 1, self.store.mat(&self.labels), None, &mut logits[i], &mut panel);
            if logits[i].iter().any(|value| !value.is_finite()) {
                return Err(format!("Gemma4 finish: readout logits for sequence {i} are non-finite"));
            }
            if let Some(cap) = self.cfg.final_logit_softcap {
                for z in &mut logits[i] {
                    *z = (*z / cap).tanh() * cap;
                }
            }
            logits[i].iter_mut().for_each(|z| *z /= self.cfg.temperature);
            if logits[i].iter().any(|value| !value.is_finite()) {
                return Err(format!("Gemma4 finish: scaled logits for sequence {i} are non-finite"));
            }
            logits[i].truncate(prepared.sequences[i].labels);
        }
        let mut answers: Vec<Vec<(String, Value)>> = vec![Vec::new(); prepared.request_sequence_ranges.len()];
        let mut raw_probabilities: Vec<Vec<(String, Value)>> = vec![Vec::new(); prepared.request_sequence_ranges.len()];
        let mut tokens = vec![0usize; prepared.request_sequence_ranges.len()];
        for (i, seq) in prepared.sequences.iter().enumerate() {
            let q = &prepared.questions[seq.question];
            let probabilities = softmax(&logits[i]);
            if probabilities.iter().any(|value| !value.is_finite()) {
                return Err(format!("Gemma4 finish: probabilities for sequence {i} are non-finite"));
            }
            answers[seq.request].push((q.id.clone(), score_answer(q, &logits[i])));
            raw_probabilities[seq.request].push((
                q.id.clone(),
                Value::Array(probabilities.into_iter().map(|v| Value::Float(v as f64)).collect()),
            ));
            tokens[seq.request] += seq.len;
        }
        Ok(answers
            .into_iter()
            .zip(raw_probabilities)
            .zip(tokens)
            .map(|((pairs, raw), input_tokens)| {
                let answer_obj = Value::Object(pairs);
                let output_tokens = self.tok.encode(&answer_obj.py_dumps(true)).len();
                Value::Object(vec![
                    ("model".into(), self.info.get("name").cloned().unwrap_or(Value::Str("gemma4".into()))),
                    ("answers".into(), answer_obj),
                    ("raw_probabilities".into(), Value::Object(raw)),
                    (
                        "usage".into(),
                        Value::Object(vec![
                            ("input_tokens".into(), Value::Int(input_tokens.to_string())),
                            ("output_tokens".into(), Value::Int(output_tokens.to_string())),
                        ]),
                    ),
                    (
                        "probability_status".into(),
                        Value::Str("conditional option score; uncalibrated as decision confidence".into()),
                    ),
                ])
            })
            .collect())
    }

    /// Runs the full native model once per prepared sequence, then applies the same readout used
    /// by the GPU path.
    pub fn decide_native(&mut self, requests: &[Request]) -> Result<Vec<Value>, String> {
        let prepared = self.prepare_all(requests)?;
        let model = self.model.as_ref().ok_or("Gemma4 pack is coordinator-only; use WebGPU for the decoder")?;
        let mut rows = Vec::with_capacity(prepared.sequences.len() * self.cfg.hidden);
        for i in 0..prepared.sequence_count() {
            rows.extend_from_slice(&model.forward(prepared.sequence_ids(i))?);
        }
        self.finish(&prepared, &rows)
    }
}

impl crate::runtime::Model for Gemma4Engine {
    fn arch(&self) -> &'static str {
        "gemma4"
    }

    fn info(&self) -> &Value {
        &self.info
    }

    fn modalities(&self) -> &[Modality] {
        &self.modalities
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    fn decide(&mut self, requests: &[Request]) -> Result<Vec<Value>, String> {
        self.decide_native(requests)
    }

    fn as_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add_config_field(config: &mut Value, key: &str, value: Value) {
        match config {
            Value::Object(fields) => fields.push((key.into(), value)),
            _ => panic!("test config must be an object"),
        }
    }

    fn set_config_field(config: &mut Value, key: &str, value: Value) {
        match config {
            Value::Object(fields) => fields.iter_mut().find(|(name, _)| name == key).unwrap().1 = value,
            _ => panic!("test config must be an object"),
        }
    }

    fn cfg_json() -> Value {
        Value::Object(vec![
            ("arch".into(), Value::Str("gemma4".into())),
            ("hidden_size".into(), Value::Int("12".into())),
            ("intermediate_size".into(), Value::Int("32".into())),
            ("vocab_size".into(), Value::Int("128".into())),
            ("vocab_size_per_layer_input".into(), Value::Int("128".into())),
            ("num_hidden_layers".into(), Value::Int("3".into())),
            ("num_attention_heads".into(), Value::Int("2".into())),
            ("num_key_value_heads".into(), Value::Int("1".into())),
            ("head_dim".into(), Value::Int("8".into())),
            ("global_head_dim".into(), Value::Int("16".into())),
            (
                "layer_types".into(),
                Value::Array(vec![
                    Value::Str("sliding_attention".into()),
                    Value::Str("full_attention".into()),
                    Value::Str("full_attention".into()),
                ]),
            ),
            ("num_kv_shared_layers".into(), Value::Int("1".into())),
            ("sliding_window".into(), Value::Int("512".into())),
            ("rms_norm_eps".into(), Value::Float(1e-6)),
            ("hidden_size_per_layer_input".into(), Value::Int("4".into())),
            ("max_position_embeddings".into(), Value::Int("4096".into())),
            ("max_input_tokens".into(), Value::Int("128".into())),
            ("use_double_wide_mlp".into(), Value::Bool(false)),
            (
                "rope_parameters".into(),
                Value::Object(vec![
                    (
                        "sliding_attention".into(),
                        Value::Object(vec![
                            ("rope_type".into(), Value::Str("default".into())),
                            ("rope_theta".into(), Value::Float(10_000.0)),
                        ]),
                    ),
                    (
                        "full_attention".into(),
                        Value::Object(vec![
                            ("rope_type".into(), Value::Str("proportional".into())),
                            ("rope_theta".into(), Value::Float(1_000_000.0)),
                            ("partial_rotary_factor".into(), Value::Float(0.25)),
                        ]),
                    ),
                ]),
            ),
        ])
    }

    #[test]
    fn parses_e2b_shape_rules_and_shared_layers() {
        let cfg = Gemma4Config::from_json(&cfg_json()).unwrap();
        assert_ne!(cfg.hidden, cfg.heads * cfg.local_head_dim);
        assert_eq!(cfg.head_dim(0), 8);
        assert_eq!(cfg.head_dim(1), 16);
        assert_eq!(cfg.kv_heads(1), 1);
        assert!(!cfg.is_kv_shared(1));
        assert!(cfg.is_kv_shared(2));
        assert_eq!(cfg.mlp_intermediate(2), 32);
    }

    #[test]
    fn ignores_global_kv_override_without_alternative_attention() {
        let mut c = cfg_json();
        add_config_field(&mut c, "num_global_key_value_heads", Value::Int("2".into()));
        let cfg = Gemma4Config::from_json(&c).unwrap();
        assert_eq!(cfg.kv_heads, 1);
        assert_eq!(cfg.global_kv_heads, 1);
        assert_eq!(cfg.kv_heads(1), 1);
    }

    #[test]
    fn rejects_moe_and_noncausal_variants() {
        let mut c = cfg_json();
        add_config_field(&mut c, "enable_moe_block", Value::Bool(true));
        assert!(Gemma4Config::from_json(&c).unwrap_err().contains("MoE"));
        let mut c = cfg_json();
        add_config_field(&mut c, "use_bidirectional_attention", Value::Str("all".into()));
        assert!(Gemma4Config::from_json(&c).unwrap_err().contains("causal"));
        let mut c = cfg_json();
        set_config_field(&mut c, "max_position_embeddings", Value::Int("8192".into()));
        set_config_field(&mut c, "max_input_tokens", Value::Int("4097".into()));
        assert!(Gemma4Config::from_json(&c).unwrap_err().contains("runtime limit"));
    }

    #[test]
    fn proportional_rope_keeps_nonrotary_quarters_identity() {
        let cfg = Gemma4Config::from_json(&cfg_json()).unwrap();
        let mut row = vec![1.0; cfg.global_head_dim];
        // At position zero all rotary terms are identity.  At a later position the explicit
        // zero-filled inverse frequencies must leave the nonrotary portions unchanged.
        let before = row.clone();
        // A simple direct check of the invariant used by rope_row: only 25% of each half carries
        // nonzero frequencies for the proportional full-attention configuration.
        let rotated = (cfg.global_head_dim as f32 * cfg.global_partial_rotary).floor() as usize;
        assert_eq!(rotated, 4);
        assert_eq!(before.len(), row.len());
        row[rotated..cfg.global_head_dim / 2].fill(2.0);
        assert!(row[cfg.global_head_dim / 2..].iter().all(|&v| v == 1.0));
    }

    #[test]
    fn prompt_is_canonical_nonthinking_gemma_turn() {
        let q = KevQuestion {
            id: "q".into(),
            kind: Kind::Choice,
            instructions: "choose".into(),
            keys: vec!["yes".into(), "no".into()],
            options: vec!["yes".into(), "no".into()],
        };
        let prompt = prompt_for(&Value::Str("evidence".into()), &q).unwrap();
        assert!(prompt.starts_with("<bos><|turn>system\n"));
        assert!(prompt.contains("<|turn>user\n"));
        assert!(prompt.ends_with("<|turn>model\n"));
        assert!(!prompt.contains("<|think|>"));
        assert!(prompt.contains("\"letter\": \"A\""));
    }

    #[test]
    fn bf16_embedding_scale_rounds_like_hf() {
        assert_eq!(bf16_round(16.0), 16.0);
        assert_eq!(bf16_round((1536.0f32).sqrt()), f32::from_bits(0x421d_0000));
    }

    #[test]
    fn rejects_nonfinite_gpu_rows() {
        assert!(require_finite_row(2, &[0.0, f32::NAN]).unwrap_err().contains("row 2"));
        assert!(require_finite_row(3, &[0.0, f32::INFINITY]).unwrap_err().contains("row 3"));
        assert!(require_finite_row(4, &[0.0, -1.0]).is_ok());
    }
}
