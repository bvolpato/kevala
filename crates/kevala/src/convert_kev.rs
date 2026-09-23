//! Converts a Kev checkpoint into a `.kevala` pack: the Qwen3.5 base (safetensors), Kev's LoRA
//! adapter folded in exactly as `kev.checkpoint` merges it (in f32, before any quantization), and
//! the pointer head from `head.pt`.
//!
//! Layout choices, all exact rewrites: q/k/v (and DeltaNet qkv/z, MLP gate/up) are concatenated
//! into one matrix each, zero-centred RMSNorm weights are stored as `1 + w`, and `A_log` as
//! `-exp(A_log)`.

use crate::convert::{f16_to_f32, quantize};
use crate::json::Value;
use crate::pack::Writer;
use crate::tokenizer::Tokenizer;
use crate::torchpt::{Py, TorchFile};

pub struct KevCheckpoint<'a> {
    /// Qwen3.5 base weights.
    pub base: &'a [u8],
    /// The base's config.json (a `text_config` inside is used when present).
    pub base_config: &'a str,
    pub base_tokenizer: &'a str,
    /// Kev's adapter_model.safetensors and adapter_config.json.
    pub adapter: &'a [u8],
    pub adapter_config: &'a str,
    /// Kev's head.pt.
    pub head: &'a [u8],
}

/// The delimiter tokens Kev reuses (kev/model.py SPECIAL): state, question, option, end of option, decide.
pub const SPECIAL: [&str; 5] = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"];
/// SemIf's direct readout accepts up to sixteen options, represented by these answer slots.
pub const SEMIF_LABELS: &str = "ABCDEFGHIJKLMNOP";

fn pointer_dimension(root: &Py, hidden: usize) -> Result<usize, String> {
    let Some(Py::Dict(tensors)) = root.get("head") else { return Err("head.pt has no head tensor dictionary".into()) };
    let dimension = match root.get("head").and_then(|head| head.get("q.weight")) {
        Some(Py::Tensor(tensor)) if tensor.shape.len() == 2 && tensor.shape[0] > 0 => tensor.shape[0],
        _ => return Err("head.pt has no rank-2 head q.weight".into()),
    };
    for (key, value) in tensors {
        let Py::Str(name) = key else { return Err("head.pt: head tensor names must be strings".into()) };
        let expected = match name.as_str() {
            "q.weight" | "k.weight" => vec![dimension, hidden],
            "q.bias" | "k.bias" => vec![dimension],
            _ => return Err(format!("head.pt: unsupported head tensor {name}")),
        };
        let Py::Tensor(tensor) = value else { return Err(format!("head.pt: {name} is not a tensor")) };
        if tensor.shape != expected {
            return Err(format!("head.pt: {name} must have shape {expected:?}, got {:?}", tensor.shape));
        }
    }
    for name in ["q.weight", "k.weight", "q.bias", "k.bias"] {
        if root.get("head").and_then(|head| head.get(name)).is_none() {
            return Err(format!("head.pt: missing {name}"));
        }
    }
    if let Py::Dict(entries) = root {
        for (name, value) in entries {
            if matches!(value, Py::Tensor(_)) && !matches!(name, Py::Str(name) if name == "temperature") {
                return Err(format!("head.pt: unconsumed top-level tensor {name:?}"));
            }
        }
    }
    Ok(dimension)
}

fn pointer_temperature(head: &TorchFile<'_>) -> Result<f64, String> {
    let temperature = match head.root.get("temperature") {
        None => 1.0,
        Some(Py::Tensor(tensor)) if tensor.shape.is_empty() || tensor.shape == [1] => {
            *head.f32(tensor)?.first().ok_or("head.pt: empty temperature tensor")? as f64
        }
        Some(value) => value.as_f64().ok_or("head.pt: temperature must be a scalar")?,
    };
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err("head.pt: temperature must be finite and positive".into());
    }
    Ok(temperature)
}

/// A safetensors file whose tensors arrive one by one: the header up front, each tensor's bytes
/// when the stream reaches it (or all at once for a file already in memory).
struct St {
    header: Value,
    /// Absolute byte offset of the first tensor byte in the safetensors file.
    data_start: usize,
    whole: Vec<u8>,
    parts: std::collections::HashMap<String, Vec<u8>>,
}

impl St {
    fn header_len(head: &[u8]) -> Result<usize, String> {
        let n = usize::try_from(u64::from_le_bytes(head.get(..8).ok_or("short safetensors")?.try_into().unwrap()))
            .map_err(|_| "safetensors header length does not fit usize")?;
        8usize.checked_add(n).ok_or_else(|| "safetensors header length overflows usize".into())
    }

    /// `head` is the file from byte 0 through at least the end of its JSON header.
    fn header_only(head: &[u8]) -> Result<St, String> {
        let end = Self::header_len(head)?;
        let n = end - 8;
        let text = std::str::from_utf8(head.get(8..8 + n).ok_or("truncated safetensors header")?)
            .map_err(|_| "safetensors header is not UTF-8")?;
        Ok(St {
            header: Value::parse(text).map_err(|e| e.to_string())?,
            data_start: end,
            whole: Vec::new(),
            parts: Default::default(),
        })
    }
    fn whole(b: &[u8]) -> Result<St, String> {
        let mut s = St::header_only(b)?;
        let n = Self::header_len(b)?;
        s.whole = b[n..].to_vec();
        Ok(s)
    }
    fn empty() -> St {
        St { header: Value::Object(Vec::new()), data_start: 0, whole: Vec::new(), parts: Default::default() }
    }
    fn has(&self, name: &str) -> bool {
        self.header.get(name).is_some()
    }
    fn shape(&self, name: &str) -> Result<Vec<usize>, String> {
        let t = self.header.get(name).ok_or_else(|| format!("checkpoint has no tensor {name}"))?;
        t.get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{name}: no shape"))?
            .iter()
            .map(|d| d.as_usize().ok_or_else(|| format!("{name}: bad shape")))
            .collect()
    }
    fn range(&self, name: &str) -> Result<(usize, usize), String> {
        let t = self.header.get(name).ok_or_else(|| format!("checkpoint has no tensor {name}"))?;
        let off = t.get("data_offsets").and_then(Value::as_array).ok_or_else(|| format!("{name}: no offsets"))?;
        if off.len() != 2 {
            return Err(format!("{name}: data_offsets must contain two values"));
        }
        let start = off[0].as_usize().ok_or_else(|| format!("{name}: bad start offset"))?;
        let end = off[1].as_usize().ok_or_else(|| format!("{name}: bad end offset"))?;
        if end < start {
            return Err(format!("{name}: data_offsets are reversed"));
        }
        Ok((start, end))
    }
    fn f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let t = self.header.get(name).ok_or_else(|| format!("checkpoint has no tensor {name}"))?;
        let raw: &[u8] = match self.parts.get(name) {
            Some(b) => b,
            None => {
                let (a, b) = self.range(name)?;
                self.whole.get(a..b).ok_or_else(|| format!("{name}: bytes not available yet"))?
            }
        };
        let shape = self.shape(name)?;
        let numel = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| format!("{name}: shape overflows usize"))?;
        let (elem, out) = match t.get("dtype").and_then(Value::as_str) {
            Some("F32") => (4, raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
            Some("BF16") => (
                2,
                raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            ),
            Some("F16") => (2, raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()),
            other => return Err(format!("{name}: unsupported dtype {other:?}")),
        };
        let expected = numel.checked_mul(elem).ok_or_else(|| format!("{name}: byte size overflows usize"))?;
        if raw.len() != expected {
            return Err(format!("{name}: expected {expected} bytes, got {}", raw.len()));
        }
        Ok(out)
    }
}

fn num(v: f64) -> Value {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        Value::Int((v as i64).to_string())
    } else {
        Value::Float(v)
    }
}

/// A Kev checkpoint being converted while its base weights stream in.
///
/// Everything but the base weights is small and arrives first (config, tokenizer, adapter,
/// head). `sources` lists the base tensors the pack needs, in file order; hand each one to
/// `add_source` as its bytes arrive and every pack tensor whose inputs are complete is written
/// into `out` right away, so at most a few source tensors are held at a time.
pub struct KevConvert {
    src: Sources,
    plan: Vec<(String, Spec)>,
    infos: Vec<crate::pack::TensorInfo>,
    done: Vec<bool>,
    uses: std::collections::HashMap<String, usize>,
    /// per job: the base tensors it reads, and how many of them are still missing
    needs: Vec<Vec<String>>,
    missing: Vec<usize>,
    /// base tensor -> the jobs that read it
    readers: std::collections::HashMap<String, Vec<usize>>,
    block: usize,
    pub prefix: Vec<u8>,
    pub total: usize,
}

impl KevConvert {
    /// base_head is the base safetensors file from byte 0 through the end of its header.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base_head: &[u8],
        base_config: &str,
        base_tokenizer: &str,
        adapter: &[u8],
        adapter_config: &str,
        head_pt: &[u8],
        block: usize,
        model: Value,
    ) -> Result<KevConvert, String> {
        let heads = [base_head];
        Self::new_inner(
            &heads,
            base_config,
            base_tokenizer,
            Some((adapter, adapter_config)),
            Some(head_pt),
            block,
            model,
            false,
        )
    }

    /// Builds a Kev converter from the headers of all base safetensors shards. The shard bodies
    /// are still streamed later through add_source, so constructing the plan never reads a
    /// checkpoint into memory.
    #[allow(clippy::too_many_arguments)]
    pub fn new_sharded(
        base_heads: &[&[u8]],
        base_config: &str,
        base_tokenizer: &str,
        adapter: &[u8],
        adapter_config: &str,
        head_pt: &[u8],
        block: usize,
        model: Value,
    ) -> Result<KevConvert, String> {
        Self::new_inner(
            base_heads,
            base_config,
            base_tokenizer,
            Some((adapter, adapter_config)),
            Some(head_pt),
            block,
            model,
            false,
        )
    }

    /// Builds a SemIf converter from one base safetensors header. SemIf uses the same Qwen3.5
    /// backbone as Kev, but has no LoRA adapter or pointer head.
    #[allow(clippy::too_many_arguments)]
    pub fn new_semif(
        base_head: &[u8],
        base_config: &str,
        base_tokenizer: &str,
        block: usize,
        model: Value,
    ) -> Result<KevConvert, String> {
        let heads = [base_head];
        Self::new_inner(&heads, base_config, base_tokenizer, None, None, block, model, true)
    }

    /// SemIf variant for a sharded Qwen3.5 checkpoint. Only shard headers are retained here;
    /// tensor bodies are supplied by the caller one range at a time.
    #[allow(clippy::too_many_arguments)]
    pub fn new_semif_sharded(
        base_heads: &[&[u8]],
        base_config: &str,
        base_tokenizer: &str,
        block: usize,
        model: Value,
    ) -> Result<KevConvert, String> {
        Self::new_inner(base_heads, base_config, base_tokenizer, None, None, block, model, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn new_inner(
        base_heads: &[&[u8]],
        base_config: &str,
        base_tokenizer: &str,
        adapter_input: Option<(&[u8], &str)>,
        head_input: Option<&[u8]>,
        block: usize,
        model: Value,
        semif: bool,
    ) -> Result<KevConvert, String> {
        if base_heads.is_empty() {
            return Err("no base safetensors shards".into());
        }
        if block == 0 {
            return Err("quantization block must be positive".into());
        }
        let mut base = Vec::with_capacity(base_heads.len());
        let mut base_index = std::collections::HashMap::new();
        for (file, head) in base_heads.iter().enumerate() {
            let st = St::header_only(head)?;
            if let Some(entries) = st.header.as_object() {
                for (name, tensor) in entries {
                    if name == "__metadata__" {
                        continue;
                    }
                    if tensor.get("data_offsets").is_none() {
                        return Err(format!("{name}: tensor entry has no data_offsets"));
                    }
                    if base_index.insert(name.clone(), file).is_some() {
                        return Err(format!("base checkpoint contains tensor {name} in multiple shards"));
                    }
                }
            }
            base.push(st);
        }
        let cfg_all = Value::parse(base_config).map_err(|e| format!("base config: {e}"))?;
        let cfg = cfg_all.get("text_config").unwrap_or(&cfg_all);
        let (adapter, scaling, adapter_rank) = if let Some((bytes, text)) = adapter_input {
            let adapter = St::whole(bytes)?;
            let acfg = Value::parse(text).map_err(|e| format!("adapter config: {e}"))?;
            let r = acfg.get("r").and_then(Value::as_f64).ok_or("adapter config: no r")?;
            let alpha = acfg.get("lora_alpha").and_then(Value::as_f64).ok_or("adapter config: no lora_alpha")?;
            if r <= 0.0 || !r.is_finite() || r.fract() != 0.0 || r > usize::MAX as f64 || !alpha.is_finite() {
                return Err("adapter config: r must be a positive integer and lora_alpha must be finite".into());
            }
            (adapter, (alpha / r) as f32, Some(r as usize))
        } else {
            (St::empty(), 0.0, None)
        };
        let head = head_input.map(TorchFile::parse).transpose()?;
        if semif && head.is_some() {
            return Err("SemIf conversion does not accept a pointer head".into());
        }
        if !semif && head.is_none() {
            return Err("Kev conversion needs head.pt".into());
        }
        let tok = Tokenizer::from_hf_json(base_tokenizer)?;

        let u = |k: &str| cfg.get(k).and_then(Value::as_usize).ok_or_else(|| format!("base config: no {k}"));
        let hidden = u("hidden_size")?;
        if hidden == 0 {
            return Err("base config: hidden_size must be positive".into());
        }
        let layer_types: Vec<String> = cfg
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or("no layer_types")?
            .iter()
            .map(|v| v.as_str().map(String::from).ok_or("layer_types must contain strings"))
            .collect::<Result<_, _>>()?;
        if let Some(n) = cfg.get("num_hidden_layers").and_then(Value::as_usize) {
            if n != layer_types.len() {
                return Err(format!("layer_types has {}, config declares {n} layers", layer_types.len()));
            }
        }
        let rope = cfg.get("rope_parameters").ok_or("no rope_parameters")?;
        let head_dim = u("head_dim")?;
        let rotary =
            (head_dim as f64 * rope.get("partial_rotary_factor").and_then(Value::as_f64).unwrap_or(1.0)) as usize;
        let key_heads = u("linear_num_key_heads")?;
        let value_heads = u("linear_num_value_heads")?;
        if key_heads == 0 || value_heads == 0 || value_heads % key_heads != 0 {
            return Err("base config: linear value heads must be a positive multiple of key heads".into());
        }
        if u("linear_key_head_dim")? == 0 || u("linear_value_head_dim")? == 0 || u("linear_conv_kernel_dim")? == 0 {
            return Err("base config: DeltaNet dimensions must be positive".into());
        }
        let (temperature, ptr_dim) = if let Some(head) = head.as_ref() {
            (pointer_temperature(head)?, pointer_dimension(&head.root, hidden)?)
        } else {
            (1.0, 0)
        };
        let label_ids: Vec<u32> = if semif {
            SEMIF_LABELS
                .chars()
                .map(|label| {
                    let text = label.to_string();
                    let ids = tok.encode(&text);
                    if ids.len() != 1 {
                        return Err(format!("SemIf answer {label:?} is not one exact tokenizer token"));
                    }
                    Ok(ids[0])
                })
                .collect::<Result<_, _>>()?
        } else {
            Vec::new()
        };
        if label_ids.iter().collect::<std::collections::HashSet<_>>().len() != label_ids.len() {
            return Err("SemIf answer tokens collide".into());
        }
        let template_ids: Vec<Value> = if semif {
            SPECIAL.iter().map(|_| Value::Int("0".into())).collect()
        } else {
            SPECIAL
                .iter()
                .map(|s| {
                    tok.token_id(s).map(|i| Value::Int(i.to_string())).ok_or_else(|| format!("tokenizer has no {s}"))
                })
                .collect::<Result<_, _>>()?
        };

        let mut config_fields = vec![
            ("arch".into(), Value::Str("kev".into())),
            ("backbone".into(), Value::Str("qwen3_5".into())),
            ("modalities".into(), Value::Array(vec![Value::Str("text".into())])),
            ("hidden_size".into(), num(hidden as f64)),
            ("intermediate_size".into(), num(u("intermediate_size")? as f64)),
            ("rms_norm_eps".into(), Value::Float(cfg.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6))),
            ("num_attention_heads".into(), num(u("num_attention_heads")? as f64)),
            ("num_key_value_heads".into(), num(u("num_key_value_heads")? as f64)),
            ("head_dim".into(), num(head_dim as f64)),
            ("rotary_dim".into(), num(rotary as f64)),
            ("rope_theta".into(), num(rope.get("rope_theta").and_then(Value::as_f64).ok_or("no rope_theta")?)),
            ("linear_num_key_heads".into(), num(key_heads as f64)),
            ("linear_num_value_heads".into(), num(value_heads as f64)),
            ("linear_key_head_dim".into(), num(u("linear_key_head_dim")? as f64)),
            ("linear_value_head_dim".into(), num(u("linear_value_head_dim")? as f64)),
            ("linear_conv_kernel_dim".into(), num(u("linear_conv_kernel_dim")? as f64)),
            ("layer_types".into(), Value::Array(layer_types.iter().map(|s| Value::Str(s.clone())).collect())),
            ("vocab_size".into(), num(u("vocab_size")? as f64)),
            ("pointer_dim".into(), num(ptr_dim as f64)),
            ("temperature".into(), Value::Float(temperature)),
            ("max_state".into(), num(8192.0)),
            ("max_branch".into(), num(8192.0)),
            ("block".into(), num(block as f64)),
            (
                "template".into(),
                Value::Object(vec![
                    ("type".into(), Value::Str(if semif { "semif" } else { "kev" }.into())),
                    ("tokens".into(), Value::Array(SPECIAL.iter().map(|s| Value::Str(s.to_string())).collect())),
                    ("token_ids".into(), Value::Array(template_ids)),
                ]),
            ),
        ];
        if semif {
            config_fields.push(("readout".into(), Value::Str("semif".into())));
            config_fields.push(("prompt_version".into(), Value::Str("direct-options-v1".into())));
            config_fields.push((
                "label_token_ids".into(),
                Value::Array(label_ids.iter().map(|&id| Value::Int(id.to_string())).collect()),
            ));
        }
        let config = Value::Object(config_fields);

        let mut head_tensors = std::collections::HashMap::new();
        if let Some(head) = head.as_ref() {
            if let Some(Py::Dict(m)) = head.root.get("head") {
                for (k, v) in m {
                    if let (Py::Str(k), Py::Tensor(t)) = (k, v) {
                        head_tensors.insert(k.clone(), (head.f32(t)?, t.shape.clone()));
                    }
                }
            }
        }
        let mut w = Writer::new(model, config, tok.to_bytes());
        let pre = crate::convert::text_tensor_prefix(|name| base_index.contains_key(name))?;
        let src = Sources { base, base_index, adapter, head: head_tensors, scaling, pre: pre.to_string() };
        let embedding = src.base_shape(&format!("{pre}embed_tokens.weight"))?;
        if embedding != [u("vocab_size")?, hidden] {
            return Err(format!("text embedding shape {embedding:?} does not match [vocab_size, hidden_size]"));
        }
        crate::convert::validate_tokenizer_vocab(tok.vocab_size(), u("vocab_size")?, embedding[0])?;
        if semif {
            crate::convert::require_prompt_tokens(&tok, &["<|im_start|>", "<|im_end|>", "<think>", "</think>"])?;
        }
        // What each pack tensor is made of, in stream order.
        let mut plan: Vec<(String, Spec)> = vec![
            ("emb".into(), Spec::Embed(format!("{pre}embed_tokens.weight"))),
            ("norm".into(), Spec::PlusOne(format!("{pre}norm.weight"))),
        ];
        if semif {
            let emb = format!("{pre}embed_tokens.weight");
            let label_source = [
                "lm_head.weight",
                "model.lm_head.weight",
                "model.language_model.lm_head.weight",
                "language_model.lm_head.weight",
            ]
            .iter()
            .find(|name| src.has_base(name))
            .map(|name| (*name).to_string())
            .or_else(|| (cfg.get("tie_word_embeddings").and_then(Value::as_bool) == Some(true)).then_some(emb))
            .ok_or("SemIf checkpoint has no LM head and does not declare tied word embeddings")?;
            let source_shape = src.base_shape(&label_source)?;
            if source_shape.len() != 2 || source_shape[1] != hidden {
                return Err(format!("SemIf readout {label_source} must have shape [vocab, {hidden}]"));
            }
            if label_ids.iter().any(|&id| id as usize >= source_shape[0]) {
                return Err(format!("SemIf answer token exceeds {label_source} vocabulary"));
            }
            plan.push((
                "semif.labels".into(),
                Spec::Rows(label_source, label_ids.iter().map(|&id| id as usize).collect()),
            ));
        } else {
            for (dst, head_name) in
                [("ptr.q", "q.weight"), ("ptr.q.b", "q.bias"), ("ptr.k", "k.weight"), ("ptr.k.b", "k.bias")]
            {
                plan.push((dst.into(), Spec::Head(head_name)));
            }
        }
        for (i, lt) in layer_types.iter().enumerate() {
            let l = format!("layers.{i}");
            let fused = |mods: &[&str], sub: &str| Spec::Fused(mods.iter().map(|m| format!("{l}.{sub}.{m}")).collect());
            plan.push((format!("L.{i}.in_norm"), Spec::PlusOne(format!("{pre}{l}.input_layernorm.weight"))));
            plan.push((format!("L.{i}.post_norm"), Spec::PlusOne(format!("{pre}{l}.post_attention_layernorm.weight"))));
            if lt == "full_attention" {
                plan.push((format!("L.{i}.qkv"), fused(&["q_proj", "k_proj", "v_proj"], "self_attn")));
                plan.push((format!("L.{i}.q_norm"), Spec::PlusOne(format!("{pre}{l}.self_attn.q_norm.weight"))));
                plan.push((format!("L.{i}.k_norm"), Spec::PlusOne(format!("{pre}{l}.self_attn.k_norm.weight"))));
                plan.push((format!("L.{i}.o"), fused(&["o_proj"], "self_attn")));
            } else if lt == "linear_attention" {
                plan.push((format!("L.{i}.qkvz"), fused(&["in_proj_qkv", "in_proj_z"], "linear_attn")));
                plan.push((format!("L.{i}.a"), Spec::Merged(format!("{l}.linear_attn.in_proj_a"))));
                plan.push((format!("L.{i}.b"), Spec::Merged(format!("{l}.linear_attn.in_proj_b"))));
                plan.push((format!("L.{i}.conv"), Spec::Conv(format!("{pre}{l}.linear_attn.conv1d.weight"))));
                plan.push((format!("L.{i}.dt_bias"), Spec::Raw(format!("{pre}{l}.linear_attn.dt_bias"))));
                plan.push((format!("L.{i}.neg_a"), Spec::NegExp(format!("{pre}{l}.linear_attn.A_log"))));
                plan.push((format!("L.{i}.gnorm"), Spec::Raw(format!("{pre}{l}.linear_attn.norm.weight"))));
                plan.push((format!("L.{i}.out"), fused(&["out_proj"], "linear_attn")));
            } else {
                return Err(format!("unsupported Qwen3.5 layer type {lt:?}"));
            }
            plan.push((format!("L.{i}.gate_up"), fused(&["gate_proj", "up_proj"], "mlp")));
            plan.push((format!("L.{i}.down"), fused(&["down_proj"], "mlp")));
        }

        if let Some(rank) = adapter_rank {
            src.validate_adapter(&plan, rank)?;
        }
        for (name, spec) in &plan {
            match src.shape(spec)? {
                (shape, true) if shape.len() == 2 && shape[1] % block == 0 => w.add_q8(name, shape[0], shape[1], block),
                (shape, true) => {
                    return Err(format!(
                    "{name}: quantized shape must be a matrix with columns divisible by block {block}, got {shape:?}"
                ))
                }
                (shape, false) => w.add_f32(name, &shape),
            }
        }
        let (prefix, infos, total) = w.layout();
        let mut uses = std::collections::HashMap::new();
        let mut readers: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
        let needs: Vec<Vec<String>> = plan.iter().map(|(_, spec)| src.needs(spec)).collect();
        for (i, ns) in needs.iter().enumerate() {
            for n in ns {
                *uses.entry(n.clone()).or_insert(0) += 1;
                readers.entry(n.clone()).or_default().push(i);
            }
        }
        let missing = needs.iter().map(Vec::len).collect();
        let done = vec![false; plan.len()];
        Ok(KevConvert { src, plan, infos, done, uses, needs, missing, readers, block, prefix, total })
    }

    /// Base tensors to feed for a single safetensors file, `(name, absolute byte offset, length)`.
    /// This is retained for the browser's one-file converter.
    pub fn sources(&self, _header_len: usize) -> Vec<(String, usize, usize)> {
        let mut v: Vec<(String, usize, usize)> = self
            .uses
            .keys()
            .filter_map(|n| {
                let st = self.src.base_st(n).ok()?;
                let (a, b) = st.range(n).ok()?;
                Some((n.clone(), st.data_start.checked_add(a)?, b - a))
            })
            .collect();
        v.sort_by_key(|x| x.1);
        v
    }

    /// Base tensors to feed from a sharded checkpoint. Each tuple is `(shard, name, absolute
    /// byte offset, length)` and is sorted in source-file order, allowing the caller to keep one
    /// file descriptor open per shard while never materializing the checkpoint.
    pub fn source_ranges(&self) -> Result<Vec<(usize, String, usize, usize)>, String> {
        let mut ranges = Vec::new();
        for (shard, st) in self.src.base.iter().enumerate() {
            for name in self.uses.keys() {
                if self.src.base_index.get(name) == Some(&shard) {
                    let (a, b) = st.range(name)?;
                    let offset =
                        st.data_start.checked_add(a).ok_or_else(|| format!("{name}: source offset overflows usize"))?;
                    ranges.push((shard, name.clone(), offset, b - a));
                }
            }
        }
        ranges.sort_by_key(|(shard, _, at, _)| (*shard, *at));
        Ok(ranges)
    }

    /// Writes pack tensor `i` and frees the source tensors nothing else still needs.
    fn run_job(&mut self, i: usize, out: &mut [u8]) -> Result<(), String> {
        let v = self.src.data(&self.plan[i].1)?;
        let info = &self.infos[i];
        if v.len() != info.numel() {
            return Err(format!("{}: converted {} values, expected {}", info.name, v.len(), info.numel()));
        }
        if info.dtype == crate::pack::DType::Q8 {
            let (q, s) = quantize(&v, info.rows(), info.cols(), self.block);
            for (j, x) in q.into_iter().enumerate() {
                out[info.offset + j] = x as u8;
            }
            for (j, x) in s.iter().enumerate() {
                out[info.scales_offset + j * 4..info.scales_offset + j * 4 + 4].copy_from_slice(&x.to_le_bytes());
            }
        } else {
            for (j, x) in v.iter().enumerate() {
                out[info.offset + j * 4..info.offset + j * 4 + 4].copy_from_slice(&x.to_le_bytes());
            }
        }
        self.done[i] = true;
        for n in std::mem::take(&mut self.needs[i]) {
            let left = self.uses.get_mut(&n).unwrap();
            *left -= 1;
            if *left == 0 {
                self.src.remove(&n);
            }
        }
        Ok(())
    }

    /// Starts writing `out` (the whole pack): the prefix and every tensor that needs no base weights.
    pub fn begin(&mut self, out: &mut [u8]) -> Result<(), String> {
        out[..self.prefix.len()].copy_from_slice(&self.prefix);
        for i in 0..self.plan.len() {
            if self.missing[i] == 0 && !self.done[i] {
                self.run_job(i, out)?;
            }
        }
        Ok(())
    }

    pub fn add_source(&mut self, name: &str, bytes: Vec<u8>, out: &mut [u8]) -> Result<(), String> {
        self.src.put(name, bytes)?;
        for i in self.readers.get(name).cloned().unwrap_or_default() {
            self.missing[i] -= 1;
            if self.missing[i] == 0 {
                self.run_job(i, out)?;
            }
        }
        Ok(())
    }

    pub fn finished(&self) -> bool {
        self.done.iter().all(|d| *d)
    }
}

/// Converts a checkpoint held in memory (the command line path), through the same stream.
pub fn convert(ck: &KevCheckpoint, block: usize, model: Value) -> Result<Vec<u8>, String> {
    let n = St::header_len(ck.base)?;
    let mut c = KevConvert::new(
        &ck.base[..n],
        ck.base_config,
        ck.base_tokenizer,
        ck.adapter,
        ck.adapter_config,
        ck.head,
        block,
        model,
    )?;
    let mut out = vec![0u8; c.total];
    c.begin(&mut out)?;
    for (name, at, len) in c.sources(n) {
        c.add_source(&name, ck.base[at..at + len].to_vec(), &mut out)?;
    }
    if !c.finished() {
        return Err("conversion ended with tensors still missing".into());
    }
    Ok(out)
}

enum Spec {
    Embed(String),
    /// Selected rows of an embedding or language-model head, kept in f32 for SemIf readout.
    Rows(String, Vec<usize>),
    /// zero-centred RMSNorm weight, stored as 1 + w
    PlusOne(String),
    Raw(String),
    NegExp(String),
    Conv(String),
    Head(&'static str),
    /// one merged matrix (f32)
    Merged(String),
    /// merged matrices stacked by rows (q8)
    Fused(Vec<String>),
}

struct Sources {
    base: Vec<St>,
    base_index: std::collections::HashMap<String, usize>,
    adapter: St,
    head: std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
    scaling: f32,
    pre: String,
}

impl Sources {
    fn validate_adapter(&self, plan: &[(String, Spec)], rank: usize) -> Result<(), String> {
        let modules: std::collections::HashSet<&str> = plan
            .iter()
            .flat_map(|(_, spec)| match spec {
                Spec::Merged(module) => vec![module.as_str()],
                Spec::Fused(modules) => modules.iter().map(String::as_str).collect(),
                _ => Vec::new(),
            })
            .collect();
        let entries = self.adapter.header.as_object().ok_or("adapter safetensors header must be an object")?;
        let mut pairs = std::collections::HashSet::new();
        for (name, _) in entries.iter().filter(|(name, _)| name != "__metadata__") {
            let module = name
                .strip_prefix("base_model.model.")
                .and_then(|name| name.strip_suffix(".lora_A.weight").or_else(|| name.strip_suffix(".lora_B.weight")))
                .ok_or_else(|| format!("unsupported pointer-adapter tensor {name}"))?;
            if !modules.contains(module) {
                return Err(format!("pointer-adapter tensor {name} targets a module this converter does not merge"));
            }
            pairs.insert(module);
        }
        if pairs.is_empty() {
            return Err("pointer adapter has no LoRA tensor pairs".into());
        }
        for module in pairs {
            let base = self.base_shape(&format!("{}{module}.weight", self.pre))?;
            let a = self.adapter.shape(&format!("base_model.model.{module}.lora_A.weight"))?;
            let b = self.adapter.shape(&format!("base_model.model.{module}.lora_B.weight"))?;
            if base.len() != 2 || a != [rank, base[1]] || b != [base[0], rank] {
                return Err(format!(
                    "{module}: LoRA shapes A={a:?}, B={b:?} do not match rank {rank} and base {base:?}"
                ));
            }
        }
        Ok(())
    }

    fn base_st(&self, name: &str) -> Result<&St, String> {
        let file = *self.base_index.get(name).ok_or_else(|| format!("checkpoint has no tensor {name}"))?;
        self.base.get(file).ok_or_else(|| format!("base shard {file} is missing"))
    }

    fn base_st_mut(&mut self, name: &str) -> Result<&mut St, String> {
        let file = *self.base_index.get(name).ok_or_else(|| format!("checkpoint has no tensor {name}"))?;
        self.base.get_mut(file).ok_or_else(|| format!("base shard {file} is missing"))
    }

    fn has_base(&self, name: &str) -> bool {
        self.base_index.contains_key(name)
    }

    fn base_shape(&self, name: &str) -> Result<Vec<usize>, String> {
        self.base_st(name)?.shape(name)
    }

    fn put(&mut self, name: &str, bytes: Vec<u8>) -> Result<(), String> {
        self.base_st_mut(name)?.parts.insert(name.to_string(), bytes);
        Ok(())
    }

    fn remove(&mut self, name: &str) {
        if let Some(&file) = self.base_index.get(name) {
            self.base[file].parts.remove(name);
        }
    }

    fn head_tensor(&self, k: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        self.head.get(k).cloned().ok_or_else(|| format!("head.pt: no {k}"))
    }

    /// Base tensors a spec reads.
    fn needs(&self, spec: &Spec) -> Vec<String> {
        match spec {
            Spec::Embed(n) | Spec::Rows(n, _) | Spec::PlusOne(n) | Spec::Raw(n) | Spec::NegExp(n) | Spec::Conv(n) => {
                vec![n.clone()]
            }
            Spec::Head(_) => Vec::new(),
            Spec::Merged(m) => vec![format!("{}{m}.weight", self.pre)],
            Spec::Fused(ms) => ms.iter().map(|m| format!("{}{m}.weight", self.pre)).collect(),
        }
    }

    /// Shape and whether the tensor is quantized.
    fn shape(&self, spec: &Spec) -> Result<(Vec<usize>, bool), String> {
        Ok(match spec {
            Spec::Embed(n) => {
                let s = self.base_shape(n)?;
                if s.len() != 2 {
                    return Err(format!("{n}: embedding must be rank 2, got {s:?}"));
                }
                (s, true)
            }
            Spec::Rows(n, rows) => {
                let s = self.base_shape(n)?;
                if s.len() != 2 || rows.iter().any(|&r| r >= s[0]) {
                    return Err(format!("{n}: selected readout rows do not fit shape {s:?}"));
                }
                (vec![rows.len(), s[1]], false)
            }
            Spec::PlusOne(n) | Spec::Raw(n) | Spec::NegExp(n) => {
                let s = self.base_shape(n)?;
                let numel = s
                    .iter()
                    .try_fold(1usize, |n, &d| n.checked_mul(d))
                    .ok_or_else(|| format!("{n}: shape overflows usize"))?;
                (vec![numel], false)
            }
            Spec::Conv(n) => {
                let s = self.base_shape(n)?;
                if s.len() != 3 || s[1] != 1 {
                    return Err(format!("{n}: convolution must have shape [channels, 1, kernel], got {s:?}"));
                }
                (vec![s[0], s[2]], false)
            }
            Spec::Head(k) => (self.head_tensor(k)?.1, false),
            Spec::Merged(m) => {
                let name = format!("{}{m}.weight", self.pre);
                let s = self.base_shape(&name)?;
                if s.len() != 2 {
                    return Err(format!("{name}: merged weight must be rank 2, got {s:?}"));
                }
                (s, false)
            }
            Spec::Fused(ms) => {
                let mut n: usize = 0;
                let mut k = None;
                for m in ms {
                    let name = format!("{}{m}.weight", self.pre);
                    let s = self.base_shape(&name)?;
                    if s.len() != 2 {
                        return Err(format!("{name}: fused weight must be rank 2, got {s:?}"));
                    }
                    if let Some(expected) = k {
                        if expected != s[1] {
                            return Err(format!("{name}: fused input width {} differs from {expected}", s[1]));
                        }
                    } else {
                        k = Some(s[1]);
                    }
                    n = n.checked_add(s[0]).ok_or_else(|| format!("{name}: fused row count overflows usize"))?;
                }
                (vec![n, k.ok_or("fused spec has no matrices")?], true)
            }
        })
    }

    /// Base matrix with the LoRA delta folded in: W + (alpha / r) B A, in f32.
    fn merged(&self, module: &str) -> Result<Vec<f32>, String> {
        let name = format!("{}{module}.weight", self.pre);
        let shape = self.base_shape(&name)?;
        if shape.len() != 2 {
            return Err(format!("{name}: merged weight must be rank 2, got {shape:?}"));
        }
        let (n, k) = (shape[0], shape[1]);
        let mut w = self.base_st(&name)?.f32(&name)?;
        let an = format!("base_model.model.{module}.lora_A.weight");
        if self.adapter.has(&an) {
            let a = self.adapter.f32(&an)?;
            let b = self.adapter.f32(&format!("base_model.model.{module}.lora_B.weight"))?;
            let a_shape = self.adapter.shape(&an)?;
            let b_shape = self.adapter.shape(&format!("base_model.model.{module}.lora_B.weight"))?;
            if a_shape.len() != 2
                || b_shape.len() != 2
                || a_shape[1] != k
                || b_shape[0] != n
                || b_shape[1] != a_shape[0]
            {
                return Err(format!("{module}: incompatible LoRA shapes A={a_shape:?}, B={b_shape:?}, base={shape:?}"));
            }
            let r = a_shape[0];
            for o in 0..n {
                let row = &mut w[o * k..(o + 1) * k];
                for j in 0..r {
                    let c = b[o * r + j] * self.scaling;
                    if c != 0.0 {
                        for (x, &av) in row.iter_mut().zip(&a[j * k..(j + 1) * k]) {
                            *x += c * av;
                        }
                    }
                }
            }
        }
        Ok(w)
    }

    fn data(&self, spec: &Spec) -> Result<Vec<f32>, String> {
        Ok(match spec {
            Spec::Embed(n) | Spec::Raw(n) | Spec::Conv(n) => self.base_st(n)?.f32(n)?,
            Spec::Rows(n, rows) => {
                let shape = self.base_shape(n)?;
                let width = shape[1];
                let values = self.base_st(n)?.f32(n)?;
                let mut selected = Vec::with_capacity(rows.len() * width);
                for &row in rows {
                    selected.extend_from_slice(&values[row * width..(row + 1) * width]);
                }
                selected
            }
            Spec::PlusOne(n) => self.base_st(n)?.f32(n)?.iter().map(|x| 1.0 + x).collect(),
            Spec::NegExp(n) => self.base_st(n)?.f32(n)?.iter().map(|x| -x.exp()).collect(),
            Spec::Head(k) => self.head_tensor(k)?.0,
            Spec::Merged(m) => self.merged(m)?,
            Spec::Fused(ms) => {
                let mut all = Vec::new();
                for m in ms {
                    all.extend(self.merged(m)?);
                }
                all
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointer_heads_require_all_four_exact_shapes_and_consume_every_tensor() {
        let tensor = |shape: Vec<usize>| {
            Py::Tensor(crate::torchpt::TensorRef {
                key: "0".into(),
                dtype: "FloatStorage".into(),
                offset: 0,
                stride: Vec::new(),
                shape,
            })
        };
        let head = vec![
            (Py::Str("q.weight".into()), tensor(vec![16, 32])),
            (Py::Str("k.weight".into()), tensor(vec![16, 32])),
            (Py::Str("q.bias".into()), tensor(vec![16])),
            (Py::Str("k.bias".into()), tensor(vec![16])),
        ];
        let root = |head: Vec<(Py, Py)>| {
            Py::Dict(vec![(Py::Str("head".into()), Py::Dict(head)), (Py::Str("temperature".into()), tensor(vec![]))])
        };
        assert_eq!(pointer_dimension(&root(head.clone()), 32).unwrap(), 16);
        assert!(pointer_dimension(&root(head.clone()), 64).unwrap_err().contains("q.weight"));
        for (index, shape) in [(1, vec![15, 32]), (2, vec![17]), (3, vec![16, 1])] {
            let mut invalid = head.clone();
            invalid[index].1 = tensor(shape);
            assert!(pointer_dimension(&root(invalid), 32).unwrap_err().contains("must have shape"));
        }
        let mut missing = head.clone();
        missing.pop();
        assert!(pointer_dimension(&root(missing), 32).unwrap_err().contains("missing k.bias"));
        let mut extra = head.clone();
        extra.push((Py::Str("unconsumed.weight".into()), tensor(vec![16, 32])));
        assert!(pointer_dimension(&root(extra), 32).unwrap_err().contains("unsupported head tensor"));
        let mut extra = root(head);
        let Py::Dict(entries) = &mut extra else { unreachable!() };
        entries.push((Py::Str("unconsumed".into()), tensor(vec![1])));
        assert!(pointer_dimension(&extra, 32).unwrap_err().contains("unconsumed top-level tensor"));
    }

    fn tiny_header(name: &str, start: usize, end: usize) -> Vec<u8> {
        let json = format!(r#"{{"{name}":{{"dtype":"F32","shape":[1],"data_offsets":[{start},{end}]}}}}"#);
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json.as_bytes());
        bytes
    }

    #[test]
    fn sharded_source_ranges_use_each_header_data_start_once() {
        let first = tiny_header("a", 2, 6);
        let second = tiny_header("long_tensor_name", 4, 8);
        let base = vec![St::header_only(&first).unwrap(), St::header_only(&second).unwrap()];
        let mut base_index = std::collections::HashMap::new();
        base_index.insert("a".to_string(), 0);
        base_index.insert("long_tensor_name".to_string(), 1);
        let mut uses = std::collections::HashMap::new();
        uses.insert("a".to_string(), 1);
        uses.insert("long_tensor_name".to_string(), 1);
        let converter = KevConvert {
            src: Sources {
                base,
                base_index,
                adapter: St::empty(),
                head: Default::default(),
                scaling: 0.0,
                pre: String::new(),
            },
            plan: Vec::new(),
            infos: Vec::new(),
            done: Vec::new(),
            uses,
            needs: Vec::new(),
            missing: Vec::new(),
            readers: Default::default(),
            block: 32,
            prefix: Vec::new(),
            total: 0,
        };
        let got = converter.source_ranges().unwrap();
        let first_data_start = first.len();
        let second_data_start = second.len();
        assert_eq!(
            got,
            vec![
                (0, "a".to_string(), first_data_start + 2, 4),
                (1, "long_tensor_name".to_string(), second_data_start + 4, 4),
            ]
        );
    }

    #[test]
    fn convolution_rejects_an_extra_middle_dimension() {
        let json = r#"{"conv":{"dtype":"F32","shape":[2,2,4],"data_offsets":[0,64]}}"#;
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json.as_bytes());
        let sources = Sources {
            base: vec![St::header_only(&bytes).unwrap()],
            base_index: [("conv".into(), 0)].into_iter().collect(),
            adapter: St::empty(),
            head: Default::default(),
            scaling: 0.0,
            pre: String::new(),
        };
        assert!(sources.shape(&Spec::Conv("conv".into())).unwrap_err().contains("[channels, 1, kernel]"));
    }

    #[test]
    fn pointer_adapters_reject_unconsumed_tensors_and_rank_mismatches() {
        let header = |text: &str| {
            let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(text.as_bytes());
            St::header_only(&bytes).unwrap()
        };
        let mut sources = Sources {
            base: vec![header(r#"{"model.layers.0.mlp.down_proj.weight":{"shape":[32,64]}}"#)],
            base_index: [("model.layers.0.mlp.down_proj.weight".into(), 0)].into_iter().collect(),
            adapter: header(
                r#"{"base_model.model.layers.0.mlp.down_proj.lora_A.weight":{"shape":[2,64]},"base_model.model.layers.0.mlp.down_proj.lora_B.weight":{"shape":[32,2]}}"#,
            ),
            head: Default::default(),
            scaling: 1.0,
            pre: "model.".into(),
        };
        let plan = vec![("down".into(), Spec::Merged("layers.0.mlp.down_proj".into()))];
        assert!(sources.validate_adapter(&plan, 2).is_ok());
        assert!(sources.validate_adapter(&plan, 4).unwrap_err().contains("do not match rank"));
        assert!(sources.validate_adapter(&[], 2).unwrap_err().contains("does not merge"));
        sources.adapter = header(r#"{"base_model.model.layers.0.mlp.down_proj.lora_A.weight":{"shape":[2,64]}}"#);
        assert!(sources.validate_adapter(&plan, 2).unwrap_err().contains("lora_B"));
    }
}
