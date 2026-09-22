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

/// A safetensors file whose tensors arrive one by one: the header up front, each tensor's bytes
/// when the stream reaches it (or all at once for a file already in memory).
struct St {
    header: Value,
    whole: Vec<u8>,
    parts: std::collections::HashMap<String, Vec<u8>>,
}

impl St {
    /// `head` is the file from byte 0 through at least the end of its JSON header.
    fn header_only(head: &[u8]) -> Result<St, String> {
        let n = u64::from_le_bytes(head.get(..8).ok_or("short safetensors")?.try_into().unwrap()) as usize;
        let text = std::str::from_utf8(head.get(8..8 + n).ok_or("truncated safetensors header")?)
            .map_err(|_| "safetensors header is not UTF-8")?;
        Ok(St { header: Value::parse(text).map_err(|e| e.to_string())?, whole: Vec::new(), parts: Default::default() })
    }
    fn whole(b: &[u8]) -> Result<St, String> {
        let mut s = St::header_only(b)?;
        let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
        s.whole = b[8 + n..].to_vec();
        Ok(s)
    }
    fn has(&self, name: &str) -> bool {
        self.header.get(name).is_some()
    }
    fn shape(&self, name: &str) -> Result<Vec<usize>, String> {
        let t = self.header.get(name).ok_or_else(|| format!("checkpoint has no tensor {name}"))?;
        Ok(t.get("shape").and_then(Value::as_array).ok_or("no shape")?.iter().filter_map(Value::as_usize).collect())
    }
    fn range(&self, name: &str) -> Result<(usize, usize), String> {
        let t = self.header.get(name).ok_or_else(|| format!("checkpoint has no tensor {name}"))?;
        let off = t.get("data_offsets").and_then(Value::as_array).ok_or("no offsets")?;
        Ok((off[0].as_usize().unwrap_or(0), off[1].as_usize().unwrap_or(0)))
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
        Ok(match t.get("dtype").and_then(Value::as_str) {
            Some("F32") => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            Some("BF16") => {
                raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
            }
            Some("F16") => raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            other => return Err(format!("{name}: unsupported dtype {other:?}")),
        })
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
    /// `base_head` is the base safetensors file from byte 0 through the end of its header.
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
        let base = St::header_only(base_head)?;
        let adapter = St::whole(adapter)?;
        let ck = KevCheckpoint { base: &[], base_config, base_tokenizer, adapter: &[], adapter_config, head: head_pt };
        let cfg_all = Value::parse(ck.base_config).map_err(|e| format!("base config: {e}"))?;
        let cfg = cfg_all.get("text_config").unwrap_or(&cfg_all);
        let acfg = Value::parse(ck.adapter_config).map_err(|e| format!("adapter config: {e}"))?;
        let head = TorchFile::parse(ck.head)?;
        let tok = Tokenizer::from_hf_json(ck.base_tokenizer)?;

        let u = |k: &str| cfg.get(k).and_then(Value::as_usize).ok_or_else(|| format!("base config: no {k}"));
        let hidden = u("hidden_size")?;
        let layer_types: Vec<String> = cfg
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or("no layer_types")?
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        let rope = cfg.get("rope_parameters").ok_or("no rope_parameters")?;
        let head_dim = u("head_dim")?;
        let rotary =
            (head_dim as f64 * rope.get("partial_rotary_factor").and_then(Value::as_f64).unwrap_or(1.0)) as usize;
        let r = acfg.get("r").and_then(Value::as_f64).ok_or("adapter config: no r")?;
        let alpha = acfg.get("lora_alpha").and_then(Value::as_f64).ok_or("adapter config: no lora_alpha")?;
        let scaling = (alpha / r) as f32;
        let temperature = head.root.get("temperature").and_then(Py::as_f64).unwrap_or(1.0);
        let ptr_dim = match head.root.get("head").and_then(|h| h.get("q.weight")) {
            Some(Py::Tensor(t)) => t.shape[0],
            _ => return Err("head.pt has no head q.weight".into()),
        };
        let ids: Vec<Value> = SPECIAL
            .iter()
            .map(|s| tok.token_id(s).map(|i| Value::Int(i.to_string())).ok_or_else(|| format!("tokenizer has no {s}")))
            .collect::<Result<_, _>>()?;

        let config = Value::Object(vec![
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
            ("linear_num_key_heads".into(), num(u("linear_num_key_heads")? as f64)),
            ("linear_num_value_heads".into(), num(u("linear_num_value_heads")? as f64)),
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
                    ("type".into(), Value::Str("kev".into())),
                    ("tokens".into(), Value::Array(SPECIAL.iter().map(|s| Value::Str(s.to_string())).collect())),
                    ("token_ids".into(), Value::Array(ids)),
                ]),
            ),
        ]);

        if u("linear_num_key_heads")? != u("linear_num_value_heads")? {
            return Err("DeltaNet layers with grouped key heads are not supported yet".into());
        }

        let mut head_tensors = std::collections::HashMap::new();
        if let Some(Py::Dict(m)) = head.root.get("head") {
            for (k, v) in m {
                if let (Py::Str(k), Py::Tensor(t)) = (k, v) {
                    head_tensors.insert(k.clone(), (head.f32(t)?, t.shape.clone()));
                }
            }
        }
        let mut w = Writer::new(model, config, tok.to_bytes());
        let pre = "model.language_model.";
        // what each pack tensor is made of, in stream order
        let mut plan: Vec<(String, Spec)> = vec![
            ("emb".into(), Spec::Embed(format!("{pre}embed_tokens.weight"))),
            ("norm".into(), Spec::PlusOne(format!("{pre}norm.weight"))),
        ];
        for (dst, src) in [("ptr.q", "q.weight"), ("ptr.q.b", "q.bias"), ("ptr.k", "k.weight"), ("ptr.k.b", "k.bias")] {
            plan.push((dst.into(), Spec::Head(src)));
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
            } else {
                plan.push((format!("L.{i}.qkvz"), fused(&["in_proj_qkv", "in_proj_z"], "linear_attn")));
                plan.push((format!("L.{i}.a"), Spec::Merged(format!("{l}.linear_attn.in_proj_a"))));
                plan.push((format!("L.{i}.b"), Spec::Merged(format!("{l}.linear_attn.in_proj_b"))));
                plan.push((format!("L.{i}.conv"), Spec::Conv(format!("{pre}{l}.linear_attn.conv1d.weight"))));
                plan.push((format!("L.{i}.dt_bias"), Spec::Raw(format!("{pre}{l}.linear_attn.dt_bias"))));
                plan.push((format!("L.{i}.neg_a"), Spec::NegExp(format!("{pre}{l}.linear_attn.A_log"))));
                plan.push((format!("L.{i}.gnorm"), Spec::Raw(format!("{pre}{l}.linear_attn.norm.weight"))));
                plan.push((format!("L.{i}.out"), fused(&["out_proj"], "linear_attn")));
            }
            plan.push((format!("L.{i}.gate_up"), fused(&["gate_proj", "up_proj"], "mlp")));
            plan.push((format!("L.{i}.down"), fused(&["down_proj"], "mlp")));
        }

        let src = Sources { base, adapter, head: head_tensors, scaling, pre: pre.to_string() };
        for (name, spec) in &plan {
            match src.shape(spec)? {
                (shape, true) => w.add_q8(name, shape[0], shape[1], block),
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

    /// Base tensors to feed, `(name, absolute byte offset, length)`, in file order.
    pub fn sources(&self, header_len: usize) -> Vec<(String, usize, usize)> {
        let mut v: Vec<(String, usize, usize)> = self
            .uses
            .keys()
            .filter_map(|n| self.src.base.range(n).ok().map(|(a, b)| (n.clone(), 8 + header_len + a, b - a)))
            .collect();
        v.sort_by_key(|x| x.1);
        v
    }

    /// Writes pack tensor `i` and frees the source tensors nothing else still needs.
    fn run_job(&mut self, i: usize, out: &mut [u8]) -> Result<(), String> {
        let v = self.src.data(&self.plan[i].1)?;
        let info = &self.infos[i];
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
                self.src.base.parts.remove(&n);
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
        self.src.base.parts.insert(name.to_string(), bytes);
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
    let n = u64::from_le_bytes(ck.base.get(..8).ok_or("short safetensors")?.try_into().unwrap()) as usize;
    let mut c = KevConvert::new(
        &ck.base[..8 + n],
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
    base: St,
    adapter: St,
    head: std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
    scaling: f32,
    pre: String,
}

impl Sources {
    fn head_tensor(&self, k: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        self.head.get(k).cloned().ok_or_else(|| format!("head.pt: no {k}"))
    }

    /// Base tensors a spec reads.
    fn needs(&self, spec: &Spec) -> Vec<String> {
        match spec {
            Spec::Embed(n) | Spec::PlusOne(n) | Spec::Raw(n) | Spec::NegExp(n) | Spec::Conv(n) => vec![n.clone()],
            Spec::Head(_) => Vec::new(),
            Spec::Merged(m) => vec![format!("{}{m}.weight", self.pre)],
            Spec::Fused(ms) => ms.iter().map(|m| format!("{}{m}.weight", self.pre)).collect(),
        }
    }

    /// Shape and whether the tensor is quantized.
    fn shape(&self, spec: &Spec) -> Result<(Vec<usize>, bool), String> {
        Ok(match spec {
            Spec::Embed(n) => (self.base.shape(n)?, true),
            Spec::PlusOne(n) | Spec::Raw(n) | Spec::NegExp(n) => (vec![self.base.shape(n)?.iter().product()], false),
            Spec::Conv(n) => {
                let s = self.base.shape(n)?;
                (vec![s[0], s[2]], false)
            }
            Spec::Head(k) => (self.head_tensor(k)?.1, false),
            Spec::Merged(m) => (self.base.shape(&format!("{}{m}.weight", self.pre))?, false),
            Spec::Fused(ms) => {
                let mut n = 0;
                let mut k = 0;
                for m in ms {
                    let s = self.base.shape(&format!("{}{m}.weight", self.pre))?;
                    n += s[0];
                    k = s[1];
                }
                (vec![n, k], true)
            }
        })
    }

    /// Base matrix with the LoRA delta folded in: W + (alpha / r) B A, in f32.
    fn merged(&self, module: &str) -> Result<Vec<f32>, String> {
        let name = format!("{}{module}.weight", self.pre);
        let shape = self.base.shape(&name)?;
        let (n, k) = (shape[0], shape[1]);
        let mut w = self.base.f32(&name)?;
        let an = format!("base_model.model.{module}.lora_A.weight");
        if self.adapter.has(&an) {
            let a = self.adapter.f32(&an)?;
            let b = self.adapter.f32(&format!("base_model.model.{module}.lora_B.weight"))?;
            let r = self.adapter.shape(&an)?[0];
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
            Spec::Embed(n) | Spec::Raw(n) | Spec::Conv(n) => self.base.f32(n)?,
            Spec::PlusOne(n) => self.base.f32(n)?.iter().map(|x| 1.0 + x).collect(),
            Spec::NegExp(n) => self.base.f32(n)?.iter().map(|x| -x.exp()).collect(),
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
