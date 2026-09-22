//! Converts an upstream Laya checkpoint (`model.safetensors`, `encoder/config.json`,
//! `rl_agent_config.json`, `tokenizer/tokenizer.json`) into a `.kevala` pack.
//!
//! Every transformer matrix becomes symmetric int8 with one f32 scale per `block` weights of a
//! row. Norms, biases, the type embedding, the scorer and the act head stay f32: they are small,
//! and the scorer produces the logits the shipped temperatures were fitted on.

use crate::json::Value;
use crate::pack::Writer;
use crate::tokenizer::Tokenizer;

pub(crate) fn text_tensor_prefix(has: impl Fn(&str) -> bool) -> Result<&'static str, String> {
    let mut prefixes = ["model.language_model.", "language_model.", "model."]
        .into_iter()
        .filter(|prefix| has(&format!("{prefix}embed_tokens.weight")));
    let prefix = prefixes.next().ok_or("checkpoint has no supported text embedding tensor namespace")?;
    if prefixes.next().is_some() {
        return Err("checkpoint has ambiguous text embedding tensor namespaces".into());
    }
    Ok(prefix)
}

pub(crate) fn validate_tokenizer_vocab(tokens: usize, configured: usize, embedding_rows: usize) -> Result<(), String> {
    if tokens > configured || tokens > embedding_rows {
        return Err(format!("tokenizer ID range 0..{tokens} exceeds the configured vocabulary ({configured}) or embedding rows ({embedding_rows})"));
    }
    Ok(())
}

pub(crate) fn require_prompt_tokens(tokenizer: &Tokenizer, tokens: &[&str]) -> Result<(), String> {
    for &token in tokens {
        let id = tokenizer.token_id(token).ok_or_else(|| format!("tokenizer has no required prompt token {token}"))?;
        if tokenizer.encode(token) != [id] {
            return Err(format!("required prompt token {token} does not encode to its exact token ID"));
        }
    }
    Ok(())
}

pub struct Checkpoint<'a> {
    pub safetensors: &'a [u8],
    pub encoder_config: &'a str,
    pub agent_config: &'a str,
    pub tokenizer_json: &'a str,
}

#[derive(Clone, Debug)]
pub struct Options {
    pub block: usize,
    /// Pack tensor name prefixes to keep in f32 instead of int8.
    pub keep_f32: Vec<String>,
    /// Provenance recorded in the pack header.
    pub model: Value,
}

struct Safetensors {
    header: Value,
}

impl Safetensors {
    fn parse(b: &[u8]) -> Result<Safetensors, String> {
        if b.len() < 8 {
            return Err("safetensors file too short".into());
        }
        let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
        let text = std::str::from_utf8(b.get(8..8 + n).ok_or("truncated safetensors header")?)
            .map_err(|_| "safetensors header is not UTF-8")?;
        Ok(Safetensors { header: Value::parse(text).map_err(|e| format!("safetensors header: {e}"))? })
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
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = match (exp, mant) {
        (0, 0) => sign,
        (0, m) => {
            // subnormal: renormalize
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (31, m) => sign | (0xff << 23) | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// Symmetric absmax int8 per `block` columns of each row.
pub fn quantize(w: &[f32], rows: usize, cols: usize, block: usize) -> (Vec<i8>, Vec<f32>) {
    assert!(cols % block == 0);
    let nb = cols / block;
    let mut q = vec![0i8; rows * cols];
    let mut scales = vec![0f32; rows * nb];
    for r in 0..rows {
        for b in 0..nb {
            let s = &w[r * cols + b * block..r * cols + (b + 1) * block];
            let amax = s.iter().fold(0f32, |m, v| m.max(v.abs()));
            let scale = amax / 127.0;
            scales[r * nb + b] = scale;
            if scale > 0.0 {
                let inv = 1.0 / scale;
                for (i, v) in s.iter().enumerate() {
                    q[r * cols + b * block + i] = (v * inv).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
    }
    (q, scales)
}

fn num(v: f64) -> Value {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        Value::Int((v as i64).to_string())
    } else {
        Value::Float(v)
    }
}

fn rope_theta(enc: &Value, kind: &str, legacy: &str) -> Result<f64, String> {
    // transformers 5 writes rope_parameters per layer type, 4.x wrote flat keys
    if let Some(t) = enc.get("rope_parameters").and_then(|r| r.get(kind)).and_then(|r| r.get("rope_theta")) {
        return t.as_f64().ok_or_else(|| format!("bad rope_theta for {kind}"));
    }
    enc.get(legacy).and_then(Value::as_f64).ok_or_else(|| format!("encoder config: no {legacy}"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SrcType {
    F32,
    F16,
    BF16,
}

/// One source tensor to widen or quantize into its place in the pack.
#[derive(Clone, Debug)]
pub struct Job {
    pub src: String,
    /// Byte range in the safetensors file (absolute, header included).
    pub src_offset: usize,
    pub src_len: usize,
    pub src_type: SrcType,
    pub rows: usize,
    pub cols: usize,
    pub q8: bool,
    pub dst_offset: usize,
    pub dst_scales_offset: usize,
}

/// Everything needed to build a pack while the checkpoint streams in: the leading bytes, one
/// job per tensor in file order, and the final size.
pub struct Plan {
    pub prefix: Vec<u8>,
    pub jobs: Vec<Job>,
    pub total: usize,
    pub block: usize,
}

/// Length of the safetensors header, from the file's first 8 bytes.
pub fn safetensors_header_len(first8: &[u8]) -> Result<usize, String> {
    let b: [u8; 8] = first8.get(..8).ok_or("safetensors file too short")?.try_into().unwrap();
    Ok(u64::from_le_bytes(b) as usize)
}

/// `st_head` is the safetensors file from byte 0 through the end of its JSON header.
pub fn plan(
    st_head: &[u8],
    encoder_config: &str,
    agent_config: &str,
    tokenizer_json: &str,
    opt: &Options,
) -> Result<Plan, String> {
    let st = Safetensors::parse(st_head)?;
    let data_start = 8 + safetensors_header_len(st_head)?;
    let enc = Value::parse(encoder_config).map_err(|e| format!("encoder config: {e}"))?;
    let agent = Value::parse(agent_config).map_err(|e| format!("rl_agent_config: {e}"))?;
    let tok = Tokenizer::from_hf_json(tokenizer_json)?;

    let u = |k: &str| enc.get(k).and_then(Value::as_usize).ok_or_else(|| format!("encoder config: no {k}"));
    let hidden = u("hidden_size")?;
    let embedding = st.shape("encoder.embeddings.tok_embeddings.weight")?;
    if embedding != [u("vocab_size")?, hidden] {
        return Err(format!("encoder embedding shape {embedding:?} does not match [vocab_size, hidden_size]"));
    }
    validate_tokenizer_vocab(tok.vocab_size(), u("vocab_size")?, embedding[0])?;
    let layers = u("num_hidden_layers")?;
    let inter = u("intermediate_size")?;
    let head_layers = agent.get("head_layers").and_then(Value::as_usize).unwrap_or(2);
    let head_ff = st.shape("head.layers.0.linear1.weight")?[0];
    let act_hidden = st.shape("act_head.0.weight")?[0];
    // block 0 keeps every matrix in f32: a reference pack for measuring quantization error
    let exact = opt.block == 0;
    let block = if exact { 32 } else { opt.block };
    if block % 16 != 0 {
        return Err(format!("block {block} must be a multiple of 16"));
    }
    for (what, k) in [("hidden", hidden), ("intermediate", inter), ("head ff", head_ff)] {
        if k % block != 0 {
            return Err(format!("{what} size {k} is not a multiple of block {block}"));
        }
    }

    let mut config = vec![
        ("hidden_size".to_string(), num(hidden as f64)),
        ("num_attention_heads".into(), num(u("num_attention_heads")? as f64)),
        ("num_hidden_layers".into(), num(layers as f64)),
        ("intermediate_size".into(), num(inter as f64)),
        ("global_attn_every_n_layers".into(), num(u("global_attn_every_n_layers")? as f64)),
        ("local_attention".into(), num(u("local_attention")? as f64)),
        ("global_rope_theta".into(), num(rope_theta(&enc, "full_attention", "global_rope_theta")?)),
        ("local_rope_theta".into(), num(rope_theta(&enc, "sliding_attention", "local_rope_theta")?)),
        ("norm_eps".into(), Value::Float(enc.get("norm_eps").and_then(Value::as_f64).unwrap_or(1e-5))),
        ("vocab_size".into(), num(u("vocab_size")? as f64)),
        ("head_layers".into(), num(head_layers as f64)),
        ("head_ff".into(), num(head_ff as f64)),
        ("head_heads".into(), num((hidden / 64).max(1) as f64)),
        ("head_norm_eps".into(), Value::Float(1e-5)),
        ("act_hidden".into(), num(act_hidden as f64)),
        ("max_len".into(), num(agent.get("max_len").and_then(Value::as_f64).unwrap_or(512.0))),
        ("head_max_len".into(), num(agent.get("head_max_len").and_then(Value::as_f64).unwrap_or(192.0))),
        ("block".into(), num(block as f64)),
    ];
    for k in ["temperature", "temperature_by_options"] {
        if let Some(v) = agent.get(k) {
            config.push((k.to_string(), v.clone()));
        }
    }

    // (source name, pack name, quantize)
    let mut names: Vec<(String, String, bool)> = Vec::new();
    let mut add = |src: &str, dst: &str, q: bool| names.push((src.to_string(), dst.to_string(), q));
    // coordinator tensors first, so a streaming loader can start tokenizing early
    add("encoder.embeddings.tok_embeddings.weight", "emb", true);
    add("encoder.embeddings.norm.weight", "emb_norm", false);
    add("encoder.final_norm.weight", "final_norm", false);
    add("type_emb.weight", "type_emb", false);
    for (src, dst) in [
        ("scorer.0", "scorer.norm"),
        ("scorer.1", "scorer.l1"),
        ("scorer.3", "scorer.l2"),
        ("act_head.0", "act.l1"),
        ("act_head.2", "act.l2"),
    ] {
        let (w, b) = if dst == "scorer.norm" {
            (format!("{dst}.w"), format!("{dst}.b"))
        } else {
            (dst.to_string(), format!("{dst}.b"))
        };
        add(&format!("{src}.weight"), &w, false);
        add(&format!("{src}.bias"), &b, false);
    }
    for i in 0..layers {
        let p = format!("encoder.layers.{i}");
        if i > 0 {
            add(&format!("{p}.attn_norm.weight"), &format!("enc.{i}.attn_norm"), false);
        }
        add(&format!("{p}.attn.Wqkv.weight"), &format!("enc.{i}.wqkv"), true);
        add(&format!("{p}.attn.Wo.weight"), &format!("enc.{i}.wo"), true);
        add(&format!("{p}.mlp_norm.weight"), &format!("enc.{i}.mlp_norm"), false);
        add(&format!("{p}.mlp.Wi.weight"), &format!("enc.{i}.wi"), true);
        add(&format!("{p}.mlp.Wo.weight"), &format!("enc.{i}.wo2"), true);
    }
    for i in 0..head_layers {
        let p = format!("head.layers.{i}");
        for (src, dst, q) in [
            ("norm1.weight", "norm1.w", false),
            ("norm1.bias", "norm1.b", false),
            ("self_attn.in_proj_weight", "in_proj", true),
            ("self_attn.in_proj_bias", "in_proj.b", false),
            ("self_attn.out_proj.weight", "out_proj", true),
            ("self_attn.out_proj.bias", "out_proj.b", false),
            ("norm2.weight", "norm2.w", false),
            ("norm2.bias", "norm2.b", false),
            ("linear1.weight", "lin1", true),
            ("linear1.bias", "lin1.b", false),
            ("linear2.weight", "lin2", true),
            ("linear2.bias", "lin2.b", false),
        ] {
            add(&format!("{p}.{src}"), &format!("head.{i}.{dst}"), q);
        }
    }

    let mut w = Writer::new(opt.model.clone(), Value::Object(config), tok.to_bytes());
    let mut srcs = Vec::new();
    for (src, dst, q) in &names {
        let shape = st.shape(src)?;
        let (rows, cols) = if shape.len() == 2 { (shape[0], shape[1]) } else { (1, shape.iter().product()) };
        let keep = opt.keep_f32.iter().any(|p| dst.starts_with(p.as_str()));
        let q = &(*q && !exact && !keep);
        if *q {
            w.add_q8(dst, rows, cols, block);
        } else {
            w.add_f32(dst, &shape);
        }
        let t = st.header.get(src).unwrap();
        let src_type = match t.get("dtype").and_then(Value::as_str) {
            Some("F32") => SrcType::F32,
            Some("F16") => SrcType::F16,
            Some("BF16") => SrcType::BF16,
            other => return Err(format!("{src}: unsupported dtype {other:?}")),
        };
        let off = t.get("data_offsets").and_then(Value::as_array).ok_or_else(|| format!("{src}: no offsets"))?;
        let (a, b) = (off[0].as_usize().unwrap_or(0), off[1].as_usize().unwrap_or(0));
        let elem = if src_type == SrcType::F32 { 4 } else { 2 };
        if b - a != rows * cols * elem {
            return Err(format!("{src}: {} bytes for shape {shape:?}", b - a));
        }
        srcs.push((src.clone(), data_start + a, b - a, src_type, rows, cols, *q));
    }
    let (prefix, infos, total) = w.layout();
    let mut jobs: Vec<Job> = srcs
        .into_iter()
        .zip(infos)
        .map(|((src, src_offset, src_len, src_type, rows, cols, q8), info)| Job {
            src,
            src_offset,
            src_len,
            src_type,
            rows,
            cols,
            q8,
            dst_offset: info.offset,
            dst_scales_offset: info.scales_offset,
        })
        .collect();
    // file order, so jobs complete as the checkpoint streams by
    jobs.sort_by_key(|j| j.src_offset);
    Ok(Plan { prefix, jobs, total, block })
}

/// Widens (or quantizes) one tensor's source bytes into `dst`, the whole pack buffer.
pub fn run_job(job: &Job, src: &[u8], block: usize, dst: &mut [u8]) {
    let v: Vec<f32> = match job.src_type {
        SrcType::F32 => src.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        SrcType::F16 => src.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        SrcType::BF16 => {
            src.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
        }
    };
    if job.q8 {
        let (q, s) = quantize(&v, job.rows, job.cols, block);
        for (d, q) in dst[job.dst_offset..job.dst_offset + q.len()].iter_mut().zip(q) {
            *d = q as u8;
        }
        for (i, sc) in s.iter().enumerate() {
            dst[job.dst_scales_offset + i * 4..job.dst_scales_offset + i * 4 + 4].copy_from_slice(&sc.to_le_bytes());
        }
    } else {
        for (i, x) in v.iter().enumerate() {
            dst[job.dst_offset + i * 4..job.dst_offset + i * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
    }
}

/// Converts a checkpoint held in memory.
pub fn convert(ck: &Checkpoint, opt: &Options) -> Result<Vec<u8>, String> {
    let head_len = 8 + safetensors_header_len(ck.safetensors)?;
    let p = plan(&ck.safetensors[..head_len], ck.encoder_config, ck.agent_config, ck.tokenizer_json, opt)?;
    let mut out = vec![0u8; p.total];
    out[..p.prefix.len()].copy_from_slice(&p.prefix);
    for j in &p.jobs {
        run_job(j, &ck.safetensors[j.src_offset..j.src_offset + j.src_len], p.block, &mut out);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_edges() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 5.960464477539063e-8);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn quantize_is_symmetric_absmax() {
        let w = [0.5, -1.0, 0.25, 0.0, 2.0, -2.0, 1.0, 0.0];
        let (q, s) = quantize(&w, 2, 4, 4);
        assert_eq!(s, vec![1.0 / 127.0, 2.0 / 127.0]);
        assert_eq!(q, vec![64, -127, 32, 0, 127, -127, 64, 0]);
    }
}

#[cfg(test)]
mod prefix_tests {
    use super::text_tensor_prefix;

    #[test]
    fn text_namespace_is_selected_from_tensors_and_rejects_ambiguity() {
        for prefix in ["model.", "model.language_model.", "language_model."] {
            let tensor = format!("{prefix}embed_tokens.weight");
            assert_eq!(text_tensor_prefix(|name| name == tensor).unwrap(), prefix);
        }
        assert!(text_tensor_prefix(|_| false).unwrap_err().contains("no supported"));
        assert!(text_tensor_prefix(|_| true).unwrap_err().contains("ambiguous"));
    }

    #[test]
    fn tokenizer_bounds_allow_padded_embeddings_and_reject_out_of_range_ids() {
        assert!(super::validate_tokenizer_vocab(248077, 248320, 248320).is_ok());
        assert!(super::validate_tokenizer_vocab(248321, 248320, 248320).is_err());
        assert!(super::validate_tokenizer_vocab(248077, 248320, 248076).is_err());
    }
}
