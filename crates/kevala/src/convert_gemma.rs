//! Converts a Gemma 4 text checkpoint into a text-only kevala pack.
//!
//! Gemma 4's published checkpoint is a multimodal conditional-generation model, but the
//! decision runtime only needs its text backbone. This converter therefore keeps the text
//! weights and tokenizer, records the supported modality as text, and deliberately does not
//! copy the vision or audio towers. The source checkpoint is streamed in bounded row chunks;
//! in particular, embed_tokens_per_layer is split into one pack tensor per decoder layer so
//! converting it never requires a multi-gigabyte temporary f32 buffer.

use crate::convert::{f16_to_f32, quantize};
use crate::json::Value;
use crate::pack::{TensorInfo, Writer};
use crate::tokenizer::Tokenizer;
use std::collections::{BTreeMap, HashMap, HashSet};

/// The direct-option readout has the same sixteen answer slots used by SemIf.
pub const LABELS: &str = "ABCDEFGHIJKLMNOP";

/// Keep source reads comfortably below the size of a large model tensor. Quantization is done
/// one job at a time, so this also bounds the temporary f32/q8 buffers.
const CHUNK_BYTES: usize = 32 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SrcDType {
    F32,
    F16,
    Bf16,
}

impl SrcDType {
    fn bytes(self) -> usize {
        match self {
            SrcDType::F32 => 4,
            SrcDType::F16 | SrcDType::Bf16 => 2,
        }
    }
}

#[derive(Clone, Debug)]
struct TensorMeta {
    shard: usize,
    dtype: SrcDType,
    shape: Vec<usize>,
    offset: usize,
}

#[derive(Clone, Debug)]
struct MatrixSpec {
    output: usize,
    source: String,
    source_row: usize,
    rows: usize,
    source_cols: usize,
    source_col: usize,
    output_row: usize,
    cols: usize,
    q8: bool,
}

#[derive(Clone, Debug)]
struct VectorSpec {
    output: usize,
    source: String,
}

#[derive(Clone, Debug)]
struct OutputChunk {
    output: usize,
    output_row: usize,
    rows: usize,
    source_col: usize,
    cols: usize,
    q8: bool,
}

#[derive(Clone, Debug)]
struct Job {
    name: String,
    shard: usize,
    offset: usize,
    len: usize,
    dtype: SrcDType,
    source_cols: usize,
    source_rows: usize,
    outputs: Vec<OutputChunk>,
}

#[derive(Clone, Debug)]
struct Header {
    tensors: HashMap<String, TensorMeta>,
}

fn num(v: f64) -> Value {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        Value::Int((v as i64).to_string())
    } else {
        Value::Float(v)
    }
}

fn bf16_round(value: f32) -> f32 {
    f32::from_bits((value.to_bits().wrapping_add(0x8000)) & 0xffff_0000)
}

fn usize_field(v: &Value, key: &str) -> Result<usize, String> {
    v.get(key).and_then(Value::as_usize).ok_or_else(|| format!("Gemma text config: missing {key}"))
}

fn bool_field(v: &Value, key: &str) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn parse_header(head: &[u8], shard: usize) -> Result<Header, String> {
    let n = u64::from_le_bytes(
        head.get(..8).ok_or("safetensors header is shorter than its length prefix")?.try_into().unwrap(),
    );
    let n = usize::try_from(n).map_err(|_| "safetensors header length does not fit usize")?;
    let data_start = 8usize.checked_add(n).ok_or("safetensors header length overflows usize")?;
    let text = std::str::from_utf8(head.get(8..data_start).ok_or("truncated safetensors header")?)
        .map_err(|_| "safetensors header is not UTF-8")?;
    let root = Value::parse(text).map_err(|e| format!("safetensors header: {e}"))?;
    let mut tensors = HashMap::new();
    for (name, entry) in root.as_object().ok_or("safetensors header is not an object")? {
        if name == "__metadata__" {
            continue;
        }
        let dtype = match entry.get("dtype").and_then(Value::as_str) {
            Some("F32") => SrcDType::F32,
            Some("F16") => SrcDType::F16,
            Some("BF16") => SrcDType::Bf16,
            Some(other) => return Err(format!("{name}: unsupported source dtype {other}")),
            None => return Err(format!("{name}: safetensors entry has no dtype")),
        };
        let shape = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{name}: safetensors entry has no shape"))?
            .iter()
            .map(|d| d.as_usize().ok_or_else(|| format!("{name}: invalid shape")))
            .collect::<Result<Vec<_>, _>>()?;
        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{name}: safetensors entry has no data_offsets"))?;
        if offsets.len() != 2 {
            return Err(format!("{name}: data_offsets must contain two values"));
        }
        let start = offsets[0].as_usize().ok_or_else(|| format!("{name}: invalid start offset"))?;
        let end = offsets[1].as_usize().ok_or_else(|| format!("{name}: invalid end offset"))?;
        if end < start {
            return Err(format!("{name}: data_offsets are reversed"));
        }
        let elements = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| format!("{name}: shape overflows usize"))?;
        let expected =
            elements.checked_mul(dtype.bytes()).ok_or_else(|| format!("{name}: byte size overflows usize"))?;
        if end - start != expected {
            return Err(format!("{name}: data_offsets cover {} bytes, expected {expected}", end - start));
        }
        let offset = data_start.checked_add(start).ok_or_else(|| format!("{name}: source offset overflows usize"))?;
        if tensors.insert(name.clone(), TensorMeta { shard, dtype, shape, offset }).is_some() {
            return Err(format!("duplicate tensor {name} across safetensors shards"));
        }
    }
    Ok(Header { tensors })
}

fn source_to_f32(dtype: SrcDType, bytes: &[u8], index: usize) -> Result<f32, String> {
    let at = index.checked_mul(dtype.bytes()).ok_or("source index overflows usize")?;
    let b = bytes.get(at..at + dtype.bytes()).ok_or("source range is shorter than its declared shape")?;
    Ok(match dtype {
        SrcDType::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        SrcDType::F16 => f16_to_f32(u16::from_le_bytes([b[0], b[1]])),
        SrcDType::Bf16 => f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16),
    })
}

fn add_f32(dst: &mut [u8], offset: usize, values: &[f32]) -> Result<(), String> {
    let end = offset.checked_add(values.len() * 4).ok_or("destination offset overflows usize")?;
    let out = dst.get_mut(offset..end).ok_or("destination is shorter than the declared tensor")?;
    for (slot, value) in out.chunks_exact_mut(4).zip(values) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

fn add_q8_output(
    writer: &mut Writer,
    outputs: &mut Vec<String>,
    name: String,
    rows: usize,
    cols: usize,
    block: usize,
) -> usize {
    let index = outputs.len();
    writer.add_q8(&name, rows, cols, block);
    outputs.push(name);
    index
}

fn add_f32_output(writer: &mut Writer, outputs: &mut Vec<String>, name: String, shape: &[usize]) -> usize {
    let index = outputs.len();
    writer.add_f32(&name, shape);
    outputs.push(name);
    index
}

/// A streaming converter for Gemma 4's text backbone.
pub struct GemmaConvert {
    jobs: Vec<Job>,
    by_name: HashMap<String, usize>,
    infos: Vec<TensorInfo>,
    prefix: Vec<u8>,
    pub total: usize,
    block: usize,
    done: Vec<bool>,
}

impl GemmaConvert {
    /// Build a plan from safetensors headers. The bodies are read later through source_ranges
    /// and add_source, so this method does not materialize any model weights.
    pub fn new_sharded(
        base_heads: &[&[u8]],
        base_config: &str,
        base_tokenizer: &str,
        block: usize,
        model: Value,
    ) -> Result<GemmaConvert, String> {
        if base_heads.is_empty() {
            return Err("no Gemma safetensors shards".into());
        }
        if block != 32 {
            return Err("Gemma conversion currently requires quantization block 32".into());
        }
        let mut catalog = HashMap::new();
        for (shard, head) in base_heads.iter().enumerate() {
            let parsed = parse_header(head, shard)?;
            for (name, tensor) in parsed.tensors {
                if catalog.insert(name.clone(), tensor).is_some() {
                    return Err(format!("duplicate tensor {name} across safetensors shards"));
                }
            }
        }
        let top = Value::parse(base_config).map_err(|e| format!("Gemma config: {e}"))?;
        let cfg = top.get("text_config").unwrap_or(&top);
        if cfg.get("model_type").and_then(Value::as_str) != Some("gemma4_text") {
            return Err("Gemma config text_config.model_type must be gemma4_text".into());
        }
        if bool_field(cfg, "enable_moe_block")
            || cfg.get("num_experts").is_some_and(|v| !v.is_null())
            || cfg.get("top_k_experts").is_some_and(|v| !v.is_null())
        {
            return Err("Gemma converter currently accepts dense Gemma 4 text checkpoints only".into());
        }
        let hidden = usize_field(cfg, "hidden_size")?;
        let intermediate = usize_field(cfg, "intermediate_size")?;
        let layers = usize_field(cfg, "num_hidden_layers")?;
        let heads = usize_field(cfg, "num_attention_heads")?;
        let kv_heads = usize_field(cfg, "num_key_value_heads")?;
        let head_dim = usize_field(cfg, "head_dim")?;
        let global_head_dim = usize_field(cfg, "global_head_dim")?;
        let ple_dim = usize_field(cfg, "hidden_size_per_layer_input")?;
        let vocab = usize_field(cfg, "vocab_size")?;
        let ple_vocab = usize_field(cfg, "vocab_size_per_layer_input")?;
        let shared_layers = usize_field(cfg, "num_kv_shared_layers")?;
        let shared_start = layers.checked_sub(shared_layers).ok_or("num_kv_shared_layers exceeds num_hidden_layers")?;
        let layer_types = cfg
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or("Gemma text config has no layer_types")?
            .iter()
            .map(|v| v.as_str().map(str::to_string).ok_or("Gemma layer_types must contain strings"))
            .collect::<Result<Vec<_>, _>>()?;
        if layer_types.len() != layers {
            return Err(format!("Gemma layer_types has {}, config declares {layers} layers", layer_types.len()));
        }
        if hidden == 0 || heads == 0 || kv_heads == 0 || head_dim == 0 || global_head_dim == 0 || ple_dim == 0 {
            return Err("Gemma text dimensions must be positive".into());
        }
        if ple_vocab != vocab {
            return Err("Gemma per-layer and main vocabularies must match".into());
        }
        if hidden % block != 0 || ple_dim % block != 0 {
            return Err(format!("Gemma matrix columns must be divisible by quantization block {block}"));
        }
        if cfg.get("rope_parameters").is_none() {
            return Err("Gemma text config has no rope_parameters".into());
        }

        let tok = Tokenizer::from_hf_json(base_tokenizer)?;
        let label_ids = LABELS
            .chars()
            .map(|label| {
                let ids = tok.encode(&label.to_string());
                if ids.len() != 1 {
                    return Err(format!("Gemma answer {label:?} is not one exact tokenizer token"));
                }
                Ok(ids[0])
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut seen = HashSet::new();
        if label_ids.iter().any(|id| !seen.insert(*id)) {
            return Err("Gemma answer tokens collide".into());
        }

        let text_prefix = "model.language_model.";
        let embed = format!("{text_prefix}embed_tokens.weight");
        let ple_embed = format!("{text_prefix}embed_tokens_per_layer.weight");
        let norm = format!("{text_prefix}norm.weight");
        let ple_proj = format!("{text_prefix}per_layer_model_projection.weight");
        let ple_norm = format!("{text_prefix}per_layer_projection_norm.weight");
        require_shape(&catalog, &embed, &[vocab, hidden])?;
        require_shape(&catalog, &ple_embed, &[vocab, layers * ple_dim])?;
        require_shape(&catalog, &norm, &[hidden])?;
        require_shape(&catalog, &ple_proj, &[layers * ple_dim, hidden])?;
        require_shape(&catalog, &ple_norm, &[ple_dim])?;

        let label_source = ["lm_head.weight", "model.lm_head.weight", &embed]
            .iter()
            .find(|name| catalog.contains_key(**name))
            .map(|name| (*name).to_string())
            .ok_or("Gemma checkpoint has no output head and does not declare tied embeddings")?;
        let tie = cfg.get("tie_word_embeddings").and_then(Value::as_bool).unwrap_or(false);
        if label_source == embed && !tie && !catalog.contains_key("lm_head.weight") {
            return Err("Gemma output head is absent but tie_word_embeddings is false".into());
        }
        let label_meta = catalog.get(&label_source).unwrap();
        if label_meta.shape.len() != 2 || label_meta.shape[0] < vocab || label_meta.shape[1] != hidden {
            return Err(format!("Gemma readout {label_source} must have shape [vocab, {hidden}]"));
        }
        if label_ids.iter().any(|&id| id as usize >= label_meta.shape[0]) {
            return Err("Gemma answer token exceeds output vocabulary".into());
        }

        // Keep the upstream text config fields intact, while adding the pack/runtime contract.
        let mut config_fields = vec![
            ("arch".into(), Value::Str("gemma4".into())),
            ("backbone".into(), Value::Str("gemma4_text".into())),
            ("modalities".into(), Value::Array(vec![Value::Str("text".into())])),
        ];
        let mut present = HashSet::new();
        present.extend(["arch", "backbone", "modalities"]);
        for (key, value) in cfg.as_object().ok_or("Gemma text config is not an object")? {
            if present.insert(key.as_str()) {
                config_fields.push((key.clone(), value.clone()));
            }
        }
        config_fields.extend([
            ("kv_shared_start".into(), num(shared_start as f64)),
            ("embedding_scale".into(), Value::Float(bf16_round((hidden as f32).sqrt()) as f64)),
            ("ple_embedding_scale".into(), Value::Float(bf16_round((ple_dim as f32).sqrt()) as f64)),
            ("ple_input_scale".into(), Value::Float(2.0f64.sqrt().recip())),
            ("ple_projection_scale".into(), Value::Float((hidden as f64).sqrt().recip())),
            ("block".into(), num(block as f64)),
            ("max_input_tokens".into(), num(4096.0)),
            ("readout".into(), Value::Str("labels".into())),
            ("prompt_version".into(), Value::Str("direct-options-v1".into())),
            ("label_token_ids".into(), Value::Array(label_ids.iter().map(|&id| Value::Int(id.to_string())).collect())),
            ("label_tokens".into(), Value::Str(LABELS.into())),
            (
                "template".into(),
                Value::Object(vec![
                    ("type".into(), Value::Str("gemma4".into())),
                    ("prompt_version".into(), Value::Str("direct-options-v1".into())),
                ]),
            ),
            (
                "excluded_modalities".into(),
                Value::Array(vec![Value::Str("image".into()), Value::Str("video".into()), Value::Str("audio".into())]),
            ),
        ]);
        let config = Value::Object(config_fields);

        let mut writer = Writer::new(model, config, tok.to_bytes());
        let mut outputs = Vec::new();

        let emb_out = add_q8_output(&mut writer, &mut outputs, "embed".into(), vocab, hidden, block);
        let norm_out = add_f32_output(&mut writer, &mut outputs, "norm".into(), &[hidden]);
        let labels_out = add_f32_output(&mut writer, &mut outputs, "readout.labels".into(), &[label_ids.len(), hidden]);
        let ple_norm_out = add_f32_output(&mut writer, &mut outputs, "ple.norm".into(), &[ple_dim]);
        let mut matrix_specs = vec![MatrixSpec {
            output: emb_out,
            source: embed.clone(),
            source_row: 0,
            rows: vocab,
            source_cols: hidden,
            source_col: 0,
            output_row: 0,
            cols: hidden,
            q8: true,
        }];
        let mut vector_specs =
            vec![VectorSpec { output: norm_out, source: norm }, VectorSpec { output: ple_norm_out, source: ple_norm }];
        for (label_row, &id) in label_ids.iter().enumerate() {
            matrix_specs.push(MatrixSpec {
                output: labels_out,
                source: label_source.clone(),
                source_row: id as usize,
                rows: 1,
                source_cols: hidden,
                source_col: 0,
                output_row: label_row,
                cols: hidden,
                q8: false,
            });
        }

        let mut ple_embed_out = Vec::with_capacity(layers);
        let mut ple_proj_out = Vec::with_capacity(layers);
        for i in 0..layers {
            ple_embed_out.push(add_q8_output(
                &mut writer,
                &mut outputs,
                format!("ple.{i}.embed"),
                vocab,
                ple_dim,
                block,
            ));
            ple_proj_out.push(add_q8_output(
                &mut writer,
                &mut outputs,
                format!("ple.{i}.proj"),
                ple_dim,
                hidden,
                block,
            ));
            matrix_specs.push(MatrixSpec {
                output: ple_embed_out[i],
                source: ple_embed.clone(),
                source_row: 0,
                rows: vocab,
                source_cols: layers * ple_dim,
                source_col: i * ple_dim,
                output_row: 0,
                cols: ple_dim,
                q8: true,
            });
            matrix_specs.push(MatrixSpec {
                output: ple_proj_out[i],
                source: ple_proj.clone(),
                source_row: i * ple_dim,
                rows: ple_dim,
                source_cols: hidden,
                source_col: 0,
                output_row: 0,
                cols: hidden,
                q8: true,
            });
        }

        for i in 0..layers {
            let layer = format!("{text_prefix}layers.{i}");
            let layer_type = layer_types[i].as_str();
            let layer_head_dim = match layer_type {
                "sliding_attention" => head_dim,
                "full_attention" => global_head_dim,
                other => return Err(format!("unsupported Gemma layer type {other:?}")),
            };
            let q_rows = heads.checked_mul(layer_head_dim).ok_or("Gemma query shape overflows usize")?;
            let kv_rows = kv_heads.checked_mul(layer_head_dim).ok_or("Gemma key/value shape overflows usize")?;
            let wide = bool_field(cfg, "use_double_wide_mlp") && i >= shared_start;
            let mlp = intermediate.checked_mul(if wide { 2 } else { 1 }).ok_or("Gemma MLP shape overflows usize")?;
            for (suffix, shape) in [
                ("input_layernorm.weight", vec![hidden]),
                ("post_attention_layernorm.weight", vec![hidden]),
                ("pre_feedforward_layernorm.weight", vec![hidden]),
                ("post_feedforward_layernorm.weight", vec![hidden]),
                ("post_per_layer_input_norm.weight", vec![hidden]),
                ("layer_scalar", vec![1]),
                ("per_layer_input_gate.weight", vec![ple_dim, hidden]),
                ("per_layer_projection.weight", vec![hidden, ple_dim]),
                ("self_attn.q_norm.weight", vec![layer_head_dim]),
                ("self_attn.q_proj.weight", vec![q_rows, hidden]),
                ("self_attn.o_proj.weight", vec![hidden, q_rows]),
                ("mlp.gate_proj.weight", vec![mlp, hidden]),
                ("mlp.up_proj.weight", vec![mlp, hidden]),
                ("mlp.down_proj.weight", vec![hidden, mlp]),
            ] {
                require_shape(&catalog, &format!("{layer}.{suffix}"), &shape)?;
            }
            let common = [
                ("attn_norm", format!("{layer}.input_layernorm.weight"), vec![hidden]),
                ("attn_post_norm", format!("{layer}.post_attention_layernorm.weight"), vec![hidden]),
                ("ffn_norm", format!("{layer}.pre_feedforward_layernorm.weight"), vec![hidden]),
                ("ffn_post_norm", format!("{layer}.post_feedforward_layernorm.weight"), vec![hidden]),
                ("ple_norm", format!("{layer}.post_per_layer_input_norm.weight"), vec![hidden]),
                ("scalar", format!("{layer}.layer_scalar"), vec![1]),
            ];
            for (name, source, shape) in common {
                let out = add_f32_output(&mut writer, &mut outputs, format!("l.{i}.{name}"), &shape);
                vector_specs.push(VectorSpec { output: out, source });
            }
            for (name, source, rows, cols) in [
                ("ple_gate", format!("{layer}.per_layer_input_gate.weight"), ple_dim, hidden),
                ("ple_out", format!("{layer}.per_layer_projection.weight"), hidden, ple_dim),
                ("q", format!("{layer}.self_attn.q_proj.weight"), q_rows, hidden),
                ("o", format!("{layer}.self_attn.o_proj.weight"), hidden, q_rows),
                ("gate", format!("{layer}.mlp.gate_proj.weight"), mlp, hidden),
                ("up", format!("{layer}.mlp.up_proj.weight"), mlp, hidden),
                ("down", format!("{layer}.mlp.down_proj.weight"), hidden, mlp),
            ] {
                let out = add_q8_output(&mut writer, &mut outputs, format!("l.{i}.{name}"), rows, cols, block);
                matrix_specs.push(MatrixSpec {
                    output: out,
                    source,
                    source_row: 0,
                    rows,
                    source_cols: cols,
                    source_col: 0,
                    output_row: 0,
                    cols,
                    q8: true,
                });
            }
            require_shape(&catalog, &format!("{layer}.self_attn.q_norm.weight"), &[layer_head_dim])?;
            let qn = add_f32_output(&mut writer, &mut outputs, format!("l.{i}.qn"), &[layer_head_dim]);
            vector_specs.push(VectorSpec { output: qn, source: format!("{layer}.self_attn.q_norm.weight") });
            if i < shared_start {
                require_shape(&catalog, &format!("{layer}.self_attn.k_norm.weight"), &[layer_head_dim])?;
                let kn = add_f32_output(&mut writer, &mut outputs, format!("l.{i}.kn"), &[layer_head_dim]);
                vector_specs.push(VectorSpec { output: kn, source: format!("{layer}.self_attn.k_norm.weight") });
                for (name, source) in [
                    ("k", format!("{layer}.self_attn.k_proj.weight")),
                    ("v", format!("{layer}.self_attn.v_proj.weight")),
                ] {
                    require_shape(&catalog, &source, &[kv_rows, hidden])?;
                    let out = add_q8_output(&mut writer, &mut outputs, format!("l.{i}.{name}"), kv_rows, hidden, block);
                    matrix_specs.push(MatrixSpec {
                        output: out,
                        source,
                        source_row: 0,
                        rows: kv_rows,
                        source_cols: hidden,
                        source_col: 0,
                        output_row: 0,
                        cols: hidden,
                        q8: true,
                    });
                }
            }
        }

        for spec in &matrix_specs {
            let meta = catalog.get(&spec.source).ok_or_else(|| format!("checkpoint has no tensor {}", spec.source))?;
            if meta.shape.len() != 2 || meta.shape[0] < spec.source_row + spec.rows || meta.shape[1] != spec.source_cols
            {
                return Err(format!(
                    "{}: source shape {:?} does not fit requested matrix slice",
                    spec.source, meta.shape
                ));
            }
            if spec.source_col + spec.cols > spec.source_cols {
                return Err(format!("{}: source column slice is out of bounds", spec.source));
            }
        }
        for spec in &vector_specs {
            let meta = catalog.get(&spec.source).ok_or_else(|| format!("checkpoint has no tensor {}", spec.source))?;
            let numel =
                meta.shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).ok_or("vector shape overflows usize")?;
            if meta.shape.len() > 1 || numel == 0 {
                return Err(format!("{}: expected a non-empty vector", spec.source));
            }
        }

        let (prefix, infos, total) = writer.layout();
        let mut jobs = Vec::new();
        let mut grouped: BTreeMap<(String, usize, usize, usize), Vec<MatrixSpec>> = BTreeMap::new();
        for spec in matrix_specs {
            grouped.entry((spec.source.clone(), spec.source_row, spec.rows, spec.source_cols)).or_default().push(spec);
        }
        for ((source, source_row, rows, source_cols), specs) in grouped {
            let meta = catalog.get(&source).unwrap();
            let row_bytes = source_cols.checked_mul(meta.dtype.bytes()).ok_or("source row size overflows usize")?;
            let chunk_rows = (CHUNK_BYTES / row_bytes.max(1)).max(1).min(rows);
            for row in (0..rows).step_by(chunk_rows) {
                let count = (rows - row).min(chunk_rows);
                let mut outputs_for_job = Vec::new();
                for spec in &specs {
                    outputs_for_job.push(OutputChunk {
                        output: spec.output,
                        output_row: spec.output_row + row,
                        rows: count,
                        source_col: spec.source_col,
                        cols: spec.cols,
                        q8: spec.q8,
                    });
                }
                let row_bytes_at = (source_row + row).checked_mul(row_bytes).ok_or("source offset overflows usize")?;
                let offset = meta.offset.checked_add(row_bytes_at).ok_or("source offset overflows usize")?;
                let len = count.checked_mul(row_bytes).ok_or("source length overflows usize")?;
                jobs.push(Job {
                    name: format!("gemma.matrix.{}", jobs.len()),
                    shard: meta.shard,
                    offset,
                    len,
                    dtype: meta.dtype,
                    source_cols,
                    source_rows: count,
                    outputs: outputs_for_job,
                });
            }
        }
        for spec in vector_specs {
            let meta = catalog.get(&spec.source).unwrap();
            let count = meta.shape.iter().product::<usize>();
            let len = count.checked_mul(meta.dtype.bytes()).ok_or("source length overflows usize")?;
            jobs.push(Job {
                name: format!("gemma.vector.{}", jobs.len()),
                shard: meta.shard,
                offset: meta.offset,
                len,
                dtype: meta.dtype,
                source_cols: count,
                source_rows: 1,
                outputs: vec![OutputChunk {
                    output: spec.output,
                    output_row: 0,
                    rows: 1,
                    source_col: 0,
                    cols: count,
                    q8: false,
                }],
            });
        }
        jobs.sort_by_key(|job| (job.shard, job.offset, job.name.clone()));
        let by_name = jobs.iter().enumerate().map(|(i, job)| (job.name.clone(), i)).collect();
        let job_count = jobs.len();
        Ok(GemmaConvert { jobs, by_name, infos, prefix, total, block, done: vec![false; job_count] })
    }

    /// Feed source ranges in safetensors file order. Every range is bounded by CHUNK_BYTES for
    /// matrices, including the packed per-layer embedding.
    pub fn source_ranges(&self) -> Vec<(usize, String, usize, usize)> {
        self.jobs.iter().map(|j| (j.shard, j.name.clone(), j.offset, j.len)).collect()
    }

    pub fn begin(&mut self, out: &mut [u8]) -> Result<(), String> {
        if out.len() < self.total {
            return Err(format!("output buffer is {} bytes, need {}", out.len(), self.total));
        }
        out[..self.prefix.len()].copy_from_slice(&self.prefix);
        Ok(())
    }

    pub fn add_source(&mut self, name: &str, bytes: Vec<u8>, out: &mut [u8]) -> Result<(), String> {
        let index = *self.by_name.get(name).ok_or_else(|| format!("unexpected Gemma source range {name}"))?;
        if self.done[index] {
            return Err(format!("Gemma source range {name} was supplied twice"));
        }
        let job = &self.jobs[index];
        if bytes.len() != job.len {
            return Err(format!("{name}: got {} bytes, expected {}", bytes.len(), job.len));
        }
        let source_values = job.source_rows.checked_mul(job.source_cols).ok_or("source shape overflows usize")?;
        if bytes.len() != source_values * job.dtype.bytes() {
            return Err(format!("{name}: source bytes do not match its declared shape"));
        }
        for chunk in &job.outputs {
            let values_len = chunk.rows.checked_mul(chunk.cols).ok_or("output shape overflows usize")?;
            let mut values = Vec::with_capacity(values_len);
            for row in 0..chunk.rows {
                for col in 0..chunk.cols {
                    values.push(source_to_f32(job.dtype, &bytes, row * job.source_cols + chunk.source_col + col)?);
                }
            }
            let info = self.infos.get(chunk.output).ok_or("invalid output tensor index")?;
            if chunk.q8 {
                if chunk.cols % self.block != 0 {
                    return Err(format!("{}: output columns are not divisible by block {}", info.name, self.block));
                }
                let (q, scales) = quantize(&values, chunk.rows, chunk.cols, self.block);
                let q_at = info
                    .offset
                    .checked_add(chunk.output_row * chunk.cols)
                    .ok_or("q8 destination offset overflows usize")?;
                let q_end = q_at.checked_add(q.len()).ok_or("q8 destination size overflows usize")?;
                let q_out =
                    out.get_mut(q_at..q_end).ok_or_else(|| format!("{}: q8 destination is too short", info.name))?;
                for (dst, value) in q_out.iter_mut().zip(q) {
                    *dst = value as u8;
                }
                let scale_at = info
                    .scales_offset
                    .checked_add(chunk.output_row * (chunk.cols / self.block) * 4)
                    .ok_or("scale destination offset overflows usize")?;
                for (i, scale) in scales.iter().enumerate() {
                    let at = scale_at + i * 4;
                    out.get_mut(at..at + 4)
                        .ok_or_else(|| format!("{}: scale destination is too short", info.name))?
                        .copy_from_slice(&scale.to_le_bytes());
                }
            } else {
                let at = info
                    .offset
                    .checked_add(chunk.output_row * chunk.cols * 4)
                    .ok_or("f32 destination offset overflows usize")?;
                add_f32(out, at, &values)?;
            }
        }
        self.done[index] = true;
        Ok(())
    }

    pub fn finished(&self) -> bool {
        self.done.iter().all(|done| *done)
    }
}

fn require_shape(catalog: &HashMap<String, TensorMeta>, name: &str, expected: &[usize]) -> Result<(), String> {
    let meta = catalog.get(name).ok_or_else(|| format!("checkpoint has no tensor {name}"))?;
    if meta.shape != expected {
        return Err(format!("{name}: expected shape {expected:?}, got {:?}", meta.shape));
    }
    Ok(())
}
