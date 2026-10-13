//! Convert H2O's merged Qwen3.5 checkpoint and its pinned serving contract.

use crate::convert::Precision;
use crate::h2o::Config;
use crate::json::Value;
use crate::kev::convert::{CustomReadout, KevConvert};
use crate::tokenizer::Tokenizer;

#[allow(clippy::too_many_arguments)]
pub fn new_sharded(
    heads: &[&[u8]],
    base_config: &str,
    tokenizer: &str,
    serve_config: &str,
    block: usize,
    model: Value,
    precision: Precision,
) -> Result<KevConvert, String> {
    let source = Value::parse(base_config).map_err(|e| e.to_string())?;
    let cfg = source.get("text_config").unwrap_or(&source);
    let serve = Value::parse(serve_config).map_err(|e| e.to_string())?;
    let contract = Config::from_json(&serve)?;
    let tok = Tokenizer::from_hf_json(tokenizer)?;
    if contract.chat {
        crate::convert::require_prompt_tokens(&tok, &["<|im_start|>", "<|im_end|>", "<think>", "</think>"])?;
    }
    let label_ids = contract.label_token_ids(&tok)?;
    let mut names = std::collections::HashMap::new();
    for head in heads {
        let bytes = head.get(..8).ok_or("H2O: short safetensors header")?;
        let n = usize::try_from(u64::from_le_bytes(bytes.try_into().unwrap()))
            .map_err(|_| "H2O: safetensors header does not fit usize")?;
        let end = 8usize.checked_add(n).ok_or("H2O: safetensors header length overflow")?;
        let json = std::str::from_utf8(head.get(8..end).ok_or("H2O: truncated safetensors header")?)
            .map_err(|_| "H2O: safetensors header is not UTF-8")?;
        let header = Value::parse(json).map_err(|e| e.to_string())?;
        for (name, tensor) in header.as_object().ok_or("H2O: safetensors header must be an object")? {
            if name != "__metadata__" && names.insert(name.clone(), tensor.clone()).is_some() {
                return Err(format!("H2O: duplicate tensor {name}"));
            }
        }
    }
    let prefix = crate::convert::text_tensor_prefix(|name| names.contains_key(name))?;
    let mut lm_heads = [
        "lm_head.weight",
        "model.lm_head.weight",
        "model.language_model.lm_head.weight",
        "language_model.lm_head.weight",
    ]
    .into_iter()
    .filter(|name| names.contains_key(*name));
    let lm_head = if let Some(head) = lm_heads.next() {
        if lm_heads.next().is_some() {
            return Err("H2O checkpoint has ambiguous LM head tensors".into());
        }
        head.to_string()
    } else if cfg.get("tie_word_embeddings").or_else(|| source.get("tie_word_embeddings")).and_then(Value::as_bool)
        == Some(true)
    {
        format!("{prefix}embed_tokens.weight")
    } else {
        return Err("H2O checkpoint has no LM head and does not declare tied word embeddings".into());
    };
    let hidden = cfg.get("hidden_size").and_then(Value::as_usize).ok_or("H2O config has no hidden_size")?;
    let shape = names[&lm_head].get("shape").and_then(Value::as_array).ok_or("H2O LM head has no shape")?;
    if shape.len() != 2
        || shape[1].as_usize() != Some(hidden)
        || label_ids.iter().any(|&id| shape[0].as_usize().is_none_or(|rows| id as usize >= rows))
    {
        return Err("H2O LM head shape is incompatible with label token IDs and hidden_size".into());
    }
    let max_prompt = cfg.get("max_position_embeddings").and_then(Value::as_usize).unwrap_or(262144);
    let custom = CustomReadout {
        config: Value::Object(vec![
            ("readout".into(), Value::Str("h2o-lightning".into())),
            ("h2o_config".into(), serve),
            ("temperature".into(), Value::Float(1.0)),
            ("max_state".into(), Value::Int(contract.max_state_tokens.to_string())),
            ("max_branch".into(), Value::Int(max_prompt.to_string())),
            ("prompt_version".into(), Value::Str("h2o-lightning-v1.2.3-text".into())),
            ("label_token_ids".into(), Value::Array(label_ids.iter().map(|id| Value::Int(id.to_string())).collect())),
        ]),
        rows: vec![("h2o.labels".into(), lm_head, label_ids.into_iter().map(|id| id as usize).collect())],
        ..CustomReadout::default()
    };
    KevConvert::new_custom_sharded(heads, base_config, tokenizer, block, model, precision, custom)
}
