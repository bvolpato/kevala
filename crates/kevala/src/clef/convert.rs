//! Converts the merged Clef text backbone and its separate joint head.

use super::head::HeadConfig;
use crate::convert::{bf16_to_f32, f16_to_f32, Precision};
use crate::json::Value;
use crate::kev::convert::{CustomReadout, KevConvert};

/// Read and validate every joint-head tensor before any backbone conversion starts.
pub fn readout(head_config: &str, head: &[u8]) -> Result<CustomReadout, String> {
    let config = Value::parse(head_config).map_err(|e| format!("joint_head_config.json: {e}"))?;
    let cfg = HeadConfig::from_json(&config)?;
    let n =
        usize::try_from(u64::from_le_bytes(head.get(..8).ok_or("short joint_head.safetensors")?.try_into().unwrap()))
            .map_err(|_| "joint head header is too large")?;
    let end = 8usize.checked_add(n).ok_or("joint head header overflows usize")?;
    let text = std::str::from_utf8(head.get(8..end).ok_or("truncated joint head header")?)
        .map_err(|_| "joint head header is not UTF-8")?;
    let header = Value::parse(text).map_err(|e| format!("joint head header: {e}"))?;
    let entries = header.as_object().ok_or("joint head header must be an object")?;
    let expected = cfg.tensors();
    for (name, _) in entries {
        if name != "__metadata__" && !expected.iter().any(|(key, _)| key == name) {
            return Err(format!("joint head: unsupported tensor {name}"));
        }
    }
    let mut tensors = Vec::new();
    let mut ranges = Vec::new();
    for (name, shape) in expected {
        let t = header.get(&name).ok_or_else(|| format!("joint head: missing tensor {name}"))?;
        let actual = t
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("joint head {name}: missing shape"))?
            .iter()
            .map(|v| v.as_usize().ok_or("bad head tensor dimension"))
            .collect::<Result<Vec<_>, _>>()?;
        if shape != actual {
            return Err(format!("joint head {name}: expected {shape:?}, got {actual:?}"));
        }
        let offsets = t
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|a| a.len() == 2)
            .ok_or_else(|| format!("joint head {name}: invalid offsets"))?;
        let a = offsets[0].as_usize().ok_or("bad head start offset")?;
        let b = offsets[1].as_usize().ok_or("bad head end offset")?;
        let dtype = t.get("dtype").and_then(Value::as_str).ok_or("joint head: missing dtype")?;
        let element = match dtype {
            "F32" => 4,
            "BF16" | "F16" => 2,
            _ => return Err(format!("joint head {name}: unsupported dtype {dtype}")),
        };
        let count =
            shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d)).ok_or("joint head shape overflows usize")?;
        let size = count.checked_mul(element).ok_or("joint head size overflows usize")?;
        if b.checked_sub(a) != Some(size) {
            return Err(format!("joint head {name}: inconsistent tensor size"));
        }
        let lo = end.checked_add(a).ok_or("joint head offset overflows usize")?;
        let hi = end.checked_add(b).ok_or("joint head offset overflows usize")?;
        let bytes = head.get(lo..hi).ok_or_else(|| format!("joint head {name}: truncated tensor"))?;
        ranges.push((a, b));
        let values: Vec<f32> = match dtype {
            "F32" => bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
            "BF16" => bytes.chunks_exact(2).map(|b| bf16_to_f32(u16::from_le_bytes(b.try_into().unwrap()))).collect(),
            _ => bytes.chunks_exact(2).map(|b| f16_to_f32(u16::from_le_bytes(b.try_into().unwrap()))).collect(),
        };
        if values.iter().any(|v| !v.is_finite()) {
            return Err(format!("joint head {name}: non-finite weights"));
        }
        tensors.push((format!("clef.{name}"), shape, values));
    }
    ranges.sort_unstable();
    let mut at = 0;
    for (a, b) in ranges {
        if a != at {
            return Err("joint head: overlapping or noncontiguous tensor data".into());
        }
        at = b;
    }
    if end.checked_add(at) != Some(head.len()) {
        return Err("joint head: unconsumed trailing bytes".into());
    }
    Ok(CustomReadout {
        config: Value::Object(vec![
            ("arch".into(), Value::Str("clef".into())),
            ("readout".into(), Value::Str("clef".into())),
            ("joint_head".into(), config),
            ("modalities".into(), Value::Array(vec![Value::Str("text".into())])),
            ("max_length".into(), Value::Int("16384".into())),
            ("max_state".into(), Value::Int("16384".into())),
            ("max_branch".into(), Value::Int("0".into())),
            (
                "template".into(),
                Value::Object(vec![
                    ("type".into(), Value::Str("clef-joint-schema-v1".into())),
                    ("token_ids".into(), Value::Array(vec![])),
                ]),
            ),
        ]),
        tensors,
        matrices: vec![("clef.lexical".into(), "lm_head.weight".into())],
        rows: vec![],
    })
}

#[allow(clippy::too_many_arguments)]
pub fn plan(
    base_heads: &[&[u8]],
    config: &str,
    tokenizer: &str,
    head_config: &str,
    head: &[u8],
    block: usize,
    model: Value,
    precision: Precision,
) -> Result<KevConvert, String> {
    let readout = readout(head_config, head)?;
    let base = Value::parse(config).map_err(|e| format!("clef config: {e}"))?;
    let hidden = base
        .get("text_config")
        .unwrap_or(&base)
        .get("hidden_size")
        .and_then(Value::as_usize)
        .ok_or("clef config: missing hidden_size")?;
    if readout.config.get("joint_head").and_then(|c| c.get("hidden_size")).and_then(Value::as_usize) != Some(hidden) {
        return Err("clef: head and backbone hidden sizes differ".into());
    }
    let vocab = base
        .get("text_config")
        .unwrap_or(&base)
        .get("vocab_size")
        .and_then(Value::as_usize)
        .ok_or("clef config: missing vocab_size")?;
    let mut lexical_shape = None;
    for bytes in base_heads {
        let length = usize::try_from(u64::from_le_bytes(
            bytes.get(..8).ok_or("short Clef backbone header")?.try_into().unwrap(),
        ))
        .map_err(|_| "Clef backbone header is too large")?;
        let end = 8usize.checked_add(length).ok_or("Clef backbone header overflows usize")?;
        let text = std::str::from_utf8(bytes.get(8..end).ok_or("truncated Clef backbone header")?)
            .map_err(|_| "Clef backbone header is not UTF-8")?;
        let header = Value::parse(text).map_err(|e| format!("Clef backbone header: {e}"))?;
        if let Some(t) = header.get("lm_head.weight") {
            if lexical_shape.is_some() {
                return Err("clef: duplicate output embedding tensor".into());
            }
            lexical_shape = Some(
                t.get("shape")
                    .and_then(Value::as_array)
                    .ok_or("clef lm_head.weight: missing shape")?
                    .iter()
                    .map(|v| v.as_usize().ok_or("clef lm_head.weight: invalid dimension"))
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
    }
    if lexical_shape.as_deref() != Some(&[vocab, hidden]) {
        return Err(format!("clef lm_head.weight: expected [{vocab}, {hidden}], got {lexical_shape:?}"));
    }
    KevConvert::new_custom_sharded(base_heads, config, tokenizer, block, model, precision, readout)
}
