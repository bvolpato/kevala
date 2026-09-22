use kevala::json::Value;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

fn safetensors_header(path: &Path) -> Result<(Vec<u8>, usize), String> {
    let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix).map_err(|e| format!("{}: {e}", path.display()))?;
    let n = usize::try_from(u64::from_le_bytes(prefix))
        .map_err(|_| format!("{}: header length does not fit usize", path.display()))?;
    let total = 8usize.checked_add(n).ok_or_else(|| format!("{}: header length overflows usize", path.display()))?;
    if total as u64 > file.metadata().map_err(|error| format!("{}: {error}", path.display()))?.len() {
        return Err(format!("{}: truncated safetensors header", path.display()));
    }
    let mut head = Vec::with_capacity(total);
    head.extend_from_slice(&prefix);
    head.resize(total, 0);
    file.read_exact(&mut head[8..]).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((head, total))
}

fn indexed_shards(dir: &str) -> Result<Vec<PathBuf>, String> {
    let root = Path::new(dir);
    let index_path = root.join("model.safetensors.index.json");
    if index_path.exists() {
        let index_bytes = std::fs::read(&index_path).map_err(|e| format!("{}: {e}", index_path.display()))?;
        let index_text =
            String::from_utf8(index_bytes).map_err(|_| format!("{} is not UTF-8", index_path.display()))?;
        let index = Value::parse(&index_text).map_err(|e| format!("{}: {e}", index_path.display()))?;
        let mut names = Vec::new();
        for (_, value) in index
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| format!("{}: missing weight_map", index_path.display()))?
        {
            let name = value
                .as_str()
                .ok_or_else(|| format!("{}: weight_map value is not a filename", index_path.display()))?;
            if !names.iter().any(|seen| seen == name) {
                names.push(name.to_string());
            }
        }
        if names.is_empty() {
            return Err(format!("{}: weight_map is empty", index_path.display()));
        }
        return names.into_iter().map(|name| safe_join(root, &name)).collect();
    }
    let single = root.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(root)
        .map_err(|e| format!("{dir}: {e}"))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    files.sort();
    match files.len() {
        0 => Err(format!("{dir}: no safetensors checkpoint or model.safetensors.index.json")),
        1 => Ok(files),
        _ => {
            Err(format!("{dir}: {} safetensors shards found but model.safetensors.index.json is missing", files.len()))
        }
    }
}

fn safe_join(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path
            .components()
            .any(|part| matches!(part, Component::ParentDir | Component::RootDir | Component::Prefix(_)))
    {
        return Err(format!("checkpoint shard path is not relative: {relative}"));
    }
    Ok(root.join(path))
}

fn output_pack(path: &str, pack: &[u8], started: Instant) -> Result<(), String> {
    let mut file = File::create(path).map_err(|e| format!("{path}: {e}"))?;
    file.write_all(pack).map_err(|e| format!("{path}: {e}"))?;
    file.flush().map_err(|e| format!("{path}: {e}"))?;
    eprintln!("wrote {path}: {:.1} MB in {:.1}s", pack.len() as f64 / 1e6, started.elapsed().as_secs_f64());
    Ok(())
}

enum StreamMode {
    Kev { adapter: Vec<u8>, adapter_config: String, head: Vec<u8> },
    Semif,
}

fn convert_sharded(
    shards: &[PathBuf],
    base_config: &str,
    base_tokenizer: &str,
    block: usize,
    model: Value,
    mode: StreamMode,
) -> Result<Vec<u8>, String> {
    let headers: Vec<Vec<u8>> =
        shards.iter().map(|path| safetensors_header(path).map(|(head, _)| head)).collect::<Result<_, _>>()?;
    let refs: Vec<&[u8]> = headers.iter().map(Vec::as_slice).collect();
    let mut converter = match mode {
        StreamMode::Kev { adapter, adapter_config, head } => kevala::convert_kev::KevConvert::new_sharded(
            &refs,
            base_config,
            base_tokenizer,
            &adapter,
            &adapter_config,
            &head,
            block,
            model,
        )?,
        StreamMode::Semif => {
            kevala::convert_kev::KevConvert::new_semif_sharded(&refs, base_config, base_tokenizer, block, model)?
        }
    };
    let mut out = vec![0u8; converter.total];
    converter.begin(&mut out)?;
    let mut files: Vec<File> = shards
        .iter()
        .map(|path| File::open(path).map_err(|e| format!("{}: {e}", path.display())))
        .collect::<Result<_, _>>()?;
    for (shard, name, offset, len) in converter.source_ranges()? {
        let file = files.get_mut(shard).ok_or("converter returned an invalid shard index")?;
        file.seek(SeekFrom::Start(offset as u64)).map_err(|e| format!("{}: {e}", shards[shard].display()))?;
        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes).map_err(|e| format!("{}: {e}", shards[shard].display()))?;
        converter.add_source(&name, bytes, &mut out)?;
    }
    if !converter.finished() {
        return Err("conversion ended with tensors still missing".into());
    }
    Ok(out)
}

fn convert_gemma_sharded(
    shards: &[PathBuf],
    base_config: &str,
    tokenizer: &str,
    block: usize,
    model: Value,
) -> Result<Vec<u8>, String> {
    let headers: Vec<Vec<u8>> =
        shards.iter().map(|path| safetensors_header(path).map(|(head, _)| head)).collect::<Result<_, _>>()?;
    let refs: Vec<&[u8]> = headers.iter().map(Vec::as_slice).collect();
    let mut converter = kevala::convert_gemma::GemmaConvert::new_sharded(&refs, base_config, tokenizer, block, model)?;
    let mut out = vec![0u8; converter.total];
    converter.begin(&mut out)?;
    let mut files: Vec<File> = shards
        .iter()
        .map(|path| File::open(path).map_err(|e| format!("{}: {e}", path.display())))
        .collect::<Result<_, _>>()?;
    for (shard, name, offset, len) in converter.source_ranges() {
        let file = files.get_mut(shard).ok_or("Gemma converter returned an invalid shard index")?;
        file.seek(SeekFrom::Start(offset as u64)).map_err(|e| format!("{}: {e}", shards[shard].display()))?;
        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes).map_err(|e| format!("{}: {e}", shards[shard].display()))?;
        converter.add_source(&name, bytes, &mut out)?;
    }
    if !converter.finished() {
        return Err("Gemma conversion ended with tensors still missing".into());
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Architecture {
    Encoder,
    Qwen35,
    Gemma4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Readout {
    EncoderHead,
    Pointer,
    DirectOptions,
}

impl Readout {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "encoder-head" => Ok(Self::EncoderHead),
            "pointer" => Ok(Self::Pointer),
            "direct-options" => Ok(Self::DirectOptions),
            _ => Err(format!("unsupported readout {value:?}; expected direct-options, pointer, or encoder-head")),
        }
    }
}

#[derive(Debug)]
struct Options {
    checkpoint: String,
    values: std::collections::BTreeMap<String, String>,
    alias: Option<Architecture>,
}

impl Options {
    fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    fn parse(command: &str, args: &[String]) -> Result<Self, String> {
        let mut values = std::collections::BTreeMap::new();
        let mut checkpoint = None;
        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            if !arg.starts_with('-') {
                if checkpoint.replace(arg.clone()).is_some() {
                    return Err("convert accepts one checkpoint directory".into());
                }
                continue;
            }
            let key = match arg.as_str() {
                "--kev" if command == "convert-kev" => "--adapter",
                "--kev-revision" if command == "convert-kev" => "--revision",
                "-o" | "--block" | "--keep-f32" | "--tokenizer" | "--base" | "--adapter" | "--readout" | "--name"
                | "--source" | "--revision" | "--base-source" | "--base-revision" | "--author" | "--license"
                | "--method-revision" => arg,
                _ => return Err(format!("unknown convert option {arg}")),
            };
            let value =
                iter.next().filter(|value| !value.starts_with('-')).ok_or_else(|| format!("{arg} needs a value"))?;
            if values.insert(key.to_string(), value.clone()).is_some() {
                return Err(format!("duplicate convert option {key}"));
            }
        }
        let checkpoint =
            checkpoint.or_else(|| values.get("--base").cloned()).ok_or("convert needs <checkpoint-dir>")?;
        if !values.contains_key("-o") {
            return Err("convert needs -o <out.kevala>".into());
        }
        let alias = match command {
            "convert-kev" => {
                if !values.contains_key("--adapter") {
                    return Err("convert-kev needs --kev <adapter-dir> (or --adapter)".into());
                }
                Some(Architecture::Qwen35)
            }
            "convert-semif" => Some(Architecture::Qwen35),
            "convert-gemma" => Some(Architecture::Gemma4),
            _ => None,
        };
        Ok(Self { checkpoint, values, alias })
    }
}

fn text(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn json_file(path: &Path) -> Result<Value, String> {
    Value::parse(&text(path)?).map_err(|e| format!("{}: {e}", path.display()))
}

fn family(value: &str) -> Option<Architecture> {
    match value {
        "modernbert" | "ModernBertModel" | "ModernBertForMaskedLM" => Some(Architecture::Encoder),
        "qwen3_5" | "qwen3_5_text" | "Qwen3_5ForConditionalGeneration" | "Qwen3_5ForCausalLM" => {
            Some(Architecture::Qwen35)
        }
        "gemma4" | "gemma4_text" | "Gemma4ForConditionalGeneration" | "Gemma4ForCausalLM" => Some(Architecture::Gemma4),
        _ => None,
    }
}

fn architecture(config: &Value) -> Result<Architecture, String> {
    if config.as_object().is_none() {
        return Err("checkpoint config must be an object".into());
    }
    let nested = config.get("text_config");
    if nested.is_some_and(|value| value.as_object().is_none()) {
        return Err("text_config must be an object".into());
    }
    let mut detected = None;
    for cfg in std::iter::once(config).chain(nested) {
        let mut ids = Vec::new();
        if let Some(kind) = cfg.get("model_type") {
            ids.push(kind.as_str().ok_or("model_type must be a string")?);
        }
        if let Some(names) = cfg.get("architectures") {
            for name in names.as_array().ok_or("architectures must be an array")? {
                ids.push(name.as_str().ok_or("architectures must contain strings")?);
            }
        }
        for id in ids {
            let arch = family(id).ok_or_else(|| format!("unsupported checkpoint architecture {id:?}; supported: ModernBERT with decision heads, dense Qwen3.5, dense Gemma4 text"))?;
            if detected.is_some_and(|previous| previous != arch) {
                return Err("conflicting model_type/architectures in checkpoint config".into());
            }
            detected = Some(arch);
        }
    }
    detected.ok_or_else(|| "checkpoint config has no supported model_type or architectures".into())
}

fn reject_quantized_or_moe(config: &Value) -> Result<(), String> {
    for cfg in std::iter::once(config).chain(config.get("text_config")) {
        if cfg.get("quantization_config").is_some_and(|value| !value.is_null()) {
            return Err("pre-quantized source checkpoints are not supported; use F32, F16, or BF16 weights".into());
        }
        if cfg.get("enable_moe_block").and_then(Value::as_bool) == Some(true)
            || ["num_experts", "num_local_experts", "top_k_experts", "num_experts_per_tok"]
                .iter()
                .any(|key| cfg.get(key).is_some_and(|value| !value.is_null() && value.as_usize() != Some(0)))
        {
            return Err(
                "mixture-of-experts checkpoints are not supported; only dense text backbones are available".into()
            );
        }
    }
    Ok(())
}

fn validate_qwen_config(config: &Value) -> Result<(), String> {
    let config = config.get("text_config").unwrap_or(config);
    for (key, expected) in [("attention_bias", false), ("attn_output_gate", true)] {
        if config.get(key).is_some_and(|value| value.as_bool() != Some(expected)) {
            return Err(format!("unsupported Qwen3.5 config: {key} must be {expected}"));
        }
    }
    if config.get("hidden_act").is_some_and(|value| value.as_str() != Some("silu")) {
        return Err("unsupported Qwen3.5 activation; expected hidden_act=silu".into());
    }
    if config.get("rope_scaling").is_some_and(|value| !value.is_null()) {
        return Err("Qwen3.5 rope_scaling is unsupported".into());
    }
    if config
        .get("rope_parameters")
        .and_then(|value| value.get("rope_type"))
        .is_some_and(|value| value.as_str() != Some("default"))
    {
        return Err("Qwen3.5 conversion currently supports only rope_type=default".into());
    }
    Ok(())
}

fn check_readout(arch: Architecture, adapter: bool, requested: Option<&str>) -> Result<Readout, String> {
    let inferred = match (arch, adapter) {
        (Architecture::Encoder, false) => Readout::EncoderHead,
        (Architecture::Qwen35, true) => Readout::Pointer,
        (_, false) => Readout::DirectOptions,
        _ => return Err("pointer adapters are currently supported only by the Qwen3.5 architecture".into()),
    };
    if let Some(requested) = requested {
        let readout = Readout::parse(requested)?;
        if readout != inferred {
            return Err(format!(
                "readout {requested:?} is incompatible with this checkpoint's architecture and head tensors"
            ));
        }
    }
    Ok(inferred)
}

fn validate_adapter(config: &Value) -> Result<(), String> {
    if config.get("peft_type").and_then(Value::as_str) != Some("LORA") {
        return Err("pointer adapter requires peft_type=LORA".into());
    }
    for flag in ["fan_in_fan_out", "use_rslora", "use_dora", "use_qalora", "lora_bias"] {
        if config.get(flag).is_some_and(|value| !value.is_null() && value.as_bool() != Some(false)) {
            return Err(format!("unsupported LoRA option {flag}; this converter merges standard alpha/r LoRA"));
        }
    }
    if config.get("bias").is_some_and(|value| value.as_str() != Some("none")) {
        return Err("LoRA bias must be none".into());
    }
    for key in [
        "rank_pattern",
        "alpha_pattern",
        "modules_to_save",
        "layer_replication",
        "target_parameters",
        "trainable_token_indices",
        "alora_invocation_tokens",
    ] {
        if config
            .get(key)
            .is_some_and(|value| !value.is_null() && value.as_object() != Some(&[]) && value.as_array() != Some(&[]))
        {
            return Err(format!("unsupported LoRA option {key}"));
        }
    }
    Ok(())
}

fn source_value(value: &str) -> String {
    let parts: Vec<_> = value.split('/').collect();
    if parts.len() == 2 && parts.iter().all(|part| !part.is_empty() && !part.starts_with('.')) && !value.contains(':') {
        format!("https://huggingface.co/{value}")
    } else {
        value.to_string()
    }
}

fn declared_source(config: &Value) -> Option<&str> {
    [config, config.get("text_config").unwrap_or(config)].into_iter().find_map(|cfg| {
        ["_name_or_path", "name_or_path"]
            .into_iter()
            .find_map(|key| cfg.get(key).and_then(Value::as_str).filter(|s| !s.is_empty()))
    })
}

fn metadata(options: &Options, config: &Value, readout: Readout, block: usize) -> Value {
    let name = options.get("--name").map(str::to_string).unwrap_or_else(|| {
        Path::new(declared_source(config).unwrap_or(&options.checkpoint))
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("checkpoint")
            .to_string()
    });
    let mut fields = vec![
        ("name".into(), Value::Str(name)),
        ("converter".into(), Value::Str(format!("kevala {}", env!("CARGO_PKG_VERSION")))),
        (
            "quantization".into(),
            Value::Str(if block == 0 {
                "f32 reference pack".into()
            } else {
                format!("int8 symmetric absmax, one f32 scale per {block} weights; architecture-specific norms and decision heads in f32")
            }),
        ),
    ];
    for (field, value) in [
        (
            "source",
            options
                .get("--source")
                .or_else(|| if readout == Readout::Pointer { None } else { declared_source(config) }),
        ),
        (
            "revision",
            options.get("--revision").or_else(|| {
                if readout == Readout::Pointer {
                    None
                } else {
                    options.get("--base-revision")
                }
            }),
        ),
        ("base", options.get("--base-source").or_else(|| declared_source(config))),
        ("base_revision", options.get("--base-revision")),
        ("author", options.get("--author")),
        ("license", options.get("--license")),
    ] {
        if let Some(value) = value {
            let explicit_source = (field == "source" && options.get("--source").is_some())
                || (field == "base" && options.get("--base-source").is_some());
            fields.push((
                field.into(),
                Value::Str(if explicit_source { source_value(value) } else { value.to_string() }),
            ));
        }
    }
    if readout == Readout::DirectOptions {
        fields.extend([
            ("inspiration".into(), Value::Str("SemIf direct-options-v1 readout".into())),
            ("method_source".into(), Value::Str("https://github.com/TheoLeeCJ/SemIf".into())),
            (
                "method_revision".into(),
                Value::Str(
                    options.get("--method-revision").unwrap_or("1f2dea3e25379f9dfc98cb83c324f00ab5deda37").into(),
                ),
            ),
            ("method_license".into(), Value::Str("mit".into())),
        ]);
    }
    Value::Object(fields)
}

fn validate_tokenizer(arch: Architecture, tokenizer: &str) -> Result<(), String> {
    if arch == Architecture::Qwen35 {
        let value = Value::parse(tokenizer).map_err(|error| format!("tokenizer: {error}"))?;
        let pre = value.get("pre_tokenizer").ok_or("tokenizer has no pre_tokenizer")?;
        if pre.to_json().contains(r"\\p{M}") {
            return Err("Qwen3.5's raw tokenizer.json requires its tokenizer_config.json; restore that file beside the checkpoint tokenizer or pass --tokenizer <tokenizer.json> saved by AutoTokenizer.save_pretrained".into());
        }
    }
    Ok(())
}

pub fn run(command: &str, args: &[String]) -> Result<(), String> {
    let options = Options::parse(command, args)?;
    let checkpoint = Path::new(&options.checkpoint);
    let has_adapter = checkpoint.join("adapter_config.json").is_file();
    let adapter =
        match (options.get("--adapter"), has_adapter) {
            (Some(_), true) => return Err(
                "ambiguous adapter: use either an adapter checkpoint with --base, or a base checkpoint with --adapter"
                    .into(),
            ),
            (Some(path), false) => Some(Path::new(path)),
            (None, true) => Some(checkpoint),
            (None, false) => None,
        };
    let base = if has_adapter {
        options.get("--base").map(Path::new).ok_or("an adapter checkpoint needs --base <checkpoint-dir>; remote base_model_name_or_path is not downloaded implicitly")?
    } else {
        if options.get("--base").is_some_and(|path| Path::new(path) != checkpoint) {
            return Err("--base is only needed when <checkpoint-dir> is an adapter; use <base-dir> --adapter <adapter-dir> otherwise".into());
        }
        checkpoint
    };
    let encoder_layout = base.join("encoder/config.json").is_file();
    let agent_layout = base.join("rl_agent_config.json").is_file();
    if encoder_layout != agent_layout {
        return Err("an encoder decision checkpoint requires both encoder/config.json and rl_agent_config.json".into());
    }
    let config_path = base.join(if encoder_layout { "encoder/config.json" } else { "config.json" });
    let config_text = text(&config_path)?;
    let config = Value::parse(&config_text).map_err(|error| format!("{}: {error}", config_path.display()))?;
    let arch = architecture(&config)?;
    reject_quantized_or_moe(&config)?;
    if arch == Architecture::Qwen35 {
        validate_qwen_config(&config)?;
    }
    if encoder_layout != (arch == Architecture::Encoder) {
        return Err("ModernBERT conversion requires the encoder decision-head checkpoint layout".into());
    }
    if options.alias.is_some_and(|expected| expected != arch) {
        return Err(format!("{command} does not match this checkpoint's architecture; use kevala convert"));
    }
    let readout = check_readout(arch, adapter.is_some(), options.get("--readout"))?;
    if options.get("--keep-f32").is_some() && readout != Readout::EncoderHead {
        return Err("--keep-f32 is currently supported only by encoder-head conversion".into());
    }
    if adapter.is_none() && (base.join("head.pt").exists() || base.join("adapter_model.safetensors").exists()) {
        return Err(
            "incomplete pointer adapter: expected adapter_config.json, adapter_model.safetensors, and head.pt".into()
        );
    }
    if let Some(dir) = adapter {
        validate_adapter(&json_file(&dir.join("adapter_config.json"))?)?;
        for file in ["adapter_model.safetensors", "head.pt"] {
            if !dir.join(file).is_file() {
                return Err(format!("pointer adapter is missing {file}"));
            }
        }
    }
    let block: usize = options.get("--block").unwrap_or("32").parse().map_err(|_| "bad --block")?;
    let tokenizer_path = options.get("--tokenizer").map(PathBuf::from).unwrap_or_else(|| {
        adapter
            .map(|dir| dir.join("tokenizer.json"))
            .filter(|path| path.is_file())
            .unwrap_or_else(|| base.join(if encoder_layout { "tokenizer/tokenizer.json" } else { "tokenizer.json" }))
    });
    let mut tokenizer = text(&tokenizer_path)?;
    if arch == Architecture::Qwen35 && options.get("--tokenizer").is_none() {
        let config_path = tokenizer_path.with_file_name("tokenizer_config.json");
        if config_path.is_file() {
            tokenizer = kevala::tokenizer::Tokenizer::normalize_qwen2_json(&tokenizer, &text(&config_path)?)?;
        }
    }
    validate_tokenizer(arch, &tokenizer)?;
    let model = metadata(&options, &config, readout, block);
    let started = Instant::now();
    eprintln!("converting {arch:?} with {readout:?} readout");
    let pack = if readout == Readout::EncoderHead {
        let safetensors = std::fs::read(base.join("model.safetensors")).map_err(|e| e.to_string())?;
        let agent_config = text(&base.join("rl_agent_config.json"))?;
        kevala::convert::convert(
            &kevala::convert::Checkpoint {
                safetensors: &safetensors,
                encoder_config: &config_text,
                agent_config: &agent_config,
                tokenizer_json: &tokenizer,
            },
            &kevala::convert::Options {
                block,
                model,
                keep_f32: options
                    .get("--keep-f32")
                    .map(|value| value.split(',').map(str::to_string).collect())
                    .unwrap_or_default(),
            },
        )?
    } else {
        let shards = indexed_shards(base.to_str().ok_or("checkpoint path is not UTF-8")?)?;
        match arch {
            Architecture::Gemma4 => {
                kevala::gemma4::Gemma4Config::from_json(&config)?;
                convert_gemma_sharded(&shards, &config_text, &tokenizer, block, model)?
            }
            Architecture::Qwen35 => {
                let mode = if let Some(dir) = adapter {
                    StreamMode::Kev {
                        adapter: std::fs::read(dir.join("adapter_model.safetensors")).map_err(|e| e.to_string())?,
                        adapter_config: text(&dir.join("adapter_config.json"))?,
                        head: std::fs::read(dir.join("head.pt")).map_err(|e| e.to_string())?,
                    }
                } else {
                    StreamMode::Semif
                };
                convert_sharded(&shards, &config_text, &tokenizer, block, model, mode)?
            }
            Architecture::Encoder => unreachable!(),
        }
    };
    output_pack(options.get("-o").unwrap(), &pack, started)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(value: &str) -> Value {
        Value::parse(value).unwrap()
    }

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn architecture_uses_config_and_ignores_model_names_and_size() {
        assert_eq!(
            architecture(&json(r#"{"model_type":"modernbert","architectures":["ModernBertForMaskedLM"]}"#)).unwrap(),
            Architecture::Encoder
        );
        for (model_type, expected) in [("qwen3_5_text", Architecture::Qwen35), ("gemma4_text", Architecture::Gemma4)] {
            for name in ["my-custom-finetune", "laya", "Qwen/misleading-name"] {
                let value =
                    json(&format!(r#"{{"model_type":"{model_type}","_name_or_path":"{name}","hidden_size":128}}"#));
                assert_eq!(architecture(&value).unwrap(), expected);
            }
        }
        assert_eq!(
            architecture(&json(r#"{"model_type":"gemma4","text_config":{"model_type":"gemma4_text"}}"#)).unwrap(),
            Architecture::Gemma4
        );
        assert_eq!(architecture(&json(r#"{"architectures":["Qwen3_5ForCausalLM"]}"#)).unwrap(), Architecture::Qwen35);
    }

    #[test]
    fn unsupported_and_conflicting_architectures_fail() {
        for value in [
            r#"{"model_type":"qwen2","_name_or_path":"google/gemma-4-E2B-it"}"#,
            r#"{"hidden_size":1536}"#,
            r#"{"model_type":"gemma4","text_config":{"model_type":"qwen3_5_text"}}"#,
            r#"{"model_type":"qwen3_5","architectures":["CustomRemoteModel"]}"#,
        ] {
            assert!(architecture(&json(value)).is_err(), "{value}");
        }
        for value in [
            r#"{"quantization_config":{"quant_method":"fp8"}}"#,
            r#"{"text_config":{"enable_moe_block":true}}"#,
            r#"{"num_experts":16}"#,
        ] {
            assert!(reject_quantized_or_moe(&json(value)).is_err());
        }
        for value in [
            r#"{"attention_bias":true}"#,
            r#"{"attn_output_gate":false}"#,
            r#"{"hidden_act":"gelu"}"#,
            r#"{"rope_parameters":{"rope_type":"yarn"}}"#,
        ] {
            assert!(validate_qwen_config(&json(value)).is_err());
        }
    }

    #[test]
    fn readout_is_inferred_from_architecture_and_adapter_presence() {
        assert_eq!(check_readout(Architecture::Qwen35, false, None).unwrap(), Readout::DirectOptions);
        assert_eq!(check_readout(Architecture::Gemma4, false, Some("direct-options")).unwrap(), Readout::DirectOptions);
        assert_eq!(check_readout(Architecture::Qwen35, true, None).unwrap(), Readout::Pointer);
        assert_eq!(check_readout(Architecture::Encoder, false, None).unwrap(), Readout::EncoderHead);
        assert!(check_readout(Architecture::Gemma4, true, None).is_err());
        assert!(check_readout(Architecture::Qwen35, true, Some("direct-options")).is_err());
        assert!(check_readout(Architecture::Qwen35, false, Some("pointer")).is_err());
    }

    #[test]
    fn legacy_aliases_normalize_into_the_same_options() {
        let legacy = Options::parse(
            "convert-kev",
            &args(&["--base", "base", "--kev", "adapter", "-o", "out", "--kev-revision", "sha"]),
        )
        .unwrap();
        let current = Options::parse(
            "convert",
            &args(&["base", "--base", "base", "--adapter", "adapter", "-o", "out", "--revision", "sha"]),
        )
        .unwrap();
        assert_eq!(legacy.values, current.values);
        assert_eq!(legacy.checkpoint, current.checkpoint);
        for input in [
            &["base", "-o", "out", "--typo", "x"][..],
            &["base", "-o", "out", "--name"],
            &["base", "-o", "out", "--name", "a", "--name", "b"],
        ] {
            assert!(Options::parse("convert", &args(input)).is_err());
        }
    }

    #[test]
    fn metadata_does_not_fabricate_source_license_or_revision() {
        let options = Options::parse("convert", &args(&["/tmp/custom-model", "-o", "out"])).unwrap();
        let model =
            metadata(&options, &json(r#"{"model_type":"gemma4_text","hidden_size":2560}"#), Readout::DirectOptions, 32);
        assert_eq!(model.get("name").and_then(Value::as_str), Some("custom-model"));
        for key in ["source", "base", "revision", "base_revision", "license", "author"] {
            assert!(model.get(key).is_none(), "{key}");
        }
        let options = Options::parse(
            "convert",
            &args(&[
                "base",
                "-o",
                "out",
                "--name",
                "fine-tune",
                "--source",
                "me/my-model",
                "--revision",
                "pinned",
                "--base-source",
                "/tmp/local-base",
                "--base-revision",
                "base-sha",
                "--license",
                "mit",
            ]),
        )
        .unwrap();
        let model = metadata(&options, &json("{}"), Readout::Pointer, 32);
        assert_eq!(model.get("source").and_then(Value::as_str), Some("https://huggingface.co/me/my-model"));
        assert_eq!(model.get("base").and_then(Value::as_str), Some("/tmp/local-base"));
        assert_eq!(model.get("revision").and_then(Value::as_str), Some("pinned"));
        assert_eq!(model.get("license").and_then(Value::as_str), Some("mit"));
    }

    #[test]
    fn nonstandard_lora_scaling_and_saved_modules_fail() {
        assert!(validate_adapter(&json(
            r#"{"peft_type":"LORA","bias":"none","rank_pattern":{},"modules_to_save":null}"#
        ))
        .is_ok());
        for extra in [
            r#""use_dora":true"#,
            r#""use_rslora":true"#,
            r#""alpha_pattern":{"a":32}"#,
            r#""modules_to_save":["lm_head"]"#,
        ] {
            assert!(validate_adapter(&json(&format!(r#"{{"peft_type":"LORA",{extra}}}"#))).is_err());
        }
        assert!(validate_adapter(&json(r#"{"peft_type":"IA3"}"#)).is_err());
    }

    #[test]
    fn unsupported_checkpoint_fails_before_creating_output() {
        let dir = std::env::temp_dir().join(format!("kevala-dispatch-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let output = dir.join("not-created.kevala");
        std::fs::write(dir.join("config.json"), r#"{"model_type":"unsupported_model"}"#).unwrap();
        let result = run("convert", &args(&[dir.to_str().unwrap(), "-o", output.to_str().unwrap()]));
        assert!(result.unwrap_err().contains("unsupported checkpoint architecture"));
        assert!(!output.exists());
        std::fs::write(dir.join("config.json"), r#"{"model_type":"qwen3_5_text"}"#).unwrap();
        std::fs::write(dir.join("head.pt"), []).unwrap();
        let result = run("convert", &args(&[dir.to_str().unwrap(), "-o", output.to_str().unwrap()]));
        assert!(result.unwrap_err().contains("incomplete pointer adapter"));
        assert!(!output.exists());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
