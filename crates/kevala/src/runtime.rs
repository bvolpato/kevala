//! The model-family registry.
//!
//! A pack's `config.arch` names the family that reads it. A family owns three pluggable pieces:
//! a template (how a request becomes token sequences), a backbone (the layers), and a head (how
//! hidden states become answer distributions); plus the modalities it reads. Every family answers
//! the same request shape (`state`, optional typed `parts`, typed `questions`) and returns its own
//! reference response format. Adding a family is one `Model` implementation and one `FAMILIES`
//! entry; nothing else in the engine, the WebAssembly ABI or the pack format changes.
//!
//! | arch | template | backbone | head |
//! |---|---|---|---|
//! | `laya` | one bidirectional sequence per question, a `[MASK]` per option | ModernBERT encoder | 2-layer transformer, marker scorer, act head |
//! | `kev` | one causal row per question sharing the state | Qwen3.5 (Gated DeltaNet + gated attention) | pointer head |

use crate::content::{Modality, Request};
use crate::json::Value;
use crate::model::AlignedBuf;
use crate::pack;
use crate::tokenizer::Tokenizer;
use std::any::Any;

/// A loaded decision model of any family.
pub trait Model: Any {
    fn arch(&self) -> &'static str;
    /// Provenance from the pack header.
    fn info(&self) -> &Value;
    fn modalities(&self) -> &[Modality];
    fn tokenizer(&self) -> &Tokenizer;
    /// Answers every request in one forward pass; one response per request.
    fn decide(&mut self, requests: &[Request]) -> Result<Vec<Value>, String>;
    /// For backends that drive a family's pieces themselves (GPU, tensor-parallel shards).
    fn as_any(&mut self) -> &mut dyn Any;
}

pub struct Family {
    pub arch: &'static str,
    pub about: &'static str,
    pub load: fn(AlignedBuf) -> Result<Box<dyn Model>, String>,
}

fn load_laya(b: AlignedBuf) -> Result<Box<dyn Model>, String> {
    Ok(Box::new(crate::engine::Engine::load(b)?))
}

fn load_kev(b: AlignedBuf) -> Result<Box<dyn Model>, String> {
    Ok(Box::new(crate::kev::KevEngine::load(b)?))
}

fn load_gemma4(b: AlignedBuf) -> Result<Box<dyn Model>, String> {
    Ok(Box::new(crate::gemma4::Gemma4Engine::load(b)?))
}

pub const FAMILIES: &[Family] = &[
    Family { arch: "laya", about: "ModernBERT encoder + decision head (convaiinnovations/laya)", load: load_laya },
    Family { arch: "kev", about: "Qwen3.5 hybrid decoder + pointer head (jaredpalmer/kev)", load: load_kev },
    Family { arch: "gemma4", about: "Gemma 4 dense text decoder + direct option scoring (Google)", load: load_gemma4 },
];

/// The family a pack declares (`laya` when the header predates `config.arch`).
pub fn arch_of(header: &pack::Header) -> String {
    header.config().get("arch").and_then(Value::as_str).unwrap_or("laya").to_string()
}

/// Loads any pack whose family this build knows.
pub fn load(buf: AlignedBuf) -> Result<Box<dyn Model>, String> {
    let h = pack::parse_header(buf.as_slice())?;
    let arch = arch_of(&h);
    match FAMILIES.iter().find(|f| f.arch == arch) {
        Some(f) => (f.load)(buf),
        None => Err(format!(
            "this build of kevala does not know the {arch:?} architecture (it knows {})",
            FAMILIES.iter().map(|f| f.arch).collect::<Vec<_>>().join(", ")
        )),
    }
}

/// Checks modalities and resolves text parts into the state each template renders.
pub fn text_requests(
    arch: &str,
    supported: &[Modality],
    requests: &[Request],
    render: impl Fn(&Value) -> String,
) -> Result<Vec<(Value, Value)>, String> {
    requests
        .iter()
        .map(|r| {
            r.check(arch, supported)?;
            let state = r.text_with(&render).map(Value::Str).unwrap_or_else(|| r.state.clone());
            Ok((state, r.questions.clone()))
        })
        .collect()
}
