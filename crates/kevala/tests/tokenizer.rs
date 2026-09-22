//! Token ids against the Hugging Face libraries, for each tokenizer kevala runs:
//!
//! - ModernBERT (Laya), tests/fixtures/tokenizer.json from `tokenizers`. Needs
//!   tmp/laya/tokenizer/tokenizer.json or $KEVALA_TOKENIZER_JSON.
//! - Qwen3.5 (Kev), tests/fixtures/tokenizer-qwen35.json from transformers' `AutoTokenizer` at the
//!   revision Kev pins. Needs the tokenizer.json AutoTokenizer materializes, which Kev
//!   checkpoints ship: tmp/kev/tokenizer.json or $KEVALA_QWEN_TOKENIZER_JSON.
//! - The Qwen3.5 base repo's own tokenizer.json (its `\p{M}` regex, 22 added tokens),
//!   tests/fixtures/tokenizer-qwen35-base.json from `tokenizers`. Needs tmp/qwen35/tokenizer.json
//!   or $KEVALA_QWEN_BASE_TOKENIZER_JSON.
//!
//! None of those files is checked in; without one its tests print a notice and pass.

use std::sync::OnceLock;

use kevala::json::Value;
use kevala::tokenizer::Tokenizer;

const MODERNBERT: (&str, &str) = ("KEVALA_TOKENIZER_JSON", "tmp/laya/tokenizer/tokenizer.json");
const QWEN: (&str, &str) = ("KEVALA_QWEN_TOKENIZER_JSON", "tmp/kev/tokenizer.json");
const QWEN_BASE: (&str, &str) = ("KEVALA_QWEN_BASE_TOKENIZER_JSON", "tmp/qwen35/tokenizer.json");

fn load(cell: &'static OnceLock<Option<Tokenizer>>, (var, default): (&str, &str)) -> Option<&'static Tokenizer> {
    cell.get_or_init(|| {
        let path = std::env::var(var).unwrap_or_else(|_| format!("{}/../../{default}", env!("CARGO_MANIFEST_DIR")));
        match std::fs::read_to_string(&path) {
            Ok(s) => Some(Tokenizer::from_hf_json(&s).unwrap_or_else(|e| panic!("{path}: {e}"))),
            Err(_) => {
                eprintln!("skipping: no tokenizer.json at {path} (set {var})");
                None
            }
        }
    })
    .as_ref()
}

fn modernbert() -> Option<&'static Tokenizer> {
    static TOK: OnceLock<Option<Tokenizer>> = OnceLock::new();
    load(&TOK, MODERNBERT)
}

fn qwen() -> Option<&'static Tokenizer> {
    static TOK: OnceLock<Option<Tokenizer>> = OnceLock::new();
    load(&TOK, QWEN)
}

fn qwen_base() -> Option<&'static Tokenizer> {
    static TOK: OnceLock<Option<Tokenizer>> = OnceLock::new();
    load(&TOK, QWEN_BASE)
}

fn corpus(name: &str) -> Vec<(String, Vec<u32>)> {
    let path = format!("{}/../../tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    let v = Value::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    v.as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let ids = c.get("ids").unwrap().as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as u32).collect();
            (c.get("text").unwrap().as_str().unwrap().to_string(), ids)
        })
        .collect()
}

fn assert_corpus(tok: &Tokenizer, name: &str) {
    let corpus = corpus(name);
    let bad: Vec<_> = corpus.iter().filter(|(text, ids)| tok.encode(text) != *ids).collect();
    if let Some((text, ids)) = bad.first() {
        panic!(
            "{name}: {} of {} differ; first {text:?}: want {ids:?}, got {:?}",
            bad.len(),
            corpus.len(),
            tok.encode(text)
        );
    }
    eprintln!("{name}: {} corpus strings match", corpus.len());
}

fn assert_round_trip(tok: &Tokenizer, name: &str) {
    let blob = tok.to_bytes();
    let t = std::time::Instant::now();
    let back = Tokenizer::from_bytes(&blob).unwrap();
    eprintln!("{name}: blob {} bytes, from_bytes {:?}", blob.len(), t.elapsed());
    assert_eq!(back.to_bytes(), blob);
    for (text, _) in corpus(name) {
        assert_eq!(back.encode(&text), tok.encode(&text), "{text:?}");
    }
    assert_eq!(
        (back.cls_id(), back.sep_id(), back.mask_id(), back.pad_id(), back.vocab_size()),
        (tok.cls_id(), tok.sep_id(), tok.mask_id(), tok.pad_id(), tok.vocab_size())
    );
    assert!(Tokenizer::from_bytes(&blob[..blob.len() - 1]).is_err());
}

#[test]
fn matches_hf_tokenizers() {
    let Some(tok) = modernbert() else { return };
    assert_corpus(tok, "tokenizer.json");
}

#[test]
fn special_ids() {
    let Some(tok) = modernbert() else { return };
    assert_eq!((tok.cls_id(), tok.sep_id(), tok.mask_id(), tok.pad_id()), (50281, 50282, 50284, 50283));
    assert_eq!(tok.vocab_size(), 50368);
    assert_eq!(tok.token_id("[unused82]"), Some(50367));
    assert_eq!(tok.token_id("  "), Some(50276));
    assert_eq!(tok.token_id("Ġthe"), Some(253));
    assert_eq!(tok.token_id("<|padding|>"), Some(1));
    assert_eq!(tok.token_id("no such token"), None);
}

#[test]
fn blob_round_trip() {
    let Some(tok) = modernbert() else { return };
    assert_round_trip(tok, "tokenizer.json");
    assert!(Tokenizer::from_bytes(b"not a tokenizer").is_err());
}

#[test]
fn encode_prefix_is_a_prefix() {
    let Some(tok) = modernbert() else { return };
    for (text, ids) in corpus("tokenizer.json") {
        for k in [0, 1, 2, 5, ids.len().saturating_sub(1), ids.len(), ids.len() + 3] {
            assert_eq!(tok.encode_prefix(&text, k), ids[..k.min(ids.len())], "{text:?} k={k}");
        }
    }
}

#[test]
fn qwen_matches_autotokenizer() {
    let Some(tok) = qwen() else { return };
    assert_corpus(tok, "tokenizer-qwen35.json");
    assert_eq!(tok.vocab_size(), 248077);
    assert_eq!(tok.token_id("<|fim_prefix|>"), Some(248060));
    assert_eq!(tok.token_id("<think>"), Some(248068));
    assert_eq!(tok.cls_id(), u32::MAX);
}

#[test]
fn qwen_blob_round_trip() {
    let Some(tok) = qwen() else { return };
    assert_round_trip(tok, "tokenizer-qwen35.json");
}

#[test]
fn qwen_base_file_with_marks_regex() {
    let Some(tok) = qwen_base() else { return };
    assert_corpus(tok, "tokenizer-qwen35-base.json");
    assert_round_trip(tok, "tokenizer-qwen35-base.json");
}

#[test]
fn rejects_unknown_pre_tokenizer() {
    let json = r#"{"pre_tokenizer": {"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "Isolated", "invert": false},
        {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}]},
        "model": {"type": "BPE", "vocab": {}, "merges": []}}"#;
    let err = Tokenizer::from_hf_json(json).err().unwrap();
    assert!(err.contains("unsupported pre_tokenizer"), "{err}");
}
