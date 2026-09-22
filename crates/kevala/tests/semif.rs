//! SemIf prompt and branch layout parity against the pinned native reference.
//!
//! These checks only load the materialized Qwen tokenizer and the JSON golden cases. They do
//! not load model weights or exercise the numerical readout.

use std::sync::OnceLock;

use kevala::json::Value;
use kevala::kev::{encode_semif, parse_questions, Encoded, KevConfig};
use kevala::tokenizer::Tokenizer;

fn tokenizer() -> Option<&'static Tokenizer> {
    static TOK: OnceLock<Option<Tokenizer>> = OnceLock::new();
    TOK.get_or_init(|| {
        let path = std::env::var("KEVALA_QWEN_TOKENIZER_JSON")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../tmp/kev/tokenizer.json").into());
        match std::fs::read_to_string(&path) {
            Ok(s) => Some(Tokenizer::from_hf_json(&s).unwrap_or_else(|e| panic!("{path}: {e}"))),
            Err(_) => {
                eprintln!("skipping: no materialized Qwen tokenizer at {path}");
                None
            }
        }
    })
    .as_ref()
}

fn golden() -> Value {
    Value::parse(
        &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/golden-semif.json"))
            .unwrap(),
    )
    .unwrap()
}

fn case<'a>(golden: &'a Value, id: &str) -> &'a Value {
    golden
        .get("cases")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .find(|case| case.get("id").and_then(Value::as_str) == Some(id))
        .unwrap_or_else(|| panic!("missing SemIf golden case {id}"))
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|id| id.as_usize().unwrap() as u32).collect()
}

fn flatten(encoded: &Encoded) -> Vec<u32> {
    assert_eq!(encoded.branches.len(), 1);
    let mut ids = encoded.state.clone();
    ids.extend_from_slice(&encoded.branches[0].ids);
    ids
}

fn config(max_branch: usize) -> KevConfig {
    let json = format!(
        r#"{{
            "hidden_size": 1,
            "intermediate_size": 1,
            "rms_norm_eps": 0.000001,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 1,
            "rotary_dim": 1,
            "rope_theta": 1.0,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 1,
            "linear_value_head_dim": 1,
            "linear_conv_kernel_dim": 1,
            "pointer_dim": 1,
            "temperature": 1.0,
            "max_state": 8192,
            "max_branch": {max_branch},
            "layer_types": ["linear_attention"],
            "template": {{"token_ids": [0, 0, 0, 0, 0]}},
            "readout": "semif"
        }}"#
    );
    KevConfig::from_json(&Value::parse(&json).unwrap()).unwrap()
}

#[test]
fn golden_input_ids_and_final_positions_match() {
    let Some(tok) = tokenizer() else { return };
    let golden = golden();
    for case in golden.get("cases").unwrap().as_array().unwrap() {
        let id = case.get("id").unwrap().as_str().unwrap();
        let questions = case.get("questions").unwrap();
        let parsed = parse_questions(questions).unwrap();
        assert_eq!(parsed.len(), 1, "{id}");
        let encoded = encode_semif(tok, &config(8192), case.get("state").unwrap(), &parsed).unwrap();
        let want = ids(case.get("input_ids").unwrap());
        assert_eq!(flatten(&encoded), want, "{id}: input IDs");
        let branch = &encoded.branches[0];
        assert_eq!(encoded.state.len() + branch.decide, want.len() - 1, "{id}: final readout row");
        assert_eq!(branch.labels, parsed[0].options.len(), "{id}: label count");
        assert!(branch.opts.is_empty(), "{id}: SemIf has no pointer option rows");
    }
}

#[test]
fn multiple_questions_keep_the_exact_shared_prefix() {
    let Some(tok) = tokenizer() else { return };
    let golden = golden();
    let first = case(&golden, "readme_ticket::department");
    let second = case(&golden, "readme_ticket::escalate");
    let questions = Value::Object(vec![
        ("department".into(), first.get("questions").unwrap().get("department").unwrap().clone()),
        ("escalate".into(), second.get("questions").unwrap().get("escalate").unwrap().clone()),
    ]);
    let parsed = parse_questions(&questions).unwrap();
    let encoded = encode_semif(tok, &config(8192), first.get("state").unwrap(), &parsed).unwrap();
    let expected_prefix = first.get("cache_prefix_tokens").unwrap().as_usize().unwrap();
    assert_eq!(encoded.state.len(), expected_prefix);

    for (branch, expected) in encoded.branches.iter().zip([first, second]) {
        let mut full = encoded.state.clone();
        full.extend_from_slice(&branch.ids);
        let want = ids(expected.get("input_ids").unwrap());
        assert_eq!(full, want, "{}: branch IDs", expected.get("id").unwrap().as_str().unwrap());
        assert_eq!(encoded.state.len() + branch.decide, want.len() - 1, "branch final row");
    }
}

#[test]
fn rejects_more_than_sixteen_options() {
    let Some(tok) = tokenizer() else { return };
    let criteria = (0..17).map(|i| (format!("option-{i}"), Value::Str(format!("description-{i}")))).collect();
    let questions = Value::Object(vec![(
        "q".into(),
        Value::Object(vec![
            ("type".into(), Value::Str("choice".into())),
            ("instructions".into(), Value::Str("Choose one.".into())),
            ("criteria".into(), Value::Object(criteria)),
        ]),
    )]);
    let parsed = parse_questions(&questions).unwrap();
    let error = encode_semif(tok, &config(8192), &Value::Str("evidence".into()), &parsed).unwrap_err();
    assert!(error.contains("at most 16"), "{error}");
}

#[test]
fn rejects_overlong_prompts_without_truncating() {
    let Some(tok) = tokenizer() else { return };
    let golden = golden();
    let sample = case(&golden, "readme_ticket::department");
    let parsed = parse_questions(sample.get("questions").unwrap()).unwrap();
    let error = encode_semif(tok, &config(64), sample.get("state").unwrap(), &parsed).unwrap_err();
    assert!(error.contains("no truncation"), "{error}");
}
