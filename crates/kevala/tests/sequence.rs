//! Sequences against the PyTorch / Hugging Face reference pipeline (tests/fixtures/golden.json).
//!
//! The golden test needs Laya's tokenizer.json, which is not checked in:
//! tmp/laya/tokenizer/tokenizer.json or $KEVALA_TOKENIZER_JSON. Without it it prints a notice and
//! passes.

use std::sync::OnceLock;

use kevala::json::Value;
use kevala::sequence::{build_sequence, parse_questions, serialize_state, QType};
use kevala::tokenizer::Tokenizer;

fn tokenizer() -> Option<&'static Tokenizer> {
    static TOK: OnceLock<Option<Tokenizer>> = OnceLock::new();
    TOK.get_or_init(|| {
        let path = std::env::var("KEVALA_TOKENIZER_JSON")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../tmp/laya/tokenizer/tokenizer.json").into());
        match std::fs::read_to_string(&path) {
            Ok(s) => Some(Tokenizer::from_hf_json(&s).expect("tokenizer.json should load")),
            Err(_) => {
                eprintln!("skipping: no tokenizer.json at {path} (set KEVALA_TOKENIZER_JSON)");
                None
            }
        }
    })
    .as_ref()
}

fn nums(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(|x| x.as_usize().unwrap()).collect()
}

#[test]
fn golden_sequences() {
    let Some(tok) = tokenizer() else { return };
    let s = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/golden.json")).unwrap();
    let golden = Value::parse(&s).unwrap();
    let cases = golden.get("cases").unwrap().as_array().unwrap();
    let mut n = 0;
    for case in cases {
        let id = case.get("id").unwrap().as_str().unwrap();
        let expect = case.get("expect").unwrap();
        let qs = parse_questions(case.get("questions").unwrap()).unwrap();
        let text = serialize_state(case.get("state").unwrap());
        assert_eq!(qs.len(), expect.as_object().unwrap().len(), "{id}");
        for q in &qs {
            let e = expect.get(&q.id).unwrap();
            let want_opts: Vec<&str> =
                e.get("options").unwrap().as_array().unwrap().iter().map(|o| o.as_str().unwrap()).collect();
            assert_eq!(q.options, want_opts, "{id}/{}: options", q.id);
            let seq = build_sequence(tok, &text, q, 512, 192, false);
            let want_ids: Vec<u32> = nums(e.get("input_ids").unwrap()).into_iter().map(|i| i as u32).collect();
            assert_eq!(seq.ids, want_ids, "{id}/{}: input_ids", q.id);
            assert_eq!(seq.markers, nums(e.get("marker_pos").unwrap()), "{id}/{}: marker_pos", q.id);
            n += 1;
        }
    }
    eprintln!("{} golden questions in {} cases match", n, cases.len());
}

#[test]
fn truncate_left_keeps_the_tail() {
    let Some(tok) = tokenizer() else { return };
    let qs =
        parse_questions(&Value::parse(r#"{"q": {"type": "noul", "instructions": "Is it late?"}}"#).unwrap()).unwrap();
    let state = "one two three four five six seven eight nine ten ".repeat(20);
    let st = tok.encode(&state);
    let full = build_sequence(tok, &state, &qs[0], 4096, 192, false);
    let prefix = full.ids.len() - st.len() - 1;

    let seq = build_sequence(tok, &state, &qs[0], prefix + 11, 192, true);
    assert_eq!(seq.ids[prefix..], [&st[st.len() - 10..], &[tok.sep_id()]].concat());
    // room == 0: Python's st[-0:] is the whole list, so a state token (not [SEP]) ends up last
    let seq = build_sequence(tok, &state, &qs[0], prefix + 1, 192, true);
    assert_eq!(seq.ids.last(), Some(&st[0]));
    let seq = build_sequence(tok, &state, &qs[0], prefix + 1, 192, false);
    assert_eq!(seq.ids.last(), Some(&tok.sep_id()));
}

fn one(q: &str) -> Result<kevala::sequence::Question, String> {
    parse_questions(&Value::parse(&format!(r#"{{"q": {q}}}"#)).unwrap()).map(|mut v| v.remove(0))
}

#[test]
fn question_parsing() {
    let q = one(r#"{"type": "choice", "instructions": {"ask": "é?"}, "criteria": ["a", 1, 2.5, true, null, "a"]}"#)
        .unwrap();
    assert_eq!(q.qtype, QType::Choice);
    assert_eq!(q.instructions, "{\"ask\": \"\\u00e9?\"}");
    assert_eq!(q.keys, ["a", "1", "2.5", "True", "None"]);
    assert_eq!(q.options, q.keys);

    let q = one(
        r#"{"type": "choice", "instructions": "x", "criteria": {"low": {"d": "é"}, "mid": 0, "hi": false, "no": ""}}"#,
    )
    .unwrap();
    assert_eq!(q.options, [r#"low: {"d": "é"}"#, "mid: 0", "hi: false", "no"]);

    let q = one(r#"{"type": "score", "instructions": "x", "criteria": ["bad", {"l": 1}]}"#).unwrap();
    assert_eq!(
        (q.keys.clone(), q.legend.clone()),
        (vec!["0".into(), "1".into()], vec!["bad".into(), r#"{"l": 1}"#.into()])
    );
    assert_eq!(q.options, ["level 0: bad", r#"level 1: {"l": 1}"#]);

    for crit in ["null", "{}", "[]", "0", "\"\""] {
        let q = one(&format!(r#"{{"type": "noul", "instructions": "x", "criteria": {crit}}}"#)).unwrap();
        assert_eq!(q.options, ["false: no, the statement does not hold", "true: yes, the statement holds"]);
    }
    let q = one(r#"{"type": "noul", "instructions": "x", "criteria": {"true": 0, "false": ""}}"#).unwrap();
    assert_eq!(q.options, ["false: no, the statement does not hold", "true: 0"]);
    assert_eq!(q.keys, ["false", "true"]);

    for bad in [
        r#"{"type": "maybe", "instructions": "x"}"#,
        r#"{"instructions": "x"}"#,
        r#"{"type": "noul"}"#,
        r#"{"type": "choice", "instructions": "x"}"#,
        r#"{"type": "choice", "instructions": "x", "criteria": {}}"#,
        r#"{"type": "choice", "instructions": "x", "criteria": [["nested"]]}"#,
        r#"{"type": "score", "instructions": "x", "criteria": []}"#,
        r#"{"type": "score", "instructions": "x", "criteria": 3}"#,
        r#"{"type": "noul", "instructions": "x", "criteria": ["a"]}"#,
    ] {
        assert!(one(bad).is_err(), "{bad}");
    }
    assert!(parse_questions(&Value::parse("{}").unwrap()).is_err());
}

#[test]
fn state_serialization() {
    let v = Value::parse(r#"{"a": "café", "n": [1, 1.0, 1e20, null, true]}"#).unwrap();
    assert_eq!(serialize_state(&v), r#"{"a": "café", "n": [1, 1.0, 1e+20, null, true]}"#);
    assert_eq!(serialize_state(&Value::Str("raw [MASK] text".into())), "raw [MASK] text");
}
