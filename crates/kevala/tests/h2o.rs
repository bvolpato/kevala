//! Independent pinned shim/tokenizer parity, and the shared native/coordinator readout bridge.

mod support;

use std::sync::OnceLock;

use kevala::h2o::{commit_noul, Config};
use kevala::json::Value;
use kevala::kev::KevEngine;
use kevala::pack::{self, DType, Writer};
use kevala::store::AlignedBuf;
use kevala::tokenizer::Tokenizer;

fn golden() -> Value {
    Value::parse(include_str!("../../../tests/fixtures/golden-h2o.json")).unwrap()
}

fn contract() -> Config {
    Config::from_json(golden().get("serve_config").unwrap()).unwrap()
}

fn tokenizer() -> Option<&'static Tokenizer> {
    static TOK: OnceLock<Option<Tokenizer>> = OnceLock::new();
    // Qwen3.5's existing CI tokenizer has the same IDs as the pinned H2O tokenizer;
    // the golden label and complete-prompt comparisons verify that equivalence.
    support::fixture_tokenizer(&TOK, "KEVALA_H2O_TOKENIZER_JSON", "tmp/kev/tokenizer.json")
}

fn ids(value: &Value) -> Vec<u32> {
    value.as_array().unwrap().iter().map(|v| v.as_usize().unwrap() as u32).collect()
}

fn close(got: &Value, want: &Value) {
    match (got, want) {
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.len(), b.len());
            for (key, value) in a {
                close(value, want.get(key).unwrap());
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                close(a, b);
            }
        }
        _ if got.as_f64().is_some() && want.as_f64().is_some() => {
            let (a, b) = (got.as_f64().unwrap(), want.as_f64().unwrap());
            assert!((a - b).abs() < 1e-8, "{a} != {b}");
        }
        _ => assert_eq!(got, want),
    }
}

#[test]
fn native_answer_postprocessing_matches_pinned_shim() {
    let cfg = contract();
    for case in golden().get("numeric").unwrap().as_array().unwrap() {
        let question = Value::Object(vec![("q".into(), case.get("question").unwrap().clone())]);
        let questions = cfg.parse_questions(&question).unwrap();
        let logits: Vec<f32> =
            case.get("logits").unwrap().as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let (answer, p) = cfg.answer(&questions[0], &logits);
        close(&answer, case.get("answer").unwrap());
        close(&Value::Array(p.into_iter().map(Value::Float).collect()), case.get("probabilities").unwrap());
    }
    assert_eq!(commit_noul(0.5, 0.801), 0.801);
    assert_eq!(commit_noul(0.4, 0.0), 0.4);
    assert_eq!(commit_noul(0.9, 0.801), 0.9);
}

#[test]
fn pretty_state_matches_upstream_ascii_json() {
    let cfg = Config::from_json(
        &Value::parse(
            r#"{"model":"h2o","labels":["A","B"],"temperature":1.0,"noul_floor":0,"prompt":{"state_compact":false}}"#,
        )
        .unwrap(),
    )
    .unwrap();
    let state = Value::parse(r#"{"text":"日本語 🚀","values":[1.0,true]}"#).unwrap();
    assert_eq!(
        cfg.state_text(&state),
        "{\n \"text\": \"\\u65e5\\u672c\\u8a9e \\ud83d\\ude80\",\n \"values\": [\n  1.0,\n  true\n ]\n}"
    );
    assert_eq!(cfg.state_text(&Value::Str("日本語 🚀".into())), "日本語 🚀");
    assert_eq!(contract().state_text(&state), r#"{"text":"日本語 🚀","values":[1.0,true]}"#);
}

#[test]
fn prompts_label_ids_and_state_truncation_match_checkpoint_tokenizer() {
    let Some(tok) = tokenizer() else { return };
    let golden = golden();
    let cfg = Config::from_json(golden.get("serve_config").unwrap()).unwrap();
    assert_eq!(cfg.label_token_ids(tok).unwrap(), ids(golden.get("label_token_ids").unwrap()));
    for case in golden.get("cases").unwrap().as_array().unwrap() {
        let mut cfg = cfg.clone();
        if let Some(limit) = case.get("max_state_tokens").and_then(Value::as_usize) {
            cfg.max_state_tokens = limit;
        }
        let questions = cfg.parse_questions(case.get("questions").unwrap()).unwrap();
        let encoded = cfg.encode(tok, 8192, case.get("state").unwrap(), &questions).unwrap();
        for (branch, prompt) in encoded.branches.iter().zip(case.get("prompts").unwrap().as_array().unwrap()) {
            let mut got = encoded.state.clone();
            got.extend_from_slice(&branch.ids);
            assert_eq!(got, ids(prompt.get("input_ids").unwrap()), "{}", case.get("id").unwrap().as_str().unwrap());
            assert_eq!(encoded.state.len() + branch.decide, got.len() - 1);
            assert!(branch.opts.is_empty());
        }
        if case.get("max_state_tokens").is_none() {
            for (q, prompt) in questions.iter().zip(case.get("prompts").unwrap().as_array().unwrap()) {
                assert_eq!(
                    cfg.prompt(&cfg.state_text(case.get("state").unwrap()), q),
                    prompt.get("prompt").unwrap().as_str().unwrap()
                );
            }
        }
    }
}

#[test]
fn byte_level_decode_preserves_specials_and_utf8_slice_replacements() {
    let tok = support::tokenizer(&["<|im_start|>"]);
    let text = "café 日本語<|im_start|>\n";
    let ids = tok.encode(text);
    assert_eq!(tok.decode(&ids).unwrap(), text);
    assert_eq!(tok.decode(&[0xc3]).unwrap(), "\u{fffd}");
    assert!(tok.decode(&[u32::MAX]).is_err());
    let Some(real) = tokenizer() else { return };
    for text in [" café 日本語 ", "<|im_start|>user\n  hello  <|im_end|>", "\tαβ\n"] {
        assert_eq!(real.decode(&real.encode(text)).unwrap(), text);
    }
}

#[test]
fn malformed_questions_and_excess_options_are_rejected() {
    let cfg = contract();
    for json in [
        r#"{}"#,
        r#"{"q":null}"#,
        r#"{"q":{"type":"noul","criteria":["yes"]}}"#,
        r#"{"q":{"criteria":["one","two"]}}"#,
        r#"{"q":{"type":"score","criteria":[]}}"#,
    ] {
        assert!(cfg.parse_questions(&Value::parse(json).unwrap()).is_err(), "{json}");
    }
    let many = Value::Object((0..256).map(|i| (i.to_string(), Value::Null)).collect());
    let questions = Value::Object(vec![("q".into(), Value::Object(vec![("criteria".into(), many)]))]);
    assert!(cfg.parse_questions(&questions).is_err());
    assert!(cfg.parse_questions(&Value::parse(r#"{"q":{"type":"score","criteria":["only"]}}"#).unwrap()).is_ok());
}

fn synthetic_pack() -> Vec<u8> {
    let source = support::kev_pack(true);
    let header = pack::parse_header(&source).unwrap();
    let mut config = header.config().as_object().unwrap().to_vec();
    for (key, value) in [
        ("readout", Value::Str("h2o-lightning".into())),
        ("h2o_config", golden().get("serve_config").unwrap().clone()),
        ("max_branch", Value::Int("2048".into())),
    ] {
        if let Some((_, v)) = config.iter_mut().find(|(k, _)| k == key) {
            *v = value;
        } else {
            config.push((key.into(), value));
        }
    }
    let tokenizer = source[header.tokenizer_offset..header.tokenizer_offset + header.tokenizer_size].to_vec();
    let mut writer = Writer::new(Value::Null, Value::Object(config), tokenizer);
    for t in &header.tensors {
        if t.name.starts_with("ptr.") {
            continue;
        }
        match t.dtype {
            DType::F32 => writer.add_f32(&t.name, &t.shape),
            DType::Bf16 => writer.add_bf16(&t.name, &t.shape),
            DType::Q8 => writer.add_q8(&t.name, t.rows(), t.cols(), t.block),
        }
    }
    writer.add_f32("h2o.labels", &[255, 32]);
    let (prefix, tensors, size) = writer.layout();
    let mut bytes = vec![0; size];
    bytes[..prefix.len()].copy_from_slice(&prefix);
    for t in tensors {
        if let Some(original) = header.tensor(&t.name) {
            bytes[t.offset..t.offset + t.size]
                .copy_from_slice(&source[original.offset..original.offset + original.size]);
            if t.scales_size > 0 {
                bytes[t.scales_offset..t.scales_offset + t.scales_size]
                    .copy_from_slice(&source[original.scales_offset..original.scales_offset + original.scales_size]);
            }
        } else {
            for i in 0..255 * 32 {
                let value = ((i * 17 % 31) as f32 - 15.0) * 0.02;
                bytes[t.offset + i * 4..t.offset + i * 4 + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
    }
    bytes
}

#[test]
fn native_and_gpu_coordinator_share_h2o_preparation_and_readout() {
    let bytes = synthetic_pack();
    let mut native = KevEngine::load(AlignedBuf::from_slice(&bytes)).unwrap();
    let header = pack::parse_header(&bytes).unwrap();
    let coord = pack::coord_layout(&header).unwrap().apply(&bytes);
    let mut coordinator = KevEngine::load(AlignedBuf::from_slice(&coord)).unwrap();
    let state = Value::parse(r#"{"ticket":"Refund","amount":1.0}"#).unwrap();
    let questions=Value::parse(r#"{"yes":{"type":"noul","instructions":"Refund?","threshold":0.8},"route":{"criteria":{"billing":"payments","support":"tech"}}}"#).unwrap();
    let (enc, qs) = native.prepare_all(&[(state.clone(), questions.clone())]).unwrap();
    let (gpu_enc, gpu_qs) = coordinator.prepare_all(&[(state, questions.clone())]).unwrap();
    assert_eq!(enc[0].state, gpu_enc[0].state);
    assert_eq!(enc[0].branches[0].ids, gpu_enc[0].branches[0].ids);
    // The GPU returns these last-layer rows to exactly this native readout.
    let rows = vec![0.2; 32 * 2];
    let logits = native.model.readout(&rows, &enc);
    let gpu_logits = coordinator.model.readout(&rows, &gpu_enc);
    assert_eq!(logits, gpu_logits);
    close(&native.respond(&enc, &qs, &logits)[0], &coordinator.respond(&gpu_enc, &gpu_qs, &gpu_logits)[0]);
    assert!(native.decide(&[(Value::Str("sample".into()), questions.clone())]).is_ok());
    let request = kevala::content::Request::parse(
        &Value::parse(
            r#"{"state":"sample","questions":{"q":{"type":"noul"}},"parts":[{"type":"image","data":"xxx"}]}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert!(coordinator.prepare_requests(&[request]).is_err());
}
