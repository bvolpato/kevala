//! Exact upstream head and encoding parity with a tiny deterministic PyTorch reference.
mod support;

use kevala::clef::{
    encoding::{self, Question, Record},
    head::{Head, HeadConfig},
};
use kevala::json::Value;
use kevala::pack::Writer;
use kevala::store::{load_store, AlignedBuf};
use std::sync::Arc;

fn fixture() -> Value {
    Value::parse(include_str!("../../../tests/fixtures/clef-head.json")).unwrap()
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}
fn dimensions(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(|x| x.as_usize().unwrap()).collect()
}
fn span(v: &Value) -> (usize, usize) {
    let a = dimensions(v);
    (a[0], a[1])
}

fn reference() -> (Head, Record, Vec<f32>, Vec<Vec<f32>>, Vec<f32>) {
    let f = fixture();
    let cfg = HeadConfig::from_json(f.get("config").unwrap()).unwrap();
    let mut writer = Writer::new(Value::Null, Value::Null, vec![]);
    let tensors = f.get("tensors").unwrap().as_object().unwrap();
    for (name, t) in tensors {
        writer.add_f32(&format!("clef.{name}"), &dimensions(t.get("shape").unwrap()));
    }
    writer.add_f32("clef.lexical", &dimensions(f.get("lexical_shape").unwrap()));
    let (prefix, infos, total) = writer.layout();
    let mut pack = vec![0u8; total];
    pack[..prefix.len()].copy_from_slice(&prefix);
    for info in infos {
        let values = if info.name == "clef.lexical" {
            floats(f.get("lexical").unwrap())
        } else {
            floats(
                f.get("tensors").unwrap().get(info.name.strip_prefix("clef.").unwrap()).unwrap().get("values").unwrap(),
            )
        };
        for (i, v) in values.iter().enumerate() {
            pack[info.offset + i * 4..info.offset + (i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    let (store, _) = load_store(AlignedBuf::from_slice(&pack)).unwrap();
    let head = Head::new(cfg, Arc::new(store)).unwrap();
    let questions = f
        .get("questions")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|q| {
            let keys: Vec<_> = q
                .get("option_ids")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            Question {
                id: q.get("id").unwrap().as_str().unwrap().into(),
                kind: q.get("kind").unwrap().as_usize().unwrap(),
                span: span(q.get("span").unwrap()),
                option_spans: q.get("option_spans").unwrap().as_array().unwrap().iter().map(span).collect(),
                option_ids: keys.clone(),
                response_ids: keys,
                criteria: Value::Null,
            }
        })
        .collect();
    let record =
        Record { ids: dimensions(f.get("input_ids").unwrap()).into_iter().map(|id| id as u32).collect(), questions };
    let hidden = floats(f.get("hidden").unwrap());
    let expected = f.get("logits").unwrap().as_array().unwrap().iter().map(floats).collect();
    (head, record, hidden, expected, floats(f.get("lexical").unwrap()))
}

#[test]
fn joint_head_matches_unmodified_upstream_pytorch() {
    let (mut head, record, hidden, expected, _) = reference();
    let got = head.forward(&hidden, &record).unwrap();
    for (got, want) in got.iter().zip(&expected) {
        for (got, want) in got.iter().zip(want) {
            assert!((got - want).abs() < 2e-5, "head logit {got} vs upstream {want}");
        }
    }
}

#[test]
fn external_lexical_rows_match_the_native_embedding_table() {
    let (mut head, record, hidden, _, lexical) = reference();
    let want = head.forward(&hidden, &record).unwrap();
    let d = head.cfg.hidden;
    let mut means = Vec::new();
    for q in &record.questions {
        for (a, b) in &q.option_spans {
            let mut mean = vec![0.; d];
            for id in &record.ids[*a..*b] {
                for i in 0..d {
                    mean[i] += lexical[*id as usize * d + i];
                }
            }
            for value in &mut mean {
                *value /= (b - a) as f32;
            }
            means.extend(mean);
        }
    }
    let got = head.forward_with_lexical(&hidden, &record, &means).unwrap();
    for (got, want) in got.iter().flatten().zip(want.iter().flatten()) {
        assert!((got - want).abs() < 2e-6, "external {got}, native {want}");
    }
    assert!(head
        .forward_with_lexical(&hidden, &record, &means[..means.len() - 1])
        .unwrap_err()
        .contains("lexical means"));
}

#[test]
fn field_self_attention_makes_decisions_joint() {
    let (mut head, record, hidden, _, _) = reference();
    let joint = head.forward(&hidden, &record).unwrap();
    let alone = Record { ids: record.ids.clone(), questions: vec![record.questions[0].clone()] };
    let single = head.forward(&hidden, &alone).unwrap();
    assert!(joint[0].iter().zip(&single[0]).any(|(a, b)| (a - b).abs() > 1e-4));
}

#[test]
fn text_encoding_matches_upstream_sorting_spans_and_truncation() {
    let f = fixture();
    let tok = support::tokenizer(&[]);
    let request = f.get("encoding_request").unwrap();
    let record =
        encoding::encode(&tok, request.get("state").unwrap(), request.get("questions").unwrap(), 4096, usize::MAX)
            .unwrap();
    let expected = f.get("encoding").unwrap();
    assert_eq!(record.ids.iter().map(|id| *id as usize).collect::<Vec<_>>(), dimensions(expected.get("ids").unwrap()));
    for (q, want) in record.questions.iter().zip(expected.get("questions").unwrap().as_array().unwrap()) {
        assert_eq!(q.span, span(want.get("span").unwrap()));
        assert_eq!(
            q.option_spans,
            want.get("option_spans").unwrap().as_array().unwrap().iter().map(span).collect::<Vec<_>>()
        );
        assert_eq!(
            q.option_ids,
            want.get("option_ids")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect::<Vec<_>>()
        );
    }
    let truncated =
        encoding::encode(&tok, request.get("state").unwrap(), request.get("questions").unwrap(), 4096, 5).unwrap();
    assert_eq!(
        truncated.ids.iter().map(|id| *id as usize).collect::<Vec<_>>(),
        dimensions(f.get("truncated_ids").unwrap())
    );
    assert!(encoding::encode(&tok, request.get("state").unwrap(), request.get("questions").unwrap(), 30, usize::MAX)
        .unwrap_err()
        .contains("schema requires"));
}

#[test]
fn malformed_spans_and_question_contracts_fail_clearly() {
    let (mut head, mut record, hidden, _, _) = reference();
    record.questions[0].span = (4, 4);
    assert!(head.forward(&hidden, &record).unwrap_err().contains("invalid token span"));
    let tok = support::tokenizer(&[]);
    let q = Value::parse(r#"{"q":{"type":"choice","criteria":["a","b"]}}"#).unwrap();
    assert!(encoding::encode(&tok, &Value::Null, &q, 4096, 4096).unwrap_err().contains("nonempty object"));
}

#[test]
fn streaming_each_backbone_layer_matches_the_complete_trunk() {
    use kevala::kev::{KevEngine, KevModel};
    use kevala::pack::{self, Slice};
    let pack = support::kev_pack(true);
    let mut engine = KevEngine::load(AlignedBuf::from_slice(&pack)).unwrap();
    let ids = engine.tok.encode("The customer was charged twice and requests a refund.");
    let expected = engine.model.forward_hidden(&ids);
    let mut hidden = engine.model.embed(&ids);
    let header = pack::parse_header(&pack).unwrap();
    for i in 0..engine.model.cfg.layers {
        let prefix = format!("L.{i}.");
        let mut selected = header.clone();
        selected.tensors.retain(|t| t.name == "norm" || t.name.starts_with(&prefix));
        for tensor in &mut selected.tensors {
            if let Some(suffix) = tensor.name.strip_prefix(&prefix) {
                tensor.name = format!("L.0.{suffix}");
            }
        }
        let layout = pack::subset(&selected, &|_| Some(Slice::Whole), false, vec![]).unwrap();
        let (store, _) = load_store(AlignedBuf::from_slice(&layout.apply(&pack))).unwrap();
        let mut cfg = engine.model.cfg.clone();
        cfg.layers = 1;
        cfg.full = vec![cfg.full[i]];
        let mut layer = KevModel::new_trunk(cfg, Arc::new(store)).unwrap();
        layer.forward_rows(&mut hidden).unwrap();
    }
    engine.model.normalize_hidden(&mut hidden);
    assert_eq!(hidden, expected);
}
