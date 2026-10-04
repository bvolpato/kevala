//! Laya's engine on a synthetic pack: the invariants that hold for any weights.

mod support;

use kevala::content::Request;
use kevala::json::Value;
use kevala::laya::model::{build_shard, Scratch, ShardPlan, Trunk};
use kevala::laya::Engine;
use kevala::pack::{self, coord_layout};
use kevala::runtime::{self, Model};
use kevala::store::AlignedBuf;
use std::sync::Arc;

const QUESTIONS: &str = r#"{
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "payments", "tech": "bugs", "other": null}},
    "churn": {"type": "noul", "instructions": "Will they leave?"},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]}
}"#;

fn request(json: &str) -> Request {
    Request::parse(&Value::parse(json).unwrap()).unwrap()
}

fn billing_request() -> Request {
    request(&format!(r#"{{"state": "We were billed twice. Refund it or we cancel.", "questions": {QUESTIONS}}}"#))
}

fn input_tokens(response: &Value) -> usize {
    response.get("usage").and_then(|u| u.get("input_tokens")).and_then(Value::as_usize).unwrap()
}

#[test]
fn tensor_parallel_shards_sum_to_the_whole_trunk() {
    let bytes = support::laya_pack();
    let mut whole = Engine::load(AlignedBuf::from_slice(&bytes)).unwrap();
    let (prepared, _) = whole.prepare_requests(&[billing_request()]).unwrap();
    let batch = &prepared.batch;
    let want = whole.forward(batch).unwrap();

    let shards: Vec<Trunk> = (0..2)
        .map(|index| {
            let store = build_shard(&whole.cfg, &bytes, ShardPlan { index, count: 2 }).unwrap();
            Trunk::new(whole.cfg.clone(), Arc::new(store), index == 0).unwrap()
        })
        .collect();
    let mut x = whole.coord.embed(batch);
    let mut partial = vec![0.0; x.len()];
    let mut scratch = Scratch::default();
    for step in 0..whole.cfg.steps() {
        if step == 2 * whole.cfg.layers {
            whole.coord.bridge(&mut x, batch);
        }
        let mut sum = vec![0.0; x.len()];
        for shard in &shards {
            shard.step(step, &x, batch, &mut partial, &mut scratch);
            sum.iter_mut().zip(&partial).for_each(|(a, b)| *a += b);
        }
        x.iter_mut().zip(&sum).for_each(|(a, b)| *a += b);
    }
    let got = whole.coord.score(&x, batch);

    assert_eq!(got.len(), 3);
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(g.logits.len(), w.logits.len());
        // The split changes only the order of f32 additions.
        for (a, b) in g.logits.iter().zip(&w.logits).chain(g.act_logits.iter().zip(&w.act_logits)) {
            assert!(a.is_finite() && (a - b).abs() < 1e-4, "sharded {a} differs from whole {b}");
        }
    }
    // Biases of row-parallel projections count once: a second primary shard would add them twice.
    assert!(
        build_shard(&whole.cfg, &bytes, ShardPlan { index: 0, count: 3 }).is_err(),
        "two heads cannot make three shards"
    );
}

#[test]
fn a_coordinator_sub_pack_refuses_to_decide_instead_of_panicking() {
    let bytes = support::laya_pack();
    let header = pack::parse_header(&bytes).unwrap();
    let coordinator = coord_layout(&header).unwrap().apply(&bytes);
    assert!(coordinator.len() < bytes.len() / 2, "the coordinator keeps no transformer layers");

    let mut model = runtime::load(AlignedBuf::from_slice(&coordinator)).unwrap();
    assert_eq!(model.arch(), "laya");
    let error = model.decide(&[billing_request()]).unwrap_err();
    assert!(error.contains("coordinator-only"), "{error}");
}

#[test]
fn text_parts_reach_the_model_through_every_preparation_path() {
    let mut engine = Engine::load(AlignedBuf::from_slice(&support::laya_pack())).unwrap();
    let plain = billing_request();
    let with_part = request(&format!(
        r#"{{"state": "We were billed twice. Refund it or we cancel.",
             "parts": [{{"type": "text", "text": "Attached: invoice 4411."}}], "questions": {QUESTIONS}}}"#
    ));
    let merged = request(&format!(
        r#"{{"state": "We were billed twice. Refund it or we cancel.\n\nAttached: invoice 4411.", "questions": {QUESTIONS}}}"#
    ));

    // The external-trunk path (GPU, shards) prepares the same tokens as `decide` scores.
    let tokens = |engine: &Engine, r: &Request| engine.prepare_requests(std::slice::from_ref(r)).unwrap().0.batch.ids;
    assert!(tokens(&engine, &with_part).len() > tokens(&engine, &plain).len());
    assert_eq!(tokens(&engine, &with_part), tokens(&engine, &merged));

    let responses = Model::decide(&mut engine, &[plain, with_part.clone(), merged]).unwrap();
    assert!(input_tokens(&responses[1]) > input_tokens(&responses[0]));
    assert_eq!(responses[1].to_json(), responses[2].to_json());

    let image = request(&format!(
        r#"{{"state": "x", "parts": [{{"type": "image", "url": "a.png"}}], "questions": {QUESTIONS}}}"#
    ));
    let error = engine.prepare_requests(&[image]).err().unwrap();
    assert!(error.contains("part 0 is image"), "{error}");
}

#[test]
fn a_batch_answers_each_request_as_if_it_were_alone() {
    let mut engine = Engine::load(AlignedBuf::from_slice(&support::laya_pack())).unwrap();
    let other = request(&format!(
        r#"{{"state": {{"subject": "Thanks", "body": "Great service."}}, "questions": {QUESTIONS}}}"#
    ));
    let together = Model::decide(&mut engine, &[billing_request(), other.clone()]).unwrap();
    let alone = [
        Model::decide(&mut engine, &[billing_request()]).unwrap().remove(0),
        Model::decide(&mut engine, &[other]).unwrap().remove(0),
    ];
    // Segments attend only to themselves, so packing two requests must not change either answer.
    assert_eq!(together[0].to_json(), alone[0].to_json());
    assert_eq!(together[1].to_json(), alone[1].to_json());

    let answers = together[0].get("answers").unwrap();
    let team = answers.get("team").unwrap();
    let probabilities: Vec<f64> =
        team.get("probabilities").unwrap().as_object().unwrap().iter().map(|(_, p)| p.as_f64().unwrap()).collect();
    assert_eq!(probabilities.len(), 3);
    assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-3, "{probabilities:?}");
    assert!(["billing", "tech", "other"].contains(&team.get("choice").unwrap().as_str().unwrap()));
    let urgency = answers.get("urgency").unwrap().get("score").unwrap().as_f64().unwrap();
    assert!((0.0..=2.0).contains(&urgency), "{urgency}");
}
