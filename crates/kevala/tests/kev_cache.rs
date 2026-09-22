//! Kev's cross-request state cache must be exact: a repeated state and a state that extends a
//! cached one give the same probabilities as a cold engine. Needs a Kev pack
//! (`KEVALA_KEV_PACK`, default tmp/kev-0.8b-q8.kevala); skipped without one.

use kevala::json::Value;
use kevala::kev::KevEngine;
use kevala::model::AlignedBuf;

fn pack() -> Option<Vec<u8>> {
    let p = std::env::var("KEVALA_KEV_PACK")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../tmp/kev-0.8b-q8.kevala").into());
    match std::fs::read(&p) {
        Ok(b) => Some(b),
        Err(_) => {
            eprintln!("skipping: no Kev pack at {p}");
            None
        }
    }
}

fn probs(e: &mut KevEngine, state: &str) -> Vec<f64> {
    let qs = Value::parse(
        r#"{"a": {"type": "noul", "instructions": "Does the customer want money back?"},
            "b": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "payments", "shipping": "deliveries", "other": null}}}"#,
    )
    .unwrap();
    let r = e.decide(&[(Value::Str(state.into()), qs)]).unwrap();
    let raw = r[0].get("raw_probabilities").unwrap();
    ["a", "b"]
        .iter()
        .flat_map(|q| raw.get(q).unwrap().as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
        .collect()
}

#[test]
fn cached_and_extended_states_match_a_cold_run() {
    let Some(bytes) = pack() else { return };
    let base = "Order #4411 arrived two weeks late and the box was crushed on one side. I was also charged twice for it on my credit card, and the second charge is still pending.";
    let longer =
        format!("{base} I called support on Monday and nobody answered, so please refund the duplicate charge today.");

    let mut cold = KevEngine::load(AlignedBuf::from_slice(&bytes)).unwrap();
    cold.model.cache_states = 0;
    let want_base = probs(&mut cold, base);
    let want_longer = probs(&mut cold, &longer);

    let mut warm = KevEngine::load(AlignedBuf::from_slice(&bytes)).unwrap();
    let first = probs(&mut warm, base);
    let again = probs(&mut warm, base); // exact hit
    let extended = probs(&mut warm, &longer); // extends the cached state
    let s = warm.model.stats;
    assert_eq!((s.misses, s.hits, s.extensions), (1, 1, 1), "{s:?}");

    let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5);
    assert!(close(&first, &want_base), "{first:?} vs {want_base:?}");
    assert!(close(&again, &want_base), "{again:?} vs {want_base:?}");
    assert!(close(&extended, &want_longer), "{extended:?} vs {want_longer:?}");
}
