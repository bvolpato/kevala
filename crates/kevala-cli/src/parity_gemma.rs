use kevala::gemma4::Gemma4Engine;
use kevala::json::Value;
use std::time::Instant;

use super::{flag, positional, read, read_pack};

pub fn run(args: &[String]) -> Result<(), String> {
    let mut engine = Gemma4Engine::load(read_pack(positional(args, 0)?)?)?;
    let golden = Value::parse(&String::from_utf8(read(positional(args, 1)?)?).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let limit: f64 = flag(args, "--max-dp").unwrap_or("0.03").parse().map_err(|_| "bad --max-dp")?;
    if !limit.is_finite() || limit < 0.0 {
        return Err("--max-dp must be finite and nonnegative".into());
    }
    let cases = golden.get("cases").and_then(Value::as_array).ok_or("no reference cases")?;
    if cases.is_empty() {
        return Err("no reference cases".into());
    }
    let mut worst = 0.0f64;
    let mut agree = 0;
    let mut timings = Vec::new();
    for case in cases {
        let id = case.get("id").and_then(Value::as_str).unwrap_or("?");
        let prepared = engine.prepare(
            case.get("state").ok_or("case has no state")?,
            case.get("questions").ok_or("case has no questions")?,
        )?;
        if prepared.sequence_count() != 1 {
            return Err(format!("{id}: reference cases must have one question"));
        }
        let expected_ids = case
            .get("input_ids")
            .and_then(Value::as_array)
            .ok_or("no reference input_ids")?
            .iter()
            .map(|v| v.as_usize().ok_or("invalid reference token ID"))
            .collect::<Result<Vec<_>, _>>()?;
        let ids = prepared.sequence_ids(0);
        if !ids.iter().map(|&id| id as usize).eq(expected_ids.iter().copied()) {
            let first = ids.iter().zip(&expected_ids).position(|(&a, &b)| a as usize != b);
            return Err(format!("{id}: token IDs differ ({} vs {}, first {first:?})", ids.len(), expected_ids.len()));
        }
        let start = Instant::now();
        let row = engine.model.as_mut().ok_or("parity-gemma requires a complete pack")?.forward(ids)?;
        timings.push(start.elapsed().as_secs_f64() * 1000.0);
        let responses = engine.finish(&prepared, &row)?;
        let question = &prepared.questions[0].id;
        let actual = responses[0]
            .get("raw_probabilities")
            .and_then(|v| v.get(question))
            .and_then(Value::as_array)
            .ok_or("no result probabilities")?;
        let expected = case
            .get("probs")
            .and_then(Value::as_array)
            .and_then(|v| v.first())
            .and_then(Value::as_array)
            .ok_or("no reference probabilities")?;
        if actual.is_empty() || actual.len() != expected.len() {
            return Err(format!("{id}: probability count differs"));
        }
        let floats =
            |a: &[Value]| a.iter().map(|v| v.as_f64().ok_or("invalid probability")).collect::<Result<Vec<_>, _>>();
        let (actual, expected) = (floats(actual)?, floats(expected)?);
        if !actual.iter().chain(&expected).all(|x| x.is_finite() && (0.0..=1.0).contains(x)) {
            return Err(format!("{id}: nonfinite or out-of-range probabilities"));
        }
        let argmax = |a: &[f64]| a.iter().enumerate().fold(0, |best, (i, &p)| if p > a[best] { i } else { best });
        let ok = argmax(&actual) == argmax(&expected);
        agree += usize::from(ok);
        let difference = actual.iter().zip(expected).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        worst = worst.max(difference);
        println!("{id}: exact tokens, argmax {ok}, max |dp| {difference:.6}, {:.1} ms", timings.last().unwrap());
    }
    timings.sort_by(f64::total_cmp);
    println!(
        "Gemma4: {agree}/{} argmax, max |dp| {worst:.6}, median native forward {:.1} ms",
        cases.len(),
        timings[timings.len() / 2]
    );
    if agree != cases.len() || worst > limit {
        return Err(format!("Gemma4 parity failed (limit {limit})"));
    }
    Ok(())
}
