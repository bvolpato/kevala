use kevala::gemma4::Gemma4Engine;
use std::time::Instant;

use super::{case_id, cases, check_ids, compare_probabilities, field, golden, max_dp, median};
use crate::{positional, read_pack};

pub fn run(args: &[String]) -> Result<(), String> {
    let mut engine = Gemma4Engine::load(read_pack(positional(args, 0)?)?)?;
    let golden = golden(args)?;
    let limit = max_dp(args, "0.03")?;
    let cases = cases(&golden)?;
    let mut worst = 0.0f64;
    let mut agree = 0;
    let mut timings = Vec::new();
    for case in cases {
        let id = case_id(case);
        let prepared = engine.prepare(field(case, "state")?, field(case, "questions")?)?;
        if prepared.sequence_count() != 1 {
            return Err(format!("{id}: reference cases must have one question"));
        }
        let ids = prepared.sequence_ids(0);
        check_ids(id, ids.iter().map(|&id| id as usize), case)?;
        let start = Instant::now();
        let row = engine.model.as_mut().ok_or("parity-gemma requires a complete pack")?.forward(ids)?;
        timings.push(start.elapsed().as_secs_f64() * 1000.0);
        let responses = engine.finish(&prepared, &row)?;
        let (ok, difference) = compare_probabilities(id, &responses[0], &prepared.questions[0].id, case)?;
        agree += usize::from(ok);
        worst = worst.max(difference);
        println!("{id}: exact tokens, argmax {ok}, max |dp| {difference:.6}, {:.1} ms", timings.last().unwrap());
    }
    println!(
        "Gemma4: {agree}/{} argmax, max |dp| {worst:.6}, median native forward {:.1} ms",
        cases.len(),
        median(&mut timings)
    );
    if agree != cases.len() || worst > limit {
        return Err(format!("Gemma4 parity failed (limit {limit})"));
    }
    Ok(())
}
