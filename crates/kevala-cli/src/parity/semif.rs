use kevala::kev::KevEngine;

use super::{case_id, cases, check_ids, compare_probabilities, field, golden, max_dp};
use crate::{positional, read_pack};

pub fn run(args: &[String]) -> Result<(), String> {
    let mut engine = KevEngine::load(read_pack(positional(args, 0)?)?)?;
    if !engine.model.cfg.semif {
        return Err("parity-semif needs a SemIf pack".into());
    }
    let golden = golden(args)?;
    let limit = max_dp(args, "0.05")?;
    let cases = cases(&golden)?;
    let mut worst = 0.0f64;
    let mut agree = 0;
    for case in cases {
        let id = case_id(case);
        let (state, questions) = (field(case, "state")?, field(case, "questions")?);
        let (encoded, qs) = engine.prepare(state, questions)?;
        if qs.len() != 1 {
            return Err("SemIf references must have one question per case".into());
        }
        let ids: Vec<usize> = encoded.state.iter().chain(&encoded.branches[0].ids).map(|&id| id as usize).collect();
        check_ids(id, ids.into_iter(), case)?;
        let responses = engine.decide(&[(state.clone(), questions.clone())])?;
        let (ok, dp) = compare_probabilities(id, &responses[0], &qs[0].id, case)?;
        agree += usize::from(ok);
        worst = worst.max(dp);
        println!("{id}: exact tokens, argmax {ok}, max |dp| {dp:.6}");
    }
    println!("SemIf: {agree}/{} argmax, max |dp| {worst:.6}", cases.len());
    if agree != cases.len() || worst > limit {
        return Err(format!("SemIf parity failed (limit {limit})"));
    }
    Ok(())
}
