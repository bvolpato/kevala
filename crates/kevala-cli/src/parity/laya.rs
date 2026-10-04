use kevala::math::argmax;
use std::time::Instant;

use super::{array, case_id, cases, f64s, field, golden, max_abs_diff, median, usizes};
use crate::sharded::Sharded;
use crate::{flag, load, positional};

pub fn run(args: &[String]) -> Result<(), String> {
    let mut e = load(positional(args, 0)?)?;
    let golden = golden(args)?;
    let shards: usize = flag(args, "--shards").unwrap_or("1").parse().map_err(|_| "bad --shards")?;
    let sharded = if shards > 1 { Some(Sharded::new(&e, shards)?) } else { None };
    let (mut n, mut agree, mut worst_dp, mut worst_logit, mut kl_sum, mut ids_ok) = (0, 0, 0f64, 0f64, 0f64, 0);
    let mut ms = Vec::new();
    for case in cases(&golden)? {
        let id = case_id(case);
        let prepared = e.prepare(&[(field(case, "state")?, field(case, "questions")?)])?;
        let t = Instant::now();
        let outs = match &sharded {
            Some(s) => s.forward(&e, &prepared.batch),
            None => e.forward(&prepared.batch)?,
        };
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        let scored = e.score(&prepared, &outs);
        let expect = field(case, "expect")?;
        for (((_, q), s), seg) in prepared.items.iter().zip(&scored).zip(&prepared.batch.segs) {
            let ex = expect.get(&q.id).ok_or_else(|| format!("{id}: golden expects no question {:?}", q.id))?;
            let want_ids: Vec<u32> =
                usizes(array(ex, "input_ids")?, "input_ids")?.into_iter().map(|v| v as u32).collect();
            if want_ids == prepared.batch.ids[seg.start..seg.start + seg.len] {
                ids_ok += 1;
            }
            let logits = f64s(array(ex, "logits")?, "golden logits")?;
            let want = tempered_softmax(&logits, s.temperature as f64);
            let got: Vec<f64> = s.probs.iter().map(|&v| v as f64).collect();
            n += 1;
            if argmax(&want) == argmax(&got) {
                agree += 1;
            }
            let dp = max_abs_diff(&want, &got);
            let got_logits: Vec<f64> = s.logits.iter().map(|&v| v as f64).collect();
            let kl: f64 =
                want.iter().zip(&got).map(|(a, b)| if *a > 0.0 { a * (a / b.max(1e-12)).ln() } else { 0.0 }).sum();
            kl_sum += kl;
            worst_dp = worst_dp.max(dp);
            worst_logit = worst_logit.max(max_abs_diff(&logits, &got_logits));
            if dp > 0.02 {
                eprintln!("  {id}/{}: max |dp| {dp:.4}  want {want:.4?} got {got:.4?}", q.id);
            }
        }
    }
    println!(
        "{n} questions  token ids {ids_ok}/{n}  argmax {agree}/{n}  max|dp| {worst_dp:.4}  max|dlogit| {worst_logit:.4}  mean KL {:.2e}  median forward {:.1} ms",
        kl_sum / n as f64,
        median(&mut ms)
    );
    // int8 weights move probabilities by up to 0.024 on these fixtures (f32 packs stay under 1e-4)
    if ids_ok != n || agree != n || worst_dp > 0.03 {
        return Err("parity gate failed (needs exact ids, full argmax agreement, max |dp| <= 0.03)".into());
    }
    Ok(())
}

/// The reference probabilities: the fixture's logits at the engine's calibrated temperature.
fn tempered_softmax(logits: &[f64], temperature: f64) -> Vec<f64> {
    let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ez: Vec<f64> = logits.iter().map(|v| ((v - m) / temperature).exp()).collect();
    let sum: f64 = ez.iter().sum();
    ez.iter().map(|v| v / sum).collect()
}
