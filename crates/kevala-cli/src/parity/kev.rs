use kevala::kev::{Branch, Encoded, KevEngine};
use kevala::math::argmax;
use std::time::Instant;

use super::{array, case_id, cases, f64s, field, golden, max_abs_diff, median, usizes};
use crate::{positional, read_pack};

/// Runs the reference token ids through the model, so a tokenizer difference shows up as its own
/// count instead of as a probability difference.
pub fn run(args: &[String]) -> Result<(), String> {
    let t = Instant::now();
    let mut e = KevEngine::load(read_pack(positional(args, 0)?)?)?;
    eprintln!("loaded in {:.2}s", t.elapsed().as_secs_f64());
    let golden = golden(args)?;
    let cases = cases(&golden)?;
    let (mut n, mut agree, mut worst, mut ids_ok) = (0, 0, 0f64, 0);
    let mut ms = Vec::new();
    for case in cases {
        let id = case_id(case);
        let ids: Vec<u32> = usizes(array(case, "ids")?, "ids")?.into_iter().map(|v| v as u32).collect();
        let seg = usizes(array(case, "seg")?, "seg")?;
        let decide = usizes(array(case, "decide_idx")?, "decide_idx")?;
        let opts: Vec<Vec<usize>> = array(case, "opt_idx")?
            .iter()
            .map(|o| usizes(o.as_array().ok_or("golden opt_idx is not a list of lists")?, "opt_idx"))
            .collect::<Result<_, _>>()?;
        if opts.len() != decide.len() {
            return Err(format!("{id}: golden has {} option lists for {} questions", opts.len(), decide.len()));
        }
        let ls = seg.iter().filter(|&&s| s == 0).count();
        let mut branches = Vec::new();
        let mut start = ls;
        for (d, opts) in decide.iter().zip(&opts) {
            let end = d + 1;
            if *d < start || end > ids.len() || opts.iter().any(|o| *o < start || *o >= end) {
                return Err(format!("{id}: golden question rows are out of order"));
            }
            branches.push(Branch {
                ids: ids[start..end].to_vec(),
                decide: d - start,
                opts: opts.iter().map(|o| o - start).collect(),
                labels: 0,
            });
            start = end;
        }
        let golden_enc = Encoded { state: ids[..ls].to_vec(), branches };
        // tokenizer + template parity, when the tokenizer can encode this model's text
        if let Ok((mine, _)) = e.prepare(field(case, "state")?, field(case, "questions")?) {
            let flat: Vec<u32> =
                mine.state.iter().copied().chain(mine.branches.iter().flat_map(|b| b.ids.iter().copied())).collect();
            if flat == ids {
                ids_ok += 1;
            } else {
                eprintln!("  {id}: token ids differ ({} vs {})", flat.len(), ids.len());
            }
        }
        let t = Instant::now();
        let logits = e.model.forward(std::slice::from_ref(&golden_enc));
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        for (q, (got, want)) in logits[0].iter().zip(array(case, "probs")?).enumerate() {
            let p: Vec<f64> = softmax_f64(got);
            let w = f64s(want.as_array().ok_or("golden probs is not a list of lists")?, "golden probs")?;
            n += 1;
            if argmax(&p) == argmax(&w) {
                agree += 1;
            }
            worst = worst.max(max_abs_diff(&p, &w));
            eprintln!("  {id}/{q}: want {w:.3?} got {p:.3?}");
        }
    }
    let cases = cases.len();
    println!("{n} questions in {cases} cases  token ids {ids_ok}/{cases}  argmax {agree}/{n}  max|dp| {worst:.4}  median forward {:.1} ms", median(&mut ms));
    Ok(())
}

/// Softmax of f32 logits accumulated in f64, as the reference script prints its probabilities.
fn softmax_f64(logits: &[f32]) -> Vec<f64> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let ez: Vec<f64> = logits.iter().map(|v| ((v - m) as f64).exp()).collect();
    let s: f64 = ez.iter().sum();
    ez.iter().map(|v| v / s).collect()
}
