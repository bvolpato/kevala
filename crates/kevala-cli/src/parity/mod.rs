//! Parity commands: replay reference fixtures recorded from each family's upstream PyTorch code
//! through the native engine, and fail when token ids or probabilities drift.

pub mod gemma;
pub mod kev;
pub mod laya;
pub mod semif;

use kevala::json::Value;
use kevala::math::argmax;

use crate::{flag, positional, read};

/// The fixture named by the second positional argument, and proof that it has cases.
fn golden(args: &[String]) -> Result<Value, String> {
    let text = String::from_utf8(read(positional(args, 1)?)?).map_err(|_| "golden is not UTF-8")?;
    let golden = Value::parse(&text).map_err(|e| e.to_string())?;
    if cases(&golden)?.is_empty() {
        return Err("golden has no cases".into());
    }
    Ok(golden)
}

fn cases(golden: &Value) -> Result<&[Value], String> {
    Ok(golden.get("cases").and_then(Value::as_array).ok_or("golden has no cases")?)
}

fn field<'a>(case: &'a Value, key: &str) -> Result<&'a Value, String> {
    case.get(key).ok_or_else(|| format!("golden case has no {key}"))
}

fn array<'a>(case: &'a Value, key: &str) -> Result<&'a [Value], String> {
    field(case, key)?.as_array().ok_or_else(|| format!("golden {key} is not an array"))
}

fn usizes(values: &[Value], what: &str) -> Result<Vec<usize>, String> {
    values.iter().map(|v| v.as_usize().ok_or_else(|| format!("golden {what} holds a non-integer"))).collect()
}

fn f64s(values: &[Value], what: &str) -> Result<Vec<f64>, String> {
    values.iter().map(|v| v.as_f64().ok_or_else(|| format!("{what} holds a non-number"))).collect()
}

fn case_id(case: &Value) -> &str {
    case.get("id").and_then(Value::as_str).unwrap_or("?")
}

/// `--max-dp`, the largest probability difference a run may show.
fn max_dp(args: &[String], default: &str) -> Result<f64, String> {
    let limit: f64 = flag(args, "--max-dp").unwrap_or(default).parse().map_err(|_| "bad --max-dp")?;
    if !limit.is_finite() || limit < 0.0 {
        return Err("--max-dp must be finite and nonnegative".into());
    }
    Ok(limit)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn median(ms: &mut [f64]) -> f64 {
    ms.sort_by(f64::total_cmp);
    ms[ms.len() / 2]
}

/// Fails unless the engine tokenized a direct-options prompt exactly as the reference did.
fn check_ids(id: &str, got: impl ExactSizeIterator<Item = usize>, case: &Value) -> Result<(), String> {
    let want = usizes(array(case, "input_ids")?, "input_ids")?;
    let got: Vec<usize> = got.collect();
    if got != want {
        let first = got.iter().zip(&want).position(|(a, b)| a != b);
        return Err(format!("{id}: token IDs differ ({} vs {}, first {first:?})", got.len(), want.len()));
    }
    Ok(())
}

/// Compares a direct-options response with the case's single reference distribution. Returns
/// whether the argmax agrees and the largest probability difference.
fn compare_probabilities(id: &str, response: &Value, question: &str, case: &Value) -> Result<(bool, f64), String> {
    let got = response
        .get("raw_probabilities")
        .and_then(|p| p.get(question))
        .and_then(Value::as_array)
        .ok_or("response has no raw probabilities")?;
    let want = array(case, "probs")?.first().and_then(Value::as_array).ok_or("golden case has no probabilities")?;
    if got.is_empty() || got.len() != want.len() {
        return Err(format!("{id}: probability count differs"));
    }
    let (got, want) = (f64s(got, "response")?, f64s(want, "golden probs")?);
    if !got.iter().chain(&want).all(|p| p.is_finite() && (0.0..=1.0).contains(p)) {
        return Err(format!("{id}: nonfinite or out-of-range probabilities"));
    }
    Ok((argmax(&got) == argmax(&want), max_abs_diff(&got, &want)))
}
