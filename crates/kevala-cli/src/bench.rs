//! Timing commands: whole requests through a pack, and the matmul kernel on synthetic weights.

use kevala::json::Value;
use kevala::laya::Engine;
use std::time::Instant;

use crate::sharded::Sharded;
use crate::{flag, positional, read_pack};

pub fn run(args: &[String]) -> Result<(), String> {
    let path = positional(args, 0)?;
    let tokens: usize = flag(args, "--tokens").unwrap_or("64").parse().map_err(|_| "bad --tokens")?;
    let nq: usize = flag(args, "--questions").unwrap_or("1").parse().map_err(|_| "bad --questions")?;
    let runs: usize = flag(args, "--runs").unwrap_or("5").parse().map_err(|_| "bad --runs")?;
    let shards: usize = flag(args, "--shards").unwrap_or("1").parse().map_err(|_| "bad --shards")?;
    if runs == 0 {
        return Err("--runs must be positive".into());
    }
    let mut m = kevala::runtime::load(read_pack(path)?)?;
    if let Some(k) = m.as_any().downcast_mut::<kevala::kev::KevEngine>() {
        // every run repeats the same state: without --warm, time the full pass, not a cache hit
        if !args.iter().any(|a| a == "--warm") {
            k.model.cache_states = 0;
        }
    }
    let words = "the quick brown fox jumps over the lazy dog while the customer waits for a refund ";
    let mut state = String::new();
    while m.tokenizer().encode(&state).len() + 24 < tokens {
        state.push_str(words);
    }
    let mut q = Vec::new();
    for i in 0..nq {
        q.push((
            format!("q{i}"),
            Value::parse(r#"{"type":"noul","instructions":"Is the customer asking for money back?"}"#).unwrap(),
        ));
    }
    let req = kevala::content::Request { state: Value::Str(state), parts: Vec::new(), questions: Value::Object(q) };
    let mut ms = Vec::new();
    let mut used = 0;
    if shards > 1 {
        // tensor-parallel shards in threads (Laya)
        let e: &mut Engine = m.as_any().downcast_mut().ok_or("--shards needs a laya pack")?;
        let sharded = Sharded::new(e, shards)?;
        let (s, q) = (req.state.clone(), req.questions.clone());
        let prepared = e.prepare(&[(&s, &q)])?;
        used = prepared.batch.tokens();
        for _ in 0..runs + 1 {
            let t = Instant::now();
            let _ = sharded.forward(e, &prepared.batch);
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
    } else {
        for _ in 0..runs + 1 {
            let t = Instant::now();
            let r = m.decide(std::slice::from_ref(&req))?;
            ms.push(t.elapsed().as_secs_f64() * 1e3);
            used = r[0].get("usage").and_then(|u| u.get("input_tokens")).and_then(Value::as_usize).unwrap_or(0);
        }
    }
    // the first run pays for cold caches
    ms.remove(0);
    ms.sort_by(f64::total_cmp);
    println!(
        "{} {used} tokens, {nq} question(s), {shards} thread(s): p50 {:.1} ms  min {:.1} ms",
        m.arch(),
        ms[ms.len() / 2],
        ms[0]
    );
    Ok(())
}

/// Matmul kernel throughput on synthetic weights (development aid).
pub fn kernels() -> Result<(), String> {
    use kevala::kernels::{linear, Mat};
    for (t, n, k) in [(64usize, 3072usize, 1024usize), (64, 1024, 2624), (256, 3072, 1024)] {
        let x: Vec<f32> = (0..t * k).map(|i| (i % 97) as f32 * 0.01 - 0.5).collect();
        let q: Vec<i8> = (0..n * k).map(|i| ((i * 31 % 251) as i32 - 125) as i8).collect();
        let s: Vec<f32> = vec![0.01; n * k / 32];
        let wf: Vec<f32> = q.iter().map(|&v| v as f32 * 0.01).collect();
        let mut out = vec![0.0; t * n];
        let mut panel = Vec::new();
        for (name, m) in [("q8 ", Mat::Q8 { n, k, block: 32, q: &q, scales: &s }), ("f32", Mat::F32 { n, k, w: &wf })] {
            linear(&x, t, m, None, &mut out, &mut panel);
            let reps = 5;
            let t0 = Instant::now();
            for _ in 0..reps {
                linear(&x, t, m, None, &mut out, &mut panel);
            }
            let secs = t0.elapsed().as_secs_f64() / reps as f64;
            println!(
                "{name} T={t} N={n} K={k}: {:.2} ms  {:.1} GFLOP/s",
                secs * 1e3,
                2.0 * (t * n * k) as f64 / secs / 1e9
            );
        }
    }
    Ok(())
}
