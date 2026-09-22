//! `kevala` command line: convert checkpoints to `.kevala` packs, answer questions, check parity
//! against the PyTorch reference, and benchmark.

use kevala::engine::Engine;
use kevala::json::Value;
use kevala::model::{build_shard, AlignedBuf, Scratch, ShardPlan, Trunk};
use std::time::Instant;

const USAGE: &str = "kevala: System 1 decision models (Laya, Kev) without Python

usage:
  kevala convert <checkpoint-dir> -o <out.kevala> [--block 32] [--keep-f32 head.,emb] [--revision <sha>]
  kevala decide <pack.kevala> --state <json|text> --questions <json>
  kevala convert-kev --base <qwen-dir> --kev <kev-dir> -o <out.kevala> [--block 32]
  kevala parity <pack.kevala> <golden.json> [--shards N]
  kevala parity-kev <pack.kevala> <golden-kev.json>
  kevala bench <pack.kevala> [--tokens 64] [--questions 1] [--runs 5] [--shards N] [--warm]
  kevala inspect <pack.kevala>
";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("convert") => convert(&args[1..]),
        Some("decide") => decide(&args[1..]),
        Some("parity") => parity(&args[1..]),
        Some("convert-kev") => convert_kev(&args[1..]),
        Some("parity-kev") => parity_kev(&args[1..]),
        Some("bench") => bench(&args[1..]),
        Some("inspect") => inspect(&args[1..]),
        Some("kbench") => kbench(),
        _ => {
            eprint!("{USAGE}");
            std::process::exit(2);
        }
    };
    if let Err(e) = r {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn positional(args: &[String], n: usize) -> Result<&str, String> {
    let mut seen = 0;
    let mut i = 0;
    while i < args.len() {
        if args[i].starts_with("--") || args[i] == "-o" {
            i += 2;
            continue;
        }
        if seen == n {
            return Ok(&args[i]);
        }
        seen += 1;
        i += 1;
    }
    Err(format!("missing argument\n{USAGE}"))
}

fn read(path: &str) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{path}: {e}"))
}

fn load(path: &str) -> Result<Engine, String> {
    let t = Instant::now();
    let bytes = read(path)?;
    let e = Engine::load(AlignedBuf::from_slice(&bytes))?;
    eprintln!("loaded {path} ({:.0} MB) in {:.2}s", bytes.len() as f64 / 1e6, t.elapsed().as_secs_f64());
    Ok(e)
}

fn convert(args: &[String]) -> Result<(), String> {
    let dir = positional(args, 0)?;
    let out = flag(args, "-o").ok_or("convert needs -o <out.kevala>")?;
    let block: usize = flag(args, "--block").unwrap_or("32").parse().map_err(|_| "bad --block")?;
    let revision = flag(args, "--revision").unwrap_or("1c5edc17a7acd8701df6fc341c0d179f1c62c982");
    let t = Instant::now();
    let st = read(&format!("{dir}/model.safetensors"))?;
    let text = |p: &str| String::from_utf8(read(&format!("{dir}/{p}"))?).map_err(|_| format!("{p} is not UTF-8"));
    let (enc, agent, tok) =
        (text("encoder/config.json")?, text("rl_agent_config.json")?, text("tokenizer/tokenizer.json")?);
    let model = Value::parse(&format!(
        r#"{{"name":"laya","source":"https://huggingface.co/convaiinnovations/laya","revision":"{revision}",
        "author":"Nandakishor M, Convai Innovations","license":"apache-2.0",
        "converter":"kevala {}","quantization":"int8 symmetric absmax, one f32 scale per {block} weights; norms, biases, type embedding, scorer and act head in f32"}}"#,
        env!("CARGO_PKG_VERSION")
    ))
    .map_err(|e| e.to_string())?;
    let pack = kevala::convert::convert(
        &kevala::convert::Checkpoint {
            safetensors: &st,
            encoder_config: &enc,
            agent_config: &agent,
            tokenizer_json: &tok,
        },
        &kevala::convert::Options {
            block,
            model,
            keep_f32: flag(args, "--keep-f32").map(|s| s.split(',').map(String::from).collect()).unwrap_or_default(),
        },
    )?;
    std::fs::write(out, &pack).map_err(|e| format!("{out}: {e}"))?;
    eprintln!("wrote {out}: {:.1} MB in {:.1}s", pack.len() as f64 / 1e6, t.elapsed().as_secs_f64());
    Ok(())
}

fn state_arg(s: &str) -> Value {
    // JSON when it parses as an object or array, otherwise the literal text
    match Value::parse(s) {
        Ok(v @ (Value::Object(_) | Value::Array(_))) => v,
        _ => Value::Str(s.to_string()),
    }
}

fn decide(args: &[String]) -> Result<(), String> {
    let path = positional(args, 0)?;
    let t = Instant::now();
    let bytes = read(path)?;
    let mut m = kevala::runtime::load(AlignedBuf::from_slice(&bytes))?;
    eprintln!("loaded {path} ({}, {:.0} MB) in {:.2}s", m.arch(), bytes.len() as f64 / 1e6, t.elapsed().as_secs_f64());
    let state = state_arg(flag(args, "--state").ok_or("decide needs --state")?);
    let qs = Value::parse(flag(args, "--questions").ok_or("decide needs --questions")?).map_err(|e| e.to_string())?;
    let t = Instant::now();
    let r = m.decide(&[kevala::content::Request { state, parts: Vec::new(), questions: qs }])?;
    eprintln!("decided in {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    println!("{}", r[0].to_json());
    Ok(())
}

fn inspect(args: &[String]) -> Result<(), String> {
    let bytes = read(positional(args, 0)?)?;
    let h = kevala::pack::parse_header(&bytes)?;
    println!("model  {}", h.json.get("model").map(Value::to_json).unwrap_or_default());
    println!("config {}", h.config().to_json());
    let (mut q8, mut f32s) = (0usize, 0usize);
    for t in &h.tensors {
        match t.dtype {
            kevala::pack::DType::Q8 => q8 += t.numel(),
            kevala::pack::DType::F32 => f32s += t.numel(),
        }
    }
    println!(
        "tensors {}  q8 params {:.1}M  f32 params {:.1}M  file {:.1} MB",
        h.tensors.len(),
        q8 as f64 / 1e6,
        f32s as f64 / 1e6,
        bytes.len() as f64 / 1e6
    );
    Ok(())
}

/// Shards the trunk `n` ways inside this process, one thread per shard and step.
struct Sharded {
    shards: Vec<Trunk>,
}

impl Sharded {
    fn new(e: &Engine, n: usize) -> Result<Sharded, String> {
        let full = e.coord.store().bytes();
        let mut shards = Vec::new();
        for i in 0..n {
            let s = build_shard(&e.cfg, full, ShardPlan { index: i, count: n })?;
            shards.push(Trunk::new(e.cfg.clone(), std::sync::Arc::new(s), i == 0)?);
        }
        Ok(Sharded { shards })
    }

    fn forward(&self, e: &Engine, batch: &kevala::model::Batch) -> Vec<kevala::model::SegOut> {
        let cfg = &e.cfg;
        let n = batch.tokens() * cfg.hidden;
        let mut x = e.coord.embed(batch);
        let mut scratch: Vec<Scratch> = self.shards.iter().map(|_| Scratch::default()).collect();
        let mut outs: Vec<Vec<f32>> = self.shards.iter().map(|_| vec![0.0; n]).collect();
        for s in 0..cfg.steps() {
            if s == 2 * cfg.layers {
                e.coord.bridge(&mut x, batch);
            }
            std::thread::scope(|sc| {
                for ((sh, scr), out) in self.shards.iter().zip(scratch.iter_mut()).zip(outs.iter_mut()) {
                    let x = &x;
                    sc.spawn(move || sh.step(s, x, batch, out, scr));
                }
            });
            for out in &outs {
                for (a, b) in x.iter_mut().zip(out) {
                    *a += b;
                }
            }
        }
        e.coord.score(&x, batch)
    }
}

fn parity(args: &[String]) -> Result<(), String> {
    let mut e = load(positional(args, 0)?)?;
    let golden = Value::parse(&String::from_utf8(read(positional(args, 1)?)?).map_err(|_| "golden is not UTF-8")?)
        .map_err(|e| e.to_string())?;
    let shards: usize = flag(args, "--shards").unwrap_or("1").parse().map_err(|_| "bad --shards")?;
    let sharded = if shards > 1 { Some(Sharded::new(&e, shards)?) } else { None };
    let (mut n, mut agree, mut worst_dp, mut worst_logit, mut kl_sum, mut ids_ok) = (0, 0, 0f64, 0f64, 0f64, 0);
    let mut ms = Vec::new();
    for case in golden.get("cases").and_then(Value::as_array).ok_or("golden has no cases")? {
        let id = case.get("id").and_then(Value::as_str).unwrap_or("?");
        let (state, qs) = (case.get("state").unwrap(), case.get("questions").unwrap());
        let prepared = e.prepare(&[(state, qs)])?;
        let t = Instant::now();
        let outs = match &sharded {
            Some(s) => s.forward(&e, &prepared.batch),
            None => e.forward(&prepared.batch),
        };
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        let scored = e.score(&prepared, &outs);
        let expect = case.get("expect").unwrap();
        for (((_, q), s), seg) in prepared.items.iter().zip(&scored).zip(&prepared.batch.segs) {
            let ex = expect.get(&q.id).unwrap();
            let want_ids: Vec<u32> = ex
                .get("input_ids")
                .and_then(Value::as_array)
                .unwrap()
                .iter()
                .map(|v| v.as_usize().unwrap() as u32)
                .collect();
            if want_ids == prepared.batch.ids[seg.start..seg.start + seg.len] {
                ids_ok += 1;
            }
            let logits: Vec<f64> =
                ex.get("logits").and_then(Value::as_array).unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let t = s.temperature as f64;
            let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let ez: Vec<f64> = logits.iter().map(|v| ((v - m) / t).exp()).collect();
            let sum: f64 = ez.iter().sum();
            let want: Vec<f64> = ez.iter().map(|v| v / sum).collect();
            let got: Vec<f64> = s.probs.iter().map(|&v| v as f64).collect();
            let am = |v: &[f64]| v.iter().enumerate().fold(0, |b, (i, &x)| if x > v[b] { i } else { b });
            n += 1;
            if am(&want) == am(&got) {
                agree += 1;
            }
            let dp = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            let dl = logits.iter().zip(&s.logits).map(|(a, &b)| (a - b as f64).abs()).fold(0.0, f64::max);
            let kl: f64 =
                want.iter().zip(&got).map(|(a, b)| if *a > 0.0 { a * (a / b.max(1e-12)).ln() } else { 0.0 }).sum();
            kl_sum += kl;
            worst_dp = worst_dp.max(dp);
            worst_logit = worst_logit.max(dl);
            if dp > 0.02 {
                eprintln!("  {id}/{}: max |dp| {dp:.4}  want {want:.4?} got {got:.4?}", q.id);
            }
        }
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "{n} questions  token ids {ids_ok}/{n}  argmax {agree}/{n}  max|dp| {worst_dp:.4}  max|dlogit| {worst_logit:.4}  mean KL {:.2e}  median forward {:.1} ms",
        kl_sum / n as f64,
        ms[ms.len() / 2]
    );
    // int8 weights move probabilities by up to 0.024 on these fixtures (f32 packs stay under 1e-4)
    if ids_ok != n || agree != n || worst_dp > 0.03 {
        return Err("parity gate failed (needs exact ids, full argmax agreement, max |dp| <= 0.03)".into());
    }
    Ok(())
}

fn bench(args: &[String]) -> Result<(), String> {
    let path = positional(args, 0)?;
    let tokens: usize = flag(args, "--tokens").unwrap_or("64").parse().map_err(|_| "bad --tokens")?;
    let nq: usize = flag(args, "--questions").unwrap_or("1").parse().map_err(|_| "bad --questions")?;
    let runs: usize = flag(args, "--runs").unwrap_or("5").parse().map_err(|_| "bad --runs")?;
    let shards: usize = flag(args, "--shards").unwrap_or("1").parse().map_err(|_| "bad --shards")?;
    let bytes = read(path)?;
    let mut m = kevala::runtime::load(AlignedBuf::from_slice(&bytes))?;
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
    ms.remove(0);
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "{} {used} tokens, {nq} question(s), {shards} thread(s): p50 {:.1} ms  min {:.1} ms",
        m.arch(),
        ms[ms.len() / 2],
        ms[0]
    );
    Ok(())
}

fn convert_kev(args: &[String]) -> Result<(), String> {
    let base = flag(args, "--base").ok_or("convert-kev needs --base <dir>")?;
    let kev = flag(args, "--kev").ok_or("convert-kev needs --kev <dir>")?;
    let out = flag(args, "-o").ok_or("convert-kev needs -o <out.kevala>")?;
    let block: usize = flag(args, "--block").unwrap_or("32").parse().map_err(|_| "bad --block")?;
    let t = Instant::now();
    let text = |p: String| String::from_utf8(read(&p)?).map_err(|_| format!("{p} is not UTF-8"));
    let st = std::fs::read_dir(base)
        .map_err(|e| format!("{base}: {e}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .ok_or("no .safetensors in --base")?;
    let weights = read(st.to_str().unwrap())?;
    let provenance = |k: &str| flag(args, k).unwrap_or("unknown").to_string();
    let model = Value::Object(vec![
        ("name".into(), Value::Str("kev-0.8b".into())),
        ("source".into(), Value::Str("https://huggingface.co/jaredpalmer/kev-0.8b".into())),
        ("revision".into(), Value::Str(provenance("--kev-revision"))),
        ("base".into(), Value::Str("https://huggingface.co/Qwen/Qwen3.5-0.8B-Base".into())),
        ("base_revision".into(), Value::Str(provenance("--base-revision"))),
        ("author".into(), Value::Str("Jared Palmer (Kev); Qwen team (base)".into())),
        ("license".into(), Value::Str("apache-2.0".into())),
        ("converter".into(), Value::Str(format!("kevala {}", env!("CARGO_PKG_VERSION")))),
        ("quantization".into(), Value::Str(format!("LoRA merged in f32, then int8 symmetric absmax, one f32 scale per {block} weights; norms, gates, conv, pointer head in f32"))),
    ]);
    let pack = kevala::convert_kev::convert(
        &kevala::convert_kev::KevCheckpoint {
            base: &weights,
            base_config: &text(format!("{base}/config.json"))?,
            // Kev ships the tokenizer AutoTokenizer really builds for the base; the base repo's own
            // tokenizer.json has a different pre-tokenizer regex and fewer added tokens
            base_tokenizer: &text(format!("{kev}/tokenizer.json"))
                .or_else(|_| text(format!("{base}/tokenizer.json")))?,
            adapter: &read(&format!("{kev}/adapter_model.safetensors"))?,
            adapter_config: &text(format!("{kev}/adapter_config.json"))?,
            head: &read(&format!("{kev}/head.pt"))?,
        },
        block,
        model,
    )?;
    std::fs::write(out, &pack).map_err(|e| format!("{out}: {e}"))?;
    eprintln!("wrote {out}: {:.1} MB in {:.1}s", pack.len() as f64 / 1e6, t.elapsed().as_secs_f64());
    Ok(())
}

fn parity_kev(args: &[String]) -> Result<(), String> {
    use kevala::kev::{Branch, Encoded, KevEngine};
    let t = Instant::now();
    let bytes = read(positional(args, 0)?)?;
    let mut e = KevEngine::load(AlignedBuf::from_slice(&bytes))?;
    eprintln!("loaded in {:.2}s", t.elapsed().as_secs_f64());
    let golden = Value::parse(&String::from_utf8(read(positional(args, 1)?)?).map_err(|_| "golden is not UTF-8")?)
        .map_err(|e| e.to_string())?;
    let (mut n, mut agree, mut worst, mut ids_ok, mut cases) = (0, 0, 0f64, 0, 0);
    let mut ms = Vec::new();
    let num = |v: &Value| v.as_usize().unwrap();
    for case in golden.get("cases").and_then(Value::as_array).ok_or("no cases")? {
        cases += 1;
        let id = case.get("id").and_then(Value::as_str).unwrap_or("?");
        let ids: Vec<u32> = case.get("ids").and_then(Value::as_array).unwrap().iter().map(|v| num(v) as u32).collect();
        let seg: Vec<usize> = case.get("seg").and_then(Value::as_array).unwrap().iter().map(num).collect();
        let decide: Vec<usize> = case.get("decide_idx").and_then(Value::as_array).unwrap().iter().map(num).collect();
        let opts: Vec<Vec<usize>> = case
            .get("opt_idx")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .map(|o| o.as_array().unwrap().iter().map(num).collect())
            .collect();
        let ls = seg.iter().filter(|&&s| s == 0).count();
        let mut branches = Vec::new();
        let mut start = ls;
        for (k, d) in decide.iter().enumerate() {
            let end = d + 1;
            branches.push(Branch {
                ids: ids[start..end].to_vec(),
                decide: d - start,
                opts: opts[k].iter().map(|o| o - start).collect(),
            });
            start = end;
        }
        let golden_enc = Encoded { state: ids[..ls].to_vec(), branches };
        // tokenizer + template parity, when the tokenizer can encode this model's text
        if let Ok((mine, _)) = e.prepare(case.get("state").unwrap(), case.get("questions").unwrap()) {
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
        let want_probs = case.get("probs").and_then(Value::as_array).unwrap();
        for (q, (got, want)) in logits[0].iter().zip(want_probs).enumerate() {
            let m = got.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let ez: Vec<f64> = got.iter().map(|v| ((v - m) as f64).exp()).collect();
            let s: f64 = ez.iter().sum();
            let p: Vec<f64> = ez.iter().map(|v| v / s).collect();
            let w: Vec<f64> = want.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let am = |v: &[f64]| v.iter().enumerate().fold(0, |b, (i, &x)| if x > v[b] { i } else { b });
            n += 1;
            if am(&p) == am(&w) {
                agree += 1;
            }
            let dp = p.iter().zip(&w).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            worst = worst.max(dp);
            eprintln!("  {id}/{q}: want {w:.3?} got {p:.3?}");
        }
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("{n} questions in {cases} cases  token ids {ids_ok}/{cases}  argmax {agree}/{n}  max|dp| {worst:.4}  median forward {:.1} ms", ms[ms.len() / 2]);
    Ok(())
}

/// Matmul kernel throughput on synthetic weights (development aid).
fn kbench() -> Result<(), String> {
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
