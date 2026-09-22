//! `kevala` command line: convert checkpoints to `.kevala` packs, answer questions, check parity
//! against the PyTorch reference, and benchmark.

use kevala::engine::Engine;
use kevala::json::Value;
use kevala::model::{build_shard, AlignedBuf, Scratch, ShardPlan, Trunk};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "kevala: System 1 decision models (Laya, Kev) without Python

usage:
  kevala convert <checkpoint-dir> -o <out.kevala> [--block 32] [--keep-f32 head.,emb] [--revision <sha>]
  kevala decide <pack.kevala> --state <json|text> --questions <json>
  kevala convert-kev --base <qwen-dir> --kev <kev-dir> -o <out.kevala> [--block 32] [--name <name>] [--source <url>] [--base-source <url>] [--kev-revision <sha>] [--base-revision <sha>]
  kevala convert-semif --base <qwen-dir> --tokenizer <tokenizer.json> -o <out.kevala> [--block 32] [--name <name>] [--source <url>] [--base-source <url>] [--base-revision <sha>] [--method-revision <sha>]
  kevala parity <pack.kevala> <golden.json> [--shards N]
  kevala parity-kev <pack.kevala> <golden-kev.json>
  kevala parity-semif <pack.kevala> <golden-semif.json> [--max-dp 0.05]
  kevala bench <pack.kevala> [--tokens 64] [--questions 1] [--runs 5] [--shards N] [--warm]
  kevala inspect <pack.kevala>
  kevala wgsl <kernel> [--f16] [--subgroups] [--rows 1-4] [--groups 1-2] [--n N --k K]
";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("convert") => convert(&args[1..]),
        Some("decide") => decide(&args[1..]),
        Some("parity") => parity(&args[1..]),
        Some("convert-kev") => convert_kev(&args[1..]),
        Some("convert-semif") => convert_semif(&args[1..]),
        Some("parity-kev") => parity_kev(&args[1..]),
        Some("parity-semif") => parity_semif(&args[1..]),
        Some("bench") => bench(&args[1..]),
        Some("inspect") => inspect(&args[1..]),
        Some("wgsl") => wgsl(&args[1..]),
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

fn read_pack(path: &str) -> Result<AlignedBuf, String> {
    let mut file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let len = file.metadata().map_err(|e| format!("{path}: {e}"))?.len();
    let len = usize::try_from(len).map_err(|_| format!("{path}: pack exceeds this platform's address space"))?;
    let mut pack = AlignedBuf::new(len);
    file.read_exact(pack.as_mut_slice()).map_err(|e| format!("{path}: {e}"))?;
    Ok(pack)
}

fn load(path: &str) -> Result<Engine, String> {
    let t = Instant::now();
    let bytes = read_pack(path)?;
    let len = bytes.len();
    let e = Engine::load(bytes)?;
    eprintln!("loaded {path} ({:.0} MB) in {:.2}s", len as f64 / 1e6, t.elapsed().as_secs_f64());
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
    let bytes = read_pack(path)?;
    let len = bytes.len();
    let mut m = kevala::runtime::load(bytes)?;
    eprintln!("loaded {path} ({}, {:.0} MB) in {:.2}s", m.arch(), len as f64 / 1e6, t.elapsed().as_secs_f64());
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

fn safetensors_header(path: &Path) -> Result<(Vec<u8>, usize), String> {
    let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix).map_err(|e| format!("{}: {e}", path.display()))?;
    let n = usize::try_from(u64::from_le_bytes(prefix))
        .map_err(|_| format!("{}: header length does not fit usize", path.display()))?;
    let total = 8usize.checked_add(n).ok_or_else(|| format!("{}: header length overflows usize", path.display()))?;
    let mut head = Vec::with_capacity(total);
    head.extend_from_slice(&prefix);
    head.resize(total, 0);
    file.read_exact(&mut head[8..]).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((head, total))
}

fn indexed_shards(dir: &str) -> Result<Vec<PathBuf>, String> {
    let root = Path::new(dir);
    let index_path = root.join("model.safetensors.index.json");
    if index_path.exists() {
        let index_bytes = std::fs::read(&index_path).map_err(|e| format!("{}: {e}", index_path.display()))?;
        let index_text =
            String::from_utf8(index_bytes).map_err(|_| format!("{} is not UTF-8", index_path.display()))?;
        let index = Value::parse(&index_text).map_err(|e| format!("{}: {e}", index_path.display()))?;
        let mut names = Vec::new();
        for (_, value) in index
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| format!("{}: missing weight_map", index_path.display()))?
        {
            let name = value
                .as_str()
                .ok_or_else(|| format!("{}: weight_map value is not a filename", index_path.display()))?;
            if !names.iter().any(|seen| seen == name) {
                names.push(name.to_string());
            }
        }
        if names.is_empty() {
            return Err(format!("{}: weight_map is empty", index_path.display()));
        }
        return names.into_iter().map(|name| safe_join(root, &name)).collect();
    }
    let single = root.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(root)
        .map_err(|e| format!("{dir}: {e}"))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    files.sort();
    match files.len() {
        0 => Err(format!("{dir}: no safetensors checkpoint or model.safetensors.index.json")),
        1 => Ok(files),
        _ => {
            Err(format!("{dir}: {} safetensors shards found but model.safetensors.index.json is missing", files.len()))
        }
    }
}

fn safe_join(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path
            .components()
            .any(|part| matches!(part, Component::ParentDir | Component::RootDir | Component::Prefix(_)))
    {
        return Err(format!("checkpoint shard path is not relative: {relative}"));
    }
    Ok(root.join(path))
}

fn repo_url(value: &str) -> String {
    if value.starts_with("http://") || value.starts_with("https://") {
        value.to_string()
    } else {
        format!("https://huggingface.co/{value}")
    }
}

fn config_repo(config: &Value, fallback_dir: &str) -> String {
    let text_config = config.get("text_config").unwrap_or(config);
    [config, text_config]
        .into_iter()
        .flat_map(|source| {
            ["_name_or_path", "name_or_path", "base_model_name_or_path"]
                .iter()
                .filter_map(move |key| source.get(key).and_then(Value::as_str))
        })
        .find(|value| !value.is_empty() && *value != "Qwen/Qwen3.5")
        .map(repo_url)
        .unwrap_or_else(|| {
            let fallback = Path::new(fallback_dir).file_name().and_then(|name| name.to_str()).unwrap_or("qwen3.5");
            repo_url(fallback)
        })
}

fn output_pack(path: &str, pack: &[u8], started: Instant) -> Result<(), String> {
    let mut file = File::create(path).map_err(|e| format!("{path}: {e}"))?;
    file.write_all(pack).map_err(|e| format!("{path}: {e}"))?;
    file.flush().map_err(|e| format!("{path}: {e}"))?;
    eprintln!("wrote {path}: {:.1} MB in {:.1}s", pack.len() as f64 / 1e6, started.elapsed().as_secs_f64());
    Ok(())
}

enum StreamMode {
    Kev { adapter: Vec<u8>, adapter_config: String, head: Vec<u8> },
    Semif,
}

fn convert_sharded(
    shards: &[PathBuf],
    base_config: &str,
    base_tokenizer: &str,
    block: usize,
    model: Value,
    mode: StreamMode,
) -> Result<Vec<u8>, String> {
    let headers: Vec<Vec<u8>> =
        shards.iter().map(|path| safetensors_header(path).map(|(head, _)| head)).collect::<Result<_, _>>()?;
    let refs: Vec<&[u8]> = headers.iter().map(Vec::as_slice).collect();
    let mut converter = match mode {
        StreamMode::Kev { adapter, adapter_config, head } => kevala::convert_kev::KevConvert::new_sharded(
            &refs,
            base_config,
            base_tokenizer,
            &adapter,
            &adapter_config,
            &head,
            block,
            model,
        )?,
        StreamMode::Semif => {
            kevala::convert_kev::KevConvert::new_semif_sharded(&refs, base_config, base_tokenizer, block, model)?
        }
    };
    let mut out = vec![0u8; converter.total];
    converter.begin(&mut out)?;
    let mut files: Vec<File> = shards
        .iter()
        .map(|path| File::open(path).map_err(|e| format!("{}: {e}", path.display())))
        .collect::<Result<_, _>>()?;
    for (shard, name, offset, len) in converter.source_ranges()? {
        let file = files.get_mut(shard).ok_or("converter returned an invalid shard index")?;
        file.seek(SeekFrom::Start(offset as u64)).map_err(|e| format!("{}: {e}", shards[shard].display()))?;
        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes).map_err(|e| format!("{}: {e}", shards[shard].display()))?;
        converter.add_source(&name, bytes, &mut out)?;
    }
    if !converter.finished() {
        return Err("conversion ended with tensors still missing".into());
    }
    Ok(out)
}

fn model_name(args: &[String], default: &str) -> String {
    flag(args, "--name").unwrap_or(default).to_string()
}

fn convert_kev(args: &[String]) -> Result<(), String> {
    let base = flag(args, "--base").ok_or("convert-kev needs --base <dir>")?;
    let kev = flag(args, "--kev").ok_or("convert-kev needs --kev <dir>")?;
    let out = flag(args, "-o").ok_or("convert-kev needs -o <out.kevala>")?;
    let block: usize = flag(args, "--block").unwrap_or("32").parse().map_err(|_| "bad --block")?;
    let started = Instant::now();
    let text = |path: String| String::from_utf8(read(&path)?).map_err(|_| format!("{path} is not UTF-8"));
    let base_config = text(format!("{base}/config.json"))?;
    let base_value = Value::parse(&base_config).map_err(|e| format!("{base}/config.json: {e}"))?;
    let adapter_config = text(format!("{kev}/adapter_config.json"))?;
    let shards = indexed_shards(base)?;
    let tokenizer = text(format!("{kev}/tokenizer.json")).or_else(|_| text(format!("{base}/tokenizer.json")))?;
    let adapter = read(&format!("{kev}/adapter_model.safetensors"))?;
    let head = read(&format!("{kev}/head.pt"))?;
    let name = model_name(args, "kev-0.8b");
    let source = flag(args, "--source")
        .map(str::to_string)
        .unwrap_or_else(|| format!("https://huggingface.co/jaredpalmer/{name}"));
    let base_source = flag(args, "--base-source").map(str::to_string).unwrap_or_else(|| config_repo(&base_value, base));
    let kev_revision = flag(args, "--kev-revision").unwrap_or("unknown");
    let base_revision = flag(args, "--base-revision").unwrap_or("unknown");
    let model = Value::Object(vec![
        ("name".into(), Value::Str(name)),
        ("source".into(), Value::Str(repo_url(&source))),
        ("revision".into(), Value::Str(kev_revision.to_string())),
        ("base".into(), Value::Str(repo_url(&base_source))),
        ("base_revision".into(), Value::Str(base_revision.to_string())),
        ("author".into(), Value::Str("Jared Palmer (Kev); Qwen team (base)".into())),
        ("license".into(), Value::Str("apache-2.0".into())),
        ("converter".into(), Value::Str(format!("kevala {}", env!("CARGO_PKG_VERSION")))),
        ("quantization".into(), Value::Str(format!("LoRA merged in f32, then int8 symmetric absmax, one f32 scale per {block} weights; norms, gates, conv, pointer head in f32"))),
    ]);
    let pack = convert_sharded(
        &shards,
        &base_config,
        &tokenizer,
        block,
        model,
        StreamMode::Kev { adapter, adapter_config, head },
    )?;
    output_pack(out, &pack, started)
}

fn convert_semif(args: &[String]) -> Result<(), String> {
    let base = flag(args, "--base").ok_or("convert-semif needs --base <dir>")?;
    let tokenizer_path = flag(args, "--tokenizer").ok_or(
        "convert-semif needs --tokenizer <tokenizer.json> saved by AutoTokenizer; use tools/convert_models.py",
    )?;
    let out = flag(args, "-o").ok_or("convert-semif needs -o <out.kevala>")?;
    let block: usize = flag(args, "--block").unwrap_or("32").parse().map_err(|_| "bad --block")?;
    let started = Instant::now();
    let text = |path: String| String::from_utf8(read(&path)?).map_err(|_| format!("{path} is not UTF-8"));
    let base_config = text(format!("{base}/config.json"))?;
    let base_value = Value::parse(&base_config).map_err(|e| format!("{base}/config.json: {e}"))?;
    let base_source = flag(args, "--base-source").map(str::to_string).unwrap_or_else(|| config_repo(&base_value, base));
    let source = flag(args, "--source").map(str::to_string).unwrap_or_else(|| base_source.clone());
    let default_name = Path::new(base)
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| format!("semif-{name}"))
        .unwrap_or_else(|| "semif-qwen3.5".into());
    let name = model_name(args, &default_name);
    let base_revision = flag(args, "--base-revision").unwrap_or("unknown");
    let method_revision = flag(args, "--method-revision").unwrap_or("1f2dea3e25379f9dfc98cb83c324f00ab5deda37");
    let tokenizer = text(tokenizer_path.to_string())?;
    let shards = indexed_shards(base)?;
    let model = Value::Object(vec![
        ("name".into(), Value::Str(name)),
        ("source".into(), Value::Str(repo_url(&source))),
        ("revision".into(), Value::Str(base_revision.to_string())),
        ("base".into(), Value::Str(repo_url(&base_source))),
        ("base_revision".into(), Value::Str(base_revision.to_string())),
        ("author".into(), Value::Str("Qwen team (weights); SemIf (decision method)".into())),
        ("license".into(), Value::Str("apache-2.0".into())),
        ("converter".into(), Value::Str(format!("kevala {}", env!("CARGO_PKG_VERSION")))),
        (
            "quantization".into(),
            Value::Str(format!(
                "int8 symmetric absmax, one f32 scale per {block} weights; norms and SemIf label readout in f32"
            )),
        ),
        ("inspiration".into(), Value::Str("SemIf direct-options-v1 readout".into())),
        ("method_source".into(), Value::Str("https://github.com/TheoLeeCJ/SemIf".into())),
        ("method_revision".into(), Value::Str(method_revision.to_string())),
        ("method_license".into(), Value::Str("mit".into())),
    ]);
    let pack = convert_sharded(&shards, &base_config, &tokenizer, block, model, StreamMode::Semif)?;
    output_pack(out, &pack, started)
}

fn parity_kev(args: &[String]) -> Result<(), String> {
    use kevala::kev::{Branch, Encoded, KevEngine};
    let t = Instant::now();
    let mut e = KevEngine::load(read_pack(positional(args, 0)?)?)?;
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
                labels: 0,
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

/// Prints a GPU kernel as the browser runtime would compile it.
fn wgsl(args: &[String]) -> Result<(), String> {
    let kernel = args
        .first()
        .filter(|a| !a.starts_with("--"))
        .ok_or_else(|| format!("which kernel? one of: {}", kevala::gpu::KERNELS.join(", ")))?;
    let number = |name: &str| -> Result<Option<u32>, String> {
        flag(args, name).map(|v| v.parse().map_err(|_| format!("bad {name}"))).transpose()
    };
    let mut spec = kevala::gpu::Spec {
        f16: args.iter().any(|a| a == "--f16"),
        subgroups: args.iter().any(|a| a == "--subgroups"),
        ..Default::default()
    };
    if let Some(r) = number("--rows")? {
        spec.rows = r;
    }
    if let Some(g) = number("--groups")? {
        spec.groups = g;
    }
    if let (Some(n), Some(k)) = (number("--n")?, number("--k")?) {
        spec.shape = Some((n, k));
    }
    print!("{}", kevala::gpu::wgsl(kernel, &spec)?);
    Ok(())
}

fn parity_semif(args: &[String]) -> Result<(), String> {
    use kevala::kev::KevEngine;
    let mut engine = KevEngine::load(read_pack(positional(args, 0)?)?)?;
    if !engine.model.cfg.semif {
        return Err("parity-semif needs a SemIf pack".into());
    }
    let golden = Value::parse(&String::from_utf8(read(positional(args, 1)?)?).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let limit: f64 = flag(args, "--max-dp").unwrap_or("0.05").parse().map_err(|_| "bad --max-dp")?;
    if !limit.is_finite() || limit < 0.0 {
        return Err("--max-dp must be finite and nonnegative".into());
    }
    let mut worst = 0.0f64;
    let mut agree = 0;
    let cases = golden.get("cases").and_then(Value::as_array).ok_or("no reference cases")?;
    if cases.is_empty() {
        return Err("no reference cases".into());
    }
    for case in cases {
        let state = case.get("state").ok_or("case has no state")?;
        let questions = case.get("questions").ok_or("case has no questions")?;
        let (encoded, qs) = engine.prepare(state, questions)?;
        if qs.len() != 1 {
            return Err("SemIf references must have one question per case".into());
        }
        let got_ids: Vec<usize> = encoded.state.iter().chain(&encoded.branches[0].ids).map(|&id| id as usize).collect();
        let want_ids: Vec<usize> = case
            .get("input_ids")
            .and_then(Value::as_array)
            .ok_or("no input_ids")?
            .iter()
            .map(|v| v.as_usize().ok_or("invalid token ID"))
            .collect::<Result<_, _>>()?;
        let id = case.get("id").and_then(Value::as_str).unwrap_or("?");
        if got_ids != want_ids {
            let first = got_ids.iter().zip(&want_ids).position(|(a, b)| a != b);
            return Err(format!("{id}: token IDs differ ({} vs {}, first {first:?})", got_ids.len(), want_ids.len()));
        }
        let responses = engine.decide(&[(state.clone(), questions.clone())])?;
        let got = responses[0]
            .get("raw_probabilities")
            .and_then(|p| p.get(&qs[0].id))
            .and_then(Value::as_array)
            .ok_or("no probabilities")?;
        let want = case
            .get("probs")
            .and_then(Value::as_array)
            .and_then(|p| p.first())
            .and_then(Value::as_array)
            .ok_or("no reference probabilities")?;
        if got.len() != want.len() || got.is_empty() {
            return Err(format!("{id}: probability count differs"));
        }
        let values =
            |a: &[Value]| a.iter().map(|v| v.as_f64().ok_or("invalid probability")).collect::<Result<Vec<_>, _>>();
        let (g, w) = (values(got)?, values(want)?);
        if !g.iter().chain(&w).all(|p| p.is_finite()) {
            return Err(format!("{id}: nonfinite probability"));
        }
        let argmax = |a: &[f64]| a.iter().enumerate().fold(0, |best, (i, &p)| if p > a[best] { i } else { best });
        let ok = argmax(&g) == argmax(&w);
        agree += usize::from(ok);
        let dp = g.iter().zip(w).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        worst = worst.max(dp);
        println!("{id}: exact tokens, argmax {ok}, max |dp| {dp:.6}");
    }
    println!("SemIf: {agree}/{} argmax, max |dp| {worst:.6}", cases.len());
    if agree != cases.len() || worst > limit {
        return Err(format!("SemIf parity failed (limit {limit})"));
    }
    Ok(())
}
