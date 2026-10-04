//! `kevala` command line: convert checkpoints to `.kevala` packs, answer questions, check parity
//! against the PyTorch reference, and benchmark.

mod bench;
mod conversion;
mod parity;
mod sharded;

use kevala::json::Value;
use kevala::laya::Engine;
use kevala::store::AlignedBuf;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

const USAGE: &str = "kevala: local decision models without Python

usage:
  kevala convert <checkpoint-dir> -o <out.kevala> [--adapter <dir>] [--base <dir>] [--readout direct-options|pointer|encoder-head] [--block 32]
    [--tokenizer <tokenizer.json>] [--name <name>] [--source <url>] [--revision <sha>]
    [--base-source <url>] [--base-revision <sha>] [--author <author>] [--license <license>] [--method-revision <sha>] [--keep-f32 <prefixes>]
  kevala decide <pack.kevala> --state <json|text> --questions <json>
  kevala parity <pack.kevala> <golden.json> [--shards N]
  kevala parity-kev <pack.kevala> <golden-kev.json>
  kevala parity-semif <pack.kevala> <golden-semif.json> [--max-dp 0.05]
  kevala parity-gemma <pack.kevala> <golden-gemma.json> [--max-dp 0.03]
  kevala bench <pack.kevala> [--tokens 64] [--questions 1] [--runs 5] [--shards N] [--warm]
  kevala kbench
  kevala inspect <pack.kevala>
  kevala wgsl <kernel> [--f16] [--subgroups] [--rows 1-4] [--groups 1-2] [--n N --k K]
";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("convert" | "convert-kev" | "convert-semif" | "convert-gemma") => conversion::run(&args[0], &args[1..]),
        Some("decide") => decide(&args[1..]),
        Some("parity") => parity::laya::run(&args[1..]),
        Some("parity-kev") => parity::kev::run(&args[1..]),
        Some("parity-semif") => parity::semif::run(&args[1..]),
        Some("parity-gemma") => parity::gemma::run(&args[1..]),
        Some("bench") => bench::run(&args[1..]),
        Some("kbench") => bench::kernels(),
        Some("inspect") => inspect(&args[1..]),
        Some("wgsl") => wgsl(&args[1..]),
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

/// Options that take no value; every other `--option` consumes the argument after it.
const SWITCHES: [&str; 3] = ["--warm", "--f16", "--subgroups"];

/// The `n`th argument that is neither an option nor an option's value.
fn positional(args: &[String], n: usize) -> Result<&str, String> {
    let mut seen = 0;
    let mut i = 0;
    while i < args.len() {
        if SWITCHES.contains(&args[i].as_str()) {
            i += 1;
            continue;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    #[test]
    fn positionals_skip_option_values_but_not_the_argument_after_a_switch() {
        let a = args(&["--tokens", "128", "pack.kevala", "--shards", "2", "golden.json"]);
        assert_eq!(positional(&a, 0).unwrap(), "pack.kevala");
        assert_eq!(positional(&a, 1).unwrap(), "golden.json");
        assert!(positional(&a, 2).is_err());
        // A switch takes no value, so the pack path after it is still the first positional.
        assert_eq!(positional(&args(&["--warm", "pack.kevala"]), 0).unwrap(), "pack.kevala");
        assert_eq!(flag(&a, "--shards"), Some("2"));
        assert_eq!(flag(&a, "--runs"), None);
    }

    #[test]
    fn state_arguments_are_json_only_for_objects_and_arrays() {
        assert!(matches!(state_arg(r#"{"a": 1}"#), Value::Object(_)));
        assert!(matches!(state_arg("[1, 2]"), Value::Array(_)));
        // Scalars that happen to parse as JSON stay literal text, as a caller typed them.
        assert!(matches!(state_arg("42"), Value::Str(s) if s == "42"));
        assert!(matches!(state_arg("refund me"), Value::Str(s) if s == "refund me"));
    }
}
