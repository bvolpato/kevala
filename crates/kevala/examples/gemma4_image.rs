//! Native decoder validation with projected rows from a local Gemma4 vision reference.
//! `cargo run --release -p kevala --example gemma4_image -- decoder.kevala fixture.json`
//! The fixture holds `requests`, flattened f32 rows per image in `projected_rows`, and optional
//! `input_ids` per sequence. Output records the prepared token IDs, image spans and responses.

use kevala::content::Request;
use kevala::gemma4::Gemma4Engine;
use kevala::json::Value;
use kevala::store::AlignedBuf;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: gemma4_image <decoder.kevala> <fixture.json>".into());
    }
    let text = std::fs::read_to_string(&args[1]).map_err(|e| format!("{}: {e}", args[1]))?;
    let fixture = Value::parse(&text).map_err(|e| e.to_string())?;
    let requests = Request::parse_many(&fixture)?;
    let projected = fixture
        .get("projected_rows")
        .and_then(Value::as_array)
        .ok_or("fixture has no projected_rows")?
        .iter()
        .map(|image| {
            image
                .as_array()
                .ok_or("projected_rows must contain flattened arrays")?
                .iter()
                .map(|v| {
                    let value = v.as_f64().ok_or("projected_rows contains a non-number")? as f32;
                    if !value.is_finite() {
                        return Err("projected_rows contains a non-finite f32".into());
                    }
                    Ok(value)
                })
                .collect::<Result<Vec<f32>, String>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut file = File::open(&args[0]).map_err(|e| format!("{}: {e}", args[0]))?;
    let len = usize::try_from(file.metadata().map_err(|e| e.to_string())?.len()).map_err(|_| "pack is too large")?;
    let mut bytes = AlignedBuf::new(len);
    file.read_exact(bytes.as_mut_slice()).map_err(|e| e.to_string())?;
    let mut engine = Gemma4Engine::load(bytes)?;
    let prepared = engine.prepare_images_all(&requests)?;
    if let Some(expected) = fixture.get("input_ids") {
        let expected = expected.as_array().ok_or("input_ids must contain arrays per sequence")?;
        if expected.len() != prepared.sequence_count() {
            return Err("input_ids sequence count differs".into());
        }
        for (i, ids) in expected.iter().enumerate() {
            let ids: Vec<_> = ids
                .as_array()
                .ok_or("input_ids must contain arrays per sequence")?
                .iter()
                .map(|v| v.as_usize().and_then(|n| u32::try_from(n).ok()).ok_or("invalid input token ID"))
                .collect::<Result<_, _>>()?;
            if ids != prepared.sequence_ids(i) {
                return Err(format!("sequence {i}: token IDs differ"));
            }
        }
    }
    let started = Instant::now();
    let responses = engine.decide_native_images(&requests, &projected)?;
    eprintln!("native image decoder: {:.1}s", started.elapsed().as_secs_f64());
    let sequences = prepared
        .sequences
        .iter()
        .enumerate()
        .map(|(i, sequence)| {
            Value::Object(vec![
                (
                    "ids".into(),
                    Value::Array(prepared.sequence_ids(i).iter().map(|id| Value::Int(id.to_string())).collect()),
                ),
                (
                    "image_spans".into(),
                    Value::Array(
                        sequence
                            .image_spans
                            .iter()
                            .map(|span| {
                                Value::Object(vec![
                                    ("start".into(), Value::Int(span.start.to_string())),
                                    ("len".into(), Value::Int(span.len.to_string())),
                                    ("image_index".into(), Value::Int(span.image_index.to_string())),
                                ])
                            })
                            .collect(),
                    ),
                ),
            ])
        })
        .collect();
    println!(
        "{}",
        Value::Object(vec![
            ("sequences".into(), Value::Array(sequences)),
            ("responses".into(), Value::Array(responses)),
        ])
        .to_json()
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
