use kevala::clef::ClefEngine;
use kevala::content::Request;
use kevala::kev::KevModel;
use kevala::math::{argmax, softmax};
use kevala::pack::{self, Header, Layout, Slice};
use kevala::store::{load_store, AlignedBuf};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;
use std::time::Instant;

use super::{array, case_id, cases, f64s, field, golden, max_abs_diff, max_dp, median, usizes};
use crate::{positional, read_pack};

fn load_subset(file: &mut File, layout: Layout) -> Result<AlignedBuf, String> {
    let mut out = AlignedBuf::new(layout.total);
    out.as_mut_slice()[..layout.prefix.len()].copy_from_slice(&layout.prefix);
    for piece in layout.pieces {
        for row in 0..piece.rows {
            file.seek(SeekFrom::Start((piece.src + row * piece.stride) as u64))
                .map_err(|e| format!("pack seek: {e}"))?;
            file.read_exact(&mut out.as_mut_slice()[piece.dst + row * piece.len..piece.dst + (row + 1) * piece.len])
                .map_err(|e| format!("pack read: {e}"))?;
        }
    }
    Ok(out)
}

struct Streamed {
    file: File,
    header: Header,
}

impl Streamed {
    fn open(path: &str) -> Result<(Self, ClefEngine), String> {
        let mut file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let mut prefix = vec![0u8; 16];
        file.read_exact(&mut prefix).map_err(|e| format!("{path}: {e}"))?;
        let length = pack::header_len(&prefix)?;
        prefix.resize(length, 0);
        file.read_exact(&mut prefix[16..]).map_err(|e| format!("{path}: {e}"))?;
        let header = pack::parse_header(&prefix)?;
        if header.total_size as u64 > file.metadata().map_err(|e| format!("{path}: {e}"))?.len() {
            return Err(format!("{path}: truncated pack"));
        }
        let layout = pack::subset(&header, &|t| (!t.name.starts_with("L.")).then_some(Slice::Whole), true, vec![])?;
        let engine = ClefEngine::load(load_subset(&mut file, layout)?)?;
        Ok((Self { file, header }, engine))
    }

    fn forward(&mut self, engine: &mut ClefEngine, ids: &[u32]) -> Result<Vec<f32>, String> {
        let mut hidden = engine.model.embed(ids);
        for i in 0..engine.model.cfg.layers {
            let prefix = format!("L.{i}.");
            let mut header = self.header.clone();
            header.tensors.retain(|t| t.name == "norm" || t.name.starts_with(&prefix));
            for t in &mut header.tensors {
                if let Some(suffix) = t.name.strip_prefix(&prefix) {
                    t.name = format!("L.0.{suffix}");
                }
            }
            let layout = pack::subset(&header, &|_| Some(Slice::Whole), false, vec![])?;
            let (store, _) = load_store(load_subset(&mut self.file, layout)?)?;
            let mut cfg = engine.model.cfg.clone();
            cfg.layers = 1;
            cfg.full = vec![cfg.full[i]];
            let mut layer = KevModel::new_trunk(cfg, Arc::new(store))?;
            layer.forward_rows(&mut hidden)?;
        }
        engine.model.normalize_hidden(&mut hidden);
        Ok(hidden)
    }
}

pub fn run(args: &[String]) -> Result<(), String> {
    let reference = golden(args)?;
    let limit = max_dp(args, "0.03")?;
    let path = positional(args, 0)?;
    let (mut stream, mut engine) = if args.iter().any(|a| a == "--stream-layers") {
        let (stream, engine) = Streamed::open(path)?;
        (Some(stream), engine)
    } else {
        (None, ClefEngine::load(read_pack(path)?)?)
    };
    if stream.is_none() && !engine.model.has_layers() {
        return Err("parity-clef requires a complete native pack".into());
    }
    let mut worst = 0.0f64;
    let mut agreements = 0;
    let mut fields = 0;
    let mut timings = Vec::new();
    for case in cases(&reference)? {
        let id = case_id(case);
        let request = Request {
            state: field(case, "state")?.clone(),
            parts: vec![],
            questions: field(case, "questions")?.clone(),
        };
        let record = engine.prepare(&[request])?.remove(0);
        let ids = usizes(array(case, "input_ids")?, "input_ids")?;
        if record.ids.iter().map(|id| *id as usize).collect::<Vec<_>>() != ids {
            return Err(format!("{id}: token IDs differ from the upstream Clef encoder"));
        }
        let questions = array(case, "encoded_questions")?;
        if record.questions.len() != questions.len() {
            return Err(format!("{id}: field count differs from the upstream encoder"));
        }
        for (q, want) in record.questions.iter().zip(questions) {
            let span = usizes(array(want, "span")?, "span")?;
            let spans = array(want, "option_spans")?
                .iter()
                .map(|v| usizes(v.as_array().ok_or("golden option span is not an array")?, "option span"))
                .collect::<Result<Vec<_>, String>>()?;
            let keys = array(want, "option_ids")?
                .iter()
                .map(|v| v.as_str().ok_or("golden option ID is not a string"))
                .collect::<Result<Vec<_>, _>>()?;
            if q.id != field(want, "id")?.as_str().ok_or("golden field ID is not a string")?
                || q.kind != field(want, "kind")?.as_usize().ok_or("golden field kind is not an integer")?
                || span != [q.span.0, q.span.1]
                || spans != q.option_spans.iter().map(|(a, b)| vec![*a, *b]).collect::<Vec<_>>()
                || keys != q.option_ids.iter().map(String::as_str).collect::<Vec<_>>()
            {
                return Err(format!("{id}/{}: spans or options differ from the upstream encoder", q.id));
            }
        }
        let start = Instant::now();
        let hidden = if let Some(stream) = &mut stream {
            stream.forward(&mut engine, &record.ids)?
        } else {
            engine.model.forward_hidden(&record.ids)
        };
        let logits = engine.head.forward(&hidden, &record)?;
        timings.push(start.elapsed().as_secs_f64() * 1000.0);
        let probabilities = array(case, "probs")?;
        let expected_logits = array(case, "logits")?;
        if probabilities.len() != logits.len() || expected_logits.len() != logits.len() {
            return Err(format!("{id}: reference distribution count differs"));
        }
        for (qi, ((q, logits), want)) in record.questions.iter().zip(&logits).zip(probabilities).enumerate() {
            let got: Vec<_> = softmax(logits).iter().map(|p| *p as f64).collect();
            let want = f64s(want.as_array().ok_or("golden distribution is not an array")?, "probabilities")?;
            let expected_logits =
                f64s(expected_logits[qi].as_array().ok_or("golden logits is not an array")?, "logits")?;
            if got.len() != want.len()
                || logits.len() != expected_logits.len()
                || want.is_empty()
                || !got.iter().chain(&want).all(|p| p.is_finite() && (0.0..=1.0).contains(p))
                || !logits.iter().all(|p| p.is_finite())
                || !expected_logits.iter().all(|p| p.is_finite())
            {
                return Err(format!("{id}/{}: malformed probabilities or logits", q.id));
            }
            let agreement = argmax(&got) == argmax(&want);
            let difference = max_abs_diff(&got, &want);
            let logit_difference =
                max_abs_diff(&logits.iter().map(|v| *v as f64).collect::<Vec<_>>(), &expected_logits);
            agreements += usize::from(agreement);
            fields += 1;
            worst = worst.max(difference);
            println!("{id}/{}: exact tokens/spans, argmax {agreement}, max |dp| {difference:.6}, max |dlogit| {logit_difference:.6}", q.id);
        }
        println!("{id}: {:.1} ms native full-record forward", timings.last().unwrap());
    }
    println!(
        "Clef: {agreements}/{fields} field argmax, max |dp| {worst:.6}, median native forward {:.1} ms",
        median(&mut timings)
    );
    if agreements != fields || worst > limit {
        return Err(format!("Clef parity failed (probability limit {limit})"));
    }
    Ok(())
}
