//! Laya: a ModernBERT encoder with a decision head. Requests in, Laya-shaped answers out.
//!
//! The response matches the upstream `laya` SDK (0.3.5) field for field: `choice` / `score` /
//! `noul` answers with probabilities rounded to four places, entropy confidence, the act head's
//! `action.act_probability`, and token usage. Temperatures are clamped to [0.5, 5] exactly as
//! the SDK does.

pub mod convert;
pub mod model;
pub mod sequence;

use crate::json::Value;
use crate::math::{argmax, round_to, softmax};
use crate::pack;
use crate::store::{AlignedBuf, Store};
use crate::tokenizer::Tokenizer;
use model::{Batch, Config, Coord, Scratch, SegOut, Trunk};
use sequence::{QType, Question};
use std::sync::Arc;

pub const TEMP_MIN: f32 = 0.5;
pub const TEMP_MAX: f32 = 5.0;

fn clamp_temperature(t: f64) -> f32 {
    if !t.is_finite() {
        return 1.0;
    }
    (t as f32).clamp(TEMP_MIN, TEMP_MAX)
}

#[derive(Clone, Debug)]
pub struct Temperatures {
    pub by_type: [f32; 3],
    pub by_options: Vec<(String, f32)>,
}

impl Temperatures {
    pub fn from_config(cfg: &Value) -> Temperatures {
        let mut by_type = [1.0; 3];
        if let Some(a) = cfg.get("temperature").and_then(Value::as_array) {
            for (i, v) in a.iter().take(3).enumerate() {
                by_type[i] = v.as_f64().map_or(1.0, clamp_temperature);
            }
        }
        let by_options = cfg
            .get("temperature_by_options")
            .and_then(Value::as_object)
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.as_f64().map_or(1.0, clamp_temperature))).collect())
            .unwrap_or_default();
        Temperatures { by_type, by_options }
    }

    /// `temp_bucket`: a 2-option noul and a 20-option choice need different scaling.
    pub fn for_question(&self, qtype: QType, k: usize) -> f32 {
        let size = if k <= 2 {
            "2"
        } else if k <= 5 {
            "3-5"
        } else if k <= 10 {
            "6-10"
        } else {
            "11+"
        };
        let key = format!("{}:{size}", qtype.name());
        self.by_options.iter().find(|(k, _)| *k == key).map_or(self.by_type[qtype as usize], |(_, t)| *t)
    }
}

/// A request turned into packed sequences.
pub struct Prepared {
    pub batch: Batch,
    /// One per segment: which request it came from and the parsed question.
    pub items: Vec<(usize, Question)>,
    pub requests: usize,
}

/// Probabilities for one question, before formatting.
#[derive(Clone, Debug)]
pub struct Scored {
    pub probs: Vec<f32>,
    pub logits: Vec<f32>,
    pub act_probability: f32,
    pub temperature: f32,
}

pub struct Engine {
    pub cfg: Config,
    pub modalities: Vec<crate::content::Modality>,
    pub tok: Tokenizer,
    pub temps: Temperatures,
    /// Provenance from the pack header.
    pub info: Value,
    pub coord: Coord,
    trunk: Option<Trunk>,
    scratch: Scratch,
}

fn confidence(p: &[f32]) -> f32 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let ent: f32 = -p.iter().map(|&v| v * v.clamp(1e-12, 1.0).ln()).sum::<f32>();
    1.0 - ent / (k as f32).ln()
}

impl Engine {
    /// Loads a complete pack that is already in memory.
    pub fn load(buf: AlignedBuf) -> Result<Engine, String> {
        let header = pack::parse_header(buf.as_slice())?;
        if buf.len() < header.total_size {
            return Err(format!("pack is truncated: {} of {} bytes", buf.len(), header.total_size));
        }
        let cfg = Config::from_json(header.config())?;
        let tok = Tokenizer::from_bytes(
            &buf.as_slice()[header.tokenizer_offset..header.tokenizer_offset + header.tokenizer_size],
        )?;
        let temps = Temperatures::from_config(header.config());
        let info = header.json.get("model").cloned().unwrap_or(Value::Null);
        // the trunk and the coordinator read the same bytes in place
        let store = Arc::new(Store::new(buf, header.tensors.clone())?);
        let coord = Coord::new(cfg.clone(), store.clone())?;
        // a coordinator sub-pack has no layers: the trunk then runs in shards or on the GPU
        let trunk = if store.has("enc.0.wqkv") { Some(Trunk::new(cfg.clone(), store, true)?) } else { None };
        let modalities = crate::content::Modality::from_config(header.config());
        Ok(Engine { cfg, modalities, tok, temps, info, coord, trunk, scratch: Scratch::default() })
    }

    /// Tokenizes every question of every request into one packed batch.
    pub fn prepare(&self, requests: &[(&Value, &Value)]) -> Result<Prepared, String> {
        let mut batch = Batch::default();
        let mut items = Vec::new();
        for (ri, (state, questions)) in requests.iter().enumerate() {
            let qs = sequence::parse_questions(questions)?;
            let text = sequence::serialize_state(state);
            for q in qs {
                let seq =
                    sequence::build_sequence(&self.tok, &text, &q, self.cfg.max_len, self.cfg.head_max_len, false);
                if seq.markers.len() != q.options.len() {
                    return Err(format!("question {:?} options exceed head_max_len={}", q.id, self.cfg.head_max_len));
                }
                batch.push(&seq.ids, q.qtype as usize, &seq.markers);
                items.push((ri, q));
            }
        }
        Ok(Prepared { batch, items, requests: requests.len() })
    }

    /// Checks each request's modalities, folds its text parts into the state, and tokenizes the
    /// batch. Returns the questions next to it, which `respond` echoes criteria from. Every
    /// backend prepares through this, so they all read the same request.
    pub fn prepare_requests(&self, requests: &[crate::content::Request]) -> Result<(Prepared, Vec<Value>), String> {
        let rs = crate::runtime::text_requests("laya", &self.modalities, requests, sequence::serialize_state)?;
        let pairs: Vec<(&Value, &Value)> = rs.iter().map(|(s, q)| (s, q)).collect();
        let prepared = self.prepare(&pairs)?;
        Ok((prepared, rs.into_iter().map(|(_, q)| q).collect()))
    }

    /// Runs the whole model in this thread. A coordinator sub-pack has no layers to run.
    pub fn forward(&mut self, batch: &Batch) -> Result<Vec<SegOut>, String> {
        let trunk =
            self.trunk.as_ref().ok_or("Laya pack is coordinator-only; its layers run in shards or on WebGPU")?;
        let cfg = &self.coord.cfg;
        let mut x = self.coord.embed(batch);
        let mut out = vec![0.0; batch.tokens() * cfg.hidden];
        for s in 0..cfg.steps() {
            if s == 2 * cfg.layers {
                self.coord.bridge(&mut x, batch);
            }
            trunk.step(s, &x, batch, &mut out, &mut self.scratch);
            for (a, b) in x.iter_mut().zip(&out) {
                *a += b;
            }
        }
        Ok(self.coord.score(&x, batch))
    }

    pub fn score(&self, prepared: &Prepared, outs: &[SegOut]) -> Vec<Scored> {
        prepared
            .items
            .iter()
            .zip(outs)
            .map(|((_, q), o)| {
                let t = self.temps.for_question(q.qtype, o.logits.len());
                let z: Vec<f32> = o.logits.iter().map(|v| v / t).collect();
                let act = softmax(&o.act_logits);
                Scored { probs: softmax(&z), logits: o.logits.clone(), act_probability: act[0], temperature: t }
            })
            .collect()
    }

    /// One Laya response object per request.
    pub fn respond(&self, prepared: &Prepared, scored: &[Scored], questions: &[&Value]) -> Vec<Value> {
        let mut answers: Vec<Vec<(String, Value)>> = vec![Vec::new(); prepared.requests];
        let mut tokens = vec![0usize; prepared.requests];
        for (((ri, q), s), seg) in prepared.items.iter().zip(scored).zip(&prepared.batch.segs) {
            tokens[*ri] += seg.len;
            let raw = questions.get(*ri).and_then(|qs| qs.get(&q.id));
            answers[*ri].push((q.id.clone(), format_answer(q, s, raw)));
        }
        answers
            .into_iter()
            .zip(tokens)
            .map(|(a, n)| {
                Value::Object(vec![
                    ("model".into(), Value::Str("laya-rl-agent".into())),
                    ("answers".into(), Value::Object(a)),
                    (
                        "usage".into(),
                        Value::Object(vec![
                            ("input_tokens".into(), Value::Int(n.to_string())),
                            ("output_tokens".into(), Value::Int("0".into())),
                        ]),
                    ),
                ])
            })
            .collect()
    }
}

fn fnum(x: f64) -> Value {
    Value::Float(round_to(x, 4))
}

fn format_answer(q: &Question, s: &Scored, raw: Option<&Value>) -> Value {
    let p = &s.probs;
    let action = Value::Object(vec![("act_probability".into(), fnum(s.act_probability as f64))]);
    let conf = fnum(confidence(p) as f64);
    match q.qtype {
        QType::Choice => Value::Object(vec![
            ("type".into(), Value::Str("choice".into())),
            ("choice".into(), Value::Str(q.keys[argmax(p)].clone())),
            (
                "probabilities".into(),
                Value::Object(q.keys.iter().zip(p).map(|(k, &v)| (k.clone(), fnum(v as f64))).collect()),
            ),
            ("confidence".into(), conf),
            ("action".into(), action),
        ]),
        QType::Score => {
            let expected: f64 = p.iter().enumerate().map(|(i, &v)| i as f64 * v as f64).sum();
            // the legend echoes the caller's criteria values, structured ones included
            let crit = raw.and_then(|r| r.get("criteria")).and_then(Value::as_array);
            let legend = (0..p.len())
                .map(|i| {
                    let v = crit.and_then(|c| c.get(i)).cloned().unwrap_or_else(|| Value::Str(q.legend[i].clone()));
                    (i.to_string(), v)
                })
                .collect();
            Value::Object(vec![
                ("type".into(), Value::Str("score".into())),
                ("score".into(), fnum(expected)),
                ("legend".into(), Value::Object(legend)),
                (
                    "probabilities".into(),
                    Value::Object(p.iter().enumerate().map(|(i, &v)| (i.to_string(), fnum(v as f64))).collect()),
                ),
                ("confidence".into(), conf),
                ("action".into(), action),
            ])
        }
        QType::Noul => {
            let t = p[1] as f64;
            Value::Object(vec![
                ("type".into(), Value::Str("noul".into())),
                ("noul".into(), fnum(t)),
                ("confidence".into(), fnum(t.max(1.0 - t))),
                ("action".into(), action),
            ])
        }
    }
}

impl crate::runtime::Model for Engine {
    fn arch(&self) -> &'static str {
        "laya"
    }
    fn info(&self) -> &Value {
        &self.info
    }
    fn modalities(&self) -> &[crate::content::Modality] {
        &self.modalities
    }
    fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }
    fn decide(&mut self, requests: &[crate::content::Request]) -> Result<Vec<Value>, String> {
        let (p, questions) = self.prepare_requests(requests)?;
        let outs = self.forward(&p.batch)?;
        let scored = self.score(&p, &outs);
        Ok(self.respond(&p, &scored, &questions.iter().collect::<Vec<_>>()))
    }
    fn as_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
