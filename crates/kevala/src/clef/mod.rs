//! Cloudflare Clef-flash text decisions: Qwen3.5 plus a joint schema transformer head.
//! The upstream image/video processor and vision tower are not part of this text runtime.

pub mod convert;
pub mod encoding;
pub mod head;

use crate::content::{Modality, Request};
use crate::json::Value;
use crate::kev::{KevConfig, KevModel};
use crate::math::{argmax, round_to, softmax};
use crate::runtime::Model;
use crate::store::{load_store, AlignedBuf};
use crate::tokenizer::Tokenizer;
use encoding::Record;
use head::{Head, HeadConfig};
use std::any::Any;
use std::sync::Arc;

pub struct ClefEngine {
    pub model: KevModel,
    pub head: Head,
    pub tok: Tokenizer,
    pub info: Value,
    pub max_length: usize,
    pub max_state: usize,
}

impl ClefEngine {
    pub fn load(buf: AlignedBuf) -> Result<Self, String> {
        let (store, header) = load_store(buf)?;
        let cfg = KevConfig::from_json(header.config())?;
        let hc = HeadConfig::from_json(header.config().get("joint_head").ok_or("clef: missing joint_head config")?)?;
        if hc.hidden != cfg.hidden {
            return Err("clef: head and backbone hidden sizes differ".into());
        }
        let tok = Tokenizer::from_bytes(
            &store.bytes()[header.tokenizer_offset..header.tokenizer_offset + header.tokenizer_size],
        )?;
        let store = Arc::new(store);
        let model = KevModel::new_trunk(cfg, store.clone())?;
        let head = Head::new(hc, store)?;
        Ok(Self {
            model,
            head,
            tok,
            info: header.json.get("model").cloned().unwrap_or(Value::Null),
            max_length: header.config().get("max_length").and_then(Value::as_usize).unwrap_or(16384),
            max_state: header.config().get("max_state_tokens").and_then(Value::as_usize).unwrap_or(usize::MAX),
        })
    }

    pub fn prepare(&self, requests: &[Request]) -> Result<Vec<Record>, String> {
        requests
            .iter()
            .map(|r| {
                r.check("clef", &[Modality::Text])?;
                let state = r.text_with(encoding::render).map(Value::Str).unwrap_or_else(|| r.state.clone());
                encoding::encode(&self.tok, &state, &r.questions, self.max_length, self.max_state)
            })
            .collect()
    }

    pub fn respond(&self, record: &Record, logits: &[Vec<f32>]) -> Result<Value, String> {
        if logits.len() != record.questions.len() {
            return Err("clef: logits do not match field count".into());
        }
        let rounded = |x: f32| Value::Float(round_to(x as f64, 4));
        let mut answers = Vec::new();
        for (q, logits) in record.questions.iter().zip(logits) {
            if logits.len() != q.option_ids.len() {
                return Err("clef: logits do not match option count".into());
            }
            let probabilities = softmax(logits);
            let probability = |id: &str| probabilities[q.option_ids.iter().position(|key| key == id).unwrap()];
            let distribution =
                || Value::Object(q.response_ids.iter().map(|id| (id.clone(), rounded(probability(id)))).collect());
            let answer = match q.kind {
                0 => Value::Object(vec![
                    ("type".into(), Value::Str("noul".into())),
                    ("noul".into(), rounded(probability("true"))),
                ]),
                1 => {
                    let ordered: Vec<_> = q.response_ids.iter().map(|id| probability(id)).collect();
                    let choice = &q.response_ids[argmax(&ordered)];
                    Value::Object(vec![
                        ("type".into(), Value::Str("choice".into())),
                        ("choice".into(), Value::Str(choice.clone())),
                        ("confidence".into(), rounded(probability(choice))),
                        ("probabilities".into(), distribution()),
                    ])
                }
                _ => {
                    let score = probabilities.iter().enumerate().map(|(i, p)| i as f64 * *p as f64).sum::<f64>();
                    let legend = Value::Object(
                        q.option_ids.iter().cloned().zip(q.criteria.as_array().unwrap().iter().cloned()).collect(),
                    );
                    Value::Object(vec![
                        ("type".into(), Value::Str("score".into())),
                        ("score".into(), Value::Float(round_to(score, 4))),
                        ("confidence".into(), rounded(probabilities[argmax(&probabilities)])),
                        ("legend".into(), legend),
                        ("probabilities".into(), distribution()),
                    ])
                }
            };
            answers.push((q.id.clone(), answer));
        }
        Ok(Value::Object(vec![
            ("model".into(), self.info.get("name").cloned().unwrap_or_else(|| Value::Str("clef-flash".into()))),
            ("answers".into(), Value::Object(answers)),
            (
                "usage".into(),
                Value::Object(vec![
                    ("input_tokens".into(), Value::Int(record.ids.len().to_string())),
                    ("output_tokens".into(), Value::Int("0".into())),
                ]),
            ),
        ]))
    }
}

impl Model for ClefEngine {
    fn arch(&self) -> &'static str {
        "clef"
    }
    fn info(&self) -> &Value {
        &self.info
    }
    fn modalities(&self) -> &[Modality] {
        &[Modality::Text]
    }
    fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }
    fn decide(&mut self, requests: &[Request]) -> Result<Vec<Value>, String> {
        if !self.model.has_layers() {
            return Err("this Clef coordinator requires an external GPU backbone".into());
        }
        let records = self.prepare(requests)?;
        records
            .iter()
            .map(|record| {
                let hidden = self.model.forward_hidden(&record.ids);
                let logits = self.head.forward(&hidden, record)?;
                self.respond(record, &logits)
            })
            .collect()
    }
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}
