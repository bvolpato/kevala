//! H2O-Lightning's text prompt and native label readout contract.

pub mod convert;

use crate::json::{self, Value};
use crate::kev::{Branch, Encoded, KevQuestion, Kind};
use crate::tokenizer::Tokenizer;

#[derive(Clone, Debug)]
pub struct Config {
    pub model: String,
    pub labels: Vec<String>,
    pub temperatures: [f64; 3], // choice, noul, score
    pub noul_floor: f64,
    pub max_state_tokens: usize,
    system: Option<String>,
    chat: bool,
    layout: String,
    compact: bool,
    prefill: String,
    default_instructions: String,
}

impl Config {
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let p = v.get("prompt").ok_or("H2O serve config: missing prompt")?;
        if p.get("enable_thinking").and_then(Value::as_bool) == Some(true) {
            return Err("H2O prompt.enable_thinking must be false".into());
        }
        let max_options = v.get("max_options").and_then(Value::as_usize).unwrap_or(255);
        let labels = v
            .get("labels")
            .and_then(Value::as_array)
            .ok_or("H2O serve config: missing labels")?
            .iter()
            .take(max_options)
            .map(|v| v.as_str().map(str::to_string).ok_or("H2O labels must be strings"))
            .collect::<Result<Vec<_>, _>>()?;
        if labels.is_empty() || labels.len() > 255 || labels.iter().enumerate().any(|(i, x)| labels[..i].contains(x)) {
            return Err("H2O labels must contain 1..255 distinct strings".into());
        }
        let t = v.get("temperature").ok_or("H2O serve config: missing temperature")?;
        if let Some(entries) = t.as_object() {
            if entries.len() != 3
                || entries.iter().any(|(kind, _)| !["choice", "noul", "score"].contains(&kind.as_str()))
            {
                return Err("H2O temperature must give exactly choice, noul, and score".into());
            }
        }
        let mut temperatures = [0.0; 3];
        for (i, kind) in ["choice", "noul", "score"].iter().enumerate() {
            temperatures[i] = t
                .as_f64()
                .or_else(|| t.get(kind).and_then(Value::as_f64))
                .ok_or_else(|| format!("H2O temperature: missing {kind}"))?;
            if !temperatures[i].is_finite() || !(0.05..=20.0).contains(&temperatures[i]) {
                return Err("H2O temperature must be finite and in [0.05, 20]".into());
            }
        }
        let noul_floor = v.get("noul_floor").and_then(Value::as_f64).ok_or("H2O serve config: missing noul_floor")?;
        if !noul_floor.is_finite() || (noul_floor != 0.0 && !(0.5 < noul_floor && noul_floor < 1.0)) {
            return Err("H2O noul_floor must be 0 or in (0.5, 1)".into());
        }
        let layout = p.get("layout").and_then(Value::as_str).unwrap_or("plain").to_string();
        if layout != "plain" && layout != "markdown" {
            return Err(format!("H2O unsupported prompt layout {layout:?}"));
        }
        Ok(Self {
            model: v.get("model").and_then(Value::as_str).ok_or("H2O serve config: missing model")?.to_string(),
            labels,
            temperatures,
            noul_floor,
            max_state_tokens: p.get("max_state_tokens").and_then(Value::as_usize).unwrap_or(32000),
            system: p.get("system").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string),
            chat: p.get("chat_template").and_then(Value::as_bool).unwrap_or(true),
            layout,
            compact: p.get("state_compact").and_then(Value::as_bool).unwrap_or(true),
            prefill: p.get("prefill").and_then(Value::as_str).unwrap_or("Answer:").to_string(),
            default_instructions: p
                .get("default_instructions")
                .and_then(Value::as_str)
                .unwrap_or("Answer the question below.")
                .to_string(),
        })
    }

    pub fn state_text(&self, state: &Value) -> String {
        match state {
            Value::Str(s) => s.clone(),
            _ => dumps(state, !self.compact, !self.compact),
        }
    }

    pub fn prompt(&self, state_text: &str, q: &KevQuestion) -> String {
        let options = self
            .labels
            .iter()
            .zip(&q.keys)
            .zip(&q.options)
            .map(|((label, name), desc)| format!("{label}) {name}: {desc}"))
            .collect::<Vec<_>>()
            .join("\n");
        let user = if self.layout == "plain" {
            format!("record: {state_text}\nquestion: {}\noptions:\n{options}", q.instructions)
        } else {
            format!("### RECORD\n{state_text}\n\n### QUESTION\n{}\n\n### OPTIONS\n{options}\n", q.instructions)
        };
        if !self.chat {
            return match &self.system {
                Some(system) => format!("{system}\n\n{user}\n{}", self.prefill),
                None => format!("{user}\n{}", self.prefill),
            };
        }
        let system =
            self.system.as_ref().map(|s| format!("<|im_start|>system\n{}<|im_end|>\n", py_trim(s))).unwrap_or_default();
        format!(
            "{system}<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{}",
            py_trim(&user),
            self.prefill
        )
    }

    pub fn label_token_ids(&self, tok: &Tokenizer) -> Result<Vec<u32>, String> {
        let n = self.labels.len();
        let q = KevQuestion {
            id: "probe".into(),
            kind: Kind::Choice,
            instructions: "Which?".into(),
            keys: (0..n).map(|i| format!("o{i}")).collect(),
            options: (0..n).map(|i| format!("d{i}")).collect(),
            threshold: None,
        };
        let prompt = self.prompt(&self.state_text(&Value::Object(vec![("x".into(), Value::Int("1".into()))])), &q);
        let base = tok.encode(&prompt);
        let mut ids = Vec::with_capacity(n);
        for label in &self.labels {
            let full = tok.encode(&format!("{prompt} {label}"));
            if full.len() != base.len() + 1 || !full.starts_with(&base) || ids.contains(full.last().unwrap()) {
                return Err(format!("H2O label {label:?} is not one distinct token at the answer slot"));
            }
            ids.push(*full.last().unwrap());
        }
        Ok(ids)
    }

    pub fn parse_questions(&self, questions: &Value) -> Result<Vec<KevQuestion>, String> {
        let questions = questions.as_object().ok_or("H2O questions must be an object")?;
        if questions.is_empty() {
            return Err("H2O questions must not be empty".into());
        }
        let mut out = Vec::new();
        for (id, q) in questions {
            if q.as_object().is_none() {
                return Err(format!("H2O question {id:?} must be an object"));
            }
            let typ = q.get("type").map(|v| v.as_str().unwrap_or("")).unwrap_or("choice");
            let criteria = q.get("criteria");
            let (kind, keys, options) = match typ {
                "noul" => {
                    let c = criteria.filter(|v| py_truthy(v));
                    if c.is_some_and(|v| v.as_object().is_none()) {
                        return Err(format!("H2O question {id:?}: noul criteria must be an object"));
                    }
                    let desc = |key, fallback, default: &str| {
                        let d = c.and_then(|c| c.get(key).or_else(|| c.get(fallback)));
                        match d {
                            None | Some(Value::Null) => default.into(),
                            Some(Value::Str(s)) if s.is_empty() => default.into(),
                            Some(v) => desc_text(v),
                        }
                    };
                    (
                        Kind::Noul,
                        vec!["true".into(), "false".into()],
                        vec![desc("true", "yes", "the statement holds"), desc("false", "no", "it does not")],
                    )
                }
                "choice" => {
                    let c = criteria.and_then(Value::as_object).ok_or_else(|| {
                        format!("H2O question {id:?}: choice criteria must map option to description")
                    })?;
                    (
                        Kind::Choice,
                        c.iter().map(|(k, _)| k.clone()).collect(),
                        c.iter().map(|(k, v)| if v.is_null() { k.clone() } else { desc_text(v) }).collect(),
                    )
                }
                "score" => {
                    let c = criteria
                        .and_then(Value::as_array)
                        .ok_or_else(|| format!("H2O question {id:?}: score criteria must be an ordered array"))?;
                    (Kind::Score, (0..c.len()).map(|i| i.to_string()).collect(), c.iter().map(desc_text).collect())
                }
                _ => return Err(format!("H2O question {id:?}: unsupported type {typ:?}")),
            };
            if keys.is_empty() || keys.len() > self.labels.len() {
                return Err(format!("H2O question {id:?}: needs 1..{} options", self.labels.len()));
            }
            let instruction = q
                .get("instructions")
                .map(|v| match v {
                    Value::Str(s) => s.clone(),
                    _ => dumps(v, true, true),
                })
                .unwrap_or_default();
            let instruction = instruction.trim_start_matches(py_whitespace);
            let instructions =
                if instruction.is_empty() { self.default_instructions.clone() } else { instruction.into() };
            let threshold = if kind == Kind::Noul {
                q.get("threshold")
                    .or_else(|| q.get("noul_threshold"))
                    .and_then(|v| {
                        // Python bool is a number, so match the upstream threshold contract.
                        v.as_f64().or_else(|| v.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
                    })
                    .filter(|t| t.is_finite() && (0.0..=1.0).contains(t))
            } else {
                None
            };
            out.push(KevQuestion { id: id.clone(), kind, keys, instructions, options, threshold });
        }
        Ok(out)
    }

    pub fn encode(
        &self,
        tok: &Tokenizer,
        max_prompt: usize,
        state: &Value,
        qs: &[KevQuestion],
    ) -> Result<Encoded, String> {
        let mut text = self.state_text(state);
        let mx = self.max_state_tokens;
        if mx != 0 && text.len() > mx {
            let ids = tok.encode(&text);
            if ids.len() > mx {
                let half = mx / 2;
                text = format!(
                    "{}\n...\n{}",
                    tok.decode(&ids[..half])?,
                    tok.decode(&ids[if half == 0 { 0 } else { ids.len() - half }..])?
                );
            }
        }
        let mut branches = Vec::new();
        for q in qs {
            let ids = tok.encode(&self.prompt(&text, q));
            if ids.is_empty() || ids.len() > max_prompt {
                return Err(format!("H2O prompt has {} tokens; maximum is {max_prompt}", ids.len()));
            }
            branches.push(Branch { decide: ids.len() - 1, ids, opts: Vec::new(), labels: q.options.len() });
        }
        let mut shared = if branches.len() > 1 { branches[0].ids[..branches[0].decide].to_vec() } else { Vec::new() };
        for b in &branches {
            let common = shared.iter().zip(&b.ids).take_while(|(a, b)| a == b).count();
            shared.truncate(common.min(b.decide));
        }
        for b in &mut branches {
            b.ids.drain(..shared.len());
            b.decide -= shared.len();
        }
        Ok(Encoded { state: shared, branches })
    }

    pub fn answer(&self, q: &KevQuestion, logits: &[f32]) -> (Value, Vec<f64>) {
        let temperature = self.temperatures[match q.kind {
            Kind::Choice => 0,
            Kind::Noul => 1,
            Kind::Score => 2,
        }];
        let values: Vec<f64> = logits.iter().map(|&z| z as f64 / temperature.max(1e-3)).collect();
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut p: Vec<f64> = values.iter().map(|&v| (v - max).exp()).collect();
        let sum: f64 = p.iter().sum();
        for v in &mut p {
            *v /= sum;
        }
        let best = crate::math::argmax(&p);
        let n = p.len();
        let confidence = if n < 2 { 1.0 } else { (p[best] - 1.0 / n as f64) / (1.0 - 1.0 / n as f64) };
        let probabilities =
            || Value::Object(q.keys.iter().zip(&p).map(|(k, &v)| (k.clone(), Value::Float(v))).collect());
        let fields = match q.kind {
            Kind::Noul => {
                let py = commit_noul(p[0], self.noul_floor);
                let mut a = vec![("type".into(), Value::Str("noul".into())), ("noul".into(), Value::Float(py))];
                if let Some(threshold) = q.threshold {
                    a.push(("threshold".into(), Value::Float(threshold)));
                    a.push(("decision".into(), Value::Bool(py >= threshold)));
                }
                a
            }
            Kind::Choice => vec![
                ("type".into(), Value::Str("choice".into())),
                ("choice".into(), Value::Str(q.keys[best].clone())),
                ("probabilities".into(), probabilities()),
                ("confidence".into(), Value::Float(confidence)),
            ],
            Kind::Score => vec![
                ("type".into(), Value::Str("score".into())),
                ("score".into(), Value::Float(p.iter().enumerate().map(|(i, p)| i as f64 * p).sum())),
                (
                    "legend".into(),
                    Value::Object(
                        q.keys.iter().zip(&q.options).map(|(k, d)| (k.clone(), Value::Str(d.clone()))).collect(),
                    ),
                ),
                ("probabilities".into(), probabilities()),
                ("confidence".into(), Value::Float(confidence)),
            ],
        };
        (Value::Object(fields), p)
    }
}

pub fn commit_noul(py: f64, floor: f64) -> f64 {
    if floor == 0.0 || py.max(1.0 - py) >= floor {
        py
    } else if py >= 0.5 {
        floor
    } else {
        1.0 - floor
    }
}

fn desc_text(v: &Value) -> String {
    match v {
        Value::Str(s) => s.clone(),
        _ => v.py_dumps(false),
    }
}

fn py_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Int(s) => s != "0" && s != "-0",
        Value::Float(f) => *f != 0.0,
        Value::Str(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(m) => !m.is_empty(),
    }
}

fn py_whitespace(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

fn py_trim(s: &str) -> &str {
    s.trim_matches(py_whitespace)
}

/// Python JSON with compact separators, or indent=1. Python float spelling is preserved.
fn dumps(v: &Value, pretty: bool, ascii: bool) -> String {
    fn write(v: &Value, out: &mut String, depth: usize, pretty: bool, ascii: bool) {
        let sep = |out: &mut String, first: bool| {
            if !first {
                out.push(',');
            }
            if pretty {
                out.push('\n');
                out.push_str(&" ".repeat(depth + 1));
            }
        };
        let close = |out: &mut String, empty: bool| {
            if pretty && !empty {
                out.push('\n');
                out.push_str(&" ".repeat(depth));
            }
        };
        match v {
            Value::Array(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    sep(out, i == 0);
                    write(v, out, depth + 1, pretty, ascii);
                }
                close(out, a.is_empty());
                out.push(']');
            }
            Value::Object(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    sep(out, i == 0);
                    json::write_str(out, k, ascii);
                    out.push(':');
                    if pretty {
                        out.push(' ');
                    }
                    write(v, out, depth + 1, pretty, ascii);
                }
                close(out, m.is_empty());
                out.push('}');
            }
            _ => v.write_py(out, ascii),
        }
    }
    let mut out = String::new();
    write(v, &mut out, 0, pretty, ascii);
    out
}
