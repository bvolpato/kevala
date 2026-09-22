//! Question parsing and token sequence construction, ported from the Laya SDK 0.3.5
//! (`laya/common.py`: `serialize_state`, `render_criterion`, `render_options`, `build_sequence`;
//! `laya/agent.py`: `Agent._to_internal`).
//!
//! A sequence is `[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]
//! state [SEP]`; the model reads each option's answer off its `[MASK]`, whose positions are the
//! markers. Every slice bound below is the SDK's, including its Python slicing quirks.

use std::borrow::Cow;

use crate::json::{write_py_float, Value};
use crate::tokenizer::Tokenizer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QType {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

impl QType {
    pub fn name(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Question {
    pub id: String,
    pub qtype: QType,
    /// The instructions as `_to_internal` leaves them: non-strings become `json.dumps(x)`.
    pub instructions: String,
    /// Answer keys in option order: the choice criteria keys, "0".."k-1" for score, and
    /// ["false", "true"] for noul.
    pub keys: Vec<String>,
    /// Score only: each level's criterion rendered as text.
    pub legend: Vec<String>,
    /// `render_options`: the option texts, in marker order.
    pub options: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sequence {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
}

/// `serialize_state`: strings verbatim, anything else as `json.dumps(state, ensure_ascii=False)`.
pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::Str(s) => s.clone(),
        v => v.py_dumps(false),
    }
}

/// `render_criterion`: strings pass through, anything else is `json.dumps(ensure_ascii=False)`
/// (its `default=str` never fires on parsed JSON).
fn render_criterion(v: &Value) -> String {
    match v {
        Value::Str(s) => s.clone(),
        v => v.py_dumps(false),
    }
}

/// Python truthiness, which the SDK's `crit or {}` applies to noul criteria.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Int(s) => s.bytes().any(|b| b.is_ascii_digit() && b != b'0'),
        Value::Float(f) => *f != 0.0,
        Value::Str(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(m) => !m.is_empty(),
    }
}

/// No description: the SDK only treats None and "" as missing, so 0 and false still render.
fn is_blank(v: Option<&Value>) -> bool {
    matches!(v, None | Some(Value::Null)) || v.and_then(Value::as_str) == Some("")
}

/// Python `str(x)` for a scalar list item. The SDK itself crashes on non-string choice keys
/// (`int.replace`), so this is where the port chooses to accept them.
fn py_str(v: &Value) -> Option<String> {
    Some(match v {
        Value::Str(s) => s.clone(),
        Value::Int(s) => s.clone(),
        Value::Bool(b) => (if *b { "True" } else { "False" }).to_string(),
        Value::Null => "None".to_string(),
        Value::Float(f) if f.is_nan() => "nan".to_string(),
        Value::Float(f) if f.is_infinite() => (if *f > 0.0 { "inf" } else { "-inf" }).to_string(),
        Value::Float(f) => {
            let mut s = String::new();
            write_py_float(&mut s, *f);
            s
        }
        Value::Array(_) | Value::Object(_) => return None,
    })
}

/// Parses a request's questions (`{id: {"type", "instructions", "criteria"?}}`), in order.
pub fn parse_questions(questions: &Value) -> Result<Vec<Question>, String> {
    let m = questions.as_object().ok_or("questions must be an object")?;
    if m.is_empty() {
        return Err("questions must not be empty".into());
    }
    m.iter().map(|(id, def)| parse_question(id, def)).collect()
}

fn parse_question(id: &str, def: &Value) -> Result<Question, String> {
    if def.as_object().is_none() {
        return Err(format!("question {id:?} must be an object"));
    }
    let qtype = match def.get("type") {
        Some(Value::Str(t)) if t == "choice" => QType::Choice,
        Some(Value::Str(t)) if t == "score" => QType::Score,
        Some(Value::Str(t)) if t == "noul" => QType::Noul,
        Some(t) => return Err(format!("question {id:?} has unknown type {}", t.to_json())),
        None => return Err(format!("question {id:?} has no type")),
    };
    let instructions = match def.get("instructions") {
        Some(Value::Str(s)) => s.clone(),
        Some(v) => v.py_dumps(true),
        None => return Err(format!("question {id:?} has no instructions")),
    };
    let crit = def.get("criteria").filter(|v| !v.is_null());
    let (mut keys, mut legend, mut options) = (Vec::new(), Vec::new(), Vec::new());
    match qtype {
        QType::Choice => match crit {
            Some(Value::Object(m)) => {
                for (k, v) in m {
                    options.push(if is_blank(Some(v)) { k.clone() } else { format!("{k}: {}", render_criterion(v)) });
                    keys.push(k.clone());
                }
            }
            // `{c: None for c in crit}`: a repeated item keeps its first position
            Some(Value::Array(items)) => {
                for item in items {
                    let k = py_str(item)
                        .ok_or_else(|| format!("question {id:?}: choice criteria items must be strings or numbers"))?;
                    if !keys.contains(&k) {
                        options.push(k.clone());
                        keys.push(k);
                    }
                }
            }
            Some(_) => return Err(format!("question {id:?}: choice criteria must be an object or a list")),
            None => return Err(format!("question {id:?}: choice needs criteria")),
        },
        QType::Score => {
            // `enumerate(crit)` also walks a dict's keys and a string's characters
            let levels: Vec<Value> = match crit {
                Some(Value::Array(a)) => a.to_vec(),
                Some(Value::Object(m)) => m.iter().map(|(k, _)| Value::Str(k.clone())).collect(),
                Some(Value::Str(s)) => s.chars().map(|c| Value::Str(c.to_string())).collect(),
                Some(_) => return Err(format!("question {id:?}: score criteria must be a list")),
                None => return Err(format!("question {id:?}: score needs criteria")),
            };
            for (i, c) in levels.iter().enumerate() {
                let text = render_criterion(c);
                options.push(format!("level {i}: {text}"));
                keys.push(i.to_string());
                legend.push(text);
            }
        }
        QType::Noul => {
            let m: &[(String, Value)] = match crit {
                Some(c) if truthy(c) => {
                    c.as_object().ok_or_else(|| format!("question {id:?}: noul criteria must be an object"))?
                }
                _ => &[],
            };
            let get = |k: &str| m.iter().find(|(kk, _)| kk == k).map(|(_, v)| v);
            for (k, default) in [("false", "no, the statement does not hold"), ("true", "yes, the statement holds")] {
                let v = get(k);
                let text = if is_blank(v) { default.to_string() } else { render_criterion(v.unwrap()) };
                options.push(format!("{k}: {text}"));
                keys.push(k.to_string());
            }
        }
    }
    if options.is_empty() {
        return Err(format!("question {id:?} has no options"));
    }
    Ok(Question { id: id.to_string(), qtype, instructions, keys, legend, options })
}

fn scrub<'a>(s: &'a str, mask: &str) -> Cow<'a, str> {
    if s.contains(mask) {
        Cow::Owned(s.replace(mask, " "))
    } else {
        Cow::Borrowed(s)
    }
}

/// `build_sequence(tok, state, q, max_len, head_max_len, truncate_left=...)` with
/// `state_text = serialize_state(state)`. The SDK rejects a question whose markers did not all
/// fit (`markers.len() != options.len()`); that check is the caller's.
pub fn build_sequence(
    tok: &Tokenizer,
    state_text: &str,
    q: &Question,
    max_len: usize,
    head_max_len: usize,
    truncate_left: bool,
) -> Sequence {
    let mask = tok.mask_token();
    let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(q.options.len());
    let mut text = String::new();
    for o in &q.options {
        text.clear();
        text.push(' ');
        text.push_str(&scrub(o, mask));
        let mut ids = Vec::with_capacity(49);
        ids.push(tok.mask_id());
        ids.extend(tok.encode_prefix(&text, 48));
        opt_ids.push(ids);
    }
    let hml = head_max_len as i64;
    let total = |o: &[Vec<u32>]| o.iter().map(Vec::len).sum::<usize>() as i64;
    let mut opt_budget = hml - total(&opt_ids);
    if opt_budget < 16 {
        // Python floor division
        let per = (hml - 16).div_euclid(opt_ids.len().max(1) as i64).max(4) as usize;
        for o in &mut opt_ids {
            o.truncate(per);
        }
        opt_budget = hml - total(&opt_ids);
    }
    let head = format!("{} question: {}", q.qtype.name(), scrub(&q.instructions, mask));
    let head_ids = tok.encode_prefix(&head, opt_budget.max(8) as usize);

    let mut ids = Vec::with_capacity(max_len + 1);
    ids.push(tok.cls_id());
    ids.extend_from_slice(&head_ids);
    ids.push(tok.sep_id());
    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in &opt_ids {
        markers.push(ids.len());
        ids.extend_from_slice(o);
    }
    ids.push(tok.sep_id());
    let room = max_len.saturating_sub(ids.len() + 1);
    let state = scrub(state_text, mask);
    if truncate_left {
        let st = tok.encode(&state);
        // `st[-room:]` is the whole list when room is 0
        let from = if room == 0 { 0 } else { st.len().saturating_sub(room) };
        ids.extend_from_slice(&st[from..]);
    } else {
        ids.extend(tok.encode_prefix(&state, room));
    }
    ids.push(tok.sep_id());
    ids.truncate(max_len);
    markers.retain(|&m| m < max_len);
    Sequence { ids, markers }
}
