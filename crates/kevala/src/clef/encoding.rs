//! Text record encoding from Cloudflare's `joint_schema_model.encode_record`.

use crate::json::Value;
use crate::tokenizer::Tokenizer;

pub const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.";

#[derive(Clone, Debug)]
pub struct Question {
    pub id: String,
    pub kind: usize,
    pub span: (usize, usize),
    pub option_spans: Vec<(usize, usize)>,
    pub option_ids: Vec<String>,
    /// Choice response ordering follows the request, while model options are sorted.
    pub response_ids: Vec<String>,
    pub criteria: Value,
}

#[derive(Clone, Debug)]
pub struct Record {
    pub ids: Vec<u32>,
    pub questions: Vec<Question>,
}

fn sorted(v: &Value) -> Value {
    match v {
        Value::Object(items) => {
            let mut items: Vec<_> = items.iter().map(|(k, v)| (k.clone(), sorted(v))).collect();
            items.sort_by(|a, b| a.0.cmp(&b.0));
            Value::Object(items)
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        _ => v.clone(),
    }
}

/// Python `json.dumps(ensure_ascii=False,separators=(",",":"),sort_keys=True)`.
/// Remove whitespace outside strings from the Python serializer, preserving Python float spelling.
pub fn render(v: &Value) -> String {
    if let Value::Str(s) = v {
        return s.clone();
    }
    let json = sorted(v).py_dumps(false);
    let mut out = String::with_capacity(json.len());
    let (mut quoted, mut escaped) = (false, false);
    for c in json.chars() {
        if quoted {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                quoted = false;
            }
        } else if c == '"' {
            quoted = true;
            out.push(c);
        } else if !c.is_ascii_whitespace() {
            out.push(c);
        }
    }
    out
}

pub fn encode(tok: &Tokenizer, state: &Value, qs: &Value, maximum: usize, max_state: usize) -> Result<Record, String> {
    let entries = qs.as_object().filter(|q| !q.is_empty()).ok_or("clef: at least one question is required")?;
    let mut schema = tok.encode("\n\nSCHEMA FIELDS:\n");
    let mut questions = Vec::new();
    for (index, (id, q)) in entries.iter().enumerate() {
        let kind = match q.get("type").and_then(Value::as_str) {
            Some("noul") => 0,
            Some("choice") => 1,
            Some("score") => 2,
            _ => return Err(format!("{id}: type must be noul, choice, or score")),
        };
        let ty = ["noul", "choice", "score"][kind];
        schema.extend(tok.encode(&format!("\nFIELD {}\nID: {id}\nTYPE: {ty}\nINSTRUCTION: ", index + 1)));
        let start = schema.len();
        let default = Value::Str(id.clone());
        let instructions = q.get("instructions").filter(|v| !v.is_null() && v.as_str() != Some("")).unwrap_or(&default);
        schema.extend(tok.encode(&render(instructions)));
        let span = (start, schema.len());
        if span.0 == span.1 {
            return Err(format!("{id}: instruction produced no tokens"));
        }
        schema.extend(tok.encode("\nALLOWED OPTIONS:\n"));
        let criteria = q.get("criteria").cloned().unwrap_or(Value::Null);
        let (options, response_ids): (Vec<(String, Value)>, Vec<String>) = match kind {
            0 => {
                if !criteria.is_null() && criteria.as_object().is_none() {
                    return Err(format!("{id}: noul criteria must be an object"));
                }
                let mut options = Vec::new();
                for (key, description) in [
                    ("true", "The proposition is true or the answer is yes."),
                    ("false", "The proposition is false or the answer is no."),
                ] {
                    options.push((
                        key.into(),
                        criteria.get(key).cloned().unwrap_or_else(|| Value::Str(description.into())),
                    ));
                }
                (options, vec!["true".into(), "false".into()])
            }
            1 => {
                let items = criteria
                    .as_object()
                    .filter(|v| !v.is_empty())
                    .ok_or_else(|| format!("{id}: choice criteria must be a nonempty object"))?;
                let response_ids = items.iter().map(|(k, _)| k.clone()).collect();
                let mut items = items.to_vec();
                items.sort_by(|a, b| a.0.cmp(&b.0));
                (items, response_ids)
            }
            _ => {
                let items = criteria
                    .as_array()
                    .filter(|v| !v.is_empty())
                    .ok_or_else(|| format!("{id}: score criteria must be a nonempty list"))?;
                let options: Vec<_> = items.iter().enumerate().map(|(i, v)| (i.to_string(), v.clone())).collect();
                let keys = options.iter().map(|(k, _)| k.clone()).collect();
                (options, keys)
            }
        };
        let mut option_ids = Vec::new();
        let mut option_spans = Vec::new();
        for (oi, (key, description)) in options.into_iter().enumerate() {
            schema.extend(tok.encode(&format!("OPTION {}: ", oi + 1)));
            let start = schema.len();
            let mut semantics = vec![("option_id".into(), Value::Str(key.clone()))];
            if !description.is_null() {
                semantics.push(("description".into(), description));
            }
            schema.extend(tok.encode(&render(&Value::Object(semantics))));
            option_spans.push((start, schema.len()));
            option_ids.push(key);
            schema.extend(tok.encode("\n"));
        }
        schema.extend(tok.encode("END FIELD\n"));
        questions.push(Question { id: id.clone(), kind, span, option_spans, option_ids, response_ids, criteria });
    }
    let mut prefix = tok.encode(&format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"));
    let suffix = tok.encode("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:");
    let fixed = prefix.len() + schema.len() + suffix.len();
    if fixed > maximum {
        return Err(format!("schema requires {fixed} tokens before state; maximum is {maximum}"));
    }
    let state = tok.encode_prefix(&render(state), (maximum - fixed).min(max_state));
    let offset = prefix.len() + state.len();
    for q in &mut questions {
        q.span.0 += offset;
        q.span.1 += offset;
        for span in &mut q.option_spans {
            span.0 += offset;
            span.1 += offset;
        }
    }
    prefix.extend(state);
    prefix.extend(schema);
    prefix.extend(suffix);
    Ok(Record { ids: prefix, questions })
}
