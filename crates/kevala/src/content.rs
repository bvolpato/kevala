//! What a request asks about, by modality.
//!
//! A request is `{"state": ..., "parts": [...], "questions": {...}}`. `state` is text or any JSON
//! value, exactly as System One APIs take it. `parts` is optional and carries typed content next
//! to it: `{"type": "text", "text": "..."}` today, `{"type": "image" | "audio", ...}` for model
//! families whose packs declare those modalities. A family lists the modalities it reads, and a
//! request with anything else is refused with an error that names the part, instead of being
//! silently dropped.

use crate::json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality {
    Text,
    Image,
    Audio,
}

impl Modality {
    pub fn name(self) -> &'static str {
        match self {
            Modality::Text => "text",
            Modality::Image => "image",
            Modality::Audio => "audio",
        }
    }

    pub fn parse(s: &str) -> Option<Modality> {
        match s {
            "text" => Some(Modality::Text),
            "image" => Some(Modality::Image),
            "audio" => Some(Modality::Audio),
            _ => None,
        }
    }

    /// The modalities a pack config declares (`"modalities": [...]`), text when absent.
    pub fn from_config(cfg: &Value) -> Vec<Modality> {
        match cfg.get("modalities").and_then(Value::as_array) {
            Some(a) => a.iter().filter_map(|v| v.as_str().and_then(Modality::parse)).collect(),
            None => vec![Modality::Text],
        }
    }
}

#[derive(Clone, Debug)]
pub struct Part {
    pub modality: Modality,
    /// The part as sent; a family's encoder for that modality reads what it needs from it.
    pub value: Value,
}

/// A parsed request.
#[derive(Clone, Debug)]
pub struct Request {
    pub state: Value,
    pub parts: Vec<Part>,
    pub questions: Value,
}

impl Request {
    pub fn parse(v: &Value) -> Result<Request, String> {
        let questions = v.get("questions").cloned().ok_or("request has no questions")?;
        let mut parts = Vec::new();
        if let Some(ps) = v.get("parts") {
            for (i, p) in ps.as_array().ok_or("parts must be an array")?.iter().enumerate() {
                let t = p.get("type").and_then(Value::as_str).ok_or_else(|| format!("part {i} has no type"))?;
                let modality = Modality::parse(t).ok_or_else(|| format!("part {i}: unknown type {t:?}"))?;
                parts.push(Part { modality, value: p.clone() });
            }
        }
        Ok(Request { state: v.get("state").cloned().unwrap_or(Value::Str(String::new())), parts, questions })
    }

    /// `{"state", "questions"}` or `{"requests": [...]}`.
    pub fn parse_many(v: &Value) -> Result<Vec<Request>, String> {
        match v.get("requests").and_then(Value::as_array) {
            Some(rs) => rs.iter().map(Request::parse).collect(),
            None => Ok(vec![Request::parse(v)?]),
        }
    }

    /// Fails unless the model reads every modality in the request.
    pub fn check(&self, arch: &str, supported: &[Modality]) -> Result<(), String> {
        for (i, p) in self.parts.iter().enumerate() {
            if !supported.contains(&p.modality) {
                let reads: Vec<&str> = supported.iter().map(|m| m.name()).collect();
                return Err(format!(
                    "part {i} is {}, but this {arch} pack reads {}; use a pack whose config lists {:?} in its modalities",
                    p.modality.name(),
                    reads.join(" and "),
                    p.modality.name()
                ));
            }
        }
        Ok(())
    }

    /// The state text a template tokenizes: `render(state)` followed by any text parts, a blank
    /// line apart. `None` when there are no text parts, so the family's reference path (which
    /// may render the state differently from plain text) stays byte-identical.
    pub fn text_with(&self, render: impl Fn(&Value) -> String) -> Option<String> {
        let texts: Vec<&str> = self
            .parts
            .iter()
            .filter(|p| p.modality == Modality::Text)
            .filter_map(|p| p.value.get("text").and_then(Value::as_str))
            .collect();
        if texts.is_empty() {
            return None;
        }
        let mut s = render(&self.state);
        for t in texts {
            if !s.is_empty() {
                s.push_str("\n\n");
            }
            s.push_str(t);
        }
        Some(s)
    }
}
