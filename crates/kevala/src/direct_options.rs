//! The direct-options prompt (`direct-options-v1`) that SemIf and Gemma 4 share.
//!
//! A frozen instruction model reads the evidence, one criterion, and up to sixteen lettered
//! options, and the readout takes the logits of the answer letters. Each family wraps the same
//! system text and JSON payload in its own chat template.

use crate::json::Value;
use crate::kev::KevQuestion;

/// One answer letter per option slot. Packs store the readout rows in this order.
pub const LABELS: &str = "ABCDEFGHIJKLMNOP";

pub const SYSTEM: &str = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.";

/// The user turn: `{"evidence", "criterion", "options": [{"letter", "description"}]}`.
pub fn payload(state: &Value, q: &KevQuestion) -> Value {
    let options = q
        .options
        .iter()
        .zip(LABELS.chars())
        .map(|(option, letter)| {
            Value::Object(vec![
                ("letter".into(), Value::Str(letter.to_string())),
                ("description".into(), Value::Str(option.clone())),
            ])
        })
        .collect();
    Value::Object(vec![
        ("evidence".into(), state.clone()),
        ("criterion".into(), Value::Str(q.instructions.clone())),
        ("options".into(), Value::Array(options)),
    ])
}
