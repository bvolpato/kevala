//! Additive staged projection ABI. No caller input buffer is retained or changed.
use super::{slice, st, State};
use kevala::clef::head::staged::{integer, Phase, StagedHead};
use kevala::clef::ClefEngine;
use kevala::json::Value;
use kevala::runtime::Model;

#[derive(Default)]
struct Ids {
    last: u32,
    active: Option<u32>,
}
impl Ids {
    fn begin(&mut self) -> Result<u32, &'static str> {
        if self.active.is_some() {
            return Err("stage-already-active");
        }
        self.last = self.last.checked_add(1).ok_or("stage-id-exhausted")?;
        self.active = Some(self.last);
        Ok(self.last)
    }
    fn matches(&self, id: u32) -> bool {
        self.active == Some(id)
    }
    fn invalidate(&mut self) {
        self.active = None;
    }
}
#[derive(Default)]
pub(super) struct Staging {
    ids: Ids,
    stage: Option<StagedHead>,
}
impl Staging {
    pub(super) fn invalidate(&mut self) {
        self.ids.invalidate();
        self.stage = None;
    }
}
fn call(f: impl FnOnce(&mut State) -> Result<Value, String>) -> u32 {
    let state = st();
    match f(state) {
        Ok(value) => {
            state.out = value.to_json().into_bytes();
            0
        }
        Err(error) => {
            state.err = error;
            1
        }
    }
}
fn engine(model: &mut Option<Box<dyn Model>>) -> Result<&mut ClefEngine, String> {
    model
        .as_mut()
        .ok_or("no model loaded")?
        .as_any()
        .downcast_mut::<ClefEngine>()
        .ok_or_else(|| "the loaded model is not a clef model".into())
}
fn status(status: &str) -> Value {
    Value::Object(vec![("version".into(), integer(1)), ("status".into(), Value::Str(status.into()))])
}
fn identified(mut value: Value, id: u32) -> Value {
    if let Value::Object(fields) = &mut value {
        fields.push(("id".into(), integer(id as usize)));
    }
    value
}
fn stale(id: u32) -> Value {
    identified(status("stale"), id)
}
fn fallback(staging: &mut Staging, reason: &str) -> Value {
    staging.invalidate();
    let mut value = status("fallback");
    if let Value::Object(fields) = &mut value {
        fields.push(("reason".into(), Value::Str(reason.into())));
    }
    value
}

/// Read-only model-resident weights. Valid until any attempted model load.
#[no_mangle]
pub extern "C" fn kevala_clef_head_layout() -> u32 {
    call(|state| Ok(engine(&mut state.model)?.head.staged_layout()))
}

/// Check pointer lifetime after an asynchronous host operation. This changes only output JSON.
/// Reacquire memory views after this call, then write and advance the phase without awaiting.
#[no_mangle]
pub extern "C" fn kevala_clef_head_status(id: u32) -> u32 {
    call(|state| {
        let staging = &state.clef_head;
        let phase = staging.stage.as_ref().filter(|_| staging.ids.matches(id)).map(StagedHead::phase);
        Ok(match phase {
            Some(Phase::Memory | Phase::Kv) => {
                let mut value = identified(status("ready"), id);
                if let Value::Object(fields) = &mut value {
                    fields.push((
                        "phase".into(),
                        Value::Str(if phase == Some(Phase::Memory) { "memory" } else { "kv" }.into()),
                    ));
                }
                value
            }
            _ => stale(id),
        })
    })
}

/// Own private copies, applying final RMSNorm then Head LayerNorm in their original order.
/// Successful ABI calls write either a ready plan or an explicit CPU-fallback reason.
/// Only `memory` is writable. All returned pointers expire on cancel, prepare, load or finish.
///
/// # Safety
/// Inputs must be valid f32 slices for their counts. No pointer is retained after this call.
#[no_mangle]
pub unsafe extern "C" fn kevala_clef_head_begin(
    hidden: *const f32,
    hidden_n: usize,
    lexical: *const f32,
    lexical_n: usize,
    validate: u32,
) -> u32 {
    call(|state| {
        if validate > 1 {
            return Err("clef staged head: validate must be 0 or 1".into());
        }
        if state.clef_head.ids.active.is_some() {
            return Err("clef staged head: stage already active".into());
        }
        let records = state.clef_records.as_ref().ok_or("nothing prepared")?;
        let engine = engine(&mut state.model)?;
        let tokens = records.iter().try_fold(0usize, |n, r| n.checked_add(r.ids.len()));
        let options =
            records.iter().flat_map(|r| &r.questions).try_fold(0usize, |n, q| n.checked_add(q.option_spans.len()));
        if tokens.and_then(|n| n.checked_mul(engine.head.cfg.hidden)) != Some(hidden_n)
            || options.and_then(|n| n.checked_mul(engine.head.cfg.hidden)) != Some(lexical_n)
        {
            return Ok(fallback(&mut state.clef_head, "hidden-or-lexical-shape"));
        }
        match engine.begin_staged_head(records, slice(hidden, hidden_n), slice(lexical, lexical_n), validate == 1) {
            Ok(stage) => match state.clef_head.ids.begin() {
                Ok(id) => {
                    let plan = identified(stage.begin_descriptor(), id);
                    state.clef_head.stage = Some(stage);
                    Ok(plan)
                }
                Err(reason) => Ok(fallback(&mut state.clef_head, reason)),
            },
            Err(reason) => Ok(fallback(&mut state.clef_head, reason)),
        }
    })
}

/// After the host fills M, return routing normalized-M inputs and decoder M aliases.
/// Only projection `output` buffers are writable; references and inputs are read-only.
#[no_mangle]
pub extern "C" fn kevala_clef_head_memory_ready(id: u32) -> u32 {
    call(|state| {
        if !state.clef_head.ids.matches(id) {
            return Ok(stale(id));
        }
        let stage = state.clef_head.stage.as_mut().ok_or("clef staged head: missing active stage")?;
        if stage.phase() != Phase::Memory {
            return Err("clef staged head: expected memory phase".into());
        }
        match stage.memory_ready(&engine(&mut state.model)?.head) {
            Ok(plan) => Ok(identified(plan, id)),
            Err(reason) => Ok(fallback(&mut state.clef_head, reason)),
        }
    })
}

/// Consume supplied KV outputs and run the unchanged attention/readout graph on the CPU.
/// With validate=1, every supplied M/KV output bit must match its CPU reference first.
#[no_mangle]
pub extern "C" fn kevala_clef_head_finish(id: u32) -> u32 {
    call(|state| {
        if !state.clef_head.ids.matches(id) {
            return Ok(stale(id));
        }
        let stage = state.clef_head.stage.as_mut().ok_or("clef staged head: missing active stage")?;
        if stage.phase() != Phase::Kv {
            return Err("clef staged head: expected KV phase".into());
        }
        let records = state.clef_records.as_ref().ok_or("nothing prepared")?;
        let engine = engine(&mut state.model)?;
        let result = stage.finish(&mut engine.head, records).and_then(|logits| {
            records
                .iter()
                .zip(&logits)
                .map(|(record, logits)| engine.respond(record, logits))
                .collect::<Result<Vec<_>, _>>()
        });
        match result {
            Ok(responses) => {
                state.clef_records = None;
                state.clef_head.invalidate();
                Ok(Value::Array(responses))
            }
            Err(reason) => Ok(fallback(&mut state.clef_head, &reason)),
        }
    })
}

/// Drop only this stage. Prepared records stay available to the ordinary CPU finish.
/// A stale id never drops a newer stage or changes its prepared records.
#[no_mangle]
pub extern "C" fn kevala_clef_head_cancel(id: u32) -> u32 {
    call(|state| {
        Ok(if state.clef_head.ids.matches(id) {
            state.clef_head.invalidate();
            identified(status("canceled"), id)
        } else {
            stale(id)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::Ids;
    #[test]
    fn stale_ids_cannot_advance_or_cancel_a_newer_stage() {
        let mut ids = Ids::default();
        let first = ids.begin().unwrap();
        assert!(ids.begin().is_err());
        ids.invalidate(); // prepare/load/ordinary finish discard the old stage.
        let second = ids.begin().unwrap();
        assert!(second > first);
        assert!(!ids.matches(first));
        assert!(ids.matches(second));
        if ids.matches(first) {
            ids.invalidate();
        }
        assert!(ids.matches(second));
        ids.invalidate();
        assert!(!ids.matches(second));
    }
    #[test]
    fn stage_ids_never_wrap_or_reuse() {
        let mut ids = Ids { last: u32::MAX, active: None };
        assert_eq!(ids.begin(), Err("stage-id-exhausted"));
        assert_eq!(ids.active, None);
    }
}
