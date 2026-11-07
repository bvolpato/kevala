//! Guarded, owned inputs and writable projection targets for the additive GPU head ABI.
use super::{Head, HeadConfig, Record, Value};
use crate::kernels::{linear, tile, Mat};
use crate::store::Store;
use std::ops::Range;
use std::sync::Arc;

const MAX_BYTES: usize = 256 * 1024 * 1024;
const MAX_RECORDS: usize = 1024;
const OPERAND_MIN: u32 = 90 << 23;
const BIAS_MIN: u32 = 30 << 23;
const MAX_MAGNITUDE: u32 = 148 << 23;

pub fn eligible(values: &[f32], bias: bool) -> bool {
    // Operand products and bias stay on the 2^-120 lattice. K<=4096 and the upper
    // bound exclude subnormal intermediates and overflow, not WGSL reassociation.
    let minimum = if bias { BIAS_MIN } else { OPERAND_MIN };
    values.iter().all(|value| {
        let bits = value.to_bits() & 0x7fff_ffff;
        bits == 0 || (minimum..MAX_MAGNITUDE).contains(&bits)
    })
}
fn memory_output_range(values: &[f32]) -> Result<(), &'static str> {
    let mut in_range = true;
    for value in values {
        let bits = value.to_bits() & 0x7fff_ffff;
        if bits >= 0x7f80_0000 {
            return Err("memory-output-nonfinite");
        }
        in_range &= bits == 0 || (OPERAND_MIN..MAX_MAGNITUDE).contains(&bits);
    }
    if in_range {
        Ok(())
    } else {
        Err("memory-input-range")
    }
}
pub fn integer(value: usize) -> Value {
    Value::Int(value.to_string())
}
pub fn descriptor(values: &[f32]) -> Value {
    Value::Object(vec![("ptr".into(), integer(values.as_ptr() as usize)), ("length".into(), integer(values.len()))])
}
fn allocate<T: Clone>(len: usize, value: T) -> Result<Vec<T>, &'static str> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| "allocation-failed")?;
    out.resize(len, value);
    Ok(out)
}
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
#[derive(Clone, Copy)]
struct BindingSize {
    records: usize,
    questions: usize,
    options: usize,
}
#[derive(Clone)]
struct QuestionBinding {
    // Only metadata read by Head math. Response-only IDs, criteria and labels are not retained.
    kind: usize,
    span: (usize, usize),
    options: Range<usize>,
}
fn scratch_bytes(
    sequence: usize,
    memory: usize,
    lexical: usize,
    routing: usize,
    count: usize,
    binding: BindingSize,
    validate: bool,
) -> Option<usize> {
    let matrices = 1usize.checked_add(routing)?.checked_add(count.checked_mul(2)?)?.checked_add(if validate {
        count.checked_mul(2)?.checked_add(1)?
    } else {
        0
    })?;
    let floats = memory.checked_mul(matrices)?.checked_add(sequence)?.checked_add(lexical)?;
    let metadata = binding
        .records
        .checked_mul(
            2 * std::mem::size_of::<usize>()
                + std::mem::size_of::<Range<usize>>()
                + std::mem::size_of::<Vec<Vec<f32>>>(),
        )?
        .checked_add(binding.questions.checked_mul(std::mem::size_of::<QuestionBinding>())?)?
        .checked_add(binding.options.checked_mul(std::mem::size_of::<(usize, usize)>())?)?
        .checked_add(routing.checked_add(count.checked_mul(2)?)?.checked_mul(std::mem::size_of::<Vec<f32>>())?)?
        .checked_add(count.checked_mul(std::mem::size_of::<&[f32]>())?)?;
    floats.checked_mul(4)?.checked_add(metadata)
}

impl Head {
    fn staged_geometry(&self) -> bool {
        let c = &self.cfg;
        c == &self.staged_config
            && (4..=4096).contains(&c.hidden)
            && (4..=4096).contains(&c.width)
            && c.hidden % 4 == 0
            && c.width % 4 == 0
            && c.routing.checked_add(c.layers).is_some_and(|n| n <= 32)
    }
    fn staged_name(&self, index: usize) -> String {
        if index == 0 {
            "memory_projection".into()
        } else if index <= self.cfg.routing {
            format!("evidence_layers.{}.attention", index - 1)
        } else {
            format!("layers.{}.multihead_attn", index - 1 - self.cfg.routing)
        }
    }
    fn staged_weights(&self, index: usize) -> (&[f32], Option<&[f32]>, usize, usize) {
        if index == 0 {
            (self.values("memory_projection.weight"), None, self.cfg.width, self.cfg.hidden)
        } else {
            let name = self.staged_name(index);
            let w = self.cfg.width;
            (
                &self.values(&format!("{name}.in_proj_weight"))[w * w..3 * w * w],
                Some(&self.values(&format!("{name}.in_proj_bias"))[w..3 * w]),
                2 * w,
                w,
            )
        }
    }
    pub fn staged_static_eligible(&self) -> bool {
        if self.cfg != self.staged_config {
            return false;
        }
        *self.staged_eligible.get_or_init(|| {
            self.staged_geometry()
                && (0..=self.cfg.routing + self.cfg.layers).all(|index| {
                    let (weights, bias, _, _) = self.staged_weights(index);
                    eligible(weights, false) && bias.is_none_or(|bias| eligible(bias, true))
                })
        })
    }
    pub fn staged_layout(&self) -> Value {
        let supported = self.staged_geometry();
        let projections = if supported {
            (0..=self.cfg.routing + self.cfg.layers)
                .map(|index| {
                    let (weights, bias, n, k) = self.staged_weights(index);
                    Value::Object(vec![
                        ("index".into(), integer(index)),
                        (
                            "kind".into(),
                            Value::Str(
                                if index == 0 {
                                    "memory"
                                } else if index <= self.cfg.routing {
                                    "routing"
                                } else {
                                    "decoder"
                                }
                                .into(),
                            ),
                        ),
                        ("name".into(), Value::Str(self.staged_name(index))),
                        ("n".into(), integer(n)),
                        ("k".into(), integer(k)),
                        ("weights".into(), descriptor(weights)),
                        ("bias".into(), bias.map_or(Value::Null, descriptor)),
                    ])
                })
                .collect()
        } else {
            Vec::new()
        };
        let eligible = self.staged_static_eligible();
        Value::Object(vec![
            ("version".into(), integer(1)),
            ("status".into(), Value::Str(if eligible { "ready" } else { "fallback" }.into())),
            ("hidden".into(), integer(self.cfg.hidden)),
            ("width".into(), integer(self.cfg.width)),
            ("staticEligible".into(), Value::Bool(eligible)),
            (
                "reason".into(),
                if eligible { Value::Null } else { Value::Str("static-geometry-or-weight-range".into()) },
            ),
            ("projections".into(), Value::Array(projections)),
        ])
    }
    fn staged_norm(&self, x: &mut [f32], name: &str, width: usize) {
        let weight = self.values(&format!("{name}.weight"));
        let bias = self.values(&format!("{name}.bias"));
        let complete_len = x.len() / width * width;
        for row in x[..complete_len].chunks_exact_mut(width) {
            let mean = row.iter().sum::<f32>() / width as f32;
            let variance = row.iter().map(|z| (z - mean) * (z - mean)).sum::<f32>() / width as f32;
            let scale = (variance + 1e-5).sqrt().recip();
            for i in 0..width {
                row[i] = (row[i] - mean) * scale * weight[i] + bias[i];
            }
        }
        x[complete_len..].fill(0.0);
    }
    fn staged_linear(&self, index: usize, input: &[f32], output: &mut [f32], segments: &[usize]) {
        let (weights, bias, n, k) = self.staged_weights(index);
        let mut offset = 0;
        for &t in segments {
            linear(
                &input[offset * k..(offset + t) * k],
                t,
                Mat::F32 { n, k, w: weights },
                bias,
                &mut output[offset * n..(offset + t) * n],
                &mut Vec::new(),
            );
            offset += t;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Memory,
    Kv,
    Finished,
}

pub struct StagedHead {
    store: Arc<Store>,
    config: HeadConfig,
    phase: Phase,
    tile: u8,
    validate: bool,
    tokens: usize,
    segments: Vec<usize>,
    options: Vec<usize>,
    record_questions: Vec<Range<usize>>,
    questions: Vec<QuestionBinding>,
    option_spans: Vec<(usize, usize)>,
    sequence: Vec<f32>,
    lexical: Vec<f32>,
    memory: Vec<f32>,
    normalized_memory: Vec<Vec<f32>>,
    kv: Vec<Vec<f32>>,
    reference_memory: Option<Vec<f32>>,
    reference_kv: Vec<Vec<f32>>,
}

impl StagedHead {
    pub fn begin(
        head: &Head,
        records: &[Record],
        hidden: &[f32],
        lexical: &[f32],
        validate: bool,
        normalize: impl FnOnce(&mut [f32]),
    ) -> Result<Self, &'static str> {
        if !head.staged_static_eligible() {
            return Err("static-geometry-or-weight-range");
        }
        if records.is_empty() || records.len() > MAX_RECORDS {
            return Err("record-count-outside-budget");
        }
        let mut segments = allocate(records.len(), 0usize)?;
        let mut options = allocate(records.len(), 0usize)?;
        let (mut tokens, mut option_count, mut question_count) = (0usize, 0usize, 0usize);
        for (i, record) in records.iter().enumerate() {
            if record.ids.is_empty() || record.questions.is_empty() {
                return Err("empty-record");
            }
            segments[i] = record.ids.len();
            options[i] = record
                .questions
                .iter()
                .try_fold(0usize, |n, q| n.checked_add(q.option_spans.len()))
                .ok_or("size-overflow")?;
            tokens = tokens.checked_add(segments[i]).ok_or("size-overflow")?;
            option_count = option_count.checked_add(options[i]).ok_or("size-overflow")?;
            question_count = question_count.checked_add(record.questions.len()).ok_or("size-overflow")?;
        }
        let (d, w, count) = (head.cfg.hidden, head.cfg.width, head.cfg.routing + head.cfg.layers);
        let sequence_len = tokens.checked_mul(d).ok_or("size-overflow")?;
        let memory_len = tokens.checked_mul(w).ok_or("size-overflow")?;
        let lexical_len = option_count.checked_mul(d).ok_or("size-overflow")?;
        if sequence_len != hidden.len() || lexical_len != lexical.len() {
            return Err("hidden-or-lexical-shape");
        }
        // All float scratch is reserved before normalization or exposing a writable target.
        let binding = BindingSize { records: records.len(), questions: question_count, options: option_count };
        if scratch_bytes(sequence_len, memory_len, lexical_len, head.cfg.routing, count, binding, validate)
            .is_none_or(|n| n > MAX_BYTES)
        {
            return Err("scratch-budget");
        }
        let mut stage = Self {
            store: Arc::clone(&head.store),
            config: head.cfg.clone(),
            phase: Phase::Memory,
            tile: tile(),
            validate,
            tokens,
            segments,
            options,
            record_questions: allocate(records.len(), 0..0)?,
            questions: allocate(question_count, QuestionBinding { kind: 0, span: (0, 0), options: 0..0 })?,
            option_spans: allocate(option_count, (0, 0))?,
            sequence: allocate(sequence_len, 0.)?,
            lexical: allocate(lexical_len, 0.)?,
            memory: allocate(memory_len, f32::NAN)?,
            normalized_memory: allocate(head.cfg.routing, Vec::new())?,
            kv: allocate(count, Vec::new())?,
            reference_memory: if validate { Some(allocate(memory_len, 0.)?) } else { None },
            reference_kv: allocate(if validate { count } else { 0 }, Vec::new())?,
        };
        let (mut question, mut option) = (0, 0);
        for (index, record) in records.iter().enumerate() {
            stage.record_questions[index] = question..question + record.questions.len();
            for q in &record.questions {
                let end = option + q.option_spans.len();
                stage.questions[question] = QuestionBinding { kind: q.kind, span: q.span, options: option..end };
                stage.option_spans[option..end].copy_from_slice(&q.option_spans);
                question += 1;
                option = end;
            }
        }
        for values in &mut stage.normalized_memory {
            *values = allocate(memory_len, 0.)?;
        }
        for values in &mut stage.kv {
            *values = allocate(memory_len * 2, f32::NAN)?;
        }
        for values in &mut stage.reference_kv {
            *values = allocate(memory_len * 2, 0.)?;
        }
        stage.sequence.copy_from_slice(hidden);
        stage.lexical.copy_from_slice(lexical);
        normalize(&mut stage.sequence);
        head.staged_norm(&mut stage.sequence, "hidden_norm", d);
        if !eligible(&stage.sequence, false) {
            return Err("sequence-input-range");
        }
        if let Some(reference) = &mut stage.reference_memory {
            head.staged_linear(0, &stage.sequence, reference, &stage.segments);
        }
        Ok(stage)
    }
    pub fn phase(&self) -> Phase {
        self.phase
    }
    pub fn begin_descriptor(&self) -> Value {
        Value::Object(vec![
            ("version".into(), integer(1)),
            ("status".into(), Value::Str("ready".into())),
            ("phase".into(), Value::Str("memory".into())),
            ("tokens".into(), integer(self.tokens)),
            ("segments".into(), Value::Array(self.segments.iter().copied().map(integer).collect())),
            ("tile".into(), integer(self.tile as usize)),
            ("validate".into(), Value::Bool(self.validate)),
            ("s".into(), descriptor(&self.sequence)),
            ("memory".into(), descriptor(&self.memory)),
            ("reference".into(), self.reference_memory.as_deref().map_or(Value::Null, descriptor)),
        ])
    }
    pub fn memory_ready(&mut self, head: &Head) -> Result<Value, &'static str> {
        if !self.matches_head(head) {
            return Err("model-or-config-changed");
        }
        if self.phase != Phase::Memory {
            return Err("expected-memory-phase");
        }
        if tile() != self.tile {
            return Err("cpu-tile-changed");
        }
        memory_output_range(&self.memory)?;
        if self.reference_memory.as_ref().is_some_and(|r| !same_bits(r, &self.memory)) {
            return Err("memory-reference-bits");
        }
        for (index, values) in self.normalized_memory.iter_mut().enumerate() {
            values.copy_from_slice(&self.memory);
            head.staged_norm(values, &format!("evidence_layers.{index}.memory_norm"), head.cfg.width);
            if !eligible(values, false) {
                return Err("normalized-memory-input-range");
            }
        }
        for (index, output) in self.reference_kv.iter_mut().enumerate() {
            let input = self.normalized_memory.get(index).map_or(self.memory.as_slice(), Vec::as_slice);
            head.staged_linear(index + 1, input, output, &self.segments);
        }
        self.phase = Phase::Kv;
        let projections = self
            .kv
            .iter()
            .enumerate()
            .map(|(index, output)| {
                let normalized = self.normalized_memory.get(index);
                Value::Object(vec![
                    ("index".into(), integer(index + 1)),
                    ("input".into(), descriptor(normalized.map_or(self.memory.as_slice(), Vec::as_slice))),
                    (
                        "inputSource".into(),
                        Value::Str(if normalized.is_some() { "normalized-memory" } else { "memory" }.into()),
                    ),
                    ("output".into(), descriptor(output)),
                    ("reference".into(), self.reference_kv.get(index).map_or(Value::Null, |values| descriptor(values))),
                ])
            })
            .collect();
        Ok(Value::Object(vec![
            ("version".into(), integer(1)),
            ("status".into(), Value::Str("ready".into())),
            ("phase".into(), Value::Str("kv".into())),
            ("tokens".into(), integer(self.tokens)),
            ("segments".into(), Value::Array(self.segments.iter().copied().map(integer).collect())),
            ("projections".into(), Value::Array(projections)),
        ]))
    }
    pub fn finish(&mut self, head: &mut Head, records: &[Record]) -> Result<Vec<Vec<Vec<f32>>>, String> {
        if !self.matches_head(head) {
            return Err("clef staged head: model or config changed".into());
        }
        if self.phase != Phase::Kv || tile() != self.tile {
            return Err("clef staged head: expected KV phase and unchanged CPU tile".into());
        }
        if !self.matches_records(records) {
            return Err("clef staged head: prepared records changed".into());
        }
        if self.kv.iter().flatten().any(|v| !v.is_finite()) {
            return Err("clef staged head: nonfinite KV output".into());
        }
        if self.validate && self.kv.iter().zip(&self.reference_kv).any(|(a, b)| !same_bits(a, b)) {
            return Err("clef staged head: KV reference bits differ".into());
        }
        let (d, w) = (head.cfg.hidden, head.cfg.width);
        let mut logits = Vec::new();
        logits.try_reserve_exact(records.len()).map_err(|_| "clef staged head: allocation failed")?;
        let mut prepared = Vec::new();
        prepared.try_reserve_exact(self.kv.len()).map_err(|_| "clef staged head: allocation failed")?;
        let (mut token, mut option) = (0, 0);
        for (index, record) in records.iter().enumerate() {
            let end = token + self.segments[index];
            prepared.clear();
            prepared.extend(self.kv.iter().map(|kv| &kv[token * 2 * w..end * 2 * w]));
            logits.push(head.forward_prepared(
                &self.sequence[token * d..end * d],
                &self.memory[token * w..end * w],
                record,
                &self.lexical[option * d..(option + self.options[index]) * d],
                Some(&prepared),
            )?);
            token = end;
            option += self.options[index];
        }
        self.phase = Phase::Finished;
        Ok(logits)
    }
    fn matches_head(&self, head: &Head) -> bool {
        Arc::ptr_eq(&self.store, &head.store) && self.config == head.cfg
    }
    fn matches_records(&self, records: &[Record]) -> bool {
        records.len() == self.segments.len()
            && records.iter().enumerate().all(|(index, record)| {
                let questions = &self.questions[self.record_questions[index].clone()];
                record.ids.len() == self.segments[index]
                    && record.questions.len() == questions.len()
                    && record.questions.iter().zip(questions).all(|(q, bound)| {
                        q.kind == bound.kind
                            && q.span == bound.span
                            && q.option_spans.as_slice() == &self.option_spans[bound.options.clone()]
                    })
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clef::encoding::Question;

    fn records() -> Vec<Record> {
        [1, 4, 5]
            .into_iter()
            .map(|t| Record {
                ids: vec![0; t],
                questions: (0..2)
                    .map(|i| Question {
                        id: format!("q{i}"),
                        kind: 0,
                        span: (0, 1),
                        option_spans: vec![(0, 1), (0, 1)],
                        option_ids: vec!["true".into(), "false".into()],
                        response_ids: vec!["true".into(), "false".into()],
                        criteria: Value::Null,
                    })
                    .collect(),
            })
            .collect()
    }
    fn bits(values: &[Vec<Vec<f32>>]) -> Vec<u32> {
        values.iter().flatten().flatten().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn prepared_projections_preserve_complete_unrounded_logits_and_record_tails() {
        let mut head = super::super::tests::head();
        let records = records();
        let d = head.cfg.hidden;
        let hidden: Vec<_> = (0..10 * d).map(|i| ((i % 31) as f32 - 11.) * 0.13).collect();
        let lexical: Vec<_> = (0..12 * d).map(|i| ((i % 13) as f32 - 3.) * 0.07).collect();
        let mut expected = Vec::new();
        let mut offset = 0;
        for (i, record) in records.iter().enumerate() {
            expected.push(
                head.forward_with_lexical(
                    &hidden[offset * d..(offset + record.ids.len()) * d],
                    record,
                    &lexical[i * 4 * d..(i + 1) * 4 * d],
                )
                .unwrap(),
            );
            offset += record.ids.len();
        }
        let original = hidden.clone();
        let mut stage = StagedHead::begin(&head, &records, &hidden, &lexical, true, |_| {}).unwrap();
        assert_eq!(hidden, original);
        assert!(stage.finish(&mut head, &records).is_err());
        stage.memory.copy_from_slice(stage.reference_memory.as_ref().unwrap());
        stage.memory_ready(&head).unwrap();
        assert!(stage.memory_ready(&head).is_err());
        for (output, reference) in stage.kv.iter_mut().zip(&stage.reference_kv) {
            output.copy_from_slice(reference);
        }
        assert_eq!(bits(&stage.finish(&mut head, &records).unwrap()), bits(&expected));
        assert!(stage.finish(&mut head, &records).is_err());
    }
    #[test]
    fn rejected_inputs_and_supplied_projection_bits_do_not_reach_attention() {
        let mut head = super::super::tests::head();
        let records = records();
        let d = head.cfg.hidden;
        let hidden: Vec<_> = (0..10 * d).map(|i| (i % 11) as f32 * 0.2).collect();
        let lexical = vec![0.1; 12 * d];
        assert!(matches!(
            StagedHead::begin(&head, &records, &hidden[..hidden.len() - 1], &lexical, true, |_| {}),
            Err("hidden-or-lexical-shape")
        ));
        let mut stage = StagedHead::begin(&head, &records, &hidden, &lexical, true, |_| {}).unwrap();
        assert_eq!(stage.memory_ready(&head).unwrap_err(), "memory-output-nonfinite"); // untouched poison.
        stage.memory.fill(2f32.powi(-38));
        assert_eq!(stage.memory_ready(&head).unwrap_err(), "memory-input-range");
        *stage.memory.last_mut().unwrap() = f32::INFINITY;
        assert_eq!(stage.memory_ready(&head).unwrap_err(), "memory-output-nonfinite");
        stage.memory.fill(2f32.powi(21));
        assert_eq!(stage.memory_ready(&head).unwrap_err(), "memory-input-range");
        stage.memory.copy_from_slice(stage.reference_memory.as_ref().unwrap());
        stage.memory[0] = f32::from_bits(stage.memory[0].to_bits() ^ 1);
        assert_eq!(stage.memory_ready(&head).unwrap_err(), "memory-reference-bits");
        stage.memory.copy_from_slice(stage.reference_memory.as_ref().unwrap());
        stage.memory_ready(&head).unwrap();
        for (output, reference) in stage.kv.iter_mut().zip(&stage.reference_kv) {
            output.copy_from_slice(reference);
        }
        stage.kv[0][0] = f32::from_bits(stage.kv[0][0].to_bits() ^ 1);
        assert!(stage.finish(&mut head, &records).unwrap_err().contains("KV reference bits"));
    }
    #[test]
    fn stages_reject_other_models_configs_and_same_size_record_changes() {
        let mut head = super::super::tests::head();
        let mut other = super::super::tests::head();
        let records = records();
        let d = head.cfg.hidden;
        let hidden: Vec<_> = (0..10 * d).map(|i| (i % 11) as f32 * 0.2).collect();
        let lexical = vec![0.1; 12 * d];
        let mut stage = StagedHead::begin(&head, &records, &hidden, &lexical, true, |_| {}).unwrap();
        assert_eq!(stage.memory_ready(&other).unwrap_err(), "model-or-config-changed");
        let config = head.cfg.clone();
        head.cfg.heads = 1;
        assert!(!head.staged_static_eligible());
        assert_eq!(stage.memory_ready(&head).unwrap_err(), "model-or-config-changed");
        head.cfg = config;
        stage.memory.copy_from_slice(stage.reference_memory.as_ref().unwrap());
        stage.memory_ready(&head).unwrap();
        for (output, reference) in stage.kv.iter_mut().zip(&stage.reference_kv) {
            output.copy_from_slice(reference);
        }
        assert!(stage.finish(&mut other, &records).unwrap_err().contains("model or config"));
        let mut changed = records.clone();
        changed[1].questions[0].span = (1, 2);
        assert!(stage.finish(&mut head, &changed).unwrap_err().contains("prepared records changed"));
        changed = records.clone();
        changed[1].questions[0].option_spans[0] = (1, 2);
        assert!(stage.finish(&mut head, &changed).unwrap_err().contains("prepared records changed"));
        changed = records.clone();
        changed[1].questions[0].kind = 1;
        assert!(stage.finish(&mut head, &changed).unwrap_err().contains("prepared records changed"));
        stage.finish(&mut head, &records).unwrap();
    }
    #[test]
    fn range_contract_and_checked_scratch_boundaries() {
        assert!(eligible(&[0., -0., f32::from_bits(OPERAND_MIN), f32::from_bits(MAX_MAGNITUDE - 1)], false));
        for bits in [1, OPERAND_MIN - 1, MAX_MAGNITUDE, 0x7f80_0000, 0x7fc0_0000] {
            assert!(!eligible(&[f32::from_bits(bits)], false));
        }
        assert!(eligible(&[f32::from_bits(BIAS_MIN)], true));
        assert!(!eligible(&[f32::from_bits(BIAS_MIN - 1)], true));
        let binding = BindingSize { records: 1, questions: 1, options: 2 };
        assert!(scratch_bytes(659 * 4096, 659 * 1024, 8192, 2, 6, binding, true).unwrap() < MAX_BYTES);
        assert!(scratch_bytes(usize::MAX, 1, 1, 2, 6, binding, true).is_none());
        assert!(scratch_bytes(1, usize::MAX, 1, 2, 6, binding, false).is_none());
        assert!(scratch_bytes(1, 1, 1, 2, 6, BindingSize { questions: usize::MAX, ..binding }, false).is_none());
        assert!(scratch_bytes(16384 * 4096, 16384 * 1024, 8192, 2, 6, binding, false).unwrap() > MAX_BYTES);
    }
}
