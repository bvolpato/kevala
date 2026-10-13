//! The joint schema head, including evidence routing and field self attention.

use super::encoding::Record;
use crate::json::Value;
use crate::kernels::{gelu, linear};
use crate::math::softmax;
use crate::pack::{DType, TensorInfo};
use crate::store::Store;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct HeadConfig {
    pub hidden: usize,
    pub width: usize,
    pub routing: usize,
    pub layers: usize,
    pub heads: usize,
    pub feedforward: usize,
}

impl HeadConfig {
    pub fn from_json(c: &Value) -> Result<Self, String> {
        let u = |key: &str| crate::pack::get_usize(c, key);
        let cfg = Self {
            hidden: u("hidden_size")?,
            width: u("width")?,
            routing: u("routing_layers")?,
            layers: u("layers")?,
            heads: u("heads")?,
            feedforward: u("feedforward")?,
        };
        if cfg.hidden == 0
            || cfg.width == 0
            || cfg.heads == 0
            || cfg.width % cfg.heads != 0
            || cfg.hidden % 4 != 0
            || cfg.width % 4 != 0
            || cfg.feedforward == 0
            || cfg.feedforward % 4 != 0
        {
            return Err("clef head dimensions must be positive, divisible by four, and width divisible by heads".into());
        }
        Ok(cfg)
    }

    /// Exact upstream state_dict names and shapes. All head weights stay f32 in the pack.
    pub fn tensors(&self) -> Vec<(String, Vec<usize>)> {
        let mut out = Vec::new();
        let mut add = |name: String, shape: Vec<usize>| out.push((name, shape));
        let (d, w, ff) = (self.hidden, self.width, self.feedforward);
        for name in ["hidden_norm.weight", "hidden_norm.bias"] {
            add(name.into(), vec![d]);
        }
        for name in [
            "memory_projection",
            "question_projection",
            "option_question_projection",
            "global_projection",
            "option_context_projection",
            "option_lexical_projection",
        ] {
            add(format!("{name}.weight"), vec![w, d]);
        }
        add("type_embedding.weight".into(), vec![3, w]);
        for i in 0..self.routing {
            let p = format!("evidence_layers.{i}");
            for norm in ["query_norm", "memory_norm", "feedforward_norm"] {
                for wb in ["weight", "bias"] {
                    add(format!("{p}.{norm}.{wb}"), vec![w]);
                }
            }
            attention_specs(&mut add, &format!("{p}.attention"), w);
            add(format!("{p}.feedforward.0.weight"), vec![ff, w]);
            add(format!("{p}.feedforward.0.bias"), vec![ff]);
            add(format!("{p}.feedforward.3.weight"), vec![w, ff]);
            add(format!("{p}.feedforward.3.bias"), vec![w]);
        }
        for norm in ["option_summary_norm", "field_norm", "option_norm"] {
            for wb in ["weight", "bias"] {
                add(format!("{norm}.{wb}"), vec![w]);
            }
        }
        for i in 0..self.layers {
            let p = format!("layers.{i}");
            for norm in ["norm1", "norm2", "norm3"] {
                for wb in ["weight", "bias"] {
                    add(format!("{p}.{norm}.{wb}"), vec![w]);
                }
            }
            for attn in ["self_attn", "multihead_attn"] {
                attention_specs(&mut add, &format!("{p}.{attn}"), w);
            }
            add(format!("{p}.linear1.weight"), vec![ff, w]);
            add(format!("{p}.linear1.bias"), vec![ff]);
            add(format!("{p}.linear2.weight"), vec![w, ff]);
            add(format!("{p}.linear2.bias"), vec![w]);
        }
        add("residual_scorer.0.weight".into(), vec![w, w * 4]);
        add("residual_scorer.0.bias".into(), vec![w]);
        add("residual_scorer.3.weight".into(), vec![1, w]);
        add("residual_scorer.3.bias".into(), vec![1]);
        for name in ["prior_logit_scale", "joint_logit_scale", "residual_gate"] {
            add(name.into(), vec![]);
        }
        out
    }
}

fn attention_specs(add: &mut impl FnMut(String, Vec<usize>), p: &str, w: usize) {
    add(format!("{p}.in_proj_weight"), vec![w * 3, w]);
    add(format!("{p}.in_proj_bias"), vec![w * 3]);
    add(format!("{p}.out_proj.weight"), vec![w, w]);
    add(format!("{p}.out_proj.bias"), vec![w]);
}

pub struct Head {
    pub cfg: HeadConfig,
    store: Arc<Store>,
    tensors: HashMap<String, TensorInfo>,
    lexical: Option<TensorInfo>,
    panel: Vec<f32>,
}

impl Head {
    pub fn new(cfg: HeadConfig, store: Arc<Store>) -> Result<Self, String> {
        let mut tensors = HashMap::new();
        for (name, shape) in cfg.tensors() {
            let t = store.info(&format!("clef.{name}"))?.clone();
            if t.shape != shape || t.dtype != DType::F32 {
                return Err(format!("clef.{name}: expected f32 {shape:?}, got {:?} {:?}", t.dtype, t.shape));
            }
            tensors.insert(name, t);
        }
        let lexical = if store.has("clef.lexical") {
            let t = store.info("clef.lexical")?.clone();
            if t.shape.len() != 2 || t.cols() != cfg.hidden || t.rows() == 0 {
                return Err("clef.lexical: wrong output embedding dimensions".into());
            }
            Some(t)
        } else {
            None
        };
        Ok(Self { cfg, store, tensors, lexical, panel: Vec::new() })
    }

    fn values(&self, name: &str) -> &[f32] {
        self.store.f32s(&self.tensors[name])
    }

    fn norm(&self, x: &[f32], name: &str, width: usize) -> Vec<f32> {
        let weight = self.values(&format!("{name}.weight"));
        let bias = self.values(&format!("{name}.bias"));
        let mut out = vec![0.; x.len()];
        for (row, dst) in x.chunks_exact(width).zip(out.chunks_exact_mut(width)) {
            let mean = row.iter().sum::<f32>() / width as f32;
            let variance = row.iter().map(|z| (z - mean) * (z - mean)).sum::<f32>() / width as f32;
            let scale = (variance + 1e-5).sqrt().recip();
            for i in 0..width {
                dst[i] = (row[i] - mean) * scale * weight[i] + bias[i];
            }
        }
        out
    }

    fn project(&mut self, x: &[f32], name: &str, biased: bool) -> Vec<f32> {
        let matrix = &self.tensors[&format!("{name}.weight")];
        let mut out = vec![0.; x.len() / matrix.cols() * matrix.rows()];
        let bias = biased.then(|| self.store.f32s(&self.tensors[&format!("{name}.bias")]));
        linear(x, x.len() / matrix.cols(), self.store.mat(matrix), bias, &mut out, &mut self.panel);
        out
    }

    fn attention(&mut self, queries: &[f32], memory: &[f32], name: &str) -> Vec<f32> {
        let w = self.cfg.width;
        let (nq, nm, hd) = (queries.len() / w, memory.len() / w, w / self.cfg.heads);
        let t = &self.tensors[&format!("{name}.in_proj_weight")];
        let mut qkv_query = vec![0.; nq * 3 * w];
        let mut qkv_memory = vec![0.; nm * 3 * w];
        let bias = self.values(&format!("{name}.in_proj_bias")).to_vec();
        linear(queries, nq, self.store.mat(t), Some(&bias), &mut qkv_query, &mut self.panel);
        linear(memory, nm, self.store.mat(t), Some(&bias), &mut qkv_memory, &mut self.panel);
        let mut output = vec![0.; nq * w];
        let scale = (hd as f32).sqrt().recip();
        for q in 0..nq {
            for h in 0..self.cfg.heads {
                let query = &qkv_query[q * 3 * w + h * hd..q * 3 * w + (h + 1) * hd];
                let logits: Vec<_> = (0..nm)
                    .map(|m| {
                        let key = &qkv_memory[m * 3 * w + w + h * hd..m * 3 * w + w + (h + 1) * hd];
                        query.iter().zip(key).map(|(q, k)| q * k).sum::<f32>() * scale
                    })
                    .collect();
                for (m, probability) in softmax(&logits).iter().enumerate() {
                    for i in 0..hd {
                        output[q * w + h * hd + i] += probability * qkv_memory[m * 3 * w + 2 * w + h * hd + i];
                    }
                }
            }
        }
        self.project(&output, &format!("{name}.out_proj"), true)
    }

    fn feedforward(&mut self, x: &[f32], first: &str, second: &str) -> Vec<f32> {
        let mut hidden = self.project(x, first, true);
        hidden.iter_mut().for_each(|x| *x = gelu(*x));
        self.project(&hidden, second, true)
    }

    /// Hidden states have already received Qwen's final RMSNorm. Head LayerNorm is separate.
    pub fn forward(&mut self, hidden: &[f32], record: &Record) -> Result<Vec<Vec<f32>>, String> {
        let d = self.cfg.hidden;
        let table = self.lexical.as_ref().ok_or("clef head: external backend must supply lexical means")?;
        if record.ids.iter().any(|id| *id as usize >= table.rows()) {
            return Err("clef head: token ID lies outside output embeddings".into());
        }
        let mut means = Vec::new();
        let mut row = vec![0.; d];
        for q in &record.questions {
            for span in &q.option_spans {
                if span.0 >= span.1 || span.1 > record.ids.len() {
                    return Err("clef head: invalid option span".into());
                }
                let mut mean = vec![0.; d];
                for id in &record.ids[span.0..span.1] {
                    self.store.row(table, *id as usize, &mut row);
                    for i in 0..d {
                        mean[i] += row[i];
                    }
                }
                mean.iter_mut().for_each(|x| *x /= (span.1 - span.0) as f32);
                means.extend(mean);
            }
        }
        self.forward_with_lexical(hidden, record, &means)
    }

    /// GPU coordinators omit the large output embedding table. Their backend supplies one
    /// mean lexical row per option, in the same field/option order as `Record`.
    pub fn forward_with_lexical(
        &mut self,
        hidden: &[f32],
        record: &Record,
        lexical_means: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        let (d, w) = (self.cfg.hidden, self.cfg.width);
        if hidden.len() != record.ids.len() * d || record.ids.is_empty() || record.questions.is_empty() {
            return Err("clef head: hidden rows do not match the encoded record".into());
        }
        let option_count: usize = record.questions.iter().map(|q| q.option_spans.len()).sum();
        if lexical_means.len() != option_count * d {
            return Err("clef head: lexical means do not match option count".into());
        }
        let sequence = self.norm(hidden, "hidden_norm", d);
        let memory = self.project(&sequence, "memory_projection", false);
        let global = &sequence[sequence.len() - d..];
        let mut question_vectors = Vec::new();
        let mut lexical = Vec::new();
        let mut queries = Vec::new();
        let mut counts = Vec::new();
        for q in &record.questions {
            let question = mean_span(&sequence, d, q.span)?;
            let option_question = self.project(&question, "option_question_projection", false);
            question_vectors.extend(&question);
            counts.push(q.option_spans.len());
            for span in &q.option_spans {
                let context = mean_span(&sequence, d, *span)?;
                if span.0 >= span.1 || span.1 > record.ids.len() {
                    return Err("clef head: invalid option span".into());
                }
                let lex = lexical_means[lexical.len() * d..(lexical.len() + 1) * d].to_vec();
                let mut projected = self.project(&context, "option_context_projection", false);
                let lexical_projected = self.project(&lex, "option_lexical_projection", false);
                for i in 0..w {
                    projected[i] += lexical_projected[i] + option_question[i];
                }
                queries.extend(projected);
                lexical.push(lex);
            }
        }
        for i in 0..self.cfg.routing {
            let p = format!("evidence_layers.{i}");
            let qn = self.norm(&queries, &format!("{p}.query_norm"), w);
            let mn = self.norm(&memory, &format!("{p}.memory_norm"), w);
            add(&mut queries, &self.attention(&qn, &mn, &format!("{p}.attention")));
            let qn = self.norm(&queries, &format!("{p}.feedforward_norm"), w);
            add(&mut queries, &self.feedforward(&qn, &format!("{p}.feedforward.0"), &format!("{p}.feedforward.3")));
        }
        let mut fields = self.project(&question_vectors, "question_projection", false);
        let mut summaries = Vec::new();
        let mut at = 0;
        for (i, count) in counts.iter().enumerate() {
            let field = &fields[i * w..(i + 1) * w];
            let logits: Vec<_> = queries[at * w..(at + count) * w]
                .chunks_exact(w)
                .map(|option| dot(option, field) / (w as f32).sqrt())
                .collect();
            let mut summary = vec![0.; w];
            for (option, prob) in queries[at * w..(at + count) * w].chunks_exact(w).zip(softmax(&logits)) {
                for k in 0..w {
                    summary[k] += option[k] * prob;
                }
            }
            summaries.extend(summary);
            at += count;
        }
        let summaries = self.norm(&summaries, "option_summary_norm", w);
        let global_projected = self.project(global, "global_projection", false);
        let types = self.values("type_embedding.weight");
        for (i, q) in record.questions.iter().enumerate() {
            for k in 0..w {
                fields[i * w + k] += summaries[i * w + k] + global_projected[k] + types[q.kind * w + k];
            }
        }
        for i in 0..self.cfg.layers {
            let p = format!("layers.{i}");
            let normalized = self.norm(&fields, &format!("{p}.norm1"), w);
            add(&mut fields, &self.attention(&normalized, &normalized, &format!("{p}.self_attn")));
            let normalized = self.norm(&fields, &format!("{p}.norm2"), w);
            add(&mut fields, &self.attention(&normalized, &memory, &format!("{p}.multihead_attn")));
            let normalized = self.norm(&fields, &format!("{p}.norm3"), w);
            add(&mut fields, &self.feedforward(&normalized, &format!("{p}.linear1"), &format!("{p}.linear2")));
        }
        let fields = self.norm(&fields, "field_norm", w);
        let options = self.norm(&queries, "option_norm", w);
        let prior_scale = self.values("prior_logit_scale")[0].min(100f32.ln()).exp();
        let joint_scale = self.values("joint_logit_scale")[0].min(100f32.ln()).exp();
        let gate = 1. / (1. + (-self.values("residual_gate")[0]).exp());
        let mut out = Vec::new();
        let mut at = 0;
        for (qi, count) in counts.iter().enumerate() {
            let field = &fields[qi * w..(qi + 1) * w];
            let anchor: Vec<_> =
                question_vectors[qi * d..(qi + 1) * d].iter().zip(global).map(|(a, b)| a + b).collect();
            let mut logits = Vec::new();
            for oi in 0..*count {
                let option = &options[(at + oi) * w..(at + oi + 1) * w];
                let mut features = Vec::with_capacity(4 * w);
                features.extend(field);
                features.extend(option);
                features.extend(field.iter().zip(option).map(|(a, b)| a * b));
                features.extend(field.iter().zip(option).map(|(a, b)| (a - b).abs()));
                let residual = self.feedforward(&features, "residual_scorer.0", "residual_scorer.3")[0];
                // functional.normalize uses 1e-12; cosine_similarity uses 1e-8.
                let prior = cosine(&lexical[at + oi], &anchor, 1e-12) * prior_scale;
                logits.push(prior + gate * (joint_scale * cosine(field, option, 1e-8) + residual));
            }
            out.push(logits);
            at += count;
        }
        Ok(out)
    }
}

fn mean_span(x: &[f32], width: usize, span: (usize, usize)) -> Result<Vec<f32>, String> {
    if span.0 >= span.1 || span.1 > x.len() / width {
        return Err("clef head: invalid token span".into());
    }
    let mut mean = vec![0.; width];
    for row in x[span.0 * width..span.1 * width].chunks_exact(width) {
        for i in 0..width {
            mean[i] += row[i];
        }
    }
    mean.iter_mut().for_each(|x| *x /= (span.1 - span.0) as f32);
    Ok(mean)
}
fn add(x: &mut [f32], y: &[f32]) {
    for (x, y) in x.iter_mut().zip(y) {
        *x += y;
    }
}
fn dot(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}
fn cosine(x: &[f32], y: &[f32], eps: f32) -> f32 {
    dot(x, y) / (dot(x, x).sqrt().max(eps) * dot(y, y).sqrt().max(eps))
}
