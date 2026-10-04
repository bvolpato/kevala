//! Tiny synthetic model packs, so tests can run whole engines without downloaded weights.
//!
//! The weights are deterministic pseudo-random numbers. The answers mean nothing, but every
//! invariant the engines promise (shards sum to the whole trunk, a cached state equals a cold
//! run, every backend reads the same request) holds for any weights.

#![allow(dead_code)]

use std::sync::OnceLock;

use kevala::json::Value;
use kevala::pack::{DType, TensorInfo, Writer};
use kevala::tokenizer::Tokenizer;

/// The text of a reference file that is not checked in: `$var`, or `default` under the repository
/// root. Without the file the caller skips its test, and this prints why. CI downloads every
/// such file and sets `KEVALA_REQUIRE_FIXTURES`, which turns a missing file into a failure, so a
/// wrong path cannot pass as a skipped test.
pub fn fixture_text(var: &str, default: &str) -> Option<String> {
    let path = std::env::var(var).unwrap_or_else(|_| format!("{}/../../{default}", env!("CARGO_MANIFEST_DIR")));
    match std::fs::read_to_string(&path) {
        Ok(text) => Some(text),
        Err(error) if std::env::var_os("KEVALA_REQUIRE_FIXTURES").is_some() => panic!("{path}: {error} (set {var})"),
        Err(_) => {
            eprintln!("skipping: no file at {path} (set {var})");
            None
        }
    }
}

/// A real `tokenizer.json` (see [`fixture_text`]), parsed once per test binary.
pub fn fixture_tokenizer(
    cell: &'static OnceLock<Option<Tokenizer>>,
    var: &str,
    default: &str,
) -> Option<&'static Tokenizer> {
    cell.get_or_init(|| {
        fixture_text(var, default)
            .map(|json| Tokenizer::from_hf_json(&json).unwrap_or_else(|e| panic!("{var} or {default}: {e}")))
    })
    .as_ref()
}

/// Quantization block of the synthetic packs. Shard boundaries fall on multiples of it.
pub const BLOCK: usize = 16;

/// GPT-2's printable spelling of every byte, as byte-level `tokenizer.json` vocabularies use it.
fn byte_level_alphabet() -> Vec<char> {
    let printable = |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
    let mut next = 256;
    (0..256u32)
        .map(|b| {
            if printable(b) {
                char::from_u32(b).unwrap()
            } else {
                next += 1;
                char::from_u32(next - 1).unwrap()
            }
        })
        .collect()
}

/// A byte-level BPE tokenizer with one token per byte, no merges, and `added` special tokens
/// numbered from 256.
pub fn tokenizer(added: &[&str]) -> Tokenizer {
    let vocab = Value::Object(
        byte_level_alphabet()
            .into_iter()
            .enumerate()
            .map(|(id, c)| (c.to_string(), Value::Int(id.to_string())))
            .collect(),
    );
    let added = Value::Array(
        added
            .iter()
            .map(|t| {
                Value::Object(vec![
                    ("content".into(), Value::Str(t.to_string())),
                    ("special".into(), Value::Bool(true)),
                ])
            })
            .collect(),
    );
    let json = format!(
        r#"{{"pre_tokenizer": {{"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}},
            "added_tokens": {}, "model": {{"type": "BPE", "vocab": {}, "merges": []}}}}"#,
        added.to_json(),
        vocab.to_json()
    );
    Tokenizer::from_hf_json(&json).expect("synthetic tokenizer")
}

/// A small deterministic generator (SplitMix64), seeded per tensor so a pack does not depend on
/// the order its tensors are declared in.
struct Rng(u64);

impl Rng {
    fn for_name(name: &str) -> Rng {
        Rng(name.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)))
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `-1.0..1.0`.
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// How an f32 tensor is filled.
#[derive(Clone, Copy)]
pub enum Fill {
    /// Uniform in `-amplitude..amplitude`.
    Around(f32),
    /// Norm weights: close to one.
    NearOne,
    /// Uniform in `lo..hi`.
    Between(f32, f32),
}

pub struct PackBuilder {
    writer: Writer,
    fills: Vec<Fill>,
}

impl PackBuilder {
    pub fn new(config: Value, tokenizer: &Tokenizer) -> PackBuilder {
        let model = Value::Object(vec![("name".into(), Value::Str("synthetic".into()))]);
        PackBuilder { writer: Writer::new(model, config, tokenizer.to_bytes()), fills: Vec::new() }
    }

    /// An int8 matrix whose weights are about `1 / sqrt(cols)` in size.
    pub fn q8(&mut self, name: &str, rows: usize, cols: usize) {
        self.writer.add_q8(name, rows, cols, BLOCK);
        self.fills.push(Fill::Around(1.0 / (cols as f32).sqrt()));
    }

    pub fn f32(&mut self, name: &str, shape: &[usize], fill: Fill) {
        self.writer.add_f32(name, shape);
        self.fills.push(fill);
    }

    pub fn build(self) -> Vec<u8> {
        let (prefix, infos, total) = self.writer.layout();
        let mut pack = vec![0u8; total];
        pack[..prefix.len()].copy_from_slice(&prefix);
        for (info, fill) in infos.iter().zip(self.fills) {
            write_tensor(&mut pack, info, fill);
        }
        pack
    }
}

fn write_tensor(pack: &mut [u8], info: &TensorInfo, fill: Fill) {
    let mut rng = Rng::for_name(&info.name);
    let mut put = |at: usize, v: f32| pack[at..at + 4].copy_from_slice(&v.to_le_bytes());
    match info.dtype {
        DType::Q8 => {
            let Fill::Around(amplitude) = fill else { unreachable!("q8 tensors are filled around zero") };
            for i in 0..info.scales_size / 4 {
                put(info.scales_offset + i * 4, amplitude * (0.75 + 0.25 * rng.unit().abs()) / 127.0);
            }
            for b in &mut pack[info.offset..info.offset + info.size] {
                *b = ((rng.unit() * 127.0) as i8) as u8;
            }
        }
        DType::F32 => {
            for i in 0..info.size / 4 {
                let v = match fill {
                    Fill::Around(amplitude) => amplitude * rng.unit(),
                    Fill::NearOne => 1.0 + 0.1 * rng.unit(),
                    Fill::Between(lo, hi) => lo + (hi - lo) * (rng.unit() + 1.0) / 2.0,
                };
                put(info.offset + i * 4, v);
            }
        }
    }
}

fn int(v: usize) -> Value {
    Value::Int(v.to_string())
}

fn object(fields: Vec<(&str, Value)>) -> Value {
    Value::Object(fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// A two-layer Laya encoder (one global layer, one sliding layer) with a one-layer decision
/// head: 32 hidden dimensions in two heads, so it splits into at most two shards.
pub fn laya_pack() -> Vec<u8> {
    let tok = tokenizer(&["[CLS]", "[SEP]", "[MASK]", "[PAD]"]);
    let (d, inter, head_ff, act) = (32, 32, 32, 16);
    let config = object(vec![
        ("arch", Value::Str("laya".into())),
        ("hidden_size", int(d)),
        ("num_attention_heads", int(2)),
        ("num_hidden_layers", int(2)),
        ("intermediate_size", int(inter)),
        ("global_attn_every_n_layers", int(2)),
        ("local_attention", int(8)),
        ("global_rope_theta", Value::Float(160000.0)),
        ("local_rope_theta", Value::Float(10000.0)),
        ("norm_eps", Value::Float(1e-5)),
        ("head_layers", int(1)),
        ("head_ff", int(head_ff)),
        ("head_heads", int(2)),
        ("head_norm_eps", Value::Float(1e-5)),
        ("max_len", int(256)),
        ("head_max_len", int(128)),
        ("act_hidden", int(act)),
        ("block", int(BLOCK)),
    ]);
    let mut p = PackBuilder::new(config, &tok);
    let bias = Fill::Around(0.05);
    p.q8("emb", tok.vocab_size(), d);
    p.f32("emb_norm", &[d], Fill::NearOne);
    p.f32("final_norm", &[d], Fill::NearOne);
    p.f32("type_emb", &[3, d], Fill::Around(0.5));
    p.f32("scorer.norm.w", &[d], Fill::NearOne);
    p.f32("scorer.norm.b", &[d], bias);
    p.f32("scorer.l1", &[d, d], Fill::Around(0.3));
    p.f32("scorer.l1.b", &[d], bias);
    p.f32("scorer.l2", &[1, d], Fill::Around(0.5));
    p.f32("scorer.l2.b", &[1], bias);
    p.f32("act.l1", &[act, d + 4], Fill::Around(0.3));
    p.f32("act.l1.b", &[act], bias);
    p.f32("act.l2", &[2, act], Fill::Around(0.5));
    p.f32("act.l2.b", &[2], bias);
    for i in 0..2 {
        if i > 0 {
            p.f32(&format!("enc.{i}.attn_norm"), &[d], Fill::NearOne);
        }
        p.q8(&format!("enc.{i}.wqkv"), 3 * d, d);
        p.q8(&format!("enc.{i}.wo"), d, d);
        p.f32(&format!("enc.{i}.mlp_norm"), &[d], Fill::NearOne);
        p.q8(&format!("enc.{i}.wi"), 2 * inter, d);
        p.q8(&format!("enc.{i}.wo2"), d, inter);
    }
    p.f32("head.0.norm1.w", &[d], Fill::NearOne);
    p.f32("head.0.norm1.b", &[d], bias);
    p.q8("head.0.in_proj", 3 * d, d);
    p.f32("head.0.in_proj.b", &[3 * d], bias);
    p.q8("head.0.out_proj", d, d);
    p.f32("head.0.out_proj.b", &[d], bias);
    p.f32("head.0.norm2.w", &[d], Fill::NearOne);
    p.f32("head.0.norm2.b", &[d], bias);
    p.q8("head.0.lin1", head_ff, d);
    p.f32("head.0.lin1.b", &[head_ff], bias);
    p.q8("head.0.lin2", d, head_ff);
    p.f32("head.0.lin2.b", &[d], bias);
    p.build()
}

/// Kev's five template delimiters, in `KevConfig::tokens` order.
pub const KEV_SPECIALS: [&str; 5] =
    ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"];

/// A two-layer Kev decoder: one Gated DeltaNet layer, one gated attention layer, and a pointer
/// head. `layers: false` builds the coordinator sub-pack, whose layers would run on the GPU.
pub fn kev_pack(layers: bool) -> Vec<u8> {
    let tok = tokenizer(&KEV_SPECIALS);
    let (d, inter, heads, kv_heads, hd) = (32, 32, 2, 1, 16);
    let (lin_heads, lin_key_heads, dk, dv, conv, ptr) = (2, 1, 16, 16, 4, 16);
    let token_ids = KEV_SPECIALS.iter().map(|t| int(tok.token_id(t).unwrap() as usize)).collect();
    let config = object(vec![
        ("arch", Value::Str("kev".into())),
        ("hidden_size", int(d)),
        ("intermediate_size", int(inter)),
        ("rms_norm_eps", Value::Float(1e-6)),
        ("num_attention_heads", int(heads)),
        ("num_key_value_heads", int(kv_heads)),
        ("head_dim", int(hd)),
        ("rotary_dim", int(8)),
        ("rope_theta", Value::Float(10000.0)),
        ("linear_num_value_heads", int(lin_heads)),
        ("linear_num_key_heads", int(lin_key_heads)),
        ("linear_key_head_dim", int(dk)),
        ("linear_value_head_dim", int(dv)),
        ("linear_conv_kernel_dim", int(conv)),
        ("layer_types", Value::Array(vec![Value::Str("linear_attention".into()), Value::Str("full_attention".into())])),
        ("pointer_dim", int(ptr)),
        ("temperature", Value::Float(1.0)),
        ("max_state", int(256)),
        ("max_branch", int(256)),
        ("template", object(vec![("token_ids", Value::Array(token_ids))])),
    ]);
    let mut p = PackBuilder::new(config, &tok);
    p.q8("emb", tok.vocab_size(), d);
    p.f32("norm", &[d], Fill::NearOne);
    p.f32("ptr.q", &[ptr, d], Fill::Around(0.5));
    p.f32("ptr.q.b", &[ptr], Fill::Around(0.05));
    p.f32("ptr.k", &[ptr, d], Fill::Around(0.5));
    p.f32("ptr.k.b", &[ptr], Fill::Around(0.05));
    if layers {
        // Gated DeltaNet: q and k per key head, v per value head, then the output gate z
        let lin = 2 * lin_key_heads * dk + lin_heads * dv;
        p.f32("L.0.in_norm", &[d], Fill::NearOne);
        p.f32("L.0.post_norm", &[d], Fill::NearOne);
        p.q8("L.0.qkvz", lin + lin_heads * dv, d);
        p.f32("L.0.a", &[lin_heads, d], Fill::Around(0.2));
        p.f32("L.0.b", &[lin_heads, d], Fill::Around(0.2));
        p.f32("L.0.conv", &[lin, conv], Fill::Around(0.5));
        p.f32("L.0.dt_bias", &[lin_heads], Fill::Around(0.5));
        p.f32("L.0.neg_a", &[lin_heads], Fill::Between(-1.0, -0.1));
        p.f32("L.0.gnorm", &[dv], Fill::NearOne);
        p.q8("L.0.out", d, lin_heads * dv);
        p.q8("L.0.gate_up", 2 * inter, d);
        p.q8("L.0.down", d, inter);
        // gated attention: q with its gate per head, then k and v per key/value head
        p.f32("L.1.in_norm", &[d], Fill::NearOne);
        p.f32("L.1.post_norm", &[d], Fill::NearOne);
        p.q8("L.1.qkv", heads * 2 * hd + 2 * kv_heads * hd, d);
        p.f32("L.1.q_norm", &[hd], Fill::NearOne);
        p.f32("L.1.k_norm", &[hd], Fill::NearOne);
        p.q8("L.1.o", d, heads * hd);
        p.q8("L.1.gate_up", 2 * inter, d);
        p.q8("L.1.down", d, inter);
    }
    p.build()
}
