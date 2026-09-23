//! The WebGPU kernels. WebGPU only runs WGSL, so every GPU kernel is a `.wgsl` source in
//! `src/wgsl/`, compiled into this crate. This module specializes them for the device and the
//! call site (tile precision, rows and column groups per workgroup, subgroup use, and optionally
//! the matrix shape) and returns the finished source. The browser runtime only turns that source
//! into pipelines and dispatches them.
//!
//! The sources use three directives, each on its own line, plus `{{NAME}}` substitutions:
//!
//! ```text
//! //#include common      the text of common.wgsl, specialized the same way
//! //#if F16              keep the lines up to //#else or //#endif when the flag is set
//! //#else
//! //#endif
//! ```

use crate::json::Value;

/// Every source, embedded at build time. Snippets (`common`, `kev_common`, `splits`) are only
/// ever included by kernels.
const SOURCES: &[(&str, &str)] = &[
    ("common", include_str!("wgsl/common.wgsl")),
    ("splits", include_str!("wgsl/splits.wgsl")),
    ("matmul", include_str!("wgsl/matmul.wgsl")),
    ("matmul_wide", include_str!("wgsl/matmul_wide.wgsl")),
    ("reduce", include_str!("wgsl/reduce.wgsl")),
    ("norm", include_str!("wgsl/norm.wgsl")),
    ("rope", include_str!("wgsl/rope.wgsl")),
    ("attention", include_str!("wgsl/attention.wgsl")),
    ("attention_subgroup", include_str!("wgsl/attention_subgroup.wgsl")),
    ("attention_tile", include_str!("wgsl/attention_tile.wgsl")),
    ("geglu", include_str!("wgsl/geglu.wgsl")),
    ("gather", include_str!("wgsl/gather.wgsl")),
    ("kev_common", include_str!("wgsl/kev_common.wgsl")),
    ("kev_gather", include_str!("wgsl/kev_gather.wgsl")),
    ("kev_rms", include_str!("wgsl/kev_rms.wgsl")),
    ("kev_gates", include_str!("wgsl/kev_gates.wgsl")),
    ("kev_conv", include_str!("wgsl/kev_conv.wgsl")),
    ("kev_save_tail", include_str!("wgsl/kev_save_tail.wgsl")),
    ("kev_qknorm", include_str!("wgsl/kev_qknorm.wgsl")),
    ("kev_recur_lanes", include_str!("wgsl/kev_recur_lanes.wgsl")),
    ("kev_recur_lanes8", include_str!("wgsl/kev_recur_lanes8.wgsl")),
    ("kev_recur_lanes16", include_str!("wgsl/kev_recur_lanes16.wgsl")),
    ("kev_gnorm", include_str!("wgsl/kev_gnorm.wgsl")),
    ("kev_aprep", include_str!("wgsl/kev_aprep.wgsl")),
    ("kev_save_kv", include_str!("wgsl/kev_save_kv.wgsl")),
    ("kev_attention_keys", include_str!("wgsl/kev_attention_keys.wgsl")),
    ("kev_attention", include_str!("wgsl/kev_attention.wgsl")),
    ("kev_attention_tile", include_str!("wgsl/kev_attention_tile.wgsl")),
    ("kev_silumul", include_str!("wgsl/kev_silumul.wgsl")),
    ("gemma4_embed", include_str!("wgsl/gemma4_embed.wgsl")),
    ("gemma4_rms", include_str!("wgsl/gemma4_rms.wgsl")),
    ("gemma4_qkv", include_str!("wgsl/gemma4_qkv.wgsl")),
    ("gemma4_attention", include_str!("wgsl/gemma4_attention.wgsl")),
    ("gemma4_gelu", include_str!("wgsl/gemma4_gelu.wgsl")),
    ("gemma4_ple", include_str!("wgsl/gemma4_ple.wgsl")),
    ("gemma4_residual", include_str!("wgsl/gemma4_residual.wgsl")),
    ("gemma4_gather", include_str!("wgsl/gemma4_gather.wgsl")),
];

/// The kernels a runtime can ask for (everything but the snippets).
pub const KERNELS: &[&str] = &[
    "matmul",
    "matmul_wide",
    "reduce",
    "norm",
    "rope",
    "attention",
    "attention_subgroup",
    "attention_tile",
    "geglu",
    "gather",
    "kev_rms",
    "kev_gather",
    "kev_gates",
    "kev_conv",
    "kev_save_tail",
    "kev_qknorm",
    "kev_recur_lanes",
    "kev_recur_lanes8",
    "kev_recur_lanes16",
    "kev_gnorm",
    "kev_aprep",
    "kev_save_kv",
    "kev_attention_keys",
    "kev_attention",
    "kev_attention_tile",
    "kev_silumul",
    "gemma4_embed",
    "gemma4_rms",
    "gemma4_qkv",
    "gemma4_attention",
    "gemma4_gelu",
    "gemma4_ple",
    "gemma4_residual",
    "gemma4_gather",
];

/// What a kernel is specialized for. Kernels ignore the fields they do not use.
#[derive(Clone, Debug, PartialEq)]
pub struct Spec {
    /// Matmul: keep the workgroup tiles in f16 (half the workgroup-memory traffic).
    pub f16: bool,
    /// Kev attention: reduce inside 32-lane subgroups.
    pub subgroups: bool,
    /// Matmul and reduce: rows per thread, 1 to 4 (a tile covers 16 * rows rows).
    pub rows: u32,
    /// Matmul and reduce: groups of 64 output columns per workgroup, 1 or 2.
    pub groups: u32,
    /// Matmul and reduce: how many workgroups split-K aims for on short inputs.
    pub split_target: u32,
    /// Matmul: N and K compiled in as constants instead of read from the uniform.
    pub shape: Option<(u32, u32)>,
    pub kev: KevSpec,
}

/// Qwen3.5 text dimensions. The optimized kernels use 256-wide attention heads,
/// 128-wide linear heads, and a four-token convolution.
#[derive(Clone, Debug, PartialEq)]
pub struct KevSpec {
    pub hidden: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub lin_key_heads: u32,
    pub lin_heads: u32,
    pub rotary: u32,
}

impl Default for KevSpec {
    fn default() -> Self {
        Self { hidden: 1024, heads: 8, kv_heads: 2, lin_key_heads: 16, lin_heads: 16, rotary: 64 }
    }
}

impl Default for Spec {
    fn default() -> Spec {
        Spec {
            f16: false,
            subgroups: false,
            rows: 4,
            groups: 1,
            split_target: 128,
            shape: None,
            kev: KevSpec::default(),
        }
    }
}

impl Spec {
    /// Reads `{ f16, subgroups, rows, groups, splitTarget, n, k }`; missing fields keep their default.
    pub fn from_json(v: &Value) -> Result<Spec, String> {
        let mut s = Spec::default();
        let flag = |key: &str| v.get(key).map(|x| matches!(x, Value::Bool(true)));
        let number = |key: &str| v.get(key).and_then(Value::as_usize).map(|n| n as u32);
        if let Some(f) = flag("f16") {
            s.f16 = f;
        }
        if let Some(f) = flag("subgroups") {
            s.subgroups = f;
        }
        if let Some(r) = number("rows") {
            s.rows = r;
        }
        if let Some(g) = number("groups") {
            s.groups = g;
        }
        if let Some(t) = number("splitTarget") {
            s.split_target = t;
        }
        if let (Some(n), Some(k)) = (number("n"), number("k")) {
            s.shape = Some((n, k));
        }
        if let Some(v) = v.get("kev") {
            for (key, dst) in [
                ("hidden", &mut s.kev.hidden),
                ("heads", &mut s.kev.heads),
                ("kv_heads", &mut s.kev.kv_heads),
                ("lin_key_heads", &mut s.kev.lin_key_heads),
                ("lin_heads", &mut s.kev.lin_heads),
                ("rotary", &mut s.kev.rotary),
            ] {
                if let Some(n) = v.get(key) {
                    *dst =
                        n.as_usize().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| format!("invalid Kev {key}"))?;
                }
            }
            let k = &s.kev;
            if k.hidden == 0
                || k.hidden > 16384
                || k.hidden % 256 != 0
                || k.heads == 0
                || k.heads > 128
                || k.kv_heads == 0
                || k.heads % k.kv_heads != 0
                || k.lin_key_heads == 0
                || k.lin_key_heads % 2 != 0
                || k.lin_heads == 0
                || k.lin_heads > 128
                || k.lin_heads % k.lin_key_heads != 0
                || k.rotary == 0
                || k.rotary > 256
                || k.rotary % 2 != 0
            {
                return Err("unsupported Qwen3.5 GPU dimensions".into());
            }
        }
        if !(1..=4).contains(&s.rows) {
            return Err(format!("rows must be 1 to 4, not {}", s.rows));
        }
        if !(1..=2).contains(&s.groups) {
            return Err(format!("groups must be 1 or 2, not {}", s.groups));
        }
        Ok(s)
    }

    /// The flags `//#if` tests and the values `{{NAME}}` stands for.
    fn vars(&self) -> Vec<(&'static str, String)> {
        let bm = 16 * self.rows;
        let bn = 64 * self.groups;
        let flag = |on: bool| if on { "1".to_string() } else { String::new() };
        let (n, k) = self.shape.unwrap_or((0, 0));
        vec![
            ("F16", flag(self.f16)),
            ("WIDE_UNROLL", flag(self.f16 && self.rows == 4)),
            ("GENERIC_UNROLL", flag(!self.f16 && self.groups == 1)),
            ("ROWS_GE_2", flag(self.rows >= 2)),
            ("ROWS_GE_3", flag(self.rows >= 3)),
            ("ROWS_GE_4", flag(self.rows >= 4)),
            ("SUBGROUPS", flag(self.subgroups)),
            ("SHAPE", flag(self.shape.is_some())),
            ("TILE", if self.f16 { "f16" } else { "f32" }.to_string()),
            ("KEY_BLOCK", if self.f16 { "16" } else { "8" }.to_string()),
            ("KEY_BUFFER_LEN", if self.f16 { "1024" } else { "512" }.to_string()),
            ("ROWS", self.rows.to_string()),
            ("GROUPS", self.groups.to_string()),
            ("BM", bm.to_string()),
            ("BM_MINUS_1", (bm - 1).to_string()),
            ("BN", bn.to_string()),
            ("BN_MINUS_1", (bn - 1).to_string()),
            ("SPLIT_TARGET", self.split_target.to_string()),
            ("WS_LEN", (8 * bn).to_string()),
            ("WIDE_WS_LEN", (16 * bn).to_string()),
            ("ACC_LEN", (self.rows * self.groups).to_string()),
            ("WV_LEN", (2 * self.groups).to_string()),
            ("N", n.to_string()),
            ("K", k.to_string()),
            ("KEV_HIDDEN", self.kev.hidden.to_string()),
            ("KEV_HEADS", self.kev.heads.to_string()),
            ("KEV_KV_HEADS", self.kev.kv_heads.to_string()),
            ("KEV_LIN_KEY_HEADS", self.kev.lin_key_heads.to_string()),
            ("KEV_LIN_HEADS", self.kev.lin_heads.to_string()),
            ("KEV_ROTARY", self.kev.rotary.to_string()),
            // linear-attention value heads that share key heads (Kev-4B and 9B), not one each
            ("KEV_GROUPED", flag(self.kev.lin_heads != self.kev.lin_key_heads)),
        ]
    }
}

/// The WGSL source of `kernel`, specialized for `spec`.
pub fn wgsl(kernel: &str, spec: &Spec) -> Result<String, String> {
    if !KERNELS.contains(&kernel) {
        return Err(format!("no GPU kernel named {kernel:?}"));
    }
    if kernel == "matmul_wide" && spec.groups != 1 {
        return Err("matmul_wide requires groups=1".into());
    }
    let vars = spec.vars();
    let mut out = String::new();
    render(kernel, &vars, &mut out, 0)?;
    Ok(out)
}

/// `wgsl` for a JSON request `{ kernel, ...spec }`, as the WebAssembly runtime asks for it.
pub fn wgsl_json(request: &Value) -> Result<String, String> {
    let kernel = request.get("kernel").and_then(Value::as_str).ok_or("the request names no kernel")?;
    wgsl(kernel, &Spec::from_json(request)?)
}

fn source(name: &str) -> Result<&'static str, String> {
    SOURCES.iter().find(|(n, _)| *n == name).map(|(_, s)| *s).ok_or_else(|| format!("no WGSL source named {name:?}"))
}

fn render(name: &str, vars: &[(&'static str, String)], out: &mut String, depth: usize) -> Result<(), String> {
    if depth > 4 {
        return Err(format!("{name}: includes nest too deeply"));
    }
    let flag = |key: &str| -> Result<bool, String> {
        vars.iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| !v.is_empty())
            .ok_or_else(|| format!("{name}: unknown flag {key}"))
    };
    // one entry per open //#if: whether its current branch is kept
    let mut open: Vec<bool> = Vec::new();
    for (number, line) in source(name)?.lines().enumerate() {
        let at = || format!("{name}.wgsl:{}", number + 1);
        let directive = line.trim_start();
        if let Some(key) = directive.strip_prefix("//#if ") {
            open.push(flag(key.trim())?);
            continue;
        }
        if directive == "//#else" {
            let branch = open.last_mut().ok_or_else(|| format!("{}: //#else without //#if", at()))?;
            *branch = !*branch;
            continue;
        }
        if directive == "//#endif" {
            open.pop().ok_or_else(|| format!("{}: //#endif without //#if", at()))?;
            continue;
        }
        if open.iter().any(|kept| !kept) {
            continue;
        }
        if let Some(included) = directive.strip_prefix("//#include ") {
            render(included.trim(), vars, out, depth + 1)?;
            continue;
        }
        out.push_str(&substitute(line, vars).map_err(|e| format!("{}: {e}", at()))?);
        out.push('\n');
    }
    if !open.is_empty() {
        return Err(format!("{name}: //#if without //#endif"));
    }
    Ok(())
}

/// Replaces every `{{NAME}}` in `line` with its value.
fn substitute(line: &str, vars: &[(&'static str, String)]) -> Result<String, String> {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(start) = rest.find("{{") {
        out.push_str(&rest[..start]);
        let end = rest[start..].find("}}").ok_or("unclosed {{")? + start;
        let key = &rest[start + 2..end];
        let value = vars.iter().find(|(k, _)| *k == key).ok_or_else(|| format!("unknown value {{{{{key}}}}}"))?;
        out.push_str(&value.1);
        rest = &rest[end + 2..];
    }
    out.push_str(rest);
    Ok(out)
}
