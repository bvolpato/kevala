//! The `.kevala` weight pack: one file holding the model configuration, the tokenizer, and every
//! tensor, laid out so a browser can stream it straight into WebAssembly or GPU memory.
//!
//! ```text
//! 0      "KVLA"            magic
//! 4      u32 version       FORMAT_VERSION
//! 8      u32 header_len    bytes of UTF-8 JSON that follow
//! 12     u32 0             reserved
//! 16     header JSON       {"format","version","model","config","tokenizer","tensors"}
//! ...    zero padding to a 64-byte boundary, then tensor data, each tensor 64-byte aligned
//! ```
//!
//! Tensor dtypes:
//! - `f32`: little-endian f32, row-major.
//! - `q8`:  `[n, k]` int8 row-major, followed at `scales_offset` by `[n, k / block]` f32 scales.
//!   Weight `w[r][c] = q[r][c] * scale[r][c / block]`, symmetric, per row block.

use crate::json::Value;

pub const MAGIC: &[u8; 4] = b"KVLA";
pub const FORMAT_VERSION: u32 = 1;
pub const ALIGN: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    Q8,
}

impl DType {
    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::Q8 => "q8",
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub size: usize,
    /// q8 only
    pub block: usize,
    pub scales_offset: usize,
    pub scales_size: usize,
}

impl TensorInfo {
    pub fn rows(&self) -> usize {
        if self.shape.len() == 2 {
            self.shape[0]
        } else {
            1
        }
    }
    pub fn cols(&self) -> usize {
        *self.shape.last().unwrap_or(&1)
    }
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Clone, Debug)]
pub struct Header {
    pub json: Value,
    pub tensors: Vec<TensorInfo>,
    pub tokenizer_offset: usize,
    pub tokenizer_size: usize,
    /// Total file size implied by the header.
    pub total_size: usize,
}

impl Header {
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
    pub fn config(&self) -> &Value {
        self.json.get("config").unwrap_or(&Value::Null)
    }
}

pub fn align_up(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Bytes needed before `parse_header` can succeed: the 16-byte prefix tells how long the rest is.
pub fn header_len(prefix: &[u8]) -> Result<usize, String> {
    if prefix.len() < 16 {
        return Err("pack is shorter than its 16-byte prefix".into());
    }
    if &prefix[0..4] != MAGIC {
        return Err("not a .kevala pack (bad magic)".into());
    }
    let version = u32_at(prefix, 4);
    if version != FORMAT_VERSION {
        return Err(format!("unsupported .kevala version {version}, this build reads {FORMAT_VERSION}"));
    }
    Ok(16 + u32_at(prefix, 8) as usize)
}

pub fn parse_header(bytes: &[u8]) -> Result<Header, String> {
    let n = header_len(bytes)?;
    if bytes.len() < n {
        return Err("truncated pack header".into());
    }
    let text = std::str::from_utf8(&bytes[16..n]).map_err(|_| "pack header is not UTF-8")?;
    let json = Value::parse(text).map_err(|e| format!("pack header: {e}"))?;
    let need =
        |v: &Value, k: &str| v.get(k).and_then(Value::as_usize).ok_or_else(|| format!("pack header: missing {k}"));
    let tok = json.get("tokenizer").ok_or("pack header: missing tokenizer")?;
    let (tokenizer_offset, tokenizer_size) = (need(tok, "offset")?, need(tok, "size")?);
    let mut tensors = Vec::new();
    let mut end = tokenizer_offset + tokenizer_size;
    for t in json.get("tensors").and_then(Value::as_array).ok_or("pack header: missing tensors")? {
        let name = t.get("name").and_then(Value::as_str).ok_or("tensor without a name")?.to_string();
        let dtype = match t.get("dtype").and_then(Value::as_str) {
            Some("f32") => DType::F32,
            Some("q8") => DType::Q8,
            other => return Err(format!("tensor {name}: unsupported dtype {other:?}")),
        };
        let shape: Vec<usize> = t
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("tensor {name}: missing shape"))?
            .iter()
            .map(|d| d.as_usize().ok_or_else(|| format!("tensor {name}: bad shape")))
            .collect::<Result<_, _>>()?;
        let info = TensorInfo {
            offset: need(t, "offset")?,
            size: need(t, "size")?,
            block: t.get("block").and_then(Value::as_usize).unwrap_or(0),
            scales_offset: t.get("scales_offset").and_then(Value::as_usize).unwrap_or(0),
            scales_size: t.get("scales_size").and_then(Value::as_usize).unwrap_or(0),
            name,
            dtype,
            shape,
        };
        let numel = info.numel();
        let ok = match dtype {
            DType::F32 => info.size == numel * 4,
            DType::Q8 => {
                info.shape.len() == 2
                    && info.block > 0
                    && info.cols() % info.block == 0
                    && info.size == numel
                    && info.scales_size == info.rows() * (info.cols() / info.block) * 4
            }
        };
        if !ok || info.offset % ALIGN != 0 {
            return Err(format!("tensor {}: inconsistent layout", info.name));
        }
        end = end.max(info.offset + info.size).max(info.scales_offset + info.scales_size);
        tensors.push(info);
    }
    Ok(Header { json, tensors, tokenizer_offset, tokenizer_size, total_size: end })
}

/// Lays out a pack: tensors are declared first (shapes only), `layout` then fixes every offset,
/// and the caller writes tensor bytes into place. Tensors keep declaration order, which is also
/// the order a streaming loader sees them.
pub struct Writer {
    meta: Vec<(String, Value)>,
    tokenizer: Vec<u8>,
    tensors: Vec<TensorInfo>,
}

impl Writer {
    pub fn new(model: Value, config: Value, tokenizer: Vec<u8>) -> Writer {
        Writer { meta: vec![("model".into(), model), ("config".into(), config)], tokenizer, tensors: Vec::new() }
    }

    pub fn add_f32(&mut self, name: &str, shape: &[usize]) {
        let size = shape.iter().product::<usize>() * 4;
        self.tensors.push(TensorInfo {
            name: name.into(),
            dtype: DType::F32,
            shape: shape.to_vec(),
            offset: 0,
            size,
            block: 0,
            scales_offset: 0,
            scales_size: 0,
        });
    }

    pub fn add_q8(&mut self, name: &str, rows: usize, cols: usize, block: usize) {
        self.tensors.push(TensorInfo {
            name: name.into(),
            dtype: DType::Q8,
            shape: vec![rows, cols],
            offset: 0,
            size: rows * cols,
            block,
            scales_offset: 0,
            scales_size: rows * (cols / block) * 4,
        });
    }

    /// Returns the bytes before the first tensor (magic, header, tokenizer), every tensor with
    /// its final offsets, and the total pack size.
    pub fn layout(&self) -> (Vec<u8>, Vec<TensorInfo>, usize) {
        // offsets depend on the header length and the header lists the offsets, so lay out
        // until the header fits the room reserved for it
        let mut infos = self.tensors.clone();
        let mut room = 0usize;
        loop {
            let tok_offset = align_up(16 + room);
            let mut at = align_up(tok_offset + self.tokenizer.len());
            for info in infos.iter_mut() {
                info.offset = at;
                at = align_up(at + info.size);
                if info.dtype == DType::Q8 {
                    info.scales_offset = at;
                    at = align_up(at + info.scales_size);
                }
            }
            let json = header_json(&self.meta, tok_offset, self.tokenizer.len(), &infos);
            if json.len() <= room {
                let mut prefix = header_prefix(&json, room);
                prefix.resize(tok_offset, 0);
                prefix.extend_from_slice(&self.tokenizer);
                let first = infos.first().map_or(at, |t| t.offset);
                prefix.resize(first, 0);
                return (prefix, infos, at);
            }
            room = json.len() + 256;
        }
    }
}

/// The header JSON: format, version, `meta` (model, config, and any sub-pack fields), the tokenizer
/// range, and the tensor table.
fn header_json(meta: &[(String, Value)], tok_offset: usize, tok_size: usize, infos: &[TensorInfo]) -> String {
    let num = |n: usize| Value::Int(n.to_string());
    let tensors = infos
        .iter()
        .map(|t| {
            let mut o = vec![
                ("name".to_string(), Value::Str(t.name.clone())),
                ("dtype".to_string(), Value::Str(t.dtype.name().into())),
                ("shape".to_string(), Value::Array(t.shape.iter().map(|&d| num(d)).collect())),
                ("offset".to_string(), num(t.offset)),
                ("size".to_string(), num(t.size)),
            ];
            if t.dtype == DType::Q8 {
                o.push(("block".into(), num(t.block)));
                o.push(("scales_offset".into(), num(t.scales_offset)));
                o.push(("scales_size".into(), num(t.scales_size)));
            }
            Value::Object(o)
        })
        .collect();
    let mut h = vec![
        ("format".to_string(), Value::Str("kevala".into())),
        ("version".to_string(), num(FORMAT_VERSION as usize)),
    ];
    h.extend(meta.iter().cloned());
    h.push((
        "tokenizer".into(),
        Value::Object(vec![("offset".into(), num(tok_offset)), ("size".into(), num(tok_size))]),
    ));
    h.push(("tensors".into(), Value::Array(tensors)));
    Value::Object(h).to_json()
}

/// The 16-byte prefix followed by `json`, padded with spaces to `room` bytes: the padding keeps
/// the declared header length valid JSON.
fn header_prefix(json: &str, room: usize) -> Vec<u8> {
    let mut prefix = Vec::with_capacity(16 + room);
    prefix.extend_from_slice(MAGIC);
    prefix.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    prefix.extend_from_slice(&(room as u32).to_le_bytes());
    prefix.extend_from_slice(&0u32.to_le_bytes());
    prefix.extend_from_slice(json.as_bytes());
    prefix.resize(16 + room, b' ');
    prefix
}

pub fn get_f64(cfg: &Value, key: &str) -> Result<f64, String> {
    cfg.get(key).and_then(Value::as_f64).ok_or_else(|| format!("config: missing {key}"))
}

pub fn get_usize(cfg: &Value, key: &str) -> Result<usize, String> {
    cfg.get(key).and_then(Value::as_usize).ok_or_else(|| format!("config: missing {key}"))
}

/// How to carve one tensor out of a source pack.
#[derive(Clone, Debug)]
pub enum Slice {
    Whole,
    /// Row ranges, concatenated. 1-D tensors count elements as rows.
    Rows(Vec<(usize, usize)>),
    /// Columns `c0..c1` of every row (q8 needs whole blocks).
    Cols(usize, usize),
}

/// Copy `rows` runs of `len` bytes from `src + r * stride` to `dst + r * len`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub src: usize,
    pub dst: usize,
    pub len: usize,
    pub rows: usize,
    pub stride: usize,
}

/// A smaller pack assembled from byte ranges of a bigger one: `prefix` goes at offset 0, then
/// every piece is copied in. A browser applies the pieces while the source streams by, so a
/// worker only ever receives the bytes it keeps.
#[derive(Clone, Debug)]
pub struct Layout {
    pub prefix: Vec<u8>,
    pub pieces: Vec<Piece>,
    pub total: usize,
}

impl Layout {
    pub fn apply(&self, src: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; self.total];
        out[..self.prefix.len()].copy_from_slice(&self.prefix);
        for p in &self.pieces {
            for r in 0..p.rows {
                let s = p.src + r * p.stride;
                out[p.dst + r * p.len..p.dst + (r + 1) * p.len].copy_from_slice(&src[s..s + p.len]);
            }
        }
        out
    }
}

/// Builds the layout of a sub-pack holding the selected tensors (sliced) and optionally the
/// tokenizer. `meta` is merged into the new header next to the source's model and config.
pub fn subset(
    h: &Header,
    select: &dyn Fn(&TensorInfo) -> Option<Slice>,
    tokenizer: bool,
    meta: Vec<(String, Value)>,
) -> Result<Layout, String> {
    let mut infos = Vec::new();
    let mut plans = Vec::new();
    for t in &h.tensors {
        let Some(mut sl) = select(t) else { continue };
        // a slice that keeps everything is a plain copy, which streams as one piece
        let full_k = if t.shape.len() == 2 { t.cols() } else { 1 };
        let full_n = if t.shape.len() == 2 { t.rows() } else { t.numel() };
        match &sl {
            Slice::Cols(c0, c1) if *c0 == 0 && *c1 == full_k => sl = Slice::Whole,
            Slice::Rows(r)
                if r.iter().map(|(a, b)| b - a).sum::<usize>() == full_n
                    && r.windows(2).all(|w| w[0].1 == w[1].0)
                    && r.first().map(|x| x.0) == Some(0) =>
            {
                sl = Slice::Whole
            }
            _ => {}
        }
        let elem = if t.dtype == DType::Q8 { 1 } else { 4 };
        let (n, k) = if t.shape.len() == 2 { (t.rows(), t.cols()) } else { (t.numel(), 1) };
        let nb = if t.dtype == DType::Q8 { k / t.block } else { 0 };
        let mut info = t.clone();
        // (src, len, rows, stride) for data and for scales, destinations assigned below
        type Span = (usize, usize, usize, usize);
        let (data, scales): (Vec<Span>, Vec<Span>) = match &sl {
            Slice::Whole => (vec![(t.offset, t.size, 1, 0)], vec![(t.scales_offset, t.scales_size, 1, 0)]),
            Slice::Rows(ranges) => {
                let rows: usize = ranges.iter().map(|(a, b)| b - a).sum();
                if ranges.iter().any(|&(a, b)| a > b || b > n) {
                    return Err(format!("{}: row slice out of range", t.name));
                }
                info.shape = if t.shape.len() == 2 { vec![rows, k] } else { vec![rows] };
                (
                    ranges.iter().map(|&(a, b)| (t.offset + a * k * elem, (b - a) * k * elem, 1, 0)).collect(),
                    ranges.iter().map(|&(a, b)| (t.scales_offset + a * nb * 4, (b - a) * nb * 4, 1, 0)).collect(),
                )
            }
            Slice::Cols(c0, c1) => {
                if t.shape.len() != 2 || c0 > c1 || *c1 > k {
                    return Err(format!("{}: bad column slice", t.name));
                }
                if t.dtype == DType::Q8 && (c0 % t.block != 0 || c1 % t.block != 0) {
                    return Err(format!("{}: column slice {c0}..{c1} is not on a block boundary", t.name));
                }
                info.shape = vec![n, c1 - c0];
                let b0 = if t.dtype == DType::Q8 { c0 / t.block } else { 0 };
                let b1 = if t.dtype == DType::Q8 { c1 / t.block } else { 0 };
                (
                    vec![(t.offset + c0 * elem, (c1 - c0) * elem, n, k * elem)],
                    vec![(t.scales_offset + b0 * 4, (b1 - b0) * 4, n, nb * 4)],
                )
            }
        };
        info.size = data.iter().map(|d| d.1 * d.2).sum();
        info.scales_size = if t.dtype == DType::Q8 { scales.iter().map(|d| d.1 * d.2).sum() } else { 0 };
        infos.push(info);
        plans.push((data, if t.dtype == DType::Q8 { scales } else { Vec::new() }));
    }
    // the source's model and config first, then the caller's fields
    let mut header_meta: Vec<(String, Value)> =
        ["model", "config"].iter().filter_map(|k| h.json.get(k).map(|v| (k.to_string(), v.clone()))).collect();
    header_meta.extend(meta);
    // header size depends on the offsets it lists, so lay out until it fits
    let mut room = 0usize;
    loop {
        let tok_offset = align_up(16 + room);
        let tok_size = if tokenizer { h.tokenizer_size } else { 0 };
        let mut at = align_up(tok_offset + tok_size);
        let mut pieces = Vec::new();
        if tokenizer {
            pieces.push(Piece { src: h.tokenizer_offset, dst: tok_offset, len: tok_size, rows: 1, stride: 0 });
        }
        for (info, (data, scales)) in infos.iter_mut().zip(&plans) {
            info.offset = at;
            for &(src, len, rows, stride) in data {
                pieces.push(Piece { src, dst: at, len, rows, stride });
                at += len * rows;
            }
            at = align_up(at);
            if info.dtype == DType::Q8 {
                info.scales_offset = at;
                for &(src, len, rows, stride) in scales {
                    pieces.push(Piece { src, dst: at, len, rows, stride });
                    at += len * rows;
                }
                at = align_up(at);
            }
        }
        let json = header_json(&header_meta, tok_offset, tok_size, &infos);
        if json.len() <= room {
            return Ok(Layout { prefix: header_prefix(&json, room), pieces, total: align_up(at) });
        }
        room = json.len() + 256;
    }
}

/// The sub-pack the coordinator needs: tokenizer, embeddings, scorer, act head.
pub fn coord_layout(h: &Header) -> Result<Layout, String> {
    let select = |t: &TensorInfo| (!is_trunk(&t.name)).then_some(Slice::Whole);
    subset(h, &select, true, Vec::new())
}

/// Tensors of the transformer layers, which run in shards or on the GPU.
pub fn is_trunk(name: &str) -> bool {
    name.starts_with("enc.") || name.starts_with("head.") || name.starts_with("L.")
}

/// Every trunk tensor, whole: what a GPU uploads.
pub fn trunk_layout(h: &Header) -> Result<Layout, String> {
    let select = |t: &TensorInfo| is_trunk(&t.name).then_some(Slice::Whole);
    subset(h, &select, false, Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pack with one f32 vector and one q8 matrix `[4, 32]` in blocks of 16, filled so every
    /// byte says where it came from: `q[r][c] = r * 32 + c`, `scale[r][b] = r * 2 + b`.
    fn sample() -> (Vec<u8>, Vec<TensorInfo>) {
        let mut w = Writer::new(Value::Str("m".into()), Value::Object(Vec::new()), b"TOKEN".to_vec());
        w.add_f32("bias", &[3]);
        w.add_q8("weight", 4, 32, 16);
        let (prefix, infos, total) = w.layout();
        let mut pack = vec![0u8; total];
        pack[..prefix.len()].copy_from_slice(&prefix);
        let (bias, weight) = (&infos[0], &infos[1]);
        for (i, v) in [1.5f32, -2.0, 0.25].iter().enumerate() {
            pack[bias.offset + i * 4..bias.offset + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        for i in 0..weight.size {
            pack[weight.offset + i] = i as u8;
        }
        for i in 0..weight.scales_size / 4 {
            pack[weight.scales_offset + i * 4..weight.scales_offset + i * 4 + 4]
                .copy_from_slice(&(i as f32).to_le_bytes());
        }
        (pack, infos)
    }

    fn scales(pack: &[u8], t: &TensorInfo) -> Vec<f32> {
        pack[t.scales_offset..t.scales_offset + t.scales_size]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn a_written_layout_parses_back_to_the_same_tensor_table() {
        let (pack, infos) = sample();
        let h = parse_header(&pack).unwrap();
        assert_eq!(&pack[h.tokenizer_offset..h.tokenizer_offset + h.tokenizer_size], b"TOKEN");
        assert_eq!(h.json.get("model").and_then(Value::as_str), Some("m"));
        assert_eq!(h.tensors.len(), 2);
        for (parsed, written) in h.tensors.iter().zip(&infos) {
            assert_eq!(
                (&parsed.name, parsed.dtype, &parsed.shape, parsed.offset, parsed.size),
                (&written.name, written.dtype, &written.shape, written.offset, written.size)
            );
            assert_eq!(
                (parsed.block, parsed.scales_offset, parsed.scales_size),
                (written.block, written.scales_offset, written.scales_size)
            );
            assert_eq!(parsed.offset % ALIGN, 0, "{} is not aligned", parsed.name);
        }
        let weight = h.tensor("weight").unwrap();
        assert_eq!((weight.rows(), weight.cols(), weight.scales_size), (4, 32, 4 * 2 * 4));
        assert_eq!(weight.scales_offset % ALIGN, 0);
        assert!(h.total_size <= pack.len() && pack.len() - h.total_size < ALIGN);
        assert!(h.tensor("missing").is_none());
    }

    #[test]
    fn damaged_headers_are_rejected_with_the_reason() {
        let (pack, _) = sample();
        assert!(header_len(&pack[..15]).unwrap_err().contains("16-byte prefix"));
        let mut bad = pack.clone();
        bad[0] = b'X';
        assert!(parse_header(&bad).unwrap_err().contains("bad magic"));
        let mut bad = pack.clone();
        bad[4] = 9;
        assert!(parse_header(&bad).unwrap_err().contains("unsupported .kevala version 9"));
        assert!(parse_header(&pack[..40]).unwrap_err().contains("truncated"));
        // A tensor whose size disagrees with its shape must not load.
        let text = String::from_utf8(pack[16..header_len(&pack).unwrap()].to_vec()).unwrap();
        let resized = text.replacen("[3]", "[4]", 1);
        assert_ne!(resized, text);
        let mut bad = pack.clone();
        bad[16..16 + text.len()].copy_from_slice(resized.as_bytes());
        assert!(parse_header(&bad).unwrap_err().contains("bias: inconsistent layout"));
    }

    #[test]
    fn subsets_copy_the_selected_rows_and_columns_with_their_scales() {
        let (pack, _) = sample();
        let h = parse_header(&pack).unwrap();
        let only_weight = |slice: Slice| move |t: &TensorInfo| (t.name == "weight").then(|| slice.clone());

        let rows = subset(&h, &only_weight(Slice::Rows(vec![(1, 3)])), false, Vec::new()).unwrap().apply(&pack);
        let rh = parse_header(&rows).unwrap();
        assert_eq!(rh.tensors.len(), 1);
        let t = &rh.tensors[0];
        assert_eq!(t.shape, [2, 32]);
        assert_eq!(rows[t.offset..t.offset + t.size], (32..96).map(|i| i as u8).collect::<Vec<_>>()[..]);
        assert_eq!(scales(&rows, t), [2.0, 3.0, 4.0, 5.0]);
        assert_eq!(rh.tokenizer_size, 0, "the tokenizer was not requested");

        let meta = vec![("shard".to_string(), Value::Int("1".into()))];
        let cols = subset(&h, &only_weight(Slice::Cols(16, 32)), true, meta).unwrap().apply(&pack);
        let ch = parse_header(&cols).unwrap();
        let t = &ch.tensors[0];
        assert_eq!(t.shape, [4, 16]);
        let want: Vec<u8> = (0..4).flat_map(|r| (16..32).map(move |c| (r * 32 + c) as u8)).collect();
        assert_eq!(cols[t.offset..t.offset + t.size], want[..]);
        assert_eq!(scales(&cols, t), [1.0, 3.0, 5.0, 7.0], "the second block of every row");
        assert_eq!(&cols[ch.tokenizer_offset..ch.tokenizer_offset + ch.tokenizer_size], b"TOKEN");
        assert_eq!(ch.json.get("shard").and_then(Value::as_usize), Some(1));
        assert_eq!(ch.json.get("model").and_then(Value::as_str), Some("m"), "the source's model is kept");

        // A slice that keeps everything streams as one plain copy.
        let whole = subset(&h, &only_weight(Slice::Rows(vec![(0, 2), (2, 4)])), false, Vec::new()).unwrap();
        assert_eq!(whole.pieces.len(), 2, "one piece for the data and one for the scales");

        let error = |slice: Slice| subset(&h, &only_weight(slice), false, Vec::new()).unwrap_err();
        assert!(error(Slice::Cols(8, 32)).contains("not on a block boundary"));
        assert!(error(Slice::Cols(0, 48)).contains("bad column slice"));
        assert!(error(Slice::Rows(vec![(2, 5)])).contains("row slice out of range"));
    }
}
