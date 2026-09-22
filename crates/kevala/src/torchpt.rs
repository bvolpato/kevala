//! Reads `torch.save` files (a stored zip holding `data.pkl` and raw storages) far enough to get
//! tensors and plain Python values out, without Python. Kev ships its pointer head this way.
//!
//! The pickle machine knows the opcodes `torch.save` emits for dicts of tensors, numbers, strings,
//! lists, tuples and `OrderedDict`s. Any other global becomes an opaque value.

use std::collections::HashMap;

#[derive(Clone, Debug)]
pub enum Py {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    List(Vec<Py>),
    Tuple(Vec<Py>),
    Dict(Vec<(Py, Py)>),
    Global(String, String),
    Storage { key: String, dtype: String },
    Tensor(TensorRef),
    Mark,
    Opaque,
}

#[derive(Clone, Debug)]
pub struct TensorRef {
    pub key: String,
    pub dtype: String,
    pub offset: usize,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
}

impl Py {
    pub fn get(&self, key: &str) -> Option<&Py> {
        match self {
            Py::Dict(m) => m.iter().find(|(k, _)| matches!(k, Py::Str(s) if s == key)).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Py::Float(f) => Some(*f),
            Py::Int(i) => Some(*i as f64),
            _ => None,
        }
    }
}

/// Files of a stored (uncompressed) zip archive.
pub fn unzip(b: &[u8]) -> Result<HashMap<String, &[u8]>, String> {
    let u16at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]) as usize;
    let u32at = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize;
    let eocd = (0..b.len().saturating_sub(21))
        .rev()
        .find(|&i| b[i..].starts_with(&[0x50, 0x4b, 0x05, 0x06]))
        .ok_or("not a zip file")?;
    let n = u16at(eocd + 10);
    let mut at = u32at(eocd + 16);
    let mut files = HashMap::new();
    for _ in 0..n {
        if !b[at..].starts_with(&[0x50, 0x4b, 0x01, 0x02]) {
            return Err("bad zip central directory".into());
        }
        let method = u16at(at + 10);
        let size = u32at(at + 20);
        let (nlen, xlen, clen) = (u16at(at + 28), u16at(at + 30), u16at(at + 32));
        let local = u32at(at + 42);
        let name = String::from_utf8_lossy(&b[at + 46..at + 46 + nlen]).to_string();
        at += 46 + nlen + xlen + clen;
        if method != 0 {
            return Err(format!("{name}: compressed zip entries are not supported"));
        }
        let data = local + 30 + u16at(local + 26) + u16at(local + 28);
        files.insert(name, b.get(data..data + size).ok_or("zip entry out of range")?);
    }
    Ok(files)
}

fn reduce(f: Py, args: Py) -> Py {
    let Py::Global(m, n) = &f else { return Py::Opaque };
    let a = match args {
        Py::Tuple(a) => a,
        _ => return Py::Opaque,
    };
    match (m.as_str(), n.as_str()) {
        ("torch._utils", "_rebuild_tensor_v2" | "_rebuild_tensor") => {
            let ints = |p: &Py| match p {
                Py::Tuple(v) => {
                    v.iter().filter_map(|x| if let Py::Int(i) = x { Some(*i as usize) } else { None }).collect()
                }
                _ => Vec::new(),
            };
            match (a.first(), a.get(1)) {
                (Some(Py::Storage { key, dtype }), Some(Py::Int(off))) => Py::Tensor(TensorRef {
                    key: key.clone(),
                    dtype: dtype.clone(),
                    offset: *off as usize,
                    shape: a.get(2).map(ints).unwrap_or_default(),
                    stride: a.get(3).map(ints).unwrap_or_default(),
                }),
                _ => Py::Opaque,
            }
        }
        ("collections", "OrderedDict") => Py::Dict(Vec::new()),
        _ => Py::Opaque,
    }
}

/// Runs a pickle and returns the object it builds.
pub fn unpickle(b: &[u8]) -> Result<Py, String> {
    let mut st: Vec<Py> = Vec::new();
    let mut memo: HashMap<usize, Py> = HashMap::new();
    let mut i = 0;
    let rd = |i: &mut usize, n: usize| -> Result<&[u8], String> {
        let s = b.get(*i..*i + n).ok_or("truncated pickle")?;
        *i += n;
        Ok(s)
    };
    let pop_mark = |st: &mut Vec<Py>| -> Vec<Py> {
        let at = st.iter().rposition(|x| matches!(x, Py::Mark)).unwrap_or(0);
        let items = st.split_off(at + 1);
        st.pop();
        items
    };
    let line = |i: &mut usize| -> String {
        let s = *i;
        while *i < b.len() && b[*i] != b'\n' {
            *i += 1;
        }
        let r = String::from_utf8_lossy(&b[s..*i]).to_string();
        *i += 1;
        r
    };
    loop {
        let op = *b.get(i).ok_or("pickle ended without STOP")?;
        i += 1;
        match op {
            0x80 => i += 1, // PROTO
            0x95 => i += 8, // FRAME
            b'.' => return st.pop().ok_or_else(|| "empty pickle".to_string()),
            b'}' => st.push(Py::Dict(Vec::new())),
            b']' => st.push(Py::List(Vec::new())),
            b')' => st.push(Py::Tuple(Vec::new())),
            b'(' => st.push(Py::Mark),
            b'N' => st.push(Py::None),
            0x88 => st.push(Py::Bool(true)),
            0x89 => st.push(Py::Bool(false)),
            b'J' => st.push(Py::Int(i32::from_le_bytes(rd(&mut i, 4)?.try_into().unwrap()) as i64)),
            b'K' => st.push(Py::Int(rd(&mut i, 1)?[0] as i64)),
            b'M' => st.push(Py::Int(u16::from_le_bytes(rd(&mut i, 2)?.try_into().unwrap()) as i64)),
            0x8a => {
                // LONG1
                let n = rd(&mut i, 1)?[0] as usize;
                let d = rd(&mut i, n)?;
                let mut v: i64 = 0;
                for (k, &x) in d.iter().enumerate().take(8) {
                    v |= (x as i64) << (8 * k);
                }
                if n > 0 && n < 8 && d[n - 1] & 0x80 != 0 {
                    v -= 1i64 << (8 * n);
                }
                st.push(Py::Int(v));
            }
            b'G' => st.push(Py::Float(f64::from_be_bytes(rd(&mut i, 8)?.try_into().unwrap()))),
            b'X' => {
                let n = u32::from_le_bytes(rd(&mut i, 4)?.try_into().unwrap()) as usize;
                st.push(Py::Str(String::from_utf8_lossy(rd(&mut i, n)?).to_string()));
            }
            0x8c => {
                let n = rd(&mut i, 1)?[0] as usize;
                st.push(Py::Str(String::from_utf8_lossy(rd(&mut i, n)?).to_string()));
            }
            0x8d => {
                let n = u64::from_le_bytes(rd(&mut i, 8)?.try_into().unwrap()) as usize;
                st.push(Py::Str(String::from_utf8_lossy(rd(&mut i, n)?).to_string()));
            }
            b'B' => {
                let n = u32::from_le_bytes(rd(&mut i, 4)?.try_into().unwrap()) as usize;
                st.push(Py::Bytes(rd(&mut i, n)?.to_vec()));
            }
            b'C' => {
                let n = rd(&mut i, 1)?[0] as usize;
                st.push(Py::Bytes(rd(&mut i, n)?.to_vec()));
            }
            b'q' => {
                let k = rd(&mut i, 1)?[0] as usize;
                memo.insert(k, st.last().cloned().unwrap_or(Py::None));
            }
            b'r' => {
                let k = u32::from_le_bytes(rd(&mut i, 4)?.try_into().unwrap()) as usize;
                memo.insert(k, st.last().cloned().unwrap_or(Py::None));
            }
            0x94 => {
                let k = memo.len();
                memo.insert(k, st.last().cloned().unwrap_or(Py::None));
            }
            b'h' => {
                let k = rd(&mut i, 1)?[0] as usize;
                st.push(memo.get(&k).cloned().ok_or("bad memo reference")?);
            }
            b'j' => {
                let k = u32::from_le_bytes(rd(&mut i, 4)?.try_into().unwrap()) as usize;
                st.push(memo.get(&k).cloned().ok_or("bad memo reference")?);
            }
            b'c' => {
                let m = line(&mut i);
                let n = line(&mut i);
                st.push(Py::Global(m, n));
            }
            0x93 => {
                let n = st.pop();
                let m = st.pop();
                match (m, n) {
                    (Some(Py::Str(m)), Some(Py::Str(n))) => st.push(Py::Global(m, n)),
                    _ => return Err("bad STACK_GLOBAL".into()),
                }
            }
            b't' => {
                let items = pop_mark(&mut st);
                st.push(Py::Tuple(items));
            }
            0x85 | 0x86 | 0x87 => {
                let n = (op - 0x84) as usize;
                let items = st.split_off(st.len().saturating_sub(n));
                st.push(Py::Tuple(items));
            }
            b's' => {
                let v = st.pop().ok_or("stack underflow")?;
                let k = st.pop().ok_or("stack underflow")?;
                if let Some(Py::Dict(m)) = st.last_mut() {
                    m.push((k, v));
                }
            }
            b'u' => {
                let items = pop_mark(&mut st);
                if let Some(Py::Dict(m)) = st.last_mut() {
                    let mut it = items.into_iter();
                    while let (Some(k), Some(v)) = (it.next(), it.next()) {
                        m.push((k, v));
                    }
                }
            }
            b'a' => {
                let v = st.pop().ok_or("stack underflow")?;
                if let Some(Py::List(l)) = st.last_mut() {
                    l.push(v);
                }
            }
            b'e' => {
                let items = pop_mark(&mut st);
                if let Some(Py::List(l)) = st.last_mut() {
                    l.extend(items);
                }
            }
            b'Q' => {
                // BINPERSID: ('storage', StorageType, key, location, numel)
                let pid = st.pop().ok_or("stack underflow")?;
                let v = match pid {
                    Py::Tuple(t) => match (t.get(1), t.get(2)) {
                        (Some(Py::Global(_, ty)), Some(Py::Str(key))) => {
                            Py::Storage { key: key.clone(), dtype: ty.clone() }
                        }
                        _ => Py::Opaque,
                    },
                    _ => Py::Opaque,
                };
                st.push(v);
            }
            b'R' => {
                let args = st.pop().ok_or("stack underflow")?;
                let f = st.pop().ok_or("stack underflow")?;
                st.push(reduce(f, args));
            }
            0x81 => {
                // NEWOBJ
                let args = st.pop().ok_or("stack underflow")?;
                let cls = st.pop().ok_or("stack underflow")?;
                st.push(reduce(cls, args));
            }
            b'b' => {
                // BUILD: state is ignored (OrderedDict metadata, object attributes)
                st.pop();
            }
            0x8f => st.push(Py::List(Vec::new())), // EMPTY_SET, read as a list
            0x90 => {
                let items = pop_mark(&mut st);
                if let Some(Py::List(l)) = st.last_mut() {
                    l.extend(items);
                }
            }
            other => return Err(format!("unsupported pickle opcode 0x{other:02x}")),
        }
    }
}

/// A `torch.save` archive: its object tree and the raw bytes of each storage.
pub struct TorchFile<'a> {
    pub root: Py,
    storages: HashMap<String, &'a [u8]>,
}

impl<'a> TorchFile<'a> {
    pub fn parse(b: &'a [u8]) -> Result<TorchFile<'a>, String> {
        let files = unzip(b)?;
        let pkl = files
            .iter()
            .find(|(k, _)| k.ends_with("/data.pkl") || *k == "data.pkl")
            .map(|(_, v)| *v)
            .ok_or("no data.pkl in archive")?;
        let root = unpickle(pkl)?;
        let mut storages = HashMap::new();
        for (k, v) in &files {
            if let Some(pos) = k.find("/data/") {
                storages.insert(k[pos + 6..].to_string(), *v);
            }
        }
        Ok(TorchFile { root, storages })
    }

    /// A tensor's values as f32 (contiguous float or half storages).
    pub fn f32(&self, t: &TensorRef) -> Result<Vec<f32>, String> {
        let raw = self.storages.get(&t.key).ok_or_else(|| format!("storage {} missing", t.key))?;
        let n: usize = t.shape.iter().product();
        // torch.save writes the tensors it gets; a pointer head's weights are always contiguous
        let mut want = 1;
        for (s, d) in t.stride.iter().zip(&t.shape).rev() {
            if *d > 1 && *s != want {
                return Err("non-contiguous tensors are not supported".into());
            }
            want *= d;
        }
        let (elem, conv): (usize, fn(&[u8]) -> f32) = match t.dtype.as_str() {
            "FloatStorage" => (4, |c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            "HalfStorage" => (2, |c| crate::convert::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))),
            "BFloat16Storage" => (2, |c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)),
            other => return Err(format!("unsupported storage type {other}")),
        };
        let start = t.offset * elem;
        let bytes = raw.get(start..start + n * elem).ok_or("tensor out of storage range")?;
        Ok(bytes.chunks_exact(elem).map(conv).collect())
    }
}
