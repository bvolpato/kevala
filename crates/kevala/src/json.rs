//! A small JSON parser and two serializers.
//!
//! `to_json` writes compact standard JSON. `py_dumps` reproduces Python's `json.dumps` byte for
//! byte (default separators, `ensure_ascii` switchable, `repr` floats), because Laya serializes
//! the state and some criteria with it before tokenizing, and a different byte is a different
//! token.
//!
//! Objects keep insertion order like a Python dict. A repeated key keeps its first position and
//! takes the last value, which is what `json.loads` does.

use std::collections::HashMap;
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    /// An integer literal, kept as its decimal digits so big integers survive unchanged.
    Int(String),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Error {
    pub msg: &'static str,
    pub pos: usize,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid JSON at byte {}: {}", self.pos, self.msg)
    }
}

impl Value {
    pub fn parse(s: &str) -> Result<Value, Error> {
        let mut p = Parser { b: s.as_bytes(), i: 0, depth: 0 };
        p.ws();
        let v = p.value()?;
        p.ws();
        if p.i != p.b.len() {
            return Err(p.err("trailing characters"));
        }
        Ok(v)
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(s) => s.parse::<f64>().ok(),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(s) => s.parse::<i64>().ok(),
            Value::Float(f) if f.fract() == 0.0 && f.abs() < 9.0e15 => Some(*f as i64),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_i64().and_then(|v| usize::try_from(v).ok())
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Value)]> {
        match self {
            Value::Object(m) => Some(m),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Compact standard JSON.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        self.write_json(&mut out);
        out
    }

    pub fn write_json(&self, out: &mut String) {
        match self {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Int(s) => out.push_str(s),
            Value::Float(f) => write_js_number(out, *f),
            Value::Str(s) => write_str(out, s, false),
            Value::Array(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write_json(out);
                }
                out.push(']');
            }
            Value::Object(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_str(out, k, false);
                    out.push(':');
                    v.write_json(out);
                }
                out.push('}');
            }
        }
    }

    /// Python `json.dumps(value, ensure_ascii=...)` with the default `(", ", ": ")` separators.
    pub fn py_dumps(&self, ensure_ascii: bool) -> String {
        let mut out = String::new();
        self.write_py(&mut out, ensure_ascii);
        out
    }

    pub fn write_py(&self, out: &mut String, ensure_ascii: bool) {
        match self {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Int(s) => out.push_str(s),
            Value::Float(f) => write_py_float(out, *f),
            Value::Str(s) => write_str(out, s, ensure_ascii),
            Value::Array(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    v.write_py(out, ensure_ascii);
                }
                out.push(']');
            }
            Value::Object(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_str(out, k, ensure_ascii);
                    out.push_str(": ");
                    v.write_py(out, ensure_ascii);
                }
                out.push('}');
            }
        }
    }
}

/// A JSON string literal with Python's escape table: `\" \\ \b \f \n \r \t`, other control
/// characters as lowercase `\u00xx`, and with `ensure_ascii` every non-ASCII code point as
/// `\uxxxx` (surrogate pairs above the BMP).
pub fn write_str(out: &mut String, s: &str, ensure_ascii: bool) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let u4 = |out: &mut String, c: u32| {
        out.push_str("\\u");
        for shift in [12, 8, 4, 0] {
            out.push(HEX[((c >> shift) & 0xf) as usize] as char);
        }
    };
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => u4(out, c as u32),
            c if ensure_ascii && (c as u32) > 0x7e => {
                let c = c as u32;
                if c > 0xffff {
                    let v = c - 0x10000;
                    u4(out, 0xd800 + (v >> 10));
                    u4(out, 0xdc00 + (v & 0x3ff));
                } else {
                    u4(out, c);
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Shortest round-trip digits and decimal point position: value = 0.DIGITS * 10^decpt.
fn shortest_digits(f: f64) -> (String, i32) {
    // `{:e}` prints the shortest digits that round-trip, e.g. "1.2345e-5"
    let s = format!("{:e}", f.abs());
    let (mant, exp) = s.split_once('e').unwrap_or((&s, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0".to_string() } else { digits.to_string() };
    (digits, exp + 1)
}

/// Python `repr(float)`, as `json.dumps` writes floats.
pub fn write_py_float(out: &mut String, f: f64) {
    if f.is_nan() {
        out.push_str("NaN");
        return;
    }
    if f.is_infinite() {
        out.push_str(if f > 0.0 { "Infinity" } else { "-Infinity" });
        return;
    }
    if f.is_sign_negative() {
        out.push('-');
    }
    if f == 0.0 {
        out.push_str("0.0");
        return;
    }
    let (digits, decpt) = shortest_digits(f);
    let d = digits.as_bytes();
    if decpt <= -4 || decpt > 16 {
        out.push(d[0] as char);
        if d.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        out.push('e');
        out.push(if e < 0 { '-' } else { '+' });
        let e = e.unsigned_abs();
        if e < 10 {
            out.push('0');
        }
        out.push_str(&e.to_string());
    } else if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..(-decpt) {
            out.push('0');
        }
        out.push_str(&digits);
    } else {
        let dp = decpt as usize;
        if d.len() <= dp {
            out.push_str(&digits);
            for _ in d.len()..dp {
                out.push('0');
            }
            out.push_str(".0");
        } else {
            out.push_str(&digits[..dp]);
            out.push('.');
            out.push_str(&digits[dp..]);
        }
    }
}

/// A number as JavaScript's `JSON.stringify` writes it, so JS can parse our output losslessly.
pub fn write_js_number(out: &mut String, f: f64) {
    if !f.is_finite() {
        out.push_str("null");
        return;
    }
    if f == 0.0 {
        out.push('0');
        return;
    }
    if f.is_sign_negative() {
        out.push('-');
    }
    let (digits, decpt) = shortest_digits(f);
    let d = digits.as_bytes();
    if decpt > 21 || decpt <= -6 {
        out.push(d[0] as char);
        if d.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        out.push('e');
        out.push(if e < 0 { '-' } else { '+' });
        out.push_str(&e.unsigned_abs().to_string());
    } else if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..(-decpt) {
            out.push('0');
        }
        out.push_str(&digits);
    } else {
        let dp = decpt as usize;
        if d.len() <= dp {
            out.push_str(&digits);
            for _ in d.len()..dp {
                out.push('0');
            }
        } else {
            out.push_str(&digits[..dp]);
            out.push('.');
            out.push_str(&digits[dp..]);
        }
    }
}

/// Objects this big find a repeated key through a hash index rather than a scan, which would make
/// parsing quadratic (a tokenizer vocabulary has 250k keys).
const INDEX_FROM: usize = 16;

fn key_hash(k: &str) -> u64 {
    // FNV-1a
    k.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// Adds a member, keeping a repeated key at its first position with the last value. `index` maps
/// key hashes to positions once the object has INDEX_FROM members.
fn insert_member(m: &mut Vec<(String, Value)>, index: &mut HashMap<u64, usize>, k: String, v: Value) {
    if m.len() >= INDEX_FROM {
        let h = key_hash(&k);
        match index.get(&h) {
            Some(&i) if m[i].0 == k => m[i].1 = v,
            // a hash collision: scan
            Some(_) => match m.iter_mut().find(|(kk, _)| *kk == k) {
                Some(slot) => slot.1 = v,
                None => m.push((k, v)),
            },
            None => {
                index.insert(h, m.len());
                m.push((k, v));
            }
        }
        return;
    }
    if let Some(slot) = m.iter_mut().find(|(kk, _)| *kk == k) {
        slot.1 = v;
        return;
    }
    m.push((k, v));
    if m.len() == INDEX_FROM {
        for (i, (kk, _)) in m.iter().enumerate() {
            index.entry(key_hash(kk)).or_insert(i);
        }
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
    depth: u32,
}

impl Parser<'_> {
    fn err(&self, msg: &'static str) -> Error {
        Error { msg, pos: self.i }
    }

    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn lit(&mut self, word: &[u8], v: Value) -> Result<Value, Error> {
        if self.b[self.i..].starts_with(word) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(self.err("unexpected literal"))
        }
    }

    fn value(&mut self) -> Result<Value, Error> {
        let Some(&c) = self.b.get(self.i) else {
            return Err(self.err("unexpected end of input"));
        };
        match c {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Value::Str(self.string()?)),
            b't' => self.lit(b"true", Value::Bool(true)),
            b'f' => self.lit(b"false", Value::Bool(false)),
            b'n' => self.lit(b"null", Value::Null),
            // Python's json accepts these; JSON.stringify never writes them
            b'N' => self.lit(b"NaN", Value::Float(f64::NAN)),
            b'I' => self.lit(b"Infinity", Value::Float(f64::INFINITY)),
            b'-' if self.b[self.i..].starts_with(b"-Infinity") => {
                self.i += 9;
                Ok(Value::Float(f64::NEG_INFINITY))
            }
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(self.err("unexpected character")),
        }
    }

    fn enter(&mut self) -> Result<(), Error> {
        self.depth += 1;
        if self.depth > 256 {
            return Err(self.err("nesting too deep"));
        }
        Ok(())
    }

    fn object(&mut self) -> Result<Value, Error> {
        self.enter()?;
        self.i += 1;
        let mut m: Vec<(String, Value)> = Vec::new();
        let mut index = HashMap::new();
        self.ws();
        if self.b.get(self.i) == Some(&b'}') {
            self.i += 1;
            self.depth -= 1;
            return Ok(Value::Object(m));
        }
        loop {
            self.ws();
            if self.b.get(self.i) != Some(&b'"') {
                return Err(self.err("expected a string key"));
            }
            let k = self.string()?;
            self.ws();
            if self.b.get(self.i) != Some(&b':') {
                return Err(self.err("expected ':'"));
            }
            self.i += 1;
            self.ws();
            let v = self.value()?;
            insert_member(&mut m, &mut index, k, v);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    self.depth -= 1;
                    return Ok(Value::Object(m));
                }
                _ => return Err(self.err("expected ',' or '}'")),
            }
        }
    }

    fn array(&mut self) -> Result<Value, Error> {
        self.enter()?;
        self.i += 1;
        let mut a = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b']') {
            self.i += 1;
            self.depth -= 1;
            return Ok(Value::Array(a));
        }
        loop {
            self.ws();
            a.push(self.value()?);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    self.depth -= 1;
                    return Ok(Value::Array(a));
                }
                _ => return Err(self.err("expected ',' or ']'")),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, Error> {
        if self.i + 4 > self.b.len() {
            return Err(self.err("short \\u escape"));
        }
        let mut v = 0u32;
        for k in 0..4 {
            let c = self.b[self.i + k];
            let d = match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                b'A'..=b'F' => c - b'A' + 10,
                _ => return Err(self.err("bad \\u escape")),
            };
            v = v * 16 + d as u32;
        }
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, Error> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let start = self.i;
            while self.i < self.b.len() && self.b[self.i] != b'"' && self.b[self.i] != b'\\' {
                if self.b[self.i] < 0x20 {
                    return Err(self.err("control character in string"));
                }
                self.i += 1;
            }
            // the input is a &str, so any run between ASCII delimiters is valid UTF-8
            out.push_str(std::str::from_utf8(&self.b[start..self.i]).map_err(|_| self.err("invalid UTF-8"))?);
            match self.b.get(self.i) {
                None => return Err(self.err("unterminated string")),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                _ => {}
            }
            self.i += 1;
            let Some(&e) = self.b.get(self.i) else {
                return Err(self.err("unterminated escape"));
            };
            self.i += 1;
            match e {
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                b'/' => out.push('/'),
                b'b' => out.push('\u{8}'),
                b'f' => out.push('\u{c}'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                b'u' => {
                    let hi = self.hex4()?;
                    let c = if (0xd800..0xdc00).contains(&hi) && self.b[self.i..].starts_with(b"\\u") {
                        let save = self.i;
                        self.i += 2;
                        let lo = self.hex4()?;
                        if (0xdc00..0xe000).contains(&lo) {
                            0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00)
                        } else {
                            self.i = save;
                            0xfffd
                        }
                    } else {
                        hi
                    };
                    // lone surrogates cannot live in a Rust string
                    out.push(char::from_u32(c).unwrap_or('\u{fffd}'));
                }
                _ => return Err(self.err("bad escape")),
            }
        }
    }

    fn number(&mut self) -> Result<Value, Error> {
        let start = self.i;
        if self.b[self.i] == b'-' {
            self.i += 1;
        }
        let int_start = self.i;
        while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
            self.i += 1;
        }
        if self.i == int_start {
            return Err(self.err("expected digits"));
        }
        if self.b[int_start] == b'0' && self.i - int_start > 1 {
            return Err(self.err("leading zero"));
        }
        let mut float = false;
        if self.b.get(self.i) == Some(&b'.') {
            float = true;
            self.i += 1;
            let f = self.i;
            while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
                self.i += 1;
            }
            if self.i == f {
                return Err(self.err("expected fraction digits"));
            }
        }
        if matches!(self.b.get(self.i), Some(b'e' | b'E')) {
            float = true;
            self.i += 1;
            if matches!(self.b.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            let e = self.i;
            while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
                self.i += 1;
            }
            if self.i == e {
                return Err(self.err("expected exponent digits"));
            }
        }
        let text = std::str::from_utf8(&self.b[start..self.i]).unwrap();
        if float {
            text.parse::<f64>().map(Value::Float).map_err(|_| self.err("bad number"))
        } else if text == "-0" {
            Ok(Value::Int("0".to_string()))
        } else {
            Ok(Value::Int(text.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn py(v: f64) -> String {
        let mut s = String::new();
        write_py_float(&mut s, v);
        s
    }

    #[test]
    fn python_float_repr() {
        let cases = [
            (0.5, "0.5"),
            (1e20, "1e+20"),
            (1e-7, "1e-07"),
            (-2.5, "-2.5"),
            (1.0, "1.0"),
            (5e-324, "5e-324"),
            (std::f64::consts::PI, "3.141592653589793"),
            (12345678901234567890.0, "1.2345678901234567e+19"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (123456789012345680.0, "1.2345678901234568e+17"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (100.0, "100.0"),
            (1.5e300, "1.5e+300"),
            (0.1, "0.1"),
            (2.0 / 3.0, "0.6666666666666666"),
        ];
        for (v, want) in cases {
            assert_eq!(py(v), want, "repr({v:e})");
        }
    }

    #[test]
    fn python_dumps() {
        let v = Value::parse(r#"{"a": [1, 2.0, -0, 1e2], "b": "é\n\u0001\"", "c": {"x": null, "y": true}, "a": 3}"#)
            .unwrap();
        assert_eq!(v.py_dumps(false), "{\"a\": 3, \"b\": \"é\\n\\u0001\\\"\", \"c\": {\"x\": null, \"y\": true}}");
        let v = Value::parse(r#"{"k": "café 😀"}"#).unwrap();
        assert_eq!(v.py_dumps(true), r#"{"k": "caf\u00e9 \ud83d\ude00"}"#);
        assert_eq!(Value::parse("[1.0, -0, 1E2]").unwrap().py_dumps(false), "[1.0, 0, 100.0]");
    }

    #[test]
    fn large_objects_keep_first_position_last_value() {
        let mut src = String::from("{");
        let mut want: Vec<(String, i64)> = Vec::new();
        for i in 0..200i64 {
            // every seventh member repeats an earlier key
            let k = if i % 7 == 6 { format!("k{}", i / 3) } else { format!("k{i}") };
            src.push_str(&format!("{}\"{k}\": {i}", if i > 0 { ", " } else { "" }));
            match want.iter_mut().find(|(kk, _)| *kk == k) {
                Some(slot) => slot.1 = i,
                None => want.push((k, i)),
            }
        }
        src.push('}');
        let got: Vec<(String, i64)> = Value::parse(&src)
            .unwrap()
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_i64().unwrap()))
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn js_numbers() {
        let mut s = String::new();
        for v in [0.9653, 1.0, 1e21, 1e-7, 123.0, -0.5] {
            write_js_number(&mut s, v);
            s.push(' ');
        }
        assert_eq!(s, "0.9653 1 1e+21 1e-7 123 -0.5 ");
    }

    #[test]
    fn rejects_bad_json() {
        for bad in ["", "{", "[1,]", "01", "\"\\x\"", "{\"a\" 1}", "1 2"] {
            assert!(Value::parse(bad).is_err(), "{bad}");
        }
    }
}
