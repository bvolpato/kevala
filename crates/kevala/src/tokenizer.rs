//! Byte-level BPE, reproducing the Hugging Face `tokenizers` pipelines of the tokenizers kevala
//! runs, token for token, with `add_special_tokens=False`: ModernBERT's (Laya) and Qwen2's
//! (Qwen3.5, the base of Kev).
//!
//! 1. split the raw text on the added tokens with `normalized: false` (leftmost-longest, with
//!    their lstrip / rstrip / single_word rules),
//! 2. NFC each remaining piece and split it on the added tokens with `normalized: true`,
//! 3. split what is left with the pre-tokenizer regex, one of the known ones below,
//! 4. BPE each regex piece over its UTF-8 bytes.
//!
//! The vocabulary is kept as raw bytes: the byte-to-char mapping of the ByteLevel pre-tokenizer
//! only exists to make bytes printable in tokenizer.json, so it is undone once at load time.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::json::Value;
use crate::unicode::{is_letter, is_mark, is_number, is_space, nfc_cow};

const MAGIC: &[u8; 8] = b"KVLATOK\0";
/// 2 added the pre-tokenizer choice, 3-byte ids and optional special tokens; 1 still loads.
const VERSION: u32 = 2;
const NONE: u32 = u32::MAX;

/// ByteLevel's own regex (`use_regex: true`).
const GPT2_REGEX: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
/// transformers' `Qwen2Tokenizer` regex, which `AutoTokenizer` applies to Qwen3.5 as well.
const QWEN2_REGEX: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
/// The variant in Qwen3.5's own tokenizer.json, with `\p{M}` joined to the letters.
const QWEN2_MARKS_REGEX: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|",
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
);

/// The pre-tokenizer regex, as recognized in tokenizer.json.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreTokenizer {
    Gpt2 = 0,
    Qwen2 = 1,
    Qwen2Marks = 2,
    /// Gemma's normalizer changes spaces to `▁`; its literal-space split then leaves one piece.
    Gemma = 3,
}

/// FxHash-style hasher (multiply-rotate). Keys are trusted, so no DoS resistance is needed.
#[derive(Default, Clone, Copy)]
struct Fx(u64);

impl Fx {
    #[inline]
    fn add(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0xf135_7aea_2e62_a9c5);
    }
}

impl Hasher for Fx {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.add(u64::from_le_bytes(c.try_into().unwrap()));
        }
        let mut tail = [0u8; 8];
        let r = chunks.remainder();
        tail[..r.len()].copy_from_slice(r);
        self.add(u64::from_le_bytes(tail) ^ (r.len() as u64) << 59);
    }

    #[inline]
    fn write_u8(&mut self, v: u8) {
        self.add(v as u64);
    }

    #[inline]
    fn write_u32(&mut self, v: u32) {
        self.add(v as u64);
    }

    #[inline]
    fn write_u64(&mut self, v: u64) {
        self.add(v);
    }

    #[inline]
    fn write_usize(&mut self, v: usize) {
        self.add(v as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        // the product's good bits are the high ones; hashbrown indexes with the low ones
        self.0.rotate_left(26)
    }
}

type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

/// Merge table: (left id, right id) -> rank << 32 | merged id. Open addressing with linear
/// probing and Fibonacci hashing, kept at most half full.
#[derive(Clone)]
struct PairMap {
    slots: Vec<[u64; 2]>,
    shift: u32,
    len: usize,
}

impl PairMap {
    const EMPTY: u64 = u64::MAX;

    fn with_capacity(n: usize) -> PairMap {
        let cap = (n * 2).next_power_of_two().max(16);
        PairMap { slots: vec![[Self::EMPTY, 0]; cap], shift: 64 - cap.trailing_zeros(), len: 0 }
    }

    #[inline]
    fn slot(&self, key: u64) -> usize {
        (key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> self.shift) as usize
    }

    fn insert(&mut self, a: u32, b: u32, rank: u32, new_id: u32) {
        let key = (a as u64) << 32 | b as u64;
        let mask = self.slots.len() - 1;
        let mut i = self.slot(key);
        loop {
            let s = &mut self.slots[i];
            if s[0] == Self::EMPTY {
                self.len += 1;
                *s = [key, (rank as u64) << 32 | new_id as u64];
                return;
            }
            if s[0] == key {
                s[1] = (rank as u64) << 32 | new_id as u64;
                return;
            }
            i = (i + 1) & mask;
        }
    }

    /// Entries as (left, right, rank, merged) in slot order, starting after an empty slot so no
    /// probe run wraps. Inserted in this order into an empty table of the same size they land in
    /// the same slots, with nearly sequential writes: the blob stores merges this way because
    /// inserting 250k merges in rank order is mostly cache misses.
    fn entries(&self) -> impl Iterator<Item = [u32; 4]> + '_ {
        let z = self.slots.iter().position(|s| s[0] == Self::EMPTY).unwrap_or(0);
        self.slots[z..]
            .iter()
            .chain(&self.slots[..z])
            .filter(|s| s[0] != Self::EMPTY)
            .map(|s| [(s[0] >> 32) as u32, s[0] as u32, (s[1] >> 32) as u32, s[1] as u32])
    }

    /// rank << 32 | merged id, or u64::MAX when the pair does not merge.
    #[inline]
    fn get(&self, a: u32, b: u32) -> u64 {
        let key = (a as u64) << 32 | b as u64;
        let mask = self.slots.len() - 1;
        let mut i = self.slot(key);
        loop {
            let s = self.slots[i];
            if s[0] == key {
                return s[1];
            }
            if s[0] == Self::EMPTY {
                return u64::MAX;
            }
            i = (i + 1) & mask;
        }
    }
}

/// A trie over the added tokens' bytes, for leftmost-longest matching.
#[derive(Clone)]
struct Matcher {
    first: Vec<bool>,
    /// per node: edges (byte, child) and the added-token index ending here (or NONE)
    nodes: Vec<(Vec<(u8, u32)>, u32)>,
}

impl Matcher {
    fn new<'a>(patterns: impl Iterator<Item = (&'a [u8], u32)>) -> Matcher {
        let mut m = Matcher { first: vec![false; 256], nodes: vec![(Vec::new(), NONE)] };
        for (p, value) in patterns {
            if p.is_empty() {
                continue;
            }
            m.first[p[0] as usize] = true;
            let mut n = 0usize;
            for &b in p {
                n = match m.nodes[n].0.iter().find(|e| e.0 == b) {
                    Some(e) => e.1 as usize,
                    None => {
                        let child = m.nodes.len();
                        m.nodes.push((Vec::new(), NONE));
                        m.nodes[n].0.push((b, child as u32));
                        child
                    }
                };
            }
            m.nodes[n].1 = value;
        }
        m
    }

    /// The leftmost match starting at or after `from`, longest among those: (start, end, value).
    fn find(&self, b: &[u8], from: usize) -> Option<(usize, usize, u32)> {
        if self.nodes.len() == 1 {
            return None;
        }
        for start in from..b.len() {
            if !self.first[b[start] as usize] {
                continue;
            }
            let (mut n, mut best) = (0usize, None);
            for (k, &c) in b[start..].iter().enumerate() {
                match self.nodes[n].0.iter().find(|e| e.0 == c) {
                    Some(e) => n = e.1 as usize,
                    None => break,
                }
                if self.nodes[n].1 != NONE {
                    best = Some((start, start + k + 1, self.nodes[n].1));
                }
            }
            if best.is_some() {
                return best;
            }
        }
        None
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AddedToken {
    id: u32,
    content: String,
    special: bool,
    normalized: bool,
    lstrip: bool,
    rstrip: bool,
    single_word: bool,
}

impl AddedToken {
    fn flags(&self) -> u8 {
        self.special as u8
            | (self.normalized as u8) << 1
            | (self.lstrip as u8) << 2
            | (self.rstrip as u8) << 3
            | (self.single_word as u8) << 4
    }
}

/// How an id appears in the BPE model vocabulary.
const KIND_ABSENT: u8 = 0;
/// A byte-level key, stored as the raw bytes it stands for.
const KIND_BYTES: u8 = 1;
/// A key outside the byte-level alphabet (e.g. the runs of plain spaces), stored verbatim. BPE
/// can never produce it; only its added token can.
const KIND_LITERAL: u8 = 2;

#[derive(Clone)]
pub struct Tokenizer {
    kinds: Vec<u8>,
    offsets: Vec<u32>,
    bytes: Vec<u8>,
    byte_ids: [u32; 256],
    /// Gemma starts BPE from Unicode scalar values and falls back to `<0xNN>` byte tokens.
    fallback_ids: [u32; 256],
    /// Only Gemma needs direct Unicode-symbol lookup; byte-level tokenizers use `byte_ids`.
    vocab_ids: Option<FxMap<String, u32>>,
    unk: u32,
    merges: PairMap,
    added: Vec<AddedToken>,
    raw_matcher: Matcher,
    norm_matcher: Matcher,
    nfc: bool,
    pre: PreTokenizer,
    cls: u32,
    sep: u32,
    mask: u32,
    pad: u32,
}

/// GPT-2 `bytes_to_unicode` maps the printable Latin-1 bytes to themselves and the other 68 to
/// U+0100.. in byte order.
const fn printable(b: u8) -> bool {
    matches!(b, b'!'..=b'~' | 0xa1..=0xac | 0xae..=0xff)
}

const NON_PRINTABLE: [u8; 68] = {
    let mut t = [0u8; 68];
    let (mut n, mut b) = (0, 0);
    while b < 256 {
        if !printable(b as u8) {
            t[n] = b as u8;
            n += 1;
        }
        b += 1;
    }
    t
};

/// The byte a tokenizer.json vocab char stands for.
fn byte_level_byte(c: char) -> Option<u8> {
    let u = c as u32;
    if u < 0x100 {
        return printable(u as u8).then_some(u as u8);
    }
    NON_PRINTABLE.get((u - 0x100) as usize).copied()
}

enum Piece<'a> {
    Text(&'a str),
    Token(u32),
}

/// Regex `\w` for single_word tokens, approximated as alphanumeric or `_` (none of Laya's
/// added tokens is single_word).
fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Letter,
    Number,
    Space,
    Other,
}

const ASCII_CLASS: [Class; 128] = {
    let mut t = [Class::Other; 128];
    let mut i = 0;
    while i < 128 {
        let c = i as u8;
        t[i] = if c.is_ascii_alphabetic() {
            Class::Letter
        } else if c.is_ascii_digit() {
            Class::Number
        } else if matches!(c, 9..=13 | b' ') {
            Class::Space
        } else {
            Class::Other
        };
        i += 1;
    }
    t
};

#[inline]
fn class_at(s: &str, i: usize) -> (Class, usize) {
    let b = s.as_bytes()[i];
    if b < 0x80 {
        return (ASCII_CLASS[b as usize], 1);
    }
    let c = s[i..].chars().next().unwrap();
    let k = if is_space(c) {
        Class::Space
    } else if is_letter(c) {
        Class::Letter
    } else if is_number(c) {
        Class::Number
    } else {
        Class::Other
    };
    (k, c.len_utf8())
}

#[inline]
fn run_end(s: &str, mut j: usize, k: Class) -> usize {
    while j < s.len() {
        let (k2, n) = class_at(s, j);
        if k2 != k {
            break;
        }
        j += n;
    }
    j
}

/// End of the GPT-2 regex match starting at `i`. The alternatives cover every character, so the
/// matches tile the string.
fn next_pretoken(s: &str, i: usize) -> usize {
    let b = s.as_bytes();
    if b[i] == b'\'' && i + 1 < b.len() {
        match (b[i + 1], b.get(i + 2)) {
            (b's' | b't' | b'm' | b'd', _) => return i + 2,
            (b'r' | b'v', Some(b'e')) | (b'l', Some(b'l')) => return i + 3,
            _ => {}
        }
    }
    let (mut k, mut n) = class_at(s, i);
    let mut start = i;
    if b[i] == b' ' && i + 1 < b.len() {
        let (k2, n2) = class_at(s, i + 1);
        if k2 != Class::Space {
            (k, n, start) = (k2, n2, i + 1);
        }
    }
    if k != Class::Space {
        return run_end(s, start + n, k);
    }
    // `\s+(?!\S)` backs off one char before a non-space; a single space falls through to `\s+`
    let (mut j, mut last) = (i, i);
    while j < b.len() {
        let (k2, n2) = class_at(s, j);
        if k2 != Class::Space {
            break;
        }
        last = j;
        j += n2;
    }
    if j == b.len() || last == i {
        j
    } else {
        last
    }
}

/// Character classes of the Qwen2 regexes, where `\r` and `\n` differ from the other spaces.
#[derive(Clone, Copy, PartialEq, Eq)]
enum QClass {
    Letter,
    Number,
    /// only for the `\p{M}` variant; otherwise marks are Other
    Mark,
    Space,
    Newline,
    Other,
}

const QWEN_ASCII: [QClass; 128] = {
    let mut t = [QClass::Other; 128];
    let mut i = 0;
    while i < 128 {
        let c = i as u8;
        t[i] = if c.is_ascii_alphabetic() {
            QClass::Letter
        } else if c.is_ascii_digit() {
            QClass::Number
        } else if matches!(c, b'\r' | b'\n') {
            QClass::Newline
        } else if matches!(c, 9 | 11 | 12 | b' ') {
            QClass::Space
        } else {
            QClass::Other
        };
        i += 1;
    }
    t
};

#[inline]
fn qclass_at<const MARKS: bool>(s: &str, i: usize) -> (QClass, usize) {
    let b = s.as_bytes()[i];
    if b < 0x80 {
        return (QWEN_ASCII[b as usize], 1);
    }
    let c = s[i..].chars().next().unwrap();
    let k = if is_space(c) {
        QClass::Space
    } else if is_letter(c) {
        QClass::Letter
    } else if is_number(c) {
        QClass::Number
    } else if MARKS && is_mark(c) {
        QClass::Mark
    } else {
        QClass::Other
    };
    (k, c.len_utf8())
}

#[inline]
fn qrun<const MARKS: bool>(s: &str, mut j: usize, pred: impl Fn(QClass) -> bool) -> usize {
    while j < s.len() {
        let (k, n) = qclass_at::<MARKS>(s, j);
        if !pred(k) {
            break;
        }
        j += n;
    }
    j
}

/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)` on the bytes after the apostrophe: the match length. Oniguruma
/// folds case by Unicode rules, so U+017F LATIN SMALL LETTER LONG S counts as an s.
fn contraction(b: &[u8]) -> Option<usize> {
    let lower = |i: usize| b.get(i).map(u8::to_ascii_lowercase);
    match lower(0)? {
        b's' | b't' | b'm' | b'd' => Some(1),
        b'r' | b'v' if lower(1) == Some(b'e') => Some(2),
        b'l' if lower(1) == Some(b'l') => Some(2),
        0xc5 if b.get(1) == Some(&0xbf) => Some(2),
        _ => None,
    }
}

/// End of the Qwen2 regex match starting at `i` (`MARKS`: the `\p{M}` variant). As with GPT-2 the
/// alternatives cover every character, so the matches tile the string.
fn next_pretoken_qwen<const MARKS: bool>(s: &str, i: usize) -> usize {
    use QClass::*;
    let b = s.as_bytes();
    if b[i] == b'\'' {
        if let Some(n) = contraction(&b[i + 1..]) {
            return i + 1 + n;
        }
    }
    let letter = |k: QClass| k == Letter || k == Mark;
    let (k, n) = qclass_at::<MARKS>(s, i);
    // `[^\r\n\p{L}\p{N}]?\p{L}+`: letters, optionally after one char of another kind
    if letter(k) {
        return qrun::<MARKS>(s, i + n, letter);
    }
    let next = (i + n < b.len()).then(|| qclass_at::<MARKS>(s, i + n));
    if k != Newline && k != Number {
        if let Some((k2, n2)) = next {
            if letter(k2) {
                return qrun::<MARKS>(s, i + n + n2, letter);
            }
        }
    }
    // `\p{N}`: digits one at a time
    if k == Number {
        return i + n;
    }
    // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
    let start = match (k, next) {
        (Other, _) => Some(i + n),
        (_, Some((Other, n2))) if b[i] == b' ' => Some(i + n + n2),
        _ => None,
    };
    if let Some(start) = start {
        let j = qrun::<MARKS>(s, start, |k| k == Other);
        return qrun::<MARKS>(s, j, |k| k == Newline);
    }
    // A run of spaces: `\s*[\r\n]+` reaches through its last line break; without one,
    // `\s+(?!\S)` backs off one char before a non-space and a single space falls to `\s+`.
    let (mut j, mut last, mut newline_end) = (i, i, None);
    while j < b.len() {
        let (k2, n2) = qclass_at::<MARKS>(s, j);
        if k2 != Space && k2 != Newline {
            break;
        }
        if k2 == Newline {
            newline_end = Some(j + n2);
        }
        last = j;
        j += n2;
    }
    match newline_end {
        Some(e) => e,
        None if j == b.len() || last == i => j,
        None => last,
    }
}

/// Reusable buffers for BPE.
#[derive(Default)]
struct Scratch {
    ids: Vec<u32>,
    pairs: Vec<u64>,
    prev: Vec<u32>,
    next: Vec<u32>,
    heap: std::collections::BinaryHeap<std::cmp::Reverse<(u32, u32, u32)>>,
}

/// Words up to this many symbols merge by rescanning; longer ones use tokenizers' heap.
const SCAN_MAX: usize = 48;

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.b.len() - self.i < n {
            return Err("tokenizer blob is truncated".into());
        }
        let s = &self.b[self.i..self.i + n];
        self.i += n;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

/// Blob merges of `fields` values (3: rank implicit, version 1) of `W` bytes each.
fn read_merges<const W: usize>(raw: &[u8], fields: usize, n: usize, merges: &mut PairMap) -> Result<(), String> {
    let at = |m: &[u8], k: usize| {
        let mut v = [0u8; 4];
        v[..W].copy_from_slice(&m[k * W..(k + 1) * W]);
        u32::from_le_bytes(v)
    };
    for (i, m) in raw.chunks_exact(fields * W).enumerate() {
        let (a, b, merged) = (at(m, 0), at(m, 1), at(m, fields - 1));
        let rank = if fields == 4 { at(m, 2) } else { i as u32 };
        if a as usize >= n || b as usize >= n || merged as usize >= n {
            return Err("merge id out of range in tokenizer blob".into());
        }
        merges.insert(a, b, rank, merged);
    }
    Ok(())
}

/// Bytes per value in the blob: the fewest that hold every value below `n`.
fn id_width(n: usize) -> usize {
    if n <= 1 << 16 {
        2
    } else if n <= 1 << 24 {
        3
    } else {
        4
    }
}

fn get_bool(v: &Value, key: &str) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn parse_pre_tokenizer(pre: &Value) -> Result<PreTokenizer, String> {
    fn kind(v: &Value) -> Option<&str> {
        v.get("type").and_then(Value::as_str)
    }
    let byte_level = |v: &Value, regex: bool| {
        kind(v) == Some("ByteLevel")
            && !get_bool(v, "add_prefix_space")
            && v.get("use_regex").and_then(Value::as_bool).unwrap_or(true) == regex
    };
    if byte_level(pre, true) {
        return Ok(PreTokenizer::Gpt2);
    }
    if kind(pre) == Some("Split")
        && pre.get("pattern").and_then(|p| p.get("String")).and_then(Value::as_str) == Some(" ")
        && pre.get("behavior").and_then(Value::as_str) == Some("MergedWithPrevious")
        && !get_bool(pre, "invert")
    {
        return Ok(PreTokenizer::Gemma);
    }
    if let (Some("Sequence"), Some([split, bl])) = (kind(pre), pre.get("pretokenizers").and_then(Value::as_array)) {
        let isolated = split.get("behavior").and_then(Value::as_str) == Some("Isolated") && !get_bool(split, "invert");
        if kind(split) == Some("Split") && isolated && byte_level(bl, false) {
            match split.get("pattern").and_then(|p| p.get("Regex")).and_then(Value::as_str) {
                Some(GPT2_REGEX) => return Ok(PreTokenizer::Gpt2),
                Some(QWEN2_REGEX) => return Ok(PreTokenizer::Qwen2),
                Some(QWEN2_MARKS_REGEX) => return Ok(PreTokenizer::Qwen2Marks),
                _ => {}
            }
        }
    }
    Err(format!(
        "unsupported pre_tokenizer {}: kevala knows ByteLevel with the GPT-2 regex, Split on a literal \
         space for Gemma, and Split on the GPT-2 or Qwen2 regex (with or without \\p{{M}}) followed by \
         ByteLevel without its regex",
        pre.to_json()
    ))
}

impl Tokenizer {
    /// Build from a Hugging Face tokenizer.json. Only configurations kevala reproduces exactly are
    /// accepted: a byte-level BPE model, NFC or no normalizer, and one of the known pre-tokenizer
    /// regexes; anything else is an error rather than a silent mismatch.
    ///
    /// For Qwen3.5 (Kev) pass the tokenizer.json `AutoTokenizer` materializes (Kev checkpoints
    /// ship it). The base repo's own file has another regex and 11 fewer added tokens, both of which
    /// transformers replaces.
    pub fn from_hf_json(json: &str) -> Result<Tokenizer, String> {
        // truncation, padding and the post-processor are ignored: transformers switches the
        // first two off per call, and the SDK encodes with add_special_tokens=False
        let root = Value::parse(json).map_err(|e| e.to_string())?;
        let mut space_to_underscore = false;
        let nfc = match root.get("normalizer") {
            None | Some(Value::Null) => false,
            Some(n) if n.get("type").and_then(Value::as_str) == Some("NFC") => true,
            Some(n)
                if n.get("type").and_then(Value::as_str) == Some("Replace")
                    && n.get("pattern").and_then(|p| p.get("String")).and_then(Value::as_str) == Some(" ")
                    && n.get("content").and_then(Value::as_str) == Some("▁") =>
            {
                space_to_underscore = true;
                false
            }
            Some(n) => return Err(format!("unsupported normalizer {}", n.to_json())),
        };
        let pre = parse_pre_tokenizer(root.get("pre_tokenizer").ok_or("tokenizer.json has no pre_tokenizer")?)?;
        if space_to_underscore != matches!(pre, PreTokenizer::Gemma) {
            return Err("Gemma's space-to-▁ normalizer must be paired with its literal-space pre_tokenizer".into());
        }
        let model = root.get("model").ok_or("tokenizer.json has no model")?;
        if model.get("type").and_then(Value::as_str) != Some("BPE") {
            return Err("unsupported model: expected BPE".into());
        }
        // an empty prefix or suffix is the same as none
        for key in ["continuing_subword_prefix", "end_of_word_suffix"] {
            if model.get(key).is_some_and(|v| !v.is_null() && v.as_str() != Some("")) {
                return Err(format!("unsupported BPE option {key}"));
            }
        }
        let gemma = matches!(pre, PreTokenizer::Gemma);
        if (gemma && model.get("unk_token").and_then(Value::as_str) != Some("<unk>"))
            || (!gemma && model.get("unk_token").is_some_and(|v| !v.is_null()))
        {
            return Err("unsupported BPE option unk_token".into());
        }
        if model.get("dropout").and_then(Value::as_f64).is_some_and(|d| d != 0.0)
            || get_bool(model, "ignore_merges")
            || (gemma && (!get_bool(model, "byte_fallback") || !get_bool(model, "fuse_unk")))
            || (!gemma && get_bool(model, "byte_fallback"))
        {
            return Err("unsupported BPE option (dropout, byte_fallback, fuse_unk or ignore_merges)".into());
        }

        let vocab = model.get("vocab").and_then(Value::as_object).ok_or("model.vocab is not an object")?;
        let mut ids: FxMap<&str, u32> = FxMap::default();
        ids.reserve(vocab.len());
        for (k, v) in vocab {
            let id = v.as_i64().and_then(|i| u32::try_from(i).ok()).filter(|&i| i < NONE);
            ids.insert(k.as_str(), id.ok_or_else(|| format!("bad id for vocab entry {k:?}"))?);
        }

        // Ids as tokenizers' AddedVocabulary::add_tokens assigns them: an existing added token or
        // vocab entry keeps its id, a new one takes the next id after the model vocabulary.
        let mut added: Vec<AddedToken> = Vec::new();
        let mut next_id = vocab.len() as u32;
        for t in root.get("added_tokens").and_then(Value::as_array).unwrap_or(&[]) {
            let content = t.get("content").and_then(Value::as_str).ok_or("added token without content")?;
            if content.is_empty() {
                continue;
            }
            let special = get_bool(t, "special");
            let mut tok = AddedToken {
                id: 0,
                content: content.to_string(),
                special,
                normalized: t.get("normalized").and_then(Value::as_bool).unwrap_or(!special),
                lstrip: get_bool(t, "lstrip"),
                rstrip: get_bool(t, "rstrip"),
                single_word: get_bool(t, "single_word"),
            };
            if let Some(prev) = added.iter_mut().find(|a| a.content == content) {
                tok.id = prev.id;
                *prev = tok;
                continue;
            }
            tok.id = match ids.get(content) {
                Some(&id) => id,
                None => {
                    next_id += 1;
                    next_id - 1
                }
            };
            added.push(tok);
        }

        let max_id = ids.values().chain(added.iter().map(|a| &a.id)).copied().max().unwrap_or(0);
        let n = max_id as usize + 1;
        let mut entries: Vec<(u8, Vec<u8>)> = vec![(KIND_ABSENT, Vec::new()); n];
        for (&k, &id) in &ids {
            // Gemma's symbols are Unicode strings. In particular, its vocab also contains the
            // printable ByteLevel spelling `Ċ` for newline next to the literal newline token;
            // decoding both through the old byte map would make one overwrite the other.
            let raw: Option<Vec<u8>> = (!gemma).then(|| k.chars().map(byte_level_byte).collect()).flatten();
            entries[id as usize] = match raw {
                Some(b) => (KIND_BYTES, b),
                None => (KIND_LITERAL, k.as_bytes().to_vec()),
            };
            if entries[id as usize].1.len() > u16::MAX as usize {
                return Err(format!("vocab entry {k:?} is too long"));
            }
        }
        let mut kinds = Vec::with_capacity(n);
        let mut offsets = Vec::with_capacity(n + 1);
        let mut bytes = Vec::new();
        offsets.push(0);
        for (kind, b) in entries {
            kinds.push(kind);
            bytes.extend_from_slice(&b);
            offsets.push(bytes.len() as u32);
        }

        let list = model.get("merges").and_then(Value::as_array).ok_or("model.merges is not an array")?;
        let mut merges = PairMap::with_capacity(list.len());
        let mut buf = String::new();
        // the legacy "a b" form skips "#version" lines and must have exactly two parts
        let list = list.iter().filter(|m| !m.as_str().is_some_and(|s| s.starts_with("#version")));
        for (rank, m) in list.enumerate() {
            let (a, b) = match m {
                Value::Str(s) => match s.split_once(' ') {
                    Some((a, b)) if !b.contains(' ') => (a, b),
                    _ => return Err(format!("bad merge {s:?}")),
                },
                Value::Array(p) if p.len() == 2 => match (p[0].as_str(), p[1].as_str()) {
                    (Some(a), Some(b)) => (a, b),
                    _ => return Err("bad merge entry".into()),
                },
                _ => return Err("bad merge entry".into()),
            };
            buf.clear();
            buf.push_str(a);
            buf.push_str(b);
            let id = |s: &str| ids.get(s).copied().ok_or_else(|| format!("merge token {s:?} is not in the vocab"));
            // like tokenizers' map, a repeated pair keeps its last rank
            merges.insert(id(a)?, id(b)?, rank as u32, id(&buf)?);
        }

        Tokenizer::build(kinds, offsets, bytes, merges, added, nfc, pre, None)
    }

    fn build(
        kinds: Vec<u8>,
        offsets: Vec<u32>,
        bytes: Vec<u8>,
        merges: PairMap,
        added: Vec<AddedToken>,
        nfc: bool,
        pre: PreTokenizer,
        specials: Option<[u32; 4]>,
    ) -> Result<Tokenizer, String> {
        let mut byte_ids = [NONE; 256];
        for (id, &kind) in kinds.iter().enumerate() {
            if kind == KIND_BYTES && offsets[id + 1] - offsets[id] == 1 {
                byte_ids[bytes[offsets[id] as usize] as usize] = id as u32;
            }
        }
        let mut vocab_ids = matches!(pre, PreTokenizer::Gemma).then(FxMap::default);
        let mut fallback_ids = [NONE; 256];
        if let Some(ids) = &mut vocab_ids {
            for id in 0..kinds.len() {
                if kinds[id] == KIND_ABSENT {
                    continue;
                }
                if let Ok(s) = std::str::from_utf8(&bytes[offsets[id] as usize..offsets[id + 1] as usize]) {
                    ids.insert(s.to_string(), id as u32);
                }
            }
            for a in &added {
                ids.insert(a.content.clone(), a.id);
            }
            for (b, id) in fallback_ids.iter_mut().enumerate() {
                *id = ids.get(&format!("<0x{b:02X}>")).copied().unwrap_or(NONE);
                if *id == NONE {
                    return Err(format!("Gemma tokenizer is missing byte fallback <0x{b:02X}>"));
                }
            }
            if !ids.contains_key("<unk>") {
                return Err("Gemma tokenizer is missing <unk>".into());
            }
        }
        let unk = vocab_ids.as_ref().and_then(|ids| ids.get("<unk>")).copied().unwrap_or(NONE);
        let patterns: Vec<String> = added
            .iter()
            .map(|a| if a.normalized && nfc { crate::unicode::nfc(&a.content) } else { a.content.clone() })
            .collect();
        let matcher = |normalized: bool| {
            Matcher::new(
                (0..added.len())
                    .filter(|&i| added[i].normalized == normalized)
                    .map(|i| (patterns[i].as_bytes(), i as u32)),
            )
        };
        let (raw_matcher, norm_matcher) = (matcher(false), matcher(true));
        let mut tok = Tokenizer {
            kinds,
            offsets,
            bytes,
            byte_ids,
            fallback_ids,
            vocab_ids,
            unk,
            merges,
            added,
            raw_matcher,
            norm_matcher,
            nfc,
            pre,
            cls: 0,
            sep: 0,
            mask: 0,
            pad: 0,
        };
        let specials = match specials {
            Some(s) => s,
            None => {
                let names = if matches!(pre, PreTokenizer::Gemma) {
                    ["<cls>", "<sep>", "<mask>", "<pad>"]
                } else {
                    ["[CLS]", "[SEP]", "[MASK]", "[PAD]"]
                };
                names.map(|name| tok.token_id(name).unwrap_or(NONE))
            }
        };
        [tok.cls, tok.sep, tok.mask, tok.pad] = specials;
        Ok(tok)
    }

    #[inline]
    fn entry(&self, id: usize) -> &[u8] {
        &self.bytes[self.offsets[id] as usize..self.offsets[id + 1] as usize]
    }

    /// Compact little-endian serialization, for embedding in a model pack.
    ///
    /// Layout: magic `KVLATOK\0`, u32 version, u32 id count, u8 width in bytes of ids and ranks
    /// (2 to 4, the least that holds them), u8 flags (bit 0: NFC; bits 1-2: pre-tokenizer, 0 GPT-2,
    /// 1 Qwen2, 2 Qwen2 with `\p{M}`, 3 Gemma), u32 ids of [CLS] [SEP] [MASK] [PAD] (u32::MAX when absent),
    /// u32 merge count, u32 added-token count; a u8 vocab kind per id, a u16 byte length per id,
    /// the entry bytes; the merges as (left, right, rank, merged) in hash-table order; per added
    /// token a u32 id, u8 flags (special, normalized, lstrip, rstrip, single_word), u16 length and
    /// its UTF-8 content. Version 1 stored the merges in rank order as (left, right, merged).
    pub fn to_bytes(&self) -> Vec<u8> {
        let n = self.kinds.len();
        let merges: Vec<[u32; 4]> = self.merges.entries().collect();
        let width = id_width(n.max(merges.iter().map(|m| m[2] as usize + 1).max().unwrap_or(0)));
        let mut out = Vec::with_capacity(64 + 3 * n + self.bytes.len() + merges.len() * 4 * width);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(n as u32).to_le_bytes());
        out.push(width as u8);
        out.push(self.nfc as u8 | (self.pre as u8) << 1);
        for v in [self.cls, self.sep, self.mask, self.pad, merges.len() as u32, self.added.len() as u32] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.kinds);
        for id in 0..n {
            out.extend_from_slice(&(self.entry(id).len() as u16).to_le_bytes());
        }
        out.extend_from_slice(&self.bytes);
        for m in &merges {
            for v in m {
                out.extend_from_slice(&v.to_le_bytes()[..width]);
            }
        }
        for a in &self.added {
            out.extend_from_slice(&a.id.to_le_bytes());
            out.push(a.flags());
            out.extend_from_slice(&(a.content.len() as u16).to_le_bytes());
            out.extend_from_slice(a.content.as_bytes());
        }
        out
    }

    /// Load what `to_bytes` wrote.
    pub fn from_bytes(b: &[u8]) -> Result<Tokenizer, String> {
        let mut r = Reader { b, i: 0 };
        if r.take(8)? != MAGIC {
            return Err("not a kevala tokenizer blob".into());
        }
        let version = r.u32()?;
        if version != 1 && version != VERSION {
            return Err(format!("unsupported tokenizer blob version {version}"));
        }
        let n = r.u32()? as usize;
        let width = r.u8()? as usize;
        let flags = r.u8()?;
        let mut head = [0u32; 6];
        for v in &mut head {
            *v = r.u32()?;
        }
        let [cls, sep, mask, pad, n_merges, n_added] = head;
        if n == 0 || !(2..=4).contains(&width) || width < id_width(n) {
            return Err("bad id count or width in tokenizer blob".into());
        }
        let pre = match flags >> 1 {
            0 => PreTokenizer::Gpt2,
            1 => PreTokenizer::Qwen2,
            2 => PreTokenizer::Qwen2Marks,
            3 => PreTokenizer::Gemma,
            _ => return Err("unknown pre-tokenizer in tokenizer blob".into()),
        };
        if [cls, sep, mask, pad].iter().any(|&i| i != NONE && i as usize >= n) {
            return Err("special token id out of range in tokenizer blob".into());
        }
        let kinds = r.take(n)?;
        if kinds.iter().any(|&k| k > KIND_LITERAL) {
            return Err("bad vocab entry kind in tokenizer blob".into());
        }
        let mut offsets = Vec::with_capacity(n + 1);
        let mut off = 0usize;
        offsets.push(0);
        for l in r.take(2 * n)?.chunks_exact(2) {
            off += u16::from_le_bytes([l[0], l[1]]) as usize;
            offsets.push(off as u32);
        }
        let bytes = r.take(off)?.to_vec();

        let fields = if version == 1 { 3 } else { 4 };
        let size = (n_merges as usize).checked_mul(fields * width).ok_or("tokenizer blob is truncated")?;
        let raw = r.take(size)?;
        let mut merges = PairMap::with_capacity(n_merges as usize);
        match width {
            2 => read_merges::<2>(raw, fields, n, &mut merges)?,
            3 => read_merges::<3>(raw, fields, n, &mut merges)?,
            _ => read_merges::<4>(raw, fields, n, &mut merges)?,
        }

        let mut added = Vec::new();
        for _ in 0..n_added {
            let id = r.u32()?;
            let f = r.u8()?;
            let len = r.u16()? as usize;
            let content = std::str::from_utf8(r.take(len)?).map_err(|_| "added token is not UTF-8")?;
            if id as usize >= n {
                return Err("added token id out of range in tokenizer blob".into());
            }
            added.push(AddedToken {
                id,
                content: content.to_string(),
                special: f & 1 != 0,
                normalized: f & 2 != 0,
                lstrip: f & 4 != 0,
                rstrip: f & 8 != 0,
                single_word: f & 16 != 0,
            });
        }
        if r.i != b.len() {
            return Err("trailing bytes in tokenizer blob".into());
        }
        Tokenizer::build(
            kinds.to_vec(),
            offsets,
            bytes,
            merges,
            added,
            flags & 1 != 0,
            pre,
            Some([cls, sep, mask, pad]),
        )
    }

    /// Token ids of `text`, as `tokenizers` encodes it with `add_special_tokens=False`.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_prefix(text, usize::MAX)
    }

    /// `encode(text)` cut to its first `max_tokens` ids, without tokenizing much past them.
    pub fn encode_prefix(&self, text: &str, max_tokens: usize) -> Vec<u32> {
        let mut out = Vec::new();
        if max_tokens == 0 {
            return out;
        }
        let mut sc = Scratch::default();
        self.split_added(text, &self.raw_matcher, &mut |piece| match piece {
            Piece::Token(id) => {
                out.push(id);
                out.len() < max_tokens
            }
            Piece::Text(seg) => {
                let norm = if matches!(self.pre, PreTokenizer::Gemma) {
                    std::borrow::Cow::Owned(seg.replace(' ', "▁"))
                } else if self.nfc {
                    nfc_cow(seg)
                } else {
                    std::borrow::Cow::Borrowed(seg)
                };
                self.split_added(&norm, &self.norm_matcher, &mut |p| match p {
                    Piece::Token(id) => {
                        out.push(id);
                        out.len() < max_tokens
                    }
                    Piece::Text(t) => self.encode_words(t, &mut out, max_tokens, &mut sc),
                })
            }
        });
        out.truncate(max_tokens);
        out
    }

    /// tokenizers' AddedVocabulary::find_matches: cut `s` around the added tokens of `m`, calling
    /// `f` for each piece in order until it returns false.
    fn split_added<'a>(&self, s: &'a str, m: &Matcher, f: &mut impl FnMut(Piece<'a>) -> bool) -> bool {
        let b = s.as_bytes();
        let (mut done, mut pos) = (0, 0);
        while let Some((mstart, mend, i)) = m.find(b, pos) {
            pos = mend;
            let tok = &self.added[i as usize];
            let (mut start, mut stop) = (mstart, mend);
            if tok.single_word {
                let before = s[..start].chars().next_back().is_some_and(is_word);
                let after = s[stop..].chars().next().is_some_and(is_word);
                if before || after {
                    continue;
                }
            }
            if tok.lstrip {
                // the spaces may already belong to the previous match
                let ws = s[..start].chars().rev().take_while(|&c| c.is_whitespace()).map(char::len_utf8).sum::<usize>();
                start = (start - ws).max(done);
            }
            if tok.rstrip {
                stop += s[stop..].chars().take_while(|&c| c.is_whitespace()).map(char::len_utf8).sum::<usize>();
            }
            if done < start && !f(Piece::Text(&s[done..start])) {
                return false;
            }
            if !f(Piece::Token(tok.id)) {
                return false;
            }
            done = stop;
        }
        done >= s.len() || f(Piece::Text(&s[done..]))
    }

    fn encode_words(&self, t: &str, out: &mut Vec<u32>, max_tokens: usize, sc: &mut Scratch) -> bool {
        match self.pre {
            PreTokenizer::Gpt2 => self.words(t, out, max_tokens, sc, next_pretoken),
            PreTokenizer::Qwen2 => self.words(t, out, max_tokens, sc, next_pretoken_qwen::<false>),
            PreTokenizer::Qwen2Marks => self.words(t, out, max_tokens, sc, next_pretoken_qwen::<true>),
            PreTokenizer::Gemma => self.gemma(t, out, max_tokens, sc),
        }
    }

    #[inline(always)]
    fn words(
        &self,
        t: &str,
        out: &mut Vec<u32>,
        max_tokens: usize,
        sc: &mut Scratch,
        next: impl Fn(&str, usize) -> usize,
    ) -> bool {
        let mut i = 0;
        while i < t.len() {
            if out.len() >= max_tokens {
                return false;
            }
            let end = next(t, i);
            self.bpe(&t.as_bytes()[i..end], out, sc);
            i = end;
        }
        out.len() < max_tokens
    }

    fn bpe(&self, word: &[u8], out: &mut Vec<u32>, sc: &mut Scratch) {
        if let [b] = word {
            // bytes missing from the vocab are dropped, as tokenizers does without an unk token
            if self.byte_ids[*b as usize] != NONE {
                out.push(self.byte_ids[*b as usize]);
            }
            return;
        }
        sc.ids.clear();
        sc.ids.extend(word.iter().map(|&b| self.byte_ids[b as usize]).filter(|&id| id != NONE));
        self.bpe_ids(out, sc);
    }

    /// Gemma's BPE starts from Unicode characters and substitutes byte-fallback symbols only for
    /// characters absent from the vocabulary. Its merge table still uses the same id-pair engine.
    fn gemma(&self, word: &str, out: &mut Vec<u32>, max_tokens: usize, sc: &mut Scratch) -> bool {
        let Some(vocab) = &self.vocab_ids else { return true };
        sc.ids.clear();
        for c in word.chars() {
            let mut encoded = [0u8; 4];
            let s = c.encode_utf8(&mut encoded);
            if let Some(&id) = vocab.get(s) {
                sc.ids.push(id);
                continue;
            }
            for &b in s.as_bytes() {
                let id = self.fallback_ids[b as usize];
                sc.ids.push(if id == NONE { self.unk } else { id });
            }
        }
        self.bpe_ids(out, sc);
        out.len() < max_tokens
    }

    fn bpe_ids(&self, out: &mut Vec<u32>, sc: &mut Scratch) {
        if sc.ids.len() <= SCAN_MAX {
            self.merge_scan(sc);
        } else {
            self.merge_heap(sc);
        }
        out.extend_from_slice(&sc.ids);
    }

    /// Repeatedly merge the lowest-ranked adjacent pair, leftmost first. This is what
    /// tokenizers' heap does, because an entry at a position is only live while that exact pair
    /// is still there.
    fn merge_scan(&self, sc: &mut Scratch) {
        let (ids, pairs) = (&mut sc.ids, &mut sc.pairs);
        if ids.len() < 2 {
            return;
        }
        pairs.clear();
        pairs.extend(ids.windows(2).map(|w| self.merges.get(w[0], w[1])));
        loop {
            let (mut best, mut at) = (u64::MAX, 0);
            for (i, &p) in pairs.iter().enumerate() {
                if p < best {
                    (best, at) = (p, i);
                }
            }
            if best == u64::MAX {
                return;
            }
            ids[at] = best as u32;
            ids.remove(at + 1);
            pairs.remove(at);
            if at > 0 {
                pairs[at - 1] = self.merges.get(ids[at - 1], ids[at]);
            }
            if at < pairs.len() {
                pairs[at] = self.merges.get(ids[at], ids[at + 1]);
            }
        }
    }

    /// tokenizers' Word::merge_all: a priority queue of (rank, position) over a linked list.
    fn merge_heap(&self, sc: &mut Scratch) {
        use std::cmp::Reverse;
        let Scratch { ids, prev, next, heap, .. } = sc;
        let n = ids.len();
        prev.clear();
        prev.extend((0..n as u32).map(|i| i.wrapping_sub(1)));
        next.clear();
        next.extend(1..=n as u32);
        heap.clear();
        let entry = |m: u64, pos: usize| Reverse(((m >> 32) as u32, pos as u32, m as u32));
        for i in 0..n - 1 {
            let m = self.merges.get(ids[i], ids[i + 1]);
            if m != u64::MAX {
                heap.push(entry(m, i));
            }
        }
        while let Some(Reverse((_, pos, new_id))) = heap.pop() {
            let pos = pos as usize;
            let nx = next[pos] as usize;
            if ids[pos] == NONE || nx >= n {
                continue;
            }
            let m = self.merges.get(ids[pos], ids[nx]);
            if m == u64::MAX || m as u32 != new_id {
                continue;
            }
            ids[pos] = new_id;
            ids[nx] = NONE;
            let nn = next[nx] as usize;
            next[pos] = nn as u32;
            if nn < n {
                prev[nn] = pos as u32;
            }
            let p = prev[pos];
            if p != NONE {
                let m = self.merges.get(ids[p as usize], new_id);
                if m != u64::MAX {
                    heap.push(entry(m, p as usize));
                }
            }
            if nn < n {
                let m = self.merges.get(new_id, ids[nn]);
                if m != u64::MAX {
                    heap.push(entry(m, pos));
                }
            }
        }
        ids.retain(|&id| id != NONE);
    }

    /// The id of a token given as tokenizer.json spells it: an added token's content, or a vocab
    /// key in the byte-level alphabet (`Ġthe`).
    pub fn token_id(&self, content: &str) -> Option<u32> {
        if let Some(a) = self.added.iter().find(|a| a.content == content) {
            return Some(a.id);
        }
        if let Some(ids) = &self.vocab_ids {
            return ids.get(content).copied();
        }
        let raw: Option<Vec<u8>> = content.chars().map(byte_level_byte).collect();
        (0..self.kinds.len())
            .find(|&id| match self.kinds[id] {
                KIND_BYTES => raw.as_deref() == Some(self.entry(id)),
                KIND_LITERAL => self.entry(id) == content.as_bytes(),
                _ => false,
            })
            .map(|id| id as u32)
    }

    /// The content of the mask token, which the SDK scrubs out of the text it tokenizes.
    pub(crate) fn mask_token(&self) -> &str {
        self.added.iter().find(|a| a.id == self.mask).map_or("[MASK]", |a| a.content.as_str())
    }

    /// Ids run from 0 to vocab_size() - 1, added tokens included.
    pub fn vocab_size(&self) -> usize {
        self.kinds.len()
    }

    // The BERT-style special tokens, u32::MAX for a tokenizer without them (Qwen: use token_id).

    pub fn cls_id(&self) -> u32 {
        self.cls
    }

    pub fn sep_id(&self) -> u32 {
        self.sep
    }

    pub fn mask_id(&self) -> u32 {
        self.mask
    }

    pub fn pad_id(&self) -> u32 {
        self.pad
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy byte-level vocab with random merges over "abc", as tokenizer.json.
    fn toy() -> Tokenizer {
        let byte_char = |b: u8| match NON_PRINTABLE.iter().position(|&x| x == b) {
            Some(i) => char::from_u32(0x100 + i as u32).unwrap(),
            None => b as char,
        };
        let mut vocab: Vec<String> = (0..=255u8).map(|b| byte_char(b).to_string()).collect();
        let mut merges = Vec::new();
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rand = |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        let mut pool: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();
        while merges.len() < 80 {
            let (x, y) = (pool[rand(pool.len())].clone(), pool[rand(pool.len())].clone());
            let xy = format!("{x}{y}");
            if !pool.contains(&xy) {
                merges.push(Value::Array(vec![Value::Str(x), Value::Str(y)]));
                vocab.push(xy.clone());
                pool.push(xy);
            }
        }
        for s in ["[CLS]", "[SEP]", "[MASK]", "[PAD]"] {
            vocab.push(s.to_string());
        }
        let vocab = vocab.into_iter().enumerate().map(|(i, k)| (k, Value::Int(i.to_string()))).collect();
        let model = Value::Object(vec![
            ("type".into(), Value::Str("BPE".into())),
            ("vocab".into(), Value::Object(vocab)),
            ("merges".into(), Value::Array(merges)),
        ]);
        let pre = Value::parse(r#"{"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}"#).unwrap();
        let root = Value::Object(vec![("pre_tokenizer".into(), pre), ("model".into(), model)]);
        Tokenizer::from_hf_json(&root.to_json()).unwrap()
    }

    /// Packs built before version 2 keep loading: rewrite a blob's merges the way version 1 did
    /// (rank order, no rank field).
    #[test]
    fn loads_version_1_blobs() {
        let tok = toy();
        let v2 = tok.to_bytes();
        let n = tok.kinds.len();
        let merges_at = 42 + 3 * n + tok.bytes.len();
        let mut merges: Vec<[u32; 4]> = tok.merges.entries().collect();
        merges.sort_by_key(|m| m[2]);
        let mut v1 = v2[..merges_at].to_vec();
        v1[8..12].copy_from_slice(&1u32.to_le_bytes());
        for m in &merges {
            for v in [m[0], m[1], m[3]] {
                v1.extend_from_slice(&(v as u16).to_le_bytes());
            }
        }
        v1.extend_from_slice(&v2[merges_at + merges.len() * 8..]);
        assert_eq!(Tokenizer::from_bytes(&v1).unwrap().to_bytes(), v2);
    }

    #[test]
    fn heap_merges_like_scan() {
        let tok = toy();
        let mut sc = Scratch::default();
        let mut seed = 7u32;
        for len in 2..160 {
            let word: Vec<u8> = (0..len)
                .map(|_| {
                    seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    b"abc"[(seed >> 16) as usize % 3]
                })
                .collect();
            let fill = |sc: &mut Scratch| {
                sc.ids.clear();
                sc.ids.extend(word.iter().map(|&b| tok.byte_ids[b as usize]));
            };
            fill(&mut sc);
            tok.merge_scan(&mut sc);
            let scan = sc.ids.clone();
            fill(&mut sc);
            tok.merge_heap(&mut sc);
            assert_eq!(sc.ids, scan, "{}", String::from_utf8_lossy(&word));
        }
    }
}
