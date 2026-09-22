//! The Unicode pieces of the Hugging Face tokenizer pipeline: the character classes of the
//! pre-tokenizer regexes and NFC normalization.
//!
//! Each follows the Unicode version tokenizers actually uses (see tools/gen_unicode.py): `\p{L}`,
//! `\p{N}` and `\p{M}` are Unicode 16.0, NFC is Unicode 9.0.

use std::borrow::Cow;

use crate::unicode_tables as t;

fn in_ranges(r: &[(u32, u32)], u: u32) -> bool {
    let i = r.partition_point(|&(lo, _)| lo <= u);
    i > 0 && u <= r[i - 1].1
}

/// General_Category L (Oniguruma `\p{L}`).
#[inline]
pub fn is_letter(c: char) -> bool {
    let u = c as u32;
    if u < 0x80 {
        return (u | 0x20).wrapping_sub('a' as u32) < 26;
    }
    in_ranges(t::LETTER, u)
}

/// General_Category N (Oniguruma `\p{N}`).
#[inline]
pub fn is_number(c: char) -> bool {
    let u = c as u32;
    if u < 0x80 {
        return u.wrapping_sub('0' as u32) < 10;
    }
    in_ranges(t::NUMBER, u)
}

/// General_Category M (Oniguruma `\p{M}`).
#[inline]
pub fn is_mark(c: char) -> bool {
    (c as u32) >= 0x300 && in_ranges(t::MARK, c as u32)
}

/// Oniguruma `\s`: Unicode White_Space.
#[inline]
pub fn is_space(c: char) -> bool {
    matches!(
        c as u32,
        0x09..=0x0d | 0x20 | 0x85 | 0xa0 | 0x1680 | 0x2000..=0x200a | 0x2028 | 0x2029 | 0x202f | 0x205f | 0x3000
    )
}

const S_BASE: u32 = 0xac00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11a7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = V_COUNT * T_COUNT;
const S_COUNT: u32 = L_COUNT * N_COUNT;

fn ccc(c: char) -> u8 {
    let u = c as u32;
    if u < 0x300 {
        return 0;
    }
    let i = t::CCC.partition_point(|&(lo, _, _)| lo <= u);
    if i > 0 && u <= t::CCC[i - 1].1 {
        t::CCC[i - 1].2
    } else {
        0
    }
}

fn decompose(c: char, out: &mut Vec<(char, u8)>) {
    let u = c as u32;
    if u < 0xc0 {
        out.push((c, ccc(c)));
        return;
    }
    if (S_BASE..S_BASE + S_COUNT).contains(&u) {
        let s = u - S_BASE;
        let jamo = |v| char::from_u32(v).unwrap();
        out.push((jamo(L_BASE + s / N_COUNT), 0));
        out.push((jamo(V_BASE + (s % N_COUNT) / T_COUNT), 0));
        if s % T_COUNT != 0 {
            out.push((jamo(T_BASE + s % T_COUNT), 0));
        }
        return;
    }
    match t::DECOMP.binary_search_by_key(&u, |&(k, _)| k) {
        Ok(i) => {
            let p = t::DECOMP[i].1 as usize;
            for &d in &t::DECOMP_CHARS[p >> 2..(p >> 2) + (p & 3) + 1] {
                out.push((d, ccc(d)));
            }
        }
        Err(_) => out.push((c, ccc(c))),
    }
}

fn compose(a: char, b: char) -> Option<char> {
    let (a, b) = (a as u32, b as u32);
    if (L_BASE..L_BASE + L_COUNT).contains(&a) && (V_BASE..V_BASE + V_COUNT).contains(&b) {
        return char::from_u32(S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT);
    }
    if (S_BASE..S_BASE + S_COUNT).contains(&a)
        && (a - S_BASE) % T_COUNT == 0
        && (T_BASE + 1..T_BASE + T_COUNT).contains(&b)
    {
        return char::from_u32(a + (b - T_BASE));
    }
    let i = t::COMPOSE.binary_search_by(|&(x, y, _)| (x, y).cmp(&(a, b))).ok()?;
    char::from_u32(t::COMPOSE[i].2)
}

/// Whether NFC leaves `s` unchanged, by the quick check: no character that NFC could alter or
/// combine with a neighbour.
fn is_nfc_quick(s: &str) -> bool {
    s.chars().all(|c| (c as u32) < 0x300 || !in_ranges(t::NFC_MAYBE, c as u32))
}

/// Unicode NFC (canonical decomposition, canonical ordering, canonical composition).
pub fn nfc(s: &str) -> String {
    nfc_cow(s).into_owned()
}

/// NFC that borrows the input when it is already normalized, the common case.
pub fn nfc_cow(s: &str) -> Cow<'_, str> {
    if s.bytes().all(|b| b < 0xcc) || is_nfc_quick(s) {
        // every byte below 0xCC means every char is below U+0300
        return Cow::Borrowed(s);
    }
    let mut d: Vec<(char, u8)> = Vec::with_capacity(s.len());
    for c in s.chars() {
        decompose(c, &mut d);
    }
    // canonical ordering: stable sort of each run of nonstarters by combining class
    let mut i = 0;
    while i < d.len() {
        if d[i].1 == 0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < d.len() && d[i].1 != 0 {
            i += 1;
        }
        if i - start > 1 {
            d[start..i].sort_by_key(|&(_, k)| k);
        }
    }
    // Canonical composition, the same state machine as unicode-normalization's Recompositions:
    // `starter` is the pending composee, `last` the class of the latest char kept after it
    // (None while nothing sits between them).
    let mut out = String::with_capacity(s.len());
    let mut starter: Option<char> = None;
    let mut last: Option<u8> = None;
    let mut between: Vec<char> = Vec::new();
    let flush = |out: &mut String, starter: &mut Option<char>, between: &mut Vec<char>| {
        if let Some(c) = starter.take() {
            out.push(c);
        }
        out.extend(between.drain(..));
    };
    for &(c, k) in &d {
        let Some(st) = starter else {
            if k == 0 {
                starter = Some(c);
            } else {
                out.push(c);
            }
            continue;
        };
        let blocked = matches!(last, Some(l) if l >= k);
        if !blocked {
            if let Some(r) = compose(st, c) {
                starter = Some(r);
                continue;
            }
        }
        if k == 0 {
            flush(&mut out, &mut starter, &mut between);
            starter = Some(c);
            last = None;
        } else {
            between.push(c);
            last = Some(k);
        }
    }
    flush(&mut out, &mut starter, &mut between);
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes() {
        assert!(is_letter('a') && is_letter('Z') && is_letter('é') && is_letter('日') && is_letter('\u{1c89}'));
        assert!(!is_letter('1') && !is_letter(' ') && !is_letter('\u{301}') && !is_letter('_'));
        assert!(is_number('7') && is_number('½') && is_number('٣') && is_number('Ⅻ') && !is_number('x'));
        assert!(is_mark('\u{301}') && is_mark('\u{93f}') && is_mark('\u{e31}') && !is_mark('a') && !is_mark('क'));
        assert!(is_space('\t') && is_space('\u{a0}') && is_space('\u{3000}') && is_space('\u{85}'));
        assert!(!is_space('\u{200b}') && !is_space('\u{180e}') && !is_space('\u{feff}') && !is_space('x'));
    }

    #[test]
    fn nfc_cases() {
        let cases = [
            ("abc", "abc"),
            ("Cafe\u{301}", "Café"),
            ("\u{1e9b}\u{323}", "\u{1e9b}\u{323}"),
            ("a\u{323}\u{302}", "\u{1ead}"),
            ("a\u{302}\u{323}", "\u{1ead}"),
            ("\u{212b}", "\u{c5}"),
            ("\u{1100}\u{1161}\u{11a8}", "\u{ac01}"),
            ("\u{ac00}\u{11a8}", "\u{ac01}"),
            ("\u{301}a", "\u{301}a"),
            ("e\u{301}\u{301}", "\u{e9}\u{301}"),
            ("\u{b47}\u{b3e}", "\u{b4b}"),
            ("\u{344}", "\u{308}\u{301}"),
            // U+1DF9 is Unicode 10: a plain starter to tokenizers, so nothing reorders or composes
            ("a\u{1df9}\u{301}", "a\u{1df9}\u{301}"),
        ];
        for (s, want) in cases {
            assert_eq!(nfc(s), want, "{s:?}");
        }
        assert!(matches!(nfc_cow("plain ascii, é"), Cow::Borrowed(_)));
    }
}
