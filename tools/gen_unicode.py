#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14,<3.15"
# ///
"""Generate crates/kevala/src/unicode_tables.rs from Python's `unicodedata`.

The tables reproduce what the Hugging Face `tokenizers` library sees, which is two different
Unicode versions:

  LETTER, NUMBER,  General_Category L*, N* and M*, for `\\p{L}`, `\\p{N}` and `\\p{M}` in the
  MARK             pre-tokenizer regexes (GPT-2, Qwen2). tokenizers runs them on Oniguruma 6.9.10,
                   whose tables are Unicode 16.0, the version of Python 3.14's `unicodedata`
                   (hence the pinned Python).
  CCC, DECOMP,     NFC data. tokenizers normalizes with `unicode-normalization-alignments` 0.1.12,
  COMPOSE,         which ships Unicode 9.0 tables: characters assigned later are plain starters
  NFC_MAYBE        there (combining class 0, no decomposition). So the NFC tables drop the code
                   points listed in POST_9_NFC.

Usage: uv run tools/gen_unicode.py
"""
import os
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "crates", "kevala", "src", "unicode_tables.rs")

UNICODE_VERSION = "16.0.0"

# Every code point assigned after Unicode 9.0 that carries NFC data (a combining class or a
# canonical decomposition) in Unicode 16.0. Removing them makes the NFC tables equal, entry for
# entry, to those of unicode-normalization-alignments 0.1.12 (Unicode 9.0).
POST_9_NFC = [
    (0x07FD, 0x07FD), (0x0897, 0x089F), (0x08CA, 0x08D3), (0x09FE, 0x09FE), (0x0C3C, 0x0C3C),
    (0x0D3B, 0x0D3C), (0x0EBA, 0x0EBA), (0x1715, 0x1715), (0x1ABF, 0x1ACE), (0x1DF6, 0x1DFA),
    (0xA82C, 0xA82C), (0x105C9, 0x105C9), (0x105E4, 0x105E4), (0x10D24, 0x10D27),
    (0x10D69, 0x10D6D), (0x10EAB, 0x10EAC), (0x10EFD, 0x10EFF), (0x10F46, 0x10F50),
    (0x10F82, 0x10F85), (0x11070, 0x11070), (0x1133B, 0x1133B), (0x11383, 0x11383),
    (0x11385, 0x11385), (0x1138E, 0x1138E), (0x11391, 0x11391), (0x113C5, 0x113C5),
    (0x113C7, 0x113C8), (0x113CE, 0x113D0), (0x1145E, 0x1145E), (0x11839, 0x1183A),
    (0x11938, 0x11938), (0x1193D, 0x1193E), (0x11943, 0x11943), (0x119E0, 0x119E0),
    (0x11A34, 0x11A34), (0x11A47, 0x11A47), (0x11A99, 0x11A99), (0x11D42, 0x11D42),
    (0x11D44, 0x11D45), (0x11D97, 0x11D97), (0x11F41, 0x11F42), (0x16121, 0x16128),
    (0x1612F, 0x1612F), (0x16D68, 0x16D6A), (0x16FF0, 0x16FF1), (0x1E08F, 0x1E08F),
    (0x1E130, 0x1E136), (0x1E2AE, 0x1E2AE), (0x1E2EC, 0x1E2EF), (0x1E4EC, 0x1E4EF),
    (0x1E5EE, 0x1E5EF),
]

HANGUL_S = range(0xAC00, 0xAC00 + 11172)
HANGUL_V = range(0x1161, 0x1161 + 21)
HANGUL_T = range(0x11A8, 0x11A7 + 28)


def ranges(cps):
    out = []
    for cp in sorted(cps):
        if out and out[-1][1] == cp - 1:
            out[-1][1] = cp
        else:
            out.append([cp, cp])
    return out


def main():
    if unicodedata.unidata_version != UNICODE_VERSION:
        sys.exit("need unicodedata %s (Python 3.14), got %s" % (UNICODE_VERSION, unicodedata.unidata_version))
    post9 = {cp for a, b in POST_9_NFC for cp in range(a, b + 1)}
    chars = [chr(cp) for cp in range(0x110000)]

    letters = [cp for cp, c in enumerate(chars) if unicodedata.category(c)[0] == "L"]
    numbers = [cp for cp, c in enumerate(chars) if unicodedata.category(c)[0] == "N"]
    marks = [cp for cp, c in enumerate(chars) if unicodedata.category(c)[0] == "M"]

    ccc = {cp: unicodedata.combining(c) for cp, c in enumerate(chars) if unicodedata.combining(c) and cp not in post9}

    def canonical(cp):
        d = unicodedata.decomposition(chars[cp])
        return [int(x, 16) for x in d.split()] if d and not d.startswith("<") else None

    def full(cp):
        d = canonical(cp)
        return [x for p in d for x in full(p)] if d else [cp]

    decomp, compose = {}, {}
    for cp in range(0x110000):
        if cp in post9 or cp in HANGUL_S:
            continue
        d = canonical(cp)
        if not d:
            continue
        decomp[cp] = full(cp)
        # a primary composite: two-char decomposition that NFC recomposes (not excluded)
        if len(d) == 2 and unicodedata.normalize("NFC", chr(d[0]) + chr(d[1])) == chars[cp]:
            compose[(d[0], d[1])] = cp
    assert max(len(v) for v in decomp.values()) <= 4
    assert all(a not in post9 and b not in post9 for a, b in compose)

    # Characters that can change under NFC or change a neighbour: nonzero combining class,
    # NFC_Quick_Check=No (decomposable but not a primary composite), or the second half of a
    # composition (NFC_Quick_Check=Maybe). A string with none of them is already NFC.
    composites = set(compose.values())
    seconds = {b for _, b in compose} | set(HANGUL_V) | set(HANGUL_T)
    maybe = set(ccc) | {cp for cp in decomp if cp not in composites} | seconds
    # nfc() also returns early when every char is below U+0300
    assert min(maybe) >= 0x300 and all(a >= 0x300 or b >= 0x300 for a, b in compose)

    ccc_ranges = []
    for cp in sorted(ccc):
        if ccc_ranges and ccc_ranges[-1][1] == cp - 1 and ccc_ranges[-1][2] == ccc[cp]:
            ccc_ranges[-1][1] = cp
        else:
            ccc_ranges.append([cp, cp, ccc[cp]])

    flat, decomp_rows = [], []
    for cp in sorted(decomp):
        decomp_rows.append((cp, len(flat) << 2 | (len(decomp[cp]) - 1)))
        flat.extend(decomp[cp])

    def rows(items, fmt, per_line):
        lines = []
        for i in range(0, len(items), per_line):
            lines.append("    " + " ".join(fmt(x) + "," for x in items[i : i + per_line]))
        return "\n".join(lines)

    pair = lambda r: "(0x%X, 0x%X)" % (r[0], r[1])
    out = [
        "//! Generated by tools/gen_unicode.py from Python %s `unicodedata` %s. Do not edit."
        % (sys.version.split()[0], unicodedata.unidata_version),
        "//!",
        "//! LETTER, NUMBER and MARK are Unicode %s (Oniguruma's tables in tokenizers). The NFC tables are"
        % unicodedata.unidata_version,
        "//! restricted to Unicode 9.0 (unicode-normalization-alignments in tokenizers).",
        "",
        "/// General_Category L*, inclusive ranges.",
        "#[rustfmt::skip]",
        "pub(crate) static LETTER: &[(u32, u32)] = &[",
        rows(ranges(letters), pair, 7),
        "];",
        "",
        "/// General_Category N*, inclusive ranges.",
        "#[rustfmt::skip]",
        "pub(crate) static NUMBER: &[(u32, u32)] = &[",
        rows(ranges(numbers), pair, 7),
        "];",
        "",
        "/// General_Category M*, inclusive ranges.",
        "#[rustfmt::skip]",
        "pub(crate) static MARK: &[(u32, u32)] = &[",
        rows(ranges(marks), pair, 7),
        "];",
        "",
        "/// Nonzero canonical combining classes: (first, last, class).",
        "#[rustfmt::skip]",
        "pub(crate) static CCC: &[(u32, u32, u8)] = &[",
        rows(ccc_ranges, lambda r: "(0x%X, 0x%X, %d)" % tuple(r), 6),
        "];",
        "",
        "/// Full canonical decompositions (Hangul excluded): (char, offset << 2 | (len - 1)) into",
        "/// DECOMP_CHARS.",
        "#[rustfmt::skip]",
        "pub(crate) static DECOMP: &[(u32, u32)] = &[",
        rows(decomp_rows, pair, 7),
        "];",
        "",
        "#[rustfmt::skip]",
        "pub(crate) static DECOMP_CHARS: &[char] = &[",
        rows(flat, lambda cp: "'\\u{%X}'" % cp, 10),
        "];",
        "",
        "/// Canonical composition pairs, sorted (Hangul excluded): (first, second, composite).",
        "#[rustfmt::skip]",
        "pub(crate) static COMPOSE: &[(u32, u32, u32)] = &[",
        rows(sorted((a, b, c) for (a, b), c in compose.items()), lambda r: "(0x%X, 0x%X, 0x%X)" % r, 5),
        "];",
        "",
        "/// Characters that NFC may change or that may combine with a neighbour, inclusive ranges.",
        "#[rustfmt::skip]",
        "pub(crate) static NFC_MAYBE: &[(u32, u32)] = &[",
        rows(ranges(maybe), pair, 7),
        "];",
        "",
    ]
    with open(OUT, "w") as f:
        f.write("\n".join(out))
    counts = [len(ranges(x)) for x in (letters, numbers, marks)] + [len(ccc_ranges), len(decomp), len(compose)]
    print("wrote %s: %d letter, %d number, %d mark, %d ccc ranges, %d decompositions, %d compositions"
          % tuple([OUT] + counts))


if __name__ == "__main__":
    main()
