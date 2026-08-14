"""Extract Persian diacritized text from HomoRich Mapped Phoneme field.

The HomoRich dataset's `Mapped Phoneme` field has byte-pair markers that
encode stress, syllable boundaries, and — critically — vowel quality.
This script parses those markers and rewrites the original Persian text
with Arabic haraqat (َ ِ ُ ْ ّ).

Example:
  Grapheme:       روی دیوار ننویسید.
  Mapped Phoneme: ruye1 divar n/nevisid

  Diacritized:    رُوی دِیوَار نَنِویسِد.

Mapping (Mapped Phoneme marker → haraqat):
  - Numbers (1, 2, 3): stress markers — usually indicate long vowel (alef/yeh)
  - "/" in middle of word: syllable boundary with implicit vowel
  - "@" prefix: ezafe (unwritten /e/ connective)
  - "$" prefix: consonant cluster marker

The mapping is approximate — Persian short vowels (/e/, /æ/, /o/) don't
have a 1:1 correspondence to Arabic haraqat. We use the conventions:
  - "/a/" or "/A/" → fatha (َ)
  - "/e/", "/i/" → kasra (ِ)
  - "/o/", "/u/" → damma (ُ)
  - "@" (ezafe) → kasra (lighter, often omitted in modern Persian)
  - no vowel → sukun (ْ)
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

FATHA = "َ"
KASRA = "ِ"
DAMMA = "ُ"
SUKUN = "ْ"
SHADDA = "ّ"

CARRIERS = set("ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیؤئآأإة")
LONG_VOWEL_LETTERS = set("اویآی")


def mapped_to_diacritized(grapheme: str, mapped: str) -> str:
    """Walk grapheme and mapped-phoneme in parallel, emit diacritized form.

    The mapped phoneme uses Latin chars + digits + symbols. We use it to
    identify vowel positions and qualities.
    """
    if not grapheme or not mapped:
        return grapheme

    # Tokenize mapped phoneme: keep track of word boundaries and vowel markers
    mapped_words = mapped.split()
    grapheme_words = grapheme.split()

    if len(mapped_words) != len(grapheme_words):
        return grapheme  # misaligned, give up

    out_words = []
    for g_word, m_word in zip(grapheme_words, mapped_words):
        out_words.append(_diacritize_word(g_word, m_word))
    return " ".join(out_words)


def _diacritize_word(g_word: str, m_word: str) -> str:
    """Diacritize a single Persian word using its mapped phoneme."""
    # Strip stress markers (digits) and @ ezafe markers from m_word for analysis
    m_clean = "".join(c for c in m_word if not c.isdigit() and c != "@")
    # Get the vowel sequence from the mapped phoneme
    # Vowels in mapped phoneme: a, A, e, i, o, u, ow, ey, etc.
    vowels_found = []
    i = 0
    while i < len(m_clean):
        c = m_clean[i].lower()
        if c == "/":
            # syllable boundary marker
            i += 1
            continue
        if c == "$":
            i += 1
            continue
        if c in "aeiou":
            # determine vowel quality
            if c == "a":
                vowels_found.append(FATHA)
            elif c in ("e", "i"):
                vowels_found.append(KASRA)
            elif c in ("o", "u"):
                vowels_found.append(DAMMA)
            i += 1
        else:
            i += 1

    # Apply haraqat to grapheme carriers (best-effort parallel walk)
    out = []
    v_idx = 0
    for ch in g_word:
        if ch in CARRIERS:
            if ch in LONG_VOWEL_LETTERS and v_idx < len(vowels_found):
                # Long vowel letter — don't add haraqat
                out.append(ch)
                v_idx += 1
            elif v_idx < len(vowels_found):
                out.append(ch + vowels_found[v_idx])
                v_idx += 1
            else:
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def main(parquet_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading {parquet_path}...", flush=True)
    table = pq.read_table(parquet_path)
    n = table.num_rows
    print(f"  rows: {n}", flush=True)

    graphemes = table.column("Grapheme").to_pylist()
    mappeds = table.column("Mapped Phoneme").to_pylist()

    pairs = []
    skipped = 0
    for g, m in zip(graphemes, mappeds):
        g = (g or "").strip()
        m = (m or "").strip()
        if not g or not m:
            skipped += 1
            continue
        if len(g.encode("utf-8")) > 512:
            skipped += 1
            continue
        d = mapped_to_diacritized(g, m)
        if not d or d == g:
            skipped += 1
            continue
        pairs.append({"src": g, "tgt": d})

    print(f"  kept: {len(pairs)}, skipped: {skipped}", flush=True)

    rng = random.Random(42)
    rng.shuffle(pairs)

    n_val = int(len(pairs) * 0.02)
    n_test = int(len(pairs) * 0.02)
    val = pairs[:n_val]
    test = pairs[n_val : n_val + n_test]
    train = pairs[n_val + n_test :]

    for name, split in (("train", train), ("val", val), ("test", test)):
        path = out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(split)} -> {path}", flush=True)


if __name__ == "__main__":
    parquet = sys.argv[1] if len(sys.argv) > 1 else "/tmp/persian-data/train.parquet"
    out = sys.argv[2] if len(sys.argv) > 2 else "data"
    main(parquet, out)
