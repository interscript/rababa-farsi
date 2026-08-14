"""Convert real Persian G2P data (HomoRich) to JSONL src/tgt format.

Source: MahtaFetrat/HomoRich-G2P-Persian (528,875 sentences)
  Grapheme (Persian sentence) -> Phoneme (Latin phonemes)

Output: data/train.jsonl, val.jsonl, test.jsonl
"""
from __future__ import annotations

import csv
import io
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq


def clean(s: str) -> str:
    return " ".join(s.strip().split())


def main(parquet_path: str, out_dir: str, val_frac: float = 0.02, test_frac: float = 0.02) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading {parquet_path}...", flush=True)
    table = pq.read_table(parquet_path)
    n = table.num_rows
    print(f"  rows: {n}", flush=True)

    graphemes = table.column("Grapheme").to_pylist()
    phonemes = table.column("Phoneme").to_pylist()

    pairs = []
    skipped = 0
    for g, p in zip(graphemes, phonemes):
        g = clean(g) if g else ""
        p = clean(p) if p else ""
        if not g or not p or len(g) < 2 or len(p) < 2:
            skipped += 1
            continue
        if len(g.encode("utf-8")) > 512 or len(p.encode("utf-8")) > 512:
            skipped += 1
            continue
        pairs.append({"src": g, "tgt": p})

    print(f"  kept: {len(pairs)}, skipped: {skipped}", flush=True)

    rng = random.Random(42)
    rng.shuffle(pairs)

    n_val = int(len(pairs) * val_frac)
    n_test = int(len(pairs) * test_frac)
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
