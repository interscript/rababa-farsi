"""Re-score Persian v5 (Mapped-Phoneme repr) on SentenceBench with a
data-derived decoding table.

Why: v5 was trained on HomoRich's `Mapped Phoneme` representation
("m|" prefix, lowercase, / $ @ 1 markers). Its raw SentenceBench score
(28.22%) compared mapped strings against plain-phoneme references — a
representation artifact, not model quality.

Fix: build a word-level Mapped -> Plain table from the HomoRich train
split (rows carry both fields, word-aligned). Majority vote on
collisions; identity fallback. Then score the decoded predictions with
the identical protocol as the v1 eval (exact + ezafe-normalized, beam 4).

Usage:
    modal run eval_persian_v5_rescore.py
"""

from __future__ import annotations

import csv
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import modal

APP_NAME = "persian-g2p"
checkpoints_volume = modal.Volume.from_name(f"{APP_NAME}-checkpoints", create_if_missing=True)
datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)

CKPT = "/checkpoints/persian_g2p_v5_ge2pe/run-001/best"
DATA_ROOT = Path("/datasets/persian-g2p-v3")
SENTENCEBENCH_URL = (
    "https://huggingface.co/datasets/MahtaFetrat/SentenceBench/resolve/main/SentenceBench.csv"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4,<3",
        "transformers>=4.40,<5",
        "sentencepiece",
        "accelerate>=1.1.0",
        "numpy>=1.26,<3",
        "tqdm>=4.66",
    )
)

app = modal.App(name=f"{APP_NAME}-v5-rescore", image=image)


@app.function(gpu="A10G", timeout=60 * 60,
              volumes={"/checkpoints": checkpoints_volume, "/datasets": datasets_volume})
def evaluate() -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    checkpoints_volume.reload()
    datasets_volume.reload()

    # ---- 1. decoding table from HomoRich train+val+test ----
    votes: dict[str, Counter] = defaultdict(Counter)
    n_rows = n_aligned = 0
    for split in ("train", "val", "test"):
        p = DATA_ROOT / f"{split}.jsonl"
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            n_rows += 1
            plain = row["tgt"].split()
            mapped = row["tgt_mapped"].split()
            if len(plain) != len(mapped):
                continue
            n_aligned += 1
            for m, t in zip(mapped, plain):
                votes[m][t] += 1
    table = {m: c.most_common(1)[0][0] for m, c in votes.items()}
    collisions = sum(1 for c in votes.values() if len(c) > 1)
    print(f"[table] rows={n_rows} aligned={n_aligned} entries={len(table)} "
          f"collisions={collisions}", flush=True)

    def decode(tok: str) -> str:
        if tok in table:
            return table[tok]
        stripped = tok.replace("1", "")
        return table.get(stripped, stripped)

    # ---- 2. SentenceBench generation ----
    local_csv = Path("/tmp/SentenceBench.csv")
    urllib.request.urlretrieve(SENTENCEBENCH_URL, local_csv)
    with local_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    hg_rows = [
        r for r in rows
        if (r.get("homograph word") or "").strip() and (r.get("pronunciation") or "").strip()
    ]
    print(f"[sb] homograph rows: {len(hg_rows)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSeq2SeqLM.from_pretrained(CKPT).to("cuda")
    model.eval()

    srcs = [r["grapheme"].strip() for r in hg_rows]
    preds = []
    with torch.no_grad():
        for i in range(0, len(srcs), 32):
            batch = ["m|" + s for s in srcs[i : i + 32]]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=256).to("cuda")
            gen = model.generate(**enc, max_new_tokens=256, num_beams=4)
            preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    print(f"[sb] generated {len(preds)}", flush=True)

    # ---- 3. score with identical protocol ----
    correct = correct_norm = total = unaligned = oov = 0
    samples = []
    for row, pred in zip(hg_rows, preds):
        grapheme = row["grapheme"].strip()
        hg_word = row["homograph word"].strip()
        pron = row["pronunciation"].strip()

        g_words = grapheme.split()
        p_words = [decode(w) for w in pred.strip().split()]
        for w in pred.strip().split():
            if w not in table:
                oov += 1

        positions = [i for i, w in enumerate(g_words) if hg_word in w]
        if not positions:
            continue
        pos = positions[0]
        if pos >= len(p_words):
            unaligned += 1
            continue

        pred_pron = p_words[pos]
        ok = pred_pron == pron
        ok_norm = ok or pred_pron.rstrip("e") == pron or pred_pron == pron.rstrip("e")
        correct += int(ok)
        correct_norm += int(ok_norm)
        total += 1
        if len(samples) < 12:
            samples.append({"homograph": hg_word, "ref": pron, "pred_mapped_raw":
                            pred.strip().split()[pos] if pos < len(pred.strip().split()) else "",
                            "pred_decoded": pred_pron, "ok": ok, "ok_norm": ok_norm})

    ha = correct / max(1, total)
    ha_norm = correct_norm / max(1, total)
    result = {
        "model": CKPT,
        "homograph_accuracy_decoded": ha,
        "homograph_accuracy_ezafe_normalized": ha_norm,
        "n_scored": total,
        "n_unaligned": unaligned,
        "n_decode_oov_tokens": oov,
        "table_entries": len(table),
        "table_collisions": collisions,
        "samples": samples,
    }
    print(f"=== v5 rescore HA (decoded): {ha:.4f} ({correct}/{total}) ===", flush=True)
    print(f"=== v5 rescore HA (ezafe-norm): {ha_norm:.4f} ===", flush=True)
    print(json.dumps(samples[:6], indent=2, ensure_ascii=False), flush=True)

    out = Path("/checkpoints/persian_g2p_v5_ge2pe/run-001/sentencebench_rescored.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    checkpoints_volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(evaluate.remote(), indent=2, ensure_ascii=False))
