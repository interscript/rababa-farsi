"""Evaluate our Persian G2P model on SentenceBench (HomoRich official test).

Protocol (mimics "Fast, Not Fancy" arXiv:2505.12973):
- For each sentence containing a homograph, generate the full phoneme output.
- Align the homograph word by space position; the predicted phoneme at that
  position must equal the reference pronunciation.
- Homograph Accuracy (HA) = fraction correct.

Usage:
    modal run eval_persian_sentencebench.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import modal

APP_NAME = "persian-g2p"
checkpoints_volume = modal.Volume.from_name(f"{APP_NAME}-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "curl")
    .pip_install(
        "torch>=2.4,<3",
        "transformers>=4.40,<5",
        "sentencepiece",
        "protobuf",
        "accelerate>=1.1.0",
        "numpy>=1.26,<3",
        "tqdm>=4.66",
    )
    .add_local_dir("src", "/opt/persian/src", copy=True)
    .workdir("/opt/persian")
    .env({"PYTHONPATH": "/opt/persian/src"})
)

app = modal.App(name=f"{APP_NAME}-sentencebench", image=image)

SENTENCEBENCH_URL = (
    "https://huggingface.co/datasets/MahtaFetrat/SentenceBench/resolve/main/SentenceBench.csv"
)


@app.function(
    gpu="A10G",
    timeout=60 * 60,
    volumes={"/checkpoints": checkpoints_volume},
)
def evaluate_sentencebench() -> dict:
    """Run our Persian G2P on SentenceBench, compute homograph accuracy."""
    import urllib.request

    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    checkpoints_volume.reload()

    # Download SentenceBench
    local_csv = Path("/tmp/SentenceBench.csv")
    urllib.request.urlretrieve(SENTENCEBENCH_URL, local_csv)
    with local_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"[sb] rows: {len(rows)}", flush=True)

    subsets = {}
    for r in rows:
        subsets[r["dataset"]] = subsets.get(r["dataset"], 0) + 1
    print(f"[sb] subsets: {subsets}", flush=True)

    # Homograph rows
    hg_rows = [
        r for r in rows if (r.get("homograph word") or "").strip() and (r.get("pronunciation") or "").strip()
    ]
    print(f"[sb] homograph rows: {len(hg_rows)}", flush=True)

    device = torch.device("cuda")
    ckpt = "/checkpoints/persian_g2p/run-001/best"
    if not Path(ckpt).is_dir():
        return {"error": f"checkpoint {ckpt} not found"}
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(device)
    model.eval()

    # Generate predictions
    srcs = [r["grapheme"].strip() for r in hg_rows]
    preds = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(srcs), batch_size):
            batch = srcs[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            gen = model.generate(**enc, max_new_tokens=256, num_beams=4)
            preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    print(f"[sb] generated {len(preds)} predictions", flush=True)

    # Homograph accuracy by word-position alignment
    correct = 0
    correct_norm = 0
    total = 0
    unaligned = 0
    samples = []
    for row, pred in zip(hg_rows, preds):
        grapheme = row["grapheme"].strip()
        hg_word = row["homograph word"].strip()
        pron = row["pronunciation"].strip()

        g_words = grapheme.split()
        p_words = pred.strip().split()

        # find homograph position(s)
        positions = [i for i, w in enumerate(g_words) if hg_word in w]
        if not positions:
            continue

        pos = positions[0]
        if pos >= len(p_words):
            unaligned += 1
            continue

        pred_pron = p_words[pos]
        ok = pred_pron == pron
        # Ezafe normalization: our model attaches the ezafe /e/ to the word
        # (qadr-e -> qadre); reference pronunciations are bare stems.
        ok_norm = pred_pron == pron or pred_pron.rstrip("e") == pron or pred_pron == pron.rstrip("e")
        correct += int(ok)
        correct_norm += int(ok_norm)
        total += 1
        if len(samples) < 10:
            samples.append(
                {
                    "grapheme": grapheme,
                    "homograph": hg_word,
                    "ref_pron": pron,
                    "pred_pron": pred_pron,
                    "ok": ok,
                    "ok_norm": ok_norm,
                }
            )

    ha = correct / max(1, total)
    ha_norm = correct_norm / max(1, total)
    result = {
        "homograph_accuracy": ha,
        "homograph_accuracy_ezafe_normalized": ha_norm,
        "n_scored": total,
        "n_unaligned": unaligned,
        "n_rows": len(rows),
        "n_homograph_rows": len(hg_rows),
        "model": ckpt,
        "samples": samples,
    }
    print(f"=== SentenceBench Homograph Accuracy: {ha:.4f} ({correct}/{total}) ===", flush=True)
    print(f"=== SentenceBench HA (ezafe-normalized): {ha_norm:.4f} ({correct_norm}/{total}) ===", flush=True)
    print(json.dumps(samples[:5], indent=2, ensure_ascii=False), flush=True)
    return result


@app.local_entrypoint()
def main():
    result = evaluate_sentencebench.remote()
    print(json.dumps(result, indent=2, ensure_ascii=False))
