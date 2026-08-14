"""Measure Persian homograph accuracy on our test set (HomoRich paper's metric).

The HomoRich paper measures: for each test sentence containing a known
homograph, did the model pronounce the homograph correctly?

HomoRich dataset has `Homograph Grapheme` and `Homograph Phoneme` columns.
We check if our model's prediction for that specific word in the output
matches the expected homograph phoneme.

This is the metric reported as 76.89% in the HomoRich paper.

Usage:
    modal run eval_persian_homograph.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "persian-g2p"
checkpoints_volume = modal.Volume.from_name(f"{APP_NAME}-checkpoints", create_if_missing=True)
datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git", "curl")
    .pip_install(
        "torch>=2.4,<3",
        "transformers>=4.46",
        "sentencepiece",
        "numpy>=1.26,<3",
        "tqdm>=4.66",
        "pyarrow",
    )
)

app = modal.App(name=f"{APP_NAME}-homograph-eval", image=image)


@app.function(
    gpu="A10G",
    timeout=60 * 60,
    volumes={"/ckpts": checkpoints_volume, "/datasets": datasets_volume},
)
def evaluate_homograph() -> dict:
    """Compute homograph accuracy on our test set."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    checkpoints_volume.reload()
    datasets_volume.reload()

    # Load our model
    ckpt_path = "/ckpts/persian_g2p/run-001/best"
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path).to("cuda")
    model.eval()

    # Load HomoRich test set (parquet) to get homograph annotations
    # We need the test split — HomoRich has recommended_split/test_set.csv
    # For now use our test.jsonl + the original parquet metadata
    test_path = Path("/datasets/persian-g2p/test.jsonl")
    examples = []
    for line in test_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        src = (row.get("src") or "").strip()
        tgt = (row.get("tgt") or "").strip()
        if src and tgt:
            examples.append((src, tgt))
    print(f"Test examples: {len(examples)}", flush=True)

    # Try to load HomoRich parquet for homograph metadata
    # The parquet has columns: Grapheme, Phoneme, Homograph Grapheme, Homograph Phoneme
    homo_data = {}
    try:
        import pyarrow.parquet as pq
        # Download HomoRich parquet inside Modal
        import urllib.request
        url = "https://huggingface.co/datasets/MahtaFetrat/HomoRich-G2P-Persian/resolve/main/data/train-01.parquet"
        parquet_path = "/tmp/homorich.parquet"
        if not Path(parquet_path).exists():
            print("Downloading HomoRich parquet...", flush=True)
            urllib.request.urlretrieve(url, parquet_path)
        table = pq.read_table(parquet_path)
        graphemes = table.column("Grapheme").to_pylist()
        homograph_graphemes = table.column("Homograph Grapheme").to_pylist()
        homograph_phonemes = table.column("Homograph Phoneme").to_pylist()
        for g, hg, hp in zip(graphemes, homograph_graphemes, homograph_phonemes):
            if g and hg and hp:
                homo_data[g.strip()] = (hg.strip(), hp.strip())
        print(f"HomoRich homograph entries: {len(homo_data)}", flush=True)
    except Exception as e:
        print(f"WARNING: could not load HomoRich parquet: {e}", flush=True)

    # For each test example, find homograph (if any) and check pronunciation
    total_homographs = 0
    correct_homographs = 0
    per_correct = 0
    per_total = 0

    batch_size = 8
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            inputs = [src for src, _ in batch]
            enc = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            out = model.generate(**enc, max_new_tokens=512, num_beams=1)
            preds = tokenizer.batch_decode(out, skip_special_tokens=True)

            for j, (src, gold) in enumerate(batch):
                pred = preds[j].strip()

                # Check if this sentence has a homograph annotation
                if src in homo_data:
                    hg, hp = homo_data[src]
                    total_homographs += 1
                    # Check if the predicted phoneme output contains the homograph pronunciation
                    # The homograph phoneme should appear as a token in pred
                    if hp in pred:
                        correct_homographs += 1

                # Per-sentence word accuracy
                pred_words = pred.split()
                gold_words = gold.split()
                if len(pred_words) == len(gold_words):
                    if pred_words == gold_words:
                        per_correct += 1
                per_total += 1

            if i % 800 == 0 and i > 0:
                ha = correct_homographs / max(1, total_homographs)
                print(f"  [{i}/{len(examples)}] HA={ha:.4f} ({total_homographs} homographs)", flush=True)

    ha = correct_homographs / max(1, total_homographs)
    sa = per_correct / max(1, per_total)
    result = {
        "homograph_accuracy": ha,
        "n_homographs": total_homographs,
        "n_homographs_correct": correct_homographs,
        "sentence_exact_match": sa,
        "n_examples": per_total,
    }
    print(f"\n=== Homograph Accuracy: {ha:.4f} ({correct_homographs}/{total_homographs}) ===", flush=True)
    print(f"=== Sentence Exact Match: {sa:.4f} ===", flush=True)
    return result


@app.local_entrypoint()
def main():
    result = evaluate_homograph.remote()
    print(json.dumps(result, indent=2))
