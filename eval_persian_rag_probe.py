"""RAG homograph probe — retrieval-augmented inference for Persian G2P v1.

The last untried model-quality lever (task #312). v1 sits at 77.34%
SentenceBench homograph accuracy (ezafe-normalized); RL was negative,
the mapped-repr line closed. Homograph resolution is a context
problem: this probe prepends top-k retrieved train examples containing
the SAME homograph as few-shot context and measures the delta — same
harness, same beam-4 decode, same scoring as eval_sentencebench.py.

Retrieval: char 3-gram cosine similarity over data-v3/train.jsonl
(445K grapheme→phoneme sentence pairs), candidates restricted to
train sentences containing the test homograph token. Contamination
guard: assert no test grapheme sentence appears in the train corpus.

Decision rule (TODO.improve-models/03):
- >= +1.5pp  → invest (retrieval cache + train-time consistency FT)
- +0.5..1.5 → ship as inference-time add-on
- < +0.5    → close the lever, record negative

Usage:
    modal run --detach eval_persian_rag_probe.py
"""

from __future__ import annotations

import csv
import json
import urllib.request
from collections import Counter
from pathlib import Path

import modal

checkpoints_volume = modal.Volume.from_name("persian-g2p-checkpoints", create_if_missing=True)

SENTENCEBENCH_URL = (
    "https://huggingface.co/datasets/MahtaFetrat/SentenceBench/resolve/main/SentenceBench.csv"
)
CKPT = "/checkpoints/persian_g2p/run-001/best"
K = 3
MAX_CANDIDATES = 2_000

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4,<3", "transformers>=4.40,<5", "sentencepiece", "protobuf", "numpy")
    .add_local_dir("data-v3", "/opt/persian/data-v3", copy=True)
    .workdir("/opt/persian")
)

app = modal.App("persian-g2p-rag-probe", image=image)


def _ngrams(s: str) -> Counter:
    s = f" {s.strip()} "
    return Counter(s[i : i + 3] for i in range(len(s) - 2))


def _cosine(a: Counter, b: Counter) -> float:
    small, big = (a, b) if len(a) < len(b) else (b, a)
    dot = sum(v * big.get(k, 0) for k, v in small.items())
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / max(1e-9, na * nb)


@app.function(gpu="A10G", timeout=6 * 60 * 60, volumes={"/checkpoints": checkpoints_volume})
def probe() -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    checkpoints_volume.reload()

    local_csv = Path("/tmp/SentenceBench.csv")
    urllib.request.urlretrieve(SENTENCEBENCH_URL, local_csv)
    with local_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    hg_rows = [
        r for r in rows if (r.get("homograph word") or "").strip() and (r.get("pronunciation") or "").strip()
    ]
    print(f"[sb] homograph rows: {len(hg_rows)}", flush=True)

    train: list[dict] = [
        json.loads(l)
        for l in Path("/opt/persian/data-v3/train.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    # contamination guard: exact test sentences are EXCLUDED from retrieval
    test_srcs = {r["grapheme"].strip() for r in hg_rows}
    train_srcs = {r["src"].strip() for r in train}
    overlap = test_srcs & train_srcs
    if overlap:
        print(f"[contam] {len(overlap)} test sentences also appear in train — excluded from retrieval", flush=True)
        train = [r for r in train if r["src"].strip() not in test_srcs]

    # index train by homograph token
    by_token: dict[str, list[int]] = {}
    for i, r in enumerate(train):
        for w in set(r["src"].split()):
            by_token.setdefault(w, []).append(i)
    print(f"[index] {len(by_token)} distinct tokens indexed", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSeq2SeqLM.from_pretrained(CKPT).to("cuda")
    model.eval()

    def generate(srcs: list[str]) -> list[str]:
        preds = []
        with torch.no_grad():
            for i in range(0, len(srcs), 32):
                batch = srcs[i : i + 32]
                enc = tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to("cuda")
                gen = model.generate(**enc, max_new_tokens=512, num_beams=4)
                preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        return preds

    plain_srcs: list[str] = []
    rag_srcs: list[str] = []
    n_retrieved = 0
    for row in hg_rows:
        grapheme = row["grapheme"].strip()
        hg_word = row["homograph word"].strip()
        plain_srcs.append(grapheme)

        cands = by_token.get(hg_word, [])[:MAX_CANDIDATES]
        if not cands:
            rag_srcs.append(grapheme)
            continue
        q = _ngrams(grapheme)
        scored = sorted(
            ((_cosine(q, _ngrams(train[i]["src"])), i) for i in cands), reverse=True
        )[:K]
        shots = [f"{train[i]['src']} => {train[i]['tgt']}" for _, i in scored]
        rag_srcs.append(" ; ".join(shots) + " ; " + grapheme)
        n_retrieved += 1
    print(f"[rag] {n_retrieved}/{len(hg_rows)} rows got few-shot context", flush=True)

    print("[gen] baseline...", flush=True)
    base_preds = generate(plain_srcs)
    print("[gen] rag...", flush=True)
    rag_preds = generate(rag_srcs)

    def score(preds: list[str]) -> dict:
        correct = correct_norm = total = unaligned = 0
        for row, pred in zip(hg_rows, preds):
            grapheme = row["grapheme"].strip()
            hg_word = row["homograph word"].strip()
            pron = row["pronunciation"].strip()
            g_words = grapheme.split()
            p_words = pred.strip().split()
            positions = [i for i, w in enumerate(g_words) if hg_word in w]
            if not positions:
                continue
            pos = positions[0]
            if pos >= len(p_words):
                unaligned += 1
                continue
            pred_pron = p_words[pos]
            correct += int(pred_pron == pron)
            correct_norm += int(
                pred_pron == pron or pred_pron.rstrip("e") == pron or pred_pron == pron.rstrip("e")
            )
            total += 1
        return {
            "homograph_accuracy": correct / max(1, total),
            "homograph_accuracy_ezafe_normalized": correct_norm / max(1, total),
            "n_scored": total,
            "n_unaligned": unaligned,
        }

    base = score(base_preds)
    rag = score(rag_preds)
    delta = rag["homograph_accuracy_ezafe_normalized"] - base["homograph_accuracy_ezafe_normalized"]
    result = {
        "baseline": base,
        "rag_k3": rag,
        "delta_ezafe_norm_pp": round(delta * 100, 2),
        "baseline_published_v1": 0.7734,
        "k": K,
        "n_with_context": n_retrieved,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    out = Path("/checkpoints/persian_g2p/rag_probe_result.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    checkpoints_volume.commit()
    return result


@app.local_entrypoint()
def main():
    probe.remote()
