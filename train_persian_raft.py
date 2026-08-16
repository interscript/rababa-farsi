"""RAFT (rejection-sampling fine-tuning) on Persian G2P v1 (77.34% SB HA).

Verifiable-reward RL (TODO.research/12) aimed at the exact stuck point:
homograph disambiguation. Samples K=4 candidates per prompt, scores each
with a word-level CER where homograph words (>=2 distinct pronunciations
in the corpus, mined from train statistics) count 3x, keeps only samples
that strictly beat greedy, fine-tunes on the winners.

Selection on the held-out val split (private dev); SentenceBench is
measured once, at the end. Base model: persian_g2p/run-001/best,
raw-src prompts (no prefix — v1 format).

Usage:
    modal run --detach train_persian_raft.py
"""

from __future__ import annotations

import random
from pathlib import Path

import modal

datasets_volume = modal.Volume.from_name("persian-g2p-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("persian-g2p-checkpoints", create_if_missing=True)

BASE = "/checkpoints/persian_g2p/run-001/best"
RAFT_RUN = "persian_g2p_raft/run-001"
DATA = Path("/datasets/persian-g2p")
PROMPT_POOL = 200_000
N_PROMPTS = 6_000
K = 4
TEMP = 0.9
TOP_P = 0.95
MAX_LEN = 256
ITERS = 3
LR = 3e-5
DEV_N = 500
KEEP_MAX_REWARD = 0.10
HG_WEIGHT = 3.0
HG_MIN_FREQ = 4

SENTENCEBENCH_URL = (
    "https://huggingface.co/datasets/MahtaFetrat/SentenceBench/resolve/main/SentenceBench.csv"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate>=1.1.0",
        "editdistance",
        "pandas",
        "tqdm",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("persian-g2p-raft", image=image)


def build_homograph_lexicon(rows: list[dict]) -> set[str]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        src_words = row["src"].split()
        tgt_words = row["tgt"].split()
        if len(src_words) != len(tgt_words):
            continue
        for w, p in zip(src_words, tgt_words):
            counts.setdefault(w, {})
            counts[w][p] = counts[w].get(p, 0) + 1
    lex = set()
    for w, prons in counts.items():
        if sum(prons.values()) >= HG_MIN_FREQ and len(prons) >= 2:
            lex.add(w)
    return lex


def hg_flags(src: str, gold: str, hg_lex: set[str]) -> list[bool]:
    sw, gw = src.split(), gold.split()
    if len(sw) != len(gw):
        return [False] * len(gw)
    return [w in hg_lex for w in sw]


def word_reward(pred: str, gold: str, flags: list[bool]) -> float:
    import editdistance

    pw, gw = pred.split(), gold.split()
    n = max(len(pw), len(gw))
    s = tot = 0.0
    for i in range(n):
        w = gw[i] if i < len(gw) else ""
        p = pw[i] if i < len(pw) else ""
        weight = HG_WEIGHT if i < len(flags) and flags[i] else 1.0
        s += weight * editdistance.eval(p, w) / max(1, len(w))
        tot += weight
    return s / tot if tot else 1.0


@app.function(
    gpu="A10G",
    timeout=11 * 60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
)
def run() -> dict:
    import json
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    datasets_volume.reload()
    checkpoints_volume.reload()

    raft_dir = Path("/checkpoints") / RAFT_RUN
    done_marker = raft_dir / "EVAL_DONE"
    if done_marker.exists():
        return {"status": "already-done"}

    train_rows = [json.loads(l) for l in (DATA / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    dev_rows = [json.loads(l) for l in (DATA / "val.jsonl").read_text(encoding="utf-8").splitlines()][:DEV_N]

    # one-time freeze of the private dev split (tamper-evident selection)
    pd_dir = Path("/datasets/private-dev/persian")
    if not (pd_dir / "FROZEN").exists():
        import hashlib

        pd_dir.mkdir(parents=True, exist_ok=True)
        raw = (DATA / "val.jsonl").read_bytes()
        (pd_dir / "MANIFEST.txt").write_text(
            f"dataset: persian private dev\n"
            f"derived_from: {DATA}/val.jsonl (v1 held-out split, first {DEV_N} used)\n"
            f"sha256: {hashlib.sha256(raw).hexdigest()}\n",
            encoding="utf-8",
        )
        (pd_dir / "FROZEN").touch()
        datasets_volume.commit()

    hg_lex = build_homograph_lexicon(train_rows)
    print(f"[lexicon] {len(hg_lex)} homograph words", flush=True)

    rng = random.Random(43)
    pool = train_rows[:PROMPT_POOL]
    rng.shuffle(pool)
    prompts = pool[:N_PROMPTS]
    print(f"[data] dev={len(dev_rows)} prompts={len(prompts)}", flush=True)

    load_dir = str(raft_dir / "best") if (raft_dir / "best").is_dir() else BASE
    print(f"[load] {load_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(load_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(load_dir).to("cuda")
    device = next(model.parameters()).device

    def greedy(texts: list[str], batch: int = 64) -> list[str]:
        out: list[str] = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = tokenizer(
                    texts[i : i + batch], return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_LEN,
                ).to(device)
                with torch.autocast("cuda", torch.bfloat16):
                    gen = model.generate(**enc, max_new_tokens=MAX_LEN, num_beams=1)
                out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        return out

    def mean_reward(rows: list[dict]) -> float:
        preds = greedy([r["src"] for r in rows])
        rs = [
            word_reward(p, r["tgt"], hg_flags(r["src"], r["tgt"], hg_lex))
            for p, r in zip(preds, rows)
        ]
        return sum(rs) / len(rs)

    metrics_path = raft_dir / "metrics.jsonl"
    raft_dir.mkdir(parents=True, exist_ok=True)

    best_dev = None
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            m = json.loads(line)
            best_dev = m.get("best_dev")
    if best_dev is None:
        best_dev = mean_reward(dev_rows)
        print(f"[dev] v1 baseline reward={best_dev:.4%}", flush=True)

    class PairDataset(Dataset):
        def __init__(self, pairs: list[tuple[str, str]]) -> None:
            self.pairs = pairs

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, idx: int) -> dict:
            src, tgt = self.pairs[idx]
            inputs = tokenizer(src, truncation=True, max_length=MAX_LEN)
            labels = tokenizer(tgt, truncation=True, max_length=MAX_LEN)
            inputs["labels"] = labels["input_ids"]
            return inputs

    srcs = [r["src"] for r in prompts]
    golds = [r["tgt"] for r in prompts]

    for it in range(1, ITERS + 1):
        iter_marker = raft_dir / f"iter{it}.done"
        if iter_marker.exists():
            continue

        print(f"[iter{it}] sampling greedy + {K} candidates on {len(srcs)} prompts", flush=True)
        greedy_preds: list[str] = []
        samples: list[list[str]] = [[] for _ in srcs]
        model.eval()
        with torch.no_grad():
            for i in range(0, len(srcs), 16):
                batch = srcs[i : i + 16]
                enc = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_LEN,
                ).to(device)
                with torch.autocast("cuda", torch.bfloat16):
                    g = model.generate(**enc, max_new_tokens=MAX_LEN, num_beams=1)
                    s = model.generate(
                        **enc, max_new_tokens=MAX_LEN, num_beams=1,
                        do_sample=True, temperature=TEMP, top_p=TOP_P,
                        num_return_sequences=K,
                    )
                greedy_preds.extend(tokenizer.batch_decode(g, skip_special_tokens=True))
                decoded = tokenizer.batch_decode(s, skip_special_tokens=True)
                for j in range(len(batch)):
                    samples[i + j] = decoded[j * K : (j + 1) * K]
                if (i // 16) % 50 == 0:
                    print(f"[iter{it}] sampled {i + len(batch)}/{len(srcs)}", flush=True)

        winners: list[tuple[str, str]] = []
        hg_wins = 0
        for src, gold, gp, cands in zip(srcs, golds, greedy_preds, samples):
            flags = hg_flags(src, gold, hg_lex)
            greward = word_reward(gp, gold, flags)
            if greward == 0.0:
                continue
            scored = sorted((word_reward(c, gold, flags), c) for c in cands)
            breward, best = scored[0]
            if breward < greward and breward <= KEEP_MAX_REWARD:
                winners.append((src, best))
                gw, bw, pw_ = gold.split(), best.split(), gp.split()
                if any(
                    f and i < len(bw) and i < len(pw_) and bw[i] == gw[i] != pw_[i]
                    for i, f in enumerate(flags)
                ):
                    hg_wins += 1
        torch.cuda.empty_cache()  # release sampling KV-cache before training
        print(f"[iter{it}] kept {len(winners)}/{len(srcs)} winners ({hg_wins} homograph fixes)", flush=True)
        if not winners:
            iter_marker.touch()
            checkpoints_volume.commit()
            continue

        args = Seq2SeqTrainingArguments(
            output_dir=str(raft_dir / f"iter{it}"),
            num_train_epochs=1,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            bf16=True,
            learning_rate=LR,
            lr_scheduler_type="cosine",
            warmup_steps=20,
            weight_decay=0.01,
            max_grad_norm=1.0,
            seed=42,
            save_strategy="no",
            logging_steps=20,
            report_to=[],
            dataloader_num_workers=2,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=PairDataset(winners),
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100),
        )
        trainer.train()
        model = trainer.model

        dev_reward = mean_reward(dev_rows)
        print(f"[iter{it}] dev reward={dev_reward:.4%} (base={best_dev:.4%})", flush=True)

        if best_dev is None or dev_reward < best_dev:
            best_dev = dev_reward
            best_dir = raft_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))
            print(f"[iter{it}] new best -> {best_dir}", flush=True)

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "iter": it, "kept": len(winners), "hg_wins": hg_wins,
                "dev_reward": dev_reward, "best_dev": best_dev,
            }) + "\n")
        iter_marker.touch()
        checkpoints_volume.commit()

    # ---- final one-shot SentenceBench homograph accuracy ----
    import csv
    import urllib.request

    eval_model = model
    best_dir = raft_dir / "best"
    if best_dir.is_dir():
        eval_model = AutoModelForSeq2SeqLM.from_pretrained(str(best_dir)).to(device)
    eval_model.eval()

    local_csv = Path("/tmp/SentenceBench.csv")
    urllib.request.urlretrieve(SENTENCEBENCH_URL, local_csv)
    with local_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    hg_rows = [
        r for r in rows
        if (r.get("homograph word") or "").strip() and (r.get("pronunciation") or "").strip()
    ]
    print(f"[sb] {len(hg_rows)} homograph rows", flush=True)

    results: dict = {"run": RAFT_RUN, "best_dev": best_dev}
    srcs_hg = [r["grapheme"].strip() for r in hg_rows]
    for beam in (4, 1):
        preds: list[str] = []
        with torch.no_grad():
            for i in range(0, len(srcs_hg), 64):
                batch = srcs_hg[i : i + 64]
                enc = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_LEN,
                ).to(device)
                with torch.autocast("cuda", torch.bfloat16):
                    gen = eval_model.generate(**enc, max_new_tokens=MAX_LEN, num_beams=beam)
                preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))

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
                pred_pron == pron
                or pred_pron.rstrip("e") == pron
                or pred_pron == pron.rstrip("e")
            )
            total += 1
        results[f"sentencebench_beam{beam}"] = {
            "ha_exact": correct / max(1, total),
            "ha_ezafe_norm": correct_norm / max(1, total),
            "n_scored": total,
            "n_unaligned": unaligned,
        }
        print(
            f"[sb] beam={beam} HA exact={correct / max(1, total):.4%} "
            f"ezafe-norm={correct_norm / max(1, total):.4%} ({correct_norm}/{total})",
            flush=True,
        )

    done_marker.touch()
    checkpoints_volume.commit()
    return results


@app.local_entrypoint()
def main():
    run.remote()
