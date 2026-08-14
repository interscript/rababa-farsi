"""Persian diacriticization trainer (Persian text → Persian with haraqat).

Input: bare Persian text (e.g., روی دیوار ننویسید.)
Output: Persian with haraqat (e.g., رُوی دِیوَار نَنِویسِد.)

Architecture: ByT5-small (Persian text is short, small model saturates)
Data: HomoRich `Mapped Phoneme` field → haraqat-extracted (445K examples)

Usage:
    modal run modal_app_diacrit.py::upload_data
    modal run --detach modal_app_diacrit.py::train
    modal run modal_app_diacrit.py::evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "persian-diacrit"

datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name(f"{APP_NAME}-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git", "curl")
    .pip_install(
        "torch>=2.4,<3",
        "transformers>=4.46",
        "sentencepiece",
        "protobuf",
        "accelerate",
        "numpy>=1.26,<3",
        "omegaconf>=2.3,<3",
        "tqdm>=4.66",
        "pyyaml>=6.0",
        "pyarrow",
    )
    .add_local_dir("src", "/opt/persian/src", copy=True)
    .add_local_dir("data", "/opt/persian/data", copy=True)
    .workdir("/opt/persian")
    .env({"PYTHONPATH": "/opt/persian/src"})
)

app = modal.App(name=APP_NAME, image=image)


@app.function(
    timeout=10 * 60,
    volumes={"/datasets": datasets_volume},
)
def upload_data() -> dict:
    from pathlib import Path as _P
    import shutil

    root = _P("/datasets/persian-diacrit")
    root.mkdir(parents=True, exist_ok=True)

    count = 0
    for split in ("train", "val", "test"):
        local = _P(f"/opt/persian/data/{split}.jsonl")
        if not local.is_file():
            print(f"[upload] WARNING: {local} not found", flush=True)
            continue
        dst = root / f"{split}.jsonl"
        shutil.copy2(local, dst)
        n = sum(1 for _ in local.open(encoding="utf-8"))
        count += n
        print(f"[upload] {split}: {n} lines", flush=True)

    datasets_volume.commit()
    return {"lines": count, "root": str(root)}


@app.function(
    gpu="A100",
    timeout=6 * 60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
)
def train() -> dict:
    """Train ByT5-small on Persian diacritization."""
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
    )
    from torch.utils.data import Dataset
    import json as _json

    datasets_volume.reload()

    model_name = "google/byt5-small"
    print(f"Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    train_path = Path("/datasets/persian-diacrit/train.jsonl")
    val_path = Path("/datasets/persian-diacrit/val.jsonl")

    class DiacritDataset(Dataset):
        def __init__(self, path, tok, max_len=512):
            self.examples = []
            for ln in Path(path).read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = _json.loads(ln)
                except Exception:
                    continue
                src = (r.get("src") or "").strip()
                tgt = (r.get("tgt") or "").strip()
                if not src or not tgt:
                    continue
                if len(src.encode("utf-8")) > max_len or len(tgt.encode("utf-8")) > max_len:
                    continue
                self.examples.append((src, tgt))
            self.tok = tok
            self.max_len = max_len

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            src, tgt = self.examples[idx]
            mi = self.tok(src, truncation=True, max_length=self.max_len)
            lab = self.tok(tgt, truncation=True, max_length=self.max_len)
            mi["labels"] = lab["input_ids"]
            return mi

    train_ds = DiacritDataset(train_path, tokenizer)
    val_ds = DiacritDataset(val_path, tokenizer)
    print(f"train={len(train_ds)}, val={len(val_ds)}", flush=True)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100
    )

    ckpt_root = Path("/checkpoints/persian_diacrit/run-001")
    ckpt_root.mkdir(parents=True, exist_ok=True)

    args = Seq2SeqTrainingArguments(
        output_dir=str(ckpt_root),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-4,
        warmup_steps=500,
        weight_decay=0.01,
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        seed=42,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        predict_with_generate=False,
        logging_steps=50,
        report_to=[],
        dataloader_num_workers=4,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    best_path = ckpt_root / "best"
    trainer.save_model(str(best_path))
    tokenizer.save_pretrained(str(best_path))

    checkpoints_volume.commit()
    return {"best": str(best_path)}


@app.function(
    gpu="A10G",
    timeout=60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
)
def evaluate() -> dict:
    """Evaluate on test set."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    checkpoints_volume.reload()
    datasets_volume.reload()

    ckpt = Path("/checkpoints/persian_diacrit/run-001/best")
    if not ckpt.is_dir():
        return {"error": f"{ckpt} not found"}

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt)).to("cuda")
    model.eval()

    test_path = Path("/datasets/persian-diacrit/test.jsonl")
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

    def _ed(a, b):
        m, n = len(a), len(b)
        if m == 0:
            return n
        if n == 0:
            return m
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            curr = [i] + [0] * n
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            prev = curr
        return prev[n]

    total_ed = 0
    total_gold = 0
    exact = 0
    n = 0
    batch_size = 8

    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            inputs = [src for src, _ in batch]
            enc = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            out = model.generate(**enc, max_new_tokens=512, num_beams=1)
            preds = tokenizer.batch_decode(out, skip_special_tokens=True)

            for j, (_, gold) in enumerate(batch):
                pred_chars = list(preds[j].strip())
                gold_chars = list(gold.strip())
                ed = _ed(pred_chars, gold_chars)
                total_ed += ed
                total_gold += max(1, len(gold_chars))
                if ed == 0:
                    exact += 1
                n += 1

            if i % 800 == 0 and i > 0:
                cer = total_ed / max(1, total_gold)
                print(f"  [{i}/{len(examples)}] CER={cer:.4f}", flush=True)

    cer = total_ed / max(1, total_gold)
    result = {
        "cer": cer,
        "exact_match": exact / max(1, n),
        "n_examples": n,
    }
    print(f"\n=== Persian Diacritization CER: {cer:.4f} ===", flush=True)
    return result


@app.local_entrypoint()
def main():
    upload_result = upload_data.remote()
    print(f"Upload: {json.dumps(upload_result, indent=2)}")

    train_result = train.remote()
    print(f"Train: {json.dumps(train_result, indent=2, default=str)}")

    eval_result = evaluate.remote()
    print(f"Evaluate: {json.dumps(eval_result, indent=2)}")
