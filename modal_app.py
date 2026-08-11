"""Modal app for Persian/Farsi diacritization training + evaluation.

Strategy: Cross-lingual transfer from Arabic.
1. Train on Arabic Tashkeela corpus (2.1M examples, same haraqat system)
2. This teaches the model haraqat prediction for Arabic script
3. Persian/Urdu share the script and haraqat — knowledge transfers
4. Fine-tune on any available Persian data later

Usage:
    modal run modal_app.py::fetch_data
    modal run --detach modal_app.py::train
    modal run modal_app.py::evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "persian-diacrit"
RABABA_DATASETS = modal.Volume.from_name("rababa-datasets", create_if_missing=True)

datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name(f"{APP_NAME}-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git", "curl")
    .pip_install(
        "torch>=2.4,<3", "numpy>=1.26,<3", "omegaconf>=2.3,<3",
        "tqdm>=4.66", "pyyaml>=6.0",
    )
    .add_local_dir("src", "/opt/persian/src", copy=True)
    .add_local_dir("configs", "/opt/persian/configs", copy=True)
    .workdir("/opt/persian")
    .env({"PYTHONPATH": "/opt/persian/src"})
)

app = modal.App(name=APP_NAME, image=image)


@app.function(
    timeout=60 * 60,
    volumes={"/datasets": datasets_volume, "/rababa-datasets": RABABA_DATASETS},
)
def fetch_data() -> dict:
    """Fetch training data. Uses Arabic Tashkeela corpus for cross-lingual transfer."""
    from pathlib import Path as _P
    import shutil

    root = _P("/datasets/persian-diacritized")
    root.mkdir(parents=True, exist_ok=True)

    # Copy Arabic Tashkeela corpus (same haraqat system, same script)
    RABABA_DATASETS.reload()
    arabic_combined = _P("/rababa-datasets/arabic-combined")
    count = 0
    for split in ("train", "val", "test"):
        src = arabic_combined / f"{split}.txt"
        dst = root / f"{split}.txt"
        if src.is_file():
            shutil.copy2(src, dst)
            lines = sum(1 for _ in src.open(encoding="utf-8"))
            count += lines
            print(f"[fetch] {split}: {lines} lines from Arabic corpus", flush=True)
        else:
            print(f"[fetch] WARNING: {src} not found", flush=True)

    # Also copy any bundled sample Persian data
    bundled = _P("/opt/persian/test-datasets")
    if (bundled / "sample.txt").is_file():
        shutil.copy2(bundled / "sample.txt", root / "persian_sample.txt")
        print(f"[fetch] Copied Persian sample data", flush=True)

    datasets_volume.commit()
    print(f"[fetch] Total: {count} lines of Arabic haraqat training data", flush=True)
    print("[fetch] Strategy: cross-lingual transfer (Arabic→Persian via shared haraqat)", flush=True)

    return {"lines": count, "source": "arabic-combined", "root": str(root)}


@app.function(
    gpu="A100",
    timeout=6 * 60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
)
def train() -> dict:
    """Train Persian diacritization model on Arabic haraqat data (cross-lingual)."""
    import torch
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf

    from persian_diacrit.encoder import PersianEncoder
    from persian_diacrit.model import PersianDiacritModel
    from persian_diacrit.dataset import PersianDataset, collate_batch
    from persian_diacrit.evaluate import haraqat_der

    cfg = OmegaConf.load("/opt/persian/configs/persian_diacrit.yaml")
    device = torch.device("cuda")

    encoder = PersianEncoder()
    model = PersianDiacritModel(
        vocab_size=encoder.vocab_size,
        dim=cfg.model.dim, layers=cfg.model.layers, heads=cfg.model.heads,
        ff_dim=cfg.model.ff_dim, dropout=cfg.model.dropout, max_len=cfg.model.max_len,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    data_root = Path(cfg.data.root)
    train_ds = PersianDataset(data_root / "train.txt", max_len=cfg.data.max_len)
    val_ds = PersianDataset(data_root / "val.txt", max_len=cfg.data.max_len)
    print(f"train={len(train_ds)}, val={len(val_ds)}", flush=True)

    if len(train_ds) == 0:
        return {"error": "No training data. Run fetch_data first."}

    # Subsample if too large (2.1M takes too long for first run)
    max_train = 200000
    if len(train_ds) > max_train:
        train_ds.examples = train_ds.examples[:max_train]
        print(f"Subsampled train to {len(train_ds)} for speed", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        collate_fn=collate_batch, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        collate_fn=collate_batch, num_workers=4, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    ckpt_root = Path("/checkpoints/persian_diacrit/run-001")
    ckpt_root.mkdir(parents=True, exist_ok=True)
    best_val_der = float("inf")

    epochs = 3  # start with 3 epochs (Arabic v2 showed 3 is enough with 2.1M data)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            src = batch["src"].to(device)
            haraqat = batch["haraqat"].to(device)
            ezafe = batch["ezafe"].to(device)
            lengths = batch["lengths"].to(device)
            padding_mask = torch.arange(src.size(1), device=device)[None, :] >= lengths[:, None]

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.train.fp16):
                haraqat_logits, ezafe_logits = model(src, src_key_padding_mask=padding_mask)
                loss_h = torch.nn.functional.cross_entropy(
                    haraqat_logits[:, 1:].reshape(-1, haraqat_logits.size(-1)),
                    haraqat[:, 1:].reshape(-1), ignore_index=-100,
                    label_smoothing=cfg.train.label_smoothing,
                )
                loss_e = torch.nn.functional.cross_entropy(
                    ezafe_logits[:, 1:].reshape(-1, 2),
                    ezafe[:, 1:].reshape(-1), ignore_index=-100,
                )
                loss = loss_h + 0.3 * loss_e

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        total_der = 0.0
        total_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch["src"].to(device)
                haraqat = batch["haraqat"].to(device)
                lengths = batch["lengths"].to(device)
                padding_mask = torch.arange(src.size(1), device=device)[None, :] >= lengths[:, None]
                hl, _ = model(src, src_key_padding_mask=padding_mask)
                total_der += haraqat_der(hl[:, 1:], haraqat[:, 1:])
                total_batches += 1

        val_der = total_der / max(1, total_batches)
        print(f"[train] epoch {epoch}: loss={running_loss/max(1,n_batches):.4f} val_der={val_der:.4f}", flush=True)

        if val_der < best_val_der:
            best_val_der = val_der
            torch.save(model.state_dict(), ckpt_root / "best.pt")
            print(f"  → saved best.pt", flush=True)

        checkpoints_volume.commit()

    return {"best_val_der": best_val_der}


@app.function(
    gpu="A10G",
    timeout=30 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
)
def evaluate() -> dict:
    """Evaluate on test set."""
    import torch
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf

    from persian_diacrit.encoder import PersianEncoder
    from persian_diacrit.model import PersianDiacritModel
    from persian_diacrit.dataset import PersianDataset, collate_batch
    from persian_diacrit.evaluate import haraqat_der

    cfg = OmegaConf.load("/opt/persian/configs/persian_diacrit.yaml")
    device = torch.device("cuda")

    encoder = PersianEncoder()
    model = PersianDiacritModel(
        vocab_size=encoder.vocab_size,
        dim=cfg.model.dim, layers=cfg.model.layers, heads=cfg.model.heads,
        ff_dim=cfg.model.ff_dim, dropout=0.0, max_len=cfg.model.max_len,
    ).to(device)

    ckpt = Path("/checkpoints/persian_diacrit/run-001/best.pt")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    data_root = Path(cfg.data.root)
    test_ds = PersianDataset(data_root / "test.txt", max_len=cfg.data.max_len)
    print(f"test examples: {len(test_ds)}", flush=True)

    loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_batch)
    total_der = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            src = batch["src"].to(device)
            haraqat = batch["haraqat"].to(device)
            lengths = batch["lengths"].to(device)
            padding_mask = torch.arange(src.size(1), device=device)[None, :] >= lengths[:, None]
            hl, _ = model(src, src_key_padding_mask=padding_mask)
            total_der += haraqat_der(hl[:, 1:], haraqat[:, 1:])
            total_batches += 1

    der = total_der / max(1, total_batches)
    print(f"=== Persian DER: {der:.4f} ({len(test_ds)} examples) ===", flush=True)
    return {"der": der, "n_examples": len(test_ds)}


@app.local_entrypoint()
def main():
    """Run full pipeline: fetch → train → evaluate."""
    fetch_result = fetch_data.remote()
    print(f"Fetch: {json.dumps(fetch_result, indent=2)}")

    train_result = train.remote()
    print(f"Train: {json.dumps(train_result, indent=2, default=str)}")

    eval_result = evaluate.remote()
    print(f"Evaluate: {json.dumps(eval_result, indent=2)}")
