"""Modal app for Persian/Farsi diacritization training + evaluation.

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

datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name(f"{APP_NAME}-checkpoints", create_if_missing=True)
models_volume = modal.Volume.from_name(f"{APP_NAME}-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git", "curl")
    .pip_install(
        "torch>=2.4,<3",
        "numpy>=1.26,<3",
        "omegaconf>=2.3,<3",
        "tqdm>=4.66",
        "pyyaml>=6.0",
    )
    .add_local_dir("src", "/opt/persian/src", copy=True)
    .add_local_dir("configs", "/opt/persian/configs", copy=True)
    .add_local_dir("test-datasets", "/opt/persian/test-datasets", copy=True)
    .workdir("/opt/persian")
    .env({"PYTHONPATH": "/opt/persian/src"})
)

app = modal.App(name=APP_NAME, image=image)


# ---- Data fetch ----

@app.function(
    timeout=60 * 60,
    volumes={"/datasets": datasets_volume},
)
def fetch_data() -> dict:
    """Fetch Persian diacritized data.

    Sources:
    1. Bundled test-datasets (if any diacritized Persian text available)
    2. Download from Ganjoor (Persian poetry with diacritics)
    3. Persian Wikipedia (undiacritized, for future distillation)
    """
    from pathlib import Path as _P

    root = _P("/datasets/persian-diacritized")
    root.mkdir(parents=True, exist_ok=True)

    # Check bundled test data
    bundled = _P("/opt/persian/test-datasets")
    count = 0
    if bundled.is_dir():
        for split in ("train", "val", "test"):
            src = bundled / f"{split}.txt"
            dst = root / f"{split}.txt"
            if src.is_file():
                import shutil
                shutil.copy2(src, dst)
                lines = sum(1 for _ in src.open(encoding="utf-8"))
                count += lines
                print(f"[fetch] {split}: {lines} lines from bundled", flush=True)

    if count == 0:
        # Download Ganjoor poetry (diacritized Persian)
        print("[fetch] No bundled data. Downloading Ganjoor poetry...", flush=True)
        import subprocess
        result = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/ganjoor/ganjoor.git",
             "/tmp/ganjoor"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            # Find diacritized poems
            poem_dir = _P("/tmp/ganjoor")
            all_lines = []
            for txt_file in poem_dir.rglob("*.txt"):
                try:
                    text = txt_file.read_text(encoding="utf-8")
                    for line in text.splitlines():
                        line = line.strip()
                        if len(line) < 5 or len(line) > 200:
                            continue
                        # Check if line has harakat
                        from persian_diacrit.constants import HARAQAT
                        if any(c in HARAQAT for c in line):
                            all_lines.append(line)
                except Exception:
                    continue

            print(f"[fetch] Found {len(all_lines)} diacritized lines from Ganjoor", flush=True)

            if all_lines:
                import random
                random.seed(42)
                random.shuffle(all_lines)
                n_test = min(1000, len(all_lines) // 20)
                n_val = min(1000, len(all_lines) // 20)
                splits = {
                    "test": all_lines[:n_test],
                    "val": all_lines[n_test:n_test+n_val],
                    "train": all_lines[n_test+n_val:],
                }
                for split, lines in splits.items():
                    out = root / f"{split}.txt"
                    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    print(f"[fetch] {split}: {len(lines)} lines", flush=True)
                count = len(all_lines)
        else:
            print(f"[fetch] Ganjoor clone failed: {result.stderr[:200]}", flush=True)

    datasets_volume.commit()

    if count == 0:
        print("[fetch] WARNING: No diacritized Persian data found!", flush=True)
        print("[fetch] Please add diacritized text to test-datasets/", flush=True)

    return {"lines": count, "root": str(root)}


# ---- Training ----

@app.function(
    gpu="A100",
    timeout=6 * 60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
)
def train() -> dict:
    """Train Persian diacritization model."""
    import torch
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf

    from persian_diacrit.encoder import PersianEncoder
    from persian_diacrit.model import PersianDiacritModel
    from persian_diacrit.dataset import PersianDataset, collate_batch
    from persian_diacrit.evaluate import haraqat_der, ezafe_metrics

    cfg = OmegaConf.load("/opt/persian/configs/persian_diacrit.yaml")
    device = torch.device("cuda")

    encoder = PersianEncoder()
    model = PersianDiacritModel(
        vocab_size=encoder.vocab_size,
        dim=cfg.model.dim,
        layers=cfg.model.layers,
        heads=cfg.model.heads,
        ff_dim=cfg.model.ff_dim,
        dropout=cfg.model.dropout,
        max_len=cfg.model.max_len,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    data_root = Path(cfg.data.root)
    train_ds = PersianDataset(data_root / "train.txt", max_len=cfg.data.max_len)
    val_ds = PersianDataset(data_root / "val.txt", max_len=cfg.data.max_len)
    print(f"train={len(train_ds)}, val={len(val_ds)}", flush=True)

    if len(train_ds) == 0:
        return {"error": "No training data. Run fetch_data first."}

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        collate_fn=collate_batch, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        collate_fn=collate_batch, num_workers=4, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    ckpt_root = Path("/checkpoints/persian_diacrit/run-001")
    ckpt_root.mkdir(parents=True, exist_ok=True)

    best_val_der = float("inf")
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.fp16)

    for epoch in range(cfg.train.epochs):
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
                haraqat_logits = haraqat_logits[:, 1:]
                ezafe_logits = ezafe_logits[:, 1:]
                haraqat_targets = haraqat[:, 1:]
                ezafe_targets = ezafe[:, 1:]

                loss_haraqat = torch.nn.functional.cross_entropy(
                    haraqat_logits.reshape(-1, haraqat_logits.size(-1)),
                    haraqat_targets.reshape(-1),
                    ignore_index=-100,
                    label_smoothing=cfg.train.label_smoothing,
                )
                loss_ezafe = torch.nn.functional.cross_entropy(
                    ezafe_logits.reshape(-1, 2),
                    ezafe_targets.reshape(-1),
                    ignore_index=-100,
                )
                loss = loss_haraqat + 0.5 * loss_ezafe

            if cfg.train.fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
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
                ezafe = batch["ezafe"].to(device)
                lengths = batch["lengths"].to(device)
                padding_mask = torch.arange(src.size(1), device=device)[None, :] >= lengths[:, None]

                haraqat_logits, ezafe_logits = model(src, src_key_padding_mask=padding_mask)
                total_der += haraqat_der(haraqat_logits[:, 1:], haraqat[:, 1:])
                total_batches += 1

        val_der = total_der / max(1, total_batches)
        train_loss = running_loss / max(1, n_batches)
        print(
            f"[train] epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_der={val_der:.4f}",
            flush=True,
        )

        if val_der < best_val_der:
            best_val_der = val_der
            torch.save(model.state_dict(), ckpt_root / "best.pt")
            print(f"  → saved best.pt (val_der={val_der:.4f})", flush=True)

        checkpoints_volume.commit()

    return {"best_val_der": best_val_der, "checkpoint": str(ckpt_root / "best.pt")}


# ---- Evaluation ----

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
    from persian_diacrit.evaluate import haraqat_der, ezafe_metrics

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
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    data_root = Path(cfg.data.root)
    test_ds = PersianDataset(data_root / "test.txt", max_len=cfg.data.max_len)
    print(f"test examples: {len(test_ds)}", flush=True)

    loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_batch)

    total_der = 0.0
    total_batches = 0
    all_ezafe_preds = []
    all_ezafe_golds = []

    with torch.no_grad():
        for batch in loader:
            src = batch["src"].to(device)
            haraqat = batch["haraqat"].to(device)
            ezafe = batch["ezafe"].to(device)
            lengths = batch["lengths"].to(device)
            padding_mask = torch.arange(src.size(1), device=device)[None, :] >= lengths[:, None]

            haraqat_logits, ezafe_logits = model(src, src_key_padding_mask=padding_mask)
            total_der += haraqat_der(haraqat_logits[:, 1:], haraqat[:, 1:])
            total_batches += 1

    der = total_der / max(1, total_batches)
    result = {"der": der, "n_examples": len(test_ds)}
    print(f"=== Persian DER: {der:.4f} ({len(test_ds)} examples) ===", flush=True)
    return result


@app.local_entrypoint()
def main():
    """Run full pipeline: fetch → train → evaluate."""
    fetch_result = fetch_data.remote()
    print(f"Fetch: {json.dumps(fetch_result, indent=2)}")

    if fetch_result.get("lines", 0) > 0:
        train_result = train.remote()
        print(f"Train: {json.dumps(train_result, indent=2, default=str)}")

        eval_result = evaluate.remote()
        print(f"Evaluate: {json.dumps(eval_result, indent=2)}")
