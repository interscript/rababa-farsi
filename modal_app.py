"""Modal app for Persian G2P training + evaluation with ByT5.

Strategy: ByT5-small fine-tuned on real Persian data (HomoRich, 528K sentences).
This replaces the previous cross-lingual Arabic haraqat approach.

Usage:
    modal run modal_app.py::upload_data
    modal run --detach modal_app.py::train
    modal run modal_app.py::evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "persian-g2p"

datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name(f"{APP_NAME}-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git", "curl")
    .pip_install(
        "torch>=2.4,<3",
        "transformers>=4.40,<5",
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
    .add_local_dir("configs", "/opt/persian/configs", copy=True)
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
    """Upload local JSONL data to the datasets volume."""
    from pathlib import Path as _P
    import shutil

    root = _P("/datasets/persian-g2p")
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
    timeout=12 * 60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
)
def train() -> dict:
    """Train ByT5-small on Persian HomoRich G2P data."""
    import torch
    from omegaconf import OmegaConf

    from persian_g2p.byt5 import train_byt5

    cfg = OmegaConf.load("/opt/persian/configs/persian_g2p.yaml")
    cfg_dict = dict(cfg)

    data_root = Path(cfg.data.root)
    train_path = data_root / "train.jsonl"
    val_path = data_root / "val.jsonl"

    if not train_path.is_file():
        return {"error": "No training data. Run upload_data first."}

    ckpt_root = Path("/checkpoints/persian_g2p/run-001")
    ckpt_root.mkdir(parents=True, exist_ok=True)
    metrics_path = Path("/checkpoints/metrics/persian_g2p-train.jsonl")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    best = train_byt5(cfg_dict, train_path, val_path, ckpt_root, metrics_path)
    checkpoints_volume.commit()
    return {"best": best}


@app.function(
    gpu="A10G",
    timeout=60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
)
def evaluate(num_beams: int = 4) -> dict:
    """Evaluate on test set."""
    import torch
    from omegaconf import OmegaConf

    from persian_g2p.byt5 import build_model, evaluate_byt5

    cfg = OmegaConf.load("/opt/persian/configs/persian_g2p.yaml")
    device = torch.device("cuda")

    ckpt = Path("/checkpoints/persian_g2p/run-001/best")
    if not ckpt.is_dir():
        return {"error": f"Checkpoint dir {ckpt} not found"}

    model, tokenizer = build_model(str(ckpt))
    model = model.to(device)

    data_root = Path(cfg.data.root)
    test_path = data_root / "test.jsonl"

    result = evaluate_byt5(model, tokenizer, test_path, device, num_beams=num_beams)
    print(f"=== Persian G2P test: {result} ===", flush=True)
    return result


@app.local_entrypoint()
def main():
    """Run full pipeline: upload → train → evaluate."""
    upload_result = upload_data.remote()
    print(f"Upload: {json.dumps(upload_result, indent=2)}")

    train_result = train.remote()
    print(f"Train: {json.dumps(train_result, indent=2, default=str)}")

    eval_result = evaluate.remote()
    print(f"Evaluate: {json.dumps(eval_result, indent=2)}")
