"""Evaluation metrics for Persian diacritization.

DER (Diacritization Error Rate) — fraction of haraqat positions predicted wrong.
Ezafe F1 — precision/recall/F1 for ezafe detection.
"""

from __future__ import annotations

import torch


def haraqat_der(haraqat_logits: torch.Tensor, haraqat_targets: torch.Tensor) -> float:
    """Per-character DER for haraqat."""
    preds = haraqat_logits.argmax(dim=-1)
    mask = haraqat_targets != -100
    if mask.sum() == 0:
        return 0.0
    wrong = (preds[mask] != haraqat_targets[mask]).sum().item()
    return wrong / mask.sum().item()


def ezafe_metrics(ezafe_logits: torch.Tensor, ezafe_targets: torch.Tensor) -> dict[str, float]:
    """Precision/recall/F1 for binary ezafe detection."""
    preds = ezafe_logits.argmax(dim=-1)
    mask = ezafe_targets != -100
    if mask.sum() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_ezafe = (preds[mask] == 1)
    gold_ezafe = (ezafe_targets[mask] == 1)

    tp = (pred_ezafe & gold_ezafe).sum().item()
    fp = (pred_ezafe & ~gold_ezafe).sum().item()
    fn = (~pred_ezafe & gold_ezafe).sum().item()

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}
