"""Persian diacritization dataset — loads diacritized text for training."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .constants import HARAQAT, KASRA
from .encoder import PersianEncoder


@dataclass
class PersianExample:
    src_ids: list[int]
    haraqat_labels: list[int]
    ezafe_labels: list[int]
    raw_src: str


class PersianDataset(Dataset):
    """Loads diacritized Persian text lines.

    Each line should be a diacritized Persian sentence.
    The encoder strips harakat to produce undiacritized input + per-char labels.
    """

    def __init__(self, data_path: str | Path, max_len: int = 200) -> None:
        self.encoder = PersianEncoder()
        self.max_len = max_len
        self.examples: list[PersianExample] = []

        for line in Path(data_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or len(line) < 5:
                continue

            cleaned = self.encoder.clean(line)
            undiacritized = self.encoder.strip_harakat(cleaned)
            if len(undiacritized) < 2 or len(undiacritized) > max_len:
                continue

            src_ids = self.encoder.encode(undiacritized)[:max_len]
            haraqat_labels = self._extract_haraqat_labels(cleaned, undiacritized)[:max_len]
            ezafe_labels = self._extract_ezafe_labels(cleaned, undiacritized)[:max_len]

            self.examples.append(PersianExample(
                src_ids=src_ids,
                haraqat_labels=haraqat_labels,
                ezafe_labels=ezafe_labels,
                raw_src=undiacritized,
            ))

    def _extract_haraqat_labels(self, diacritized: str, undiacritized: str) -> list[int]:
        labels = [0]  # 0 = no haraka
        from .constants import HARAQAT_LIST
        undiac_idx = 0
        current_label = 0
        for c in diacritized:
            if c in HARAQAT:
                idx = HARAQAT_LIST.index(c) if c in HARAQAT_LIST else 0
                if idx > 0:
                    current_label = idx
            else:
                labels.append(current_label)
                current_label = 0
                undiac_idx += 1
        if current_label > 0 and len(labels) < len(undiacritized) + 1:
            labels.append(current_label)
        while len(labels) < len(undiacritized):
            labels.append(0)
        return labels[:len(undiacritized)]

    def _extract_ezafe_labels(self, diacritized: str, undiacritized: str) -> list[int]:
        """Ezafe = kasra at the end of a word (before space).

        Returns binary labels: 1 = ezafe present, 0 = no ezafe.
        """
        from .constants import KASRA
        labels = [0] * len(undiacritized)
        undiac_idx = 0
        in_word = False
        last_char_idx = -1

        for c in diacritized:
            if c in HARAQAT:
                if c == KASRA and last_char_idx >= 0:
                    next_is_space = False
                    labels[last_char_idx] = 1
            else:
                if c == ' ':
                    in_word = False
                    last_char_idx = -1
                else:
                    in_word = True
                    last_char_idx = undiac_idx
                    undiac_idx += 1

        return labels[:len(undiacritized)]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {
            "src": torch.tensor(ex.src_ids, dtype=torch.long),
            "haraqat": torch.tensor(ex.haraqat_labels, dtype=torch.long),
            "ezafe": torch.tensor(ex.ezafe_labels, dtype=torch.long),
        }


def collate_batch(batch: list[dict], pad_id: int = 0) -> dict:
    """Pad sequences to batch max length."""
    max_len = max(b["src"].size(0) for b in batch)
    B = len(batch)

    src = torch.full((B, max_len), pad_id, dtype=torch.long)
    haraqat = torch.full((B, max_len), -100, dtype=torch.long)
    ezafe = torch.full((B, max_len), -100, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        T = b["src"].size(0)
        src[i, :T] = b["src"]
        haraqat[i, :T] = b["haraqat"]
        ezafe[i, :T] = b["ezafe"]
        lengths[i] = T

    return {"src": src, "haraqat": haraqat, "ezafe": ezafe, "lengths": lengths}
