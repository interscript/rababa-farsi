"""Persian text encoder — character-level encoding for diacritization.

Takes diacritized Persian text, strips harakat to get undiacritized form,
and maps each character to an integer ID. Also extracts per-character
haraka labels for training.
"""

from __future__ import annotations

from .constants import (
    HARAQAT, HARAQAT_LIST, STRIP_CHARS, SHADDA, PAD_ID,
)


class PersianEncoder:
    """Character-level encoder for Persian diacritization."""

    def __init__(self) -> None:
        chars = sorted(set(
            "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیةيك"
            "0123456789"
            " \t\n،.؟!؛-«»()[]«»"
            "‌"
        ))
        self.char_to_id: dict[str, int] = {PAD_ID_CHAR: 0}
        for i, c in enumerate(chars, 1):
            self.char_to_id[c] = i
        self.id_to_char: dict[int, str] = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id) + 1

    def clean(self, text: str) -> str:
        out = []
        for c in text:
            if c in STRIP_CHARS:
                continue
            out.append(c)
        return "".join(out)

    def encode(self, text: str) -> list[int]:
        text = self.clean(text)
        return [self.char_to_id.get(c, 1) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char.get(i, "") for i in ids)

    def strip_harakat(self, text: str) -> str:
        return "".join(c for c in text if c not in HARAQAT)

    def extract_labels(self, text: str) -> tuple[str, list[int]]:
        undiacritized = []
        labels = []
        for c in text:
            if c in HARAQAT:
                if c == SHADDA and labels:
                    pass
                else:
                    idx = HARAQAT_LIST.index(c) if c in HARAQAT_LIST else 0
                    if labels:
                        labels[-1] = idx
                    continue
            else:
                undiacritized.append(c)
                labels.append(0)
        return "".join(undiacritized), labels


PAD_ID_CHAR = "\0"
