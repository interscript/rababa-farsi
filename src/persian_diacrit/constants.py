"""Persian/Farsi diacritization constants.

Persian uses Arabic script with harakat (short vowel diacritics) that are
normally omitted. This module defines the character sets and harakat marks.

Persian-specific additions:
  - Letters: پ چ ژ گ (not in Arabic)
  - Ezafe: kasra (ـِ) at word boundary = /e/ connective
  - Persian uses the same harakat as Arabic: fatha, damma, kasra, sukun, shadda
"""

PERSIAN_LETTERS = set("ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")
ARABIC_LETTERS_USED_IN_PERSIAN = set("ةيﻙﻭ")
ALL_LETTERS = PERSIAN_LETTERS | ARABIC_LETTERS_USED_IN_PERSIAN

# Harakat (diacritics) — same Unicode points as Arabic
FATHA = "َ"      # ـَ  (a)
DAMMA = "ُ"      # ـُ  (u)
KASRA = "ِ"      # ـِ  (i) — also used for ezafe
SUKUN = "ْ"      # ـْ  (no vowel)
SHADDA = "ّ"     # ـّ  (gemination)
FATHATAN = "ً"   # ـً  (an)
DAMMATAN = "ٌ"   # ـٌ  (un)
KASRATAN = "ٍ"   # ـٍ  (in)
DAGGER = "ٰ"     # ـٰ  (superscript alef)
HAMZA_ABOVE = "ٔ"

HARAQAT = {
    FATHA, DAMMA, KASRA, SUKUN, SHADDA,
    FATHATAN, DAMMATAN, KASRATAN, DAGGER,
}

# Ordered list for classification (index = class ID)
HARAQAT_LIST = [
    "",          # 0: no haraka
    FATHA,       # 1: a
    DAMMA,       # 2: u
    KASRA,       # 3: i / ezafe
    SUKUN,       # 4: no vowel
    SHADDA,      # 5: gemination
    FATHATAN,    # 6: tanwin an
    DAMMATAN,    # 7: tanwin un
    KASRATAN,    # 8: tanwin in
    DAGGER,      # 9: superscript alef
]

NUM_HARAQAT = len(HARAQAT_LIST)

# Shadda is special — it doubles the consonant and combines with a vowel
SHADDA_COMBOS = {SHADDA + h: i + NUM_HARAQAT for i, h in enumerate(HARAQAT_LIST[1:], 1)}
# This gives us composite labels like SHADDA+FATHA, SHADDA+DAMMA, etc.

# TATWEEL (kashida) — decorative elongation, strip
TATWEEL = "ـ"

# Characters to strip (not letters, not harakat)
STRIP_CHARS = set("‌‍‎‏") | {TATWEEL}

# ZWNJ (zero-width non-joiner) — common in Persian, affects word boundaries
ZWNJ = "‌"

PAD_ID = 0

# Ezafe marker — kasra at end of word that connects to next word
EZAFA_HARAKA = KASRA
