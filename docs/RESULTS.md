# rababa-farsi — SOTA Results (Persian G2P + diacritization)

All numbers from the 2026-08 SOTA campaign on real Persian data
(HomoRich-G2P-Persian, 528K sentences).

## G2P (Persian text → Latin phonemes)

### Best result (ByT5-small)

| Metric | Value | Test set |
|---|---|---|
| PER (word-level) | 6.18% | 10,576 held-out |
| **CER (char-level)** | **1.64%** | 10,576 held-out |
| Exact match | 63.8% | 10,576 held-out |
| **Homograph accuracy** | **89.45%** | 6,529 homograph-bearing sentences in split |

ByT5-base (1.2B) gave identical quality (CER 1.62%, PER 6.24%, EM 63.4%) —
the task is saturated by ByT5-small (300M).

- Checkpoint: `/checkpoints/persian_g2p/run-001/best`
- Eval: `modal run modal_app.py::evaluate` (run ap-234QpyIs..., ap-47CYSFDRDp...)
- HA eval: `eval_homograph.py` — aligns homograph words by position, compares
  predicted pronunciation to reference

### Comparison

| System | Homograph accuracy | Test |
|---|---|---|
| **ours (ByT5 on HomoRich)** | **89.45%** | our 528K random split |
| **ours, ezafe-normalized** | **77.34%** (157/203) | SentenceBench (their split) |
| ours, exact match | 59.11% (120/203) | SentenceBench |
| Homo-GE2PE (published SOTA) | 76.89% | SentenceBench (their split) |
| HomoFast eSpeak (rule-based) | 74.53% | SentenceBench |

SentenceBench protocol (eval_sentencebench.py): word-position alignment of
the homograph; exact string match against reference pronunciation. The
ezafe-normalized variant strips a trailing /e/ from the prediction (our
model attaches the Persian ezafe connective to the word: qadr-e → "qadre";
the reference stores bare stems "qadar"). With this single normalization
our untuned model matches the published SOTA on its own curated hard set;
on the natural-distribution split it reaches 89.45%.

## Diacritization (Persian text → text + haraqat)

### Best result

| Metric | Value | Test set |
|---|---|---|
| **CER** | **0.52%** | 9,288 held-out |

- Derived labels: HomoRich `Mapped Phoneme` field → haraqat via deterministic
  byte-pair parsing (scripts/extract_diacritics.py, 464K usable pairs)
- Model: ByT5-base, 3 epochs (modal_app_diacrit.py)
- Checkpoint: `/checkpoints/persian_diacrit/run-001/best`
- 99.48% of characters correctly diacritized

This is, to our knowledge, the first Persian diacritization trained from
HomoRich — the dataset was published for G2P only; we show its `Mapped
Phoneme` field deterministically yields full haraqat annotation.

## Key findings

1. **HomoRich dual-use**: one dataset, two tasks — G2P (89.45% HA) and
   diacritization (0.52% CER) with zero additional annotation.
2. **Model-size saturation**: ByT5-small ≈ ByT5-base for this task family.
3. **Metric non-comparability is endemic**: HomoRich's 76.89% scores one
   homograph per sentence; PER/CER score everything. Comparisons require
   implementing the HA protocol explicitly.
