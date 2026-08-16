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

## G2P v3–v5 ablation series (2026-08-16)

| Run | Config | SB ezafe-norm | Our-split CER | Note |
|---|---|---|---|---|
| v1 (standing best) | plain Phoneme, 3ep linear | **77.34%** | 1.64% | production model |
| v3 | dual-task + cosine 3ep | 74.75% | 1.40% | dual-task hurts homographs |
| v4 | plain, 5ep cosine | 74.88% | 1.43% | more training hurts homographs |
| v5 | Mapped Phoneme (Repr 2) | 28.22%* | 16.65%* | *repr mismatch: @/? tokens, syllable marks survive two normalization passes |

Findings: (1) homograph accuracy peaks at v1's early-stopped linear recipe
— both longer training and auxiliary-task regularization trade ~2.5 SB
points for better CER; (2) the two HomoRich representations are NOT
interchangeable under simple normalization — scoring a Repr-2 model
against plain refs requires their exact decoding table; recorded as a
metric-interoperability negative result for the paper. Next lever: RAG
homograph evidence on v1 (TODO.research/05), not more recipe roulette.
