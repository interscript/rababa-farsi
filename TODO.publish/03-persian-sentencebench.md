# 03 — Persian: evaluate on HomoRich official test (SentenceBench)

## Why
Our Persian G2P reports 89.45% homograph accuracy on our own random split.
HomoRich paper reports 76.89% on their recommended split / SentenceBench.
For a fair SOTA claim we evaluate on their split.

## Tasks
- [x] Download MahtaFetrat/SentenceBench (or HomoRich recommended_split test CSV)
- [x] Run our Persian G2P model on their test sentences
- [x] Compute homograph accuracy with their protocol
- [x] Record result

## Result
See rababa-farsi/docs/RESULTS.md.
