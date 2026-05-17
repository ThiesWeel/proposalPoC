# OU Admissibility Detection — Phase 3C PoC

**Notebook:** https://github.com/ThiesWeel/proposalPoC/blob/master/OU_Admissibility_CNN_Phase3C.ipynb

Can a CNN learn, from the raw shape of a 126-day financial return window, whether an OU/AR(1) model fits better than a random walk? This repo is a proof of concept for the detection layer of a two-stage OU trading framework.

## Notebook

**[OU_Admissibility_CNN_Phase3C.ipynb](OU_Admissibility_CNN_Phase3C.ipynb)** — full results and diagnostic validation, including:
- mathematical setup and label construction
- CNN architecture and training details
- out-of-sample evaluation (unseen tickers, post-2020 era)
- real vs. synthetic AR(1) diagnostic comparison across the score spectrum
- detrending robustness check

## Key results (Phase 3-C test set)

| Metric | Value |
|---|---:|
| LR baseline AUC | 0.9165 |
| CNN AUC @ threshold 0.03 | **0.9871** |
| CNN Average Precision | 0.9634 |
| CNN R² (regression) | 0.8926 |
| Delta AUC (CNN - LR) | **+0.0706** |

Test set: 14,058 windows from 50 tickers never seen during training, evaluated on post-2020 data only (joint out-of-sample on both cross-section and time).

## Split design

```
Train : 80% of tickers, windows before 2020-01-01   (511,671 windows)
Val   : 10% of tickers, windows before 2020-01-01   ( 61,136 windows)
Test  : 10% of tickers, windows from 2020-01-01 on  ( 14,058 windows)
```

## Repo structure

```
train_phase3_C.py          training script (self-contained)
OU_Admissibility_CNN_Phase3C.ipynb   results notebook
models/best_model_phase3_C.pt        trained weights
data/train_gen_CNN/                  FF5 portfolio CSVs + extracted windows
results/phase3_C/summary.md          numerical summary
figures/phase3_C/                    output figures
```

## Requirements

```
pip install -r requirements.txt
```
