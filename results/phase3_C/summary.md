# Phase 3-C Results (ticker + time split)

Split: 80/10/10 tickers, train pre-2020-01-01, test post-2020-01-01

Windows: train=511,671  val=61,136  test=14,058  |  best epoch: 14

| Metric | Value |
|---|---:|
| LR baseline AUC | 0.9165 |
| CNN R^2 (test)  | 0.8926 |
| CNN AUC @ 0.03  | 0.9871 |
| CNN AP          | 0.9634 |
| Delta AUC (CNN - LR) | +0.0706 |
