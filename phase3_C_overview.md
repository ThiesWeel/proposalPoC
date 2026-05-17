# Phase 3-C: CNN-based OU-Admissibility Detector

## 1. Problem

The goal is to predict, from a fixed-length window of a financial return series, how much better an OU (Ornstein-Uhlenbeck / AR(1)) model fits compared to a random-walk. The model outputs a continuous score. A binary yes/no decision is only made at evaluation time by applying a threshold.

---

## 2. Data

Two data sources are used:

- **Individual stocks** - rolling 252-day FF5 OLS residuals, Jensen-corrected and cumulated into idiosyncratic log-index series
- **Fama-French sorted portfolios** - 381 portfolios covering 1990-2026

Each window has length $W = 126$ trading days, extracted with a step of 5 days, giving roughly 786,000 windows in total.

---

## 3. Train / Validation / Test Split

A standard random ticker split was used (80/10/10), but with an extra time constraint to make the test set harder. Tickers are shuffled once (seed 42) and divided:

$$
\text{tickers} \xrightarrow{\text{shuffle (seed 42)}} \underbrace{80\%}_{\text{train}} \;/\; \underbrace{10\%}_{\text{val}} \;/\; \underbrace{10\%}_{\text{test}}
$$

On top of that, a time boundary is applied:

| Set | Tickers | Window period |
|---|---|---|
| Train | 80% | window_mid before 2020-01-01 |
| Val   | 10% | window_mid before 2020-01-01 |
| **Test** | **10%** | window_mid from 2020-01-01 onwards |

So the test set has both unseen tickers and a completely different time period (post-2020). This is more conservative than a simple random split.

| Split | Windows |
|---|---:|
| Train | 511,671 |
| Val   |  61,136 |
| Test  |  14,058 |

---

## 4. Label Construction

Let $\{Y_t\}_{t=1}^{W}$ be the Z-scored window (normalized per window, not globally).

The AR(1) coefficient is estimated by OLS:

$$
\hat{\phi} = \frac{\sum_{t=1}^{W-1} Y_t \, Y_{t+1}}{\sum_{t=1}^{W-1} Y_t^2}
$$

One-step-ahead forecast errors under each model:

$$
e_t^{\text{RW}} = Y_{t+1} - Y_t
$$

$$
e_t^{\text{OU}} = Y_{t+1} - \hat{\phi}\,Y_t
$$

The relative MAE improvement of OU over RW:

$$
\Delta\text{MAE}_{\text{rel}} = \frac{\text{MAE}_{\text{RW}} - \text{MAE}_{\text{OU}}}{\text{MAE}_{\text{RW}} + 10^{-12}}
$$

This is the regression target, clipped to $[-0.05,\; 0.30]$ during training to limit the effect of outliers.

A binary label is derived for evaluation only:

$$
z = \mathbf{1}\bigl\{\Delta\text{MAE}_{\text{rel}} > 0.03\bigr\}
$$

---

## 5. Input Representation

Three channels are computed from each Z-scored window $Y_1, \dots, Y_W$:

| Channel | Definition | Rationale |
|---|---|---|
| $c_0$ | $Y_t$ | Overall shape and mean-reversion envelope |
| $c_1$ | $\Delta_t = Y_t - Y_{t-1}$, with $\Delta_1 = 0$ | OU increments are negatively autocorrelated, RW increments are not |
| $c_2$ | $\Delta_t^2$ | Captures volatility clustering |

The model input is $\mathbf{X} \in \mathbb{R}^{3 \times 126}$. There is no Yule-Walker fitting or any other hand-crafted feature in the preprocessing; the CNN sees only the raw series in these three representations.

---

## 6. Model Architecture

A 1D convolutional network with roughly 150,000 parameters. The forward pass is:

$$
\mathbf{X} \xrightarrow{\text{conv blocks}} \mathbf{H} \in \mathbb{R}^{128 \times 126} \xrightarrow{\text{GAP}} \mathbf{h} \in \mathbb{R}^{128} \xrightarrow{\text{MLP}} \hat{y} \in \mathbb{R}
$$

**Convolutional stack:**

| Layer | Channels | Kernel | Dilation |
|---|---|---|---|
| Conv1 | $3 \to 32$    | 4  | 1 |
| Conv2 | $32 \to 64$   | 8  | 1 |
| Conv3 | $64 \to 128$  | 16 | 1 |
| Conv4 | $128 \to 128$ | 8  | 2 |

All layers use `padding="same"`. Each is followed by BatchNorm and ReLU. The kernel sizes grow (4, 8, 16) to progressively increase the receptive field. The last layer uses dilation 2 to widen it further without adding parameters.

**Pooling:** Global Average Pooling collapses the time dimension, giving a single 128-dimensional vector per window.

**Head:**

$$
\mathbb{R}^{128} \xrightarrow{\text{Linear}} \mathbb{R}^{64} \xrightarrow{\text{ReLU}} \xrightarrow{\text{Dropout}(0.2)} \mathbb{R}^{32} \xrightarrow{\text{ReLU}} \mathbb{R}^{1}
$$

---

## 7. Training

**Loss function:** Huber loss with $\delta = 0.02$:

$$
\mathcal{L}_\delta(r) =
\begin{cases}
\tfrac{1}{2} r^2 & |r| \leq \delta \\
\delta\bigl(|r| - \tfrac{1}{2}\delta\bigr) & |r| > \delta
\end{cases}
$$

where $r = \hat{y} - y_{\text{clipped}}$. The small $\delta$ keeps the loss close to MAE behaviour, which is useful here because most targets are near zero and only a small fraction of windows have large positive $\Delta\text{MAE}_{\text{rel}}$.

**Optimizer:** Adam, initial learning rate $\eta_0 = 10^{-3}$.

**Learning rate schedule:** Cosine annealing over $T_{\max} = 30$ epochs down to $\eta_{\min} = 10^{-5}$:

$$
\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_0 - \eta_{\min})\!\left(1 + \cos\frac{\pi t}{T_{\max}}\right)
$$

**Early stopping:** patience of 5 epochs on validation Huber loss. Best checkpoint was at epoch 14 (val $R^2 = 0.910$).

**Batch size:** 512.

---

## 8. Baseline

A logistic regression on four scalar features computed per window:

$$
\text{features} = \bigl[\hat{\phi}_{\text{clipped}},\; \text{hit\_rate},\; \text{zero\_crossings},\; |\text{trend\_tstat}|\bigr]
$$

These summarize the most obvious linear OU signal. If the CNN does not beat this, the extra complexity is not worth it.

---

## 9. Evaluation Metrics

Computed on the test set only (unseen tickers, post-2020 windows).

| Metric | Definition |
|---|---|
| $R^2$ | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ on continuous target |
| AUC | ROC-AUC with $\hat{y}$ as score and $z$ as label at threshold 0.03 |
| AP | Average Precision (area under precision-recall curve) |

---

## 10. Results

| Metric | Value |
|---|---:|
| LR baseline AUC | 0.9165 |
| CNN $R^2$ (test) | 0.8926 |
| CNN AUC @ 0.03  | 0.9871 |
| CNN AP          | 0.9634 |
| $\Delta$ AUC (CNN - LR) | +0.0706 |

The CNN reaches AUC 0.987, compared to 0.917 for the logistic regression baseline, a gap of about 7 AUC points. This holds on test windows from tickers and a time period that were fully held out during training, which suggests the model is picking up on genuine path-shape features rather than overfitting to specific tickers or the pre-2020 regime.
