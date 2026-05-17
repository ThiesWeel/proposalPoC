"""
Phase 3 — Ticker + Time split validation.

Same model and training loop as train_phase3.py.
Split: tickers randomly divided 80/10/10, PLUS a time boundary:
  train : 80% of tickers, window_mid < 2020-01-01
  val   : 10% of tickers, window_mid < 2020-01-01
  test  : 10% of tickers, window_mid >= 2020-01-01  (unseen tickers, new era)

This answers: does the model generalise to BOTH new tickers AND a new time period?

Run:
    python train_phase3_C.py
"""
import sys, json, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── config ─────────────────────────────────────────────────────────────────────
STOCK_PATH = Path("data/train_gen_CNN/windows/stock_windows.parquet")
FF_PATH    = Path("data/train_gen_CNN/windows/ff_portfolio_windows.parquet")
OUT_DIR    = Path("results/phase3_C")
FIG_DIR    = Path("figures/phase3_C")
MODEL_PATH = Path("models/best_model_phase3_C.pt")
HIST_PATH  = Path("models/history_phase3_C.json")

TEST_CUTOFF = "2020-01-01"   # test tickers: only post-2020 windows
                              # train/val tickers: only pre-2020 windows

THRESHOLD  = 0.03
CLIP_LO    = -0.05
CLIP_HI    = 0.30

BATCH      = 512
MAX_EPOCHS = 30
PATIENCE   = 5
LR_INIT    = 1e-3
LR_MIN     = 1e-5
HUBER_D    = 0.02

SERIES_COLS = [f"series_{k:03d}" for k in range(126)]


# ── model (identical to train_phase3.py) ───────────────────────────────────────

class Phase3CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(3,   32,  kernel_size=4,  padding="same"), nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Conv1d(32,  64,  kernel_size=8,  padding="same"), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,  128, kernel_size=16, padding="same"), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=8,  padding="same",
                      dilation=2),                               nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,  32), nn.ReLU(),
            nn.Linear(32,   1),
        )

    def forward(self, x):
        return self.head(self.pool(self.convs(x))).squeeze(-1)


# ── dataset ────────────────────────────────────────────────────────────────────

class Phase3Dataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        s        = df[SERIES_COLS].values.astype(np.float32)
        d        = np.empty_like(s)
        d[:, 0]  = 0.0
        d[:, 1:] = np.diff(s, axis=1)
        self.X   = torch.from_numpy(np.stack([s, d, d ** 2], axis=1))
        raw      = df["rel_delta_mae"].values.astype(np.float32)
        self.y_reg = torch.from_numpy(np.clip(raw, CLIP_LO, CLIP_HI))
        self.y_bin = (raw > THRESHOLD).astype(np.float32)

    def __len__(self):  return len(self.y_reg)
    def __getitem__(self, i): return self.X[i], self.y_reg[i]


# ── training helpers ───────────────────────────────────────────────────────────

def run_epoch(model, loader, crit, opt, device, training):
    model.train(training)
    tot_loss, all_pred, all_true = 0.0, [], []
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = crit(pred, yb)
            if training:
                opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * len(yb)
            all_pred.append(pred.detach().cpu().numpy())
            all_true.append(yb.cpu().numpy())
    pred_np = np.concatenate(all_pred)
    true_np = np.concatenate(all_true)
    r2 = float(1 - np.sum((true_np - pred_np)**2) / (np.sum((true_np - true_np.mean())**2) + 1e-12))
    return tot_loss / len(true_np), r2


@torch.no_grad()
def predict(model, ds, device):
    model.eval()
    preds = []
    for xb, _ in DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0):
        preds.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(preds)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────────────
    print("Loading windows...")
    df = pd.concat([
        pd.read_parquet(STOCK_PATH),
        pd.read_parquet(FF_PATH),
    ], ignore_index=True)
    df["window_mid"] = pd.to_datetime(df["window_mid"])
    df["label_bin"]  = (df["rel_delta_mae"] > THRESHOLD).astype(int)
    df["trend_tstat_abs"] = df["trend_tstat"].abs()
    print(f"  n={len(df):,}  tickers={df['ticker'].nunique()}  "
          f"pos_rate={df.label_bin.mean():.3f}  "
          f"date range: {df['window_mid'].min().date()} - {df['window_mid'].max().date()}")

    # ── ticker + time split ───────────────────────────────────────────────────
    print(f"\nSplit: ticker (80/10/10) + time cutoff {TEST_CUTOFF}")
    tickers = np.array(df["ticker"].unique())
    rng = np.random.default_rng(42)
    rng.shuffle(tickers)
    n = len(tickers)
    train_tickers = set(tickers[:int(0.8 * n)])
    val_tickers   = set(tickers[int(0.8 * n):int(0.9 * n)])
    test_tickers  = set(tickers[int(0.9 * n):])

    cutoff = pd.Timestamp(TEST_CUTOFF)

    # train: 80% of tickers, pre-cutoff windows only
    tr = df[df["ticker"].isin(train_tickers) & (df["window_mid"] < cutoff)].reset_index(drop=True)
    # val:   10% of tickers, pre-cutoff windows only
    va = df[df["ticker"].isin(val_tickers)   & (df["window_mid"] < cutoff)].reset_index(drop=True)
    # test:  10% of tickers, post-cutoff windows only (unseen tickers in new era)
    te = df[df["ticker"].isin(test_tickers)  & (df["window_mid"] >= cutoff)].reset_index(drop=True)

    print(f"  Tickers   train={len(train_tickers)}  val={len(val_tickers)}  test={len(test_tickers)}")
    print(f"  Windows   train={len(tr):,}  val={len(va):,}  test={len(te):,}")
    print(f"  Train period: {tr['window_mid'].min().date()} - {tr['window_mid'].max().date()}")
    print(f"  Test  period: {te['window_mid'].min().date()} - {te['window_mid'].max().date()}")
    print(f"  Test pos_rate: {te['label_bin'].mean():.3f}")

    # ── LR baseline ───────────────────────────────────────────────────────────
    print("\n--- LR baseline ---")
    feat_cols = ["phi_clipped", "hit_rate", "zero_crossings", "trend_tstat_abs"]
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(tr[feat_cols].fillna(0).values, tr["label_bin"].values)
    p_lr   = lr.predict_proba(te[feat_cols].fillna(0).values)[:, 1]
    auc_lr = roc_auc_score(te["label_bin"].values, p_lr)
    print(f"  LR baseline AUC = {auc_lr:.4f}")

    # ── datasets + loaders ────────────────────────────────────────────────────
    print("\nBuilding CNN datasets...")
    train_ds = Phase3Dataset(tr)
    val_ds   = Phase3Dataset(va)
    test_ds  = Phase3Dataset(te)
    tl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=True)
    vl = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

    # ── train ─────────────────────────────────────────────────────────────────
    print("\n--- Training ---")
    model = Phase3CNN().to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    crit  = nn.HuberLoss(delta=HUBER_D)
    opt   = torch.optim.Adam(model.parameters(), lr=LR_INIT)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=LR_MIN)

    best_val, no_imp, history = np.inf, 0, []
    MODEL_PATH.parent.mkdir(exist_ok=True)

    for ep in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_r2 = run_epoch(model, tl, crit, opt,  device, training=True)
        vl_loss, vl_r2 = run_epoch(model, vl, crit, None, device, training=False)
        sched.step()
        flag = ""
        if vl_loss < best_val:
            best_val, no_imp = vl_loss, 0
            torch.save(model.state_dict(), MODEL_PATH)
            flag = " *"
        else:
            no_imp += 1
        lr_now = opt.param_groups[0]["lr"]
        print(f"ep{ep:3d} | tr_r2={tr_r2:.4f}  vl_r2={vl_r2:.4f}  "
              f"vl_loss={vl_loss:.5f}  lr={lr_now:.2e}  {time.time()-t0:.1f}s{flag}")
        history.append(dict(epoch=ep, train_r2=tr_r2, val_r2=vl_r2, val_loss=vl_loss))
        if no_imp >= PATIENCE:
            print(f"  Early stop (best val_loss={best_val:.5f})")
            break

    with open(HIST_PATH, "w") as f:
        json.dump(history, f, indent=2)

    # ── evaluate ──────────────────────────────────────────────────────────────
    print("\n--- Evaluation on test set ---")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    pred_raw = predict(model, test_ds, device)

    y_true = te["rel_delta_mae"].values
    y_bin  = te["label_bin"].values

    r2      = float(1 - np.sum((y_true - pred_raw)**2) / (np.sum((y_true - y_true.mean())**2) + 1e-12))
    auc_cnn = roc_auc_score(y_bin, pred_raw)
    ap_cnn  = average_precision_score(y_bin, pred_raw)

    print(f"\n  LR baseline AUC  = {auc_lr:.4f}")
    print(f"  CNN R^2          = {r2:.4f}")
    print(f"  CNN AUC @ 0.03   = {auc_cnn:.4f}  (delta = {auc_cnn - auc_lr:+.4f} vs LR)")
    print(f"  CNN AP           = {ap_cnn:.4f}")

    for src in te["source"].unique():
        m = te["source"].values == src
        a = roc_auc_score(y_bin[m], pred_raw[m]) if len(np.unique(y_bin[m])) > 1 else float("nan")
        print(f"  [{src:<14}]  n={m.sum():6,}  auc={a:.4f}")

    # ── quick plots ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # calibration
    bins = np.linspace(CLIP_LO, CLIP_HI, 20)
    idx  = np.digitize(pred_raw, bins) - 1
    xs, ys = [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() > 10:
            xs.append(pred_raw[m].mean()); ys.append(y_true[m].mean())
    axes[0].scatter(xs, ys, s=40)
    lo, hi = min(xs + ys) - 0.01, max(xs + ys) + 0.01
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=0.8)
    axes[0].set_xlabel("Mean predicted"); axes[0].set_ylabel("Mean actual")
    axes[0].set_title("Calibration")
    # score dist
    bins2 = np.linspace(pred_raw.min(), pred_raw.max(), 60)
    axes[1].hist(pred_raw[y_bin == 0], bins=bins2, alpha=0.6, density=True,
                 color="#B71C1C", label="RW-dominant")
    axes[1].hist(pred_raw[y_bin == 1], bins=bins2, alpha=0.6, density=True,
                 color="#1565C0", label="OU-admissible")
    axes[1].axvline(THRESHOLD, color="k", lw=1, ls="--")
    axes[1].set_title("Score distribution"); axes[1].legend()
    plt.suptitle(f"Phase 3-C  AUC={auc_cnn:.4f}  R2={r2:.4f}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eval.png", bbox_inches="tight", dpi=110)
    plt.close()

    # ── save summary ──────────────────────────────────────────────────────────
    best_ep = history[int(np.argmin([h["val_loss"] for h in history]))]["epoch"]
    summary = (
        "# Phase 3-C Results (ticker + time split)\n\n"
        f"Split: 80/10/10 tickers, train pre-{TEST_CUTOFF}, test post-{TEST_CUTOFF}\n\n"
        f"Windows: train={len(tr):,}  val={len(va):,}  test={len(te):,}  |  best epoch: {best_ep}\n\n"
        "| Metric | Value |\n|---|---:|\n"
        f"| LR baseline AUC | {auc_lr:.4f} |\n"
        f"| CNN R^2 (test)  | {r2:.4f} |\n"
        f"| CNN AUC @ 0.03  | {auc_cnn:.4f} |\n"
        f"| CNN AP          | {ap_cnn:.4f} |\n"
        f"| Delta AUC (CNN - LR) | {auc_cnn - auc_lr:+.4f} |\n"
    )
    (OUT_DIR / "summary.md").write_text(summary, encoding="utf-8")
    print(f"\nSaved: {MODEL_PATH}  |  {OUT_DIR}/summary.md  |  {FIG_DIR}/eval.png")
    print("Done.")


if __name__ == "__main__":
    main()
