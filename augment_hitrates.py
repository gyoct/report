#!/usr/bin/env python3
"""Regenerate every ic_by_pred_quantile_ret_<h>.csv (linear & lgbm, all symbols)
UNIFORMLY from the validation set X_dt_valid_ret_<h>.csv (`target` + `Y_hat_<h>`),
reproducing pred_diagnostics.ic_by_pred_quantile exactly AND adding two columns:

  hit_rate            = mean(sign(pred)==sign(real))              0-move = MISS
  hit_rate_with_zero  = mean((sign(pred)==sign(real)) | real==0)  0-move = HIT
  hit_rate_move       = mean(sign(pred)==sign(real) | real!=0)    moved rows only

X_dt_valid IS the exact (y_valid, y_pred_valid) the trainer scored, so this
reproduces the pipeline's own ic_pearson/ic_spearman/hit_rate and cleans up schema
drift (padded headers, btc-lgbm's earlier mask-based regen, missing columns).
"""
import csv, os, numpy as np, pandas as pd

STATS = "/home/guanyang/work/alpha_replayer_config/statistics"
SYMS, MODELS, HZ = ["btc", "eth", "sol"], ["linear", "lgbm"], [1, 10, 30, 60]
PROBS = np.array([0.0, .10, .25, .50, .75, .90, .95, .96, .97, .98, .985, .99, .995, 1.0])
COLS = ["bin", "q_lo", "q_hi", "n", "abs_pred_min", "abs_pred_mean", "abs_pred_max",
        "real_mean", "real_median", "dir_return", "ic_pearson", "ic_spearman",
        "hit_rate", "hit_rate_with_zero", "hit_rate_move"]


def regen(sym, model, h):
    d = os.path.join(STATS, sym, model)
    xv = os.path.join(d, f"X_dt_valid_ret_{h}.csv")
    ic = os.path.join(d, f"ic_by_pred_quantile_ret_{h}.csv")
    if not os.path.isfile(xv):
        return f"skip {sym}/{model}/ret_{h}: no X_dt_valid"
    df = pd.read_csv(xv, usecols=["target", f"Y_hat_{h}"], engine="c")
    y = df["target"].to_numpy(float); p = df[f"Y_hat_{h}"].to_numpy(float)
    keep = np.isfinite(y) & np.isfinite(p); y, p = y[keep], p[keep]
    ap = np.abs(p)
    # canonical binning
    raw = np.quantile(ap, PROBS)
    _, first = np.unique(raw, return_index=True); first = np.sort(first)
    breaks = raw[first]; probs_used = PROBS[first]
    b = pd.cut(ap, bins=breaks, include_lowest=True, labels=False)
    b = np.where(np.isnan(b), 0, b).astype(int)
    b = np.clip(b, 0, breaks.size - 2) + 1
    g = pd.DataFrame({"bin": b, "p": p, "y": y, "ap": ap})
    out = []
    for bi, s in g.groupby("bin", sort=True):
        pv, yv, av = s["p"].to_numpy(), s["y"].to_numpy(), s["ap"].to_numpy()
        n = len(s); sp, sy = np.sign(pv), np.sign(yv); hit = sp == sy; moved = yv != 0
        out.append({
            "bin": bi, "q_lo": probs_used[bi - 1], "q_hi": probs_used[bi], "n": n,
            "abs_pred_min": av.min(), "abs_pred_mean": av.mean(), "abs_pred_max": av.max(),
            "real_mean": yv.mean(), "real_median": float(np.median(yv)),
            "dir_return": (sp * yv).mean(),
            "ic_pearson": float(np.corrcoef(pv, yv)[0, 1]) if n > 1 else np.nan,
            "ic_spearman": float(pd.Series(pv).corr(pd.Series(yv), method="spearman")) if n > 1 else np.nan,
            "hit_rate": float(hit.mean()),
            "hit_rate_with_zero": float((hit | ~moved).mean()),
            "hit_rate_move": float(hit[moved].mean()) if moved.any() else np.nan,
        })
    with open(ic, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader()
        for r in out:
            w.writerow({k: (f"{r[k]:.10g}" if isinstance(r[k], float) else r[k]) for k in COLS})
    return f"ok   {sym}/{model}/ret_{h}  ({len(out)} bins, n={sum(r['n'] for r in out):,})"


if __name__ == "__main__":
    for s in SYMS:
        for m in MODELS:
            for h in HZ:
                print(regen(s, m, h), flush=True)
