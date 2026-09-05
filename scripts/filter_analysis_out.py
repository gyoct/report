#!/usr/bin/env python3
"""Window a product's analyze-side artifacts to ts >= SINCE, faithfully.

Reads <full_analysis_out>/resampled.parquet, filters to ts >= SINCE, and recomputes
step5..step10 with the SAME formulas as analyze_alpha.py, writing resampled.parquet +
step5/6/7/8/9/10 CSVs into <out_dir>. The order-side artifacts (order_hourly,
cum_pnl_commission, position_midprice, trade_markout, holding_pnl) are produced
separately by order_multi.R with ORDER_SINCE_UTC. build_report.py then renders both.

Usage: python filter_analysis_out.py <full_analysis_out> <SINCE iso> <out_dir>
"""
import os, sys, numpy as np, pandas as pd

FULL = sys.argv[1]
SINCE = pd.Timestamp(sys.argv[2])
OUT = sys.argv[3]
os.makedirs(OUT, exist_ok=True)


def quantile_grid():
    q = np.concatenate([np.arange(0, 1, 0.1), np.arange(1, 10, 1), np.arange(10, 90, 10),
                        np.arange(90, 99, 1), np.arange(99, 100, 0.1), [100.0]])
    return np.unique(np.round(q, 4))


df = pd.read_parquet(os.path.join(FULL, "resampled.parquet"))
df = df[pd.to_datetime(df["ts"]) >= SINCE].copy()
if df.empty:
    sys.exit(f"no resampled rows >= {SINCE}")
df.to_parquet(os.path.join(OUT, "resampled.parquet"))            # windowed contribution source

contrib_cols = [c for c in df.columns if c.startswith("contrib_")]
feat_cols = [c[len("contrib_"):] for c in contrib_cols]
valid = df.dropna(subset=["fwd_ret", "alpha"]).copy()

# Step 5: corr(alpha, fwd_ret) by hour
g = valid.groupby("hour")
corr_alpha = g.apply(lambda x: x["alpha"].corr(x["fwd_ret"]), include_groups=False)
cnt = g.size()
(corr_alpha.rename("corr").to_frame().assign(n=cnt)
 .to_csv(os.path.join(OUT, "step5_corr_alpha_by_hour.csv")))

# Step 6: mean |contribution share| = |contrib/alpha| averaged per feature
fr = valid[contrib_cols].div(valid["alpha"].values, axis=0)
fr.columns = feat_cols
rank = fr.abs().mean().sort_values(ascending=False)
rank.to_csv(os.path.join(OUT, "step6_mean_abs_contribution_share.csv"))

# Step 7: corr(col*coef, fwd_ret) by hour -> matrix (feat x hour)
hours = sorted(valid["hour"].unique())
mat = pd.DataFrame(index=feat_cols, columns=[pd.Timestamp(h) for h in hours], dtype=float)
for h in hours:
    sub = valid[valid["hour"] == h]
    for c, cc in zip(feat_cols, contrib_cols):
        mat.loc[c, pd.Timestamp(h)] = sub[cc].corr(sub["fwd_ret"])
mat.columns = [pd.Timestamp(h).strftime("%m-%d %H:%M") for h in hours]
mat.to_csv(os.path.join(OUT, "step7_corr_feature_by_hour.csv"))

# Step 8 / 9: quantile tables
qg = quantile_grid()
for series, name in ((valid["fwd_ret"], "step8_return"), (valid["alpha"], "step9_alpha")):
    s = series.dropna().values
    pd.DataFrame({"percentile": qg, "value": np.percentile(s, qg)}).to_csv(
        os.path.join(OUT, f"{name}_quantiles.csv"), index=False)

# Step 10: corr(alpha, fwd_ret) bucketed by abs(alpha) percentile
v = valid.copy()
v["abs_alpha"] = v["alpha"].abs()
raw = np.percentile(v["abs_alpha"].values, qg)
edges, edge_pct = [], []
for p, e in zip(qg, raw):
    if not edges or e > edges[-1]:
        edges.append(e); edge_pct.append(p)
edges = np.array(edges)
v["bucket"] = pd.cut(v["abs_alpha"], bins=edges, include_lowest=True)
grp = v.groupby("bucket", observed=False)
size, m_aa, m_rt = grp.size(), grp["abs_alpha"].mean(), grp["fwd_ret"].mean()
corr = grp.apply(lambda gg: gg["alpha"].corr(gg["fwd_ret"]) if len(gg) > 1 else np.nan,
                 include_groups=False)
m_dir = grp.apply(lambda gg: float((np.sign(gg["alpha"]) * gg["fwd_ret"]).mean()),
                  include_groups=False)  # dir_return = mean(sign(alpha)*ret)
rows = []
for i, interval in enumerate(v["bucket"].cat.categories):
    p_lo, p_hi = edge_pct[i], edge_pct[i + 1]
    rows.append({"pct_lo": p_lo, "pct_hi": p_hi, "pct_range": f"{p_lo:g}-{p_hi:g}",
                 "abs_alpha_lo": interval.left, "abs_alpha_hi": interval.right,
                 "n": int(size.loc[interval]), "corr_alpha_ret": corr.loc[interval],
                 "mean_abs_alpha": m_aa.loc[interval], "mean_ret": m_rt.loc[interval],
                 "dir_return": m_dir.loc[interval]})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "step10_corr_by_absalpha_quantile.csv"), index=False)

print(f"wrote windowed analyze artifacts to {OUT} | rows {len(df):,} valid {len(valid):,} "
      f"| {len(hours)} hours since {SINCE.date()}")
