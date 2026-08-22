#!/usr/bin/env python3
"""Pipeline model-training comparison page: linear vs LGBM vs MLP, one run family.

Reads <model>_horizon_summary.csv + ic_by_conviction_ret_<h>.csv (+ lgbm
importance / mlp arch) from the unified-pipeline stats dirs
(statistics/<sym>/<model>/20260602/pipeline_sqrtw) and renders
<sym>_model_training.html in the site's dark theme. Models with no artifacts
yet are skipped gracefully — rerun after each training lands.

    python gen_pipeline_models.py            # btc
Then make_index.py to encrypt + publish.
"""
import html
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

HERE = os.path.dirname(os.path.abspath(__file__))
SYM = (sys.argv[1] if len(sys.argv) > 1 else "btc").lower()
# optional data-version arg (e.g. "v2" = extended 20260427-20260813 window):
# reads *_<ver> generation dirs and writes <sym>_model_<ver>.html
VER = (sys.argv[2] if len(sys.argv) > 2 else "").lower()
GEN = "pipeline_sqrtw"
STATS = "/home/guanyang/work/alpha_replayer_config/statistics"
# (key, label, color, model-dir, generation) — tuned passes overlay as extra columns
MODELS = [("lasso_ols", "Linear (LASSO→WLS)", "#818cf8", "lasso_ols", "pipeline_sqrtw"),
          ("lgbm", "LightGBM", "#2dd4bf", "lgbm", "pipeline_sqrtw"),
          # per-horizon Optuna run (2026-08-08, 40 trials x 4 horizons, all symbols);
          # supersedes the earlier BTC-only 30/60 pass (pipeline_sqrtw_tuned3060)
          ("lgbm_tuned", "LGBM tuned (per-horizon)", "#34d399", "lgbm", "pipeline_tuned_all"),
          ("mlp", "MLP (torch)", "#fbbf24", "mlp", "pipeline_sqrtw"),
          ("mlp_tuned", "MLP tuned (1s)", "#f59e0b", "mlp", "pipeline_sqrtw_tuned1")]
if VER:
    MODELS = [(k, l, c, d, f"{g}_{VER}") for k, l, c, d, g in MODELS]
HORIZONS = [1, 10, 30, 60]
OUT = os.path.join(HERE, f"{SYM}_model{('_' + VER) if VER else ''}.html")

def path_of(key, name):
    _k, _l, _c, mdir, gen = next(m for m in MODELS if m[0] == key)
    return f"{STATS}/{SYM}/{mdir}/20260602/{gen}/{name}"

loaded = {}
for key, _l, _c, _d, _g in MODELS:
    p = path_of(key, f"{next(m for m in MODELS if m[0]==key)[3]}_horizon_summary.csv")
    if os.path.isfile(p):
        loaded[key] = pd.read_csv(p).set_index("prediction_horizon")

def fmt(v, n=4):
    try:
        return "—" if v is None or pd.isna(v) else f"{float(v):.{n}f}"
    except (TypeError, ValueError):
        return "—"

# ---- headline comparison table ---------------------------------------------
rows = []
for h in HORIZONS:
    cells = [f"<td class='hz'>ret_{h}</td>"]
    for metric in ("validation_IC", "validation_R2"):
        vals = {k: (float(loaded[k].loc[h, metric]) if k in loaded and h in loaded[k].index else None)
                for k, *_ in MODELS}
        ok = [v for v in vals.values() if v is not None]
        best = max(ok) if ok else None
        for k, *_ in MODELS:
            v = vals[k]
            cls = " class='best'" if v is not None and best is not None and v >= best else ""
            cells.append(f"<td{cls}>{fmt(v)}</td>")
    rows.append("<tr>" + "".join(cells) + "</tr>")

model_heads = "".join(f"<th style='color:{c}'>{html.escape(l)}</th>" for _, l, c, *_ in MODELS)
cmp_table = f"""<table class="cmp">
<tr><th rowspan="2">horizon</th><th colspan="{len(MODELS)}">validation IC</th><th colspan="{len(MODELS)}">validation R&sup2;</th></tr>
<tr>{model_heads}{model_heads}</tr>
{''.join(rows)}
</table>"""

# ---- conviction ladders (ret_10) -------------------------------------------
ladders = []
for key, label, color, _d, _g in MODELS:
    p = path_of(key, "ic_by_conviction_ret_10.csv")
    if not os.path.isfile(p):
        continue
    t = pd.read_csv(p)
    icmax = max(0.35, float(t.ic.max()))
    bars = []
    for _, r in t.iterrows():
        w = max(2, int(100 * float(r.ic) / icmax))
        bars.append(f"<div class='brow'><span class='blab'>d{int(r.tier)}</span>"
                    f"<div class='bar' style='width:{w}%;background:{color}'></div>"
                    f"<span class='bval'>{float(r.ic):.3f}</span></div>")
    ladders.append(f"<div class='ladder'><h3 style='color:{color}'>{html.escape(label)}</h3>"
                   + "".join(bars) + "</div>")

# ---- per-model detail cards ------------------------------------------------
details = []
for key, label, color, _d, _g in MODELS:
    if key not in loaded:
        continue
    s = loaded[key]
    lines = []
    for h in HORIZONS:
        if h not in s.index:
            continue
        r = s.loc[h]
        lines.append(f"ret_{h:>2}: IC {fmt(r['validation_IC'])}  R2 {fmt(r['validation_R2'], 5)}  "
                     f"RMSE {float(r['validation_RMSE']):.3e}  beta {fmt(r.get('beta'), 3)}")
    extra = ""
    if key.startswith("lgbm"):
        ip = path_of(key, "lgbm_importance_ret_10.csv")
        if os.path.isfile(ip):
            imp = pd.read_csv(ip)
            top = imp.nlargest(8, "gain")
            tot = imp.gain.sum()
            extra = "\n\ntop gain (ret_10):\n" + "\n".join(
                f"  {r.feature:<26s} {100 * r.gain / tot:5.1f}%" for r in top.itertuples())
    if key.startswith("mlp"):
        ap = path_of(key, "mlp_arch_ret_10.json" if key == "mlp" else "mlp_arch_ret_1.json")
        if os.path.isfile(ap):
            a = json.load(open(ap))
            extra = f"\n\narch: {a.get('hidden')}  y_scale {a.get('y_scale'):.3e}"
            if a.get("tuned_params"):
                extra += f"\ntuned: {a['tuned_params']}"
    details.append(f"<div class='mcard'><h3 style='color:{color}'>{html.escape(label)}</h3>"
                   f"<pre class='rpt'>{html.escape(chr(10).join(lines) + extra)}</pre></div>")

# ---- model performance: IC by |pred| quantile (same schema as the legacy
# per-horizon tables in <sym>_model.html / augment_hitrates.py, computed here
# straight from the pipeline's valid_pred_ret_<h>.csv so EVERY symbol gets it) --
PROBS = np.array([0.0, .10, .25, .50, .75, .90, .95, .96, .97, .98, .985, .99, .995, 1.0])

# FR = cov(pred, ret) / sd(pred) = IC * sd(ret) = beta_{ret|pred} * sd(pred).
# Expected return per 1-sd of alpha exposure, in RETURN units (rendered in bps).
# NOTE it is IC x sd(RETURN), not IC x sd(alpha) — the alpha sd cancels.
FR_CACHE = {}


def _fr(p, y):
    sd = np.std(p)
    return float(np.cov(p, y, ddof=0)[0, 1] / sd) if sd > 0 else float("nan")


ICSD_CACHE = {}
PN_CACHE = {}   # (key, h) -> (n_pred_positive, n_pred_negative)


def _ic_sd_alpha(p, y):
    """IC x sd(alpha): the IC scaled by the SPREAD OF THE PREDICTIONS themselves.
    Distinct from FR (= IC x sd(return)) — this one grows when a model makes
    bolder predictions, so it reads as 'signal strength x how much the model is
    willing to bet', in prediction units."""
    if np.std(p) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(p, y)[0, 1] * np.std(p))


def perf_table(path, h, key=None):
    has_raw = f"Y_hat_{h}_raw" in pd.read_csv(path, nrows=0).columns
    usecols = ["target", f"Y_hat_{h}"] + ([f"Y_hat_{h}_raw"] if has_raw else [])
    df = pd.read_csv(path, usecols=usecols, engine="c")
    y = df["target"].to_numpy(float); p = df[f"Y_hat_{h}"].to_numpy(float)
    praw = df[f"Y_hat_{h}_raw"].to_numpy(float) if has_raw else None
    keep = np.isfinite(y) & np.isfinite(p); y, p = y[keep], p[keep]
    if praw is not None:
        praw = praw[keep]
    if len(y) < 100:
        return ""
    if key is not None:
        FR_CACHE[(key, h)] = _fr(p, y)
        ICSD_CACHE[(key, h)] = _ic_sd_alpha(p, y)
        PN_CACHE[(key, h)] = ((p > 0).sum(), (p < 0).sum())
    ap = np.abs(p)
    raw = np.quantile(ap, PROBS)
    _, first = np.unique(raw, return_index=True); first = np.sort(first)
    breaks, probs_used = raw[first], PROBS[first]
    b = pd.cut(ap, bins=breaks, include_lowest=True, labels=False)
    b = np.clip(np.where(np.isnan(b), 0, b).astype(int), 0, breaks.size - 2) + 1
    g = pd.DataFrame({"bin": b, "p": p, "y": y})
    if praw is not None:
        g["praw"] = praw
    rows = []
    for bi, sdf in g.groupby("bin", sort=True):
        pv, yv = sdf["p"].to_numpy(), sdf["y"].to_numpy()
        if praw is not None:
            rv = sdf["praw"].to_numpy()
            calib_raw = f"{(rv * yv).mean() / max((rv ** 2).mean(), 1e-300):.3f}"
        else:
            calib_raw = "&mdash;"
        sp, sy = np.sign(pv), np.sign(yv)
        hit, moved = sp == sy, yv != 0
        ic_pe = float(np.corrcoef(pv, yv)[0, 1]) if len(pv) > 1 else float("nan")
        ic_sp = float(pd.Series(pv).corr(pd.Series(yv), method="spearman")) if len(pv) > 1 else float("nan")
        av = np.abs(pv)
        pn = f"{(pv > 0).sum() / max((pv < 0).sum(), 1):.3f}"   # alpha P/N ratio in bin
        rows.append(
            f"<tr><td>{bi}</td><td>{probs_used[bi-1]:g}</td><td>{probs_used[bi]:g}</td>"
            f"<td>{len(sdf):,}</td>"
            f"<td>{av.min():.3e}</td><td>{av.mean():.3e}</td><td>{av.max():.3e}</td>"
            f"<td>{yv.mean():.3e}</td><td>{np.median(yv):.3e}</td>"
            f"<td>{(sp * yv).mean() * 1e4:.3f}</td>"
            f"<td>{(pv * yv).mean() / max(av.mean(), 1e-300) * 1e4:.3f}</td>"
            f"<td>{(pv * yv).mean() / max((pv ** 2).mean(), 1e-300):.3f}</td>"
            f"<td>{calib_raw}</td>"
            f"<td>{_fr(pv, yv) * 1e4:.3f}</td>"
            f"<td>{_ic_sd_alpha(pv, yv) * 1e4:.3f}</td>"
            f"<td>{ic_pe:.4f}</td><td>{ic_sp:.4f}</td>"
            f"<td>{pn}</td>"
            f"<td>{hit.mean():.4f}</td><td>{(hit | ~moved).mean():.4f}</td>"
            f"<td>{(hit[moved].mean() if moved.any() else float('nan')):.4f}</td></tr>")
    # full legacy schema (btc_weighted.html / augment_hitrates.py), dir_return in bps
    return ("<div class='scrollx'><table class='cmp perf'><tr>"
            "<th>bin</th><th>q_lo</th><th>q_hi</th><th>n</th>"
            "<th>abs_pred_min</th><th>abs_pred_mean</th><th>abs_pred_max</th>"
            "<th>real_mean</th><th>real_median</th><th>dir_return (bps)</th>"
            "<th>prop_return (bps)</th><th>calib</th><th>calib_raw</th><th>FR (bps)</th><th>IC&middot;&sigma;(alpha) (bps)</th>"
            "<th>ic_pearson</th><th>ic_spearman</th><th>P/N ratio</th>"
            "<th>hit_rate</th><th>hit_rate_with_zero</th><th>hit_rate_move</th></tr>"
            + "".join(rows) + "</table></div>")


perf_blocks = []
for h in HORIZONS:
    cards = []
    for key, label, color, _d, _g in MODELS:
        if key not in loaded:
            continue
        p = path_of(key, f"valid_pred_ret_{h}.csv")
        if not os.path.isfile(p):
            continue
        t = perf_table(p, h, key)
        if t:
            _pl, _ps = PN_CACHE.get((key, h), (0, 0))
            _pn = f"{_pl / max(_ps, 1):.3f}"
            cards.append(f"<div class='perfcard'><h3 style='color:{color}'>"
                         f"{html.escape(label)} &mdash; VALIDATION "
                         f"<span class='note'>(alpha P/N ratio {_pn}: {_pl:,} positive "
                         f"vs {_ps:,} negative predictions)</span></h3>{t}</div>")
    if cards:
        perf_blocks.append(f"<details class='perf'{' open' if h == 10 else ''}>"
                           f"<summary>ret_{h} &mdash; IC by |pred| quantile "
                           f"(dir return, Pearson/Spearman IC, hit rates)</summary>"
                           f"<div class='perfrow'>{''.join(cards)}</div></details>")
perf_html = "".join(perf_blocks)

# ---- MERGED FROM THE LEGACY gen_model_report.py (so one generator produces
# every model page): full transposed per-horizon metric table, top features per
# horizon, and the predicted-vs-realized scatter — all now sourced from PIPELINE
# artifacts (horizon_summary / ols_coefficients / lgbm_importance / valid_pred)
# instead of the retired R-trainer outputs. -------------------------------------
FULL_METRIC_ROWS = [
    ("in_sample_R2", "IS R²"), ("validation_R2", "Val R²"),
    ("validation_R2_zero", "Val R² (vs 0)"), ("validation_IC", "Val IC"),
    ("validation_RMSE", "Val RMSE"), ("validation_IR", "Val IR"), ("beta", "β realized|pred"),
    ("realized_vol", "realized vol"), ("predicted_vol", "predicted vol"),
    ("mean_realized", "mean realized"), ("mean_predicted", "mean predicted"),
    ("skew_realized", "skew realized"), ("skew_predicted", "skew predicted"),
    ("kurt_realized", "kurt realized"), ("kurt_predicted", "kurt predicted"),
    ("real_q5", "real q05"), ("real_q50", "real q50"), ("real_q95", "real q95"),
    ("pred_q5", "pred q05"), ("pred_q50", "pred q50"), ("pred_q95", "pred q95"),
]


def per_model_summary(key):
    """Section 1 of the legacy per-model report: metrics as ROWS, horizons as
    COLUMNS, for ONE model (btc_weighted.html '1. Per-horizon summary')."""
    sdf = loaded[key]
    hz = [h for h in HORIZONS if h in sdf.index]
    head = "".join(f"<th>ret_{h}</th>" for h in hz)
    body = []
    for field in sdf.columns:
        if field == "prediction_horizon":
            continue
        cells = "".join(f"<td>{fmt(sdf.loc[h].get(field), 6)}</td>" for h in hz)
        body.append(f"<tr><td class='hz'>{html.escape(field)}</td>{cells}</tr>")
    fr_cells = "".join(f"<td>{fmt(FR_CACHE.get((key, h), float('nan')) * 1e4, 4)}</td>"
                       for h in hz)
    body.append(f"<tr><td class='hz'><b>FR (bps) = cov(pred,ret)/&sigma;(pred) = IC&middot;&sigma;(ret)</b></td>{fr_cells}</tr>")
    ics_cells = "".join(f"<td>{fmt(ICSD_CACHE.get((key, h), float('nan')) * 1e4, 4)}</td>"
                        for h in hz)
    body.append(f"<tr><td class='hz'><b>IC&middot;&sigma;(alpha) (bps)</b></td>{ics_cells}</tr>")
    return (f"<div class='scrollx'><table class='cmp'><tr><th>metric</th>{head}</tr>"
            + "".join(body) + "</table></div>")


def full_metric_table():
    """Transposed metrics-as-rows / (model × horizon)-as-columns — the legacy
    page's 'Full per-horizon summary', now from the pipeline summaries."""
    cols = [(k, l, h) for k, l, _c, _d, _g in
            [(m[0], m[1], m[2], m[3], m[4]) for m in MODELS] if k in loaded
            for h in HORIZONS if h in loaded[k].index]
    if not cols:
        return ""
    head = "".join(f"<th>{html.escape(l.split(' (')[0])}<br>ret_{h}</th>" for _k, l, h in cols)
    body = []
    for field, label in FULL_METRIC_ROWS:
        cells = []
        for k, _l, h in cols:
            v = loaded[k].loc[h].get(field)
            cells.append(f"<td>{fmt(v, 5)}</td>")
        body.append(f"<tr><td class='hz'>{label}</td>{''.join(cells)}</tr>")
    fr = "".join(f"<td>{fmt(FR_CACHE.get((k, h), float('nan')) * 1e4, 4)}</td>"
                 for k, _l, h in cols)
    body.append(f"<tr><td class='hz'><b>FR (bps)</b></td>{fr}</tr>")
    ics = "".join(f"<td>{fmt(ICSD_CACHE.get((k, h), float('nan')) * 1e4, 4)}</td>"
                  for k, _l, h in cols)
    body.append(f"<tr><td class='hz'><b>IC&middot;&sigma;(alpha) (bps)</b></td>{ics}</tr>")
    return (f"<div class='scrollx'><table class='cmp'><tr><th>metric</th>{head}</tr>"
            + "".join(body) + "</table></div>")


def top_features(key, h, n=12):
    """Top features for this model/horizon: |t| for the linear refit, gain for
    trees (legacy 'top features' block, from pipeline artifacts)."""
    p = path_of(key, f"ols_coefficients_ret_{h}.csv")
    if os.path.isfile(p):
        d = pd.read_csv(p)
        d = d[d.term != "const"].assign(t=lambda x: x.t_value.abs()).nlargest(n, "t")
        rows = "".join(f"<tr><td>{html.escape(str(r.term))}</td><td>{r.estimate:.3e}</td>"
                       f"<td>{r.t_value:+.1f}</td></tr>" for r in d.itertuples())
        return ("<table class='cmp perf'><tr><th>feature</th><th>β</th><th>t</th></tr>"
                + rows + "</table>")
    p = path_of(key, f"lgbm_importance_ret_{h}.csv")
    if os.path.isfile(p):
        d = pd.read_csv(p); tot = d.gain.sum()
        rows = "".join(f"<tr><td>{html.escape(str(r.feature))}</td>"
                       f"<td>{100 * r.gain / tot:.1f}%</td><td>{int(r.split)}</td></tr>"
                       for r in d.nlargest(n, "gain").itertuples())
        return ("<table class='cmp perf'><tr><th>feature</th><th>gain</th><th>splits</th></tr>"
                + rows + "</table>")
    return ""


def scatter_fig(key, label, color, h, sample=9000):
    """Predicted vs realized (legacy scatter PNG -> interactive plotly):
    sampled points + mean-realized per prediction ventile (the calibration line;
    a 45° trend means predictions are correctly scaled)."""
    p = path_of(key, f"valid_pred_ret_{h}.csv")
    if not os.path.isfile(p):
        return ""
    d = pd.read_csv(p, usecols=["target", f"Y_hat_{h}"], engine="c").dropna()
    if len(d) < 100:
        return ""
    yh, y = d[f"Y_hat_{h}"].to_numpy(float), d["target"].to_numpy(float)
    idx = np.random.default_rng(7).choice(len(y), size=min(sample, len(y)), replace=False)
    qs = np.quantile(yh, np.linspace(0, 1, 21))
    qs = np.unique(qs)
    b = np.clip(np.digitize(yh, qs[1:-1]), 0, len(qs) - 2)
    bm = pd.DataFrame({"b": b, "yh": yh, "y": y}).groupby("b").agg(
        x=("yh", "mean"), m=("y", "mean"), n=("y", "size")).reset_index()
    # legacy title stats (pred_diagnostics.R): full-sample n, IC, OLS beta/alpha
    n_all = len(y)
    ic = float(np.corrcoef(yh, y)[0, 1])
    beta, alpha = np.polyfit(yh, y, 1)
    fig = go.Figure()
    fig.add_scatter(x=yh[idx] * 1e4, y=y[idx] * 1e4, mode="markers",
                    marker=dict(size=2.5, color=color, opacity=0.30), name="rows",
                    hovertemplate="pred=%{x:.3f} bps<br>real=%{y:.3f} bps<extra></extra>")
    fig.add_scatter(x=bm.x * 1e4, y=bm.m * 1e4, mode="lines+markers",
                    line=dict(color="#e53e3e", width=2), marker=dict(size=6),
                    name="mean realized per ventile",
                    customdata=bm.n,
                    hovertemplate=("pred=%{x:.3f} bps<br>mean real=%{y:.3f} bps"
                                   "<br>n=%{customdata}<extra></extra>"))
    lim = float(np.nanquantile(np.abs(yh), 0.999) * 1e4)
    xs = np.array([-lim, lim])
    fig.add_scatter(x=xs, y=(beta * xs / 1e4 + alpha) * 1e4, mode="lines",
                    name="OLS fit", line=dict(color="#e53e3e", width=1.6),
                    hovertemplate="OLS fit<extra></extra>")
    fig.add_scatter(x=xs, y=xs, mode="lines", name="y = x",
                    line=dict(color="#68d391", width=1, dash="dot"),
                    hovertemplate="y = x<extra></extra>")
    # EQUAL SCALES: realized has far fatter tails than the prediction, so an
    # auto y-range (±10 bps vs ±1.4 bps of pred) squashed y=x into a flat line.
    # Matching the y-range to the x-range makes y=x a true 45° reference and the
    # OLS/ventile slopes readable; the clipped tail share is stated in the title.
    clipped = float(np.mean(np.abs(y) * 1e4 > lim))
    fig.update_layout(title=dict(text=(f"Predicted vs realized return — ret_{h} ({label})<br>"
                                       f"<sub>n={n_all:,}  IC={ic:.4f}  beta={beta:.4f}  "
                                       f"alpha={alpha:.3e}  &nbsp;|&nbsp; equal axes; "
                                       f"{clipped * 100:.1f}% of rows outside the y-range"
                                       f"</sub>"), font=dict(size=13)),
                      xaxis_title="prediction (bps)", yaxis_title="realized (bps)",
                      xaxis_range=[-lim, lim], yaxis_range=[-lim, lim],
                      template="plotly_dark", autosize=True,
                      paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                      margin=dict(t=62, l=60, r=20, b=44), height=380, showlegend=True,
                      legend=dict(x=0.01, y=0.99, bgcolor="rgba(15,23,42,0.6)",
                                  font=dict(size=10)))
    # responsive: the figure lives in a grid cell, so it must re-layout on resize
    # instead of baking the (narrow) width it happened to see at load time.
    return to_html(fig, include_plotlyjs=False, full_html=False,
                   config={"responsive": True}, default_width="100%")


def ic_by_hour_fig():
    """IC (and n) by UTC hour — session structure. Metals trade in sessions:
    active London/NY-overlap hours carry several times the volatility and
    non-zero-return share of the dead hours, so a whole-sample IC hides where
    the model actually works. Computed on the validation set, per model, at the
    horizons in HORIZONS."""
    traces, buttons = [], []
    per_h = {}
    for h in HORIZONS:
        rows = []
        for key, label, color, _d, _g in MODELS:
            if key not in loaded:
                continue
            p = path_of(key, f"valid_pred_ret_{h}.csv")
            if not os.path.isfile(p):
                continue
            d = pd.read_csv(p, usecols=["mark_ts", "target", f"Y_hat_{h}"], engine="c").dropna()
            d["hr"] = (pd.to_datetime(d.mark_ts, unit="ns").dt.hour)
            g = d.groupby("hr").apply(
                lambda x: pd.Series({
                    "ic": (np.corrcoef(x[f"Y_hat_{h}"], x.target)[0, 1]
                           if len(x) > 2 and x.target.std() > 0 else np.nan),
                    # dir_return: mean signed return of trading the predicted
                    # direction, in bps — the P&L-flavoured twin of IC (second axis)
                    "dr": float((np.sign(x[f"Y_hat_{h}"]) * x.target).mean() * 1e4),
                    "n": len(x),
                    "nz": float((x.target != 0).mean()),
                    "vol": float(x.target.std() * 1e4)}), include_groups=False).reset_index()
            rows.append((label, color, g))
        if rows:
            per_h[h] = rows
    if not per_h:
        return ""
    hs = sorted(per_h)
    ranges = {}
    for hi, h in enumerate(hs):
        ranges[h] = []
        for label, color, g in per_h[h]:
            ranges[h].append(len(traces))
            traces.append(go.Scatter(
                x=g.hr, y=g.ic, mode="lines+markers", name=f"{label} — IC",
                line=dict(color=color, width=1.8), marker=dict(size=5),
                visible=(hi == 0), yaxis="y",
                customdata=np.stack([g.n, g.nz * 100, g.vol, g.dr], axis=-1),
                hovertemplate=("%{x}:00 UTC<br>IC=%{y:.4f}"
                               "<br>dir_return=%{customdata[3]:.3f} bps"
                               "<br>n=%{customdata[0]:,}"
                               "<br>non-zero ret=%{customdata[1]:.1f}%"
                               "<br>ret vol=%{customdata[2]:.2f} bps<extra></extra>")))
            ranges[h].append(len(traces))
            traces.append(go.Scatter(
                x=g.hr, y=g.dr, mode="lines+markers", name=f"{label} — dir_return",
                line=dict(color=color, width=1.4, dash="dot"),
                marker=dict(size=4, symbol="diamond"),
                visible=(hi == 0), yaxis="y2",
                customdata=np.stack([g.n, g.ic], axis=-1),
                hovertemplate=("%{x}:00 UTC<br>dir_return=%{y:.3f} bps"
                               "<br>IC=%{customdata[1]:.4f}"
                               "<br>n=%{customdata[0]:,}<extra></extra>")))
    # ---- session-hours bands: legend-toggleable shaded rectangles on a hidden
    # 0..1 axis (y3) so they span the full plot height at any IC scale. They
    # start hidden ('legendonly') — click the legend entry to show a session.
    # Hours are UTC for the CURRENT (summer/DST) offsets; local hours in the label.
    SESSIONS = [("Asia — Tokyo 09–15 JST (00–06 UTC)", 0.0, 6.0, "234,179,8"),
                ("Shanghai day — SGE/SHFE 09–15 CST (01–07 UTC)", 1.0, 7.0, "239,68,68"),
                # SHFE au/ag night session 21:00–02:30 CST = 13:00–18:30 UTC
                # (crosses CST midnight but is contiguous in UTC)
                ("Shanghai night — SHFE 21–02:30 CST (13–18:30 UTC)", 13.0, 18.5, "249,115,22"),
                ("London — 08–16:30 UK (07–15:30 UTC)", 7.0, 15.5, "56,189,248"),
                ("New York — 08–17 ET (12–21 UTC)", 12.0, 21.0, "167,139,250")]
    session_idx = []
    for sname, s0, s1, rgb in SESSIONS:
        session_idx.append(len(traces))
        traces.append(go.Scatter(
            x=[s0, s1, s1, s0, s0], y=[0, 0, 1, 1, 0], mode="none",
            fill="toself", fillcolor=f"rgba({rgb},0.16)", name=sname,
            yaxis="y3", visible="legendonly", hoverinfo="skip",
            showlegend=True))
    # NOTE trace counts differ per horizon (e.g. a tuned model trained on ret_1
    # only), so track each horizon's own index range instead of assuming a
    # constant group size — a fixed stride overruns the trace list.
    for hi, h in enumerate(hs):
        vis = [False] * len(traces)
        for j in ranges[h]:
            vis[j] = True
        for j in session_idx:            # session bands survive horizon switches
            vis[j] = "legendonly"
        buttons.append(dict(label=f"ret_{h}", method="update",
                            args=[{"visible": vis},
                                  {"title": f"IC &amp; dir_return by UTC hour — ret_{h} (validation)"}]))
    fig = go.Figure(data=traces)
    fig.add_hline(y=0, line_color="#94a3b8", line_width=1)
    fig.update_layout(
        title=f"IC &amp; dir_return by UTC hour — ret_{hs[0]} (validation)",
        xaxis_title="hour of day (UTC)",
        yaxis=dict(title="IC (solid)"),
        yaxis2=dict(title="dir_return, bps (dotted)", overlaying="y", side="right",
                    showgrid=False, zeroline=True, zerolinecolor="#475569"),
        yaxis3=dict(overlaying="y", range=[0, 1], visible=False, fixedrange=True),
        xaxis=dict(dtick=1), template="plotly_dark", autosize=True,
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", height=560,
        # title on its own line, horizon buttons on a SECOND line beneath it,
        # legend below the x-axis — nothing overlaps the plotting area or the
        # modebar (top-right), which is why the top margin is this generous.
        margin=dict(t=118, l=60, r=70, b=120),
        title_x=0.01, title_xanchor="left", title_y=0.985, title_yanchor="top",
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        updatemenus=[dict(type="buttons", direction="right",
                          x=0.01, xanchor="left", y=1.10, yanchor="top",
                          pad=dict(t=2, b=2, l=2, r=2),
                          buttons=buttons, font=dict(size=10),
                          bgcolor="#1e293b", bordercolor="#334155")])
    return to_html(fig, include_plotlyjs=False, full_html=False,
                   config={"responsive": True}, default_width="100%",
                   default_height="560px")


detail_blocks = []
for h in HORIZONS:
    cards = []
    for key, label, color, _d, _g in MODELS:
        if key not in loaded or h not in loaded[key].index:
            continue
        tf, sc = top_features(key, h), scatter_fig(key, label, color, h)
        if tf or sc:
            cards.append(f"<div class='perfcard'><h3 style='color:{color}'>"
                         f"{html.escape(label)}</h3>{tf}{sc}</div>")
    if cards:
        detail_blocks.append(f"<details class='perf'{' open' if h == 10 else ''}>"
                             f"<summary>ret_{h} &mdash; top features &amp; predicted-vs-realized"
                             f"</summary><div class='perfrow'>{''.join(cards)}</div></details>")
detail_html = "".join(detail_blocks)

SETUPS = {
    "btc": "dump btc_v6 (12.10M rows, 2026-01-01 &rarr; 05-20, 100% coverage)",
    "xau": "dump xau_v1 (2026-06-01 &rarr; 07-20, 100% coverage, live-book re-record)",
    "xag": "dump xag_v2 (2026-06-01 &rarr; 07-19, 100% coverage, live-book re-record)",
}
if VER:   # v2/v3 read the EXTENDED-window dumps (/mnt/nvme2) — override the v1 blurb
    SETUPS.update({
        "xau": "dump xau_v1 (2026-04-27 &rarr; 08-13, 100% coverage, /mnt/nvme2)"
               + (" &middot; CALIBRATED predictions" if VER == "v3" else ""),
        "xag": "dump xag_v2 (2026-04-27 &rarr; 08-13, 100% coverage, /mnt/nvme2)"
               + (" &middot; CALIBRATED predictions" if VER == "v3" else ""),
    })
def _fmt_ts(ns):
    return pd.to_datetime(int(ns)).strftime("%Y-%m-%d %H:%M")


def period_line():
    """training/validation date ranges from the first available run's dumps:
    train start = first row of the all-rows Y_hat dump; validation range =
    first/last mark_ts of valid_pred (the 80/20 split boundary)."""
    for key, _l, _c, mdir, gen in MODELS:
        vp = f"{STATS}/{SYM}/{mdir}/20260602/{gen}/valid_pred_ret_10.csv"
        yh = f"{STATS}/{SYM}/{mdir}/20260602/{gen}/{mdir}_Y_hat.csv"
        if not os.path.isfile(vp):
            continue
        try:
            m = pd.read_csv(vp, usecols=["mark_ts"])["mark_ts"]
            v0, v1 = _fmt_ts(m.min()), _fmt_ts(m.max())
            t0 = (_fmt_ts(pd.read_csv(yh, usecols=["mark_ts"], nrows=1)
                          ["mark_ts"].iloc[0]) if os.path.isfile(yh) else "?")
            return (f"<b>training</b> {t0} &rarr; {v0} (first 80%) &middot; "
                    f"<b>validation</b> {v0} &rarr; {v1} (last 20%, all "
                    f"validation metrics on this page)")
        except Exception:
            continue
    return ""


setup = (f"{SETUPS.get(SYM, 'unified-pipeline run')} &middot; report-driven feature "
         "auto-selection &middot; 80/20 time split &middot; cost &radic;|Y| sample weights "
         "&middot; train-only target clip q0.001/0.999 &middot; linear+MLP further-normalized "
         "per x&sigma;/kurt audit, LGBM raw &middot; identical splits across models")

body = f"""<header>
  <a class="back-link" href="index.html">&larr; All reports</a>
  <h1>{SYM.upper()}{f" {VER}" if VER else ""} &mdash; model report (unified pipeline)</h1>
  <div class="meta">{setup}</div>
  <div class="meta">{period_line()}</div>
</header>
<section>
  <h2>Validation metrics by horizon <span class="note">(green = best per cell group)</span></h2>
  {cmp_table}
  <h2>IC by conviction decile (ret_10, |prediction| deciles)</h2>
  <details class="defs-c"><summary>&#9432; How the conviction-decile buckets are built</summary><div class="note defs">
   Rows are split into ten equal-size buckets by <b>|prediction|</b> (d0 = least
   confident 10% of rows, d9 = most confident), and IC is computed
   <b>within</b> each bucket &mdash; so this measures how well the model ranks
   returns <i>among rows of similar conviction</i>, not across them.<br><br>
   <b>What to look for:</b> a <b>monotone rise d0&rarr;d9</b> means the model
   knows when it knows &mdash; bigger predictions really are more reliable, which
   is what makes a |pred| trading threshold work (see the alpha-decay page).
   A flat ladder means conviction carries no information and thresholding buys
   nothing. <b>A dip at d9</b> (common) means the very largest predictions are
   partly noise/outliers: their <i>direction</i> can still be right (check
   <code>dir_return</code> in the table above) while their <i>ordering</i> within
   the bucket degrades &mdash; typically extreme-leaf variance in trees, or
   heavy-tailed inputs in a linear fit.<br><br>
   IC here is far higher than the headline whole-sample IC because within a
   narrow conviction band the prediction spread is small, so correlation is
   dominated by genuine signal rather than by scale.
  </div></details>
  <div class="ladders">{''.join(ladders)}</div>
  <details class="defs-c"><summary>&#9432; How to read the ladders — calibration, top-decile dip, position sizing</summary><div class="note defs">
   <b>How to read this:</b> validation rows are bucketed by |prediction| into deciles &mdash;
   d0 = weakest 10% of signals (lowest conviction), d9 = strongest 10%; each bar is
   IC = corr(prediction, realized ret_10) computed <i>within that decile only</i>.
   It answers: when the model speaks louder, is it more right?<br><br>
   &bull; <b>Rising ladder = calibrated conviction.</b> IC increasing with |pred| means signal
   size is real information &mdash; it justifies conviction-proportional position sizing
   (vs flat size per signal). A flat ladder would mean |pred| carries no extra edge.<br>
   &bull; <b>Top-decile dip = tail overconfidence.</b> The largest predictions come from
   extreme/saturated feature values. A LINEAR model extrapolates (coef &times; outlier input
   &rarr; outsized forecast), so its d9 typically drops hardest; LGBM leaf values cannot
   exceed the training targets, so its top decile usually holds up better. The d9 bucket
   also concentrates high-vol/event regimes where realized noise is largest.<br>
   &bull; <b>Actionable:</b> size with conviction up to ~d8; CAP or shrink beyond (prediction
   soft-cap, e.g. tanh at the d8/d9 boundary, or isotonic recalibration of the top bucket)
   &mdash; otherwise the biggest positions are taken exactly where accuracy degrades.<br>
   &bull; Within-decile ICs are computed on thin slices &mdash; the overall IC is NOT their
   average, and d0's near-zero IC is expected (tiny predictions are mostly noise-range).
  </div></details>
  <h2>Model performance &mdash; IC by |pred| quantile</h2>
  <details class="defs-c"><summary>&#9432; Metric definitions — hit-rate variants, dir_return, FR, IC&middot;&sigma;(alpha), P/N ratio</summary><div class="note defs">
   <b>Hit-rate columns</b> (differ only in how a realized return of exactly 0 is scored):<br>
   &bull; <code>hit_rate</code> = sign(pred)==sign(real); a 0 move counts as a MISS (pessimistic).<br>
   &bull; <code>hit_rate_with_zero</code> = same, but a 0 move counts as a HIT (no adverse move);
     inflated at short horizons where many returns are exactly 0, so it mostly tracks inactivity.<br>
   &bull; <code>hit_rate_move</code> = sign accuracy over rows that MOVED (real&ne;0) only &mdash;
     the cleanest directional-skill measure.<br>
   The gap <code>hit_rate_with_zero &minus; hit_rate</code> = share of no-move rows.<br><br>
   <b><code>dir_return</code></b> = mean(sign(p) &middot; y) &nbsp;(p = prediction, y = realized return):
   the average realized return of trading the predicted direction (long if pred&gt;0, short if
   pred&lt;0, unit size) in that bin &mdash; a signed-P&amp;L edge proxy. Sign = directional skill;
   it is dollar-weighted (big moves dominate, unlike hit_rate), and 0-moves contribute exactly 0.
   <i>(shown in bps)</i><br><br>
   <b><code>prop_return</code></b> = mean(p &middot; y) / mean(|p|) &nbsp;(bps): realized PnL per unit
   of GROSS exposure when position size is PROPORTIONAL to the prediction &mdash; the sizing-aware
   twin of <code>dir_return</code> (flat size). <code>prop_return &gt; dir_return</code> in a bin
   means conviction-proportional sizing beats flat sizing within that bin. (Raw
   mean(p&middot;y) alone would rise mechanically with |p| even at constant edge, so it is
   normalized by gross.)<br><br>
   <b><code>calib</code></b> = mean(p &middot; y) / mean(p&sup2;): the through-origin regression of
   realized on predicted &mdash; <b>realized return per unit of predicted return</b>. It answers
   &ldquo;when the model predicts 2&times;, do we earn 2&times;?&rdquo;: &asymp;1 and flat across
   bins = prediction sizes are in true return units and trustworthy; &lt;1 = predictions overshoot
   (trade calib&times;size); decaying toward the top bins = tail overconfidence, quantified as the
   exact shrink factor for the biggest predictions. On runs trained WITH the
   calibration layer (calibration_ret_&lt;h&gt;.json), <code>Y_hat</code> is already
   calibrated: <code>calib</code> is the ACCEPTANCE TEST (target &asymp;1.0, flat across
   bins) and <code>calib_raw</code> shows the pre-calibration value; on older runs
   calib_raw is &mdash;.<br><br>
   <b><code>FR</code></b> = cov(pred, ret) / &sigma;(pred) &nbsp;=&nbsp; <b>IC &middot; &sigma;(ret)</b>
   &nbsp;=&nbsp; &beta;<sub>ret|pred</sub> &middot; &sigma;(pred). The expected return from a
   <b>1-standard-deviation alpha exposure</b>, i.e. IC re-expressed in RETURN units instead of
   correlation units &mdash; so unlike IC it is comparable across horizons and instruments with
   different return volatility. (Note the &sigma; that survives is the <i>return</i>&rsquo;s, not the
   alpha&rsquo;s: the alpha&rsquo;s cancels.) <code>dir_return</code> is the sign-only cousin
   (unit size, direction only); FR is the size-aware version &mdash; it rewards being big when
   right, so FR &gt; dir_return means conviction is well calibrated.<br><br>
   <b><code>IC&middot;&sigma;(alpha)</code></b> &mdash; the same IC scaled by the spread of the
   PREDICTIONS instead of the returns. It rises when a model both ranks well and makes bolder
   predictions, so it is the natural companion to FR: <code>FR</code> answers &ldquo;how much
   return does a unit-&sigma; exposure earn&rdquo;, <code>IC&middot;&sigma;(alpha)</code> answers
   &ldquo;how much signal is this model actually emitting&rdquo;. The two differ by
   &sigma;(alpha)/&sigma;(ret), so a model with tiny predictions can post a healthy FR yet a small
   IC&middot;&sigma;(alpha).<br><br>
   <b><code>P/N ratio</code></b> = count(pred &gt; 0) / count(pred &lt; 0) &mdash; the
   sign balance of the alpha. ~1.0 means the model is directionally neutral; a
   persistent skew means it leans one way, which shows up as inventory drift in any
   unit-per-signal trading rule (see the alpha-decay page) and is worth checking
   against the realized return skew before trusting a headline PnL.
  </div></details>
  {perf_html}
  <h2>IC &amp; dir_return by hour of day <span class="note">(validation; solid = IC
      (left axis), dotted = dir_return in bps (right axis); hover for n, non-zero
      share and realized vol &mdash; the buttons switch horizon)</span></h2>
  <details class="defs-c"><summary>&#9432; How to read IC by hour — metals session structure</summary><div class="note defs">
   Metals trade in <b>sessions</b>: the London/NY overlap (~13&ndash;15 UTC) and the
   Asia open carry several times the volatility and non-zero-return share of the dead
   hours (04, 10, 20&ndash;23 UTC), so a single whole-sample IC averages a working
   model over hours where there is nothing to predict. Read this chart for
   <b>where the edge lives</b>: hours with high IC <i>and</i> a high non-zero share
   are the tradeable window; hours with near-zero IC and a low non-zero share are
   mostly quote noise and are candidates for session filtering (train and trade on
   active hours only). A model that is strong only in dead hours is a red flag &mdash;
   usually stale-quote artefacts rather than signal.<br><br>
   The dotted <code>dir_return</code> line (right axis) is the P&amp;L-flavoured twin:
   IC says how well the model <i>ranks</i> within the hour, dir_return says what a
   unit-size directional trade actually <i>earned</i> there (bps). They can diverge —
   an hour with decent IC but flat dir_return has ranking skill on moves too small to
   monetise (typically dead hours), while an hour where dir_return spikes above its
   IC is where a few large moves went the right way. Trade the hours where BOTH are
   high.
  </div></details>
  {ic_by_hour_fig()}
  <h2>Per-horizon summary by model <span class="note">(every metric the trainer
      emits, one table per model &mdash; legacy trainer-report layout)</span></h2>
  {"".join(f"<details class='perf'><summary>{html.escape(l)}</summary>{per_model_summary(k)}</details>" for k, l, _c, _d, _g in MODELS if k in loaded)}
  <h2>Per-horizon detail &mdash; top features &amp; predicted-vs-realized</h2>
  {detail_html}
  <h2>Full per-horizon metrics <span class="note">(every metric; models &times; horizons as columns)</span></h2>
  {full_metric_table()}
  <h2>Per-model detail</h2>
  <div class="mcards">{''.join(details)}</div>
</section>"""

CSS = """body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;
background:#0f172a;color:#e2e8f0}
header{background:linear-gradient(90deg,#1a202c,#2d3748);padding:18px 28px}
header h1{margin:6px 0 0;font-size:19px}
header .meta{color:#94a3b8;font-size:12px;margin-top:6px;max-width:920px;line-height:1.5}
header .back-link{color:#90cdf4;font-size:13px;text-decoration:none}
section{padding:18px 28px;max-width:1100px}
h2{font-size:15px;color:#cbd5e1;margin:26px 0 10px}
h2 .note{font-size:11px;color:#64748b;font-weight:400}
details.defs-c{margin:6px 0 10px}
details.defs-c>summary{cursor:pointer;color:#975a16;font-size:12.5px;font-weight:600;
list-style:none;padding:4px 0}
details.defs-c>summary:hover{color:#744210}
table.cmp{border-collapse:collapse;font-size:13px}
table.cmp th,table.cmp td{border:1px solid #334155;padding:6px 12px;text-align:right}
table.cmp th{background:#1e293b;color:#94a3b8;font-weight:600}
table.cmp td.hz{color:#94a3b8}
table.cmp td.best{color:#4ade80;font-weight:700}
.ladders{display:flex;gap:28px;flex-wrap:wrap}
.ladder{min-width:280px}
.ladder h3{font-size:13px;margin:0 0 8px}
.brow{display:flex;align-items:center;gap:6px;margin:3px 0}
.blab{width:24px;color:#64748b;font-size:11px}
.bar{height:10px;border-radius:2px}
.bval{font-size:11px;color:#94a3b8}
.mcards{display:flex;gap:20px;flex-wrap:wrap}
.mcard{background:#1e293b;border-radius:8px;padding:12px 16px;min-width:300px}
.mcard h3{font-size:13px;margin:0 0 8px}
.scrollx{overflow-x:auto;max-width:100%}
.note.defs{background:#111c33;border:1px solid #1e293b;border-radius:8px;padding:12px 16px;margin:8px 0 14px;font-size:12px;line-height:1.65;color:#cbd5e1;max-width:1000px}
.note.defs code{color:#e2c08d}
details.perf{margin:8px 0}
details.perf summary{cursor:pointer;color:#e2c08d;font-size:13px}
.perfrow{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:22px;margin-top:10px}
.perfcard{min-width:0;background:#111c33;border:1px solid #1e293b;border-radius:8px;padding:12px 14px}
.perfcard table{width:100%}
.perfcard h3{font-size:12.5px;margin:0 0 6px}
table.perf{font-size:12px}
table.perf td{color:#e2e8f0}
pre.rpt{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;
line-height:1.55;color:#e2e8f0;margin:0;white-space:pre-wrap}
"""

plotly_js = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
page = ("<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{SYM.upper()}{f' {VER}' if VER else ''} model report</title>{plotly_js}<style>{CSS}</style></head>"
        f"<body>{body}</body></html>")
open(OUT, "w").write(page)
print(f"wrote {OUT} | models present: {', '.join(loaded) or 'none'}")
