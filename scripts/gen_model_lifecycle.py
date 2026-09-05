#!/usr/bin/env python3
"""Model LIFECYCLE page — the bridge between research and production.

Training pages answer "is the model good?"; production pages answer "what did it
earn?". Neither answers "is the live model still the model we trained, and where
does the researched edge go?". This page puts both on the same axes:

  1. EDGE WATERFALL      gross virtual edge -> fees -> slippage/impact -> realized
                         (research bps/trip vs production bps/trip, decomposed)
  2. MARKOUT OVERLAY     virtual-trade markout (validation) vs production
                         qty-weighted fill markout, same horizons, same formula
  3. HOLDING OVERLAY     virtual round-trip holding distribution vs the live
                         FIFO-matched one
  4. FEATURE DRIFT       per-feature CAP-HIT RATE and PSI of the LIVE feature
                         distribution vs the TRAINING window. feature_caps_ret_h
                         .csv already defines each feature's training domain, so
                         a feature drifting outside it is the earliest signal the
                         model is being asked to extrapolate.
  5. ALPHA DRIFT         live alpha distribution vs validation predictions
                         (sigma, P/N ratio, |alpha| quantiles) + live IC by hour.

Sources: pipeline artifacts (statistics/<sym>/<model>/.../pipeline_sqrtw) and the
production analysis_out (resampled.parquet, trade_markout.csv, holding_pnl.csv,
order_summary.csv).

    python gen_model_lifecycle.py [btc] [prod_dir_name]
Then make_index.py to encrypt + publish.
"""
import html
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import to_html

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYM = (sys.argv[1] if len(sys.argv) > 1 else "btc").lower()
PROD_NAME = sys.argv[2] if len(sys.argv) > 2 else "btc_monetization2"
PROD = f"/home/guanyang/work/CR_TRAINING/PY/prod/{PROD_NAME}/analysis_out"
STATS = "/home/guanyang/work/alpha_replayer_config/statistics"
GEN = "pipeline_sqrtw"
MODEL, H_REF = "lgbm", 10          # reference model/horizon for the research side
Q_REF = 0.99                        # conviction threshold used for the research trip
NS = 1_000_000_000
OUT = os.path.join(HERE, f"{SYM}_model_lifecycle.html")

DARK = dict(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", template="plotly_dark")


def _fig(fig, height="430px"):
    return to_html(fig, include_plotlyjs=False, full_html=False,
                   config={"responsive": True}, default_width="100%",
                   default_height=height)


def _path(model, name):
    return f"{STATS}/{SYM}/{model}/20260602/{GEN}/{name}"


# ─────────────────── research side: virtual trips at Q_REF ────────────────────
def research_trips():
    p = _path(MODEL, f"valid_pred_ret_{H_REF}.csv")
    if not os.path.isfile(p):
        return None
    d = pd.read_csv(p, usecols=["mark_ts", "AlphaPx", f"Y_hat_{H_REF}"]).dropna()
    d = d.sort_values("mark_ts")
    ts = d.mark_ts.to_numpy(np.int64)
    px = d.AlphaPx.to_numpy(float)
    pred = d[f"Y_hat_{H_REF}"].to_numpy(float)
    T = float(np.quantile(np.abs(pred), Q_REF))
    # markout curve (prod formula: 1s grid + LOCF, simple returns)
    sec = ts // NS
    g0, g1 = int(sec[0]), int(sec[-1])
    grid = np.full(g1 - g0 + 1, np.nan)
    grid[sec - g0] = px
    grid = pd.Series(grid).ffill().to_numpy()
    side = np.where(pred > T, 1.0, np.where(pred < -T, -1.0, 0.0))
    e = np.flatnonzero(side != 0)
    hs = np.arange(1, 301)
    mo = []
    for h in hs:
        t = sec[e] - g0 + h
        ok = t <= g1 - g0
        mo.append(np.nanmean(side[e][ok] * (grid[t[ok]] / px[e][ok] - 1.0)) * 1e4)
    # unit trips, FIFO, cap 20 (same rule as the alpha-decay page)
    logpx = np.log(px)
    seg = np.cumsum(np.r_[False, np.diff(ts) != NS])
    dsig = np.where(pred > T, 1, np.where(pred < -T, -1, 0)).astype(np.int8)
    hold, pnl, q, inv, cs = [], [], [], 0, -1
    for i in np.flatnonzero(dsig != 0):
        if seg[i] != cs:
            q, inv, cs = [], 0, seg[i]
        s = int(dsig[i])
        if inv != 0 and s != inv:
            o_ts, o_px = q.pop(0)
            hold.append((ts[i] - o_ts) / 1e9)
            pnl.append(inv * (np.exp(logpx[i] - o_px) - 1.0) * 1e4)
            if not q:
                inv = 0
            continue
        if len(q) >= 20:
            continue
        q.append((ts[i], logpx[i]))
        inv = s
    return dict(T=T, markout=np.asarray(mo), horizons=hs,
                hold=np.asarray(hold), pnl=np.asarray(pnl))


# ───────────────────────────── 1. EDGE WATERFALL ──────────────────────────────
def waterfall(res):
    """gross research edge -> fees -> execution slippage -> realized production."""
    hp = pd.read_csv(f"{PROD}/holding_pnl.csv") if os.path.isfile(f"{PROD}/holding_pnl.csv") else None
    osum = pd.read_csv(f"{PROD}/order_summary.csv") if os.path.isfile(f"{PROD}/order_summary.csv") else None
    if res is None or hp is None:
        return "", []
    gross = float(np.nanmean(res["pnl"]))                      # research bps / trip
    realized = float(np.average(hp.realized_bps, weights=hp.qty)) \
        if hp.qty.sum() > 0 else float(hp.realized_bps.mean())
    fees_bps = np.nan
    if osum is not None and {"commission", "turnover"}.issubset(osum.columns):
        turn = osum.turnover.sum()
        if turn > 0:
            # commission per unit turnover, expressed in bps of a round trip (x2 legs)
            fees_bps = float(osum.commission.sum() / turn * 1e4)
    fees = -abs(fees_bps) if np.isfinite(fees_bps) else np.nan
    # residual = everything else (queue position, latency, adverse selection at fill)
    slip = realized - gross - (fees if np.isfinite(fees) else 0.0)
    labels = ["research gross<br>(virtual trip)", "fees", "execution<br>(slippage/latency)",
              "realized<br>(live FIFO trip)"]
    vals = [gross, fees if np.isfinite(fees) else 0.0, slip, realized]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute", "relative", "relative", "total"],
        x=labels, y=vals, text=[f"{v:+.3f}" for v in vals], textposition="outside",
        connector=dict(line=dict(color="#475569")),
        increasing=dict(marker_color="#38a169"), decreasing=dict(marker_color="#e53e3e"),
        totals=dict(marker_color="#2b6cb0"),
        hovertemplate="%{x}<br>%{y:+.4f} bps<extra></extra>"))
    fig.update_layout(title=(f"Edge waterfall — where the researched edge goes "
                             f"({MODEL} ret_{H_REF}, q{Q_REF:g} vs live fills)"),
                      yaxis_title="bps per round trip", height=430,
                      margin=dict(t=60, b=40), **DARK)
    notes = [f"research gross **{gross:+.3f} bps/trip** (virtual, unit size, FIFO cap 20)",
             (f"fees **{fees:+.3f} bps**" if np.isfinite(fees) else "fees — not derivable from order_summary"),
             f"execution residual **{slip:+.3f} bps** (queue position, latency, adverse selection)",
             f"realized **{realized:+.3f} bps/trip** (live FIFO round trips, qty-weighted)"]
    return _fig(fig), notes


# ───────────────────────── 2. MARKOUT: research vs live ───────────────────────
def markout_overlay(res):
    tm = pd.read_csv(f"{PROD}/trade_markout.csv") if os.path.isfile(f"{PROD}/trade_markout.csv") else None
    if res is None and tm is None:
        return ""
    fig = go.Figure()
    if res is not None:
        fig.add_scatter(x=res["horizons"], y=res["markout"], mode="lines",
                        name=f"research (virtual, q{Q_REF:g})",
                        line=dict(color="#2dd4bf", width=2),
                        hovertemplate="+%{x}s<br>%{y:.4f} bps<extra></extra>")
    if tm is not None:
        y = tm["bps_wmean"] if "bps_wmean" in tm else tm["pnl_wmean"]
        fig.add_scatter(x=tm.horizon_s, y=y, mode="lines+markers",
                        name="production (real fills, qty-weighted)",
                        line=dict(color="#f59e0b", width=2), marker=dict(size=3),
                        customdata=np.stack([tm.trades], axis=-1),
                        hovertemplate=("+%{x}s<br>%{y:.4f} bps"
                                       "<br>trades=%{customdata[0]:,}<extra></extra>"))
    fig.add_hline(y=0, line_color="#475569")
    fig.update_layout(title="Markout — research expectation vs production reality "
                            "(identical formula: 1s AlphaPx grid, LOCF, simple returns)",
                      xaxis_title="seconds after fill", yaxis_title="markout (bps)",
                      height=430, margin=dict(t=60), legend=dict(orientation="h", y=1.08),
                      **DARK)
    return _fig(fig)


# ─────────────────────── 3. HOLDING: research vs live ─────────────────────────
BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 90),
           (90, 120), (120, 180), (180, 300), (300, 600), (600, 900),
           (900, 1200), (1200, np.inf)]
BLAB = [f"{int(a)}-{int(b)}s" if np.isfinite(b) else f">{int(a)}s" for a, b in BUCKETS]


def _bucketize(hold, pnl, w=None):
    n, m = [], []
    for a, b in BUCKETS:
        k = (hold >= a) & (hold < b)
        n.append(int(k.sum()))
        if k.any():
            m.append(float(np.average(pnl[k], weights=w[k]) if w is not None else np.mean(pnl[k])))
        else:
            m.append(np.nan)
    return n, m


def holding_overlay(res):
    hp = pd.read_csv(f"{PROD}/holding_pnl.csv") if os.path.isfile(f"{PROD}/holding_pnl.csv") else None
    if res is None and hp is None:
        return ""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=("share of round trips per holding bucket (%)",
                                        "realized PnL (bps) per bucket"))
    if res is not None:
        n, m = _bucketize(res["hold"], res["pnl"])
        tot = max(sum(n), 1)
        fig.add_bar(x=BLAB, y=[100 * v / tot for v in n], name="research (virtual)",
                    marker_color="#2dd4bf", opacity=0.75, row=1, col=1)
        fig.add_bar(x=BLAB, y=m, name="research (virtual)", marker_color="#2dd4bf",
                    opacity=0.75, showlegend=False, row=2, col=1)
    if hp is not None:
        n, m = _bucketize(hp.life_secs.to_numpy(), hp.realized_bps.to_numpy(),
                          hp.qty.to_numpy())
        tot = max(sum(n), 1)
        fig.add_bar(x=BLAB, y=[100 * v / tot for v in n], name="production (real)",
                    marker_color="#f59e0b", opacity=0.75, row=1, col=1)
        fig.add_bar(x=BLAB, y=m, name="production (real)", marker_color="#f59e0b",
                    opacity=0.75, showlegend=False, row=2, col=1)
    fig.update_layout(title="Holding time — research virtual trips vs live FIFO round trips",
                      barmode="group", height=560, margin=dict(t=90),
                      legend=dict(orientation="h", y=1.06), **DARK)
    fig.update_yaxes(title_text="% of trips", row=1, col=1)
    fig.update_yaxes(title_text="realized (bps)", row=2, col=1)
    fig.update_xaxes(title_text="holding time bucket", row=2, col=1)
    return _fig(fig, "560px")


# ───────────────── 4. FEATURE DRIFT: cap-hit rate + PSI vs training ───────────
def _psi(train_q, live, bins=10):
    """population stability index of live vs the training quantile grid."""
    edges = np.unique(np.nanquantile(train_q, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.nan
    t = np.histogram(train_q, bins=edges)[0].astype(float)
    l = np.histogram(live, bins=edges)[0].astype(float)
    t, l = t / max(t.sum(), 1), l / max(l.sum(), 1)
    m = (t > 0) & (l > 0)
    return float(np.sum((l[m] - t[m]) * np.log(l[m] / t[m])))


def feature_drift():
    # NOTE tree models emit an EMPTY cap file by design (kind="tree" -> no
    # transforms), so pick the first caps file that actually has rows — the
    # linear run's caps define the same training-window feature domain.
    caps = None
    for m in (MODEL, "lasso_ols", "mlp"):
        p = _path(m, f"feature_caps_ret_{H_REF}.csv")
        if os.path.isfile(p) and os.path.getsize(p) > 64:
            try:
                c = pd.read_csv(p)
            except Exception:
                continue
            if len(c):
                caps = c.set_index("feature")
                break
    rp = f"{PROD}/resampled.parquet"
    if caps is None or not os.path.isfile(rp):
        return "", "no populated feature_caps (tree runs emit none) or no live parquet"
    live = pd.read_parquet(rp)
    # live feature columns use '__' where configs use '_' suffixes; match loosely
    norm = {c.replace("__", "_"): c for c in live.columns}
    rows = []
    for f, r in caps.iterrows():
        col = norm.get(f) or (f if f in live.columns else None)
        if col is None or not np.issubdtype(live[col].dtype, np.number):
            continue
        v = live[col].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) < 100:
            continue
        lo, hi = float(r.cap_lo_raw), float(r.cap_hi_raw)
        hit = float(np.mean((v < lo) | (v > hi))) * 100
        # training reference sample: reconstruct from the cap params (zcap: mean/sd)
        if r.transform == "zcap" and np.isfinite(r.get("mean", np.nan)):
            ref = np.random.default_rng(7).normal(r["mean"], r["sd"], 20000)
        else:
            ref = np.linspace(lo, hi, 20000)
        rows.append(dict(feature=f, cap_hit_pct=hit, psi=_psi(ref, v),
                         live_mean=float(v.mean()), train_lo=lo, train_hi=hi))
    if not rows:
        return "", "no feature overlap between caps and live parquet"
    d = pd.DataFrame(rows).sort_values("cap_hit_pct", ascending=False)
    top = d.head(30)
    fig = go.Figure()
    fig.add_bar(x=top.feature, y=top.cap_hit_pct, name="% live rows outside training cap",
                marker_color=["#e53e3e" if v > 1 else "#38a169" for v in top.cap_hit_pct],
                customdata=np.stack([top.psi, top.train_lo, top.train_hi, top.live_mean], axis=-1),
                hovertemplate=("%{x}<br>cap-hit=%{y:.2f}%<br>PSI=%{customdata[0]:.3f}"
                               "<br>training range [%{customdata[1]:.3g}, %{customdata[2]:.3g}]"
                               "<br>live mean=%{customdata[3]:.3g}<extra></extra>"))
    fig.update_layout(title="Feature drift — % of LIVE rows outside the TRAINING cap "
                            f"(top 30 of {len(d)}; red > 1%)",
                      yaxis_title="% of live rows clipped", height=470,
                      margin=dict(t=60, b=150), xaxis_tickangle=-45, **DARK)
    worst = ", ".join(f"{r.feature} ({r.cap_hit_pct:.1f}%)" for r in d.head(5).itertuples())
    return _fig(fig, "470px"), worst


# ───────────────── 5. ALPHA DRIFT: live alpha vs validation preds ─────────────
def alpha_drift():
    rp = f"{PROD}/resampled.parquet"
    vp = _path(MODEL, f"valid_pred_ret_{H_REF}.csv")
    if not (os.path.isfile(rp) and os.path.isfile(vp)):
        return "", ""
    live = pd.read_parquet(rp, columns=["alpha"]).alpha.dropna().to_numpy()
    val = pd.read_csv(vp, usecols=[f"Y_hat_{H_REF}"])[f"Y_hat_{H_REF}"].dropna().to_numpy()
    qs = np.linspace(0.01, 0.99, 99)
    fig = go.Figure()
    fig.add_scatter(x=qs, y=np.quantile(val, qs) / np.std(val), name="validation (research)",
                    line=dict(color="#2dd4bf", width=2),
                    hovertemplate="q%{x:.2f}<br>%{y:.3f} sd<extra></extra>")
    fig.add_scatter(x=qs, y=np.quantile(live, qs) / np.std(live), name="live (production)",
                    line=dict(color="#f59e0b", width=2),
                    hovertemplate="q%{x:.2f}<br>%{y:.3f} sd<extra></extra>")
    fig.update_layout(title="Alpha shape drift — standardized quantile curves "
                            "(same shape = the live signal is distributed like the trained one)",
                      xaxis_title="quantile", yaxis_title="value / sd", height=430,
                      margin=dict(t=60), legend=dict(orientation="h", y=1.08), **DARK)
    pn_l = (live > 0).sum() / max((live < 0).sum(), 1)
    pn_v = (val > 0).sum() / max((val < 0).sum(), 1)
    note = (f"P/N ratio — research **{pn_v:.3f}** vs live **{pn_l:.3f}**; "
            f"sd research {np.std(val):.3e} vs live {np.std(live):.3e}; "
            f"|alpha| q99 research {np.quantile(np.abs(val), .99):.3e} vs "
            f"live {np.quantile(np.abs(live), .99):.3e}")
    return _fig(fig), note


def main():
    res = research_trips()
    wf, wf_notes = waterfall(res)
    mo = markout_overlay(res)
    ho = holding_overlay(res)
    fd, fd_note = feature_drift()
    ad, ad_note = alpha_drift()

    def sec(title, body, note=""):
        if not body:
            return ""
        n = f"<div class='note'>{note}</div>" if note else ""
        return f"<section><h2>{title}</h2>{n}{body}</section>"

    body = f"""<header>
  <a class="back-link" href="index.html">&larr; All reports</a>
  <h1>{SYM.upper()} &mdash; model lifecycle: research &rarr; production</h1>
  <div class="meta">research side = unified-pipeline {MODEL} ret_{H_REF} on the
   validation set (virtual unit trades at the q{Q_REF:g} conviction threshold, FIFO,
   inventory cap 20) &middot; production side = {PROD_NAME} live fills
   (order_multi.R / markout.py) &middot; both use the SAME markout formula, so any
   gap is execution or drift, not methodology</div>
</header>
{sec("1. Edge waterfall &mdash; where the researched edge goes", wf,
     " &middot; ".join(wf_notes))}
{sec("2. Markout &mdash; research vs production", mo)}
{sec("3. Holding time &mdash; research vs production", ho)}
{sec("4. Feature drift &mdash; live vs training domain", fd,
     f"worst cap-hit: {fd_note}" if fd_note else "")}
{sec("5. Alpha drift &mdash; live signal vs trained signal", ad, ad_note)}"""

    CSS = """body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;
background:#0f172a;color:#e2e8f0}
header{background:linear-gradient(90deg,#1a202c,#2d3748);padding:18px 28px}
header h1{margin:6px 0 0;font-size:19px}
header .meta{color:#94a3b8;font-size:12px;margin-top:6px;max-width:960px;line-height:1.6}
header .back-link{color:#90cdf4;font-size:13px;text-decoration:none}
section{padding:12px 28px;max-width:1280px}
h2{font-size:15px;color:#cbd5e1;margin:18px 0 8px}
.note{background:#111c33;border:1px solid #1e293b;border-radius:8px;padding:10px 14px;
margin:6px 0 12px;font-size:12px;line-height:1.6;color:#cbd5e1;max-width:1000px}
"""
    js = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    page = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{SYM.upper()} model lifecycle</title>{js}<style>{CSS}</style>"
            f"</head><body>{body}</body></html>")
    open(OUT, "w").write(page)
    print(f"wrote {OUT} ({len(page)//1024} KB)")


if __name__ == "__main__":
    main()
