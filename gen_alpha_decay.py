#!/usr/bin/env python3
"""Alpha-decay markout page: VIRTUAL trades from |pred|-quantile thresholds.

Production markout uses real fills; this is the research twin for alpha decay:
for each threshold T = quantile(|Y_hat|, q), q in the conviction-table tail
buckets, virtual trades are every validation row with Y_hat > +T (long) or
Y_hat < -T (short). Markout(h) = mean over entries of side * (px[t+h]/px[t]-1),
h = 1..300 s, computed on AlphaPx from the validation snapshot (1 s cadence,
exact-span check so data gaps never fabricate a markout).

Entries are per-row and OVERLAPPING (no qty, plain mean — the production
plot's qty-weighting has no analogue for virtual trades).

    python gen_alpha_decay.py [btc] [10]     # symbol, alpha horizon
Then make_index.py to encrypt + publish.
"""
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import to_html

HERE = os.path.dirname(os.path.abspath(__file__))
SYM = (sys.argv[1] if len(sys.argv) > 1 else "btc").lower()
ALPHA_HORIZONS = [int(a) for a in sys.argv[2:]] or [1, 10]
GEN = "pipeline_sqrtw"
STATS = "/home/guanyang/work/alpha_replayer_config/statistics"
MODELS = [("lasso_ols", "Linear (LASSO→WLS)"), ("lgbm", "LightGBM")]
# q=0 is the NO-FILTER baseline: T=0, so every row trades (long if alpha>0,
# short if alpha<0) — the reference line every threshold must beat.
QUANTILES = [0.0, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 0.999]
HORIZONS = np.arange(1, 301)          # seconds after (virtual) fill
NS = 1_000_000_000

DARK = {"bg": "#0f172a", "panel": "#0f172a", "fg": "#e2e8f0", "grid": "#334155"}
LINE_COLORS = ["#64748b", "#818cf8", "#2dd4bf", "#fbbf24", "#f472b6", "#f87171", "#4ade80"]


TOLERANCE_SEC = 2      # max px staleness at t+h — mirrors prod markout.py


def markout_curves(df, H_ALPHA):
    """-> {q: (n_trades, curve bps, T, coverage)} — PROD-ALIGNED logic
    (order_multi.R / markout.py): complete per-second AlphaPx grid with LOCF
    carry, price at t+h looked up on that grid and accepted only if the carried
    px is <= TOLERANCE_SEC stale; SIMPLE return sign*(px_h/px_0 - 1)*1e4.
    coverage = accepted / total entries (prod's trades_total/coverage stat)."""
    ts = df["mark_ts"].to_numpy(np.int64)
    px = df["AlphaPx"].to_numpy(np.float64)
    pred = df[f"Y_hat_{H_ALPHA}"].to_numpy(np.float64)
    sec = ts // NS
    # complete 1s grid + LOCF (exactly order_multi.R's reindex + nafill(locf))
    g0, g1 = int(sec[0]), int(sec[-1])
    grid_px = np.full(g1 - g0 + 1, np.nan)
    grid_px[sec - g0] = px
    grid_age = np.zeros(len(grid_px), np.int64)
    last = np.nan; age = 0
    for i in range(len(grid_px)):
        if np.isnan(grid_px[i]):
            age += 1; grid_px[i] = last
        else:
            age = 0; last = grid_px[i]
        grid_age[i] = age
    thr = {q: float(np.quantile(np.abs(pred), q)) for q in QUANTILES}
    out = {}
    for q, T in thr.items():
        side = np.where(pred > T, 1.0, np.where(pred < -T, -1.0, 0.0))
        eidx = np.flatnonzero(side != 0.0)
        e_sec = sec[eidx] - g0
        e_px = px[eidx]
        e_side = side[eidx]
        curve = np.full(len(HORIZONS), np.nan)
        cov = np.full(len(HORIZONS), np.nan)
        for j, h in enumerate(HORIZONS):
            t_sec = e_sec + h
            ok = t_sec <= g1 - g0
            tp = grid_px[t_sec[ok]]
            fresh = grid_age[t_sec[ok]] <= TOLERANCE_SEC
            good = fresh & ~np.isnan(tp)
            if good.any():
                curve[j] = np.mean(e_side[ok][good]
                                   * (tp[good] / e_px[ok][good] - 1.0)) * 1e4
                cov[j] = good.sum() / max(ok.sum(), 1)
        out[q] = (int(len(eidx)), curve, T, float(np.nanmean(cov)))
    return out


# Production's bucket grid up to 600s, then 300s steps to 1200s (the capped
# unit-per-signal rule holds inventory far longer than the production strategy,
# so a single ">600s" bucket hid most of the mass).
HOLD_BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 45), (45, 60),
                (60, 90), (90, 120), (120, 180), (180, 300), (300, 600),
                (600, 900), (900, 1200), (1200, float("inf"))]


MAX_INVENTORY = 20     # position cap in units (stacking allowed up to this)


def round_trips(ts, logpx, pred, T, max_inventory=MAX_INVENTORY, **_ignored):
    """Virtual round trips — UNIT-SIZE trades, FIFO-matched (production
    order_multi.R semantics with qty = 1) and an INVENTORY CAP:

      alpha >  +T  ->  BUY  1 unit   (skipped if already long max_inventory)
      alpha <  -T  ->  SELL 1 unit   (skipped if already short max_inventory)

    Opposing units match FIFO; each matched pair is one round trip
    (hold = close_ts - open_ts, realized = open_dir*(close_px/open_px - 1)).
    Inventory is flattened at a cadence gap so a stalled feed cannot fabricate
    a long hold.

    WHY THE CAP: the signal is not sign-balanced (btc ret_10 fires 9.5% more
    shorts than longs on the validation set = ~11.5k unmatched units), so an
    UNCAPPED unit-per-signal rule accumulates inventory for days — FIFO then
    reports 40-hour median holds and tens of bps per trip, which measures
    directional drift, not the alpha. max_inventory=1 is the plain reading of
    "each trade is size 1": flat or +/-1, a signal on the other side closes the
    open unit (and the next opposite signal opens the other way).
    """
    n = len(ts)
    seg = np.cumsum(np.r_[False, np.diff(ts) != NS])
    d = np.where(pred > T, 1, np.where(pred < -T, -1, 0)).astype(np.int8)
    idx = np.flatnonzero(d != 0)
    hold, pnl = [], []
    n_long = int((d > 0).sum()); n_short = int((d < 0).sum())   # signal P/N counts
    queue = []                       # open units (ts, logpx), all one side
    inv_dir, cur_seg = 0, -1
    for i in idx:
        if seg[i] != cur_seg:        # new contiguous block -> start flat
            queue.clear(); inv_dir = 0; cur_seg = seg[i]
        s_i = int(d[i])
        if inv_dir != 0 and s_i != inv_dir:          # opposite -> close one FIFO unit
            o_ts, o_px = queue.pop(0)
            hold.append((ts[i] - o_ts) / 1e9)
            pnl.append(inv_dir * (np.exp(logpx[i] - o_px) - 1.0) * 1e4)
            if not queue:
                inv_dir = 0
            continue
        if len(queue) >= max_inventory:              # at the cap -> no new unit
            continue
        queue.append((ts[i], logpx[i]))
        inv_dir = s_i
    return np.asarray(hold), np.asarray(pnl), (n_long, n_short)


def _fig_html(fig, height="480px"):
    """Embeddable div, plotly.js loaded once per page — prod build_report.py helper."""
    return to_html(fig, include_plotlyjs=False, full_html=False,
                   default_height=height)


def render_holding_fig(hold, pnl, label, T, q, pn=(0, 0)):
    labels = [f"{int(a)}-{int(b)}s" if np.isfinite(b) else f">{int(a)}s"
              for a, b in HOLD_BUCKETS]
    counts, means = [], []
    for a, b in HOLD_BUCKETS:
        m = (hold >= a) & (hold < b)
        counts.append(int(m.sum()))
        means.append(float(np.nanmean(pnl[m])) if m.any() else np.nan)
    g = pd.DataFrame({"bucket": labels, "trades": counts, "bps": means})
    g = g[g.trades > 0]
    g["trade_pct"] = 100.0 * g.trades / g.trades.sum()
    # NOTE the figure title is two lines (stats subtitle), so the first subplot
    # title must start lower or they overlap — hence the extra top margin plus a
    # taller figure and explicit subplot-title placement.
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                        subplot_titles=("trades per holding bucket",
                                        "realized PnL (bps) per bucket"))
    fig.add_bar(x=g.bucket, y=g.trades, name="trades", marker_color="#2b6cb0",
                customdata=np.stack([g.trade_pct], axis=-1),
                hovertemplate=("%{x}<br>trades=%{y} (%{customdata[0]:.1f}%)"
                               "<extra></extra>"),
                row=1, col=1)
    fig.add_bar(x=g.bucket, y=g.bps, name="realized bps",
                marker_color=["#38a169" if v >= 0 else "#e53e3e" for v in g.bps],
                hovertemplate="%{x}<br>realized=%{y:.3f} bps<extra></extra>",
                row=2, col=1)
    fig.add_hline(y=0, line_color="#999", row=2, col=1)
    overall = float(np.nanmean(pnl)) if len(pnl) else float("nan")
    avg_hold = float(np.mean(hold)) if len(hold) else float("nan")
    med_hold = float(np.median(hold)) if len(hold) else float("nan")
    ratio = (pn[0] / pn[1]) if pn[1] else float("nan")
    fig.update_layout(
        title=((f"{label} — ALL alpha, no threshold: " if q == 0 else
                f"{label} — q{q:g} (T={T:.2e}): ")
               + f"{len(pnl):,} virtual round trips, "
               f"overall {overall:+.3f} bps<br><sub>avg hold {avg_hold:,.0f}s "
               f"(median {med_hold:,.0f}s) &nbsp;|&nbsp; alpha P/N ratio {ratio:.3f} "
               f"({pn[0]:,} long vs {pn[1]:,} short signals)</sub>"),
        template="plotly_white", margin=dict(t=118, b=60), showlegend=False,
        height=540, title_y=0.97, title_yanchor="top")
    fig.update_yaxes(title_text="trades", row=1, col=1)
    fig.update_yaxes(title_text="realized (bps)", row=2, col=1)
    fig.update_xaxes(title_text="holding time bucket", row=2, col=1)
    for ann in fig.layout.annotations:          # subplot titles: shift below the header
        ann.y = ann.y - 0.045
        ann.font.size = 13
    return _fig_html(fig, height="540px")


LINE_COLORS = ["#0f172a", "#94a3b8", "#818cf8", "#2b6cb0", "#d69e2e", "#ed64a6",
               "#e53e3e", "#38a169", "#805ad5"]


def render_fig(curves, label, H_ALPHA):
    fig = go.Figure()
    for (q, (n, c, T, cv)), col in zip(curves.items(), LINE_COLORS):
        fig.add_scatter(x=HORIZONS, y=c, mode="lines+markers",
                        name=("ALL alpha, no threshold "
                              f"(n={n:,})" if q == 0 else f"q{q:g} (n={n:,})"),
                        line=dict(color=col, width=2.4 if q == 0 else 1.6,
                                  dash="dash" if q == 0 else "solid"),
                        marker=dict(size=0 if q == 0 else 3),
                        hovertemplate=("+%{x}s after fill<br>markout=%{y:.4f} bps"
                                       + ("<br>NO threshold filter (all rows)"
                                          if q == 0 else
                                          f"<br>threshold q{q:g} (T={T:.2e})")
                                       + f"<br>trades={n:,}  coverage={cv * 100:.1f}%"
                                       + "<extra></extra>"))
    fig.add_hline(y=0, line_color="#999")
    fig.add_vline(x=H_ALPHA, line_color="#94a3b8", line_dash="dash")
    fig.update_layout(
        title=f"{label} — virtual-trade markout (bps) vs horizon after fill, "
              f"by |pred| quantile threshold (long > +T / short < −T)",
        xaxis_title="seconds after fill (mark_ts + h)",
        yaxis_title="markout (bps)",
        template="plotly_white", margin=dict(t=50))
    return _fig_html(fig)

sections = []
for H_ALPHA in ALPHA_HORIZONS:
  for key, label in MODELS:
    p = f"{STATS}/{SYM}/{key}/20260602/{GEN}/valid_pred_ret_{H_ALPHA}.csv"
    if not os.path.isfile(p):
        print(f"skip {key} ret_{H_ALPHA}: no {p}")
        continue
    df = pd.read_csv(p, usecols=["mark_ts", "AlphaPx", f"Y_hat_{H_ALPHA}"])
    df = df.sort_values("mark_ts").reset_index(drop=True)
    curves = markout_curves(df, H_ALPHA)
    fig_div = render_fig(curves, f"{label} — alpha ret_{H_ALPHA}", H_ALPHA)
    peak = {q: (float(np.nanmax(c)), int(HORIZONS[int(np.nanargmax(c))]))
            for q, (n, c, T, _cv) in curves.items()}
    for q in QUANTILES:
        print(f"  [{key} ret_{H_ALPHA}] q{q:g}: n={curves[q][0]:,} "
              f"@{H_ALPHA}s {curves[q][1][H_ALPHA-1]:+.3f}bp "
              f"peak {peak[q][0]:+.3f}bp@{peak[q][1]}s  @300s {curves[q][1][-1]:+.3f}bp")
    rows = "".join(
        f"<tr><td>{'ALL (no filter)' if q == 0 else f'q{q:g}'}</td>"
        f"<td>{curves[q][2]:.3e}</td><td>{curves[q][0]:,}</td>"
        f"<td>{curves[q][1][H_ALPHA - 1]:+.3f}</td>"
        f"<td>{peak[q][0]:+.3f} @ {peak[q][1]}s</td>"
        f"<td>{curves[q][1][-1]:+.3f}</td><td>{curves[q][3]:.3f}</td></tr>" for q in QUANTILES)
    # holding-time vs realized PnL, one compact two-panel per threshold
    ts_a = df["mark_ts"].to_numpy(np.int64)
    logpx_a = np.log(df["AlphaPx"].to_numpy(np.float64))
    pred_a = df[f"Y_hat_{H_ALPHA}"].to_numpy(np.float64)
    hold_imgs = []
    for q in QUANTILES:
        T = curves[q][2]
        hold, pnl, pn = round_trips(ts_a, logpx_a, pred_a, T)
        hold_imgs.append(render_holding_fig(
            hold, pnl, f"{label} alpha ret_{H_ALPHA}", T, q, pn))
        print(f"  [{key} ret_{H_ALPHA}] holding q{q:g}: {len(pnl):,} round trips, "
              f"overall {np.nanmean(pnl):+.3f}bp, avg hold {np.mean(hold):.0f}s "
              f"(median {np.median(hold):.0f}s), P/N {pn[0]/max(pn[1],1):.3f}")
    sections.append(f"""
<section>
  <h2>{label} — alpha ret_{H_ALPHA}</h2>
  {fig_div}
  <table class="cmp">
   <tr><th>threshold</th><th>T (|pred|)</th><th>virtual trades</th>
       <th>markout @alpha hzn (bps)</th><th>peak (bps)</th><th>@300s (bps)</th><th>px coverage</th></tr>
   {rows}
  </table>
  <details class="hold" open><summary>Holding time vs realized PnL — virtual round trips
   (UNIT-SIZE trades, FIFO-matched like production: BUY 1 when alpha &gt; +T, SELL 1 when
   alpha &lt; &minus;T, position capped at &plusmn;20 units, opposing units matched
   first-in-first-out, inventory flattened at a data gap), one panel per threshold</summary>
   {''.join(hold_imgs)}
  </details>
</section>""")
    print(f"{key}: done ({len(df):,} rows)")

body = f"""<header>
  <a class="back-link" href="index.html">&larr; All reports</a>
  <h1>{SYM.upper()} &mdash; alpha decay: virtual-trade markout by conviction threshold</h1>
  <div class="meta">VIRTUAL trades (research twin of the production fill-markout plot):
   entry at every validation row with Y_hat &gt; +T (long) / &lt; &minus;T (short),
   T = |pred| quantile from the conviction table&rsquo;s highlighted tail buckets &middot;
   markout logic PROD-ALIGNED (order_multi.R): 1s AlphaPx grid + LOCF, staleness tolerance 2s, simple returns, coverage reported &middot; alpha horizons: {", ".join(f"ret_{h}" for h in ALPHA_HORIZONS)} &middot;
   overlapping entries, unweighted mean (no qty for virtual trades) &middot;
   validation set only (last 20%)</div>
</header>
{''.join(sections)}"""

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;
background:#ffffff;color:#1a202c}
header{background:#f7fafc;border-bottom:1px solid #e2e8f0;padding:18px 28px}
header h1{margin:6px 0 0;font-size:19px}
header .meta{color:#4a5568;font-size:12px;margin-top:6px;max-width:960px;line-height:1.5}
header .back-link{color:#2b6cb0;font-size:13px;text-decoration:none}
section{padding:14px 28px;max-width:1280px}
h2{font-size:15px;color:#2d3748;margin:16px 0 8px}
table.cmp{border-collapse:collapse;font-size:12.5px;margin-top:10px}
table.cmp th,table.cmp td{border:1px solid #cbd5e0;padding:5px 12px;text-align:right}
table.cmp th{background:#edf2f7;color:#4a5568;font-weight:600}
details.hold{margin-top:14px}
details.hold summary{cursor:pointer;color:#975a16;font-size:13px;max-width:960px;line-height:1.5}
"""

plotly_js = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
page = ("<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{SYM.upper()} alpha decay</title>{plotly_js}<style>{CSS}</style></head>"
        f"<body>{body}</body></html>")
OUT = os.path.join(HERE, f"{SYM}_alpha_decay.html")
open(OUT, "w").write(page)
print(f"wrote {OUT} ({len(page)//1024} KB)")
