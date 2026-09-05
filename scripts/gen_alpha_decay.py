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

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYM = (sys.argv[1] if len(sys.argv) > 1 else "btc").lower()
# non-numeric trailing arg = data version (e.g. "v2"): *_<ver> generations,
# <sym>_alpha_decay_<ver>.html output
# `cal`/`calibrated` flag = use the CALIBRATED Y_hat (else Y_hat_raw); INDEPENDENT
# of the data version. VER (v2/v3) selects the dump; calibration is orthogonal.
_a2 = [a.lower() for a in sys.argv[2:]]
CALIB = any(x in ("cal", "calibrated", "--calibrated") for x in _a2)
_a2 = [a for a in _a2 if a not in ("cal", "calibrated", "--calibrated")]
VER = next((a for a in _a2 if not a.isdigit()), "")
ALPHA_HORIZONS = [int(a) for a in _a2 if a.isdigit()] or [1, 10]
GEN = "pipeline_sqrtw"
STATS = "/home/guanyang/work/alpha_replayer_config/statistics"
# (key, label, generation) — keep in step with gen_pipeline_models.MODELS so the
# decay view always covers the same models the model page compares.
MODELS = [("lasso_ols", "Linear (LASSO→WLS)", "pipeline_sqrtw"),
          ("lgbm", "LightGBM", "pipeline_sqrtw"),
          ("lgbm_tuned", "LGBM tuned (per-horizon)", "pipeline_tuned_all"),
          # horizon-combined signal (combine_signals.py); NOTE its weights are
          # fitted on the FIRST half of the validation window, so slice-A rows
          # are mildly in-sample for this line (4 fitted weights — small).
          ("combo", "Combined (horizon stack)", "pipeline_combo"),
          ("lgbm_capped", "LGBM capped (uniform norm)", "pipeline_capstats")]
if VER:
    MODELS = [(k, l, f"{g}_{VER}") for k, l, g in MODELS]
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
# A missing SECOND on the mark grid is a quiet market, not a feed outage — the
# metals grid skips 67k seconds in 22 days (vs ONE real gap > 60s), and the old
# flatten-on-any-missing-second rule discarded 95% of open units there (1,795
# signals -> 45 trips; with this threshold -> 854). Only a real stall flattens.
GAP_FLATTEN_SEC = 60
GRID_N = 1500          # points per position/equity timeseries (page-size cap)
# transaction cost per round trip, booked at trip close — crypto perps pay
# taker fees ~1.8bp/trip; metals (tokenized) modeled at 0.6bp
COST_BPS = {"btc": 1.8, "eth": 1.8, "sol": 1.8}.get(SYM, 0.6)


def round_trips(ts, logpx, pred, T, max_inventory=MAX_INVENTORY,
                t_long=None, t_short=None, close_level=None, **_ignored):
    """Virtual round trips — UNIT-SIZE trades, FIFO-matched (production
    order_multi.R semantics with qty = 1) and an INVENTORY CAP:

      alpha >  +T  ->  BUY  1 unit   (skipped if already long max_inventory)
      alpha <  -T  ->  SELL 1 unit   (skipped if already short max_inventory)

    Opposing units match FIFO; each matched pair is one round trip
    (hold = close_ts - open_ts, realized = open_dir*(close_px/open_px - 1)).
    Inventory is flattened at a data gap > GAP_FLATTEN_SEC (a stalled feed
    must not fabricate a long hold; a missing second is just a quiet market).

    WHY THE CAP: the signal is not sign-balanced (btc ret_10 fires 9.5% more
    shorts than longs on the validation set = ~11.5k unmatched units), so an
    UNCAPPED unit-per-signal rule accumulates inventory for days — FIFO then
    reports 40-hour median holds and tens of bps per trip, which measures
    directional drift, not the alpha. max_inventory=1 is the plain reading of
    "each trade is size 1": flat or +/-1, a signal on the other side closes the
    open unit (and the next opposite signal opens the other way).
    """
    n = len(ts)
    seg = np.cumsum(np.r_[False, np.diff(ts) > GAP_FLATTEN_SEC * NS])
    if t_long is not None:      # BALANCED per-side thresholds (P/N = 1 by design)
        d = np.where(pred > t_long, 1, np.where(pred < t_short, -1, 0)).astype(np.int8)
    else:
        d = np.where(pred > T, 1, np.where(pred < -T, -1, 0)).astype(np.int8)
    if close_level is not None:      # asymmetric band rule: wider event set
        idx = np.flatnonzero((d != 0) | (pred > close_level) | (pred < -close_level))
    else:
        idx = np.flatnonzero(d != 0)
    hold, pnl, trip_sides, trip_ts = [], [], [], []
    n_long = int((d > 0).sum()); n_short = int((d < 0).sum())   # signal P/N counts
    sgn = d[idx]
    n_flips = int((np.diff(sgn) != 0).sum()) if len(sgn) > 1 else 0
    cap_skips = 0; gap_discards = 0
    queue = []                       # open units (ts, logpx), all one side
    inv_dir, cur_seg = 0, -1
    # Event replay records the queue state AFTER each signal event; equity is
    # then evaluated at EVERY price row vectorized (real + inv*(e^lp * S - n),
    # S = sum(e^-lp_open) over the queue), in bps SUMMED PER UNIT. Max drawdown
    # comes from this FULL-RESOLUTION series — a plot-grid-sampled equity curve
    # understates it (measured -3959 vs true -4473 bps on xau v2 q0.999); the
    # GRID_N uniform time grid is used for PLOTTING only.
    ev_row, ev_inv, ev_S, ev_n, ev_real, ev_trips = [], [], [], [], [], []
    real_cum = 0.0; S = 0.0
    for i in idx:
        if seg[i] != cur_seg:        # new contiguous block -> start flat
            gap_discards += len(queue)
            queue.clear(); inv_dir = 0; S = 0.0; cur_seg = seg[i]
        s_i = int(d[i])
        if close_level is not None:                  # band rule: close on the
            p_i = pred[i]                            # OPPOSITE close level
            do_close = ((inv_dir > 0 and p_i < -close_level)
                        or (inv_dir < 0 and p_i > close_level))
        else:
            do_close = inv_dir != 0 and s_i != inv_dir
        if do_close:                                 # close one FIFO unit
            o_ts, o_px = queue.pop(0); S -= np.exp(-o_px)
            hold.append((ts[i] - o_ts) / 1e9)
            trip = inv_dir * (np.exp(logpx[i] - o_px) - 1.0) * 1e4
            pnl.append(trip); trip_sides.append(inv_dir); trip_ts.append(ts[i])
            real_cum += trip
            if not queue:
                inv_dir = 0
        elif s_i != 0 and (inv_dir == 0 or s_i == inv_dir) \
                and len(queue) < max_inventory:      # open one unit
            queue.append((ts[i], logpx[i])); S += np.exp(-logpx[i])
            inv_dir = s_i
        elif s_i != 0:
            cap_skips += 1
        ev_row.append(i); ev_inv.append(inv_dir); ev_S.append(S)
        ev_n.append(len(queue)); ev_real.append(real_cum)
        ev_trips.append(len(pnl))

    if ev_row:
        ev_row_a = np.asarray(ev_row)
        k = np.searchsorted(ev_row_a, np.arange(n), side="right") - 1
        valid = k >= 0
        inv_a = np.where(valid, np.asarray(ev_inv)[k], 0)
        S_a = np.where(valid, np.asarray(ev_S)[k], 0.0)
        n_a = np.where(valid, np.asarray(ev_n)[k], 0)
        real_a = np.where(valid, np.asarray(ev_real)[k], 0.0)
        cost_a = np.where(valid, np.asarray(ev_trips)[k], 0) * COST_BPS
        # a row in a LATER segment than its state event sits past a data gap:
        # inventory was flattened AT the gap (discarded, never realized) — mark
        # it flat there instead of fabricating cross-gap unrealized swings
        gap_flat = valid & (seg != seg[ev_row_a[np.clip(k, 0, None)]])
        inv_a = np.where(gap_flat, 0, inv_a)
        S_a = np.where(gap_flat, 0.0, S_a)
        n_a = np.where(gap_flat, 0, n_a)
        eq_full = real_a + inv_a * (np.exp(logpx) * S_a - n_a) * 1e4
        pos_full = inv_a * n_a
        cummax = np.maximum.accumulate(eq_full)
        t_tr = int(np.argmin(eq_full - cummax))
        maxdd = float(eq_full[t_tr] - cummax[t_tr])
        t_pk = int(np.argmax(eq_full[:t_tr + 1]))     # first peak before trough
        fmt = lambda t: pd.to_datetime(t).strftime("%Y-%m-%d %H:%M:%S")
        ddinfo = dict(
            pk=fmt(ts[t_pk]), tr=fmt(ts[t_tr]),
            px_pk=float(np.exp(logpx[t_pk])), px_tr=float(np.exp(logpx[t_tr])),
            move_bp=float((np.exp(logpx[t_tr] - logpx[t_pk]) - 1.0) * 1e4),
            pos_tr=int(pos_full[t_tr]),
            real_chg=float(real_a[t_tr] - real_a[t_pk]))
    else:
        eq_full = np.zeros(n); pos_full = np.zeros(n); real_a = np.zeros(n)
        cost_a = np.zeros(n)
        maxdd = 0.0; ddinfo = None
    grid = np.linspace(ts[0], ts[-1], GRID_N).astype(np.int64)
    j = np.searchsorted(ts, grid, side="right") - 1
    j = np.clip(j, 0, n - 1)
    curves = (pd.to_datetime(grid), pos_full[j], real_a[j], eq_full[j], maxdd,
              ddinfo, cost_a[j])
    acct = dict(n_long=n_long, n_short=n_short, n_flips=n_flips,
                trips=len(pnl), cap_skips=cap_skips,
                gap_discards=gap_discards, open_end=len(queue),
                trip_sides=np.asarray(trip_sides),
                trip_ts=np.asarray(trip_ts, dtype=np.int64))
    return np.asarray(hold), np.asarray(pnl), (n_long, n_short), curves, acct


def _fig_html(fig, height="480px"):
    """Embeddable div, plotly.js loaded once per page — prod build_report.py helper."""
    return to_html(fig, include_plotlyjs=False, full_html=False,
                   default_height=height)


# ---- open/close threshold GRID SEARCH ---------------------------------------
# Asymmetric band rule generalizing the page's base rule (which is the
# diagonal open==close case): OPEN a unit when pred crosses +/-T_open
# (stacking to the cap), CLOSE one FIFO unit when pred crosses the OPPOSITE
# close level -/+C. C < T_open exits on weaker counter-evidence (earlier,
# shorter holds, more trips x cost); C == T_open is the base rule.
GRID_Q_OPEN = [round(0.80 + 0.01 * i, 2) for i in range(16)] \
              + [0.96, 0.98, 0.99, 0.995, 0.999]   # 0.80..0.95 by 0.01 + tail
GRID_Q_CLOSE = [0.0, 0.5, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97,
                0.98, 0.99]


def grid_cell(ts, logpx, pred, t_open, c_close, max_inventory=MAX_INVENTORY):
    n = len(ts)
    seg = np.cumsum(np.r_[False, np.diff(ts) > GAP_FLATTEN_SEC * NS])
    lvl = min(t_open, c_close)
    cand = (np.flatnonzero((pred > lvl) | (pred < -lvl)) if lvl > 0
            else np.arange(n))
    hold, pnl = [], []
    n_ltrips = 0; n_strips = 0; n_discard = 0
    ev_row, ev_inv, ev_S, ev_n, ev_real = [], [], [], [], []
    queue = []; inv = 0; cur = -1; S = 0.0; real = 0.0
    for i in cand:
        p = pred[i]
        if seg[i] != cur:
            n_discard += len(queue)
            queue.clear(); inv = 0; S = 0.0; cur = seg[i]
        if inv > 0 and p < -c_close:            # close one long unit
            o_ts, o_px = queue.pop(0); S -= np.exp(-o_px)
            hold.append((ts[i] - o_ts) / 1e9)
            tr = (np.exp(logpx[i] - o_px) - 1.0) * 1e4
            pnl.append(tr); real += tr; n_ltrips += 1
            if not queue: inv = 0
        elif inv < 0 and p > c_close:           # close one short unit
            o_ts, o_px = queue.pop(0); S -= np.exp(-o_px)
            hold.append((ts[i] - o_ts) / 1e9)
            tr = -(np.exp(logpx[i] - o_px) - 1.0) * 1e4
            pnl.append(tr); real += tr; n_strips += 1
            if not queue: inv = 0
        elif p > t_open and inv >= 0 and len(queue) < max_inventory:
            queue.append((ts[i], logpx[i])); S += np.exp(-logpx[i]); inv = 1
        elif p < -t_open and inv <= 0 and len(queue) < max_inventory:
            queue.append((ts[i], logpx[i])); S += np.exp(-logpx[i]); inv = -1
        else:
            continue
        ev_row.append(i); ev_inv.append(inv); ev_S.append(S)
        ev_n.append(len(queue)); ev_real.append(real)
    if not pnl:
        return None
    if ev_row:
        ev_row_a = np.asarray(ev_row)
        k = np.searchsorted(ev_row_a, np.arange(n), side="right") - 1
        valid = k >= 0
        inv_a = np.where(valid, np.asarray(ev_inv)[k], 0)
        S_a = np.where(valid, np.asarray(ev_S)[k], 0.0)
        n_a = np.where(valid, np.asarray(ev_n)[k], 0)
        real_a = np.where(valid, np.asarray(ev_real)[k], 0.0)
        gap_flat = valid & (seg != seg[ev_row_a[np.clip(k, 0, None)]])
        inv_a = np.where(gap_flat, 0, inv_a)
        S_a = np.where(gap_flat, 0.0, S_a); n_a = np.where(gap_flat, 0, n_a)
        eq = real_a + inv_a * (np.exp(logpx) * S_a - n_a) * 1e4
        maxdd = float((eq - np.maximum.accumulate(eq)).min())
        total = float(eq[-1])
    else:
        maxdd, total = 0.0, 0.0
    pnl = np.asarray(pnl)
    opened = len(pnl) + n_discard + len(queue)
    return dict(trips=len(pnl), bps=float(pnl.mean()),
                net=float(pnl.mean()) - COST_BPS,
                net_total=total - len(pnl) * COST_BPS,
                maxdd=maxdd, hold=float(np.mean(hold)),
                pn=n_ltrips / max(n_strips, 1),
                realized_share=len(pnl) / max(opened, 1),
                n_discard=n_discard, open_end=len(queue))


def render_grid(ts, logpx, pred, label):
    """-> html table: rows q_open, cols q_close (+ base diagonal), heat by
    NET total after cost; best cells marked."""
    cells = {}
    for qo in GRID_Q_OPEN:
        t_open = float(np.quantile(np.abs(pred), qo))
        for qc in GRID_Q_CLOSE + ["base"]:
            c = t_open if qc == "base" else float(np.quantile(np.abs(pred), qc))
            cells[(qo, qc)] = grid_cell(ts, logpx, pred, t_open, c)
    vals = [c["net_total"] for c in cells.values() if c]
    if not vals:
        return ""
    vmax = max(abs(v) for v in vals) or 1.0
    best_nt = max((k for k in cells if cells[k]), key=lambda k: cells[k]["net_total"])
    best_np = max((k for k in cells if cells[k]), key=lambda k: cells[k]["net"])
    best_cal = max((k for k in cells if cells[k]),
                   key=lambda k: cells[k]["net_total"] / max(abs(cells[k]["maxdd"]), 1e-9))
    cols = GRID_Q_CLOSE + ["base"]
    head = "".join(f"<th>{'close = open (base)' if c == 'base' else f'q_close {c:g}'}</th>"
                   for c in cols)
    rows_html = []
    for qo in GRID_Q_OPEN:
        tds = []
        for qc in cols:
            c = cells[(qo, qc)]
            if not c:
                tds.append("<td>&mdash;</td>"); continue
            a = c["net_total"] / vmax
            bg = (f"rgba(56,161,105,{0.12 + 0.45 * a:.2f})" if a >= 0
                  else f"rgba(229,62,62,{0.12 + 0.45 * (-a):.2f})")
            marks = (('<span class="gm gm1">&#9733;</span>' if (qo, qc) == best_nt else "")
                     + ('<span class="gm gm2">&#9670;</span>' if (qo, qc) == best_np else "")
                     + ('<span class="gm gm3">&#9650;</span>' if (qo, qc) == best_cal else ""))
            tip = (f"open q{qo:g} / close {'=open' if qc == 'base' else f'q{qc:g}'}: "
                   f"{c['trips']:,} trips (long/short trip ratio {c['pn']:.2f}), "
                   f"{c['bps']:+.3f} gross ({c['net']:+.3f} net) "
                   f"bps/trade, avg hold {c['hold']:,.0f}s, "
                   f"net total {c['net_total']:+,.0f} bps, maxDD {c['maxdd']:+,.0f} bps, "
                   f"realized share {c['realized_share']*100:.0f}% "
                   f"({c['n_discard']} gap-discards, {c['open_end']} open at end)")
            ntr = (f"{c['trips'] / 1000:.1f}k" if c['trips'] >= 1000
                   else f"{c['trips']}")
            tds.append(f'<td style="background:{bg}" title="{tip}">'
                       f"<b>{c['net_total']:+,.0f}</b>{marks}"
                       f"<span class='gsub'>net {c['net']:+.2f}/tr &middot; "
                       f"gr {c['bps']:+.2f}</span>"
                       f"<span class='gsub'>{ntr} trips &middot; "
                       f"P/N {c['pn']:.2f}</span></td>")
        rows_html.append(f"<tr><td>q_open {qo:g}</td>{''.join(tds)}</tr>")
    return (f"<details class='acct'><summary>OPEN/CLOSE threshold GRID research "
            f"({label}): open a unit at |pred| &gt; T_open(q_open), close one FIFO "
            f"unit when pred crosses the OPPOSITE close level (q_close); "
            f"'close = open' = the base rule above. Cell = NET total after "
            f"{COST_BPS}bp/trade (heat), net bps/trade and trips below; "
            f"&#9733; best net total &middot; &#9670; best net/trade &middot; "
            f"&#9650; best net-total/maxDD. CAVEAT: picked on THIS validation "
            f"window &mdash; confirm on the next window before trading it.</summary>"
            f"<div class='scrollx'><table class='grid'><tr><th></th>{head}</tr>"
            f"{''.join(rows_html)}</table></div></details>"), cells


# ---- WINNER SELECTION over the open/close grid ------------------------------
# Criteria (also summarized on the page):
#   1. VIABILITY: net/trade >= 0.5 x cost (buffer vs cost-model error),
#      trips >= 200, net total > 0.
#   2. PLATEAU, not peak: rank by the 3x3-neighborhood MEAN of net total on the
#      (q_open x q_close) lattice — an isolated green cell is luck.
#   3. CONFIDENCE: per-trip t-stat of NET PnL >= 2.
#   4. CONSISTENCY: positive net total in >= 3 of 4 equal sub-periods of the
#      validation window (a cell that earned everything in one burst is regime
#      luck, not a rule).
#   Survivors ranked by plateau score, capped at MAX_WINNERS.
MAX_WINNERS = 10


def select_winners(cells, ts, logpx, pred):
    cols = GRID_Q_CLOSE + ["base"]
    n_r, n_c = len(GRID_Q_OPEN), len(cols)
    M = np.full((n_r, n_c), np.nan)
    for i, qo in enumerate(GRID_Q_OPEN):
        for j, qc in enumerate(cols):
            c = cells.get((qo, qc))
            if c:
                M[i, j] = c["net_total"]
    smooth = np.full_like(M, np.nan)
    for i in range(n_r):
        for j in range(n_c):
            blk = M[max(0, i - 1):i + 2, max(0, j - 1):j + 2]
            if np.isfinite(blk).any():
                smooth[i, j] = np.nanmean(blk)
    cand = []
    for i, qo in enumerate(GRID_Q_OPEN):
        for j, qc in enumerate(cols):
            c = cells.get((qo, qc))
            if not c:
                continue
            if (c["net"] >= 0.5 * COST_BPS and c["trips"] >= 200
                    and c["net_total"] > 0 and np.isfinite(smooth[i, j])
                    and c.get("realized_share", 1.0) >= 0.8):
                cand.append((float(smooth[i, j]), qo, qc, c))
    cand.sort(key=lambda x: -x[0])
    q_edges = np.linspace(ts[0], ts[-1], 5).astype(np.int64)
    winners = []
    for sm, qo, qc, c in cand[:3 * MAX_WINNERS]:
        t_open = float(np.quantile(np.abs(pred), qo))
        cl = None if qc == "base" else float(np.quantile(np.abs(pred), qc))
        hold, pnl, pn, curves, acct = round_trips(ts, logpx, pred, t_open,
                                                  close_level=cl)
        if len(pnl) < 200:
            continue
        pnl_a = np.asarray(pnl)
        sd = float(pnl_a.std())
        tstat = float((pnl_a.mean() - COST_BPS) / sd * np.sqrt(len(pnl_a))) if sd > 0 else 0.0
        tt = acct["trip_ts"]
        qnet = []
        for k in range(4):
            mk = (tt >= q_edges[k]) & (tt < q_edges[k + 1] if k < 3 else tt <= q_edges[4])
            qnet.append(float(pnl_a[mk].sum() - COST_BPS * mk.sum()))
        consec = int(sum(v > 0 for v in qnet))
        if tstat < 2.0 or consec < 3:
            continue
        winners.append(dict(qo=qo, qc=qc, cell=c, smooth=sm, tstat=tstat,
                            qnet=qnet, consec=consec, hold=hold, pnl=pnl_a,
                            pn=pn, curves=curves, acct=acct, t_open=t_open,
                            close_level=cl))
        if len(winners) >= MAX_WINNERS:
            break
    return winners


def winner_reason(w, cells, h_alpha):
    """Auto-generated WHY for one selected threshold pair, from its measured
    diagnostics vs the base rule at the same q_open."""
    c = w["cell"]; qo, qc = w["qo"], w["qc"]
    base = cells.get((qo, "base"))
    parts = []
    parts.append(
        f"<b>Entry quality clears cost with margin:</b> gross {c['bps']:+.2f} "
        f"bps/trade vs {COST_BPS} bp cost &rarr; net {c['net']:+.2f}, "
        f"t = {w['tstat']:.1f} over {c['trips']:,} trips — the edge is "
        f"statistically real, not a handful of lucky fills.")
    if qc != "base" and base:
        dd_cut = (1 - abs(c["maxdd"]) / max(abs(base["maxdd"]), 1e-9)) * 100
        if dd_cut > 10:
            parts.append(
                f"<b>The close level does the risk work:</b> exiting on weaker "
                f"counter-evidence (q_close {qc:g} &lt; q_open {qo:g}) cuts maxDD "
                f"{dd_cut:.0f}% vs close=open at the same entry "
                f"({c['maxdd']:+,.0f} vs {base['maxdd']:+,.0f} bps) — early exits "
                f"stop adverse-run stacking before the book fills to the cap.")
        else:
            parts.append(
                f"<b>Asymmetric band:</b> q_close {qc:g} vs q_open {qo:g} trades "
                f"{c['trips']:,} vs the base rule's {base['trips']:,} trips "
                f"(net total {c['net_total']:+,.0f} vs {base['net_total']:+,.0f} bps).")
    elif base:
        parts.append(
            "<b>The symmetric rule wins at this entry level:</b> holding until "
            "equal-and-opposite conviction captures the full swing between "
            "signal extremes.")
    hold_mean = float(np.mean(w["hold"])) if len(w["hold"]) else float("nan")
    if np.isfinite(hold_mean) and hold_mean > 3 * h_alpha:
        parts.append(
            f"<b>It harvests trend persistence past the label horizon:</b> avg "
            f"hold {hold_mean:,.0f}s vs the {h_alpha}s alpha label — consistent "
            f"with the markout curves accruing well beyond the nominal horizon.")
    elif np.isfinite(hold_mean):
        parts.append(f"<b>Holding profile:</b> avg hold {hold_mean:,.0f}s, "
                     f"in line with the {h_alpha}s alpha horizon.")
    qtxt = " / ".join(f"{v:+,.0f}" for v in w["qnet"])
    parts.append(
        f"<b>Consistent, not one lucky burst:</b> net total by quarter of the "
        f"window: {qtxt} bps — positive in {w['consec']}/4 sub-periods.")
    parts.append(
        f"<b>Plateau, not peak:</b> the 3&times;3 neighborhood around this cell "
        f"averages {w['smooth']:+,.0f} bps net — adjacent thresholds also work, "
        f"so the choice is structural rather than an isolated best cell.")
    if qc != "base" and isinstance(qc, float) and qc > qo:
        parts.append(
            f"<b>Slow-exit variant (q_close {qc:g} &gt; q_open {qo:g}):</b> the book "
            f"holds through opposite signals weaker than the close level, so trips "
            f"ride longer swings; realized share {c['realized_share']*100:.0f}% of "
            f"opened units ({c['n_discard']} gap-discarded, {c['open_end']} still "
            f"open at end) — the selection required &ge;80%, but treat the maxDD "
            f"and holding profile as trend-riding risk, not quick alpha capture.")
    if c["pn"] > 1.5 or c["pn"] < 0.67:
        parts.append(
            f"<b>Caveat — side skew:</b> long/short trip ratio {c['pn']:.2f}; "
            f"part of this cell's PnL leans on one side of the market (check the "
            f"balanced-thresholds table before trusting it across regimes).")
    return "<div class='winwhy'>" + "<br>".join(parts) + "</div>"


MAX_TRIPS_SHOWN = 1500     # marker cap — an uncapped 5s/1s tenor made a 500MB page

def render_orders_fig(ts_a, logpx_a, w, label):
    """Price chart with every ORDER of one winner threshold pair: green
    triangle-up = BUY, red triangle-down = SELL (open + close legs, from the
    FIFO trip ledger: open at trip_ts - hold, side = trip side; close at
    trip_ts, opposite side). Trips beyond MAX_TRIPS_SHOWN are evenly
    subsampled; units still open at window end are not shown."""
    acct = w["acct"]
    tt = np.asarray(acct["trip_ts"], dtype=np.int64)
    sides = np.asarray(acct["trip_sides"], dtype=int)
    if not len(tt):
        return ""
    hold_ns = (np.asarray(w["hold"]) * 1e9).astype(np.int64)
    open_ts = tt - hold_ns
    n_all = len(tt)
    if n_all > MAX_TRIPS_SHOWN:
        pick = np.linspace(0, n_all - 1, MAX_TRIPS_SHOWN).astype(int)
        tt, sides, open_ts = tt[pick], sides[pick], open_ts[pick]
    ev_ts = np.concatenate([open_ts, tt])
    ev_side = np.concatenate([sides, -sides])
    j = np.clip(np.searchsorted(ts_a, ev_ts, side="right") - 1, 0, len(ts_a) - 1)
    ev_px = np.exp(logpx_a[j])
    ev_kind = np.r_[np.zeros(len(tt)), np.ones(len(tt))]
    g = np.linspace(ts_a[0], ts_a[-1], 3000).astype(np.int64)
    gj = np.clip(np.searchsorted(ts_a, g, side="right") - 1, 0, len(ts_a) - 1)
    fig = go.Figure()
    fig.add_scatter(x=pd.to_datetime(g), y=np.exp(logpx_a[gj]), mode="lines",
                    name="px", line=dict(color="#94a3b8", width=1),
                    hovertemplate="%{x}<br>px=%{y:.2f}<extra></extra>")
    for sd, nm, sym_m, colr in ((1, "BUY", "triangle-up", "#38a169"),
                                (-1, "SELL", "triangle-down", "#e53e3e")):
        m = ev_side == sd
        if not m.any():
            continue
        kind_txt = np.where(ev_kind[m] > 0, "close leg", "open leg")
        fig.add_scatter(x=pd.to_datetime(ev_ts[m]), y=ev_px[m], mode="markers",
                        name=f"{nm} ({int(m.sum()):,})",
                        marker=dict(symbol=sym_m, size=7, color=colr,
                                    line=dict(width=0.5, color="#1a202c")),
                        customdata=kind_txt,
                        hovertemplate=(f"{nm} " + "%{customdata}<br>%{x}<br>"
                                       "px=%{y:.2f}<extra></extra>"))
    fig.update_layout(
        title=(f"{label} — price & orders ("
               + (f"showing {len(tt):,} of {n_all:,} trips, evenly sampled; "
                  if n_all > MAX_TRIPS_SHOWN else f"{n_all:,} round trips; ")
               + f"{acct['open_end']} unit(s) open at window end not shown)"),
        xaxis_title="time (zoom for detail)", yaxis_title="price",
        template="plotly_white", height=420, margin=dict(t=60, b=40),
        legend=dict(orientation="h", y=-0.18))
    return _fig_html(fig, height="440px")


def build_winner_cards(ts, logpx, pred, cells, label, h_alpha):
    """Selection-note + one collapsible card per selected winner (max
    MAX_WINNERS), each with the WHY note + holding + position/cum-PnL panels."""
    winners = select_winners(cells, ts, logpx, pred)
    note = (
        "<div class='selnote'><b>Threshold selection logic</b> (applied to the "
        "grid above, winners below): "
        "1) <b>viability</b> — net/trade &ge; 0.5&times;cost (buffer against "
        "cost-model error), &ge;200 trips, positive net total; "
        "2) <b>plateau over peak</b> — cells ranked by the 3&times;3 "
        "neighborhood MEAN of net total, so an isolated lucky cell cannot win; "
        "3) <b>confidence</b> — per-trip net t-stat &ge; 2; "
        "4) <b>consistency</b> — positive net in &ge;3 of 4 sub-periods of the "
        "window; "
        "5) <b>realization-dominance</b> — &ge;80% of opened units must CLOSE as "
        "realized trips (cells whose PnL lives in unrealized mark-to-market or "
        "gap-discarded units are trend-riding, not alpha capture — a hazard of "
        "slow-exit cells where q_close &gt; q_open). Survivors ranked by plateau "
        "score, capped at "
        f"{MAX_WINNERS}. All numbers remain IN-WINDOW: confirm on the next "
        "window (walk-forward) before trading a selected pair.</div>")
    if not winners:
        return (note + "<div class='winwhy'><b>No cell met all four criteria"
                "</b> — nothing at this alpha/cost combination trades robustly "
                "net of cost. The grid above shows the closest calls; treat "
                "them as research leads, not tradeable settings.</div>")
    cards = []
    for rank, w in enumerate(winners, 1):
        c = w["cell"]
        qc_txt = "close = open" if w["qc"] == "base" else f"q_close {w['qc']:g}"
        rule_txt = (f"OPEN q{w['qo']:g} (T={w['t_open']:.2e}) / CLOSE "
                    + ("= open (opposite signal at T)" if w["qc"] == "base"
                       else f"q{w['qc']:g} (C={w['close_level']:.2e})"))
        summ = (f"#{rank} &mdash; <b>OPEN q_open {w['qo']:g} "
                f"(T={w['t_open']:.2e})</b> / <b>CLOSE {qc_txt}"
                + ("" if w["qc"] == "base" else f" (C={w['close_level']:.2e})")
                + "</b> &middot; "
                f"{c['trips']:,} trips &middot; net {c['net']:+.2f} bps/trade "
                f"&middot; net total {c['net_total']:+,.0f} bps &middot; "
                f"maxDD {c['maxdd']:+,.0f} bps &middot; t {w['tstat']:.1f} "
                f"&middot; {w['consec']}/4 quarters positive")
        body = (winner_reason(w, cells, h_alpha)
                + render_holding_fig(w["hold"], w["pnl"], label,
                                     w["t_open"], w["qo"], w["pn"],
                                     rule_txt=rule_txt)
                + render_position_fig(w["curves"], label,
                                      w["t_open"], w["qo"],
                                      rule_txt=rule_txt)
                + render_orders_fig(ts, logpx, w, label))
        cards.append(f'<details class="win"{" open" if rank == 1 else ""}>'
                     f'<summary>{summ}</summary>{body}</details>')
    return note + "".join(cards)


def render_holding_fig(hold, pnl, label, T, q, pn=(0, 0), rule_txt=None):
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
        title=((f"{label} — {rule_txt}: " if rule_txt else
                (f"{label} — ALL alpha, no threshold: " if q == 0 else
                 f"{label} — q{q:g} (T={T:.2e}): "))
               + f"{len(pnl):,} virtual round trips, "
               f"overall {overall:+.3f} bps/trade<br><sub>avg hold {avg_hold:,.0f}s "
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


def render_position_fig(curves, label, T, q, rule_txt=None):
    """Position + cumulative-PnL timeseries for one threshold: shows how much
    of the headline PnL is REALIZED vs still open (unrealized = equity gap),
    and the equity drawdown. A drop in equity without a matching move in
    realized is an open-inventory mark-to-market swing; units discarded at a
    data-gap flatten take their unrealized PnL with them (never realized)."""
    g_dt, pos, real, eq, maxdd, ddinfo, cost = curves   # maxdd is tick-resolution
    real_net = real - cost
    eq_net = eq - cost
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                        subplot_titles=("position (units, cap ±20)",
                                        "cumulative PnL (bps per unit)"))
    fig.add_scatter(x=g_dt, y=pos, mode="lines", name="position",
                    line=dict(color="#2b6cb0", width=1.2, shape="hv"),
                    fill="tozeroy", fillcolor="rgba(43,108,176,0.15)",
                    hovertemplate="%{x}<br>position=%{y:.0f}<extra></extra>",
                    row=1, col=1)
    fig.add_scatter(x=g_dt, y=eq, mode="lines",
                    name="total (realized + unrealized)",
                    line=dict(color="#d69e2e", width=1.6),
                    hovertemplate="%{x}<br>total=%{y:.2f} bps<extra></extra>",
                    row=2, col=1)
    fig.add_scatter(x=g_dt, y=real, mode="lines", name="realized",
                    line=dict(color="#38a169", width=1.6, shape="hv"),
                    hovertemplate="%{x}<br>realized=%{y:.2f} bps<extra></extra>",
                    row=2, col=1)
    fig.add_scatter(x=g_dt, y=eq_net, mode="lines",
                    name=f"total net of {COST_BPS}bp/trade",
                    line=dict(color="#dd6b20", width=1.4, dash="dash"),
                    hovertemplate="%{x}<br>total net=%{y:.2f} bps<extra></extra>",
                    row=2, col=1)
    fig.add_scatter(x=g_dt, y=real_net, mode="lines",
                    name=f"realized net of {COST_BPS}bp/trade",
                    line=dict(color="#e53e3e", width=1.4, dash="dot", shape="hv"),
                    hovertemplate="%{x}<br>realized net=%{y:.2f} bps<extra></extra>",
                    row=2, col=1)
    fig.add_hline(y=0, line_color="#999", row=2, col=1)
    fig.update_layout(
        title=((f"{label} — {rule_txt}: " if rule_txt else
                (f"{label} — ALL alpha, no threshold: " if q == 0 else
                 f"{label} — q{q:g} (T={T:.2e}): "))
               + f"position &amp; cumulative PnL "
               f"<br><sub>final: total {eq[-1]:+,.1f} bps, realized {real[-1]:+,.1f} bps "
               f"(gap = unrealized/discarded-at-gaps); "
               f"NET of {COST_BPS}bp/trade cost: total {eq_net[-1]:+,.1f}, "
               f"realized {real_net[-1]:+,.1f} bps &nbsp;|&nbsp; "
               f"maxDD (total, tick-resolution) {maxdd:+,.1f} bps "
               f"≈ {maxdd / MAX_INVENTORY:+,.1f} bp on max-cap equity (÷{MAX_INVENTORY})</sub>"
               + (f"<br><sub>maxDD window: {ddinfo['pk']} → {ddinfo['tr']} — "
                  f"px {ddinfo['px_pk']:,.2f} → {ddinfo['px_tr']:,.2f} "
                  f"({ddinfo['move_bp']:+,.0f} bp move) against position {ddinfo['pos_tr']:+d} "
                  f"at the trough; {ddinfo['real_chg']:+,.1f} bps realized inside the window</sub>"
                  if ddinfo else "")),
        template="plotly_white", margin=dict(t=138, b=50), height=580,
        title_y=0.97, title_yanchor="top",
        legend=dict(orientation="h", y=-0.14))
    fig.update_yaxes(title_text="units", row=1, col=1)
    fig.update_yaxes(title_text="cum PnL (bps)", row=2, col=1)
    for ann in fig.layout.annotations:
        ann.y = ann.y - 0.045
        ann.font.size = 13
    return _fig_html(fig, height="580px")


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
nav_entries = []            # (anchor_id, model label, tenor) for the floating index
PERIOD = ""                 # filled from the first loaded valid_pred (dates line)
for H_ALPHA in ALPHA_HORIZONS:
  for key, label, gen in MODELS:
    mdir = "lgbm" if key.startswith("lgbm") else key
    p = f"{STATS}/{SYM}/{mdir}/20260602/{gen}/valid_pred_ret_{H_ALPHA}.csv"
    if not os.path.isfile(p):
        print(f"skip {key} ret_{H_ALPHA}: no {p}")
        continue
    _yc = f"Y_hat_{H_ALPHA}"
    if not CALIB:                                    # uncalibrated -> prefer raw
        if f"Y_hat_{H_ALPHA}_raw" in pd.read_csv(p, nrows=0).columns:
            _yc = f"Y_hat_{H_ALPHA}_raw"
    df = pd.read_csv(p, usecols=["mark_ts", "AlphaPx", _yc])
    if _yc != f"Y_hat_{H_ALPHA}":                    # normalize to canonical name
        df = df.rename(columns={_yc: f"Y_hat_{H_ALPHA}"})
    df = df.sort_values("mark_ts").reset_index(drop=True)
    if not PERIOD:
        _f = lambda ns: pd.to_datetime(int(ns)).strftime("%Y-%m-%d %H:%M")
        PERIOD = (" &middot; <b>validation period</b> "
                  f"{_f(df.mark_ts.iloc[0])} &rarr; {_f(df.mark_ts.iloc[-1])}")
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
    grid_html, grid_cells = render_grid(ts_a, logpx_a, pred_a,
                                        f"{label} alpha ret_{H_ALPHA}")
    winners_html = build_winner_cards(ts_a, logpx_a, pred_a, grid_cells,
                                      f"{label} alpha ret_{H_ALPHA}", H_ALPHA)
    acct_rows = []
    bal_rows = []
    for q in QUANTILES[1:]:
        tl = float(np.quantile(pred_a, 1 - (1 - q) / 2))
        tsh = float(np.quantile(pred_a, (1 - q) / 2))
        bh, bp, bpn, bcur, bacct = round_trips(ts_a, logpx_a, pred_a, 0.0,
                                               t_long=tl, t_short=tsh)
        sides = bacct["trip_sides"]
        lm = float(np.mean(bp[sides > 0])) if (sides > 0).any() else float("nan")
        sm = float(np.mean(bp[sides < 0])) if (sides < 0).any() else float("nan")
        mean_bp = float(np.mean(bp)) if len(bp) else float("nan")
        bal_rows.append(
            f"<tr><td>q{q:g}</td><td>{tl:+.3e}</td><td>{tsh:+.3e}</td>"
            f"<td>{bacct['n_long']:,} / {bacct['n_short']:,}</td>"
            f"<td>{len(bp):,}</td><td>{mean_bp:+.3f}</td>"
            f"<td>{mean_bp - COST_BPS:+.3f}</td>"
            f"<td>{lm:+.3f}</td><td>{sm:+.3f}</td>"
            f"<td>{bcur[3][-1]:+,.0f}</td><td>{bcur[4]:+,.0f}</td></tr>")
    for q in QUANTILES:
        T = curves[q][2]
        hold, pnl, pn, pcurves, acct = round_trips(ts_a, logpx_a, pred_a, T)
        n_sig = acct["n_long"] + acct["n_short"]
        runs = acct["n_flips"] + 1
        consumed = 2 * acct["trips"] + acct["cap_skips"] + acct["gap_discards"] + acct["open_end"]
        acct_rows.append(
            f"<tr><td>{'ALL (no filter)' if q == 0 else f'q{q:g}'}</td>"
            f"<td>{n_sig:,}</td><td>{acct['n_long']:,} / {acct['n_short']:,}</td>"
            f"<td>{acct['n_flips']:,}</td><td>{n_sig / max(runs, 1):.1f}</td>"
            f"<td>{acct['trips']:,}</td><td>{2 * acct['trips']:,}</td>"
            f"<td>{acct['cap_skips']:,}</td><td>{acct['gap_discards']:,}</td>"
            f"<td>{acct['open_end']}</td>"
            f"<td>{'&#10003;' if consumed == n_sig else f'{consumed:,} MISMATCH'}</td></tr>")
        # per-quantile cards replaced by the SELECTED WINNERS section
        print(f"  [{key} ret_{H_ALPHA}] holding q{q:g}: {len(pnl):,} round trips, "
              f"overall {np.nanmean(pnl):+.3f}bp, avg hold {np.mean(hold):.0f}s "
              f"(median {np.median(hold):.0f}s), P/N {pn[0]/max(pn[1],1):.3f}")
    anchor = f"sec-{key}-ret{H_ALPHA}"
    nav_entries.append((anchor, label, H_ALPHA))
    sections.append(f"""
<section id="{anchor}">
  <h2>{label} — alpha ret_{H_ALPHA}</h2>
  {fig_div}
  <table class="cmp">
   <tr><th>threshold</th><th>T (|pred|)</th><th>virtual trades</th>
       <th>markout @alpha hzn (bps)</th><th>peak (bps)</th><th>@300s (bps)</th><th>px coverage</th></tr>
   {rows}
  </table>
  <details class="acct"><summary>Signals &rarr; round-trips accounting (why trips &laquo; signals):
   each trip consumes TWO signals (open + close); same-side signal RUNS stack units only to the
   &plusmn;{MAX_INVENTORY} cap (rest skipped); trips are governed by sign FLIPS, not signal counts</summary>
   <table class="cmp">
    <tr><th>threshold</th><th>signals</th><th>long / short</th><th>sign flips</th>
        <th>mean run</th><th>round trips</th><th>&times;2 consumed</th>
        <th>cap-skips</th><th>gap-discards</th><th>open at end</th><th>sums</th></tr>
    {''.join(acct_rows)}
   </table>
  </details>
  <details class="acct"><summary>BALANCED per-side thresholds (P/N forced to 1):
   a single |pred| threshold turns a few percent of tail asymmetry into a 2&ndash;4&times;
   long/short count imbalance (rally-beta loading); here T_long / T_short are separate
   per-side quantiles admitting the SAME row fraction each way &mdash; the short-side
   bps column shows whether genuine short alpha exists</summary>
   <table class="cmp">
    <tr><th>threshold</th><th>T_long</th><th>T_short</th><th>signals L/S</th>
        <th>round trips</th><th>bps/trade</th><th>net of {COST_BPS}bp</th>
        <th>long-side bps</th><th>short-side bps</th><th>final (bps)</th><th>maxDD (bps)</th></tr>
    {''.join(bal_rows)}
   </table>
  </details>
  {grid_html}
  <details class="hold" open><summary>SELECTED WINNERS &mdash; open/close threshold pairs
   passing the four selection criteria (viability &middot; plateau-over-peak &middot;
   t&ge;2 &middot; 3/4-quarter consistency), capped at 10 &mdash; each card carries the
   WHY, holding buckets and position/cum-PnL panels (UNIT-SIZE FIFO trades, cap
   &plusmn;20, gap-flatten &gt; 60s)</summary>
   {winners_html}
  </details>
</section>""")
    print(f"{key}: done ({len(df):,} rows)")

# ---- floating model/tenor index (fixed top-right) --------------------------
nav_groups = {}
for anchor, label, h in nav_entries:
    nav_groups.setdefault(h, []).append((anchor, label))
nav_rows = "".join(
    '<div class="fnav-g"><span class="fnav-h">ret_' + str(h) + '</span>'
    + "".join(f'<a href="#{a}">{l}</a>' for a, l in rows) + '</div>'
    for h, rows in nav_groups.items())
fnav = (f'<details class="fnav" open><summary>index</summary>{nav_rows}</details>'
        if nav_entries else "")

body = f"""{fnav}<header>
  <a class="back-link" href="index.html">&larr; All reports</a>
  <h1>{SYM.upper()}{f" {VER}" if VER else ""}{" (calibrated)" if CALIB else ""} &mdash; alpha decay: virtual-trade markout by conviction threshold</h1>
  <div class="meta">VIRTUAL trades (research twin of the production fill-markout plot):
   entry at every validation row with Y_hat &gt; +T (long) / &lt; &minus;T (short),
   T = |pred| quantile from the conviction table&rsquo;s highlighted tail buckets &middot;
   markout logic PROD-ALIGNED (order_multi.R): 1s AlphaPx grid + LOCF, staleness tolerance 2s, simple returns, coverage reported &middot; alpha horizons: {", ".join(f"ret_{h}" for h in ALPHA_HORIZONS)} &middot;
   overlapping entries, unweighted mean (no qty for virtual trades) &middot;
   validation set only (last 20%){PERIOD}</div>
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
html{scroll-behavior:smooth}
section{scroll-margin-top:12px}
details.fnav{position:fixed;top:12px;right:12px;z-index:1000;background:#ffffff;
border:1px solid #cbd5e0;border-radius:10px;box-shadow:0 2px 10px rgba(15,23,42,.12);
padding:6px 10px;max-width:230px;font-size:12px}
details.fnav>summary{cursor:pointer;font-weight:700;color:#2b6cb0;list-style:none}
details.fnav>summary::before{content:"\2630  ";font-weight:400}
details.fnav .fnav-g{margin-top:6px;padding-top:6px;border-top:1px solid #edf2f7}
details.fnav .fnav-h{display:block;color:#718096;font-weight:700;font-size:11px;
text-transform:uppercase;letter-spacing:.4px;margin-bottom:2px}
details.fnav a{display:block;color:#2d3748;text-decoration:none;padding:1.5px 4px;
border-radius:5px;line-height:1.45}
details.fnav a:hover{background:#ebf8ff;color:#2b6cb0}
details.acct{margin-top:12px}
details.acct>summary{cursor:pointer;color:#2b6cb0;font-size:13px;max-width:960px;line-height:1.5}
div.scrollx{overflow-x:auto;margin-top:10px}
table.grid{border-collapse:separate;border-spacing:3px;font-size:12.5px}
table.grid th{position:sticky;top:0;background:#edf2f7;color:#4a5568;font-weight:600;
font-size:11.5px;padding:6px 10px;border-radius:6px;white-space:nowrap}
table.grid td{text-align:center;min-width:104px;padding:6px 9px;border-radius:7px;
line-height:1.3;white-space:nowrap}
table.grid td:first-child{background:#f7fafc;font-weight:600;color:#2d3748;
text-align:right;font-size:11.5px}
table.grid td:hover{outline:2px solid #2b6cb0;outline-offset:-1px;cursor:default}
table.grid b{font-size:13px;letter-spacing:.2px}
table.grid .gsub{display:block;font-size:10px;color:#4a5568;margin-top:1px}
table.grid .gm{font-size:12px;margin-left:3px}
table.grid .gm1{color:#d69e2e}
table.grid .gm2{color:#2b6cb0}
table.grid .gm3{color:#805ad5}
details.win summary{cursor:pointer;padding:8px 14px;background:#f0fff4;
border:1px solid #9ae6b4;border-radius:8px;font-size:13px;color:#22543d;
font-weight:600;list-style:none;margin-top:8px}
details.win[open]>summary{border-radius:8px 8px 0 0}
div.winwhy{background:#fffbeb;border:1px solid #f6e05e;border-radius:6px;
padding:10px 14px;margin:10px 0;font-size:12.5px;line-height:1.6;color:#2d3748;
max-width:1100px}
div.selnote{background:#ebf8ff;border:1px solid #90cdf4;border-radius:6px;
padding:10px 14px;margin:8px 0;font-size:12.5px;line-height:1.6;color:#2c5282;
max-width:1100px}
details.thr{margin:8px 0;border:1px solid #e2e8f0;border-radius:8px}
details.thr>summary{cursor:pointer;padding:8px 14px;background:#f7fafc;border-radius:8px;
font-size:13px;color:#2d3748;font-weight:600;list-style:none}
details.thr>summary::before{content:"\25B8  ";color:#2b6cb0}
details.thr[open]>summary::before{content:"\25BE  ";color:#2b6cb0}
details.thr[open]>summary{border-bottom:1px solid #e2e8f0;border-radius:8px 8px 0 0}
details.thr>summary:hover{background:#edf2f7}
"""

plotly_js = ('<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
             "<script>document.addEventListener('DOMContentLoaded',function(){"
             "document.querySelectorAll('details').forEach(function(d){"
             "d.addEventListener('toggle',function(){if(d.open){"
             "d.querySelectorAll('.plotly-graph-div').forEach(function(g){"
             "try{Plotly.Plots.resize(g)}catch(e){}})}})})});</script>")
page = ("<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{SYM.upper()}{f' {VER}' if VER else ''} alpha decay</title>{plotly_js}<style>{CSS}</style></head>"
        f"<body>{body}</body></html>")
# version (v2/v3 = which dump) and calibration are INDEPENDENT axes: VER -> page
# prefix directly; the `cal` flag -> calibrated Y_hat + `_calibrated` suffix.
_PV, _SFX = (VER or "v1"), ("_calibrated" if CALIB else "")
OUT = os.path.join(HERE, f"{_PV}_{SYM}_alpha_decay{_SFX}.html")
open(OUT, "w").write(page)
print(f"wrote {OUT} ({len(page)//1024} KB)")
