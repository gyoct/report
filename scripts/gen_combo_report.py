#!/usr/bin/env python3
"""Signal-combination report page: <sym>_signal_combo.html.

Renders the output of pipeline/combine_signals.py (combine_report.json) and
answers the two questions with the measured numbers next to their baselines:

  1. IS MODEL ENSEMBLING WORTH IT?   per-horizon table: pairwise prediction
     correlation, effective breadth, best single model IC vs equal / IC-weighted
     ensemble IC — verdict + reason generated from the data.
  2. HORIZON COMBINATION — WHICH METHOD, WHAT EFFECT?  per target horizon:
     own-horizon single-model baseline vs equal / ic_weight / shrunk_mvo /
     ridge_stack (all fitted on slice A, scored on slice B), with % gain and
     the shipped method's weights.

  3. COMBO vs SINGLE — ALPHA DECAY.  Prod-aligned virtual-trade markout
     (same formulas as gen_alpha_decay.py: 1 s AlphaPx grid + LOCF, 2 s
     staleness tolerance, simple returns) overlaying the combined signal on
     the reference single model at q0.99 and the q=0 no-filter baseline,
     one panel per horizon, plus the summary table.

Prerequisites: statistics/<sym>/combo/20260602/pipeline_combo/ populated by
`python3 pipeline/combine_signals.py <sym>` (combine_report.json +
valid_pred_ret_<h>.csv) and the reference model's valid_pred files.

    python3 gen_combo_report.py [btc]
Then make_index.py to encrypt + publish.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYM = (sys.argv[1] if len(sys.argv) > 1 else "btc").lower()
STATS = "/home/guanyang/work/alpha_replayer_config/statistics"
COMBO_DIR = f"{STATS}/{SYM}/combo/20260602/pipeline_combo"
HORIZONS_S = [1, 10, 30, 60]
MARKOUT_H = np.arange(1, 301)
NS = 1_000_000_000
TOLERANCE_SEC = 2
Q_SHOW = [0.99, 0.995, 0.999]      # conviction thresholds for the decay overlay
Q_DASH = {0.99: "solid", 0.995: "dash", 0.999: "dashdot", 0.0: "dot"}
METHOD_LABELS = {"equal": "equal", "ic_weight": "IC-weight",
                 "shrunk_mvo": "shrunk MVO", "ridge_stack": "ridge stack"}
MODEL_LABELS = {"lasso_ols": "Linear (LASSO→WLS)", "lgbm": "LightGBM",
                "lgbm_tuned": "LGBM tuned (per-horizon)", "mlp": "MLP (torch)"}

report = json.load(open(f"{COMBO_DIR}/combine_report.json"))
try:
    _m = pd.read_csv(f"{COMBO_DIR}/valid_pred_ret_10.csv", usecols=["mark_ts"])["mark_ts"]
    _f = lambda ns: pd.to_datetime(int(ns)).strftime("%Y-%m-%d %H:%M")
    PERIOD = f" <b>Validation period</b> {_f(_m.min())} &rarr; {_f(_m.max())}."
except Exception:
    PERIOD = ""
REF = report["ref_model"]
REF_LABEL = MODEL_LABELS.get(REF, REF)
REF_MDIR = "lgbm" if REF.startswith("lgbm") else REF
REF_GEN = "pipeline_tuned_all" if REF == "lgbm_tuned" else "pipeline_sqrtw"


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if x is not None and np.isfinite(x) else "—"


def pct(new, base):
    if not (np.isfinite(new) and np.isfinite(base)) or base == 0:
        return "—"
    return f"{(new / base - 1) * 100:+.1f}%"


# ---------------------------------------------------------------- section 1
ens_rows, gains = [], []
for h in HORIZONS_S:
    e = report["ensemble"].get(str(h))
    if not e:
        continue
    gains.append(e["gain_vs_best"])
    combo_best = max(e["equal_ic"], e["icw_ic"])
    ens_rows.append(
        f"<tr><td>ret_{h}</td><td>{e['n_models']}</td>"
        f"<td>{e['corr_mean']:.3f} ({e['corr_min']:.3f}–{e['corr_max']:.3f})</td>"
        f"<td>{e['n_effective']:.2f}</td>"
        f"<td>{MODEL_LABELS.get(e['best'].split('__')[0], e['best'])} "
        f"({fmt(e['best_ic'])})</td>"
        f"<td>{fmt(e['equal_ic'])}</td><td>{fmt(e['icw_ic'])}</td>"
        f"<td>{fmt(e['gain_vs_best'])} ({pct(combo_best, e['best_ic'])})</td></tr>")

max_gain = max(gains) if gains else float("nan")
corr_means = [report["ensemble"][str(h)]["corr_mean"]
              for h in HORIZONS_S if str(h) in report["ensemble"]]
neffs = [report["ensemble"][str(h)]["n_effective"]
         for h in HORIZONS_S if str(h) in report["ensemble"]]
worth_it = max_gain >= 0.005
verdict = ("WORTH IT" if worth_it else "NOT WORTH IT")
verdict_cls = "good" if worth_it else "bad"
reason = (
    f"Mean pairwise prediction correlation is "
    f"{min(corr_means):.2f}–{max(corr_means):.2f}, so the "
    f"{report['ensemble'][str(HORIZONS_S[0])]['n_models']} models carry only "
    f"{min(neffs):.2f}–{max(neffs):.2f} effective independent signals — they are "
    f"re-reading the same features. The best ensemble beats the best single "
    f"model by at most {max_gain:+.4f} IC "
    f"({max(pct(max(report['ensemble'][str(h)]['equal_ic'], report['ensemble'][str(h)]['icw_ic']), report['ensemble'][str(h)]['best_ic']) for h in HORIZONS_S if str(h) in report['ensemble'])} "
    f"at best), which does not pay for running and monitoring "
    f"{report['ensemble'][str(HORIZONS_S[0])]['n_models']}× the models in "
    f"production. Pick the best single model per horizon and spend the "
    f"complexity budget on horizon combination instead."
    if not worth_it else
    f"The best ensemble beats the best single model by {max_gain:+.4f} IC — "
    f"large enough to justify running multiple models.")

sec1 = f"""
<section>
 <h2>1. Is MODEL ensembling worth it? <span class="badge {verdict_cls}">{verdict}</span></h2>
 <div class="note">All ICs are SLICE-B (second half of the validation window);
  ensemble weights fitted on slice A only. n_eff = N / (1 + (N−1)·ρ̄) =
  effective number of independent signals.</div>
 <table class="cmp">
  <tr><th>horizon</th><th>models</th><th>pred corr mean (min–max)</th><th>n_eff</th>
      <th>best single (IC)</th><th>equal-weight IC</th><th>IC-weight IC</th>
      <th>gain vs best</th></tr>
  {''.join(ens_rows)}
 </table>
 <div class="reason"><b>Reason:</b> {reason}</div>
</section>"""

# ---------------------------------------------------------------- section 2
hz_rows, w_rows = [], []
for h in HORIZONS_S:
    z = report["horizon"].get(str(h))
    best = report["best_method"].get(str(h), {})
    if not z:
        continue
    single = z["single"]
    bm = best.get("method")
    cells = "".join(
        f"<td class=\"{'win' if m == bm else ''}\">"
        f"{fmt(z[m])} ({pct(z[m], single)})</td>"
        for m in ("equal", "ic_weight", "shrunk_mvo", "ridge_stack"))
    hz_rows.append(
        f"<tr><td>ret_{h}</td><td>{fmt(single)}</td>{cells}"
        f"<td>{METHOD_LABELS.get(bm, '—')}</td></tr>")
    w = best.get("weights", {})
    if w:
        w_rows.append(f"<tr><td>ret_{h}</td><td>{METHOD_LABELS.get(bm)}</td>"
                      + "".join(f"<td>{w.get(f'h{s}', 0):+.3f}</td>"
                                for s in HORIZONS_S) + "</tr>")

sec2 = f"""
<section>
 <h2>2. HORIZON combination — method × target horizon</h2>
 <div class="note">Reference model: <b>{REF_LABEL}</b>. Its four horizon signals
  (z-scored with slice-A stats) are combined to predict EACH target; weights
  fitted on slice A, IC scored on slice B against the own-horizon single model.
  Shrunk MVO = inv(shrunk corr)·IC (Grinold–Kahn); ridge stack = ridge
  regression of the target on the four z-scores.</div>
 <table class="cmp">
  <tr><th>target</th><th>single (own-horizon)</th><th>equal</th><th>IC-weight</th>
      <th>shrunk MVO</th><th>ridge stack</th><th>best</th></tr>
  {''.join(hz_rows)}
 </table>
 <h3>Shipped weights (best method per target, applied to z-scored signals)</h3>
 <table class="cmp">
  <tr><th>target</th><th>method</th>{''.join(f'<th>w(h{s})</th>' for s in HORIZONS_S)}</tr>
  {''.join(w_rows)}
 </table>
 <div class="note">The shipped combo (per-horizon <code>valid_pred_ret_&lt;h&gt;.csv</code>
  under <code>combo/20260602/pipeline_combo</code>) is rescaled by FR = IC·σ(ret) so
  Y_hat is in return units, then consumed by every downstream tool unchanged —
  including the <a href="{SYM}_alpha_decay.html">alpha-decay page</a>, where
  &ldquo;Combined (horizon stack)&rdquo; now appears beside the single models.</div>
</section>"""

# ---------------------------------------------------------------- section 3
DARK_LINES = {"combo": "#f59e0b", "single": "#818cf8"}


def markout_curves(df, h_alpha, quantiles):
    """PROD-ALIGNED markout (copy of gen_alpha_decay.markout_curves)."""
    ts = df["mark_ts"].to_numpy(np.int64)
    px = df["AlphaPx"].to_numpy(np.float64)
    pred = df[f"Y_hat_{h_alpha}"].to_numpy(np.float64)
    sec = ts // NS
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
    out = {}
    for q in quantiles:
        T = float(np.quantile(np.abs(pred), q))
        side = np.where(pred > T, 1.0, np.where(pred < -T, -1.0, 0.0))
        eidx = np.flatnonzero(side != 0.0)
        e_sec = sec[eidx] - g0
        e_px = px[eidx]
        e_side = side[eidx]
        curve = np.full(len(MARKOUT_H), np.nan)
        for j, h in enumerate(MARKOUT_H):
            t_sec = e_sec + h
            ok = t_sec <= g1 - g0
            tp = grid_px[t_sec[ok]]
            fresh = grid_age[t_sec[ok]] <= TOLERANCE_SEC
            good = fresh & ~np.isnan(tp)
            if good.any():
                curve[j] = np.mean(e_side[ok][good]
                                   * (tp[good] / e_px[ok][good] - 1.0)) * 1e4
        out[q] = (int(len(eidx)), curve, T)
    return out


decay_panels, decay_rows = [], []
for h in HORIZONS_S:
    p_combo = f"{COMBO_DIR}/valid_pred_ret_{h}.csv"
    p_ref = f"{STATS}/{SYM}/{REF_MDIR}/20260602/{REF_GEN}/valid_pred_ret_{h}.csv"
    if not (os.path.isfile(p_combo) and os.path.isfile(p_ref)):
        print(f"skip decay ret_{h}: missing input")
        continue
    fig = go.Figure()
    panel_stats = {}
    for tag, path, label in (("combo", p_combo, "Combined (horizon stack)"),
                             ("single", p_ref, REF_LABEL)):
        df = pd.read_csv(path, usecols=["mark_ts", "AlphaPx", f"Y_hat_{h}"]
                         ).sort_values("mark_ts").reset_index(drop=True)
        curves = markout_curves(df, h, [0.0] + Q_SHOW)
        # q0.99 entry set (signed) for the entry-overlap diagnostic below
        pv = df[f"Y_hat_{h}"].to_numpy(np.float64)
        T99 = float(np.quantile(np.abs(pv), 0.99))
        m_ts = df["mark_ts"].to_numpy(np.int64) // NS
        panel_stats[tag] = {
            "label": label, "curves": curves,
            "long": set(m_ts[pv > T99]), "short": set(m_ts[pv < -T99])}
        for q in Q_SHOW + [0.0]:
            n, c, T = curves[q]
            nm = f"{label} {'q' + str(q) if q else 'ALL (no filter)'}"
            if q and (n == 0 or np.all(np.isnan(c))):
                # degenerate signal (near-constant preds — e.g. xag lgbm_tuned
                # ret_60): the strict tail threshold admits no entries
                decay_rows.append(
                    f"<tr><td>ret_{h}</td><td>{label}</td><td>q{q:g}</td>"
                    f"<td>{T:.3e}</td><td>{n:,}</td><td colspan=3>no entries "
                    f"above threshold (near-constant predictions)</td></tr>")
                print(f"  [ret_{h}] {tag}: q{q:g} DEGENERATE (n={n}) — skipped")
                continue
            fig.add_trace(go.Scatter(
                x=MARKOUT_H, y=c, mode="lines", name=nm,
                line=dict(color=DARK_LINES[tag], width=2 if q else 1,
                          dash=Q_DASH[q]),
                hovertemplate=nm + " · %{x}s: %{y:+.3f}bp<extra></extra>"))
            if q:
                pk = int(np.nanargmax(c))
                decay_rows.append(
                    f"<tr><td>ret_{h}</td><td>{label}</td><td>q{q:g}</td>"
                    f"<td>{T:.3e}</td><td>{n:,}</td>"
                    f"<td>{c[h - 1]:+.3f}</td>"
                    f"<td>{np.nanmax(c):+.3f} @ {MARKOUT_H[pk]}s</td>"
                    f"<td>{c[-1]:+.3f}</td></tr>")
                print(f"  [ret_{h}] {tag}: q{q:g} n={n:,} @{h}s {c[h-1]:+.3f}bp "
                      f"peak {np.nanmax(c):+.3f}bp@{MARKOUT_H[pk]}s "
                      f"@300s {c[-1]:+.3f}bp")
    fig.update_layout(
        title=f"Virtual-trade markout — combo vs {REF_LABEL} — alpha ret_{h} "
              f"(q0.99 solid / q0.995 dash / q0.999 dash-dot, dotted no-filter)",
        xaxis_title="seconds after fill (mark_ts + h)", yaxis_title="markout (bps)",
        template="plotly_white", margin=dict(t=60), height=420,
        legend=dict(orientation="h", y=-0.18))
    panel_html = to_html(fig, include_plotlyjs=False, full_html=False,
                         default_width="100%", default_height="420px")
    # ---- WHY-IS-THE-COMBO-WORSE diagnostic (rendered below the plot) -------
    # Triggered whenever the single model beats the combo on any q0.99 summary
    # stat (markout @alpha horizon, peak, or @300s). Reasons are MEASURED, not
    # asserted: entry-set overlap + the shipped weights' horizon loading.
    cc = panel_stats.get("combo", {}).get("curves", {}).get(0.99)
    sc = panel_stats.get("single", {}).get("curves", {}).get(0.99)
    if cc and sc and cc[0] > 0 and sc[0] > 0 and not np.all(np.isnan(sc[1])):
        c_at, s_at = cc[1][h - 1], sc[1][h - 1]
        c_pk, s_pk = np.nanmax(cc[1]), np.nanmax(sc[1])
        c_end, s_end = cc[1][-1], sc[1][-1]
        if (s_at > c_at) or (s_pk > c_pk) or (s_end > c_end):
            inter = (len(panel_stats["combo"]["long"] & panel_stats["single"]["long"])
                     + len(panel_stats["combo"]["short"] & panel_stats["single"]["short"]))
            union = (len(panel_stats["combo"]["long"] | panel_stats["single"]["long"])
                     + len(panel_stats["combo"]["short"] | panel_stats["single"]["short"]))
            overlap = inter / max(union, 1)
            w = report["best_method"].get(str(h), {}).get("weights", {})
            wa = {k: abs(v) for k, v in w.items()}
            short_share = ((wa.get("h1", 0) + wa.get("h10", 0))
                           / max(sum(wa.values()), 1e-12)) * 100
            lost = []
            if s_at > c_at:
                lost.append(f"at the alpha horizon ({c_at:+.2f} vs {s_at:+.2f} bp)")
            if s_pk > c_pk:
                lost.append(f"at peak ({c_pk:+.2f} vs {s_pk:+.2f} bp)")
            if s_end > c_end:
                lost.append(f"at 300 s ({c_end:+.2f} vs {s_end:+.2f} bp)")
            comment = (
                f"<b>Why the combo trails the single model {' and '.join(lost)} "
                f"(q0.99 tail):</b><ol>"
                f"<li><b>The tail trades are DIFFERENT trades, not the same trades "
                f"re-ranked</b> — the two q0.99 entry sets overlap only "
                f"{overlap * 100:.0f}% (same-side, same-second). The combo's IC gain "
                f"is measured across the WHOLE distribution; a better overall "
                f"ordering does not imply a better top-1% selection.</li>"
                f"<li><b>The combo's tail conviction is short-horizon conviction</b> "
                f"— {short_share:.0f}% of the shipped |weights| sit on h1+h10 "
                f"({', '.join(f'{k} {v:+.2f}' for k, v in w.items())}), so its extreme "
                f"entries fire on fast signals whose edge is realized (and decays) "
                f"within ~a minute, while the single ret_{h} model's tail entries are "
                f"slower and keep accruing toward 300 s.</li>"
                f"<li><b>Averaging correlated signals shrinks the tail</b> — the "
                f"z-score blend pulls extreme predictions toward the consensus, so "
                f"the combo's q0.99 threshold admits less-extreme own-horizon "
                f"conviction (diluted selectivity where only the tail is traded).</li>"
                f"<li><b>Practical read:</b> use the combo for full-distribution "
                f"sizing (where its IC gain lives) and keep the own-horizon single "
                f"signal as the extreme-tail trigger for slow-horizon trades — or "
                f"fit the stack on tail-weighted loss if tail selection is the "
                f"goal.</li></ol>")
            panel_html += f'<div class="reason">{comment}</div>'
            print(f"  [ret_{h}] combo-trails comment added "
                  f"(overlap {overlap*100:.0f}%, short-share {short_share:.0f}%)")
    decay_panels.append(panel_html)

sec3 = f"""
<section>
 <h2>3. Combo vs single — alpha decay (prod-aligned virtual-trade markout)</h2>
 <div class="note">Same formulas as the <a href="{SYM}_alpha_decay.html">alpha-decay
  page</a> (order_multi.R semantics: 1 s AlphaPx grid + LOCF, staleness tolerance
  2 s, simple returns). Conviction thresholds q0.99 (solid) / q0.995 (dash) /
  q0.999 (dash-dot); dotted = every row trades. CAVEAT: combo weights are fitted
  on the first half of the validation window, so that half is mildly in-sample
  for the combo lines (4 fitted weights).</div>
 {''.join(decay_panels)}
 <table class="cmp">
  <tr><th>alpha</th><th>signal</th><th>threshold</th><th>T (|pred|)</th><th>virtual trades</th>
      <th>markout @alpha hzn (bps)</th><th>peak (bps)</th><th>@300s (bps)</th></tr>
  {''.join(decay_rows)}
 </table>
</section>"""

# ---------------------------------------------------------------- page
CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;
background:#ffffff;color:#1a202c}
header{background:#f7fafc;border-bottom:1px solid #e2e8f0;padding:18px 28px}
header h1{margin:6px 0 0;font-size:19px}
header .meta{color:#4a5568;font-size:12px;margin-top:6px;max-width:960px;line-height:1.5}
header .back-link{color:#2b6cb0;font-size:13px;text-decoration:none}
section{padding:14px 28px;max-width:1280px}
h2{font-size:15px;color:#2d3748;margin:16px 0 8px}
h3{font-size:13.5px;color:#2d3748;margin:14px 0 4px}
table.cmp{border-collapse:collapse;font-size:12.5px;margin-top:10px}
table.cmp th,table.cmp td{border:1px solid #cbd5e0;padding:5px 12px;text-align:right}
table.cmp th{background:#edf2f7;color:#4a5568;font-weight:600}
table.cmp td.win{background:#f0fff4;font-weight:700}
.badge{font-size:12px;padding:2px 10px;border-radius:10px;vertical-align:2px}
.badge.good{background:#c6f6d5;color:#22543d}
.badge.bad{background:#fed7d7;color:#822727}
.note{color:#4a5568;font-size:12px;max-width:960px;line-height:1.5;margin-top:4px}
.reason{color:#2d3748;font-size:12.5px;max-width:960px;line-height:1.6;margin-top:10px;
background:#fffbeb;border:1px solid #f6e05e;border-radius:6px;padding:10px 14px}
"""

n_rows = report["n_rows"]
body = f"""<header>
 <a class="back-link" href="index.html">&larr; All reports</a>
 <h1>{SYM.upper()} &mdash; signal combination: models &times; horizons &rarr; one trade signal</h1>
 <div class="meta">Built by <code>pipeline/combine_signals.py</code> (methodology
  steps in its docstring). Panel = {n_rows:,} validation rows, all four models
  &times; four horizons joined on mark_ts. NESTED evaluation: slice A = first
  {report['n_fit']:,} rows (fit z-stats / ICs / covariances / stacker betas),
  slice B = last {report['n_score']:,} rows (every IC reported).{PERIOD} Shipped combo:
  best method per target horizon, FR-rescaled to return units, written in
  feature-dump format under <code>combo/20260602/pipeline_combo</code>.</div>
</header>
{sec1}{sec2}{sec3}"""

plotly_js = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
page = ("<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{SYM.upper()} signal combination</title>{plotly_js}"
        f"<style>{CSS}</style></head><body>{body}</body></html>")
OUT = os.path.join(HERE, f"v1_{SYM}_signal_combo.html")
open(OUT, "w").write(page)
print(f"wrote {OUT} ({len(page)//1024} KB)")
