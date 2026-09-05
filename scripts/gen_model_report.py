#!/usr/bin/env python3
"""Render a per-symbol MODEL report (Linear/OLS+Lasso vs LGBM) into a dark HTML page.

Usage: python gen_model_report.py <symbol>   e.g. btc, eth, sol
Reads  ../alpha_replayer_config/statistics/<symbol>/{linear,lgbm}/...
Writes ./<symbol>_model.html   (self-contained; scatter PNGs embedded as data URIs)
Then run make_index.py to (re)encrypt + publish.
"""
import base64, csv, html, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYM = (sys.argv[1] if len(sys.argv) > 1 else "btc").lower()
ROOT = os.path.join(HERE, "..", "alpha_replayer_config", "statistics", SYM)
LIN, LGB = os.path.join(ROOT, "linear"), os.path.join(ROOT, "lgbm")
HORIZONS = [1, 10, 30, 60]
if not os.path.isdir(ROOT):
    sys.exit(f"no statistics dir for {SYM!r}: {ROOT}")


def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    # LGBM summaries pad headers/values with spaces -> strip keys and values
    return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in rows]


def f(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "&mdash;"


def bps(x, nd=3):
    try:
        return f"{float(x)*1e4:.{nd}f}"
    except (TypeError, ValueError):
        return "&mdash;"


def sci(x, nd=2):
    try:
        return f"{float(x):.{nd}e}"
    except (TypeError, ValueError):
        return "&mdash;"


def data_uri(path):
    if not os.path.isfile(path):
        return ""
    b = base64.b64encode(open(path, "rb").read()).decode()
    return f"data:image/png;base64,{b}"


# ---- headline: one row per (horizon, model) ---------------------------------
lin = {r["prediction_horizon"].strip(): r for r in read_csv(os.path.join(LIN, "linear_horizon_summary.csv"))}
lgb = {r["prediction_horizon"].strip(): r for r in read_csv(os.path.join(LGB, "lgbm_horizon_summary.csv"))}

def headline_rows():
    out = []
    for h in HORIZONS:
        hs = str(h)
        L = lin.get(hs)
        if L:
            out.append(("Linear/OLS", h, L["in_sample_R2"], L["validation_R2"], L["validation_IC"],
                        L["validation_IR"], L["validation_RMSE"], L["beta"], ""))
            if L.get("lasso_validation_IC"):
                out.append(("Lasso", h, L["lasso_in_sample_R2"], L["lasso_validation_R2"],
                            L["lasso_validation_IC"], "", "", "", ""))
        G = lgb.get(hs)
        if G:
            out.append(("LGBM", h, G["in_sample_R2"], G["validation_R2"], G["validation_IC"],
                        G["validation_IR"], G["validation_RMSE"], G["beta"], G.get("model_best_iter", "")))
    return out

hrows = "".join(
    f'<tr class="m-{m.split("/")[0].lower()}"><td>{m}</td><td>{h}s</td>'
    f'<td>{f(isr)}</td><td>{f(vr)}</td><td class="ic">{f(ic)}</td>'
    f'<td>{f(ir)}</td><td>{bps(rmse)}</td><td>{f(beta,3)}</td><td>{bi or "&mdash;"}</td></tr>'
    for (m, h, isr, vr, ic, ir, rmse, beta, bi) in headline_rows())

headline = f'''<table class="tbl">
<thead><tr><th>Model</th><th>Horizon</th><th>IS R&sup2;</th><th>Val R&sup2;</th>
<th>Val IC</th><th>Val IR</th><th>RMSE (bps)</th><th>&beta;</th><th>LGBM iter</th></tr></thead>
<tbody>{hrows}</tbody></table>'''


# ---- full transposed per-horizon summary (all ~40 metrics) ------------------
def gfmt(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return html.escape(v) if v else "&mdash;"
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e5:
        return f"{x:.4g}"
    if x == int(x):
        return f"{int(x):,}"
    return f"{x:.4g}"


def full_summary(rows_by_h, model):
    hs = [str(h) for h in HORIZONS if str(h) in rows_by_h]
    if not hs:
        return ""
    metrics = [k for k in rows_by_h[hs[0]] if k != "prediction_horizon"]
    head = "".join(f"<th>ret_{h}</th>" for h in hs)
    body = "".join(
        f'<tr><td class="feat">{html.escape(m)}</td>'
        + "".join(f"<td>{gfmt(rows_by_h[h].get(m))}</td>" for h in hs)
        + "</tr>" for m in metrics)
    return (f'<div class="ic-h">{model}</div><div class="tblwrap">'
            f'<table class="tbl sm"><thead><tr><th>metric</th>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')

full_tbl = full_summary(lin, "Linear (OLS + Lasso)") + full_summary(lgb, "LGBM")


# ---- per-horizon detail -----------------------------------------------------
def top_ols(h, n=12):
    rows = read_csv(os.path.join(LIN, f"ols_coefficients_ret_{h}.csv"))
    rows = [r for r in rows if r.get("term") != "(Intercept)"]
    rows.sort(key=lambda r: abs(float(r.get("t_value") or 0)), reverse=True)
    body = "".join(
        f'<tr><td class="feat">{html.escape(r["term"])}</td><td>{sci(r["estimate"])}</td>'
        f'<td>{f(r["t_value"],1)}</td></tr>' for r in rows[:n])
    return (f'<table class="tbl sm"><thead><tr><th>OLS term (top |t|)</th><th>coef</th><th>t</th></tr></thead>'
            f'<tbody>{body}</tbody></table>')


def top_lgbm(h, n=12):
    rows = read_csv(os.path.join(LGB, f"lgbm_importance_ret_{h}.csv"))
    rows.sort(key=lambda r: float(r.get("gain") or 0), reverse=True)
    body = "".join(
        f'<tr><td class="feat">{html.escape(r["feature"])}</td><td>{float(r["gain"])*100:.1f}%</td></tr>'
        for r in rows[:n])
    return (f'<table class="tbl sm"><thead><tr><th>LGBM feature (top gain)</th><th>gain</th></tr></thead>'
            f'<tbody>{body}</tbody></table>')


def ic_quantile(h, where, tag):
    rows = read_csv(os.path.join(where, f"ic_by_pred_quantile_ret_{h}.csv"))
    if not rows:
        return ""
    def qlabel(r):   # newer files carry q_lo/q_hi; older ones only a bin index
        return f'{r["q_lo"]}&ndash;{r["q_hi"]}' if r.get("q_lo") else f'bin {r.get("bin","")}'
    def dret(r):     # dir_return where present, else the signed real_mean
        return bps(r["dir_return"]) if r.get("dir_return") else bps(r.get("real_mean"))
    hr = lambda r, k: f(r.get(k), 3)
    body = "".join(
        f'<tr><td>{qlabel(r)}</td><td>{int(float(r["n"])):,}</td>'
        f'<td>{dret(r)}</td><td>{f(r.get("ic_pearson"))}</td><td>{f(r.get("ic_spearman"))}</td>'
        f'<td>{hr(r,"hit_rate")}</td><td>{hr(r,"hit_rate_with_zero")}</td><td>{hr(r,"hit_rate_move")}</td></tr>'
        for r in rows)
    return (f'<div class="ictbl"><div class="ic-h">{tag} &mdash; IC by |pred| quantile '
            f'<span class="val-tag">validation</span></div>'
            f'<table class="tbl sm"><thead><tr><th>|pred| q</th><th>n</th><th>dir ret (bps)</th>'
            f'<th>IC&nbsp;pe</th><th>IC&nbsp;sp</th>'
            f'<th title="0-move = miss">hit</th>'
            f'<th title="0-move = hit">hit&#43;0</th>'
            f'<th title="moved rows only">hit&#183;mv</th></tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def scatter(h):
    imgs = ""
    for where, tag in ((LIN, "OLS"), (LGB, "LGBM")):
        uri = data_uri(os.path.join(where, f"pred_vs_real_scatter_ret_{h}.png"))
        if uri:
            imgs += f'<figure><img src="{uri}" alt="{tag} pred vs real ret_{h}"><figcaption>{tag}</figcaption></figure>'
    return f'<div class="scatter">{imgs}</div>' if imgs else ""


sections = ""
for h in HORIZONS:
    if str(h) not in lin and str(h) not in lgb:
        continue
    detail = (f'<div class="cols"><div>{top_ols(h)}</div><div>{top_lgbm(h)}</div></div>'
              f'<div class="cols">{ic_quantile(h, LIN, "OLS")}{ic_quantile(h, LGB, "LGBM")}</div>'
              f'{scatter(h)}')
    op = " open" if h == 10 else ""
    sections += (f'<details class="hz"{op}><summary>Horizon ret_{h} &mdash; '
                 f'Val IC {f(lin.get(str(h),{}).get("validation_IC"))} (OLS) / '
                 f'{f(lgb.get(str(h),{}).get("validation_IC"))} (LGBM)</summary>{detail}</details>')

CSS = """body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;
     background:#0f172a;color:#e2e8f0}
header{background:linear-gradient(90deg,#1a202c,#2d3748);color:#fff;padding:18px 28px}
header h1{margin:6px 0 0;font-size:20px}
header .meta{color:#94a3b8;font-size:12px;margin-top:6px}
header .back-link{color:#90cdf4;font-size:13px;text-decoration:none}
header .back-link:hover{text-decoration:underline}
section{margin:18px 24px}
h2{font-size:15px;color:#7dd3fc;border-bottom:1px solid #334155;padding-bottom:6px}
.tbl{border-collapse:collapse;font-size:12px;font-family:ui-monospace,Menlo,Consolas,monospace;
     width:100%;margin:6px 0}
.tbl th{text-align:right;color:#94a3b8;font-weight:600;padding:4px 10px;border-bottom:1px solid #334155;
     position:sticky;top:0;background:#0f172a}
.tbl td{text-align:right;padding:3px 10px;border-bottom:1px solid #1e293b}
.tbl td.feat,.tbl th:first-child,.tbl td:first-child{text-align:left}
.tbl td.ic{color:#34d399;font-weight:700}
.tbl.sm{font-size:11px}
tr.m-lgbm td{background:rgba(56,189,248,.06)}
tr.m-lasso td{color:#94a3b8}
details.hz{background:#111827;border:1px solid #334155;border-radius:10px;margin:12px 0;padding:6px 16px}
details.hz>summary{cursor:pointer;font-size:14px;font-weight:600;color:#fbbf24;padding:6px 0}
.cols{display:flex;gap:24px;flex-wrap:wrap}.cols>div{flex:1;min-width:280px}
.ictbl{flex:1;min-width:300px}.ic-h{font-size:12px;color:#94a3b8;margin-top:8px}
.scatter{display:flex;gap:18px;flex-wrap:wrap;margin-top:10px}
.scatter figure{margin:0}.scatter img{max-width:420px;width:100%;border:1px solid #334155;border-radius:6px}
.scatter figcaption{color:#94a3b8;font-size:11px;text-align:center;margin-top:2px}
.note{color:#94a3b8;font-size:12px;margin:4px 0 0}
.note code{color:#cbd5e0}
.val-tag{font-size:10px;font-weight:700;color:#0f172a;background:#34d399;border-radius:4px;
     padding:1px 6px;margin-left:6px;letter-spacing:.3px;text-transform:uppercase}
details.foot{background:#111827;border:1px solid #334155;border-radius:10px;padding:8px 18px;margin-bottom:14px}
details.foot>summary{cursor:pointer;font-size:14px;font-weight:700;color:#fbbf24;padding:4px 0}
.foot-b{font-size:12.5px;color:#cbd5e0;line-height:1.55}
.foot-b code{background:#0f172a;color:#7dd3fc;padding:1px 4px;border-radius:4px;font-size:12px}
.foot-b ul{margin:6px 0;padding-left:20px}.foot-b b{color:#e2e8f0}
details.full{background:#111827;border:1px solid #334155;border-radius:10px;padding:8px 18px;margin-top:12px}
details.full>summary{cursor:pointer;font-size:14px;font-weight:700;color:#7dd3fc;padding:4px 0}
.tblwrap{overflow-x:auto;border:1px solid #1e293b;border-radius:8px;margin:8px 0 16px}
.tblwrap .tbl{margin:0}.tblwrap .tbl th{white-space:nowrap}"""

# Unified-pipeline comparison fragment (written by gen_pipeline_models.py);
# injected as the first section so this page combines BOTH model reports.
_frag_path = os.path.join(HERE, f"_fragment_{SYM}_pipeline_models.inc")
pipeline_frag = open(_frag_path).read() if os.path.isfile(_frag_path) else ""

body = f'''<header>
  <a class="back-link" href="index.html">&larr; All reports</a>
  <h1>{SYM.upper()} &mdash; model report (Linear, LGBM &amp; MLP)</h1>
  <div class="meta">source: alpha_replayer_config/statistics/{SYM}/&#123;linear,lgbm&#125; &middot;
     horizons 1/10/30/60s &middot; IC = validation rank-corr(pred, fwd ret)</div>
</header>
<section>{pipeline_frag}</section>
<section>
  <h2>Horizon summary &mdash; validation metrics by model (legacy trainers)</h2>
  {headline}
  <p class="note">Columns prefixed <b>Val</b> (Val R&sup2;, Val IC, Val IR) are on the
     <b>validation holdout</b> (last 25% by time); <b>IS R&sup2;</b> is in-sample (training). IC is the headline
     skill metric (validation corr of prediction vs realized forward return); R&sup2;/IR are per-observation and
     therefore small. RMSE shown in bps. &beta; = slope of realized on predicted.
     LGBM rows shaded; Lasso shown where it differs from OLS.</p>
</section>
<section>
  <details class="full" open>
    <summary>Full per-horizon summary &mdash; every metric (realized &amp; predicted distribution, vol / skew / kurt, Lasso)</summary>
    <p class="note">Transposed: metrics as rows, horizons as columns. <code>real_*</code> / <code>pred_*</code>
       are distribution quantiles of the realized return and the model prediction respectively;
       <code>*_vol</code> = std, <code>skew/kurt</code> as labelled; <code>lasso_*</code> = the L1-regularized
       counterpart of each fit metric. All validation figures are on the holdout.</p>
    {full_tbl}
  </details>
</section>
<section>
  <details class="foot" open>
    <summary>Interpreting the metrics &mdash; two things that look wrong but aren&rsquo;t</summary>
    <div class="foot-b">
      <p><b>1. IC-Pearson falls in the extreme tail quantiles.</b> Within a |pred| tier the Pearson IC
      factors exactly as <code>pearson = &beta; &middot; (sd_pred / sd_real)</code>, where
      <code>&beta;</code> is the slope of realized on predicted. Two things happen as conviction rises into
      the top bins (numbers = BTC LGBM ret_10, validation):</p>
      <ul>
        <li><b>Realized-vol explosion (dominant).</b> The most extreme predictions fire in the most volatile
        moments, so <code>sd_real</code> nearly doubles in the top bin (2.3e-4 &rarr; 4.3e-4). It sits in the
        denominator, so the <i>fraction</i> of variance explained (correlation/R&sup2;) mechanically deflates &mdash;
        even though the directional signal is intact (<b>hit-rate stays ~0.63&ndash;0.67 and dir-return keeps
        rising monotonically</b>).</li>
        <li><b>Mild saturation (secondary).</b> <code>&beta;</code> slips from ~1.0 to ~0.88 in the last two
        bins &mdash; extreme predictions overshoot (realized is only ~88% of predicted magnitude:
        regression-to-the-mean at the extreme).</li>
      </ul>
      <p>Check: top bin <code>0.88 &times; 9.0e-5 / 4.3e-4 = 0.186</code>. So the tail drop is a
      vol-inflated denominator, <b>not</b> a loss of skill.</p>
      <p><b>2. Val R&sup2; is higher than IS R&sup2; (for most horizons).</b> This is <b>not</b> overfitting
      (that would make Val &lt; IS). For a calibrated predictor <code>R&sup2; &asymp; IC&sup2;</code>
      (train 0.164&sup2;=0.0269&asymp;0.0263; valid 0.181&sup2;=0.0328&asymp;0.0327), so the gap reduces to
      <b>validation IC &gt; training IC</b>: the last 25% of the sample (the validation holdout) carries a
      <b>stronger, cleaner signal and lower realized vol</b> (1.96e-4 vs 2.68e-4) than the longer, noisier
      training span. With ~10M rows and few effective parameters the overfit gap is negligible, so the IS-vs-Val
      difference is a <b>regime / non-stationarity</b> effect &mdash; meaning these validation numbers reflect
      that specific recent regime and may be optimistic for a different future one. (LGBM ret_60 is the lone case
      where Val &lt; IS.)</p>
    </div>
  </details>
  <h2>Per-horizon detail &mdash; top features, decile IC, pred-vs-real</h2>
  <p class="note"><b>All IC-by-|pred|-quantile tables and pred-vs-real scatters below are computed on the
     VALIDATION holdout</b> (last 25% of rows by <code>mark_ts</code>, returns clipped at the 0.1/99.9
     percentiles) &mdash; for both OLS and LGBM. OLS coefficients / LGBM importances are fit on the
     training portion.<br>
     coef = OLS estimate (scientific notation); t = t-stat (rows sorted by |t|).
     In the IC-by-quantile tables, rows are tiers of |pred| conviction;
     <b>IC pe</b> = Pearson (linear) corr(pred, realized), <b>IC sp</b> = Spearman rank
     corr(pred, realized) within the tier, and
     <b>dir ret</b> <code>= mean(sign(pred)&middot;realized)</code> in bps &mdash; the average realized
     return of trading the predicted direction (dollar-weighted: big moves dominate, unlike hit-rate;
     0-moves contribute exactly 0), the cleanest signed-edge proxy.</p>
    <p class="note"><b>The three hit-rate columns</b> differ only in how a realized return of
     <b>exactly 0</b> (no move) is scored:
     <ul class="hitleg">
       <li><code>hit</code> (<code>hit_rate</code>) = <code>sign(pred)==sign(real)</code>; a
           <b>0-move counts as a miss</b> (pessimistic).</li>
       <li><code>hit+0</code> (<code>hit_rate_with_zero</code>) = same, but a <b>0-move counts as a hit</b>
           (no adverse move); inflated at short horizons where many returns are exactly 0, so it mostly
           tracks inactivity.</li>
       <li><code>hit&middot;mv</code> (<code>hit_rate_move</code>) = sign accuracy over <b>rows that moved
           (real&ne;0)</b> only &mdash; the cleanest directional-skill measure.</li>
     </ul>
     The gap <code>hit+0 &minus; hit</code> = share of no-move rows in that tier.</p>
  {sections}
</section>'''

page = ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{SYM.upper()} model report</title><style>{CSS}</style></head><body>{body}</body></html>")
OUT = os.path.join(HERE, f"{SYM}_model.html")
open(OUT, "w").write(page)
print(f"wrote {OUT} | {SYM} | {len(headline_rows())} model-rows ({len(page)//1024} KB)")

# Also publish the per-feature TRAINING REPORT(s) written by train_pipeline.py
# (why-selected + further transform + potential problems). Layout-independent:
# glob under the symbol's stats dir and take the newest per model. Skipped
# silently if a run hasn't produced one yet.
import glob
import re as _re
import shutil
_newest = {}                                   # model-suffix -> newest training report path
for _p in glob.glob(os.path.join(ROOT, "**", "training_report_*.html"), recursive=True):
    _mm = _re.match(r"training_report_(.+)\.html$", os.path.basename(_p))
    if not _mm:
        continue
    _mdl = _mm.group(1)                        # linear | lgbm | lasso_ols | mlp | ...
    if _mdl not in _newest or os.path.getmtime(_p) > os.path.getmtime(_newest[_mdl]):
        _newest[_mdl] = _p
for _mdl, _p in _newest.items():
    _dst = os.path.join(HERE, f"{SYM}_training_report_{_mdl}.html")
    shutil.copyfile(_p, _dst)
    print(f"published {os.path.basename(_dst)} <- {_p}")

# (pre_training.py writes <sym>_pre_training.html straight into the report repo,
# so no copy step is needed here — make_index picks it up directly.)
