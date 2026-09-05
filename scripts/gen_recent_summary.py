#!/usr/bin/env python3
"""Summarize a production monetization report's stats from a cutoff date onward.

Usage: python gen_recent_summary.py <product_dir> <since YYYY-MM-DD> <out.html> [Title]
  e.g. python gen_recent_summary.py \
         ../CR_TRAINING/PY/prod/btc_monetization2 2026-08-19 btc_prod2_recent.html \
         "BTC Monetization 2"

Reads analysis_out/{order_hourly.csv, cum_pnl_commission.csv, position_midprice.csv},
filters to >= the cutoff, and renders a dark, self-contained page: headline KPIs,
per-day + per-hour order stats, a rebased cumulative PnL/alpnl/commission curve, and
position vs mid. Markout / holding-PnL are all-history aggregates in the source and are
NOT date-sliceable without re-running order_multi.R, so they are intentionally omitted.
"""
import csv, os, sys, json, datetime as dt

PROD = sys.argv[1] if len(sys.argv) > 1 else "../CR_TRAINING/PY/prod/btc_monetization2"
SINCE = sys.argv[2] if len(sys.argv) > 2 else "2026-08-19"
OUT = sys.argv[3] if len(sys.argv) > 3 else "btc_prod2_recent.html"
TITLE = sys.argv[4] if len(sys.argv) > 4 else "BTC Monetization 2"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AO = os.path.join(HERE, PROD, "analysis_out")
SINCE_KEY = SINCE.replace("-", "")            # YYYYMMDD for the hour-label filter
SINCE_ISO = SINCE + "T00:00:00Z"


def rd(name):
    with open(os.path.join(AO, name), newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


oh = rd("order_hourly.csv")
days = [r for r in oh if len(r["hour"]) == 8 and r["hour"] >= SINCE_KEY]
hours = [r for r in oh if len(r["hour"]) == 10 and r["hour"][:8] >= SINCE_KEY]
days.sort(key=lambda r: r["hour"]); hours.sort(key=lambda r: r["hour"])
if not days:
    sys.exit(f"no day rows >= {SINCE_KEY} in {AO}/order_hourly.csv")

# ---- period totals (sum day rows; count-weighted rates) ---------------------
tot = dict(Count=0.0, TotalVolume=0.0, turnover=0.0, commission=0.0, pnl=0.0, alpnl=0.0)
fr_num = hold_num = 0.0
for r in days:
    c = fnum(r["Count"])
    for k in tot:
        tot[k] += fnum(r[k])
    fr_num += fnum(r["fillRate"]) * c
    hold_num += fnum(r["avgHoldingSec"]) * c
fill_rate = fr_num / tot["Count"] if tot["Count"] else 0.0
avg_hold = hold_num / tot["Count"] if tot["Count"] else 0.0
# bps = pnl / TotalVolume * 1e4 (matches the strategy's own pnl_pct = pnl/TotalVolume).
# Volume in USD = TotalVolume = sum(trd_qty * trd_px), the notional traded.
alpnl_bps = tot["alpnl"] / tot["TotalVolume"] * 1e4 if tot["TotalVolume"] else 0.0
pnl_bps = tot["pnl"] / tot["TotalVolume"] * 1e4 if tot["TotalVolume"] else 0.0


def usd(v):
    a = abs(v)
    if a >= 1e6:
        return f"${v/1e6:.2f}M"
    if a >= 1e3:
        return f"${v/1e3:.1f}k"
    return f"${v:,.0f}"


def cls(v):
    return "pos" if v >= 0 else "neg"


def pnl_cell(u, pct):
    return f'<td class="{cls(u)}">{u:+.3f} <span class="bps">{pct*1e4:+.2f} bps</span></td>'
span = f'{days[0]["startTime"][:16].replace("T"," ")} &rarr; {days[-1]["endTime"][:16].replace("T"," ")} UTC'

# ---- cumulative PnL curve for the period (rebased to 0 at cutoff) -----------
cum = [r for r in rd("cum_pnl_commission.csv") if r["ts"] >= SINCE_ISO]
if cum:
    b_c, b_p, b_a = fnum(cum[0]["cum_commission"]), fnum(cum[0]["cum_pnl"]), fnum(cum[0]["cum_alpnl"])
    cts = [r["ts"] for r in cum]
    c_comm = [fnum(r["cum_commission"]) - b_c for r in cum]
    c_pnl = [fnum(r["cum_pnl"]) - b_p for r in cum]
    c_alp = [fnum(r["cum_alpnl"]) - b_a for r in cum]
else:
    cts = c_comm = c_pnl = c_alp = []

# ---- position vs mid for the period ----------------------------------------
pos = [r for r in rd("position_midprice.csv") if r["ts"] >= SINCE_ISO]
pts = [r["ts"] for r in pos]
p_pos = [fnum(r["position"]) for r in pos]
p_mid = [fnum(r["mid_px"]) for r in pos]

# ---- per-hour alpnl bars ----------------------------------------------------
h_lab = [f'{r["hour"][:4]}-{r["hour"][4:6]}-{r["hour"][6:8]} {r["hour"][8:10]}h' for r in hours]
h_alp = [fnum(r["alpnl"]) for r in hours]
h_pnl = [fnum(r["pnl"]) for r in hours]


def tile(label, val, sub=""):
    return (f'<div class="tile"><div class="t-val">{val}</div>'
            f'<div class="t-lab">{label}</div>'
            + (f'<div class="t-sub">{sub}</div>' if sub else "") + '</div>')


kpis = "".join([
    tile("Fills", f'{int(tot["Count"]):,}', f'{len(days)} day(s)'),
    tile("Fill rate", f'{fill_rate*100:.1f}%'),
    tile("Volume (USD)", usd(tot["TotalVolume"]), "notional traded"),
    tile("Avg holding", f'{avg_hold:.1f}s'),
    tile("Gross alpnl", f'{tot["alpnl"]:+.2f}', f'{alpnl_bps:+.2f} bps'),
    tile("Net PnL", f'{tot["pnl"]:+.2f}', f'{pnl_bps:+.2f} bps'),
    tile("Commission", f'{tot["commission"]:.2f}'),
])

day_rows = "".join(
    f'<tr><td>{r["hour"][:4]}-{r["hour"][4:6]}-{r["hour"][6:8]}</td>'
    f'<td>{int(fnum(r["Count"])):,}</td><td>{fnum(r["fillRate"])*100:.1f}%</td>'
    f'<td>{usd(fnum(r["TotalVolume"]))}</td><td>{fnum(r["avgHoldingSec"]):.1f}s</td>'
    + pnl_cell(fnum(r["alpnl"]), fnum(r["alpnl_pct"]))
    + pnl_cell(fnum(r["pnl"]), fnum(r["pnl_pct"]))
    + f'<td>{fnum(r["commission"]):.3f}</td></tr>' for r in days)

hour_rows = "".join(
    f'<tr><td>{r["hour"][:4]}-{r["hour"][4:6]}-{r["hour"][6:8]} {r["hour"][8:10]}h</td>'
    f'<td>{int(fnum(r["Count"])):,}</td><td>{fnum(r["fillRate"])*100:.1f}%</td>'
    + pnl_cell(fnum(r["alpnl"]), fnum(r["alpnl_pct"]))
    + pnl_cell(fnum(r["pnl"]), fnum(r["pnl_pct"]))
    + f'<td>{fnum(r["commission"]):.3f}</td></tr>' for r in hours)

FIG = json.dumps  # shorthand

pnl_fig = {
    "data": [
        {"x": cts, "y": c_alp, "name": "cum alpnl (gross)", "line": {"color": "#34d399", "width": 2}},
        {"x": cts, "y": c_pnl, "name": "cum pnl (net)", "line": {"color": "#f87171", "width": 2}},
        {"x": cts, "y": c_comm, "name": "cum commission", "line": {"color": "#94a3b8", "width": 1, "dash": "dot"}},
    ],
    "layout": {"template": "plotly_dark", "height": 340, "margin": {"l": 50, "r": 20, "t": 10, "b": 30},
               "paper_bgcolor": "#111827", "plot_bgcolor": "#111827",
               "legend": {"orientation": "h", "y": 1.12}, "hovermode": "x unified",
               "yaxis": {"title": "cum (rebased to 0 at cutoff)"}},
}
bar_fig = {
    "data": [{"x": h_lab, "y": h_alp, "type": "bar", "name": "alpnl",
              "marker": {"color": ["#34d399" if v >= 0 else "#f87171" for v in h_alp]}}],
    "layout": {"template": "plotly_dark", "height": 300, "margin": {"l": 50, "r": 20, "t": 10, "b": 80},
               "paper_bgcolor": "#111827", "plot_bgcolor": "#111827",
               "yaxis": {"title": "alpnl per hour"}},
}
pos_fig = {
    "data": [
        {"x": pts, "y": p_pos, "name": "position", "line": {"color": "#7dd3fc", "width": 1}, "yaxis": "y"},
        {"x": pts, "y": p_mid, "name": "mid px", "line": {"color": "#fbbf24", "width": 1}, "yaxis": "y2"},
    ],
    "layout": {"template": "plotly_dark", "height": 300, "margin": {"l": 50, "r": 55, "t": 10, "b": 30},
               "paper_bgcolor": "#111827", "plot_bgcolor": "#111827", "hovermode": "x unified",
               "legend": {"orientation": "h", "y": 1.12},
               "yaxis": {"title": "position"}, "yaxis2": {"title": "mid", "overlaying": "y", "side": "right"}},
}

CSS = """body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;
     background:#0f172a;color:#e2e8f0}
header{background:linear-gradient(90deg,#1a202c,#2d3748);color:#fff;padding:18px 28px}
header h1{margin:6px 0 0;font-size:20px}
header .meta{color:#94a3b8;font-size:12px;margin-top:6px}
header .back-link{color:#90cdf4;font-size:13px;text-decoration:none}
header .back-link:hover{text-decoration:underline}
section{margin:18px 24px}
h2{font-size:15px;color:#7dd3fc;border-bottom:1px solid #334155;padding-bottom:6px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.tile{background:#111827;border:1px solid #334155;border-radius:12px;padding:14px 16px}
.t-val{font-size:22px;font-weight:700}
.t-lab{color:#94a3b8;font-size:12px;margin-top:4px}
.t-sub{color:#64748b;font-size:11px;margin-top:2px}
.card{background:#111827;border:1px solid #334155;border-radius:12px;padding:10px 14px}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-family:ui-monospace,Menlo,Consolas,monospace}
th{text-align:right;color:#94a3b8;font-weight:600;padding:6px 10px;border-bottom:1px solid #334155}
td{text-align:right;padding:5px 10px;border-bottom:1px solid #1e293b}
th:first-child,td:first-child{text-align:left}
td.pos{color:#34d399}td.neg{color:#f87171}
.bps{color:#94a3b8;font-size:10.5px;font-weight:400}
details>summary{cursor:pointer;color:#7dd3fc;font-size:13px;font-weight:600;padding:6px 0}
.note{color:#94a3b8;font-size:12px}"""

now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
body = f'''<header>
  <a class="back-link" href="index.html">&larr; All reports</a>
  <h1>{TITLE} &mdash; stats since {SINCE}</h1>
  <div class="meta">{span} &middot; source: analysis_out/order_hourly &middot; generated {now}</div>
</header>
<section><div class="tiles">{kpis}</div>
  <p class="note" style="margin-top:10px">Gross <b>alpnl</b> = alpha PnL before fees; <b>Net PnL</b> = after commission.
  bps = per-turnover. Markout / holding-PnL curves are all-history aggregates upstream and are not shown here (they
  cannot be date-sliced without re-running <code>order_multi.R</code>).</p>
</section>
<section><h2>Cumulative PnL over the period (rebased to 0 at cutoff)</h2>
  <div class="card"><div id="pnl"></div></div></section>
<section><h2>Per-day</h2>
  <div class="card"><table><thead><tr><th>day</th><th>fills</th><th>fill%</th><th>volume</th>
  <th>avg hold</th><th>alpnl</th><th>pnl</th><th>comm</th></tr></thead><tbody>{day_rows}</tbody></table></div>
</section>
<section><h2>alpnl by hour</h2>
  <div class="card"><div id="bar"></div></div>
  <details><summary>per-hour table ({len(hours)} hours)</summary>
  <div class="card" style="margin-top:8px"><table><thead><tr><th>hour</th><th>fills</th><th>fill%</th>
  <th>alpnl</th><th>pnl</th><th>comm</th></tr></thead><tbody>{hour_rows}</tbody></table></div></details>
</section>
<section><h2>Position vs mid</h2>
  <div class="card"><div id="pos"></div></div></section>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
Plotly.newPlot('pnl', {FIG(pnl_fig['data'])}, {FIG(pnl_fig['layout'])}, {{displayModeBar:false, responsive:true}});
Plotly.newPlot('bar', {FIG(bar_fig['data'])}, {FIG(bar_fig['layout'])}, {{displayModeBar:false, responsive:true}});
Plotly.newPlot('pos', {FIG(pos_fig['data'])}, {FIG(pos_fig['layout'])}, {{displayModeBar:false, responsive:true}});
</script>'''

page = ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{TITLE} since {SINCE}</title><style>{CSS}</style></head><body>{body}</body></html>")
out_path = os.path.join(HERE, OUT)
open(out_path, "w").write(page)
print(f"wrote {out_path} | {len(days)} days, {len(hours)} hours since {SINCE} "
      f"| alpnl {tot['alpnl']:+.2f} pnl {tot['pnl']:+.2f} ({len(page)//1024} KB)")
