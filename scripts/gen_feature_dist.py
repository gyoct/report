#!/usr/bin/env python3
"""Render a *_distribution.txt feature report into a styled dark HTML page.

Usage:
    python gen_feature_dist.py <version>            # BTC (legacy)   e.g. v4, v5
    python gen_feature_dist.py <symbol> <version>   # any symbol     e.g. eth v1, sol v1

Sources per symbol (first existing path wins):
    btc: ../CR_TRAINING/PY/btc_<ver>_110200172_..._distribution.txt
    eth: ../alpha_replayer_config/eth/20260602/eth_<ver>_110200089_..._distribution.txt
    sol: ../alpha_replayer_config/sol/20260602/sol_<ver>_110200132_..._distribution.txt
    (CR_TRAINING/PY is also searched for every symbol.)

Writes ./<sym>_feature_distribution[_<ver>].html — the FIRST version of each
symbol (btc v4, eth/sol v1) keeps the unsuffixed name so the index pill stays
stable. Then run make_index.py to (re)encrypt + publish.
"""
import html, re, sys, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SYMS = {
    "btc": dict(ukey="110200172", base_ver="v4"),
    "eth": dict(ukey="110200089", base_ver="v1"),
    "sol": dict(ukey="110200132", base_ver="v1"),
    # metals (stock-perp namespace): data window differs from crypto — recording
    # starts 2026-06-01; XAU's config ends 0720 (partial last day), XAG's 0719.
    # 2026-08-14: window extended to 4.5 months; dumps moved to /mnt/nvme2/ghu/output
    "xau": dict(ukey="110800001", base_ver="v1", window="202604270000_202608132359"),
    "xag": dict(ukey="110800002", base_ver="v2", window="202604270000_202608132359"),
}
WINDOW = "202601010000_202605202359"  # default (crypto); per-symbol override via SYMS["window"]

# DATA-GAP warnings per (symbol, version): the parallel replayer can silently drop
# worker chunks (see memory/project_replay_gap_audit). Audited 2026-07-31 by mark_ts
# scan; rendered as an amber banner so the coverage caveat travels with the numbers.
GAPS = {
    # ("btc", "v6") banner REMOVED 2026-08-04: the gappy v6 was deleted and re-run
    # (2026-08-02 generation); fresh stats show 100.0% coverage, 0 gaps >= 60s.
    # ("eth","v2")/("sol","v2") banners REMOVED 2026-08-04: those described the DELETED
    # 07-28 dumps; the 08-02 reruns measure 100.0% coverage so far (see the in-page
    # DATA COVERAGE panel, which is now measured per-report — hardcoded banners obsolete).
}

# ---- args: [symbol] [version] with legacy single-arg = btc version ----------
args = sys.argv[1:]
# optional --src PATH: render an explicit .txt (bypasses the SYMS-window lookup);
# used to (re)generate a FROZEN historical page whose window differs from the
# symbol's current SYMS window (e.g. the Jun-Jul base page while SYMS is on the
# extended window).
SRC_OVERRIDE = None
if '--src' in args:
    _i = args.index('--src')
    SRC_OVERRIDE = args[_i + 1]
    del args[_i:_i + 2]
if len(args) == 1 and args[0] in SYMS:
    SYM, VER = args[0], SYMS[args[0]]["base_ver"]
elif len(args) == 1:                      # legacy: gen_feature_dist.py v5
    SYM, VER = "btc", args[0]
elif len(args) >= 2:
    SYM, VER = args[0], args[1]
else:
    SYM, VER = "btc", "v5"
if SYM not in SYMS:
    sys.exit(f"unknown symbol {SYM!r} (expected {', '.join(SYMS)})")
UKEY = SYMS[SYM]["ukey"]

WINDOW = SYMS[SYM].get("window", WINDOW)
# env override for regenerating HISTORICAL pages whose TXT carries an older
# window than SYMS' current default (e.g. the metals Jun-Jul base pages)
WINDOW = os.environ.get("GFD_WINDOW", WINDOW)
fname = f"{SYM}_{VER}_{UKEY}_{WINDOW}_distribution.txt"
CANDIDATES = [
    os.path.join(HERE, "..", "CR_TRAINING", "PY", fname),
    os.path.join(HERE, "..", "alpha_replayer_config", SYM, "20260602", fname),
]
SRC = SRC_OVERRIDE or next((p for p in CANDIDATES if os.path.isfile(p)), None)
if SRC is None or not os.path.isfile(SRC):
    sys.exit("source not found: " + (SRC_OVERRIDE or "\n  " + "\n  ".join(CANDIDATES)))

# optional 3rd arg = page label (e.g. "v2" for the extended-window rerun):
# names the PAGE independently of the dump version embedded in the filename
PAGE_LABEL = (args[2] if len(args) >= 3 else "").lower()
# version-PREFIXED naming: v<N>_<sym>_feature_distribution.html (PAGE_LABEL,
# when given, is the page version — e.g. metals window-v2 pages on a v1 dump)
_pv = PAGE_LABEL or VER
OUT = os.path.join(HERE,
                   f"{_pv}_{SYM}_feature_distribution.html")

raw = open(SRC).read()
# drop the icS/icR corr definitions from the header line (kept elsewhere in the report)
raw = raw.replace("icS=corr(feat, fwd log-ret)  icR=corr(|feat|, |fwd ret|)  ", "")

m = re.search(r"rows=([\d,]+)\s+features=(\d+)", raw)
rows, feats = m.group(1), m.group(2)

# Pull the DATA/DAILY COVERAGE block out of the main report body and render it
# as a COLLAPSIBLE <details> under the header: the one-line summary stays
# visible; the gap list + per-day table expand on click.
# Pull the REPLAY LOG REVIEW section out FIRST (it sits between the coverage
# block and the feature table — leaving it in would be swallowed into the
# coverage panel below). Rendered as its own collapsible; auto-opens when the
# log flags broken raw data.
logrev_html = ""
lm = re.search(r"^={78}\nREPLAY LOG REVIEW.*?(?=^feature\s|^={78}\n[A-Z])", raw, flags=re.M | re.S)
if lm:
    block = lm.group(0).rstrip()
    raw = raw[:lm.start()] + raw[lm.end():]
    body_lines = [l for l in block.split("\n") if not set(l) == {"="}]
    title, rest = body_lines[0], "\n".join(body_lines[1:])
    rest_esc = html.escape(rest)
    rest_esc = re.sub(r"^(\s*\d+x\s+.*)$", r'<span class="lr-class">\1</span>', rest_esc, flags=re.M)
    rest_esc = rest_esc.replace("COMPROMISED", '<span class="miss">COMPROMISED</span>')
    rest_esc = rest_esc.replace("RAW RECORD ENCODING BROKEN", '<span class="miss">RAW RECORD ENCODING BROKEN</span>')
    warn = ' open' if 'RAW RECORD ENCODING BROKEN' in block else ''
    logrev_html = (f'<details class="cov"{warn}><summary>{html.escape(title)}</summary>'
                   f'<pre class="rpt covbody">{rest_esc}</pre></details>')

cov_html = ""
cm = re.search(r"^DATA COVERAGE:.*?(?=^feature\s)", raw, flags=re.M | re.S)
if cm:
    block = cm.group(0).rstrip()
    raw = raw[:cm.start()] + raw[cm.end():]
    first, _, rest = block.partition("\n")
    rest_esc = html.escape(rest)
    rest_esc = re.sub(r"^(.*MISSING.*)$", r'<span class="miss">\1</span>', rest_esc, flags=re.M)
    warn = ' open' if 'GAPS' in first else ''
    cov_html = (f'<details class="cov"{warn}><summary>{html.escape(first)}</summary>'
                f'<pre class="rpt covbody">{rest_esc}</pre></details>')

esc = html.escape(raw)
esc = re.sub(r"(?<= )(BLOWUP)(?=\s|$)", r'<span class="f-blow">\1</span>', esc, flags=re.M)
esc = re.sub(r"(?<= )(blowup\?)", r'<span class="f-blow">\1</span>', esc)
esc = re.sub(r"(?<= )(bounded)(?=\s|$)", r'<span class="f-bnd">\1</span>', esc, flags=re.M)
# new curation columns (feature_distribution.py): outlier HEAVY/MILD + divide-by-small SUSP
esc = re.sub(r"(?<= )(HEAVY|SUSP)(?=\s|$)", r'<span class="f-blow">\1</span>', esc, flags=re.M)
esc = re.sub(r"(?<= )(MILD)(?=\s|$)", r'<span class="f-bnd">\1</span>', esc, flags=re.M)
esc = re.sub(r"(?<= )(ok)(?=\s*$)", r'<span class="f-ok">\1</span>', esc, flags=re.M)
esc = re.sub(r"^(=+)$", r'<span class="hr">\1</span>', esc, flags=re.M)
esc = re.sub(r"^(\[\d+\][^\n]*)$", r'<span class="sec">\1</span>', esc, flags=re.M)

# column-formula legend (only xσ was documented on-page; spike/capSug added
# 2026-08-26). Injected so EVERY existing page carries it without re-profiling;
# skipped when the source TXT already contains the new-format legend.
LEGEND = ("<b>x&sigma;</b> = max|x|/std &mdash; effective sigma a StandardScaler-fed "
          "MLP sees at the feature max &middot; "
          "<b>spike</b> = max/p99 &mdash; denominator-collapse detector (healthy tail "
          "= a few &times;; thousands &times; = the rolling normalizer denominator "
          "collapsed for some window) &middot; "
          "<b>capSug</b> = p99.9(|x|) &mdash; suggested production feature <code>cap</code> "
          "in raw emission units; only filled for outlier / denom-collapse rows, "
          "blank = leave uncapped")
# (new-format TXTs carry plain-text spike/capSug lines; they are stripped at
# the body-split stage below, where the collapsible replaces them)

# body split: the collapsible sits WITH the report-header comment lines
# (file / rows / xσ / horizons), right above the feature table
_lines = [l for l in esc.split("\n")
          if not l.startswith(("spike = max/p99", "capSug = p99.9"))]
_cut = next((i for i, l in enumerate(_lines) if l.startswith("feature ")), 0)
if _cut > 0 and LEGEND:
    body_pre = (f'<pre class="rpt">{chr(10).join(_lines[:_cut])}</pre>'
                f'<details class="defs-c"><summary>&#9432; Column formulas — '
                f'x&sigma; / spike / capSug</summary>'
                f'<div class="meta legend">{LEGEND}</div></details>'
                f'<pre class="rpt">{chr(10).join(_lines[_cut:])}</pre>')
else:
    body_pre = f'<pre class="rpt">{esc}</pre>'

gap_note = GAPS.get((SYM, VER), "")
gap_html = (f'<div class="gapwarn">&#9888; {html.escape(gap_note)}</div>' if gap_note else "")

body = f'''<header>
  <a class="back-link" href="index.html">&larr; All reports</a>
  <h1>{SYM.upper()} {PAGE_LABEL or VER} &mdash; feature distribution &amp; IC report</h1>
  <div class="meta">source: {SYM}_{VER}_{UKEY}_{WINDOW}.csv &middot;
     {rows} rows &middot; {feats} features &middot; window
     {WINDOW[:4]}-{WINDOW[4:6]}-{WINDOW[6:8]} &rarr; {WINDOW[13:17]}-{WINDOW[17:19]}-{WINDOW[19:21]}</div>
  {gap_html}
  {cov_html}
</header>
<section>{body_pre}</section>
<section>{logrev_html}</section>'''

CSS = """body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;
     background:#0f172a;color:#e2e8f0}
header{background:linear-gradient(90deg,#1a202c,#2d3748);color:#fff;padding:18px 28px}
header h1{margin:6px 0 0;font-size:19px}
header .meta{color:#94a3b8;font-size:12px;margin-top:6px}
header .back-link{color:#90cdf4;font-size:13px;text-decoration:none}
header .back-link:hover{text-decoration:underline}
details.defs-c{margin:8px 0 2px}
details.defs-c>summary{cursor:pointer;color:#f6ad55;font-size:12.5px;font-weight:600;list-style:none}
details.defs-c>summary:hover{color:#fbd38d}
details.defs-c .legend{margin-top:6px;line-height:1.6;max-width:1100px}
section{background:#111827;margin:18px 24px;padding:8px 14px;border:1px solid #334155;
     border-radius:10px;overflow-x:auto}
pre.rpt{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;
     line-height:1.45;white-space:pre;margin:0;color:#cbd5e0}
.f-blow{color:#f87171;font-weight:700}.f-bnd{color:#fbbf24}.f-ok{color:#34d399}
details.cov{margin:10px 0 0;border:1px solid #334155;border-radius:10px;padding:8px 14px;
  background:#16233b;}
details.cov summary{cursor:pointer;font-size:13px;font-weight:600;color:#fbbf24;
  list-style-position:inside;}
details.cov summary:hover{color:#fde68a;}
pre.covbody{margin:8px 0 2px;max-height:340px;overflow-y:auto;}
.miss{color:#f87171;font-weight:600;}
.lr-class{color:#e2c08d;}
.hr{color:#475569}.sec{color:#7dd3fc;font-weight:700}
.gapwarn{margin-top:10px;padding:8px 12px;border-left:3px solid #fbbf24;background:rgba(251,191,36,.09);
     color:#fbbf24;font-size:12.5px;border-radius:0 6px 6px 0}"""

page = ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{SYM.upper()} {VER} feature distribution</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>")
open(OUT, "w").write(page)
print(f"wrote {OUT} | {SYM} {VER} rows={rows} features={feats} ({len(page)//1024} KB)")
