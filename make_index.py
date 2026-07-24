#!/usr/bin/env python3
"""Build index.html in this folder -- a landing page linking every per-model
report, grouped by symbol. For each symbol a card shows TRAINING reports on
top and PRODUCTION reports on the bottom.

Reports are auto-discovered from *.html files in DIR (index.html itself is
skipped). The symbol and stage are inferred from the file name:

  <symbol>_eqweights.html   -> Training / Equally weighted
  <symbol>_weighted.html    -> Training / Weighted
  <symbol>_prod.html        -> Production / Monetization
  <symbol>_prod2.html       -> Production / Monetization 2
  <symbol>_prodN.html       -> Production / Monetization N
  anything else             -> Production / <prettified name>

So new reports show up automatically -- no table to maintain. To override a
label or force ordering, add an entry to OVERRIDES below.

Run:  python make_index.py            (or: python make_index.py --dir <path>)
"""
from __future__ import annotations

import argparse
import os
import re
import glob
import subprocess
from datetime import datetime

# symbols listed here keep this order; any other discovered symbol is appended.
SYMBOL_ORDER = ["btc", "eth", "sol"]

# optional per-file overrides: filename -> (symbol, stage, label).
# stage must be "Training" or "Production". Within a stage, entries are ordered
# alphabetically by label, so a label after "Monetization 2" lands below it.
OVERRIDES: dict[str, tuple[str, str, str]] = {
    # bgb/btc_taker report: show under the BTC card's Production section,
    # below "Monetization 2" ("Taker..." sorts after "Monetization 2").
    "bgb_btc.html": ("btc", "Production", "Taker (BGB)"),
}


def classify(fname: str):
    """Infer (symbol, stage, label) from a report file name."""
    if fname in OVERRIDES:
        return OVERRIDES[fname]

    stem = re.sub(r"\.html$", "", fname)
    # symbol = leading token before the first underscore
    parts = stem.split("_", 1)
    symbol = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    rl = rest.lower()
    if "eqweight" in rl:
        return symbol, "Training", "Equally weighted"
    if "weighted" in rl:
        return symbol, "Training", "Weighted"
    m = re.search(r"prod(\d*)", rl)
    if m:
        n = m.group(1)
        label = "Monetization" if n in ("", "1") else f"Monetization {n}"
        return symbol, "Production", label
    if "train" in rl:
        return symbol, "Training", rest.replace("_", " ").strip().title() or "Training"
    # fallback: treat as a production report, prettify the remaining name
    pretty = rest.replace("_", " ").strip().title() or "Report"
    return symbol, "Production", pretty


def discover(dir_: str):
    """Return {symbol: {stage: [(label, file), ...]}} from *.html in dir_."""
    out: dict[str, dict[str, list[tuple[str, str]]]] = {}
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(dir_, "*.html")))
    # index.html is the page itself; spread.html is a standalone page (SpreadArb
    # order summary) that is intentionally kept off the landing page.
    skip = {"index.html", "spread.html", "plot.html"}
    for f in files:
        if f in skip:
            continue
        symbol, stage, label = classify(f)
        out.setdefault(symbol, {"Training": [], "Production": []})
        out[symbol].setdefault(stage, []).append((label, f))
    # stable label sort within each stage
    for stages in out.values():
        for lst in stages.values():
            lst.sort(key=lambda x: x[0])
    return out


def report_pill(label: str, file: str, dir_: str) -> str:
    if os.path.exists(os.path.join(dir_, file)):
        return (f'<a class="pill" href="{file}">'
                f'<span class="pill-label">{label}</span>'
                f'<span class="pill-file">{file}</span></a>')
    return ('<span class="pill missing">'
            f'<span class="pill-label">{label}</span>'
            f'<span class="pill-file">not found: {file}</span></span>')


def stage_block(reports: list[tuple[str, str]], stage: str, dir_: str) -> str:
    if not reports:
        pills = '<div class="empty">&mdash;</div>'
    else:
        pills = ('<div class="pills">'
                 + "".join(report_pill(lbl, f, dir_) for lbl, f in reports)
                 + '</div>')
    return (f'<div class="stage stage-{stage.lower()}">'
            f'<div class="stage-head">{stage}<span class="count">{len(reports)}</span></div>'
            f'{pills}</div>')


def build_card(symbol: str, stages: dict[str, list[tuple[str, str]]], dir_: str) -> str:
    return ('<div class="card">'
            f'<div class="card-head"><span class="sym">{symbol.upper()}</span></div>'
            + stage_block(stages.get("Training", []), "Training", dir_)
            + stage_block(stages.get("Production", []), "Production", dir_)
            + '</div>')


CSS = """
:root{
  --bg:#0f172a; --panel:#1e293b; --panel2:#27354a; --line:#334155;
  --txt:#e2e8f0; --muted:#94a3b8; --train:#38bdf8; --prod:#34d399;
}
*{box-sizing:border-box;}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
     margin:0;background:radial-gradient(1200px 600px at 20% -10%,#1e293b,#0f172a);
     color:var(--txt);line-height:1.5;min-height:100vh;}
.wrap{max-width:1040px;margin:0 auto;padding:48px 24px 64px;}
header h1{margin:0;font-size:30px;letter-spacing:.3px;}
header .meta{color:var(--muted);font-size:13px;margin-top:8px;}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));
      gap:24px;margin-top:32px;}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;
      overflow:hidden;box-shadow:0 10px 30px rgba(0,0,0,.35);
      transition:transform .15s ease,box-shadow .15s ease;}
.card:hover{transform:translateY(-3px);box-shadow:0 16px 40px rgba(0,0,0,.45);}
.card-head{padding:18px 22px;background:linear-gradient(90deg,#334155,#1e293b);
           border-bottom:1px solid var(--line);}
.sym{font-size:22px;font-weight:700;letter-spacing:1px;}
.stage{padding:16px 22px;}
.stage + .stage{border-top:1px dashed var(--line);}
.stage-head{display:flex;align-items:center;gap:8px;font-size:12px;
            text-transform:uppercase;letter-spacing:1.5px;font-weight:700;
            margin-bottom:12px;}
.stage-training .stage-head{color:var(--train);}
.stage-production .stage-head{color:var(--prod);}
.count{font-size:11px;color:var(--muted);background:var(--panel2);
       border-radius:10px;padding:1px 8px;font-weight:600;letter-spacing:0;}
.pills{display:flex;flex-direction:column;gap:8px;}
.pill{display:flex;flex-direction:column;text-decoration:none;color:var(--txt);
      background:var(--panel2);border:1px solid var(--line);border-radius:10px;
      padding:10px 14px;transition:border-color .15s,background .15s;}
.pill:hover{background:#30425c;border-color:#475569;}
.stage-training .pill:hover{border-color:var(--train);}
.stage-production .pill:hover{border-color:var(--prod);}
.pill-label{font-weight:600;font-size:14px;}
.pill-file{font-size:11px;color:var(--muted);margin-top:2px;
           font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
.pill.missing{opacity:.5;cursor:not-allowed;}
.pill.missing .pill-file{color:#ef9a9a;}
.empty{color:var(--muted);font-size:13px;padding:4px 0;}
footer{color:var(--muted);font-size:12px;margin-top:40px;text-align:center;}
"""


def git_publish(dir_: str, msg: str) -> None:
    """History-free deploy: publish the current tree as a SINGLE parentless commit and
    FORCE-push it, so the repo only ever holds the latest snapshot.

    The report pages are large and (being encrypted) do NOT delta-compress, so each
    publish is ~95MB of fresh, incompressible blobs. Keeping per-commit history would
    balloon the repo ~2GB/day and eventually blow past GitHub's pack (2GB) / repo-size
    limits -- which is exactly what jammed this pipeline before. A single orphan commit
    keeps every push to one snapshot's worth of objects and the repo bounded. Report
    history is intentionally discarded (the pages are regenerated hourly; old snapshots
    have no value)."""
    # Sanitize the env so git's ssh transport uses the system libssl, not the
    # conda env's (which triggers "OpenSSL version mismatch" and a push fail).
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    env.setdefault("GIT_SSH_COMMAND", "/usr/bin/ssh")

    def run(*cmd, capture=False, check=True):
        return subprocess.run(["git", "-C", dir_, *cmd], check=check,
                              text=True, capture_output=capture, env=env)

    if run("rev-parse", "--is-inside-work-tree", capture=True, check=False).stdout.strip() != "true":
        print(f"[publish] {dir_} is not a git repo -- skipping push")
        return

    branch = run("rev-parse", "--abbrev-ref", "HEAD", capture=True).stdout.strip()
    run("add", "-A")
    tree = run("write-tree", capture=True).stdout.strip()
    prev = run("rev-parse", "-q", "--verify", "HEAD^{tree}",
               capture=True, check=False).stdout.strip()
    if tree and tree == prev:
        print("[publish] nothing changed -- skipping commit/push")
        return

    commit = run("commit-tree", tree, "-m", msg, capture=True).stdout.strip()  # no parent
    run("update-ref", f"refs/heads/{branch}", commit)
    run("push", "--force", "origin", branch)
    print(f"[publish] force-pushed snapshot to origin/{branch} ({commit[:9]})")

    # Each force-push orphans the PREVIOUS snapshot's objects (~95MB of incompressible
    # encrypted blobs). Nothing prunes them automatically, so without this the local
    # .git would grow ~95MB every publish. Expire the reflog and prune now so the repo
    # stays bounded to ~one snapshot. Cheap on a repo this small (a few seconds).
    run("reflog", "expire", "--expire=now", "--all", check=False)
    run("gc", "--prune=now", "--quiet", check=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="Folder holding the *.html reports (default: this script's dir).")
    ap.add_argument("--no-push", action="store_true",
                    help="Only rebuild index.html; do not git commit/push.")
    ap.add_argument("--message", default=None,
                    help="Commit message (default: 'publish reports <timestamp>').")
    args = ap.parse_args()

    found = discover(args.dir)
    ordered = [s for s in SYMBOL_ORDER if s in found] + \
              [s for s in found if s not in SYMBOL_ORDER]

    cards = "".join(build_card(s, found[s], args.dir) for s in ordered)
    ts_now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Alpha Reports</title>"
        f"<style>{CSS}</style></head><body>"
        "<div class='wrap'>"
        "<header><h1>Alpha Reports</h1>"
        f"<div class='meta'>generated: {ts_now}</div></header>"
        "<div class='grid'>"
        f"{cards}"
        "</div>"
        "<footer>Training reports on top &middot; Production reports below &middot; per symbol"
        "</footer></div></body></html>"
    )

    out = os.path.join(args.dir, "index.html")
    with open(out, "w") as f:
        f.write(html)
    n = sum(len(v) for st in found.values() for v in st.values())
    print(f"wrote {out}  ({len(found)} symbol(s), {n} report(s))")

    if not args.no_push:
        # password-protect every page before it is committed (encrypt_pages.py:
        # StatiCrypt-style AES-256-GCM wrapper, idempotent on already-encrypted files)
        import encrypt_pages
        encrypt_pages.encrypt_dir(args.dir)
        msg = args.message or f"publish reports {ts_now}"
        git_publish(args.dir, msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
