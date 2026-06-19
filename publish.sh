#!/bin/bash
# One-command publish: regenerate every per-symbol report, rebuild index.html,
# then git commit + push the report folder. Run:  ./publish.sh
#
# Add a new report by adding a "build_report.py --name X.html" line below;
# make_index.py auto-discovers it into the index, no table edit needed.
set -euo pipefail

REPORT_DIR="/home/guanyang/work/report"
PROD="/home/guanyang/work/CR_TRAINING/PY/prod"

echo "==> syncing prod logs/config from byb_prod"
bash "$PROD/btc_monetization/rsync_bybprod.sh"
bash "$PROD/btc_monetization2/rsync_bybprod.sh"
bash "$PROD/btc_monetization3/rsync_bybprod.sh"
bash "$PROD/eth/rsync_bybprod.sh"
bash "$PROD/bgb/btc_taker/rsync_bybprod.sh"   # bgb pulls from bgb_prod

echo "==> analyzing raw logs -> analysis_out (3 parallel processes)"
# analyze_alpha.py resumes from a byte-offset cache (only re-reads new bytes).
# Self-heal against drift / schema change with a full --rebuild, but STAGGER it:
# at 00:00 UTC rebuild exactly ONE product (rotating by day-of-year, so each is
# rebuilt every 3 days); the other two stay incremental. Avoids three concurrent
# ~100GB re-reads in the same hour. 10# forces base-10 (%j is zero-padded -> the
# 08/09 days would otherwise be read as invalid octal).
products=(btc_monetization btc_monetization2 btc_monetization3 eth bgb/btc_taker)
rebuild_today=""
if [ "$(date -u +%H)" = "00" ]; then
  rebuild_today="${products[$(( 10#$(date -u +%j) % ${#products[@]} ))]}"
  echo "    00:00 UTC -> full rebuild for: $rebuild_today (others incremental)"
fi
# The three folders are independent, so analyze them concurrently. Each logs to
# its own file; we wait for all and fail the publish if any one fails.
pids=()
for d in "${products[@]}"; do
  flag=""
  [ "$d" = "$rebuild_today" ] && flag="--rebuild"
  python "$PROD/$d/analyze_alpha.py" $flag > "$PROD/$d/analyze.log" 2>&1 &
  pids+=("$!:$d")
done
analyze_failed=0
for pd in "${pids[@]}"; do
  pid="${pd%%:*}"; d="${pd##*:}"
  if wait "$pid"; then
    echo "    [ok]   $d"
  else
    echo "    [FAIL] $d -- see $PROD/$d/analyze.log" >&2
    analyze_failed=1
  fi
done
[ "$analyze_failed" -eq 0 ] || { echo "ERROR: analyze_alpha.py failed" >&2; exit 1; }

echo "==> building order summaries (order_multi.R)"
# Reconstructs per-order state from the raw order logs and writes
# order_summary.csv / order_hourly.csv into each analysis_out, which
# build_report.py picks up for the Order summary section.
Rscript "$PROD/btc_monetization2/order/order_multi.R"
Rscript "$PROD/btc_monetization3/order/order_multi.R"
Rscript "$PROD/bgb/btc_taker/order/order_multi.R"

echo "==> regenerating reports"
python "$PROD/btc_monetization/build_report.py" \
  --analysis_out "$PROD/btc_monetization/analysis_out" \
  --out_dir "$REPORT_DIR" --name btc_prod.html

python "$PROD/btc_monetization2/build_report.py" \
  --analysis_out "$PROD/btc_monetization2/analysis_out" \
  --out_dir "$REPORT_DIR" --name btc_prod2.html

python "$PROD/btc_monetization3/build_report.py" \
  --analysis_out "$PROD/btc_monetization3/analysis_out" \
  --out_dir "$REPORT_DIR" --name btc_prod3.html

python "$PROD/eth/build_report.py" \
  --analysis_out "$PROD/eth/analysis_out" \
  --out_dir "$REPORT_DIR" --name eth_prod.html

python "$PROD/bgb/btc_taker/build_report.py" \
  --analysis_out "$PROD/bgb/btc_taker/analysis_out" \
  --out_dir "$REPORT_DIR" --name bgb_btc.html

echo "==> building SpreadArb summary (standalone page, kept out of index)"
# Pull bgb logs, rebuild Summary, render summary.html into the report repo.
# Non-fatal: a failure here must NOT block the main reports, so it runs inside an
# `if` (set -e is suppressed for the condition).
if bash /home/guanyang/work/CR_TRAINING/SpreadArb/build_summary.sh; then
  echo "    [ok]   summary.html"
else
  echo "    [WARN] SpreadArb summary failed -- skipping" >&2
fi

echo "==> rebuilding index + publishing"
# make_index.py rebuilds index.html and (unless --no-push) commits + pushes.
python "$REPORT_DIR/make_index.py" --dir "$REPORT_DIR"

echo "==> done"
