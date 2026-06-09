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
bash "$PROD/eth/rsync_bybprod.sh"

echo "==> analyzing raw logs -> analysis_out (3 parallel processes)"
# The three folders are independent, so analyze them concurrently. Each logs to
# its own file; we wait for all and fail the publish if any one fails.
pids=()
for d in btc_monetization btc_monetization2 eth; do
  python "$PROD/$d/analyze_alpha.py" > "$PROD/$d/analyze.log" 2>&1 &
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

echo "==> building btc_monetization2 order summary (order_multi.R)"
# Reconstructs per-order state from the raw order logs and writes
# order_summary.csv / order_hourly.csv into btc_monetization2/analysis_out,
# which build_report.py picks up for the Order summary section.
Rscript "$PROD/btc_monetization2/order/order_multi.R"

echo "==> regenerating reports"
python "$PROD/btc_monetization/build_report.py" \
  --analysis_out "$PROD/btc_monetization/analysis_out" \
  --out_dir "$REPORT_DIR" --name btc_prod.html

python "$PROD/btc_monetization2/build_report.py" \
  --analysis_out "$PROD/btc_monetization2/analysis_out" \
  --out_dir "$REPORT_DIR" --name btc_prod2.html

python "$PROD/eth/build_report.py" \
  --analysis_out "$PROD/eth/analysis_out" \
  --out_dir "$REPORT_DIR" --name eth_prod.html

echo "==> rebuilding index + publishing"
# make_index.py rebuilds index.html and (unless --no-push) commits + pushes.
python "$REPORT_DIR/make_index.py" --dir "$REPORT_DIR"

echo "==> done"
