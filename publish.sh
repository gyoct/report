#!/bin/bash
# One-command publish: regenerate every per-symbol report, rebuild index.html,
# then git commit + push the report folder. Run:  ./publish.sh
#
# Add a new report by adding a "build_report.py --name X.html" line below;
# make_index.py auto-discovers it into the index, no table edit needed.
set -euo pipefail

REPORT_DIR="/home/guanyang/work/report"
PROD="/home/guanyang/work/CR_TRAINING/PY/prod"

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
