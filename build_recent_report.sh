#!/bin/bash
# Rebuild the "since <DATE>" full btc_prod2-style report for a product and publish.
# Usage: build_recent_report.sh <product_dir> <SINCE 'YYYY-MM-DD HH:MM:SS'> <out.html> [--no-publish]
set -euo pipefail
PROD="${1:?product dir}"; SINCE="${2:?since 'YYYY-MM-DD HH:MM:SS'}"; OUT="${3:?out html}"
PUBLISH=1; [ "${4:-}" = "--no-publish" ] && PUBLISH=0
REPORT=/home/guanyang/work/report
REC="$PROD/analysis_out_recent"
mkdir -p "$REC"; ln -sf ../analysis_out/alphapx_per_second.csv "$REC/alphapx_per_second.csv"
echo "==> order_multi.R (since $SINCE)"
ORDER_SINCE_UTC="$SINCE" ORDER_OUT_DIR="$REC" Rscript "$PROD/order/order_multi.R" >/tmp/rec_order.log 2>&1
echo "==> filter analyze artifacts"
python "$REPORT/scripts/filter_analysis_out.py" "$PROD/analysis_out" "${SINCE%% *}" "$REC"
echo "==> build_report -> $OUT"
python "$PROD/build_report.py" --analysis_out "$REC" --out_dir "$REPORT" --name "$OUT"
python3 -c "p='$REPORT/$OUT'; s=open(p).read().replace('analysis_out_recent','since ${SINCE%% *}'); open(p,'w').write(s)"
[ "$PUBLISH" = 1 ] && { echo "==> publish"; (cd "$REPORT" && python scripts/make_index.py >/dev/null && echo published); } || echo "(skipped publish)"
