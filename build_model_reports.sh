#!/bin/bash
# Regenerate the per-symbol MODEL reports (Linear/OLS+Lasso vs LGBM) and publish
# the report site. This is the REPORT-BUILD half only -- it consumes the model
# statistics already written by the trainers under
#   alpha_replayer_config/statistics/<sym>/{linear,lgbm}/
# (retraining is a separate, upstream step -- see doc/model_report_process.md).
#
# Usage:
#   build_model_reports.sh                 # btc eth sol, no hit-rate rebuild, publish
#   build_model_reports.sh --augment       # first rebuild the 3 hit-rate columns
#   build_model_reports.sh --no-publish     # build HTML only, skip encrypt/push
#   build_model_reports.sh btc             # a subset of symbols
#   build_model_reports.sh --augment eth sol
set -euo pipefail

REPORT_DIR="/home/guanyang/work/report"
cd "$REPORT_DIR"

AUGMENT=0
PUBLISH=1
SYMS=()
for a in "$@"; do
  case "$a" in
    --augment)    AUGMENT=1 ;;
    --no-publish) PUBLISH=0 ;;
    -*)           echo "unknown flag: $a" >&2; exit 2 ;;
    *)            SYMS+=("$a") ;;
  esac
done
[ ${#SYMS[@]} -eq 0 ] && SYMS=(btc eth sol)

echo "==> symbols: ${SYMS[*]}   augment=$AUGMENT   publish=$PUBLISH"

# 1. (optional) rebuild ic_by_pred_quantile_*.csv with the 3 hit-rate variants
#    (hit_rate / hit_rate_with_zero / hit_rate_move) from X_dt_valid. Slow (reads
#    the multi-GB validation matrices). LEGACY: pred_diagnostics.{py,R} now emit all three
#    natively (patched 2026-08-06), so this is only needed for statistics trained before then.
if [ "$AUGMENT" -eq 1 ]; then
  echo "==> augmenting hit-rate columns (this reads the validation matrices, ~minutes)"
  python "$REPORT_DIR/augment_hitrates.py"
fi

# 2. render one self-contained HTML per symbol -> <sym>_model.html
for s in "${SYMS[@]}"; do
  python "$REPORT_DIR/gen_model_report.py" "$s"
done

# 3. rebuild index.html, encrypt changed pages, force-push the snapshot
if [ "$PUBLISH" -eq 1 ]; then
  echo "==> publishing"
  python "$REPORT_DIR/make_index.py"
else
  echo "==> --no-publish: skipping make_index.py (run it later to push)"
fi
echo "==> done"
