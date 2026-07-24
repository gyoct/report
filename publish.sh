#!/bin/bash
# One-command publish: regenerate every per-symbol report, rebuild index.html,
# then git commit + push the report folder. Run:  ./publish.sh
#
# Add a new report by adding a "build_report.py --name X.html" line below;
# make_index.py auto-discovers it into the index, no table edit needed.
set -euo pipefail

# ---- single-instance guard -------------------------------------------------
# This script is on an HOURLY cron but a slow run can exceed the hour (e.g. the
# 2026-07-22 backfill: 77GB of lr_S10 logs after the rsync-glob bug was fixed).
# Two concurrent runs would both read-modify-write analyze_alpha.py's per-product
# state -- data/_resample_cache.parquet (~500MB) and data/_resample_state.json
# (byte offsets) -- which can silently corrupt the incremental history; recovering
# needs a full --rebuild over the raw logs.
#
# flock takes an exclusive kernel lock on fd 9. -n = don't wait: if a previous run
# still holds it, skip this hour entirely (exit 0 so cron sends no error mail). The
# kernel releases the lock when the process exits, including on crash or kill, so no
# stale lockfile can wedge the pipeline.
exec 9>/tmp/publish_reports.lock
flock -n 9 || { echo "publish.sh: previous run still in progress -- skipping this hour"; exit 0; }

REPORT_DIR="/home/guanyang/work/report"
PROD="/home/guanyang/work/CR_TRAINING/PY/prod"

echo "==> syncing prod logs/config from byb_prod"
# Parallelised ACROSS HOSTS ONLY, deliberately: 4 of these pull from the same LIVE
# trading box (byb_prod) and one from bgb_prod. Running all 5 at once would put 5
# concurrent readers on a production trading server's disk/network; keeping each host
# to a single sequential stream bounds that while still overlapping the two hosts.
( bash "$PROD/btc_monetization/rsync_bybprod.sh"
  bash "$PROD/btc_monetization2/rsync_bybprod.sh"
  bash "$PROD/btc_monetization3/rsync_bybprod.sh"
  bash "$PROD/eth/rsync_bybprod.sh" ) & rs_byb=$!
bash "$PROD/bgb/btc_taker/rsync_bybprod.sh" &  rs_bgb=$!     # bgb_prod: separate host
rsync_failed=0
wait "$rs_byb" || rsync_failed=1
wait "$rs_bgb" || rsync_failed=1
[ "$rsync_failed" -eq 0 ] || echo "WARN: an rsync stream failed -- continuing with what we have" >&2

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
# Independent per product (each reads only its own order_* logs and writes only its
# own analysis_out), so run them concurrently and fail the publish if any one fails.
om_products=(btc_monetization2 btc_monetization3 bgb/btc_taker)
om_pids=()
for d in "${om_products[@]}"; do
  Rscript "$PROD/$d/order/order_multi.R" > "$PROD/$d/order_multi.log" 2>&1 &
  om_pids+=("$!:$d")
done
om_failed=0
for pd in "${om_pids[@]}"; do
  pid="${pd%%:*}"; d="${pd##*:}"
  if wait "$pid"; then echo "    [ok]   $d"
  else echo "    [FAIL] $d -- see $PROD/$d/order_multi.log" >&2; om_failed=1; fi
done
[ "$om_failed" -eq 0 ] || { echo "ERROR: order_multi.R failed" >&2; exit 1; }

echo "==> regenerating reports"
# Each build_report.py reads only its own analysis_out and writes its own HTML, so
# these are independent -> run concurrently. (They are memory-hungry: plotly over a
# ~1GB parquet each. Five at once is fine on this 32-core box; cap with a job pool if
# that ever changes.)
declare -A REPORT_OF=(
  [btc_monetization]=btc_prod.html
  [btc_monetization2]=btc_prod2.html
  [btc_monetization3]=btc_prod3.html
  [eth]=eth_prod.html
  [bgb/btc_taker]=bgb_btc.html
)
br_pids=()
for d in "${!REPORT_OF[@]}"; do
  python "$PROD/$d/build_report.py" \
    --analysis_out "$PROD/$d/analysis_out" \
    --out_dir "$REPORT_DIR" --name "${REPORT_OF[$d]}" > "$PROD/$d/build_report.log" 2>&1 &
  br_pids+=("$!:$d")
done
br_failed=0
for pd in "${br_pids[@]}"; do
  pid="${pd%%:*}"; d="${pd##*:}"
  if wait "$pid"; then echo "    [ok]   $d -> ${REPORT_OF[$d]}"
  else echo "    [FAIL] $d -- see $PROD/$d/build_report.log" >&2; br_failed=1; fi
done
[ "$br_failed" -eq 0 ] || { echo "ERROR: build_report.py failed" >&2; exit 1; }

echo "==> building SpreadArb summary (standalone page, kept out of index)"
# Pull bgb logs, rebuild Summary, render spread.html into the report repo.
# Non-fatal: a failure here must NOT block the main reports, so it runs inside an
# `if` (set -e is suppressed for the condition).
if bash /home/guanyang/work/CR_TRAINING/SpreadArb/build_summary.sh; then
  echo "    [ok]   spread.html"
else
  echo "    [WARN] SpreadArb summary failed -- skipping" >&2
fi

echo "==> rebuilding index + publishing"
# make_index.py rebuilds index.html and (unless --no-push) commits + pushes.
python "$REPORT_DIR/make_index.py" --dir "$REPORT_DIR"

echo "==> done"
