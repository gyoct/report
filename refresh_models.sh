#!/bin/bash
# Refresh the MODEL pages and their ALPHA-DECAY twins together — the two views
# read the same artifacts (gen_pipeline_models.MODELS and gen_alpha_decay.MODELS
# are kept in step), so refreshing one without the other leaves the site
# inconsistent. Usage: ./refresh_models.sh [sym ...]   (default: btc xau xag)
set -e
cd "$(dirname "$0")"
SYMS=${@:-"btc xau xag"}
for s in $SYMS; do
  python3 scripts/gen_pipeline_models.py "$s"
  python3 scripts/gen_alpha_decay.py "$s" 1 10
  # signal-combination page (reads combine_report.json — rerun
  # pipeline/combine_signals.py first if the model artifacts changed)
  python3 scripts/gen_combo_report.py "$s" || echo "WARN: no combo artifacts for $s"
done
env -u LD_LIBRARY_PATH python3 scripts/make_index.py
