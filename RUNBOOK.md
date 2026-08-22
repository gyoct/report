# Report runbook — what to run, what it needs, what it publishes

Three stages: **FEATURES → MODELS → PRODUCTION**. Every stage ends with
`make_index.py`, which rebuilds the index, encrypts the pages and force-pushes
the published snapshot.

Published pages are **encrypted in place** — never `grep` a published file to
check content. Verify with `python3 make_index.py --no-push` (plaintext index)
or regenerate the page and inspect it before publishing.

## AUPY — fut-qyas Python feature and model pages

These pages do not use the crypto alpha-replayer pipeline. They are generated
from the `fut-qyas/output/au0001_panel_*.parquet` panels, its daily IC CSV, and
machine-readable walk-forward model artifacts. Build the artifacts first:

```bash
cd /home/guanyang/work/OnDev/fut-qyas
python fzoo/DemoAlpha/predict_test.py --h 10 --folds \
  --min-train-days 20 --test-block-days 10 \
  --save-dir output/au0001_predict_test_10s_walk_forward

python tools/reports/gen_feature_report.py --out-dir /home/guanyang/work/report
python tools/reports/gen_model_report.py --out-dir /home/guanyang/work/report
# -> aupy_feature_distribution.html
# -> aupy_model.html

cd /home/guanyang/work/report
python3 make_index.py --no-push      # local plaintext index check
```

The distribution report uses a deterministic 1-in-100 snapshot sample for
moments, quantiles, xσ and kurtosis; IC values use the stored daily evaluation.
Its model-input flags follow the shared rule: xσ>30 or |kurtosis|>75 requires
rank-gauss, while xσ>15 or |kurtosis|>25 requires review. The model page reads
the saved OOS parquet/CSV/JSON and computes per-fold stability, conviction
deciles, FR, IC·σ(alpha), P/N, three hit-rate definitions, execution-cost
tables and feature importance. `AUPY` is intentionally separate from `XAU`:
the former is the SHFE `au0001` Python zoo, while the latter is the stock-perp
feature pipeline.

---

## 1. FEATURES — distribution & IC reports

Per-dump statistics: distribution shape, xσ/kurtosis (MLP-scaler hazard), IC by
horizon, data-coverage audit, replay-log review.

**Prerequisites**

| file | produced by |
|---|---|
| `/mnt/v0a2d/ghu/output/<sym>/20260602/<sym>_<ver>_<ukey>_<window>.csv` (crypto) — **xau/xag live under `/mnt/nvme2/ghu/output/` since 2026-08-14** | alpha_replayer (`alpha_replayer_config/<sym>/20260602/run_alpha_*.sh`) |
| `alpha_replayer_config/<sym>/20260602/alpha_offline_cross_alpha_parallel_*.json` or `crypto_alpha/config/alpha_online_research_<sym>.json` | the config that produced the dump — sources the per-feature `result_mode` column |
| `alpha_replayer/output/alpha_replayer_<sym>_*.log` | the same replay — auto-matched for the REPLAY LOG REVIEW section |

**Run**

```bash
cd /home/guanyang/work/CR_TRAINING/PY
python3 feature_distribution.py \
    /mnt/v0a2d/ghu/output/btc/20260602/btc_v6_110200172_202601010000_202605202359.csv \
    --config /home/guanyang/work/alpha_replayer_config/btc/20260602/alpha_offline_cross_alpha_parallel_20260728.json
# useful flags: --no-ic  --horizons 1 10 60  --gap-min 60  --log <explicit log>  --no-log-review

cd /home/guanyang/work/report
python3 gen_feature_dist.py btc v6      # <sym> <ver>; writes btc_feature_distribution_v6.html
python3 make_index.py                   # index + encrypt + publish
```

**Outputs** — `<stem>_distribution.txt` (human table + DATA COVERAGE + DAILY
COVERAGE + REPLAY LOG REVIEW + MLP INPUT TRANSFORM), `<stem>_distribution.csv`
(machine-readable — **this feeds model feature selection**), `<stem>_gaps.csv`,
`<stem>_daily.csv`; page `<sym>_feature_distribution[_<ver>].html`.

---

## 2. MODELS — training + model report

### 2a. Returns (targets) — only if missing for the symbol

```bash
cd /home/guanyang/work/alpha_replayer/output
export LD_LIBRARY_PATH=/home/guanyang/work/alpha_replayer/build/lib
/home/guanyang/work/alpha_replayer/build/bin/alpha_replayer \
    /home/guanyang/work/alpha_replayer_config/<sym>/20260602/alpha_offline_ret.json
# -> /mnt/v0a2d/ghu/output/<sym>/20260602/smid/ret_<ukey>_<window>_0_{1,5,10,30,60}.csv
```

### 2b. Train (unified pipeline — one code path, model is a config key)

**Prerequisites**: the feature dump, the `ret_*` CSVs above, and the feature
**distribution CSV** from stage 1 (drives auto feature selection).

```bash
cd /home/guanyang/work/alpha_replayer_config/pipeline
python3 train_pipeline.py run_btc_lasso_ols.json          # linear: weighted LASSO -> WLS refit
python3 train_pipeline.py run_btc_lgbm.json               # LightGBM (+ gain_prune)
python3 train_pipeline.py config_btc_mlp.json             # torch MLP
python3 train_pipeline.py run_btc_lgbm.json --tune        # + Optuna (inner TRAIN holdout)
python3 train_pipeline.py cfg.json --model lgbm --horizons 1 10
```

Config blocks: `data` (loader + paths), `selection` (distribution_csv, corr
prune), `normalization` (xσ/kurt thresholds, zcap), `split` (explicit periods or
`train_frac`), `cost` (`sample_weight` / `custom_module`), `model_params`,
`tune`. Full stage docs: `alpha_replayer_config/pipeline/README.md`.

**Per-run outputs** (`statistics/<sym>/<model>/20260602/<generation>/`):
`<model>_horizon_summary.csv`, `valid_pred_ret_<h>.csv`, `ic_by_conviction_ret_<h>.csv`,
`ols_coefficients_ret_<h>.csv` / `lgbm_importance_ret_<h>.csv`,
**`feature_caps_ret_<h>.csv` + `normalizer_ret_<h>.json` + `calibration_ret_<h>.json`
(SHIP THESE TO PROD — production must apply the same caps/transforms AND the
prediction-calibration multipliers; never pair with another run's model)**,
`linear_model_ret_<h>.json` (linear runs — C++-servable: intercept + coefs +
normalization data for `LinearAutoAlpha`; see doc/models.md "Automated C++
serving"), `<model>_Y_hat.csv` (feature-dump format, CALIBRATED units since 2026-08-16;
`valid_pred_ret_<h>.csv` keeps the raw prediction as `Y_hat_<h>_raw`),
`selection_report.csv`, `needs_further_normalization.csv`.

### 2c. Model page (single generator, every symbol)

```bash
cd /home/guanyang/work/report
./refresh_models.sh                     # btc xau xag: BOTH pages each + publish
./refresh_models.sh xau                 # one symbol
# equivalent long form (keep the two views in step — same artifacts):
python3 gen_pipeline_models.py btc      # -> btc_model.html
python3 gen_alpha_decay.py btc 1 10     # -> btc_alpha_decay.html
python3 gen_model_lifecycle.py btc btc_monetization2   # research->prod bridge
env -u LD_LIBRARY_PATH python3 make_index.py
```

Adding a tuned generation = one line in `gen_pipeline_models.MODELS` **and** one
in `gen_alpha_decay.MODELS` (both are `(key, label, ..., generation)` lists),
then `./refresh_models.sh`.

PUBLISH GOTCHAS: run make_index with `env -u LD_LIBRARY_PATH` — a leaked
replayer `LD_LIBRARY_PATH` breaks git's OpenSSL; and files > 100 MB are
auto-excluded from the snapshot (GitHub's pre-receive hard limit) with a
printed warning.

`gen_pipeline_models.py` reads ONLY pipeline artifacts and renders: model
comparison table, IC-by-conviction ladders, per-horizon summary per model,
IC-by-|pred|-quantile performance tables (full 15-column legacy schema),
top features + predicted-vs-realized per horizon, full metric matrix.
Which model columns appear is the `MODELS` list at the top of the file
(key, label, colour, model-dir, generation) — add a row to surface a new
tuned generation.

`gen_alpha_decay.py` reads `valid_pred_ret_<h>.csv` and renders virtual-trade
markout (1–300 s) + holding-time vs realized PnL per |pred|-quantile threshold
(q0.95 → q0.999), prod-aligned formulas (1 s AlphaPx grid + LOCF, 2 s staleness
tolerance, simple returns).

### 2d. Signal combination (models × horizons → one signal)

**Prerequisites**: `valid_pred_ret_<h>.csv` for every model generation in
`combine_signals.MODELS` (lasso_ols / lgbm / lgbm_tuned / mlp).

```bash
cd /home/guanyang/work/alpha_replayer_config/pipeline
python3 combine_signals.py btc          # -> statistics/btc/combo/20260602/pipeline_combo/
                                        #    combo_Y_hat.csv, valid_pred_ret_<h>.csv, combine_report.json
cd /home/guanyang/work/report
python3 gen_combo_report.py btc         # -> btc_signal_combo.html
```

Methodology (nested A/B validation split, ensemble worth-it test, the four
horizon-combination methods, FR rescaling) is documented step by step in
`combine_signals.py`'s module docstring. The combo ships in feature-dump
format, so `gen_alpha_decay.py` renders it as just another model
("Combined (horizon stack)"). `refresh_models.sh` reruns `gen_combo_report.py`
(cheap, reads the JSON) but NOT `combine_signals.py` — rerun that manually
after retraining.

*Legacy, superseded:* `gen_model_report.py` + `augment_hitrates.py` (R-trainer
artifacts) — kept only for eth/sol until their pipeline runs land.

---

## 3. PRODUCTION — live monitoring

**Prerequisites**: the live logs pulled from the prod box, plus the running
config (`config/alpha_online.json` — feature coefs for the alpha reconstruction).

```bash
P=/home/guanyang/work/CR_TRAINING/PY/prod/btc_monetization2   # or btc_monetization3, bgb/btc_taker

# 1. pull live data (lr_S10_* alpha logs + order_* fill logs -> $P/data/)
bash $P/rsync_bybprod.sh

# 2. alpha analysis: 1s resample, fwd returns, alpha = sum(col_i * coef_i),
#    IC by hour / by |alpha| quantile, contribution series
python3 $P/analyze_alpha.py          # -> $P/analysis_out/*.csv, alphapx_per_second.csv

# 3. order/fill analysis: PnL, FIFO round trips, markout vs AlphaPx
Rscript $P/order/order_multi.R       # -> order_summary/hourly, holding_pnl.csv, trade_markout.csv
python3 $P/order/markout.py          # per-fill markout CSV (same formulas)

# 4. render + publish
python3 $P/build_report.py --analysis_out $P/analysis_out \
        --out_dir /home/guanyang/work/report --name btc_prod2.html
cd /home/guanyang/work/report && python3 make_index.py
```

Report sections: IC by hour / by |alpha| quantile, feature-correlation heatmap,
return & alpha distributions, contribution timeseries, PnL curve, position vs
mid, **trade markout (1 s–5 min after fill)**, **holding time vs realized PnL
(FIFO round trips)**. The research twins of the last two live on
`<sym>_alpha_decay.html` (stage 2c) — same formulas, virtual trades.

---

## Publishing (all stages)

```bash
cd /home/guanyang/work/report
python3 make_index.py            # rebuild index, encrypt, force-push
python3 make_index.py --no-push  # plaintext build for verification only
```

Index cards come from `PAGES` in `make_index.py` (explicit `<file> -> (symbol,
stage, label)`), with a fallback classifier for `<sym>_feature_distribution*`.
Add new pages there; stages are `Features`, `Model`, `Production`.
