#!/bin/bash
set -e
cd /home/guanyang/work/report/scripts
python3 gen_pipeline_models.py btc
python3 gen_alpha_decay.py btc 1 10
env -u LD_LIBRARY_PATH python3 make_index.py
echo BTC_CAPPED_PAGES_DONE
