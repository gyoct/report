#!/bin/bash
cd /home/guanyang/work/report/scripts
for a in "xau 1 10 v3" "xag 1 10 v3" "xau 1 10 v2" "xag 1 10 v2" "btc 1 10" "xau 1 10" "xag 1 10"; do
  python3 gen_alpha_decay.py $a || exit 1
done
cd /home/guanyang/work/alpha_replayer_config/pipeline
python3 pre_training.py run_xau_lasso_ols.json || exit 1
cd /home/guanyang/work/report
rm -f btc_alpha_decay.html xau_alpha_decay.html xag_alpha_decay.html \
      xau_alpha_decay_v2.html xag_alpha_decay_v2.html \
      xau_alpha_decay_v3.html xag_alpha_decay_v3.html xau_pre_training.html
env -u LD_LIBRARY_PATH python3 scripts/make_index.py && echo VNAMING_DONE
