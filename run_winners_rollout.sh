#!/bin/bash
# Full decay-page rollout: winners + open/close labels, then aupy, then publish.
cd /home/guanyang/work/report
for a in "xau 1 10 v3" "xag 1 10 v3" "xau 1 10 v2" "xag 1 10 v2" "btc 1 10" "xau 1 10" "xag 1 10"; do
  python3 scripts/gen_alpha_decay.py $a || exit 1
done
( cd /home/guanyang/work/OnDev/fut-qyas && \
  python3 tools/reports/gen_decay_report.py --out-dir /home/guanyang/work/report ) || exit 1
cd /home/guanyang/work/report
env -u LD_LIBRARY_PATH python3 scripts/make_index.py && echo WINNERS_FINAL_DONE
