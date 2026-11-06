#!/bin/bash
W=/shared/s3/zqy/dac2027_p1; cd $W
until [ -f logs/real_q14.done ]; do python3 -c "import time;time.sleep(60)"; done
export QWEN_CAL=$W/results/first_order_cal_final.json FREQ_STATES=34 STA_CAT=1 RAILS=4 ON_ALLDOM=1 ON_FAIR=1 ON_COST=aware ON_RLIM_G=32 ON_LAGQ=.25 ON_GUARD=.9 ON_AGN=1 REAL_N=32 REAL_G=.9,.8
PY=/shared/s3/zqy/DSE/npu_mac/.venv/bin/python
for m in l8bm_ l8_ q32_; do $PY scripts/real_trace_v1.py 18 $m > logs/real_$m.log 2>&1; done
echo done > logs/real_rest.done
