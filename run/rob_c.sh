#!/bin/bash
# tab:robust and single-rail runs, 4 rails, 34 points.
W=/shared/s3/zqy/dac2027_p1; PY=/shared/s3/zqy/DSE/npu_mac/.venv/bin/python; cd $W
export ON_AGN=1 QWEN_CAL=$W/results/first_order_cal_final.json FREQ_STATES=34 STA_CAT=1 RAILS=4 ON_ALLDOM=1 ON_FAIR=1 ON_COST=worst,mean,aware ON_RLIM_G=32 ON_LAGQ=.25 ON_GUARD=.9
for k in nominal switch_x10 eff_m10 dldo lowv20; do ROB_KNOB=$k $PY scripts/online_robust_v1.py > logs/rob_$k.log 2>&1 & done
ROB_KNOB=no045 FREQ_VMIN=.46 $PY scripts/online_robust_v1.py > logs/rob_no045.log 2>&1 &
ROB_KNOB=no06 FREQ_VMIN=.6 $PY scripts/online_robust_v1.py > logs/rob_no06.log 2>&1 &
wait; echo done > logs/rob_c.done
