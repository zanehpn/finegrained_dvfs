#!/bin/bash
W=/shared/s3/zqy/dac2027_p1; cd $W
export QWEN_CAL=$W/results/first_order_cal_final.json FREQ_STATES=34 STA_CAT=1 RAILS=4 ON_ALLDOM=1 ON_FAIR=1 ON_COST=aware ON_RLIM_G=32 ON_LAGQ=.25 ON_GUARD=.9 ON_AGN=1 REAL_N=32 REAL_B=.05,.25 REAL_GUARD=.9 PYTHONPATH=$W/scripts REAL_DEST=$W/results/real_b
PY=/shared/s3/zqy/DSE/npu_mac/.venv/bin/python
$PY scripts/real_slo_v1.py 2 q14_ > logs/real_b_q14.log 2>&1
$PY scripts/real_slo_v1.py 24 l8bm_ > logs/real_b_l8bm.log 2>&1
$PY scripts/real_slo_v1.py 24 l8_ > logs/real_b_l8.log 2>&1
$PY scripts/real_slo_v1.py 14 q32_ > logs/real_b_q32.log 2>&1
echo done > logs/real_b.done
