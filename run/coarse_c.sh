#!/bin/bash
W=/shared/s3/zqy/dac2027_p1; PY=/shared/s3/zqy/DSE/npu_mac/.venv/bin/python; cd $W
export QWEN_CAL=$W/results/first_order_cal_final.json FREQ_STATES=34 STA_CAT=1 RAILS=4 ON_ALLDOM=1 CAL_DEST=$W/results/abl_c
$PY scripts/coarse_granularity_v1.py main batch > logs/coarse_c.log 2>&1
echo done > logs/coarse_c.done
