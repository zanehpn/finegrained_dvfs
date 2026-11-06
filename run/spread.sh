#!/bin/bash
W=/shared/s3/zqy/dac2027_p1; cd $W
export ON_AGN=1 QWEN_CAL=$W/results/first_order_cal_final.json FREQ_STATES=34 STA_CAT=1 RAILS=4 ON_ALLDOM=1 ON_FAIR=1 ON_COST=worst,mean,aware ON_RLIM_G=32 ON_LAGQ=.25 ON_GUARD=.9
/shared/s3/zqy/DSE/npu_mac/.venv/bin/python scripts/spread_v1.py 12 > logs/spread.log 2>&1
echo done > logs/spread.done
