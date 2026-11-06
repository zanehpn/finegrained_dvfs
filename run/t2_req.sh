#!/bin/bash
W=/shared/s3/zqy/dac2027_p1; cd $W
export QWEN_CAL=$W/results/first_order_cal_final.json FREQ_STATES=34 STA_CAT=1 RAILS=4 ON_ALLDOM=1 ON_FAIR=1 ON_COST=aware ON_RLIM_G=32 ON_LAGQ=.25 ON_GUARD=.9
/shared/s3/zqy/DSE/npu_mac/.venv/bin/python scripts/t2_request_v1.py 9 > logs/t2_req.log 2>&1
echo done > logs/t2_req.done
