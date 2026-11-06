#!/bin/bash
# 4-rail replays: nominal first, then aware, mean, worst. Same gates as before: slice+cap <= 800G, host keeps 400G. 4K arms 30 min apart, 2K 10 min. No retries.
L=/data/zqy/npu_mac_logs; C=outputs/flexposit_enpu_paper/asap7_calibration; A=$C/qwen14b_longctx_assignments_ip1
S=/sys/fs/cgroup/user.slice/user-2013.slice/user@2013.service/zqyjobs.slice/memory.current
log() { echo "$(date "+%m-%d %T") $*" >> $L/qc.log; }
go() { # cost t p
  c=$1; t=$2; p=$3
  if [ $p = 3968 ]; then cap=200; gap=1800; wl="WORKLOAD=$C/qwen14b_wk4_w4k"; else cap=88; gap=600; wl=; fi
  tag=c_p${p}_${c}_ttft$t; f=b5.5_fine_${c}_ttft${t}_tpot5_self_online_p$p.json
  while :; do s=$(($(cat $S)/1073741824)); av=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo); [ $((s+cap)) -le 800 ] && [ "$av" -gt 400 ] && break; python3 -c "import time;time.sleep(60)"; done
  env $wl $L/run_replay_g1.sh $tag $cap 16 --assignment $A/$f >/dev/null 2>&1; log "start $tag realG=$s av=$av rc=$?"
  python3 -c "import time;time.sleep($gap)"
}
go nominal 5 1920; go nominal 5 3968
for c in aware mean worst; do for t in 5 25; do for p in 1920 3968; do go $c $t $p; done; done; done
log "queue empty"
