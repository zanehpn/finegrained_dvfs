#!/bin/bash
# Pulls each finished online-plan replay from gaia and records its objective use. Replays that died are reported, not restarted. Ends when none is left.
W=/shared/s3/zqy/dac2027_p1; G=/data/zqy/npu_mac/outputs/flexposit_enpu_paper/asap7_calibration; L=$W/logs/queue_c.log
G1="ssh -o BatchMode=yes -o ConnectTimeout=20 -i $HOME/.ssh/id_ed25519_gaia2 -o IdentitiesOnly=yes zqy@gaia1.ddns.comp.nus.edu.sg"
say() { echo "$(date '+%m-%d %T') $*" >> $L; }
todo="c_p1920_nominal_ttft5 c_p3968_nominal_ttft5"; for c in aware mean worst; do for t in 5 25; do for p in 1920 3968; do todo="$todo c_p${p}_${c}_ttft$t"; done; done; done
while [ -n "${todo// /}" ]; do
  next=
  for t in $todo; do
    d=qwen14b_longctx_replay_k256_$t
    if $G1 "test -f $G/$d/result.json" </dev/null; then
      mkdir -p $W/replays_gaia2/$d; rsync -a gaia2:$G/$d/result.json $W/replays_gaia2/$d/ </dev/null && say "$t done: $(python3 $W/scripts/fb_check.py $W/replays_gaia2/$d 2>&1 | tail -1 | cut -c1-60)" || { say "$t pull failed"; next="$next $t"; }
      continue
    fi
    if [ "$($G1 "grep -c 'start $t ' /data/zqy/npu_mac_logs/qc.log; systemctl --user is-active cal_r_$t" </dev/null 2>/dev/null | tr '\n' ' ')" = "1 inactive " ] || [ "$($G1 "grep -c 'start $t ' /data/zqy/npu_mac_logs/qc.log; systemctl --user is-active cal_r_$t" </dev/null 2>/dev/null | tr '\n' ' ')" = "1 failed " ]; then say "$t died, left for review"; continue; fi
    next="$next $t"
  done
  todo=$next; python3 -c 'import time;time.sleep(600)'
done
say "queue empty"
