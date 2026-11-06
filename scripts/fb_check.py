"""Print replay use of the objectives: fb_check.py <replay_dir> [nominal_dir] -> max dTTFT%, max dTPOT% against the nominal replay.
Without nominal_dir the reference follows the workload: p3968 replays use the 4K nominal, others the 2K one."""
import json, sys
from pathlib import Path
N2K = '/shared/s3/zqy/DSE/npu_mac/outputs/flexposit_enpu_paper/asap7_calibration/qwen14b_longctx_p1920_o128_b55_replay_k256_nominal'
N4K = '/shared/s3/zqy/dac2027_p1/replays_gaia2/qwen14b_longctx_replay_k256_p3968_nominal'
N2Kc = '/shared/s3/zqy/dac2027_p1/replays_gaia2/qwen14b_longctx_replay_k256_c_p1920_nominal_ttft5'; N4Kc = '/shared/s3/zqy/dac2027_p1/replays_gaia2/qwen14b_longctx_replay_k256_c_p3968_nominal_ttft5'
def pr(f):
    out = []
    for r in sorted(json.loads(Path(f).read_text())['requests'], key=lambda r: r['request']):
        t = r['emitted_times_ps']; out.append((t[0]-(r.get('arrival_ps') or 0), (t[-1]-t[0])/(len(t)-1)))
    return out
d = sys.argv[1].rstrip('/'); N = sys.argv[2] if len(sys.argv) > 2 else ((N4Kc if 'p3968' in d else N2Kc) if '/qwen14b_longctx_replay_k256_c_' in d or d.endswith('_c') else (N4K if 'p3968' in d else N2K))
a, b = pr(d+'/result.json'), pr(N+'/result.json')
print(max(100*(x[0]/y[0]-1) for x, y in zip(a, b)), max(100*(x[1]/y[1]-1) for x, y in zip(a, b)))
