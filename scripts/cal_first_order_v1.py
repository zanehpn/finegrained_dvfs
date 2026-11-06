"""Fit the first-order timing correction on the nominal replay only, then hold it out on every DVFS replay.
usage: cal_first_order_v1.py fit|holdout [cal.json]
Fit: coordinate search over (a, x) per (engine, family), minimising per-family busy-time error of q.run against the nominal timeline.
Hold-out: per-request TTFT/mean TPOT and makespan of each arm, predicted with the frozen file."""
import collections, json, os, re, sys
from pathlib import Path
sys.dont_write_bytecode = True
W = Path('/shared/s3/zqy/dac2027_p1'); MODE = sys.argv[1]; CAL = sys.argv[2] if len(sys.argv) > 2 else str(W/'results/first_order_cal_v1.json')
if MODE == 'holdout': os.environ['QWEN_CAL'] = CAL
sys.argv = ['x']; sys.path.insert(0, str(W/'scripts'))
import energy_from_qwen_replay_v1 as X
q = X.q; rid = X.load_work('b5.5'); NOM = X.O/'qwen14b_longctx_p1920_o128_b55_replay_k256_nominal'; A = X.O/'qwen14b_longctx_assignments_ip1'


def busy_by_key(run_dur=None, tl=None):
    b = collections.Counter()
    if tl is not None:
        for x in tl:
            i = rid.get(x['id'])
            if i is not None: o = q.OPS[i]; b[f"{o['engine']}/{o['family']}"] += x['end_ps']-x['start_ps']
    else:
        for i, (c, t) in run_dur.items(): o = q.OPS[i]; b[f"{o['engine']}/{o['family']}"] += t
    return b


def fit():
    res = json.loads((NOM/'scheduler_result.json').read_text()); tl = json.loads((NOM/'timeline.json').read_text())
    tgt = busy_by_key(tl=tl); lat = X.latencies(res['requests']); pr = X.per_req(lat)
    keys = [k for k in tgt if tgt[k] > 5e10 and not k.endswith('state')]; print({k: round(v*1e-12, 3) for k, v in tgt.items()}, flush=True)
    cal = {k: dict(a=0., x=0.) for k in keys}; q.MEM_MODE = 'cg'; q.CAL = cal
    def loss():
        s = q.run(X.K, {}, X.SUP, False); b = busy_by_key(run_dur=q.DUR); got = X.per_req(s[3])
        return sum((b[k]/tgt[k]-1)**2 for k in keys)+sum((a[0]/c[0]-1)**2+(a[1]/c[1]-1)**2 for a, c in zip(got, pr))+(s[0]/res['makespan_ps']-1)**2
    best = loss(); print('start', best, flush=True)
    for step in (.4, .2, .1, .05, .025):
        for _ in range(3):
            for k in keys:
                for f, lo, hi in (('a', 0., 1.), ('x', 0., 3.)):
                    for d in (step, -step):
                        v = cal[k][f]; cal[k][f] = min(hi, max(lo, v+d)); l = loss()
                        if l < best-1e-7: best = l
                        else: cal[k][f] = v
        print(step, round(best, 6), json.dumps(cal), flush=True)
    Path(CAL).write_text(json.dumps(cal, indent=1)+'\n')


def holdout():
    print('cal', json.dumps(q.CAL)); q.MEM_MODE = 'cg'; s0 = q.run(X.K, {}, X.SUP, False); q.MEM_MODE = 'dvfs'; R0 = X.per_req(s0[3])
    res0 = json.loads((NOM/'result.json').read_text()); P0 = X.per_req(X.latencies(res0['requests']))
    fmt = lambda a, b: f"{100*(a/b-1):+.1f}"
    print('nominal T err', fmt(s0[0], res0['makespan_ps']), 'TTFT', [fmt(a[0], b[0]) for a, b in zip(R0, P0)], 'TPOT', [fmt(a[1], b[1]) for a, b in zip(R0, P0)], flush=True)
    errs = []
    for d in sorted((W/'replays_gaia2').iterdir()):
        arm = d.name.split('k256_')[1].rsplit('_', 1)[0]
        f = W/'assignments_repair'/f'{arm}.json'; f = f if f.exists() else A/f'{arm}.json'; asg = json.loads(f.read_text())
        assign = {rid[n]: p for n, p in asg['node_points'].items() if n in rid}
        s = q.run(X.K, assign, X.SUP, asg['coarse']); R = X.per_req(s[3]); res = json.loads((d/'result.json').read_text()); P = X.per_req(X.latencies(res['requests']))
        e = [100*(a[j]/b[j]-1) for a, b in zip(R, P) for j in (0, 1)]; dp = [100*(a[1]/b[1]-1) for a, b in zip(P, P0)]; df = [100*(a[1]/b[1]-1) for a, b in zip(R, R0)]
        errs += [abs(x) for x in e]
        print(f"{arm[5:-11]:26s} T {fmt(s[0], res['makespan_ps'])}% lat err {min(e):+.1f}..{max(e):+.1f}%  dTPOT replay {max(dp):+.1f} pred {max(df):+.1f}  dTTFT replay {max(100*(a[0]/b[0]-1) for a, b in zip(P, P0)):+.1f} pred {max(100*(a[0]/b[0]-1) for a, b in zip(R, R0)):+.1f}", flush=True)
    print('max abs latency err', max(errs), 'mean', sum(errs)/len(errs))


{'fit': fit, 'holdout': holdout}[MODE]()
