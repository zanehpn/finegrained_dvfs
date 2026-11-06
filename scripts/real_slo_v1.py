"""Azure traces under a two-branch TTFT limit, so the chip's real limits do not hide the comparison.
Limit per request = max(MLPerf target, TTFT at peak V/f x (1+b)); TPOT stays the absolute target. Peak TTFT is the measured nominal run, the same limit for every costing.
Six windows evenly spaced through each trace (0..2 are real_trace_v1's). Guard g acts on the target and on the allowance b. Result files carry every limit's outcome."""
import json, os, sys, csv
os.environ.setdefault('REAL_DEST', '/shared/s3/zqy/dac2027_p1/results/real_b')
import real_trace_v1 as R
from concurrent.futures import ProcessPoolExecutor, as_completed
E, ON, K, SUP, q, P = R.E, R.ON, R.K, R.SUP, R.q, R.P
BS = tuple(float(x) for x in os.environ.get('REAL_B', '.05,.25').split(',')); G = float(os.environ.get('REAL_GUARD', .9)); DEST = R.DEST
RHOS = (.25, .5, .8); WINS = range(6); TIERS = dict(q14=('server',), q32=('server',), l8=('server', 'interactive'), l8bm=('server', 'interactive'))  # Qwen peak TPOT is above the 40 ms Interactive target, so that tier has no budget to compare
_w0 = P.window


def window(kind, w, n=32, limit=2048):
    if w < 3: return _w0(kind, w, n, limit)
    rows = list(csv.DictReader(open(P.TRACE % kind))); start = len(rows)*(w-3)//3; out = []
    for x in rows[start:]:
        c, g = int(x['ContextTokens']), int(x['GeneratedTokens'])
        if c+g <= limit and g >= 2: out.append((R.dt.datetime.fromisoformat(x['TIMESTAMP'][:26]), c, g))
        if len(out) == n: break
    return out
P.window = window


def job(c):
    m, kind, w, rho, tier = c; name = f'{m}_{kind}{w}_r{rho}_{tier}'; f = DEST/f'{name}.json'
    if f.exists(): return name, 'skip'
    outs, arr = R.install(m, kind, w, rho); T, Td = R.TIER[m][tier]
    def per(lat):
        o = []; i = 0
        for n in outs: o.append((lat[i], sum(lat[i+1:i+n])/max(n-1, 1))); i += n
        return o
    q.MEM_MODE = 'cg'; s0 = q.run(K, {}, SUP, False); Eref, comp, _ = q.price(K, SUP, {}, s0, pg=True); E0 = q.price(K, SUP, {}, s0, pg=False, mem_dvfs=False)[0]; q.MEM_MODE = 'dvfs'
    R0 = per(s0[3]); bg = E.IP._bg_power(comp, s0[0]); n = len(R0); tp0 = [x[1] <= Td+1 for x in R0]
    out = dict(model=m, kind=kind, w=w, rho=rho, tier=tier, T=T, Td=Td, requests=n, tokens=sum(outs), makespan_s=s0[0]*1e-12, E_pgcg=Eref, E_nodvfs=E0, pgcg_vs_nodvfs=100*(1-Eref/E0),
               nominal=dict(ttft_abs=sum(x[0] <= T+1 for x in R0), tpot_met=sum(tp0), ttft_p50=sorted(x[0] for x in R0)[n//2]*1e-12, tpot_p50=sorted(x[1] for x in R0)[n//2]*1e-9), arms={})
    for cost in ('worst', 'mean', 'aware'):
        q.MEM_MODE = 'cg'; own = ON.own_twin(K, SUP, *E.FIELDS[cost], per, n, arr); q.MEM_MODE = 'dvfs'
        for b in BS:
            lim = [max(T, x[0]*(1+b)) for x in R0]; slo = dict(ttft=[max(T*G, x[0]*(1+G*b)) for x in R0], tpot=[Td*G]*n)
            A, sc, info = ON.run_online(K, SUP, 0., 0., 0., 1., bg, *E.FIELDS[cost], lag_q=E.LQ, hedge=E.HEDGE, warm=8, nom=None, slo=slo, own=own); Rr = per(sc[3]); Ex = ON.energy(K, SUP, A, sc)
            tf = [r[0] <= l+1 for r, l in zip(Rr, lim)]; tp = [r[1] <= Td+1 for r in Rr]
            out['arms'][f'{cost}_b{b}'] = dict(E=Ex, saving_pgcg=100*(1-Ex/Eref), saving_nodvfs=100*(1-Ex/E0), ttft_miss=n-sum(tf), tpot_added=sum(a and not c_ for a, c_ in zip(tp0, tp)), relaxed=sum(x[0]*(1+b) > T for x in R0), ttft_abs=sum(r[0] <= T+1 for r in Rr),
                                               over_max_pct=max(100*(r[0]/l-1) for r, l in zip(Rr, lim)), stretch_max_pct=max(100*(r[0]/y[0]-1) for r, y in zip(Rr, R0)), decisions=info['decisions'])
    DEST.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(out, indent=1)+'\n')
    return name, n, out['nominal']['ttft_abs'], {k: (round(v['saving_pgcg'], 2), v['ttft_miss'], v['tpot_added']) for k, v in out['arms'].items()}


def cases():
    return [(m, kind, w, rho, t) for m in ('q14', 'l8bm', 'l8', 'q32') for kind in ('conv', 'code') for w in WINS for rho in RHOS for t in TIERS[m]]


def main():
    DEST.mkdir(parents=True, exist_ok=True); js = [c for c in cases() if R.FILT in f'{c[0]}_{c[1]}{c[2]}_r{c[3]}_{c[4]}']; print(len(js), 'cases', flush=True)
    with ProcessPoolExecutor(min(R.W, len(js)), max_tasks_per_child=1, initializer=E.limit_memory) as ex:
        for m in ('q14', 'l8bm', 'l8', 'q32'):
            for t in ('server', 'interactive'):
                fs = {ex.submit(job, c): c for c in js if c[0] == m and c[4] == t}
                for x in as_completed(fs):
                    try: print(x.result(), flush=True)
                    except Exception as e: print('FAILED', fs[x], repr(e)[:400], flush=True)


if __name__ == '__main__': main()
