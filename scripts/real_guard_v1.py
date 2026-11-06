"""Saving against miss rate on the Azure traces: sweep the TTFT guard per costing, one fixed guard for all traces, so costings compare at equal misses.
Guard g acts on the target and on the allowance: target = max(g T, R (1+g b)). TPOT guard stays 0.9; misses are on the limit of real_slo_v1. Same g on every trace, no per-trace tuning."""
import json, os
os.environ['REAL_DEST'] = os.environ.get('REAL_DEST_G', '/shared/s3/zqy/dac2027_p1/results/real_g')
import real_slo_v1 as S
from concurrent.futures import ProcessPoolExecutor, as_completed
R, E, ON, K, SUP, q = S.R, S.E, S.ON, S.K, S.SUP, S.q; DEST = S.DEST; GS = tuple(float(x) for x in os.environ.get('REAL_GS', '1,.8,.6,.4,.2,0').split(','))


def job(c):
    m, kind, w, rho, tier = c; name = f'{m}_{kind}{w}_r{rho}_{tier}'; f = DEST/f'{name}.json'
    if f.exists(): return name, 'skip'
    outs, arr = R.install(m, kind, w, rho); T, Td = R.TIER[m][tier]
    def per(lat):
        o = []; i = 0
        for n in outs: o.append((lat[i], sum(lat[i+1:i+n])/max(n-1, 1))); i += n
        return o
    q.MEM_MODE = 'cg'; s0 = q.run(K, {}, SUP, False); Eref, comp, _ = q.price(K, SUP, {}, s0, pg=True); E0 = q.price(K, SUP, {}, s0, pg=False, mem_dvfs=False)[0]; q.MEM_MODE = 'dvfs'
    R0 = per(s0[3]); bg = E.IP._bg_power(comp, s0[0]); n = len(R0); tp0 = [x[1] <= Td+1 for x in R0]; out = dict(requests=n, E_pgcg=Eref, E_nodvfs=E0, arms={})
    for cost in ('worst', 'mean', 'aware'):
        q.MEM_MODE = 'cg'; own = ON.own_twin(K, SUP, *E.FIELDS[cost], per, n, arr); q.MEM_MODE = 'dvfs'
        for b in S.BS:
            lim = [max(T, x[0]*(1+b)) for x in R0]
            for g in GS:
                slo = dict(ttft=[max(T*g, x[0]*(1+g*b)) for x in R0], tpot=[Td*.9]*n)
                A, sc, _ = ON.run_online(K, SUP, 0., 0., 0., 1., bg, *E.FIELDS[cost], lag_q=E.LQ, hedge=E.HEDGE, warm=8, nom=None, slo=slo, own=own); Rr = per(sc[3]); Ex = ON.energy(K, SUP, A, sc)
                out['arms'][f'{cost}_b{b}_g{g}'] = dict(E=Ex, ttft_miss=sum(r[0] > l+1 for r, l in zip(Rr, lim)), tpot_added=sum(a and r[1] > Td+1 for a, r in zip(tp0, Rr)), ttft_abs=sum(r[0] <= T+1 for r in Rr))
    DEST.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(out)+'\n'); return name, 'ok'


def main():
    DEST.mkdir(parents=True, exist_ok=True); js = [c for c in S.cases() if R.FILT in f'{c[0]}_{c[1]}{c[2]}_r{c[3]}_{c[4]}']; print(len(js), 'cases', flush=True)
    with ProcessPoolExecutor(min(R.W, len(js)), max_tasks_per_child=1, initializer=E.limit_memory) as ex:
        fs = {ex.submit(job, c): c for c in js}
        for x in as_completed(fs):
            try: print(x.result(), flush=True)
            except Exception as e: print('FAILED', fs[x], repr(e)[:300], flush=True)


if __name__ == '__main__': main()
