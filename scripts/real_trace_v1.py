"""Real scenarios: Azure LLM traces at MLPerf latency limits, on allocations with measured quality only (Fisher P4-P8 plans, BitMoD 4.0-bit), first-order model, causal controller.
Trace: n consecutive requests with context+output <= 4096 tokens (the RTL-calibrated range), from three windows of each trace; arrivals dilated to a prefill load rho of one accelerator.
Objectives are absolute and shared by all arms, and neither costing nor controller sees future arrivals or output lengths:
  Llama-3.1-8B: Server 2 s / 100 ms, Interactive 0.5 s / 30 ms (MLPerf Inference, that model).
  Qwen2.5-14B/32B: Server 2 s / 200 ms, Interactive 0.45 s / 40 ms (MLPerf Llama-2-70B, the nearest larger model).
Requests that miss at peak V/f have no budget and keep free steps only. Idle time between requests is priced under PG+CG.
Guard g: the controller aims at g x limit. attain = both objectives met; added = requests met at peak V/f and missed here."""
import datetime as dt, json, os, sys, csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
_r = sys.argv[1:] if len(sys.argv) > 1 and sys.argv[1].isdigit() else []; W = int(_r[0]) if _r else 16; FILT = _r[1] if len(_r) > 1 else ''
R = '/shared/s3/zqy/dac2027_p1/results'; os.environ.update(ON_CACHE=f'{R}/cache_real', ON_DEST=f'{R}/real/x')
sys.argv = ['x', 'o']; sys.path.insert(0, str(Path(__file__).resolve().parent))
import online_eval_v1 as E
q = E.q; ON = E.ON; K, SUP = E.K, E.SUP; P = E.P; v3 = E.v3; LL = E.LL; DEST = Path(os.environ.get('REAL_DEST', f'{R}/real'))
N = int(os.environ.get('REAL_N', 64)); GUARDS = tuple(float(x) for x in os.environ.get('REAL_G', '.9').split(',')); LIMIT = 4096
_t = ON.Online._take; ON.Online._take = lambda s, key, b: _t(s, key, max(b, 0.))
QW = dict(server=(2e12, 200e9), interactive=(.45e12, 40e9)); LM = dict(server=(2e12, 100e9), interactive=(.5e12, 30e9))
MODELS = {m: (h, p) for m, h, p in E.MODELS}; TIER = dict(q14=QW, q32=QW, l8=LM, l8bm=LM); SPT = dict(q14=1., q32=1.67, l8=.54, l8bm=.54)  # prefill time per token relative to Qwen2.5-14B, from the 2K nominal


def trace(m, kind, w, rho):
    rows = P.window(kind, w, N, LIMIT); t0 = rows[0][0]; raw = [(r[0]-t0).total_seconds() for r in rows]; prompts = [r[1] for r in rows]; outs = [r[2] for r in rows]
    scale = (sum(prompts)*P.S_PER_TOKEN*SPT[m]/rho)/max(raw[-1], 1e-9)
    return prompts, outs, [a*scale*1e12 for a in raw]


def install(m, kind, w, rho):
    how, plan = MODELS[m]; name = f'{m}_{kind}{w}_r{rho}_n{N}'; prompts, outs, arr = trace(m, kind, w, rho)
    if m != 'q14':
        import serve_models_v1 as SM
        if how == 'bm':
            import llama_bitmod_first_order_v1 as BM; LL.make_matrix = BM.make_matrix
        E.SB._graph = SM.graph_for('q32' if how == 'q32' else 'llama')
    E.install(name, 'azure', plan, dict(prompts=prompts, outputs=outs, arrivals=arr)); E.add_costings()
    return outs, arr


def job(c):
    m, kind, w, rho, tier = c; name = f'{m}_{kind}{w}_r{rho}_{tier}'; f = DEST/f'{name}.json'
    if f.exists(): return name, 'skip'
    outs, arr = install(m, kind, w, rho); T, Td = TIER[m][tier]
    def per(lat):
        o = []; i = 0
        for n in outs: o.append((lat[i], sum(lat[i+1:i+n])/max(n-1, 1))); i += n
        return o
    q.MEM_MODE = 'cg'; s0 = q.run(K, {}, SUP, False); Eref, comp, _ = q.price(K, SUP, {}, s0, pg=True); E0 = q.price(K, SUP, {}, s0, pg=False, mem_dvfs=False)[0]; q.MEM_MODE = 'dvfs'
    R0 = per(s0[3]); bg = E.IP._bg_power(comp, s0[0]); met = lambda r: r[0] <= T+1 and r[1] <= Td+1; ok0 = [met(x) for x in R0]
    out = dict(model=m, kind=kind, w=w, rho=rho, tier=tier, T=T, Td=Td, requests=len(R0), tokens=sum(outs), makespan_s=s0[0]*1e-12, E_pgcg=Eref, E_nodvfs=E0, pgcg_vs_nodvfs=100*(1-Eref/E0), nominal=dict(attain=sum(ok0), ttft_p50=sorted(x[0] for x in R0)[len(R0)//2]*1e-12, ttft_met=sum(x[0] <= T+1 for x in R0), tpot_met=sum(x[1] <= Td+1 for x in R0), tpot_p50=sorted(x[1] for x in R0)[len(R0)//2]*1e-9), arms={})
    for cost in ('worst', 'mean', 'aware'):
        q.MEM_MODE = 'cg'; own = ON.own_twin(K, SUP, *E.FIELDS[cost], per, len(R0), arr); q.MEM_MODE = 'dvfs'
        for g in GUARDS:
            slo = dict(ttft=[T*g]*len(R0), tpot=[Td*g]*len(R0))
            A, sc, info = ON.run_online(K, SUP, 0., 0., 0., 1., bg, *E.FIELDS[cost], lag_q=E.LQ, hedge=E.HEDGE, warm=8, nom=None, slo=slo, own=own); Rr = per(sc[3]); Ex = ON.energy(K, SUP, A, sc); ok = [met(x) for x in Rr]
            out['arms'][f'{cost}_g{g}'] = dict(E=Ex, saving_pgcg=100*(1-Ex/Eref), saving_nodvfs=100*(1-Ex/E0), attain=sum(ok), added=sum(a and not b for a, b in zip(ok0, ok)), ttft_over_ms=max(0., max((x[0]-max(T, y[0]))*1e-9 for x, y in zip(Rr, R0))), decisions=info['decisions'])
    DEST.mkdir(exist_ok=True); f.write_text(json.dumps(out, indent=1)+'\n')
    return name, out['nominal']['attain'], {k: (round(v['saving_pgcg'], 2), v['added']) for k, v in out['arms'].items()}


def cases():
    return [(m, kind, w, rho, tier) for m in ('q14', 'q32', 'l8', 'l8bm') for kind in ('conv', 'code') for w in range(3) for rho in (.25, .5, .8) for tier in ('server', 'interactive')]


def main():
    DEST.mkdir(parents=True, exist_ok=True); js = [c for c in cases() if FILT in f'{c[0]}_{c[1]}{c[2]}_r{c[3]}_{c[4]}']; print(len(js), 'cases', flush=True)
    with ProcessPoolExecutor(min(W, len(js)), max_tasks_per_child=1, initializer=E.limit_memory) as ex:
        for t in ('server', 'interactive'):  # one tier at a time: the second reuses the trace cache the first built
            fs = {ex.submit(job, c): c for c in js if c[4] == t}
            for x in as_completed(fs):
                try: print(x.result(), flush=True)
                except Exception as e: print('FAILED', fs[x], repr(e)[:400], flush=True)


if __name__ == '__main__': main()
