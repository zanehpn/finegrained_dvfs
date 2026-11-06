"""Effect of weight precision on each component and voltage domain, over models and prompt lengths (first-order model, 4 rails, 34 points).

Grid: Qwen2.5-14B, Qwen2.5-32B, Llama-3.1-8B x uniform P4/P5/P6/P8 and the Fisher 5.5-bit plan x three requests of 512, 1920, 3968 prompt tokens + 128 output.
At nominal: busy time per unit, MAC-weighted compute-bound share, PG+CG energy per domain.
Under the per-block controller at TTFT/TPOT 5/5 and 25/5: energy per domain, energy-weighted voltage per rail.
Objectives are each case's own nominal latency, so precision changes are compared at the same relative slack.
usage: QWEN_CAL=... FREQ_STATES=34 STA_CAT=1 RAILS=4 ON_ALLDOM=1 ON_FAIR=1 ON_LAGQ=.25 motivation_precision_v1.py [workers] [filter]"""
import json, os, sys, collections
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
_r = sys.argv[1:] if len(sys.argv) > 1 and sys.argv[1].isdigit() else []; W = int(_r[0]) if _r else 12; FILT = _r[1] if len(_r) > 1 else ''  # workers re-import with another argv
R = '/shared/s3/zqy/dac2027_p1/results'; os.environ.setdefault('ON_CACHE', f'{R}/cache_mot'); os.environ.setdefault('ON_DEST', f'{R}/motivation_c/x')
sys.argv = ['x', 'o']; sys.path.insert(0, str(Path(__file__).resolve().parent))
import online_eval_v1 as E
import array_domains_v1 as AD
q = E.q; ON = E.ON; K, SUP = E.K, E.SUP; DEST = Path(f'{R}/motivation_c')
HBM = ('dram_die_background', 'dram_activity', 'io_activity', 'controller_background', 'controller_activity', 'dma', 'hbm_controller_pg_credit')


def domains(c, sav=0.):
    m = c['matrix_core']; lg = sum(AD.SHARE[d] for d in AD.HEADROOM)
    return dict(MAC=m*(1-lg), LOGIC=m*lg-sav, VEC=c['vu']+c['att'], MEM=c['retained_logic_busy']+c['sram'], HBM=sum(c.get(k, 0.) for k in HBM),
                OTHER=c['frontend_controller']+c['pd_link_fifo']+c['array_leakage']+c['other_logic_work']+c['retained_logic_idle'])


def phase(o): return 'prefill' if '/t0/' in o['id'] else 'decode'


def nominal_stats():
    busy = collections.defaultdict(lambda: collections.defaultdict(float)); cb = collections.Counter(); mac = collections.Counter()
    for o in q.OPS:
        p = phase(o); c = o['C_true']/K*1e-12; m = o['B_true']/q.BW_R
        if o['unit'] == 'MATRIX':
            busy[p]['MAC'] += c; mac[p] += o['macs']
            if c > m: cb[p] += o['macs']
        if o['unit'] in ('VU', 'ATT'): busy[p]['VEC'] += c
        busy[p]['HBM'] += m
        if o['sram_bytes']: busy[p]['MEM'] += o['sram_bytes']/K/q.CAP*1e-12
    return dict(busy_s={p: dict(v) for p, v in busy.items()}, compute_bound={p: cb[p]/max(mac[p], 1) for p in mac})


def volts(A):
    """Energy-weighted voltage of the MAC and VEC rails per phase; LOGIC follows the MAC point within its clock headroom."""
    st = q.J.MAC_STRETCH; h = min(AD.HEADROOM.values()); s = collections.defaultdict(lambda: [0., 0.])
    for i, o in enumerate(q.OPS):
        if o['unit'] not in ('MATRIX', 'VU', 'ATT'): continue
        pi = A.get(i, 0); p = phase(o); v = q.points[pi]['v']; w = o['macs'] if o['unit'] == 'MATRIX' else o['C']
        if o['unit'] == 'MATRIX':
            a = max(j for j in range(len(q.points)) if st[j] <= h*st[pi]+1e-12)
            for rail, x in (('MAC', v), ('LOGIC', q.points[a]['v'])): s[f'{p}/{rail}'][0] += w*x; s[f'{p}/{rail}'][1] += w
        else: s[f'{p}/VEC'][0] += w*v; s[f'{p}/VEC'][1] += w
    return {k: a/b for k, (a, b) in s.items() if b}


def job(c):
    model, how, plan, prec, prompt = c; name = f'{model}_p{prompt}_{prec}'; f = DEST/f'{name}.json'
    if f.exists(): return name, 'skip'
    kw = dict(mode='full', prompts=[prompt]*3, outputs=128)
    if prec != 'mixed':
        import qwen32b_first_order_v1 as Q32
        m = dict(qwen=E.v3, q32=Q32, llama=E.LL)[how]; lb = m.load_bits; P = int(prec[1:])
        m.load_bits = lambda p, lb=lb, P=P: {n: [P]*len(x) for n, x in lb(p).items()}
    E.install(f'mot_{name}', how, plan, kw); E.add_costings()
    def per(lat):
        o = []
        for i in range(0, len(lat), 128): o.append((lat[i], sum(lat[i+2:i+128])/126))
        return o
    q.MEM_MODE = 'cg'; s0 = q.run(K, {}, SUP, False); Eref, comp, _ = q.price(K, SUP, {}, s0, pg=True); nom = ON.nominal_dispatch(K, SUP); q.MEM_MODE = 'dvfs'
    R0 = per(s0[3]); bg = E.IP._bg_power(comp, s0[0]); arr = [q.ARRIVAL_END.get(f'arr{j}', 0.) for j in range(len(R0))]
    slo = dict(ttft=[a for a, _ in R0], tpot=[d for _, d in R0])
    q.MEM_MODE = 'cg'; own = ON.own_twin(K, SUP, 'C_true', 'B_true', per, len(R0), arr); q.MEM_MODE = 'dvfs'
    out = dict(model=model, prec=prec, prompt=prompt, ttft_s=[a*1e-12 for a in slo['ttft']], tpot_ms=[d*1e-9 for d in slo['tpot']], makespan_s=s0[0]*1e-12, nominal=nominal_stats(), E_pgcg=Eref, dom_pgcg=domains(comp), plans={})
    for bt, bd in ((.05, .05), (.25, .05)):
        g = 1.; rounds = 0
        while True:
            A, sc, info = ON.run_online(K, SUP, bt, bd, 0., g, bg, 'C_true', 'B_true', lag_q=E.LQ, hedge=E.HEDGE, warm=8, nom=None, slo=slo, own=own); Rr = per(sc[3])
            over = [(x/y-1)/b-1 for (t, d), (x0, y0) in zip(R0, Rr) for y, x, b in ((t, x0, bt), (d, y0, bd)) if x > y*(1+b)+1]  # budget use beyond 1
            if not over or rounds >= 8: break
            g *= min(.995, .95/(1+max(over))); rounds += 1
        Et, c1, _ = q.price(K, SUP, A, sc, pg=True); sav = AD.saving(A, SUP)
        out['plans'][f't{int(bt*100)}_d{int(bd*100)}'] = dict(saving=100*(1-(Et-sav)/Eref), dom=domains(c1, sav), volts=volts(A), viol=len(over), rounds=rounds, guard=g)
    DEST.mkdir(exist_ok=True); f.write_text(json.dumps(out, indent=1)+'\n'); return name, round(out['plans']['t5_d5']['saving'], 2), round(out['plans']['t25_d5']['saving'], 2)


def cases():
    Mm = {m: (h, p) for m, h, p in E.MODELS}
    return [(m, Mm[m][0], Mm[m][1], pr, p) for m in ('q14', 'q32', 'l8') for pr in ('P4', 'P5', 'P6', 'P8', 'mixed') for p in (512, 1920, 3968)]


def main():
    DEST.mkdir(exist_ok=True); js = [c for c in cases() if FILT in f'{c[0]}_p{c[4]}_{c[3]}']; print(len(js), 'cases', flush=True)
    with ProcessPoolExecutor(min(W, len(js)), max_tasks_per_child=1, initializer=E.limit_memory) as ex:
        fs = {ex.submit(job, c): c for c in js}
        for x in as_completed(fs):
            try: print(x.result(), flush=True)
            except Exception as e: print('FAILED', fs[x][:1], fs[x][3:], repr(e)[:400], flush=True)


if __name__ == '__main__': main()
