"""What T2 buys at request level: export-only VU vs write-back-all (4-entry return queue) vs no local forwarding, first-order model, per-block controller.
VU time per kernel is scaled by the lockstep RTL ratio at 1.2/4.2 ns (results/t2_real). Kernels without an RTL run (casts, biases, rope, ksplit) use residual_add, the nearest element-wise kernel.
All arms get the same absolute objectives: export-only nominal latency x (1+budget), so a slower VU has less to spend. Stall energy is bounded two ways:
gated: stalled VU cycles cost no dynamic energy; clocked: VU dynamic energy scales with its time. Extra crossing energy of write-back is not modelled, which favors write-back."""
import json, os, sys, re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
_r = sys.argv[1:] if len(sys.argv) > 1 and sys.argv[1].isdigit() else []; W = int(_r[0]) if _r else 12; FILT = _r[1] if len(_r) > 1 else ''
R = '/shared/s3/zqy/dac2027_p1/results'; os.environ.update(ON_CACHE=f'{R}/cache_mot', ON_DEST=f'{R}/t2_req/x')
sys.argv = ['x', 'o']; sys.path.insert(0, str(Path(__file__).resolve().parent))
import online_eval_v1 as E
q = E.q; ON = E.ON; K, SUP = E.K, E.SUP; DEST = Path(f'{R}/t2_req')
import array_domains_v1 as AD
_t = ON.Online._take
ON.Online._take = lambda s, key, b: _t(s, key, max(b, 0.))  # an arm already past its objective keeps its free steps
RTL = {}  # kernel -> (write-back 4 entries, no forwarding)
_w = json.loads(open(f'{R}/t2_real/writeback.json').read())['rows']; _o = json.loads(open(f'{R}/t2_real/shadow_off.json').read())['rows']
for r in _w:
    if r['vu_ps'] == 4200 and r['policy'] == 'all' and r['out_entries'] == 4: RTL[r['kernel']] = [1+r['vs_v2_export_out4'], None]
for r in _o:
    if r['vu_ps'] == 4200 and r['variant'] == 'noshadow': RTL[r['kernel']][1] = 1+r['vs_shadow']


def kernel(oid, prompt):
    if '/softmax' in oid: return 'softmax_w512' if prompt <= 512 else 'softmax_w2048'
    if 'silu_mul' in oid: return 'silu_mul'
    if oid.endswith('norm'): return 'rmsnorm'
    if oid.endswith('emit'): return 'argmax_chunk'
    return 'residual_add'


def per(lat): return [(lat[i], sum(lat[i+2:i+128])/126) for i in range(0, len(lat), 128)]


def job(c):
    m, how, plan, p = c; name = f'{m}_p{p}'; f = DEST/f'{name}.json'
    if f.exists(): return name, 'skip'
    E.install(f'mot_{m}_p{p}_mixed', how, plan, dict(mode='full', prompts=[p]*3, outputs=128)); E.add_costings()
    q.MEM_MODE = 'cg'; s0 = q.run(K, {}, SUP, False); E0, comp0, _ = q.price(K, SUP, {}, s0, pg=True); R0 = per(s0[3]); q.MEM_MODE = 'dvfs'
    arr = [q.ARRIVAL_END.get(f'arr{j}', 0.) for j in range(len(R0))]; slo = dict(ttft=[a for a, _ in R0], tpot=[d for _, d in R0])
    orig = [(o['C'], o['C_true']) for o in q.OPS]; out = dict(model=m, prompt=p, E_pgcg=E0, arms={})
    for arm in ('export', 'wb4_gated', 'wb4_clocked', 'nofwd_gated', 'nofwd_clocked'):
        ix = 0 if arm.startswith('wb4') else 1
        for o, (cc, t) in zip(q.OPS, orig):
            fx = 1. if arm == 'export' or o['unit'] != 'VU' else RTL[kernel(o['id'], p)][ix]
            o['C_true'] = t*fx; o['C'] = cc*(fx if arm.endswith('clocked') else 1.)
        q.MEM_MODE = 'cg'; s1 = q.run(K, {}, SUP, False); E1, comp, _ = q.price(K, SUP, {}, s1, pg=True); R1 = per(s1[3]); own = ON.own_twin(K, SUP, 'C_true', 'B_true', per, len(R1), arr); q.MEM_MODE = 'dvfs'
        bg = E.IP._bg_power(comp, s1[0])
        a = dict(ttft_max=max(x[0]/y[0] for x, y in zip(R1, R0)), tpot_max=max(x[1]/y[1] for x, y in zip(R1, R0)), makespan=s1[0]/s0[0], E_nom=E1, plans={})
        for bt, bd in ((.05, .05), (.10, .10), (.25, .05)):
            g = 1.; rounds = 0
            while True:
                A, sc, info = ON.run_online(K, SUP, bt, bd, 0., g, bg, 'C_true', 'B_true', lag_q=E.LQ, hedge=E.HEDGE, warm=8, nom=None, slo=slo, own=own); Rr = per(sc[3])
                use = [(x/y-1)/b for (t, d), (x0, y0) in zip(R0, Rr) for y, x, b in ((t, x0, bt), (d, y0, bd))]; miss = sum(u > 1+1e-9 for u in use)
                if not miss or rounds >= 6 or a['ttft_max']-1 > bt or a['tpot_max']-1 > bd: break  # past the objective at peak V/f: nothing to tighten
                g *= min(.995, .95/max(use)); rounds += 1
            Et = ON.energy(K, SUP, A, sc)
            a['plans'][f't{int(bt*100)}_d{int(bd*100)}'] = dict(E=Et, saving_vs_export_pgcg=100*(1-Et/E0), miss=miss, objectives=len(use), max_use=max(use), rounds=rounds, guard=g, ttft_max=max(x[0]/y[0] for x, y in zip(Rr, R0)), tpot_max=max(x[1]/y[1] for x, y in zip(Rr, R0)))
        out['arms'][arm] = a
    DEST.mkdir(exist_ok=True); f.write_text(json.dumps(out, indent=1)+'\n'); return name, 'ok'


def main():
    DEST.mkdir(exist_ok=True); Mm = {m: (h, x) for m, h, x in E.MODELS}
    js = [(m, *Mm[m], p) for m in ('q14', 'q32', 'l8') for p in (512, 1920, 3968) if FILT in f'{m}_p{p}']; print(len(js), flush=True)
    with ProcessPoolExecutor(min(W, len(js)), max_tasks_per_child=1, initializer=E.limit_memory) as ex:
        fs = {ex.submit(job, c): c for c in js}
        for x in as_completed(fs):
            try: print(x.result(), flush=True)
            except Exception as e: print('FAILED', fs[x][0], fs[x][3], repr(e)[:400], flush=True)


if __name__ == '__main__': main()
