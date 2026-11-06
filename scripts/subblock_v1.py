"""Can the per-block costing go finer than mean and worst can? Split each mixed-precision prefill matrix block into one contiguous run per precision, each with its own point.
Only the precision map tells where these runs are, so average and worst-case costing cannot follow. Tile order inside a block is assumed reorderable into such runs.
Shares come from the uniform-precision layer times (results/motivation_c): C(p) ~ 3.29+.275(p-4), M(p) ~ .6625p+.17, so a P4 run is compute-bound and a P8 run HBM-bound.
Same absolute objectives as the block-level arm (block nominal x (1+budget)); a split block loses the smoothing of max(C,M) over the whole block, which the arm pays for. First-order, exploratory."""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
_r = sys.argv[1:] if len(sys.argv) > 1 and sys.argv[1].isdigit() else []; W = int(_r[0]) if _r else 8
R = '/shared/s3/zqy/dac2027_p1/results'; os.environ.update(ON_CACHE=f'{R}/cache_mot', ON_DEST=f'{R}/subblock/x')
sys.argv = ['x', 'o']; sys.path.insert(0, str(Path(__file__).resolve().parent))
import online_eval_v1 as E
q = E.q; ON = E.ON; K, SUP = E.K, E.SUP; DEST = Path(f'{R}/subblock')
_t = ON.Online._take; ON.Online._take = lambda s, key, b: _t(s, key, max(b, 0.))
Cp = lambda p: 3.29+.275*(p-4); Mp = lambda p: .6625*p+.17


def split():
    ops = []; n = 0
    for o in q.OPS:
        if o['unit'] != 'MATRIX' or '/t0/' not in o['id'] or len(o['prec']) < 2: ops.append(o); continue
        ps = sorted(o['prec']); tn = sum(o['prec'].values()); tc = sum(Cp(p)*o['prec'][p] for p in ps); tb = sum(Mp(p)*o['prec'][p] for p in ps); prev = None; n += 1
        for j, p in enumerate(ps):
            x = o['prec'][p]; fc = Cp(p)*x/tc; fb = Mp(p)*x/tb; fn = x/tn
            s = dict(o); s['id'] = o['id'] if j == len(ps)-1 else f"{o['id']}_s{j}"; s['prec'] = {p: x}
            s.update(C=o['C']*fc, C_true=o['C_true']*fc, B=o['B']*fb, B_true=o['B_true']*fb, C_p5=o['C_p5']*fc, B_p5=o['B_p5']*fb, macs=o['macs']*fn, sram_bytes=o['sram_bytes']*fb, residual=o['residual']*fn)
            ops.append(s)
            prev_id = prev; prev = s['id']
            if j == 0: dep0 = q.DEPS[o['id']]
            q.DEPS[s['id']] = list(dep0) if j == 0 else [prev_id]
    q.OPS[:] = ops; return n


def per(lat): return [(lat[i], sum(lat[i+2:i+128])/126) for i in range(0, len(lat), 128)]


def job(c):
    name, plan, p = c; f = DEST/f'{name}.json'
    if f.exists(): return name, 'skip'
    E.install(f'sub_{name}', 'qwen', plan, dict(mode='full', prompts=[p]*3, outputs=128)); E.add_costings()
    q.MEM_MODE = 'cg'; s0 = q.run(K, {}, SUP, False); E0, comp0, _ = q.price(K, SUP, {}, s0, pg=True); R0 = per(s0[3]); q.MEM_MODE = 'dvfs'
    arr = [q.ARRIVAL_END.get(f'arr{j}', 0.) for j in range(len(R0))]; slo = dict(ttft=[a for a, _ in R0], tpot=[d for _, d in R0]); out = dict(name=name, E_pgcg=E0, arms={})
    B = ((.02, .02), (.05, .05), (.10, .10), (.25, .05))
    for arm in ('block', 'split'):
        n = split() if arm == 'split' else 0
        q.MEM_MODE = 'cg'; s1 = q.run(K, {}, SUP, False); E1, comp, _ = q.price(K, SUP, {}, s1, pg=True); R1 = per(s1[3]); own = ON.own_twin(K, SUP, 'C_true', 'B_true', per, len(R1), arr); q.MEM_MODE = 'dvfs'
        bg = E.IP._bg_power(comp, s1[0]); a = dict(split_blocks=n, ttft_max=max(x[0]/y[0] for x, y in zip(R1, R0)), makespan=s1[0]/s0[0], E_nom=E1, plans={})
        for bt, bd in B:
            g = 1.; rounds = 0
            while True:
                A, sc, info = ON.run_online(K, SUP, bt, bd, 0., g, bg, 'C_true', 'B_true', lag_q=E.LQ, hedge=E.HEDGE, warm=8, nom=None, slo=slo, own=own); Rr = per(sc[3])
                use = [(x/y-1)/b for (t, d), (x0, y0) in zip(R0, Rr) for y, x, b in ((t, x0, bt), (d, y0, bd))]; miss = sum(u > 1+1e-9 for u in use)
                if not miss or rounds >= 8: break
                g *= min(.995, .95/max(use)); rounds += 1
            a['plans'][f't{int(bt*100)}_d{int(bd*100)}'] = dict(saving=100*(1-ON.energy(K, SUP, A, sc)/E0), miss=miss, rounds=rounds, guard=g, use=max(use))
        out['arms'][arm] = a
    DEST.mkdir(exist_ok=True); f.write_text(json.dumps(out, indent=1)+'\n'); return name, {k: (round(out['arms']['block']['plans'][k]['saving'], 2), round(v['saving'], 2), v['miss']) for k, v in out['arms']['split']['plans'].items()}


def main():
    DEST.mkdir(exist_ok=True); V = E.P.VARIANCE; print(list(V), flush=True)
    js = [('fisher55_p512', E.F, 512), ('fisher55_p1920', E.F, 1920), ('fisher55_p3968', E.F, 3968)]+[(f'{n}_p1920', pf, 1920) for n, pf in V.items() if n != 'fisher_b5.5']
    with ProcessPoolExecutor(min(W, len(js)), max_tasks_per_child=1, initializer=E.limit_memory) as ex:
        fs = {ex.submit(job, c): c for c in js}
        for x in as_completed(fs):
            try: print(x.result(), flush=True)
            except Exception as e: print('FAILED', fs[x][0], repr(e)[:500], flush=True)


if __name__ == '__main__': main()
