"""Every workload of the paper through one causal controller and one set of costings.
usage: QWEN_CAL=<cal.json> FREQ_STATES=34 ON_DEST=<dir> online_eval_v1.py [workers] [name filter]
Env: ON_GUARD (.9), ON_LAGQ (.25, 'none' to disable), TTFT_FLOOR_PS (50e9, traces only), ON_CACHE (installed workloads), ON_TWIN (1: nominal twin clock), ON_HEDGE (0: lag_q is a hard ceiling), ON_COST (costings), ON_RLIM_G (per-process address-space cap).
A workload is installed once and cached, then each costing and budget takes one schedule pass. Finished results are skipped."""
import collections, json, os, pickle, random, re, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
_a = sys.argv[1:] if len(sys.argv) > 1 and sys.argv[1].isdigit() else []  # imported workers see another argv
WORKERS = int(_a[0]) if _a else 8; FILT = _a[1] if len(_a) > 1 else ''
sys.argv = ['x', 'o']; sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('P1_DEST', '/tmp'); os.environ.setdefault('P1_WORKERS', '1'); os.environ.setdefault('P1_SWEEPS', 'azure'); os.environ.setdefault('PLAN_FIX2', '1')
import llama_first_order_v2 as LL
import p1_eval_v1 as P, online_planner_v1 as ON
import serve_batch_v1 as SB
IE, q, SV, v3 = P.IE, P.q, P.SV, P.v3; IP = IE.IP; S6 = IE.S6
W = Path('/shared/s3/zqy/dac2027_p1'); DEST = Path(os.environ.get('ON_DEST', W/'results/online')); CACHE = Path(os.environ.get('ON_CACHE', W/'results/online_cache'))
K, SUP = IE.K, IE.SUP; G = float(os.environ.get('ON_GUARD', .9)); FLOOR = float(os.environ.get('TTFT_FLOOR_PS', 50e9))
_lq = os.environ.get('ON_LAGQ', '.25'); LQ = None if _lq == 'none' else float(_lq)
COST = tuple(os.environ.get('ON_COST', 'worst,mean,layer,module,aware').split(',')); F = S6.F_QWEN; TWIN = os.environ.get('ON_TWIN', '0') == '1'; HEDGE = os.environ.get('ON_HEDGE', '1') == '1'; FAIR = os.environ.get('ON_FAIR', '0') == '1'


def _arr(gap, n):
    rng = random.Random(7); t = 0.; a = []
    for _ in range(n): a.append(t*1e12); t += rng.expovariate(1/gap)
    return a


def workloads():
    full = dict(mode='full', prompts=[1920]*3, outputs=128); B = ((0., 0.), (.02, .02), (.05, .05), (.10, .10), (.25, .05)); o = []
    o.append(('main', 'qwen', F, full, B, 0.))
    for b in (1, 4, 16, 64): o.append((f'qwen_batch{b}', 'qwen', F, dict(mode='decode', L=1920, R=b, B=b, outputs=32), ((.05, .05),), 0.))
    o.append(('llama', 'llama', S6.F_LLAMA, full, ((.05, .05), (.25, .05)), 0.)); o.append(('llama_batch16', 'llama', S6.F_LLAMA, dict(mode='decode', L=1920, R=16, B=16, outputs=32), ((.05, .05),), 0.))
    for pn, pf in P.VARIANCE.items(): o.append((f'var_{pn}', 'qwen', pf, full, ((.05, .05), (.10, .05)), 0.))
    tb = ((.02, .02), (.05, .05), (.10, .10))
    for load, gap in (('low', 8.), ('mid', 4.), ('high', 2.5)): o.append((f'serve_{load}', 'qwen', F, dict(mode='full', prompts=[1920]*8, outputs=64, arrivals=_arr(gap, 8)), tb, FLOOR))
    for load, gap in (('low', 4.), ('mid', 2.), ('high', 1.)):
        o.append((f'servebatch_{load}', 'serve', F, dict(prompts=[512]*8, outputs=128, arrivals=_arr(gap, 8)), tb, FLOOR))
        o.append((f'servebatch32_{load}', 'serve', F, dict(prompts=[512]*32, outputs=128, arrivals=_arr(gap, 32)), ((.05, .05),), FLOOR))
    for kind in ('conv', 'code'):
        for w in range(3):
            for lname, rho in (('low', .25), ('mid', .5), ('high', .8)):
                kw = P.azure_kw(kind, w, rho)[0]; o.append((f'az_{kind}{w}_{lname}', 'azure', F, kw, ((.05, .05),), FLOOR))
    if os.environ.get('ON_SET') == 'cov': o = coverage()
    return [x for x in o if FILT in x[0]]


Q_ = '/shared/s3/zqy/DSE/npu_mac/outputs/flexposit_precision_quality/'
MODELS = (('q14', 'qwen', F), ('q32', 'q32', Q_+'qwen32b_fisher_p4p8_plan_v1/plan_b5.5.json'), ('l8', 'llama', S6.F_LLAMA), ('l8bm', 'bm', Q_+'llama3_8b_bitmod_fisher_plan_v1/plan_b4.0.json'))
MIXP = [256, 512, 1920, 512, 3968, 256]; MIXO = 128


def coverage():
    """Every model against every request type, all through the same fair flow."""
    o = []; B = ((.05, .05), (.10, .10), (.25, .05))
    for m, how, plan in MODELS:
        if not plan: continue
        for tag, pr, n in (('short', 256, 128), ('mid', 512, 128), ('long', 1920, 128), ('ctx4k', 3968, 128), ('outlong', 512, 512), ('bothlong', 1920, 512)):
            o.append((f'cov_{m}_{tag}', how, plan, dict(mode='full', prompts=[pr]*3, outputs=n), B, 0.))
        o.append((f'cov_{m}_mixed', how, plan, dict(mode='full', prompts=MIXP, outputs=MIXO), B, 0.))
        for load, gap in (('low', 16.), ('high', 5.)):
            o.append((f'covt_{m}_poisson_{load}', how, plan, dict(mode='full', prompts=MIXP*2, outputs=MIXO, arrivals=_arr(gap, 12)), B[:2], FLOOR))
    return o


def install(name, how, plan, kw):
    if os.environ.get('ON_AGN') == '1': name += '_agn'
    import qwen32b_first_order_v1 as Q32, llama_bitmod_first_order_v1 as BM
    f = CACHE/f'{name}.pkl'; mods = [q, v3, SV, SB, LL, S6, Q32, BM]
    if f.exists():
        for m, d in zip(mods, pickle.loads(f.read_bytes())):
            for n, x in d.items(): setattr(m, n, x)
        return
    before = [{n: id(x) for n, x in vars(m).items()} for m in mods]; v3.AGN_BYTES = 'worst' if os.environ.get('ON_AGN') == '1' else 'true'; b = v3.load_bits(plan)
    if how == 'qwen': v3.install(b, **kw)
    elif how == 'llama': LL.install(LL.load_bits(plan), **kw)
    elif how == 'q32': Q32.install(Q32.load_bits(plan), **kw)
    elif how == 'bm': BM.install(BM.load_bits(plan), **kw)
    elif how == 'serve': SB.install(b, kw['prompts'], kw['outputs'], kw['arrivals'])
    else: SV.install(b, kw['prompts'], kw['outputs'], kw['arrivals'])
    st = []
    for m, bf in zip(mods, before):
        d = {}
        for n, x in vars(m).items():
            if bf.get(n) != id(x) or n in ('ARRIVAL_END', 'OPS', 'DEPS', 'DUR', 'OUT_LIST'):
                try: pickle.dumps(x); d[n] = x
                except Exception: pass
        st.append(d)
    CACHE.mkdir(exist_ok=True); f.write_bytes(pickle.dumps(st))


def add_costings():
    ops = [o for o in q.OPS if o['prec']]
    def add(field, key):
        g = collections.defaultdict(lambda: [0., 0.])
        for o in ops:
            for p, n in o['prec'].items(): a = g[key(o)]; a[0] += p*n; a[1] += n
        for o in q.OPS:
            if o['prec']:
                m = sum(p*x for p, x in o['prec'].items())/sum(o['prec'].values()); a = g[key(o)]
                o[field] = o['C']*(94+20*(a[0]/a[1]-4))/(94+20*(m-4))
            else: o[field] = o['C']
    tot = sum(sum(o['prec'].values()) for o in ops); pbar = sum(p*n for o in ops for p, n in o['prec'].items())/tot
    for o in q.OPS:  # bytes follow precision, so a costing that lacks the map also lacks each block's bytes
        o['B_mean'] = o['B']*pbar/(sum(p*x for p, x in o['prec'].items())/sum(o['prec'].values())) if o['prec'] and os.environ.get('ON_AGN') == '1' else o['B']
    add('C_mean', lambda o: 0); add('C_module', lambda o: o['id'].split('/')[-1].split('_k')[0])
    add('C_layer', lambda o: (re.search(r'/L(\d+)/', o['id']) or [0, 'x'])[1])


FIELDS = dict(worst=('C_p5', 'B_p5'), mean=('C_mean', 'B_mean'), layer=('C_layer', 'B_true'), module=('C_module', 'B_true'), aware=('C_true', 'B_true'))


def job(w):
    name, how, plan, kw, budgets, floor = w; t0 = time.time(); install(name, how, plan, kw); add_costings(); outs = kw.get('outputs', 32)
    ns = list(outs) if isinstance(outs, (list, tuple)) else None
    def per(lat):  # TPOT from the second gap on: batch membership is fixed by the nominal run, so the wait to join a late step is a model artifact
        o = []; i = 0; m = ns or [outs]*(len(lat)//outs)
        for n in m: o.append((lat[i], sum(lat[i+2:i+n])/max(n-2, 1))); i += n
        return o
    q.MEM_MODE = 'cg'; s0 = q.run(K, {}, SUP, False); Eref, comp, _ = q.price(K, SUP, {}, s0, pg=True); E0 = q.price(K, SUP, {}, s0, pg=False, mem_dvfs=False)[0]; q.MEM_MODE = 'dvfs'
    nom = None
    if TWIN: q.MEM_MODE = 'cg'; nom = ON.nominal_dispatch(K, SUP); q.MEM_MODE = 'dvfs'
    R0 = per(s0[3]); bg = IP._bg_power(comp, s0[0]); done = []
    for bt, bd in budgets:
        for c in COST:
            f = DEST/f'{name}_t{int(bt*100)}_d{int(bd*100)}_{c}.json'
            if f.exists(): continue
            slo = own = None; g = G; rounds = 0; tr = name.startswith(('az_', 'serve', 'covt_'))
            if FAIR:  # the objective is given to every costing; each costing's beliefs come from its own twin
                arr = [q.ARRIVAL_END.get(f'arr{j}', 0.) for j in range(len(R0))]; slo = dict(ttft=[a for a, _ in R0], tpot=[d for _, d in R0])
                q.MEM_MODE = 'cg'; own = ON.own_twin(K, SUP, *FIELDS[c], per, len(R0), arr); q.MEM_MODE = 'dvfs'
                if not tr: g = 1.
            while True:
                A, sc, info = ON.run_online(K, SUP, bt, bd, floor, g, bg, *FIELDS[c], lag_q=LQ, hedge=HEDGE, warm=8, nom=None if FAIR else nom, slo=slo, own=own); R = per(sc[3]); E = ON.energy(K, SUP, A, sc)
                v = [dict(req=r, phase=ph, over=100*(x/y-1), nominal_ms=y/1e9, excess_ms=(x-y-max(y*b, floor if ph == 'P' else 0.))/1e9) for r, ((t, d), (x0, y0)) in enumerate(zip(R0, R)) for ph, y, x, b in (('P', t, x0, bt), ('D', d, y0, bd)) if x > y+max(y*b, floor if ph == 'P' else 0.)+1]
                if not (FAIR and v and not tr and rounds < 3 and bt > 0): break
                # budget feedback, same for every costing: tighten only, when over target; traces keep the preset guard, their misses are unseen arrivals
                use = max(x['over']/(100*(bt if x['phase'] == 'P' else bd)) for x in v); g *= .95/use; rounds += 1
            f.write_text(json.dumps(dict(workload=name, bt=bt, bd=bd, costing=c, saving_vs_pg_cg=100*(1-E/Eref), saving_vs_nodvfs=100*(1-E/E0), requests=len(R0), viol=len(v), viol_detail=v, max_over=max([x['over'] for x in v], default=0.),
                                     points=len(q.points), guard=G, lag_q=LQ, floor=floor, twin=TWIN, fair=FAIR, guard_final=g, rounds=rounds, **info))+'\n'); done.append(f.stem)
    return name, len(done), time.time()-t0


def limit_memory():
    import resource; cap = int(os.environ.get('ON_RLIM_G', 16)) << 30  # conv0_low with the twin needs more than the default 16G
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))


def main():
    DEST.mkdir(parents=True, exist_ok=True); W_ = workloads(); print('workloads', len(W_), flush=True)
    with ProcessPoolExecutor(min(WORKERS, len(W_)), max_tasks_per_child=1, initializer=limit_memory) as ex:
        fs = {ex.submit(job, w): w[0] for w in W_}
        for f in as_completed(fs):
            try: print(f.result(), flush=True)
            except Exception as e: print('FAILED', fs[f], repr(e)[:300], flush=True)


if __name__ == '__main__': main()
