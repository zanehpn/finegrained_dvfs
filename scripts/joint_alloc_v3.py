"""Energy-aware precision allocation: same Fisher score as the base plan, lowest request energy under the online controller.
A window upgrade is ranked by Fisher reduction per unit of marginal energy instead of per weight bit. Marginal energy per module type is
measured on the calibrated model with the fair controller, then re-measured on each new plan. Screening only: no quality measured here."""
import copy, heapq, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
W = Path('/shared/s3/zqy/dac2027_p1'); DEST = W/'results/joint_v3'; DEST.mkdir(exist_ok=True)
Q = Path('/shared/s3/zqy/DSE/npu_mac/outputs/flexposit_precision_quality/qwen14b_fisher_p4p8_plan_v1'); BASE = Q/'plan_b5.5.json'; WIN = 128
TYPES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
ttype = lambda n: next(t for t in TYPES if n.endswith(t))


FIXG = False


def fixed(a):
    global FIXG
    FIXG = True
    return evaluate(a)


def evaluate(a):
    """Fair online aware on 3x2K, tighten-only from guard 1. Returns absolute energies."""
    path, bt, cost = a; sys.argv = ['x', 'o']; sys.path.insert(0, str(W/'scripts'))
    import planner_qwen_longctx_v1 as P, online_eval_v1 as E
    q, v, ON, IP, K, SUP = E.q, P.v, E.ON, E.IP, P.K, P.SUP
    v.AGN_BYTES = 'true'; v.install(v.load_bits(path), mode='full', prompts=[1920]*3, outputs=P.OUTS); E.add_costings(); per = P.per_req
    q.MEM_MODE = 'cg'; s0 = q.run(K, {}, SUP, False); Ecg, comp, _ = q.price(K, SUP, {}, s0, pg=True); R0 = per(s0[3]); bg = IP._bg_power(comp, s0[0])
    own = ON.own_twin(K, SUP, *E.FIELDS[cost], per, 3, [0., 0., 0.]); q.MEM_MODE = 'dvfs'; slo = dict(ttft=[x for x, _ in R0], tpot=[x for _, x in R0]); g = 1.
    for rnd in range(1 if FIXG else 4):
        A, sc, _ = ON.run_online(K, SUP, bt, .05, 0., g, bg, *E.FIELDS[cost], lag_q=None, hedge=False, warm=8, slo=slo, own=own); R = per(sc[3])
        over = [x/y/(1+b) for (t0, d0), (t, d) in zip(R0, R) for x, y, b in ((t, t0, bt), (d, d0, .05))]
        if max(over) <= 1+1e-6: break
        use = max(((t/t0-1)/bt, (d/d0-1)/.05) for (t0, d0), (t, d) in zip(R0, R))[0]; g *= .95/use
    return dict(path=str(path), bt=bt, cost=cost, Ecg=Ecg, E=ON.energy(K, SUP, A, sc), g=g, rounds=rnd, met=max(over) <= 1+1e-6, t0=max(x for x, _ in R0), d0=max(x for _, x in R0))


def load(path): return {n: m['channel_precision'] for n, m in json.loads(Path(path).read_text())['modules'].items()}
def write(bits, name):
    d = json.loads(BASE.read_text()); d['status'] = 'ENERGY_AWARE_SCREEN_NOT_QUALITY_MEASURED'
    for n, b in bits.items(): d['modules'][n]['channel_precision'] = b
    p = DEST/f'{name}.json'; p.write_text(json.dumps(d)); return p
def windows(bits): return {n: b[::WIN] for n, b in bits.items()}
def expand(lv, shapes): return {n: [p for p in l for _ in range(WIN)][:shapes[n][0]] for n, l in lv.items()}


def allocate(scores, shapes, target, c):
    """Upgrade from P4 by Fisher reduction per unit cost until the Fisher score reaches target. c: cost per weight bit by module type."""
    lv = {n: [4]*len(scores[n]['4']) for n in scores}; S = sum(sum(scores[n]['4']) for n in scores); heap = []
    def push(n, w):
        p = lv[n][w]
        if p < 8:
            width = min(WIN, shapes[n][0]-w*WIN); r = (scores[n][str(p)][w]-scores[n][str(p+1)][w]); heapq.heappush(heap, (-r/(width*shapes[n][1]*c[ttype(n)]), n, w, r))
    for n in scores:
        for w in range(len(lv[n])): push(n, w)
    while heap and S > target:
        _, n, w, r = heapq.heappop(heap); lv[n][w] += 1; S -= r; push(n, w)
    return lv, S


def main():
    t0 = time.time(); sens = json.loads((Q/'sensitivity.json').read_text()); scores, shapes = sens['window_scores'], sens['shapes']
    base = load(BASE); lv0 = windows(base); S0 = sum(scores[n][str(p)][w] for n, l in lv0.items() for w, p in enumerate(l))
    avg = lambda lv: sum(p*min(WIN, shapes[n][0]-w*WIN)*shapes[n][1] for n, l in lv.items() for w, p in enumerate(l))/sum(s[0]*s[1] for s in shapes.values())
    log = open(DEST/'log.txt', 'a'); say = lambda *x: (print(*x, file=log, flush=True), print(*x, flush=True))
    say('base fisher', S0, 'avg bits', avg(lv0))
    ex = ProcessPoolExecutor(12, max_tasks_per_child=1); plans = {'base': BASE}; costs = {}; res = {}
    def run(items): return list(ex.map(evaluate, items))
    def marginal(tag, path):
        cur = windows(load(path)); items = [(path, .10, 'aware')]; dbits = {}
        for t in TYPES:
            lv = {n: [min(p+1, 8) if ttype(n) == t else p for p in l] for n, l in cur.items()}; items.append((write(expand(lv, shapes), f'm_{tag}_{t}'), .10, 'aware')); dbits[t] = avg(lv)-avg(cur)
        r = list(ex.map(fixed, items)); e0 = r[0]['E']; say('raw marginal', tag, {t: (round(x['E']-e0, 3), round(dbits[t], 4), x['met'], round(x['g'], 3)) for t, x in zip(TYPES, r[1:])}); c = {t: max((x['E']-e0)/max(dbits[t], 1e-9), 1e-9) for t, x in zip(TYPES, r[1:])}; return c, e0
    slope = {b: json.loads((Q/f'plan_b{b}.json').read_text()) for b in ('4.5', '5.0', '6.0')}
    sr = run([(Q/f'plan_b{b}.json', bt, 'aware') for b in slope for bt in (.05, .25)]+[(BASE, .05, 'aware'), (BASE, .25, 'aware')])
    for x in sr: say('slope', Path(x['path']).stem, x['bt'], 'E', round(x['E'], 2), 'Ecg', round(x['Ecg'], 2), 'fisher', round(sum(scores[n][str(p)][w] for n, l in windows(load(x['path'])).items() for w, p in enumerate(l)), 4), 'met', x['met'])
    c, e0 = marginal('base', BASE); m = sum(c.values())/len(c); c = {t: v/m for t, v in c.items()}; say('marginal cost base', {k: round(v, 3) for k, v in c.items()})
    for beta in (.5, 1., 2.):
        cc = dict(c)
        for rnd in range(3):
            lv, S = allocate(scores, shapes, S0, {t: v**beta for t, v in cc.items()}); name = f'plan_beta{beta}_r{rnd}'; p = write(expand(lv, shapes), name); plans[name] = p
            say(name, 'fisher ratio', round(S/S0, 4), 'avg bits', round(avg(lv), 3), 'by type', {t: round(sum(sum(l) for n, l in lv.items() if ttype(n) == t)/sum(len(l) for n, l in lv.items() if ttype(n) == t), 2) for t in TYPES})
            if rnd < 2: c2, _ = marginal(name, p); m = sum(c2.values())/len(c2); cc = {t: .5*cc[t]+.5*v/m for t, v in c2.items()}
    items = [(p, bt, cost) for p in plans.values() for bt in (.05, .25) for cost in ('aware', 'mean', 'worst')]; r = run(items); (DEST/'evals.json').write_text(json.dumps(r, indent=1))
    ref = {(x['bt'],): x['Ecg'] for x in r if x['path'] == str(BASE) and x['cost'] == 'aware'}
    for x in r: say(Path(x['path']).stem, x['bt'], x['cost'], 'save vs base PG+CG', round(100*(1-x['E']/ref[(x['bt'],)]), 2), 'own PG+CG', round(100*(1-x['Ecg']/ref[(x['bt'],)]), 2), 'g', round(x['g'], 3), 'met', x['met'], 'ttft', round(x['t0']/1e12, 2))
    say('wall', time.time()-t0)


if __name__ == '__main__': main()
