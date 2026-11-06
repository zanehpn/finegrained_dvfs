"""2x2 DVFS ablation: spatial granularity (coarse/fine) x precision awareness (agnostic/aware).

Same workload, k, alpha, supply, latency budget and PG model in every arm.
  coarse : one compute-domain V/f per core per operator (MATRIX/VU/ATT and the
           MEM-side retained logic share it; MEM clock follows the compute stretch).
  fine   : MATRIX, VU, ATT and MEM retained logic each pick their own V/f.
  agnostic : planner sees worst-case uniform P5 cycles and one group per layer type.
  aware    : planner sees the actual per-channel P4/P5 cycles and groups layers by P4-share tercile.
Plans are always re-evaluated on the true mixed-precision workload.
First-order estimate on the captured replay; not a calibrated simulator run.
"""
import collections, contextlib, importlib.util, io, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('j', HERE/'joint_dvfs_pg_v2.py')
argv, sys.argv = sys.argv, ['x', 'out_tmp']
with contextlib.redirect_stdout(io.StringIO()):
    J = importlib.util.module_from_spec(spec); spec.loader.exec_module(J)
sys.argv = argv
L, S, ops, points = J.L, J.S, J.ops, J.points

# worst-case uniform-P5 compute time seen by the precision-agnostic planner
for o in ops:
    o['C_true'] = o['C']; o['C_p5'] = o['C']
    if o['family'] == 'matrix' and o['prec']:
        w = J.templates[J.profiles[o['id']]['work_key']]['precision_work']
        steps = w['serial_dp4_group_steps_by_precision']; macs = w['mac_operations_by_precision']
        if '5' in steps:
            per5 = macs['5']/steps['5']
            o['C_p5'] = o['C']*(sum(macs.values())/per5)/sum(steps.values())
    o['p4'] = (o['prec'].get(4, 0)/sum(o['prec'].values())) if o['prec'] else 0.
    o['B_true'] = o['B']; o['B_p5'] = o['B']
    if o['family'] == 'matrix' and o['prec']:
        wb = J.templates[J.profiles[o['id']]['work_key']]['counts']['hbm_weight_read_bytes']
        o['B_p5'] = o['B']+wb*(5/(5-o['p4'])-1)  # packed weight bytes scale with bit width (D fit exact)


def groups(aware):
    g = collections.defaultdict(list)
    for i, o in enumerate(ops):
        u = J.unit_of(o)
        if u == 'MATRIX' and o['macs']:
            key = L.group_of(o)+((min(2, int(o['p4']*3)),) if aware else ())
            g[key].append(i)
        elif u in ('VU', 'ATT'):
            g[(o['engine'], o['family'])].append(i)
    return g


def run(k, bw, assign, supply, coarse, cfield, bfield='B_true'):
    eta = L.SUPPLY[supply]
    end, free, lastv = {}, collections.defaultdict(int), {}
    busy = collections.defaultdict(list); mem_rho_t = collections.defaultdict(float); mem_mix = collections.Counter()
    for i, o in enumerate(ops):
        unit = J.unit_of(o); pi = assign.get(i, 0)
        c = o[cfield]/k if o['family'] in S.SCALED else o[cfield]
        st = J.MAC_STRETCH[pi] if unit == 'MATRIX' else J.VU_STRETCH[pi] if unit else 1.
        c *= st
        m = o[bfield]/bw*1e3
        trans = 0.
        if unit:
            key = (o['engine'], 'CORE' if coarse else unit); pv = lastv.get(key, .7); v = points[pi]['v']
            trans = abs(pv-v)*J.IVR_PS_PER_V; lastv[key] = v
        mem_busy = 0.
        if o['family'] == 'matrix' and o['sram_bytes'] and J.cap[o['engine']]:
            mem_busy = (o['sram_bytes']/k)/J.cap[o['engine']]
        t = max(c+trans, m)
        if coarse and unit:
            t = max(t, mem_busy*J.MEM_STRETCH[pi]+trans)  # MEM slowed with the shared domain
        t += o['residual']
        s = max([free[o['engine']]]+[end[d] for d in J.deps.get(o['id'], []) if d in end])
        end[o['id']] = s+t; free[o['engine']] = s+t; busy[o['engine']].append((s, s+t))
        if coarse:
            if unit:
                q = points[pi]; rho = L.ratio(q['v'], eta)/J.MEM_STRETCH[pi]; vq = q['v']
            else:
                rho, vq = 1., .7
        else:
            if mem_busy:
                u = min(1., mem_busy/max(t-o['residual'], 1.))
            elif unit:  # no SRAM counts: MEM demand follows the compute rate (same rule as coarse)
                u = min(1., (o['C_true']/k if o['family'] in S.SCALED else o['C_true'])/max(t-o['residual'], 1.))
            else:
                u = 1.
            rho, vq = 1., .7
            for qi, q in enumerate(points):
                if J.MEM_STRETCH[qi] <= 1/max(u, 1e-9)+1e-12:
                    r = (q['v']/.7)**2*eta(.7)/eta(q['v'])*u
                    if r < rho: rho, vq = r, q['v']
        mem_rho_t[o['engine']] += rho*t; mem_mix[vq] += t
        o['_t'] = t
    return max(end.values()), end, busy, mem_rho_t, mem_mix


WARM = None


def arm(args):
    global WARM
    k, alpha, supply, budget, coarse, aware, pg = args[:7]
    WARM = args[7] if len(args) > 7 else None
    bw, ch = 85., 4  # Ramulator q256 streaming; first-order within 3.9% of calibrated replay at k=128
    cfield = 'C_true' if aware else 'C_p5'
    bfield = 'B_true' if aware else 'B_p5'
    r = [L.ratio(p['v'], L.SUPPLY[supply]) for p in points]
    true0 = run(k, bw, {}, supply, coarse, 'C_true'); lat_true0 = L.latency(true0[1])
    nominal_mem = {e: sum(b-a for a, b in true0[2][e]) for e in true0[2]}
    E_ref, comp_ref, _ = J.price(k, bw, ch, alpha, supply, {}, (*true0[:3], nominal_mem, {.7: 1}), pg=False)
    E_pg, _, _ = J.price(k, bw, ch, alpha, supply, {}, (*true0[:3], nominal_mem, {.7: 1}), pg=True)

    def ok(end_, lat0): return all(b <= a*(1+budget)+1 for a, b in zip(lat0, L.latency(end_)))

    def plan_energy(assign):
        sc = run(k, bw, assign, supply, coarse, cfield, bfield)
        if not ok(sc[1], lat_true0): return None  # absolute SLO from the true baseline
        return J.price(k, bw, ch, alpha, supply, assign, sc, pg=pg)[0]
    # free slack on the planner's view
    assign = {}
    for i, o in enumerate(ops):
        if J.unit_of(o) != 'MATRIX' or not o['macs']: continue
        c = o[cfield]/k; m = o[bfield]/bw*1e3; best = 0
        for pi in range(1, len(points)):
            if c*J.MAC_STRETCH[pi]+(.7-points[pi]['v'])*J.IVR_PS_PER_V*2 <= m and r[pi] < r[best]: best = pi
        if best: assign[i] = best
    E = plan_energy(assign)
    if E is None:
        assign = {}; E = plan_energy({})
    if WARM is not None:  # fine arm also starts from the coarse arm's plan (fine space includes it)
        Ew = plan_energy(dict(WARM))
        if Ew is not None and Ew < E:
            E, assign = Ew, dict(WARM)
    # coordinate descent over groups (same search for every arm)
    G = groups(aware); level = collections.Counter(); changed = True
    while changed:
        changed = False
        for g, idx in G.items():
            while level[g]+1 < len(points):
                trial = dict(assign)
                for i in idx: trial[i] = max(trial.get(i, 0), level[g]+1)
                Et = plan_energy(trial)
                if Et is None or Et >= E-1e-9: break
                E, assign = Et, trial; level[g] += 1; changed = True
    # evaluate on the true workload
    sc = run(k, bw, assign, supply, coarse, 'C_true')
    E_true, comp, info = J.price(k, bw, ch, alpha, supply, assign, sc, pg=pg)
    lat = L.latency(sc[1])
    worst = max(b/a-1 for a, b in zip(lat_true0, lat))
    prec = collections.defaultdict(collections.Counter)
    for i, o in enumerate(ops):
        if J.unit_of(o) == 'MATRIX':
            for p, n in o['prec'].items(): prec[f'P{p}'][points[assign.get(i, 0)]['v']] += n
    norm = lambda c: {str(v): x/sum(c.values()) for v, x in sorted(c.items())}
    return dict(k=k, alpha=alpha, supply=supply, budget=budget, coarse=coarse, aware=aware, pg=pg,
                reference_j=E_ref, pg_only_j=E_pg, final_j=E_true,
                saving_vs_reference_percent=100*(1-E_true/E_ref),
                saving_vs_pg_only_percent=100*(1-E_true/E_pg) if pg else None,
                worst_latency_increase_percent=100*worst, budget_met=worst <= budget+1e-9,
                groups=len(G), assignment={str(i): p for i, p in assign.items()}, warm_started=WARM is not None, mac_by_precision_and_v={p: norm(c) for p, c in prec.items()},
                mem_time_by_v=info['mem_time_by_v'], components=comp, reference_components=comp_ref)


if __name__ == '__main__':
    dest = Path(sys.argv[1]); dest.mkdir(parents=True, exist_ok=True)
    jobs = [(128, a, 'hybrid_scvr_dldo', b, coarse, aware, pg)
            for a in (0, 1) for b in (0., .05) for coarse in (True, False) for aware in (False, True) for pg in (True, False)]
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        rows = list(ex.map(arm, jobs))
    (dest/'result.json').write_text(json.dumps(dict(status='FIRST_ORDER_DVFS_ABLATION_NOT_REPLAYED', rows=rows), indent=1)+'\n')
    for r in rows:
        name = ('coarse' if r['coarse'] else 'fine')+'+'+('aware' if r['aware'] else 'agnostic')
        print(f"a={r['alpha']} b={r['budget']:.0%} pg={int(r['pg'])} {name:16s} E {r['final_j']:6.2f}J  vs ref {r['saving_vs_reference_percent']:5.2f}%  "
              f"vs PG-only {r['saving_vs_pg_only_percent'] if r['pg'] else float('nan'):5.2f}%  lat +{r['worst_latency_increase_percent']:.2f}% met={r['budget_met']}", flush=True)
