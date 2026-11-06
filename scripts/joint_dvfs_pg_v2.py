"""Per-precision, per-component fine-grained DVFS + PG estimate on scaled arrays (v2).

Adds to lowv_dvfs_v1:
  * IVR transitions are hidden when the operator has slack (memory-bound MAC, or
    MEM utilization < 1 behind the FIFO); otherwise they add latency.
  * VU/ATT get the same voltage list (stretch anchored on the 0.6 V low_vu point).
  * MEM-side retained logic (bank controllers, matrix/VU peripherals) is clocked
    down to its per-operator utilization and moved to the lowest V that still
    meets that clock (DVFS of the non-bottleneck component).
  * PG: HBM controller (v4 proxy) and engine-idle windows of retained logic
    longer than BET (10% residual, ReGate-style).
  * alpha: retained peripheral power scales with k**alpha (0 shared, 1 replicated).
First-order estimate; not a calibrated replay.
"""
import collections, contextlib, importlib.util, io, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE/file)
    argv, sys.argv = sys.argv, ['x', str(HERE/'out_tmp')]
    with contextlib.redirect_stdout(io.StringIO()):
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    sys.argv = argv
    return m


L = load('lowv', 'lowv_dvfs_v1.py')
S = L.S
ops, deps, points = S.ops, S.deps, L.points
templates = S.templates
profiles = S.profiles

# stretch of each point relative to nominal, per unit class
MAC_STRETCH = [p['period']/4000. for p in points]
VU_STRETCH = [1.]+[7500/5500*L.worst[str(p['v'])]/L.worst['0.6'] for p in points[1:]]
MEM_STRETCH = MAC_STRETCH  # same cell mix assumption, nominal slack similar to MAC
ATT_STRETCH = VU_STRETCH
import os as _o, math as _m
if _o.environ.get('STA_CAT') == '1':  # catalog bounded by the 7-voltage STA of the patched netlists: period = path x 1.01, log-linear between anchors
    _V = (.45, .5, .525, .55, .6, .65, .7)
    _P = dict(MAC=(11977, 8344, 7201, 6325, 5083, 4264, 3690), VU=(13661, 9506, 8199, 7194, 5770, 4822, 4157), ATT=(12586, 8750, 7542, 6609, 5285, 4409, 3796))
    def _path(n, v):
        for a, b, x, y in zip(_V, _V[1:], _P[n], _P[n][1:]):
            if a <= v <= b: return _m.exp(_m.log(x)+(v-a)/(b-a)*(_m.log(y)-_m.log(x)))
        return _P[n][0] if v < _V[0] else _P[n][-1]
    MAC_STRETCH = [max(s, 1.01*_path('MAC', p['v'])/4000.) for s, p in zip(MAC_STRETCH, points)]; MEM_STRETCH = MAC_STRETCH
    VU_STRETCH = [max(s, 1.01*_path('VU', p['v'])/4000.) for s, p in zip(VU_STRETCH, points)]
    ATT_STRETCH = [max(1., 1.01*_path('ATT', p['v'])/4000.) for p in points]

VU_FAM = {'nonlinear', 'elementwise', 'argmax'}
ATT_FAM = {'attention', 'fp16_matrix'}

# per-op static data
for o in ops:
    w = templates[profiles[o['id']]['work_key']]
    c = w.get('counts') or {}
    o['sram_bytes'] = sum(v for k, v in c.items() if k.startswith('sram_') and k.endswith('_bytes'))
    o['prec'] = {int(p): n for p, n in (w.get('precision_work', {}).get('mac_operations_by_precision') or {}).items()}

# MEM utilization normalisation: busiest matrix op at k=1 defines 100% per engine
cap = collections.defaultdict(float)
for o in ops:
    if o['family'] == 'matrix' and o['sram_bytes'] and o['span']:
        cap[o['engine']] = max(cap[o['engine']], o['sram_bytes']/o['span'])

logic_in = S.logic_in; load_total = 17.99914
vu_in = logic_in*0.08388293397880367/load_total
att_in = logic_in*0.6506163943611687/load_total
other_work_in = S.ledger_const['logic_work_part']-S.mac_in-vu_in-att_in
vu_C = sum(o['C'] for o in ops if o['family'] in VU_FAM)
att_C = sum(o['C'] for o in ops if o['family'] in ATT_FAM)
RET_W_PER_ENGINE = S.ledger_time['logic_time_part']/(S.T0*1e-12)/2  # retained logic at k=1, W
BET_PS = (412+120)*1000
PG_RESIDUAL = .1
IVR_PS_PER_V = 4000/.054


def unit_of(o):
    if o['family'] == 'matrix': return 'MATRIX'
    if o['family'] in VU_FAM: return 'VU'
    if o['family'] in ATT_FAM: return 'ATT'
    return None


def run(k, bw, assign, supply):
    """assign: op index -> point index for its compute unit. Returns schedule and per-op info."""
    eta = L.SUPPLY[supply]
    end, free, lastv = {}, collections.defaultdict(int), {}
    busy = collections.defaultdict(list)
    mem_rho_t = collections.defaultdict(float); mem_mix = collections.Counter()
    for i, o in enumerate(ops):
        unit = unit_of(o); pi = assign.get(i, 0)
        c = o['C']/k if o['family'] in S.SCALED else o['C']
        if unit == 'MATRIX': c *= MAC_STRETCH[pi]
        elif unit in ('VU', 'ATT'): c *= VU_STRETCH[pi]
        m = o['B']/bw*1e3
        trans = 0.
        if unit:
            key = (o['engine'], unit); pv = lastv.get(key, .7); v = points[pi]['v']
            trans = abs(pv-v)*IVR_PS_PER_V; lastv[key] = v
        # transition hidden if it fits inside memory slack
        t = max(c+trans, m) if m > 0 else c+trans
        t += o['residual']
        s = max([free[o['engine']]]+[end[d] for d in deps.get(o['id'], []) if d in end])
        end[o['id']] = s+t; free[o['engine']] = s+t
        busy[o['engine']].append((s, s+t))
        # MEM retained logic: clock down to utilisation, lowest V meeting that clock
        u = 1.
        if o['family'] == 'matrix' and o['sram_bytes'] and cap[o['engine']]:
            u = min(1., (o['sram_bytes']/k)/(t-o['residual'] or 1)/cap[o['engine']])
        best = (1., .7)
        for qi, q in enumerate(points):
            if MEM_STRETCH[qi] <= 1/max(u, 1e-9)+1e-12:
                rho = (q['v']/.7)**2*eta(.7)/eta(q['v'])*u
                if rho < best[0]: best = (rho, q['v'])
        mem_rho_t[o['engine']] += best[0]*t; mem_mix[best[1]] += t
        o['_t'] = t
    return max(end.values()), end, busy, mem_rho_t, mem_mix


def price(k, bw, ch, alpha, supply, assign, sched, pg=True):
    T, end, busy, mem_rho_t, mem_mix = sched
    eta = L.SUPPLY[supply]
    r = [L.ratio(p['v'], eta) for p in points]
    s = T/S.T0
    e = dict(
        dram_die_background=S.ledger_time['dram_die_background']*s,
        controller_background=S.ledger_time['controller_background']*s*ch/4,
        frontend_controller=S.ledger_time['frontend_controller']*s,
        pd_link_fifo=S.ledger_time['pd_link_fifo']*s,
        array_leakage=S.LEAK_J_K1*k*s)
    e.update({kk: v for kk, v in S.ledger_const.items() if kk != 'logic_work_part'})
    e['other_logic_work'] = other_work_in
    mac = vu = att = 0.
    for i, o in enumerate(ops):
        pi = assign.get(i, 0); u = unit_of(o)
        if u == 'MATRIX' and o['macs']: mac += S.mac_in*o['macs']/S.macs_total*r[pi]
        elif u == 'VU' and vu_C: vu += vu_in*o['C']/vu_C*r[pi]
        elif u == 'ATT' and att_C: att += att_in*o['C']/att_C*r[pi]
    e['matrix_core'] = mac; e['vu'] = vu; e['att'] = att
    pw = RET_W_PER_ENGINE*k**alpha*1e-12
    ret_busy = ret_idle = 0.
    idle_pg_saved = 0.
    for eng in ('P', 'D'):
        iv = sorted(busy[eng]); t_busy = sum(b-a for a, b in iv)
        ret_busy += pw*mem_rho_t[eng]
        gaps = [iv[0][0]]+[max(0, iv[j+1][0]-iv[j][1]) for j in range(len(iv)-1)]+[T-iv[-1][1]]
        idle = sum(gaps); ret_idle += pw*idle
        if pg:
            idle_pg_saved += pw*sum(g-BET_PS for g in gaps if g > BET_PS)*(1-PG_RESIDUAL)
    for eng in busy:
        if eng not in ('P', 'D'):  # SHARED engine time keeps nominal retained power on both cores
            pass
    e['retained_logic_busy'] = ret_busy; e['retained_logic_idle'] = ret_idle-idle_pg_saved
    hbm_u = S.B_total/(bw*1e9)/(T*1e-12)
    u1 = S.B_total/(24e9)/(S.T0*1e-12)
    e['hbm_controller_pg_credit'] = -(e['controller_background']*S.PG_FRACTION_K1*max(0, 1-hbm_u)/(1-u1) if pg else 0.)
    return sum(e.values()), e, dict(hbm_util=hbm_u, logic_idle_pg_saved_j=idle_pg_saved,
                                    mem_time_by_v={str(v): x/sum(mem_mix.values()) for v, x in sorted(mem_mix.items())})


def latency(end): return L.latency(end)


GROUPS = collections.defaultdict(list)
for i, o in enumerate(ops):
    u = unit_of(o)
    if u == 'MATRIX' and o['macs']:
        GROUPS[L.group_of(o)].append(i)
    elif u in ('VU', 'ATT'):
        GROUPS[(o['engine'], o['family'])].append(i)


def optimize(args):
    k, bw, ch, alpha, supply, budget = args
    r = [L.ratio(p['v'], L.SUPPLY[supply]) for p in points]
    base_sched = run(k, bw, {}, supply)
    lat0 = latency(base_sched[1])
    # reference: no DVFS, no PG, MEM retained at nominal (rho=1) -> recompute with u forced 1
    E_ref, comp_ref, _ = price(k, bw, ch, alpha, supply, {}, (base_sched[0], base_sched[1], base_sched[2],
                               {e: sum(b-a for a, b in base_sched[2][e]) for e in base_sched[2]}, {0.7: 1}), pg=False)

    def ok(end_): return all(b <= a*(1+budget)+1 for a, b in zip(lat0, latency(end_)))
    # free slack per matrix op (MAC stretch + transition hidden under memory time)
    assign = {}
    for i, o in enumerate(ops):
        if unit_of(o) != 'MATRIX' or not o['macs']: continue
        c = o['C']/k; m = o['B']/bw*1e3
        best = 0
        for pi in range(1, len(points)):
            if c*MAC_STRETCH[pi]+(.7-points[pi]['v'])*IVR_PS_PER_V*2 <= m and r[pi] < r[best]: best = pi
        if best: assign[i] = best
    sched = run(k, bw, assign, supply)
    if not ok(sched[1]):
        assign = {}; sched = base_sched
    E, comp, info = price(k, bw, ch, alpha, supply, assign, sched)
    E_free = E
    level = collections.Counter()
    while True:
        best = None
        for g, idx in GROUPS.items():
            nxt = level[g]+1
            if nxt >= len(points): continue
            trial = dict(assign)
            for i in idx: trial[i] = max(trial.get(i, 0), nxt)
            sc = run(k, bw, trial, supply)
            if not ok(sc[1]): continue
            Et, ct, it = price(k, bw, ch, alpha, supply, trial, sc)
            if Et < E-1e-9 and (best is None or Et < best[0]): best = (Et, g, trial, sc, ct, it)
        if not best: break
        E, g, assign, sched, comp, info = best; level[g] += 1
    lat = latency(sched[1])
    prec_mix = collections.defaultdict(collections.Counter)
    for i, o in enumerate(ops):
        if unit_of(o) == 'MATRIX':
            for p, n in o['prec'].items(): prec_mix[f'P{p}'][points[assign.get(i, 0)]['v']] += n
    unit_mix = collections.defaultdict(collections.Counter)
    for i, o in enumerate(ops):
        u = unit_of(o)
        if u: unit_mix[f"{o['engine']}.{u}"][points[assign.get(i, 0)]['v']] += o['_t']
    norm = lambda c: {str(v): x/sum(c.values()) for v, x in sorted(c.items())}
    return dict(k=k, alpha=alpha, supply=supply, budget=budget, reference_j=E_ref, reference_components=comp_ref,
                free_slack_j=E_free, final_j=E, components=comp, info=info,
                saving_percent=100*(1-E/E_ref), free_slack_saving_percent=100*(1-E_free/E_ref),
                makespan_s=sched[0]*1e-12, reference_makespan_s=base_sched[0]*1e-12,
                worst_latency_increase_percent=100*max(b/a-1 for a, b in zip(lat0, lat)),
                mac_by_precision_and_v={p: norm(c) for p, c in prec_mix.items()},
                unit_time_by_v={u: norm(c) for u, c in unit_mix.items()},
                group_levels={':'.join(map(str, g)): points[l]['v'] for g, l in level.items()})


if __name__ == '__main__':
    dest = Path(sys.argv[1]); dest.mkdir(parents=True, exist_ok=True)
    jobs = [(k, .8*204.8, 4, a, s, b) for k in (1, 64, 128) for a in (0, 1)
            for s in ('dldo', 'hybrid_scvr_dldo') for b in (0., .05)]
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        rows = list(ex.map(optimize, jobs))
    (dest/'result.json').write_text(json.dumps(dict(status='FIRST_ORDER_JOINT_DVFS_PG_ESTIMATE_NOT_REPLAYED',
                                                    points=points, rows=rows), indent=1)+'\n')
    for r in rows:
        print(f"k={r['k']:3d} a={r['alpha']} {r['supply']:17s} b={r['budget']:.0%} ref {r['reference_j']:7.2f}J -> {r['final_j']:7.2f}J "
              f"save {r['saving_percent']:5.1f}% (free {r['free_slack_saving_percent']:5.1f}%) lat +{r['worst_latency_increase_percent']:.2f}% "
              f"HBMu {100*r['info']['hbm_util']:.0f}%")
