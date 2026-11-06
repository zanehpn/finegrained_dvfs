"""Single-rail (coarse) DVFS at realistic switching granularity vs MIDAS.

A coarse design has one rail per core (all arrays, VU, attention unit and memory logic). Our default
coarse arm may switch that rail at every operator with nanosecond regulators, an upper bound for a
core-wide rail. This study restricts the coarse rail to
  'layer' : one point per (core, transformer layer index, phase), no per-operator free slowdowns;
  'phase' : one point per core for the whole prefill or decode phase (Ascend-style epochs);
and compares them with the per-operator coarse arm and fine-grained MIDAS, all with per-tile costing.
The greedy allocation, objectives and energy model are planner_qwen_longctx_v1's.
usage: coarse_granularity_v1.py <sweep> ...   sweeps: main batch
"""
import collections, json, re, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
ARGS = sys.argv[1:]; sys.argv = ['x', 'o']; sys.path.insert(0, str(Path(__file__).resolve().parent))
import planner_qwen_longctx_v1 as P
import array_domains_v1 as AD
q = P.q
import os
DEST = Path(os.environ.get('CAL_DEST', P.O))/'coarse_granularity_v1'
F_QWEN = '/shared/s3/zqy/DSE/npu_mac/outputs/flexposit_precision_quality/qwen14b_fisher_p4p8_plan_v1/plan_b5.5.json'


def key(o, gran):
    eng = o['engine']; ph = 'P' if '/t0/' in o['id'] else 'D'
    if gran == 'phase': return (eng, ph)
    m = re.search(r'/L(\d+)/', o['id']); return (eng, ph, int(m.group(1)) if m else -1)


def plan_coarse(gran, b_ttft, b_tpot, outs):
    """Greedy lowering of single-rail groups at the given granularity (per-tile costing, coarse ledger)."""
    lat0 = q.run(P.K, {}, P.SUP, True)[3]; R0 = P.per_req(lat0)
    def ok(lat): return all(t <= t0*(1+b_ttft)+1 and d <= d0*(1+b_tpot)+1 for (t0, d0), (t, d) in zip(R0, P.per_req(lat)))
    def plan_E(asg):
        sc = q.run(P.K, asg, P.SUP, True)
        return q.price(P.K, P.SUP, asg, sc, pg=True)[0] if ok(sc[3]) else None
    G = collections.defaultdict(list)
    for i, o in enumerate(q.OPS):
        if o['unit']: G[key(o, gran)].append(i)
    assign = {}; E = plan_E(assign); level = collections.Counter(); changed = True; evals = 0
    while changed:
        changed = False
        for g, idx in G.items():
            while level[g]+1 < len(q.points):
                trial = dict(assign)
                for i in idx: trial[i] = level[g]+1
                Et = plan_E(trial); evals += 1
                if Et is None or Et >= E-1e-9: break
                E, assign = Et, trial; level[g] += 1; changed = True
    return assign, E, evals


def job(spec):
    name, kw, gran, bt, bd = spec
    P.v = P.v; P.KW = kw; outs = kw.get('outputs', 32); P.OUTS = outs
    P.per_req = lambda lat: [(lat[i], sum(lat[i+1:i+outs])/(outs-1)) for i in range(0, len(lat), outs)]
    P.v.install(P.v.load_bits(F_QWEN), **kw)
    q.MEM_MODE = 'cg'; s0 = q.run(P.K, {}, P.SUP, False)
    E0 = q.price(P.K, P.SUP, {}, s0, pg=False, mem_dvfs=False)[0]; Eref = q.price(P.K, P.SUP, {}, s0, pg=True)[0]; q.MEM_MODE = 'dvfs'
    t0 = time.monotonic(); assign, E, evals = plan_coarse(gran, bt, bd, outs)
    return dict(name=name, granularity=gran, budget_ttft=bt, budget_tpot=bd, workload=kw, energy_j=E, nodvfs_j=E0, pg_cg_j=Eref,
                saving_vs_nodvfs=100*(1-E/E0), saving_vs_pg_cg=100*(1-E/Eref), evaluations=evals, groups_lowered=len(set(assign.values())),
                planner_wall_s=time.monotonic()-t0)


def specs(sweep):
    full = dict(mode='full', prompts=[1920]*3, outputs=128); out = []
    if sweep == 'main':
        for bt, bd in ((0., 0.), (.02, .02), (.05, .05), (.10, .10), (.25, .05)):
            out += [(f'main_t{int(100*bt)}_d{int(100*bd)}', full, g, bt, bd) for g in ('layer', 'phase')]
    if sweep == 'batch':
        for b in (1, 4, 16, 64):
            out += [(f'qwen_batch{b}', dict(mode='decode', L=1920, R=b, B=b, outputs=32), g, .05, .05) for g in ('layer', 'phase')]
    return out


def limit_memory():
    import resource; cap = 16 << 30; resource.setrlimit(resource.RLIMIT_AS, (cap, cap))


def main():
    DEST.mkdir(exist_ok=True); jobs = [s for sw in ARGS for s in specs(sw)]
    with ProcessPoolExecutor(min(24, len(jobs)), max_tasks_per_child=1, initializer=limit_memory) as ex:
        futs = {ex.submit(job, j): j for j in jobs}
        for f in as_completed(futs):
            try: r = f.result()
            except Exception as e: print('FAILED', futs[f][:3], repr(e)[:300], flush=True); continue
            (DEST/f"{r['name']}_{r['granularity']}.json").write_text(json.dumps(r, indent=1)+'\n')
            print(f"{r['name']:18s} {r['granularity']:6s} vs NoDVFS {r['saving_vs_nodvfs']:5.1f}%  vs PG+CG {r['saving_vs_pg_cg']:5.1f}%  wall {r['planner_wall_s']:.0f}s", flush=True)


if __name__ == '__main__': main()
