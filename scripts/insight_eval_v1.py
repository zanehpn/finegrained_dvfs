"""Evaluate the search-free insight planner against the search-based planner and a Lagrangian bound.

For each scenario and tile costing (worst case, average precision, per tile), fine domains:
plan with insight_planner_v1 using that costing, then evaluate with true costs (first-order schedule and
ledger). A plan that misses an objective under true costs is repaired as in sensitivity_v6 (raise the
violating phase of a request one level per round). Reported: saving over PG+CG and NoDVFS, repairs,
planning wall time, the planning-model energy change against the Lagrangian bound, and the search
planner's saving for the same scenario (pivot_sensitivity_v7/v6/v3 when present).
usage: insight_eval_v1.py
"""
import json, glob, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
sys.argv = ['x', 'o']; sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_first_order_v3 as v3, llama_first_order_v2 as LL, qwen_first_order_v1 as q
import insight_planner_v1 as IP
import sensitivity_v6 as S6

import os
O = Path('/shared/s3/zqy/DSE/npu_mac/outputs/flexposit_enpu_paper/asap7_calibration'); ROOT = Path(os.environ.get('CAL_DEST', O)); DEST = ROOT/'insight_planner_eval_v1'
K, SUP = 256, 'hybrid_scvr_dldo'


def job(spec):
    name, model, plan, kw, bt, bd, costing = spec
    base = LL if model == 'llama' else v3
    mod = S6.mean_cost_module(base) if costing == 'mean' else base
    v3.AGN_BYTES = 'true'; mod.install(mod.load_bits(plan), **kw)
    outs = kw.get('outputs', 32); per_req = S6.per_req_fn(outs)
    cf, bf = ('C_true', 'B_true') if costing == 'aware' else ('C_p5', 'B_p5')
    assign, info = IP.plan(K, SUP, bt, bd, cf, bf, per_req, outs)
    if os.environ.get('PLAN_ASSIGN_DUMP'):
        import pickle; pickle.dump(dict(assign=assign, info=info, outs=outs), open(os.environ['PLAN_ASSIGN_DUMP'], 'wb'))
    bound = IP.lagrange_bound(K, SUP, bt, bd, cf, bf, per_req)
    q.MEM_MODE = 'cg'; s0 = q.run(K, {}, SUP, False); E0 = q.price(K, SUP, {}, s0, pg=False, mem_dvfs=False)[0]
    Eref = q.price(K, SUP, {}, s0, pg=True)[0]; q.MEM_MODE = 'dvfs'; R0 = per_req(s0[3])
    FIX2 = os.environ.get('PLAN_FIX2') == '1'; floor = float(os.environ.get('TTFT_FLOOR_PS', 0))
    def bad(R): return [(r, ph) for r, ((t0, d0), (t, d)) in enumerate(zip(R0, R)) for ph, x, y, b in (('P', t, t0, bt), ('D', d, d0, bd)) if x > y+max(y*b, floor if ph == 'P' else 0.)+1]
    sc = q.run(K, assign, SUP, False); R = per_req(sc[3]); pre = bad(R); rounds = 0
    while bad(R) and rounds < 8:
        for r, ph in bad(R):
            for i in list(assign):
                oid = q.OPS[i]['id']; tag = oid.split('/')[0]
                mem = q.OPS[i].get('members')
                mine = tag == f'r{r}' if ph == 'P' else (tag == f'r{r}' or (tag.startswith('g') and (r in mem if FIX2 and mem else True)))
                if mine and ((ph == 'P') == ('/t0/' in oid)):
                    assign[i] -= 1
                    if not assign[i]: del assign[i]
        sc = q.run(K, assign, SUP, False); R = per_req(sc[3]); rounds += 1
    E = q.price(K, SUP, assign, sc, pg=True)[0]
    final_viol = [dict(req=r, phase=ph, nominal_ps=R0[r][0 if ph == 'P' else 1], actual_ps=R[r][0 if ph == 'P' else 1]) for r, ph in bad(R)]
    return dict(final_viol=final_viol, name=name, model=model, costing=costing, budget_ttft=bt, budget_tpot=bd, workload=kw,
                saving_vs_pg_cg=100*(1-E/Eref), saving_vs_nodvfs=100*(1-E/E0), met=not bad(R), pre_repair_violations=len(pre),
                repair_rounds=rounds, plan_wall_s=info['plan_wall_s'], backoff_rounds=info['backoff_rounds'], steps=info['steps'],
                accepted=info['accepted'], model_delta_j=info['model_delta_j'], bound_delta_j=bound,
                bound_gap_pct=100*(1-info['model_delta_j']/bound) if bound < 0 else None, energy_j=E, pg_cg_j=Eref, nodvfs_j=E0)


def specs():
    full = lambda p: dict(mode='full', prompts=[p]*3, outputs=128); out = []
    for c in ('worst', 'mean', 'aware'):
        for bt, bd in ((0., 0.), (.02, .02), (.05, .05), (.10, .10), (.25, .05)):
            out.append((f'main_t{int(100*bt)}_d{int(100*bd)}', 'qwen', S6.F_QWEN, full(1920), bt, bd, c))
        for b in (1, 4, 16, 64):
            out.append((f'qwen_batch{b}', 'qwen', S6.F_QWEN, dict(mode='decode', L=1920, R=b, B=b, outputs=32), .05, .05, c))
        out += [('llama_t5_d5', 'llama', S6.F_LLAMA, full(1920), .05, .05, c), ('llama_t25_d5', 'llama', S6.F_LLAMA, full(1920), .25, .05, c),
                ('llama_batch16', 'llama', S6.F_LLAMA, dict(mode='decode', L=1920, R=16, B=16, outputs=32), .05, .05, c)]
    return out


def search_result(name, costing):
    for d in ('pivot_sensitivity_v7', 'pivot_sensitivity_v6', 'pivot_sensitivity_v5'):
        for f in glob.glob(str(O/d/f'{name}_fine_{costing}.json')):
            return json.loads(Path(f).read_text())['saving_vs_pg_cg']
    return None


def main():
    DEST.mkdir(exist_ok=True); jobs = [j for j in specs() if not (DEST/f'{j[0]}_{j[-1]}.json').exists()]
    with ProcessPoolExecutor(min(24, len(jobs)), max_tasks_per_child=1, initializer=S6.limit_memory) as ex:
        futs = {ex.submit(job, j): j for j in jobs}
        for f in as_completed(futs):
            try: r = f.result()
            except Exception as e: print('FAILED', futs[f][0], futs[f][-1], repr(e)[:300], flush=True); continue
            r['search_saving_vs_pg_cg'] = search_result(r['name'], r['costing'])
            (DEST/f"{r['name']}_{r['costing']}.json").write_text(json.dumps(r, indent=1)+'\n')
            gap = f"{r['bound_gap_pct']:.1f}%" if r['bound_gap_pct'] is not None else 'n/a'
            print(f"{r['name']:16s} {r['costing']:6s} insight {r['saving_vs_pg_cg']:5.1f}% (search {r['search_saving_vs_pg_cg']}) vsNoDVFS {r['saving_vs_nodvfs']:5.1f}% "
                  f"met={r['met']} viol={r['pre_repair_violations']} repair={r['repair_rounds']} backoff={r['backoff_rounds']} gap-to-bound {gap} plan {r['plan_wall_s']:.2f}s", flush=True)


if __name__ == '__main__': main()
