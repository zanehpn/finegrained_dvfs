"""v6: v5 plus (a) high-variance precision allocations and (b) pre-repair violation statistics.

New sweep 'variance': Qwen plans with large tile/operator precision variance -- bimodal P4/P8 Fisher
plans at 5.5 bits (128-channel windows; whole modules), the real kv8_rest4 and Fisher b4.5 plans --
under worst-case, average and per-tile costing, fine domains, at TTFT/TPOT 5%/5%, TTFT 25%, and decode
batches 4 and 16. Every job now records, before any repair, which requests miss which objective and by
how much (true-cost latency over the objective), so the safety of average costing can be reported.
Sweep 'serve' as in v5, re-run to collect these statistics.

v5 description:

Planner = best of two greedy orders (as sensitivity_v4). New pieces:
  * 'mean' baseline (precision-agnostic, average-precision costing): every matrix tile is costed at the
    plan's weight-averaged precision (5.5 bits for b5.5) instead of the worst case; weight bytes are
    exact. Its plan can miss the objectives under true costs, so it is repaired like a replay: while a
    request misses TTFT (TPOT), the points of that request's prefill (decode) ops are raised by one
    level, then re-checked. Reported: planner estimate, true result, repair rounds.
  * sweep 'slo2': TTFT budget 10%/50% at TPOT 5%, TPOT budget 10%/20% at TTFT 5% (Qwen b5.5).
  * sweep 'serve': N=8 Qwen requests (prompt 1920, 64 outputs) with seeded Poisson arrivals at three
    rates (prefill-engine load about 0.25/0.5/0.8); objectives per request against the nominal run of
    the same trace. Reports energy per output token and absolute-SLO attainment
    (TTFT <= 2x the isolated prefill time, mean TPOT <= 1.1x the isolated decode step).
usage: sensitivity_v5.py <sweep> ...   sweeps: agnmean slo2 serve
"""
import json, random, sys, time, types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
ARGS = sys.argv[1:]; sys.argv = ['x', 'o']; sys.path.insert(0, str(Path(__file__).resolve().parent))
import planner_qwen_longctx_v1 as P1, planner_qwen_longctx_v2 as P2
import qwen_first_order_v3 as v3, llama_first_order_v2 as LL, qwen_first_order_v1 as q

DEST = P2.O/'pivot_sensitivity_v6'
F_QWEN = '/shared/s3/zqy/DSE/npu_mac/outputs/flexposit_precision_quality/qwen14b_fisher_p4p8_plan_v1/plan_b5.5.json'
F_LLAMA = '/shared/s3/zqy/DSE/npu_mac/outputs/flexposit_precision_quality/llama3_8b_fisher_p4p8_plan_v1/plan_b5.5.json'


def mean_cost_module(base):
    """Wrap a workload module so install() costs matrix tiles at the plan's weight-averaged precision."""
    def install(bits, **kw):
        n = base.install(bits, **kw)
        tot = sum(len(b) for b in bits.values()); pbar = sum(sum(b) for b in bits.values())/tot
        for o in q.OPS:
            if o['prec']:
                c = sum(o['prec'].values()); mean = sum(p*x for p, x in o['prec'].items())/c
                o['C_p5'] = o['C']*(94+20*(pbar-4))/(94+20*(mean-4)); o['B_p5'] = o['B']
        return n
    return types.SimpleNamespace(install=install, load_bits=base.load_bits)


def per_req_fn(outs): return lambda lat: [(lat[i], sum(lat[i+1:i+outs])/(outs-1)) for i in range(0, len(lat), outs)]


def plan_once(P, model, plan, kw, coarse, aware, bt, bd, mean_cost):
    base = LL if model == 'llama' else v3
    P.v = mean_cost_module(base) if mean_cost else base; P.PLANS = {'x': plan}; P.KW = kw
    outs = kw.get('outputs', 32); P.OUTS = outs; P.per_req = per_req_fn(outs)
    t0 = time.monotonic(); _, asg = P.arm(('x', coarse, aware, bt, bd, 'self', None))
    return asg, time.monotonic()-t0


def true_eval(asg, coarse, outs):
    """Rebuild the op-index assignment from node ids, run with true costs, return (assign, E, per-request)."""
    rid = {P1.replay_id(o['id']): i for i, o in enumerate(q.OPS) if P1.replay_id(o['id'])}
    assign = {rid[n]: p for n, p in asg['node_points'].items() if n in rid}
    sc = q.run(P1.K, assign, P1.SUP, coarse); E = q.price(P1.K, P1.SUP, assign, sc, pg=True)[0]
    return assign, E, per_req_fn(outs)(sc[3]), sc


def job(spec):
    name, model, plan, kw, coarse, aware, bt, bd, mean_cost = spec
    outs = kw.get('outputs', 32); orders = {}; best = None
    for tag, P in (('direct', P1), ('hierarchical', P2)):
        asg, wall = plan_once(P, model, plan, kw, coarse, aware, bt, bd, mean_cost)
        fo = asg['first_order']; R0 = [(a*1e12, b*1e12) for a, b in fo['ref_ttft_tpot_s']]
        assign, E, R, sc = true_eval(asg, coarse, outs); rounds = 0
        def bad(R): return [(r, ph) for r, ((t0, d0), (t, d)) in enumerate(zip(R0, R)) for ph, x, y, b in (('P', t, t0, bt), ('D', d, d0, bd)) if x > y*(1+b)+1]
        pre = [dict(request=r, objective='TTFT' if ph == 'P' else 'TPOT',
                    over_objective_pct=100*((R[r][0]/(R0[r][0]*(1+bt))) if ph == 'P' else (R[r][1]/(R0[r][1]*(1+bd))))-100) for r, ph in bad(R)]
        E_pre = E
        while bad(R) and rounds < 8:  # repair: raise the violating phase of each violating request by one level
            for r, ph in bad(R):
                for i in list(assign):
                    oid = q.OPS[i]['id']
                    if oid.startswith(f'r{r}/t') and ((ph == 'P') == (oid.split('/')[1] == 't0')):
                        assign[i] -= 1
                        if not assign[i]: del assign[i]
            sc = q.run(P1.K, assign, P1.SUP, coarse); E = q.price(P1.K, P1.SUP, assign, sc, pg=True)[0]; R = per_req_fn(outs)(sc[3]); rounds += 1
        orders[tag] = dict(planner_estimate_saving=fo['saving_vs_pg_cg'], saving_vs_pg_cg=100*(1-E/fo['pg_cg_j']), final_j=E, pg_cg_j=fo['pg_cg_j'],
                           met=not bad(R), repair_rounds=rounds, certified=asg['planner_certified_nominal'], planner_wall_s=wall,
                           per_request_s=[(a*1e-12, b*1e-12) for a, b in R], nominal_per_request_s=fo['ref_ttft_tpot_s'],
                           makespan_s=sc[0]*1e-12, tokens=len(R)*outs,
                           pre_repair_violations=pre, requests=len(R0), pre_repair_saving=100*(1-E_pre/fo['pg_cg_j']))
    ok = [t for t in orders if orders[t]['met']] or list(orders)
    best = min(ok, key=lambda t: orders[t]['final_j'])
    return dict(name=name, model=model, plan=plan, workload=kw, coarse=coarse, aware=aware, baseline='mean' if mean_cost else ('aware' if aware else 'worst'),
                budget_ttft=bt, budget_tpot=bd, choice=best, orders=orders, **orders[best])


def specs(sweep):
    full = lambda p, o=128, arr=None: dict(mode='full', prompts=[p]*(len(arr) if arr else 3), outputs=o, **({'arrivals': arr} if arr else {}))
    out = []
    if sweep == 'agnmean':
        for c in (True, False):
            out += [('qwen_b5.5_t5_d5', 'qwen', F_QWEN, full(1920), c, False, .05, .05, True),
                    ('qwen_b5.5_t25_d5', 'qwen', F_QWEN, full(1920), c, False, .25, .05, True),
                    ('llama_b5.5_t5_d5', 'llama', F_LLAMA, full(1920), c, False, .05, .05, True)]
            for b in (4, 16, 64):
                out.append((f'qwen_batch{b}', 'qwen', F_QWEN, dict(mode='decode', L=1920, R=b, B=b, outputs=32), c, False, .05, .05, True))
            out.append(('llama_batch16', 'llama', F_LLAMA, dict(mode='decode', L=1920, R=16, B=16, outputs=32), c, False, .05, .05, True))
    if sweep == 'variance':
        Fq = '/shared/s3/zqy/DSE/npu_mac/outputs/flexposit_precision_quality'
        plans = {'bimodal_window': f'{Fq}/qwen14b_bimodal_plan_v1/plan_bimodal_window_b5.5.json',
                 'bimodal_module': f'{Fq}/qwen14b_bimodal_plan_v1/plan_bimodal_module_b5.5.json',
                 'kv8_rest4': f'{Fq}/qwen14b_fisher_p4p8_plan_v1/plan_kv8_rest4.json', 'b4.5': f'{Fq}/qwen14b_fisher_p4p8_plan_v1/plan_b4.5.json'}
        for pn, pf in plans.items():
            for a, m in ((False, False), (False, True), (True, False)):
                out += [(f'{pn}_t5_d5', 'qwen', pf, full(1920), False, a, .05, .05, m), (f'{pn}_t25_d5', 'qwen', pf, full(1920), False, a, .25, .05, m)]
                out += [(f'{pn}_batch{b}', 'qwen', pf, dict(mode='decode', L=1920, R=b, B=b, outputs=32), False, a, .05, .05, m) for b in (4, 16)]
    if sweep == 'slo2':
        for bt, bd in ((.10, .05), (.50, .05), (.05, .10), (.05, .20)):
            out += [(f'slo_t{int(100*bt)}_d{int(100*bd)}', 'qwen', F_QWEN, full(1920), c, a, bt, bd, False) for c in (True, False) for a in (False, True)]
    if sweep == 'serve':
        for load, gap in (('low', 8.0), ('mid', 4.0), ('high', 2.5)):
            rng = random.Random(7); t = 0.; arr = []
            for _ in range(8): arr.append(t*1e12); t += rng.expovariate(1/gap)
            for c in (True, False):
                for a, m in ((False, False), (False, True), (True, False)):
                    out.append((f'serve_{load}', 'qwen', F_QWEN, full(1920, 64, arr), c, a, .05, .05, m))
    return out


def limit_memory():
    import resource; cap = 16 << 30  # a runaway job fails alone (MemoryError) instead of pushing the host into OOM
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))


def main():
    DEST.mkdir(exist_ok=True); jobs = [s for sw in ARGS for s in specs(sw)]
    with ProcessPoolExecutor(min(24, len(jobs)), max_tasks_per_child=1, initializer=limit_memory) as ex:
        futs = {ex.submit(job, j): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            try: r = f.result()
            except Exception as e: print('FAILED', j[0], j[4], j[5], j[8], repr(e)[:300], flush=True); continue
            tag = f"{r['name']}_{'coarse' if r['coarse'] else 'fine'}_{r['baseline']}"
            (DEST/f'{tag}.json').write_text(json.dumps(r, indent=1)+'\n')
            print(f"{tag:42s} saving {r['saving_vs_pg_cg']:5.1f}% (est {r['planner_estimate_saving']:.1f}, pre-repair {r['pre_repair_saving']:.1f}, {r['choice']}) met={r['met']} repair={r['repair_rounds']} "
                  f"viol {len(r['pre_repair_violations'])}/{2*r['requests']} max {max([v['over_objective_pct'] for v in r['pre_repair_violations']] or [0]):.1f}% wall {r['planner_wall_s']:.0f}s", flush=True)


if __name__ == '__main__': main()
