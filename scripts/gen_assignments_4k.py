"""4K (3968+128) assignments: gen_assignments_v2 on the tiled workload, per-tile or coarse, replay node ids expanded to KV tiles.
usage: QWEN_CAL=<cal.json> gen_assignments_4k.py <worst|mean|aware|coarse> <target_ttft> <plan_ttft> <round>
target_ttft is the objective the replay is judged on; plan_ttft is the budget given to the planner (scaled from the previous replay's measured use).
Output: assignments_4k/b5.5_<fine|coarse>_<costing>_ttft<T>_tpot5_self_fb<round>_p3968.json, in the format of gen_insight_assignments_v1."""
import json, sys, time
from pathlib import Path
COST, TARGET, PLAN, RND = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
COARSE = COST == 'coarse'; COST = 'aware' if COARSE else COST; PROMPT = 3968
sys.argv = ['x', 'o']; sys.path.insert(0, str(Path(__file__).resolve().parent))
import planner_qwen_longctx_v1 as P
import insight_planner_v1 as IP
import sensitivity_v6 as S6
v, q = P.v, P.q
P.KW = dict(mode='full', prompts=[PROMPT]*3, outputs=P.OUTS)
DEST = Path('/shared/s3/zqy/dac2027_p1/assignments_4k'); DEST.mkdir(exist_ok=True); BD = .05
assert q.CAL, 'QWEN_CAL not set'
mod = S6.mean_cost_module(v) if COST == 'mean' else v
v.AGN_BYTES = 'true'; mod.install(mod.load_bits(P.PLANS['b5.5']), **P.KW)
cf, bf = ('C_true', 'B_true') if COST == 'aware' else ('C_p5', 'B_p5'); per_req = P.per_req
t0 = time.monotonic(); assign, info = IP.plan(P.K, P.SUP, PLAN, BD, cf, bf, per_req, P.OUTS, coarse=COARSE); wall = time.monotonic()-t0
R0 = per_req(q.run(P.K, {}, P.SUP, COARSE)[3])
def bad(R): return [(r, ph) for r, ((a, b), (t, d)) in enumerate(zip(R0, R)) for ph, x, y, bb in (('P', t, a, PLAN), ('D', d, b, BD)) if x > y*(1+bb)+1]
sc = q.run(P.K, assign, P.SUP, COARSE); R = per_req(sc[3]); pre = len(bad(R)); rounds = 0
while bad(R) and rounds < 8:
    for r, ph in bad(R):
        for i in list(assign):
            oid = q.OPS[i]['id']
            if oid.startswith(f'r{r}/t') and ((ph == 'P') == ('/t0/' in oid)):
                assign[i] -= 1
                if not assign[i]: del assign[i]
    sc = q.run(P.K, assign, P.SUP, COARSE); R = per_req(sc[3]); rounds += 1
E = q.price(P.K, P.SUP, assign, sc, pg=True)[0]
q.MEM_MODE = 'cg'; Ecg = q.price(P.K, P.SUP, {}, q.run(P.K, {}, P.SUP, False), pg=True)[0]; q.MEM_MODE = 'dvfs'
W = json.loads((P.O/'qwen14b_wk4_w4k/request0.json').read_text()); WL = [json.loads((P.O/f'qwen14b_wk4_w4k/request{r}.json').read_text()) for r in range(3)]
nodes = {n['id'] for g in WL for n in g['nodes']}
def tiles(rid):
    """Replay ids of a first-order op: attention ops map to every KV tile of that op."""
    if rid in nodes: return [rid]
    out = [f'{rid}.t{i}' for i in range(3) if f'{rid}.t{i}' in nodes]+[f'{rid}.h{i}' for i in range(2) if f'{rid}.h{i}' in nodes]
    return out
node_points, node_units, unmapped = {}, {}, 0
for o in q.OPS:
    rid = P.replay_id(o['id'])
    if not rid or not o['unit']: continue
    t = tiles(rid); unmapped += not t
    for x in t: node_units[x] = o['unit']
for i, p in assign.items():
    rid = P.replay_id(q.OPS[i]['id'])
    if rid:
        for x in tiles(rid): node_points[x] = p
T = int(round(TARGET*100)); name = f"b5.5_{'coarse' if COARSE else 'fine'}_{COST}_ttft{T}_tpot5_self_fb{RND}_p3968"
pts = [dict(v=p['v'], mac_stretch=q.J.MAC_STRETCH[j], vu_stretch=q.J.VU_STRETCH[j]) for j, p in enumerate(q.points)]
(DEST/f'{name}.json').write_text(json.dumps(dict(status='ABLATION_VF_ASSIGNMENT', arm=name, plan='b5.5', k=P.K, coarse=COARSE, aware=COST == 'aware', costing=COST,
    budget_ttft=TARGET, budget_tpot=BD, plan_budget_ttft=PLAN, slo_reference='self', points=pts, node_points=node_points, node_units=node_units,
    planner='insight_planner_v1 on first_order_cal_v2a, replay feedback on TTFT budget', planner_certified_nominal=True,
    first_order=dict(pg_cg_j=Ecg, final_j=E, saving_vs_pg_cg=100*(1-E/Ecg), met=not bad(R), pre_repair_violations=pre, repair_rounds=rounds, planner_wall_s=wall,
                     backoff_checks=info['backoff_rounds'], lowered_ops=len(assign), ref_ttft_tpot_s=[(a*1e-12, b*1e-12) for a, b in R0])))+'\n')
print(name, 'unmapped', unmapped, 'plan_budget', PLAN, 'fo saving', round(100*(1-E/Ecg), 2), 'lowered', len(assign), 'repair', rounds, 'checks', info['backoff_rounds'], 'pre', pre, 'dTTFT', round(max(100*(t/a-1) for (a, b), (t, d) in zip(R0, R)), 2), 'dTPOT', round(max(100*(d/b-1) for (a, b), (t, d) in zip(R0, R)), 2), 'wall', round(wall, 1))
