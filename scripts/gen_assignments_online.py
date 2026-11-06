"""Causal assignments for replay: the online controller decides each op at dispatch on the calibrated first-order model, and the decisions are written in replay node ids.
usage: QWEN_CAL=<cal.json> FREQ_STATES=34 ON_TWIN=1 ON_LAGQ=<q> gen_assignments_online.py <worst|mean|aware> <target_ttft> <1920|3968>
Replay of these assignments is open loop: the decisions were taken on the model's clock, not the replay's.
Output: assignments_online/b5.5_fine_<costing>_ttft<T>_tpot5_self_online_p<prompt>.json, in the format of gen_insight_assignments_v1."""
import json, sys, time
from pathlib import Path
COST, TARGET, PROMPT = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
sys.argv = ['x', 'o']; sys.path.insert(0, str(Path(__file__).resolve().parent))
import planner_qwen_longctx_v1 as P
import online_eval_v1 as E
q, v, ON, IP, K, SUP = E.q, P.v, E.ON, E.IP, P.K, P.SUP
assert q.CAL, 'QWEN_CAL not set'
W = Path('/shared/s3/zqy/dac2027_p1'); DEST = W/'assignments_online'; OUT = Path(__import__('os').environ.get('ON_ASSIGN_DEST', DEST)); BD = .05; OUTS = P.OUTS
v.AGN_BYTES = 'worst' if __import__('os').environ.get('ON_AGN') == '1' else 'true'; v.install(v.load_bits(P.PLANS['b5.5']), mode='full', prompts=[PROMPT]*3, outputs=OUTS); E.add_costings(); per_req = P.per_req
q.MEM_MODE = 'cg'; s0 = q.run(K, {}, SUP, False); Ecg, comp, _ = q.price(K, SUP, {}, s0, pg=True); nom = ON.nominal_dispatch(K, SUP) if E.TWIN else None; q.MEM_MODE = 'dvfs'
R0 = per_req(s0[3]); bg = IP._bg_power(comp, s0[0])
FAIR = __import__('os').environ.get('ON_FAIR') == '1'; SLO = OWN = None
if FAIR and COST != 'nominal':
    SLO = dict(ttft=[a for a, b in R0], tpot=[b for a, b in R0]); q.MEM_MODE = 'cg'; OWN = ON.own_twin(K, SUP, *E.FIELDS[COST], per_req, len(R0), [q.ARRIVAL_END.get(f'arr{j}', 0.) for j in range(len(R0))]); q.MEM_MODE = 'dvfs'; nom = None
if COST == 'nominal':  # reference replay of the catalog: no lowered node, every node at point 0 (VU base period included)
    assign = {}; sc = s0; info = dict(decisions=0, decide_s=0.); wall = 0.
else: t0 = time.monotonic(); assign, sc, info = ON.run_online(K, SUP, TARGET, BD, 0., E.G, bg, *E.FIELDS[COST], lag_q=E.LQ, hedge=True, warm=8, nom=nom, open_loop=__import__('os').environ.get('ON_OPENLOOP') == '1', slo=SLO, own=OWN); wall = time.monotonic()-t0
R = per_req(sc[3]); Efin = ON.energy(K, SUP, assign, sc)
bad = [(r, ph) for r, ((a, b), (t, d)) in enumerate(zip(R0, R)) for ph, x, y, bb in (('P', t, a, TARGET), ('D', d, b, BD)) if x > y*(1+bb)+1]
def tiles(rid):
    if rid in NODES: return [rid]
    return [f'{rid}.t{i}' for i in range(3) if f'{rid}.t{i}' in NODES]+[f'{rid}.h{i}' for i in range(2) if f'{rid}.h{i}' in NODES]
if PROMPT == 1920: tiles = lambda rid: [rid]
else: NODES = set(json.loads((DEST/'wk4_w4k_nodes.json').read_text()))
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
node_mem = {}
for i, p in q.MEMPLAN.items():
    rid = P.replay_id(q.OPS[i]['id'])
    if rid and p:
        for x in tiles(rid): node_mem[x] = q.J.MEM_STRETCH[p]
T = int(round(TARGET*100)); name = f'b5.5_fine_{COST}_ttft{T}_tpot5_self_online_p{PROMPT}'
pts = [dict(v=p['v'], mac_stretch=q.J.MAC_STRETCH[j], vu_stretch=q.J.VU_STRETCH[j], att_stretch=q.J.ATT_STRETCH[j]) for j, p in enumerate(q.points)]
(OUT/f'{name}.json').write_text(json.dumps(dict(status='ABLATION_VF_ASSIGNMENT', arm=name, rails=int(__import__('os').environ.get('RAILS', 7)), node_mem=node_mem, plan='b5.5', k=K, coarse=False, aware=COST == 'aware', costing=COST,
    budget_ttft=TARGET, budget_tpot=BD, plan_budget_ttft=TARGET, slo_reference='self', points=pts, node_points=node_points, node_units=node_units,
    planner=f'online_planner_v1 guard={E.G} lag_q={E.LQ} twin={E.TWIN}; replay is open loop', planner_certified_nominal=True,
    first_order=dict(pg_cg_j=Ecg, final_j=Efin, saving_vs_pg_cg=100*(1-Efin/Ecg), met=not bad, planner_wall_s=wall, decisions=info['decisions'], decide_s=info['decide_s'], lowered_ops=len(assign), ref_ttft_tpot_s=[(a*1e-12, b*1e-12) for a, b in R0])))+'\n')
print(name, 'lq', E.LQ, 'twin', E.TWIN, 'pts', len(q.points), 'saving', round(100*(1-Efin/Ecg), 2), 'lowered', len(assign), 'nodes', len(node_points), 'unmapped', unmapped, 'bad', bad, 'dTTFT', round(max(100*(t/a-1) for (a, b), (t, d) in zip(R0, R)), 2), 'dTPOT', round(max(100*(d/b-1) for (a, b), (t, d) in zip(R0, R)), 2), 'ms/decision', round(1e3*info['decide_s']/max(info['decisions'], 1), 3), flush=True)
