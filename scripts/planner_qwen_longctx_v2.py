"""PIVOT planner v2: hierarchical slack allocation (precision-aware arms refine the agnostic grouping).

v1 lowered precision-aware groups (core, family, layer type, precision class) directly; on decode
batches its greedy could end below the agnostic arm. v2 first lowers the agnostic groups (with true
costs for aware arms), then, for aware arms, splits each group by precision class and continues,
so an aware arm never ends below the agnostic grouping evaluated with exact costs.

v1 description:

Planner = the paper's Section 4 algorithm on the first-order request model (shared-HBM FCFS
schedule, uniform clock-gating ledger, power gating):
  1. free slowdowns: per matrix op, the lowest MAC point whose stretched compute + IVR
     transition stays within its HBM time;
  2. slack allocation: groups (core, family, layer type[, precision class if aware]) lowered
     greedily while energy drops and every request meets TTFT <= (1+b_ttft) TTFT_ref and
     mean TPOT <= (1+b_tpot) TPOT_ref.
Precision-agnostic arms cost every matrix tile at the plan's widest precision (cycles only;
weight bytes are the real DMA byte counts, AGN_BYTES='true'). Coarse arms use one point for all
domains of a core per op. ref='p8': objectives provisioned for uniform-P8 weights (first-order
P8 latencies). The assignment is written in replay node ids; the calibrated replay verifies it
and violating windows are repaired by repair_qwen_longctx_v1.py.
"""
import collections, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ARGS = sys.argv[1:]; sys.argv = ['x', 'o']; sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_first_order_v3 as v, qwen_first_order_v1 as q
v.AGN_BYTES = 'true'
SUP = 'hybrid_scvr_dldo'; K = 256; PROMPT, OUTS = 1920, 128
O = Path('/shared/s3/zqy/DSE/npu_mac/outputs/flexposit_enpu_paper/asap7_calibration')
F = '/shared/s3/zqy/DSE/npu_mac/outputs/flexposit_precision_quality/qwen14b_fisher_p4p8_plan_v1'
PLANS = {'p45': 'p45', 'b5.5': f'{F}/plan_b5.5.json'}
KW = dict(mode='full', prompts=[PROMPT]*3, outputs=OUTS)
NAMES = {'input_norm': 'input_layernorm/compiled_row_program', 'q_proj': 'self_attn.q_proj', 'k_proj': 'self_attn.k_proj',
         'v_proj': 'self_attn.v_proj', 'q_bias': 'self_attn.q_proj.bias', 'k_bias': 'self_attn.k_proj.bias', 'v_bias': 'self_attn.v_proj.bias',
         'rope_q': 'rotary_qk.q', 'rope_k': 'rotary_qk.k', 'kv_append': 'kv_append', 'attn_qk': 'self_attn.attn_qk',
         'softmax': 'masked_softmax/compiled_row_program', 'attn_av': 'self_attn.attn_v', 'o_proj': 'self_attn.o_proj',
         'residual1': 'residual_add.attention', 'post_norm': 'post_attention_layernorm/compiled_row_program',
         'gate_proj': 'mlp.gate_proj', 'up_proj': 'mlp.up_proj', 'silu_mul': 'mlp.act_mul/compiled_row_program',
         'down_k0': 'mlp.down_proj.k0', 'down_k1': 'mlp.down_proj.k1', 'ksplit_add': 'mlp.down_proj.ksplit_add',
         'residual2': 'residual_add.mlp'}
TOP = {'final_norm': 'final_norm/compiled_row_program', 'lm_head': 'lm_head', 'emit': 'emit', 'embedding': 'embedding'}


def replay_id(fid):
    """First-order op id -> replay node id (None for first-order-only ops such as the FP32 cast)."""
    parts = fid.split('/')
    if len(parts) == 4:
        r, t, layer, name = parts
        return f'{r}/{t}/model.layers.{layer[1:]}.{NAMES[name]}' if name in NAMES else None
    r, t, name = parts
    return f'{r}/{t}/{TOP[name]}' if name in TOP else None


def per_req(lat):
    return [(lat[i], sum(lat[i+1:i+OUTS])/(OUTS-1)) for i in range(0, len(lat), OUTS)]


def ref_latency(plan, ref):
    if ref == 'self': return None
    bits = v.load_bits(PLANS[plan]); v.install({n: [8]*len(b) for n, b in bits.items()}, **KW)
    return q.run(K, {}, SUP, False)[3]


def arm(args):
    plan, coarse, aware, b_ttft, b_tpot, ref, lat_ref = args
    v.install(v.load_bits(PLANS[plan]), **KW)
    r = q.RATIO(SUP); cf, bf = ('C_true', 'B_true') if aware else ('C_p5', 'B_p5')
    lat0 = lat_ref or q.run(K, {}, SUP, coarse)[3]; R0 = per_req(lat0)
    q.MEM_MODE = 'cg'; Ecg = q.price(K, SUP, {}, q.run(K, {}, SUP, False), pg=True, mem_dvfs=True)[0]; q.MEM_MODE = 'dvfs'
    def ok(lat): return all(t <= t0*(1+b_ttft)+1 and d <= d0*(1+b_tpot)+1 for (t0, d0), (t, d) in zip(R0, per_req(lat)))
    def plan_E(asg):
        sc = q.run(K, asg, SUP, coarse, cf, bf)
        return q.price(K, SUP, asg, sc, pg=True)[0] if ok(sc[3]) else None
    assign = {}
    for i, o in enumerate(q.OPS):
        if o['unit'] != 'MATRIX': continue
        c = o[cf]/K; m = o[bf]/q.BW_R*1e12; best = 0
        for pi in range(1, len(q.points)):
            if c*q.J.MAC_STRETCH[pi]+(.7-q.points[pi]['v'])*q.J.IVR_PS_PER_V*2 <= m and r[pi] < r[best]: best = pi
        if best: assign[i] = best
    E = plan_E(assign)
    if E is None: assign = {}; E = plan_E({})
    certified = E is not None
    evals = 0
    for refine in ((False, True) if aware else (False,)):
        G = collections.defaultdict(list)
        for i, o in enumerate(q.OPS):
            if o['unit'] and certified:
                G[q.group_of(o)+((min(2, int(o['p4']*3)),) if refine and o['unit'] == 'MATRIX' else ())].append(i)
        level = collections.Counter({g: min(assign.get(i, 0) for i in idx) for g, idx in G.items()}); changed = True
        while changed:
            changed = False
            for g, idx in G.items():
                while level[g]+1 < len(q.points):
                    trial = dict(assign)
                    for i in idx: trial[i] = max(trial.get(i, 0), level[g]+1)
                    Et = plan_E(trial); evals += 1
                    if Et is None or Et >= E-1e-9: break
                    E, assign = Et, trial; level[g] += 1; changed = True
    sc = q.run(K, assign, SUP, coarse); Ef = q.price(K, SUP, assign, sc, pg=True)[0]
    node_points, node_units, dropped = {}, {}, 0
    for i, pi in assign.items():
        rid = replay_id(q.OPS[i]['id'])
        if rid is None: dropped += 1; continue
        node_points[rid] = pi; node_units[rid] = q.OPS[i]['unit']
    for i, o in enumerate(q.OPS):  # units of every replayed compute node, so the replay can price transitions
        rid = replay_id(o['id'])
        if rid and o['unit']: node_units.setdefault(rid, o['unit'])
    name = f"{plan}_{'coarse' if coarse else 'fine'}_{'aware' if aware else 'agnostic'}_ttft{int(100*b_ttft)}_tpot{int(100*b_tpot)}_{ref}"
    pts = [dict(v=p['v'], mac_stretch=q.J.MAC_STRETCH[j], vu_stretch=q.J.VU_STRETCH[j]) for j, p in enumerate(q.points)]
    return name, dict(status='ABLATION_VF_ASSIGNMENT', arm=name, plan=plan, k=K, coarse=coarse, aware=aware, budget_ttft=b_ttft,
                      budget_tpot=b_tpot, slo_reference=ref, points=pts, node_points=node_points, node_units=node_units,
                      planner='PIVOT first-order (planner_qwen_longctx_v2, hierarchical)', planner_certified_nominal=certified,
                      first_order=dict(pg_cg_j=Ecg, final_j=Ef, saving_vs_pg_cg=100*(1-Ef/Ecg), met=ok(sc[3]), evaluations=evals,
                                       lowered_ops=len(assign), dropped_first_order_only_ops=dropped,
                                       ref_ttft_tpot_s=[(a*1e-12, b*1e-12) for a, b in R0]))


if __name__ == '__main__':
    dest = O/'qwen14b_longctx_assignments_v2'; dest.mkdir(exist_ok=True)
    jobs = [(p, c, a, .05, .05, 'self', None) for p in PLANS for c in (True, False) for a in (False, True)]
    lat8 = ref_latency('b5.5', 'p8')
    jobs += [('b5.5', c, a, .25, .05, 'p8', lat8) for c in (True, False) for a in (False, True)]
    with ProcessPoolExecutor(len(jobs)) as ex:
        for name, asg in ex.map(arm, jobs):
            (dest/f'{name}.json').write_text(json.dumps(asg)+'\n'); fo = asg['first_order']
            print(f"{name:45s} lowered {fo['lowered_ops']:6d} first-order vs PG+CG {fo['saving_vs_pg_cg']:5.1f}% met={fo['met']} certified={asg['planner_certified_nominal']}", flush=True)
