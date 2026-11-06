"""Qwen2.5-14B first-order request model + 2x2 DVFS ablation (same method as Phi-2 ablation_v4/v5).

Operators per token follow Qwen2 (serial attention->MLP, GQA 40/8, qkv bias, SwiGLU,
down_proj as two K=6912 halves + FP16 partial-sum add, FP16 lm_head on ATT with 32-column
tiles, 4-chunk argmax). Compute times at k=1 come from the Qwen calibrations:
  matrix: context tile anchors (holdout error 0%), x ceil(m/16) row blocks
  vector: phased-SRAM per-row records; attention: phased per-command anchors;
  lm_head: phased per-tile anchor (last-row projection); HBM: 16 channels.
Energy coefficients are the calibrated Phi-2 per-component ledger rescaled to Qwen work
(MACs, VU/ATT busy time, HBM/SRAM bytes); HBM controller background x4 for 16 channels.
First-order: per-op max(C/k, B/BW), no HBM contention between P and D. Not a replay.
"""
import collections, contextlib, functools, importlib.util, io, json, math, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARGV = list(sys.argv)
spec = importlib.util.spec_from_file_location('a5', HERE/'ablation_v5.py'); sys.argv = ['x', 'out_tmp']
with contextlib.redirect_stdout(io.StringIO()):
    A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
spec = importlib.util.spec_from_file_location('r5', HERE/'ablation_v5_run.py')
with contextlib.redirect_stdout(io.StringIO()):
    R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
sys.argv = ARGV
J, L, S, points = A.J, A.L, A.S, A.points
O = Path('/shared/s3/zqy/DSE/npu_mac/outputs/flexposit_enpu_paper/asap7_calibration')
Q = Path('/shared/s3/zqy/DSE/npu_mac/outputs/flexposit_precision_quality')

H, I, LAYERS, HEADS, KV, HD, VOCAB = 5120, 13824, 48, 40, 8, 128, 152064
plan = json.loads((Q/'qwen14b_artifact_channel_plan_v1/result.json').read_text())['modules']
cap = json.loads((O/'qwen14b_nonlinear_real_vectors_v1/result.json').read_text())
anch = json.loads((O/'qwen14b_phased_request_timing_calibration_v1/tile_anchor_model.json').read_text())['anchors']
vec = {r['case']: r['elapsed_ps'] for r in json.loads((O/'qwen14b_phased_vector_calibration_v1/result.json').read_text())['rows']}
att = {r['case']: r for r in json.loads((O/'qwen14b_phased_attention_head_v1/result.json').read_text())['rows']}
BW_R, BW_W = 338e9, 394e9  # 16-channel streaming occupancy (hbm_streaming_occupancy_ch16_q512_v1)


def V(name): return vec[f'VEC_{name}_nominal']


FALLBACK = {}
for _k, _v in anch.items():
    FALLBACK.setdefault(_k.rsplit('|', 1)[0], _v)


def tile_ps(k, phase, p, nxt, idx, n_tiles):
    ctx = 1 if idx == 0 else (4 if idx == n_tiles-1 else 2)
    key = f'{k}|{phase}|{p}|1111|1200|{ctx}|{nxt if ctx != 4 else None}'
    # a precision pattern absent from the five-tile cases falls back to the same context
    return anch[key] if key in anch else FALLBACK[key.rsplit('|', 1)[0]]


import functools


sys.path.insert(0, '/shared/s3/zqy/DSE/npu_mac/scripts')
from flexposit_channel_prefetch_plan import compile_channel_matrix  # noqa: E402
from flexposit_channel_matrix_memory_plan import summarize  # noqa: E402


@functools.lru_cache(None)
def matrix_C(name, k, phase, m, kslice=None):
    """Compute time from tile anchors; HBM/SRAM traffic from the real channel compiler."""
    bits = plan[name]['channel_precision']; tiles = [bits[i] for i in range(0, len(bits), 32)]
    per = sum(tile_ps(k, phase, p, tiles[i+1] if i+1 < len(tiles) else None, i, len(tiles)) for i, p in enumerate(tiles))
    mp = m if m == 1 else math.ceil(m/16)*16  # learned-array row contract: whole 16-row blocks
    c = summarize(compile_channel_matrix(mp, len(bits), k, bits, 'activation_resident', a_capacity=524288, w_capacity=524288))
    tot = c[0]  # summarize() -> (counts, ...); counts matches the Phi-2 template 'counts' field
    sram = sum(v for kk, v in tot.items() if kk.startswith('sram_') and kk.endswith('_bytes'))
    prec = collections.Counter(); [prec.__setitem__(p, prec[p]+32*k*mp) for p in tiles]
    return per*(mp//16 if mp > 1 else 1), tot['hbm_total_bytes'], tuple(sorted(prec.items())), tot['array_mac_operations'], sram


ATT_PS = {(ph, kind): att[f'r1_l23_t{0 if ph=="P" else 31}_{kind}_nominal_base']['elapsed_ps'] /
          (att[f'r1_l23_t{0 if ph=="P" else 31}_{kind}_nominal_base']['shape']['m'] *
           (1 if kind == 'QK' else 1)*att[f'r1_l23_t{0 if ph=="P" else 31}_{kind}_nominal_base']['shape']['k'])
          for ph in ('P', 'D') for kind in ('QK', 'AV')}
HEAD_TILE = att['D_qwen14b_fp16_head_nominal_base']['elapsed_ps']/4


def build_ops():
    ops, deps = [], {}
    def add(oid, engine, family, unit, C, B=0, prec=None, macs=0, sram=0, after=()):
        ops.append(dict(id=oid, engine=engine, family=family, unit=unit, C=C, C_true=C, B=B, B_true=B, prec=prec or {},
                        macs=macs, sram_bytes=sram, residual=0.)); deps[oid] = list(after); return oid
    for r in cap['requests']:
        rid = r['request']; P0 = len(r['prompt_ids']); prev = None
        for t in range(32):
            ph = 'P' if t == 0 else 'D'; m = P0 if t == 0 else 1; ctx = P0+t; eng = ph; tag = f'r{rid}/t{t}'
            first = [prev] if prev else []
            x = add(f'{tag}/embedding', eng, 'state', None, 0., B=m*H*2, after=first)
            for l in range(LAYERS):
                pre = f'{tag}/L{l}'
                n1 = add(f'{pre}/input_norm', eng, 'vector', 'VU', V('rmsnorm')*m, after=[x])
                qkv = []
                for proj, n in (('q', H), ('k', KV*HD), ('v', KV*HD)):
                    name = f'model.layers.{l}.self_attn.{proj}_proj'
                    C, B, prec, macs, sr = matrix_C(name, H, ph, m); prec = dict(prec)
                    mm = add(f'{pre}/{proj}_proj', eng, 'matrix', 'MATRIX', C, B, prec, macs, sram=sr, after=[n1])
                    qkv.append(add(f'{pre}/{proj}_bias', eng, 'vector', 'VU', V('bias_add_n5120' if proj == 'q' else 'bias_add_n1024')*m, after=[mm]))
                rq = add(f'{pre}/rope_q', eng, 'vector', 'VU', V('rotary_n128')*m*HEADS, after=[qkv[0]])
                rk = add(f'{pre}/rope_k', eng, 'vector', 'VU', V('rotary_n128')*m*KV, after=[qkv[1]])
                kv = add(f'{pre}/kv_append', eng, 'state', None, 0., B=m*2*KV*HD*2, after=[rk, qkv[2]])
                cv = add(f'{pre}/fp32_cast', eng, 'vector', 'VU', V('FP16->FP32_n128')*(m*HEADS+ctx*KV), B=ctx*2*KV*HD*2, after=[rq, kv])
                qk = add(f'{pre}/attn_qk', eng, 'attention', 'ATT', ATT_PS[(ph, 'QK')]*HEADS*m*math.ceil(ctx/32)*HD, after=[cv])
                sm = add(f'{pre}/softmax', eng, 'vector', 'VU', V('scaled_masked_softmax')*m*HEADS*ctx/270, after=[qk])
                av = add(f'{pre}/attn_av', eng, 'attention', 'ATT', ATT_PS[(ph, 'AV')]*HEADS*m*(HD//32)*ctx, after=[sm])
                C, B, prec, macs, sr = matrix_C(f'model.layers.{l}.self_attn.o_proj', H, ph, m); prec = dict(prec)
                o = add(f'{pre}/o_proj', eng, 'matrix', 'MATRIX', C, B, prec, macs, sram=sr, after=[av])
                x1 = add(f'{pre}/residual1', eng, 'vector', 'VU', V('residual_add_n5120')*m, after=[o, x])
                n2 = add(f'{pre}/post_norm', eng, 'vector', 'VU', V('rmsnorm')*m, after=[x1])
                gu = []
                for proj in ('gate', 'up'):
                    C, B, prec, macs, sr = matrix_C(f'model.layers.{l}.mlp.{proj}_proj', H, ph, m); prec = dict(prec)
                    gu.append(add(f'{pre}/{proj}_proj', eng, 'matrix', 'MATRIX', C, B, prec, macs, sram=sr, after=[n2]))
                act = add(f'{pre}/silu_mul', eng, 'vector', 'VU', V('silu_mul')*m, after=gu)
                halves = []
                for h in range(2):
                    name = f'model.layers.{l}.mlp.down_proj'
                    C, B, prec, macs, sr = matrix_C(name, 6912, ph, m); prec = dict(prec)
                    halves.append(add(f'{pre}/down_k{h}', eng, 'matrix', 'MATRIX', C, B, prec, macs, sram=sr, after=[act]))
                ps = add(f'{pre}/ksplit_add', eng, 'vector', 'VU', V('residual_add_n5120')*m, after=halves)
                x = add(f'{pre}/residual2', eng, 'vector', 'VU', V('residual_add_n5120')*m, after=[ps, x1])
            fn = add(f'{tag}/final_norm', eng, 'vector', 'VU', V('rmsnorm'), after=[x])  # last-row projection
            lm = add(f'{tag}/lm_head', eng, 'fp16', 'ATT', HEAD_TILE*(VOCAB//32), B=VOCAB*H*2, after=[fn])
            prev = add(f'{tag}/emit', eng, 'vector', 'VU', 4*V('argmax_chunk_n38016')+V('argmax_combine_n4'), after=[lm])
    return ops, deps


OPS, DEPS = build_ops()
for o in OPS:
    o['p4'] = o['prec'].get(4, 0)/max(1, sum(o['prec'].values()))
    o['C_p5'] = o['C']*(sum(o['prec'].values())*5/max(1, sum(p*n for p, n in o['prec'].items()))) if o['prec'] else o['C']
    o['B_p5'] = o['B']*(5/(5-o['p4'])) if o['prec'] else o['B']
UNIT = {'MATRIX': 'MATRIX', 'VU': 'VU', 'ATT': 'ATT'}
RATIO = lambda supply: [L.ratio(p['v'], L.SUPPLY[supply]) for p in points]
import os as _o2
RAIL = {'VU': 'VEC', 'ATT': 'VEC'} if _o2.environ.get('RAILS') == '4' else {}  # 4 rails per core: MAC, VU+ATT, CTRL+DEC+OUT, MEM
STRETCH = {'MATRIX': J.MAC_STRETCH, 'VU': J.VU_STRETCH, 'ATT': J.ATT_STRETCH}
CAP = sum(J.cap.values())/len(J.cap)  # per-array MEM bytes/ps capacity (same peripheral hardware as Phi-2)
TOTAL_MACS = sum(o['macs'] for o in OPS); TOTAL_B = sum(o['B'] for o in OPS)


LEDGER = 'strict'  # 'strict': clock gating in every arm; 'enpu': eNPU convention (coarse arms not gated)
ARRIVAL_END = {}  # pseudo-op id -> request arrival time (ps)
OUT_TOKENS = 32
OUT_LIST = None  # per-request output counts (serve_var_v1); None: every request emits OUT_TOKENS
MEM_MODE = 'dvfs'  # 'dvfs': MEM retained logic DVFS to utilisation; 'cg': clock gating at nominal V
DUR = {}  # last run: op index -> (compute ps, occupied ps)
# Replay calibration, frozen from the nominal replay only (cal_first_order_v1.py). Off unless QWEN_CAL names the JSON.
# For (engine, family): t = max(c', wait+m) + a*min(c0, m) + y*c0 + b*(c'-c0) + pc*m while the other core is active, c0 the nominal compute time, c' = c0+g*(c-c0);
# 'absorb' subtracts HBM wait from that overhead. a, y, pc come from the nominal replay; g, b from one DVFS replay (fine_aware_ttft10); every other arm is hold-out.
import os as _os, json as _json
CAL = _json.loads(open(_os.environ['QWEN_CAL']).read()) if _os.environ.get('QWEN_CAL') else None


MEMPLAN = {}  # op -> MEM point past its utilisation: the op takes at least the stretched MEM busy time
LIVE = None  # causal controller: called at each op's dispatch with what is known at that time, returns the op's point


def run(k, assign, supply, coarse, cfield='C_true', bfield='B_true'):
    eta = L.SUPPLY[supply]; end, free, last = dict(ARRIVAL_END), collections.defaultdict(float), {}
    busy = collections.defaultdict(list); rho_t = collections.defaultdict(float); emit = {}
    # Shared HBM: P and D engines each run their ops in program order; the HBM transfer of
    # every op is booked FCFS (by op start time) on one shared bandwidth server, so total
    # HBM busy time can never exceed the makespan.
    queues = {e: collections.deque(i for i, o in enumerate(OPS) if o['engine'] == e) for e in ('P', 'D')}
    hbm_free = 0.
    while any(queues.values()):
        cand = []
        for e, qq in queues.items():
            if qq and all(d in end for d in DEPS[OPS[qq[0]]['id']]):
                cand.append((max([free[e]]+[end[d] for d in DEPS[OPS[qq[0]]['id']]]), e))
        s, e = min(cand); i = queues[e].popleft(); o = OPS[i]
        pi = LIVE(i, s, hbm_free, queues, free) if LIVE else assign.get(i, 0); u = o['unit']
        c = o[cfield]/k*(STRETCH[u][pi] if u else 1.)
        m = o[bfield]/BW_R*1e12
        trans = 0.
        if u:
            key = (o['engine'], 'CORE' if coarse else RAIL.get(u, u)); trans = abs(last.get(key, .7)-points[pi]['v'])*J.IVR_PS_PER_V; last[key] = points[pi]['v']
        mem_busy = o['sram_bytes']/k/CAP if o['sram_bytes'] else 0.
        cal = CAL.get(f"{o['engine']}/{o['family']}") if CAL else None
        c0 = o[cfield]/k
        if cal and 'g' in cal: c = c0+cal['g']*(c-c0)  # replay's response to a stretch, fitted on one DVFS arm
        hs = max(s, hbm_free) if m else s; hbm_free = hs+m if m else hbm_free
        t = max(c+trans, hs+m-s)
        if cal:  # fixed overhead at nominal cost: does not stretch with V/f
            extra = cal.get('a', 0.)*min(c0, m)+cal.get('y', 0.)*c0
            extra += cal.get('b', 0.)*max(0., c-c0)
            if cal.get('pc') and m:  # the other core is active: fine-grained HBM sharing that the op-level FCFS booking misses
                oe = 'D' if o['engine'] == 'P' else 'P'
                if queues[oe] and free[oe] > s: extra += cal['pc']*m
            if cal.get('absorb'): extra = max(0., extra-(hs-s))
            t += extra
        if MEMPLAN.get(i): t = max(t, mem_busy*J.MEM_STRETCH[MEMPLAN[i]])
        if coarse and u: t = max(t, mem_busy*J.MEM_STRETCH[pi]+trans)
        end[o['id']] = s+t; free[o['engine']] = s+t; DUR[i] = (c, t)
        busy[o['engine']].append((s, s+t))
        uu = min(1., mem_busy/t) if mem_busy else (min(1., (o[cfield]/k)/t) if u and t > 0 else 1.)
        if coarse and LEDGER == 'enpu':  # eNPU convention: one compute domain, MEM clocked with it, no separate gating
            rho = (L.ratio(points[pi]['v'], eta)/J.MEM_STRETCH[pi]) if u else 1.
        elif coarse:  # strict: uniform clock gating, MEM shares the core rail, idle MEM cycles gated (v5 ledger)
            rho = L.ratio(points[pi]['v'], eta)*uu if u else uu
        else:
            rho = 1.
            for qi, q in enumerate(points):
                if MEM_MODE == 'cg' and qi: break  # clock gating only: nominal V, idle MEM cycles gated
                if J.MEM_STRETCH[qi] > 1/max(uu, 1e-9)+1e-12: break  # stretch only grows along the catalog
                rho = min(rho, (q['v']/.7)**2*eta(.7)/eta(q['v'])*uu)
        rho_t[o['engine']] += rho*t
        if o['id'].endswith('/emit'): emit[o['id']] = s+t
    lat = []
    for r in cap['requests']:
        times = [emit[f"r{r['request']}/t{t}/emit"] for t in range(OUT_LIST[r['request']] if OUT_LIST else OUT_TOKENS)]
        lat += [times[0]-ARRIVAL_END.get(f"arr{r['request']}", 0.)]+[b-a for a, b in zip(times, times[1:])]
    return max(end.values()), busy, rho_t, lat


def price(k, supply, assign, sched, pg=True, mem_dvfs=True, alpha=None, coarse=False):
    T, busy, rho_t, _ = sched; eta = L.SUPPLY[supply]; r = RATIO(supply); Ts = T*1e-12
    alpha = (R.alpha_for(k) if k > 1 else 0.) if alpha is None else alpha
    lt, lc = S.ledger_time, S.ledger_const; T0 = S.T0*1e-12
    phi_sram = sum(o['sram_bytes'] for o in A.ops); q_sram = sum(o['sram_bytes'] for o in OPS)
    e = dict(dram_die_background=lt['dram_die_background']/T0*Ts, controller_background=lt['controller_background']/T0*Ts*4,
             frontend_controller=lt['frontend_controller']/T0*Ts, pd_link_fifo=lt['pd_link_fifo']/T0*Ts,
             array_leakage=S.LEAK_J_K1/T0*k*Ts,
             dram_activity=lc['dram_activity']*TOTAL_B/S.B_total, io_activity=lc['io_activity']*TOTAL_B/S.B_total,
             controller_activity=lc['controller_activity']*TOTAL_B/S.B_total, dma=lc['dma']*TOTAL_B/S.B_total,
             sram=lc['sram']*q_sram/phi_sram, other_logic_work=J.other_work_in*TOTAL_MACS/S.macs_total)
    mac = vu = at = 0.
    for i, o in enumerate(OPS):
        pi = assign.get(i, 0)
        if o['unit'] == 'MATRIX': mac += S.mac_in/S.macs_total*o['macs']*o.get('mac_e', 1.)*r[pi]
        elif o['unit'] == 'VU': vu += J.vu_in/J.vu_C*o['C']*r[pi]
        elif o['unit'] == 'ATT': at += J.att_in/J.att_C*o['C']*r[pi]
    e.update(matrix_core=mac, vu=vu, att=at)
    pw = J.RET_W_PER_ENGINE*k**alpha*1e-12; idle = pgsave = 0.; ret = 0.
    for eng in ('P', 'D'):
        iv = sorted(busy[eng]); ret += pw*(rho_t[eng] if mem_dvfs else sum(b-a for a, b in iv))
        gaps = [iv[0][0]]+[max(0, iv[j+1][0]-iv[j][1]) for j in range(len(iv)-1)]+[T-iv[-1][1]] if iv else [T]
        idle += pw*sum(gaps)
        if pg: pgsave += pw*sum(g-J.BET_PS for g in gaps if g > J.BET_PS)*(1-J.PG_RESIDUAL)
    e['retained_logic_busy'] = ret; e['retained_logic_idle'] = idle-pgsave
    hbm_u = TOTAL_B/BW_R/Ts
    e['hbm_controller_pg_credit'] = -(e['controller_background']*S.PG_FRACTION_K1*max(0, 1-hbm_u)/(1-S.B_total/24e9/T0) if pg else 0.)
    if TRANS_DECAP and assign: e['rail_transition'] = rail_transition_j(k, supply, assign, coarse)
    return sum(e.values()), e, hbm_u


TRANS_DECAP = float(_os.environ.get('QWEN_TRANS_DECAP', 0))  # rail capacitance / switched capacitance; 0 keeps the pre-existing ledger
F_NOM = 1e9  # Hz, conservative lower clock gives a larger switched capacitance


def rail_transition_j(k, supply, assign, coarse=False):
    """Upper bound on rail recharge energy: every upward step costs C_rail*V*dV/eta and downward steps recover nothing.
    C_rail = TRANS_DECAP * switched capacitance of the domain, from its nominal dynamic power; MEM logic follows the MATRIX rail's steps."""
    eta = L.SUPPLY[supply]; E = {'MATRIX': S.mac_in/S.macs_total*sum(o['macs']*o.get('mac_e', 1.) for o in OPS if o['unit'] == 'MATRIX'),
                                 'VU': J.vu_in/J.vu_C*sum(o['C'] for o in OPS if o['unit'] == 'VU'), 'ATT': J.att_in/J.att_C*sum(o['C'] for o in OPS if o['unit'] == 'ATT')}
    T = {u: sum(o['C_true'] for o in OPS if o['unit'] == u)/k*1e-12 for u in E}
    Csw = {u: E[u]/T[u]/(F_NOM*.49) for u in E if T[u] > 0}
    Cmem = J.RET_W_PER_ENGINE*k**R.alpha_for(k)/(F_NOM*.49)
    last, tot = {}, 0.
    for i, o in enumerate(OPS):
        u = o['unit']
        if not u: continue
        key = (o['engine'], 'CORE' if coarse else RAIL.get(u, u)); v = points[assign.get(i, 0)]['v']; pv = last.get(key, .7); last[key] = v
        if v > pv:
            c = (sum(Csw.values())+Cmem) if coarse else Csw[u]+(Cmem if u == 'MATRIX' else 0.)
            tot += TRANS_DECAP*c*v*(v-pv)/eta(v)
    return tot


def group_of(o):
    layer_type = o['id'].split('/')[-1].split('_k')[0]
    return (o['engine'], o['family'], layer_type)


def arm(args):
    k, supply, budget, coarse, aware, pg = args
    r = RATIO(supply); cf, bf = ('C_true', 'B_true') if aware else ('C_p5', 'B_p5')
    base = run(k, {}, supply, coarse); lat0 = base[3]
    E_ref, _, hu = price(k, supply, {}, base, pg=False, mem_dvfs=False); E_pg, _, _ = price(k, supply, {}, base, pg=True, mem_dvfs=False)
    def ok(lat): return all(b <= a*(1+budget)+1 for a, b in zip(lat0, lat))
    def plan_E(asg):
        sc = run(k, asg, supply, coarse, cf, bf)
        return price(k, supply, asg, sc, pg=pg)[0] if ok(sc[3]) else None
    assign = {}
    for i, o in enumerate(OPS):
        if o['unit'] != 'MATRIX': continue
        c = o[cf]/k; m = o[bf]/BW_R*1e12; best = 0
        for pi in range(1, len(points)):
            if c*J.MAC_STRETCH[pi]+(.7-points[pi]['v'])*J.IVR_PS_PER_V*2 <= m and r[pi] < r[best]: best = pi
        if best: assign[i] = best
    E = plan_E(assign)
    if E is None: assign = {}; E = plan_E({})
    G = collections.defaultdict(list)
    for i, o in enumerate(OPS):
        if not o['unit']: continue
        g = group_of(o)+((min(2, int(o['p4']*3)),) if aware and o['unit'] == 'MATRIX' else ())
        G[g].append(i)
    level = collections.Counter(); changed = True
    while changed:
        changed = False
        for g, idx in G.items():
            while level[g]+1 < len(points):
                trial = dict(assign)
                for i in idx: trial[i] = max(trial.get(i, 0), level[g]+1)
                Et = plan_E(trial)
                if Et is None or Et >= E-1e-9: break
                E, assign = Et, trial; level[g] += 1; changed = True
    sc = run(k, assign, supply, coarse); Et, comp, _ = price(k, supply, assign, sc, pg=pg)
    worst = max(b/a-1 for a, b in zip(lat0, sc[3]))
    return dict(k=k, budget=budget, coarse=coarse, aware=aware, pg=pg, ref=E_ref, pg_only=E_pg, final=Et,
                saving_vs_ref=100*(1-Et/E_ref), saving_vs_pg=100*(1-Et/E_pg) if pg else None, worst=100*worst,
                met=worst <= budget+1e-9, makespan_s=sc[0]*1e-12, nominal_makespan_s=base[0]*1e-12, hbm_util=hu,
                mem_only=None)


if __name__ == '__main__':
    out = Path(ARGV[1]); out.mkdir(exist_ok=True)
    print('ops', len(OPS), 'MACs %.3e' % TOTAL_MACS, 'HBM GB %.1f' % (TOTAL_B/1e9))
    for k in (1, 32, 64, 128, 256):
        sc = run(k, {}, 'hybrid_scvr_dldo', False); E, comp, hu = price(k, 'hybrid_scvr_dldo', {}, sc, pg=False, mem_dvfs=False)
        print(f'k={k:3d} nominal T {sc[0]*1e-12:8.3f}s HBM util {hu:5.1%} ref energy {E:8.2f} J  TTFT {[round(x*1e-12,3) for x in sc[3][0::32]]}')
    jobs = [(k, 'hybrid_scvr_dldo', b, c, a, pg) for k in (32, 64, 128, 256) for b in (0., .05) for c in (True, False)
            for a in (False, True) for pg in (True, False)]
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex: rows = list(ex.map(arm, jobs))
    (out/'result.json').write_text(json.dumps(dict(status='QWEN14B_FIRST_ORDER_ABLATION_NOT_REPLAYED', rows=rows), indent=1)+'\n')
    for r in sorted(rows, key=lambda r: (r['k'], r['budget'], not r['pg'], not r['coarse'], r['aware'])):
        name = ('coarse' if r['coarse'] else 'fine')+'+'+('aware' if r['aware'] else 'agnostic')
        print(f"k={r['k']:3d} b={r['budget']:.0%} pg={int(r['pg'])} {name:16s} vs ref {r['saving_vs_ref']:5.1f}%  vs PG {r['saving_vs_pg'] if r['pg'] else float('nan'):5.1f}%  worst +{r['worst']:.2f}% met={r['met']} T {r['makespan_s']:.3f}s HBMu {r['hbm_util']:.0%}")
