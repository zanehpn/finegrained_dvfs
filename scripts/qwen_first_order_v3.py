"""Qwen2.5-14B first-order v3: configurable workload (P4..P8 plans, long context, batch).

On top of v1 (uniform clock-gating ledger, shared-HBM FCFS):
  * P6..P8 tiles: tile time linear in bits from the P4/P5 context anchors (bit-serial;
    ASAP7 PE scan 94/114/134/154/174 cycles for P4..P8 is exactly linear); weight bytes
    per tile = align(K*32*P/8)+64 per activation row block; MAC energy per op scaled by
    the ASAP7 PE scan energy/MAC (1.241 + 0.199*(P-4) pJ) relative to the Phi-2 ledger mix.
  * precision-agnostic planning assumes every matrix op runs at the plan's worst-case
    precision (P5 for the P4/P5 plan, P8 for P4..P8 plans).
  * mode 'decode': KV of length L already resident; R requests x 32 decode steps, batched
    in groups of B (rows m=B through the array, 16-row anchors when B>1).
"""
import collections, json, math, sys
from pathlib import Path
sys.argv = ['x', 'o']; sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_first_order_v1 as q
from qwen_first_order_v1 import H, I, LAYERS, HEADS, KV, HD, VOCAB, V, ATT_PS, HEAD_TILE

E_PE = lambda p: 1.2412+0.19874*(p-4)  # ASAP7 PE scan, pJ per useful MAC at 0.7 V / 1200 ps
E_REF = E_PE(4.5)                       # Phi-2 ledger mix (4.5-bit plan)
AGN_BYTES = 'worst'  # precision-agnostic planner: 'worst' scales weight bytes to the worst-case precision too; 'true' keeps real bytes
PLANS = {'p45': Path('/shared/s3/zqy/DSE/npu_mac/outputs/flexposit_precision_quality/qwen14b_artifact_channel_plan_v1/result.json')}


def tile(k, phase, p, nxt, idx, n):
    lo = q.tile_ps(k, phase, 4, min(nxt, 5) if nxt else nxt, idx, n); hi = q.tile_ps(k, phase, 5, min(nxt, 5) if nxt else nxt, idx, n)
    return lo+(p-4)*(hi-lo)


def make_matrix(bits_by_module):
    cache = {}
    def matrix(name, k, phase, m):
        key = (name, k, phase, m)
        if key in cache: return cache[key]
        bits = bits_by_module[name]; tiles = [bits[i] for i in range(0, len(bits), 32)]
        anchor_phase = 'P' if m > 1 else 'D'; mp = m if m == 1 else math.ceil(m/16)*16
        per = sum(tile(k, anchor_phase, p, tiles[i+1] if i+1 < len(tiles) else None, i, len(tiles)) for i, p in enumerate(tiles))
        b45 = [min(5, max(4, b)) for b in bits]
        c = q.summarize(q.compile_channel_matrix(mp, len(bits), k, [4]*len(bits), 'activation_resident', a_capacity=524288, w_capacity=524288))[0]
        blocks = c['load_activation_events']; slot = lambda p: (k*32*p//8+63)//64*64+64
        extra = blocks*sum(slot(p)-slot(4) for p in tiles)
        hbm = c['hbm_total_bytes']+extra; sram = sum(v for kk, v in c.items() if kk.startswith('sram_') and kk.endswith('_bytes'))+extra
        prec = collections.Counter()
        for p in tiles: prec[p] += 32*k*mp
        mac_e = sum(E_PE(p) for p in tiles)/len(tiles)/E_REF
        cache[key] = (per*(mp//16 if mp > 1 else 1), hbm, dict(prec), c['array_mac_operations'], sram, mac_e, max(tiles))
        return cache[key]
    return matrix


def build(bits, mode='full', L=None, R=3, B=1, prompts=None, outputs=32, arrivals=None):
    matrix = make_matrix(bits); ops, deps = [], {}
    def add(oid, engine, family, unit, C, Bt=0, prec=None, macs=0, sram=0, after=(), mac_e=1., pmax=None):
        ops.append(dict(id=oid, engine=engine, family=family, unit=unit, C=C, C_true=C, B=Bt, B_true=Bt, prec=prec or {},
                        macs=macs, sram_bytes=sram, residual=0., mac_e=mac_e, pmax=pmax)); deps[oid] = list(after); return oid
    def layer(pre, eng, ph, m, ctxs, x):
        # ctxs: list of per-request contexts in this step (len == number of request rows for attention)
        n1 = add(f'{pre}/input_norm', eng, 'vector', 'VU', V('rmsnorm')*m, after=[x]); qkv = []
        def mm(name, k, after, tag):
            C, Bt, prec, macs, sr, me, pm = matrix(name, k, ph, m)
            return add(f'{pre}/{tag}', eng, 'matrix', 'MATRIX', C, Bt, prec, macs, sram=sr, after=after, mac_e=me, pmax=pm)
        for proj, n in (('q', H), ('k', KV*HD), ('v', KV*HD)):
            o = mm(f'model.layers.{L_}.self_attn.{proj}_proj', H, [n1], f'{proj}_proj')
            qkv.append(add(f'{pre}/{proj}_bias', eng, 'vector', 'VU', V('bias_add_n5120' if proj == 'q' else 'bias_add_n1024')*m, after=[o]))
        rq = add(f'{pre}/rope_q', eng, 'vector', 'VU', V('rotary_n128')*m*HEADS, after=[qkv[0]])
        rk = add(f'{pre}/rope_k', eng, 'vector', 'VU', V('rotary_n128')*m*KV, after=[qkv[1]])
        kv = add(f'{pre}/kv_append', eng, 'state', None, 0., Bt=m*2*KV*HD*2, after=[rk, qkv[2]])
        rows = m//len(ctxs)
        cv = add(f'{pre}/fp32_cast', eng, 'vector', 'VU', V('FP16->FP32_n128')*sum(rows*HEADS+c*KV for c in ctxs), Bt=sum(c*2*KV*HD*2 for c in ctxs), after=[rq, kv])
        apn = 'P' if rows > 1 else 'D'
        qk = add(f'{pre}/attn_qk', eng, 'attention', 'ATT', sum(ATT_PS[(apn, 'QK')]*HEADS*rows*math.ceil(c/32)*HD for c in ctxs), after=[cv])
        sm = add(f'{pre}/softmax', eng, 'vector', 'VU', sum(V('scaled_masked_softmax')*rows*HEADS*c/270 for c in ctxs), after=[qk])
        av = add(f'{pre}/attn_av', eng, 'attention', 'ATT', sum(ATT_PS[(apn, 'AV')]*HEADS*rows*(HD//32)*c for c in ctxs), after=[sm])
        o = mm(f'model.layers.{L_}.self_attn.o_proj', H, [av], 'o_proj')
        x1 = add(f'{pre}/residual1', eng, 'vector', 'VU', V('residual_add_n5120')*m, after=[o, x])
        n2 = add(f'{pre}/post_norm', eng, 'vector', 'VU', V('rmsnorm')*m, after=[x1])
        gu = [mm(f'model.layers.{L_}.mlp.{p}_proj', H, [n2], f'{p}_proj') for p in ('gate', 'up')]
        act = add(f'{pre}/silu_mul', eng, 'vector', 'VU', V('silu_mul')*m, after=gu)
        halves = [mm(f'model.layers.{L_}.mlp.down_proj', 6912, [act], f'down_k{h}') for h in range(2)]
        ps = add(f'{pre}/ksplit_add', eng, 'vector', 'VU', V('residual_add_n5120')*m, after=halves)
        return add(f'{pre}/residual2', eng, 'vector', 'VU', V('residual_add_n5120')*m, after=[ps, x1])
    requests = []
    if mode == 'full':
        for rid, P0 in enumerate(prompts):
            prev = None; requests.append(dict(request=rid, prompt_ids=[0]*P0))
            for t in range(outputs):
                ph = 'P' if t == 0 else 'D'; m = P0 if t == 0 else 1; tag = f'r{rid}/t{t}'
                entry = [prev] if prev else ([f'arr{rid}'] if arrivals else [])
                x = add(f'{tag}/embedding', ph, 'state', None, 0., Bt=m*H*2, after=entry)
                for L_ in range(LAYERS):
                    globals()['L_'] = L_; x = layer(f'{tag}/L{L_}', ph, ph, m, [P0+t], x)
                fn = add(f'{tag}/final_norm', ph, 'vector', 'VU', V('rmsnorm'), after=[x])
                lm = add(f'{tag}/lm_head', ph, 'fp16', 'ATT', HEAD_TILE*(VOCAB//32), Bt=VOCAB*H*2, after=[fn])
                prev = add(f'{tag}/emit', ph, 'vector', 'VU', 4*V('argmax_chunk_n38016')+V('argmax_combine_n4'), after=[lm])
    else:  # decode-only, KV of length L resident, R requests batched in groups of B
        for rid in range(R): requests.append(dict(request=rid, prompt_ids=[0]*L))
        prev = {}
        for g0 in range(0, R, B):
            group = list(range(g0, min(R, g0+B))); m = len(group)
            for t in range(32):
                tag = f'g{g0}/t{t}'; after = [prev[r] for r in group if r in prev]
                x = add(f'{tag}/embedding', 'D', 'state', None, 0., Bt=m*H*2, after=after)
                for L_ in range(LAYERS):
                    globals()['L_'] = L_; x = layer(f'{tag}/L{L_}', 'D', 'D', m, [L+t]*m, x)
                fn = add(f'{tag}/final_norm', 'D', 'vector', 'VU', V('rmsnorm')*m, after=[x])
                lm = add(f'{tag}/lm_head', 'D', 'fp16', 'ATT', HEAD_TILE*(VOCAB//32)*m, Bt=VOCAB*H*2, after=[fn])
                for r in group:
                    prev[r] = add(f'r{r}/t{t}/emit', 'D', 'vector', 'VU', 4*V('argmax_chunk_n38016')+V('argmax_combine_n4'), after=[lm])
    return ops, deps, requests


def install(bits, **kw):
    """Replace v1's workload in place (run/price/arm read these module globals)."""
    ops, deps, requests = build(bits, **kw)
    pmax_plan = max(max(v) for v in bits.values())
    for o in ops:
        if o['prec']:
            tot = sum(o['prec'].values()); mean = sum(p*n for p, n in o['prec'].items())/tot
            o['p4'] = min(.999, (mean-4)/4) if pmax_plan > 5 else o['prec'].get(4, 0)/tot
            o['C_p5'] = o['C']*(94+20*(pmax_plan-4))/(94+20*(mean-4)) if pmax_plan > 5 else o['C']*(5/mean)  # worst-case precision assumption
            o['B_p5'] = o['B']*pmax_plan/mean if AGN_BYTES == 'worst' else o['B']  # 'true': DMA byte counts are known to the compiler
        else:
            o['p4'] = 0.; o['C_p5'] = o['C']; o['B_p5'] = o['B']
    q.OPS[:] = ops; q.DEPS.clear(); q.DEPS.update(deps); q.cap = dict(requests=requests)
    q.OUT_TOKENS = kw.get('outputs', 32); q.ARRIVAL_END.clear()
    if kw.get('arrivals'): q.ARRIVAL_END.update({f'arr{i}': float(a) for i, a in enumerate(kw['arrivals'])})
    q.TOTAL_MACS = sum(o['macs'] for o in ops); q.TOTAL_B = sum(o['B'] for o in ops)
    return len(ops)


def load_bits(name):
    path = PLANS.get(name) or Path(name)
    return {n: m['channel_precision'] for n, m in json.loads(path.read_text())['modules'].items()}
