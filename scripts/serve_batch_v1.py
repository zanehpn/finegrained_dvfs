"""Continuous-batching service workload for the first-order model (Qwen2.5-14B).

Requests arrive at given times. The prefill core runs each prompt in arrival order (m = prompt rows).
The decode core runs global steps; a step batches every request whose prefill has finished before
the step starts and that still has tokens to emit (m = batch size; attention per request at its own
context; one LM head and argmax per member). Batch membership depends on timing, so it is solved on
the nominal (no-DVFS) schedule by fixed-point iteration and then held fixed for every DVFS plan,
like a batching decision made by the serving scheduler at nominal speed; DVFS plans are still checked
against every request's TTFT and mean TPOT. Op ids follow the replay naming (r<i>/t<j>/..., g<s>/...)
so planner_qwen_longctx_v1.arm and its objectives apply unchanged.
"""
import math, sys
from pathlib import Path
sys.argv = ['x', 'o']; sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_first_order_v1 as q
import qwen_first_order_v3 as v3
from qwen_first_order_v1 import H, LAYERS, HEADS, KV, HD, VOCAB, V, ATT_PS, HEAD_TILE


def _graph(bits, prompts, outputs, arrivals, steps):
    matrix = v3.make_matrix(bits); ops, deps = [], {}
    def add(oid, engine, family, unit, C, Bt=0, prec=None, macs=0, sram=0, after=(), mac_e=1., pmax=None):
        ops.append(dict(id=oid, engine=engine, family=family, unit=unit, C=C, C_true=C, B=Bt, B_true=Bt, prec=prec or {},
                        macs=macs, sram_bytes=sram, residual=0., mac_e=mac_e, pmax=pmax)); deps[oid] = list(after); return oid
    def layer(l, pre, eng, ph, m, ctxs, rows, x):
        def mm(name, k, after, tag):
            C, Bt, prec, macs, sr, me, pm = matrix(name, k, ph, m)
            return add(f'{pre}/{tag}', eng, 'matrix', 'MATRIX', C, Bt, prec, macs, sram=sr, after=after, mac_e=me, pmax=pm)
        n1 = add(f'{pre}/input_norm', eng, 'vector', 'VU', V('rmsnorm')*m, after=[x]); qkv = []
        for p in 'qkv':
            o = mm(f'model.layers.{l}.self_attn.{p}_proj', H, [n1], f'{p}_proj')
            qkv.append(add(f'{pre}/{p}_bias', eng, 'vector', 'VU', V('bias_add_n5120' if p == 'q' else 'bias_add_n1024')*m, after=[o]))
        rq = add(f'{pre}/rope_q', eng, 'vector', 'VU', V('rotary_n128')*m*HEADS, after=[qkv[0]])
        rk = add(f'{pre}/rope_k', eng, 'vector', 'VU', V('rotary_n128')*m*KV, after=[qkv[1]])
        kv = add(f'{pre}/kv_append', eng, 'state', None, 0., Bt=m*2*KV*HD*2, after=[rk, qkv[2]])
        cv = add(f'{pre}/fp32_cast', eng, 'vector', 'VU', V('FP16->FP32_n128')*sum(rows*HEADS+c*KV for c in ctxs), Bt=sum(c*2*KV*HD*2 for c in ctxs), after=[rq, kv])
        apn = 'P' if rows > 1 else 'D'
        qk = add(f'{pre}/attn_qk', eng, 'attention', 'ATT', sum(ATT_PS[(apn, 'QK')]*HEADS*rows*math.ceil(c/32)*HD for c in ctxs), after=[cv])
        sm = add(f'{pre}/softmax', eng, 'vector', 'VU', sum(V('scaled_masked_softmax')*rows*HEADS*c/270 for c in ctxs), after=[qk])
        av = add(f'{pre}/attn_av', eng, 'attention', 'ATT', sum(ATT_PS[(apn, 'AV')]*HEADS*rows*(HD//32)*c for c in ctxs), after=[sm])
        o = mm(f'model.layers.{l}.self_attn.o_proj', H, [av], 'o_proj')
        x1 = add(f'{pre}/residual1', eng, 'vector', 'VU', V('residual_add_n5120')*m, after=[o, x])
        n2 = add(f'{pre}/post_norm', eng, 'vector', 'VU', V('rmsnorm')*m, after=[x1])
        gu = [mm(f'model.layers.{l}.mlp.{p}_proj', H, [n2], f'{p}_proj') for p in ('gate', 'up')]
        act = add(f'{pre}/silu_mul', eng, 'vector', 'VU', V('silu_mul')*m, after=gu)
        halves = [mm(f'model.layers.{l}.mlp.down_proj', 6912, [act], f'down_k{h}') for h in range(2)]
        ps = add(f'{pre}/ksplit_add', eng, 'vector', 'VU', V('residual_add_n5120')*m, after=halves)
        return add(f'{pre}/residual2', eng, 'vector', 'VU', V('residual_add_n5120')*m, after=[ps, x1])
    amax = 4*V('argmax_chunk_n38016')+V('argmax_combine_n4')
    for rid, P0 in enumerate(prompts):  # prefill: token 0 of each request on the P core
        tag = f'r{rid}/t0'; x = add(f'{tag}/embedding', 'P', 'state', None, 0., Bt=P0*H*2, after=[f'arr{rid}'])
        for l in range(LAYERS): x = layer(l, f'{tag}/L{l}', 'P', 'P', P0, [P0], P0, x)
        fn = add(f'{tag}/final_norm', 'P', 'vector', 'VU', V('rmsnorm'), after=[x])
        lm = add(f'{tag}/lm_head', 'P', 'fp16', 'ATT', HEAD_TILE*(VOCAB//32), Bt=VOCAB*H*2, after=[fn])
        add(f'{tag}/emit', 'P', 'vector', 'VU', amax, after=[lm])
    prev = None
    for s, members in enumerate(steps):  # members: list of (request, token index >= 1)
        m = len(members); tag = f'g{s}'
        after = ([prev] if prev else [])+[f'r{r}/t{t-1}/emit' for r, t in members]
        x = add(f'{tag}/embedding', 'D', 'state', None, 0., Bt=m*H*2, after=after)
        for l in range(LAYERS): x = layer(l, f'{tag}/L{l}', 'D', 'D', m, [prompts[r]+t for r, t in members], 1, x)
        fn = add(f'{tag}/final_norm', 'D', 'vector', 'VU', V('rmsnorm')*m, after=[x])
        lm = add(f'{tag}/lm_head', 'D', 'fp16', 'ATT', HEAD_TILE*(VOCAB//32)*m, Bt=VOCAB*H*2, after=[fn])
        for r, t in members: add(f'r{r}/t{t}/emit', 'D', 'vector', 'VU', amax, after=[lm])
        prev = lm
    for o in ops:  # requests served by each decode step, for per-request TPOT budgets
        if o['id'][0] == 'g': o['members'] = sorted({r for r, _ in steps[int(o['id'][1:].split('/')[0])]})
    return ops, deps


def _simulate_steps(pf, outputs, dur):
    """Membership of each decode step for prefill completion times pf and step duration dur(m)."""
    n = len(pf); nxt = [1]*n; steps = []; t = min(pf)
    while any(x < outputs for x in nxt):
        mem = [(r, nxt[r]) for r in range(n) if nxt[r] < outputs and pf[r] <= t]
        if not mem: t = min(pf[r] for r in range(n) if nxt[r] < outputs); continue
        steps.append(mem); t += dur(len(mem))
        for r, _ in mem: nxt[r] += 1
    return steps


def _measure(lat, arrivals, outputs, steps):
    """Prefill end times and a per-batch-size step duration fit from a nominal run's emit latencies."""
    n = len(arrivals); emit = []
    for r in range(n):
        t = arrivals[r]; row = []
        for j in range(outputs): t += lat[r*outputs+j]; row.append(t)
        emit.append(row)
    pf = [emit[r][0] for r in range(n)]; samples = {}; prev_end = None
    for mem in steps:
        e = emit[mem[0][0]][mem[0][1]]; st = max([prev_end or 0.]+[emit[r][t-1] for r, t in mem])
        samples.setdefault(len(mem), []).append(e-st); prev_end = e
    pts = sorted((m, sum(v)/len(v)) for m, v in samples.items())
    def dur(m):
        if len(pts) == 1: return pts[0][1]
        lo = max([p for p in pts if p[0] <= m] or [pts[0]]); hi = min([p for p in pts if p[0] >= m] or [pts[-1]])
        if lo[0] == hi[0]: return lo[1]
        return lo[1]+(hi[1]-lo[1])*(m-lo[0])/(hi[0]-lo[0])
    return pf, dur


def install(bits, prompts, outputs, arrivals, supply='hybrid_scvr_dldo', k=256, iters=8):
    """Solve batch membership on the nominal schedule (fixed point), then leave that graph installed."""
    q.ARRIVAL_END.clear(); q.ARRIVAL_END.update({f'arr{i}': float(a) for i, a in enumerate(arrivals)})
    pf = [a+2.e12 for a in arrivals]; steps = _simulate_steps(pf, outputs, lambda m: 40e9+.5e9*m); hist = []
    for it in range(iters):
        ops, deps = _graph(bits, prompts, outputs, arrivals, steps); _load(ops, deps, prompts, outputs, bits)
        lat = q.run(k, {}, supply, False)[3]
        pf, dur = _measure(lat, arrivals, outputs, steps); new = _simulate_steps(pf, outputs, dur)
        hist.append(len(steps))
        if new == steps: break
        steps = new
    else:
        ops, deps = _graph(bits, prompts, outputs, arrivals, steps); _load(ops, deps, prompts, outputs, bits)
    return dict(ops=len(q.OPS), steps=len(steps), mean_batch=sum(map(len, steps))/len(steps), iterations=len(hist),
                converged=len(hist) < iters or hist[-1] == len(steps), step_counts=hist)


def _load(ops, deps, prompts, outputs, bits):
    pmax_plan = max(max(b) for b in bits.values())
    for o in ops:
        if o['prec']:
            tot = sum(o['prec'].values()); mean = sum(p*n for p, n in o['prec'].items())/tot
            o['p4'] = min(.999, (mean-4)/4) if pmax_plan > 5 else o['prec'].get(4, 0)/tot
            o['C_p5'] = o['C']*(94+20*(pmax_plan-4))/(94+20*(mean-4)) if pmax_plan > 5 else o['C']*(5/mean)
            o['B_p5'] = o['B']*pmax_plan/mean if v3.AGN_BYTES == 'worst' else o['B']
        else: o['p4'] = 0.; o['C_p5'] = o['C']; o['B_p5'] = o['B']
    q.OPS[:] = ops; q.DEPS.clear(); q.DEPS.update(deps)
    q.cap = dict(requests=[dict(request=r, prompt_ids=[0]*p) for r, p in enumerate(prompts)])
    q.OUT_TOKENS = outputs; q.TOTAL_MACS = sum(o['macs'] for o in ops); q.TOTAL_B = sum(o['B'] for o in ops)
