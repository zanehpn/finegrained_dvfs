"""Continuous-batching graphs for Qwen2.5-32B and Llama-3.1-8B: serve_batch_v1._graph with each model's own layers, FFN split and head, as in their full-mode builds."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_first_order_v1 as q, qwen_first_order_v3 as v3
from qwen_first_order_v1 import V, ATT_PS, HEAD_TILE
import llama_first_order_v2 as LL


def graph_for(model):
    if model == 'q32':
        import qwen32b_first_order_v1 as M
        c = dict(mk=M.make_matrix, H=q.H, LAYERS=M.LAYERS, HEADS=q.HEADS, KV=q.KV, HD=q.HD, VOCAB=q.VOCAB, bias=True, down=6912, nd=M.I//6912, rms=V('rmsnorm'), silu=V('silu_mul')*M.I/13824, res=V('residual_add_n5120'), ks=M.I//6912-1,
                 head=HEAD_TILE*(q.VOCAB//32), amax=4*V('argmax_chunk_n38016')+V('argmax_combine_n4'))
    else:
        S = LL.SCALE
        c = dict(mk=lambda b: LL.make_matrix(b), H=LL.H, LAYERS=LL.LAYERS, HEADS=LL.HEADS, KV=LL.KV, HD=LL.HD, VOCAB=LL.VOCAB, bias=False, down=LL.KDOWN, nd=2, rms=V('rmsnorm')*S['rmsnorm'], silu=V('silu_mul')*S['silu_mul'], res=V('residual_add_n5120')*S['residual'], ks=1,
                 head=HEAD_TILE*LL.H/5120*(LL.VOCAB//32), amax=4*V('argmax_chunk_n38016')*S['argmax']+V('argmax_combine_n4'))
    H, LAYERS, HEADS, KV, HD, VOCAB = c['H'], c['LAYERS'], c['HEADS'], c['KV'], c['HD'], c['VOCAB']; rms, silu, res, amax = c['rms'], c['silu'], c['res'], c['amax']

    def graph(bits, prompts, outputs, arrivals, steps):
        matrix = c['mk'](bits); ops, deps = [], {}
        def add(oid, engine, family, unit, C, Bt=0, prec=None, macs=0, sram=0, after=(), mac_e=1., pmax=None):
            ops.append(dict(id=oid, engine=engine, family=family, unit=unit, C=C, C_true=C, B=Bt, B_true=Bt, prec=prec or {}, macs=macs, sram_bytes=sram, residual=0., mac_e=mac_e, pmax=pmax)); deps[oid] = list(after); return oid
        def layer(l, pre, eng, ph, m, ctxs, rows, x):
            def mm(name, k, after, tag):
                C, Bt, prec, macs, sr, me, pm = matrix(name, k, ph, m)
                return add(f'{pre}/{tag}', eng, 'matrix', 'MATRIX', C, Bt, prec, macs, sram=sr, after=after, mac_e=me, pmax=pm)
            n1 = add(f'{pre}/input_norm', eng, 'vector', 'VU', rms*m, after=[x]); qkv = []
            for p in 'qkv':
                o = mm(f'model.layers.{l}.self_attn.{p}_proj', H, [n1], f'{p}_proj')
                qkv.append(add(f'{pre}/{p}_bias', eng, 'vector', 'VU', V('bias_add_n5120' if p == 'q' else 'bias_add_n1024')*m, after=[o]) if c['bias'] else o)
            rq = add(f'{pre}/rope_q', eng, 'vector', 'VU', V('rotary_n128')*m*HEADS, after=[qkv[0]])
            rk = add(f'{pre}/rope_k', eng, 'vector', 'VU', V('rotary_n128')*m*KV, after=[qkv[1]])
            kv = add(f'{pre}/kv_append', eng, 'state', None, 0., Bt=m*2*KV*HD*2, after=[rk, qkv[2]])
            cv = add(f'{pre}/fp32_cast', eng, 'vector', 'VU', V('FP16->FP32_n128')*sum(rows*HEADS+x_*KV for x_ in ctxs), Bt=sum(x_*2*KV*HD*2 for x_ in ctxs), after=[rq, kv])
            apn = 'P' if rows > 1 else 'D'
            qk = add(f'{pre}/attn_qk', eng, 'attention', 'ATT', sum(ATT_PS[(apn, 'QK')]*HEADS*rows*math.ceil(x_/32)*HD for x_ in ctxs), after=[cv])
            sm = add(f'{pre}/softmax', eng, 'vector', 'VU', sum(V('scaled_masked_softmax')*rows*HEADS*x_/270 for x_ in ctxs), after=[qk])
            av = add(f'{pre}/attn_av', eng, 'attention', 'ATT', sum(ATT_PS[(apn, 'AV')]*HEADS*rows*(HD//32)*x_ for x_ in ctxs), after=[sm])
            o = mm(f'model.layers.{l}.self_attn.o_proj', H, [av], 'o_proj')
            x1 = add(f'{pre}/residual1', eng, 'vector', 'VU', res*m, after=[o, x])
            n2 = add(f'{pre}/post_norm', eng, 'vector', 'VU', rms*m, after=[x1])
            gu = [mm(f'model.layers.{l}.mlp.{p}_proj', H, [n2], f'{p}_proj') for p in ('gate', 'up')]
            act = add(f'{pre}/silu_mul', eng, 'vector', 'VU', silu*m, after=gu)
            halves = [mm(f'model.layers.{l}.mlp.down_proj', c['down'], [act], f'down_k{h}') for h in range(c['nd'])]
            ps = add(f'{pre}/ksplit_add', eng, 'vector', 'VU', res*m*c['ks'], after=halves)
            return add(f'{pre}/residual2', eng, 'vector', 'VU', res*m, after=[ps, x1])
        for rid, P0 in enumerate(prompts):
            tag = f'r{rid}/t0'; x = add(f'{tag}/embedding', 'P', 'state', None, 0., Bt=P0*H*2, after=[f'arr{rid}'])
            for l in range(LAYERS): x = layer(l, f'{tag}/L{l}', 'P', 'P', P0, [P0], P0, x)
            fn = add(f'{tag}/final_norm', 'P', 'vector', 'VU', rms, after=[x])
            lm = add(f'{tag}/lm_head', 'P', 'fp16', 'ATT', c['head'], Bt=VOCAB*H*2, after=[fn])
            add(f'{tag}/emit', 'P', 'vector', 'VU', amax, after=[lm])
        prev = None
        for s, members in enumerate(steps):
            m = len(members); tag = f'g{s}'
            x = add(f'{tag}/embedding', 'D', 'state', None, 0., Bt=m*H*2, after=([prev] if prev else [])+[f'r{r}/t{t-1}/emit' for r, t in members])
            for l in range(LAYERS): x = layer(l, f'{tag}/L{l}', 'D', 'D', m, [prompts[r]+t for r, t in members], 1, x)
            fn = add(f'{tag}/final_norm', 'D', 'vector', 'VU', rms*m, after=[x])
            lm = add(f'{tag}/lm_head', 'D', 'fp16', 'ATT', c['head']*m, Bt=VOCAB*H*2, after=[fn])
            for r, t in members: add(f'r{r}/t{t}/emit', 'D', 'vector', 'VU', amax, after=[lm])
            prev = lm
        for o in ops:
            if o['id'][0] == 'g': o['members'] = sorted({r for r, _ in steps[int(o['id'][1:].split('/')[0])]})
        return ops, deps
    return graph
