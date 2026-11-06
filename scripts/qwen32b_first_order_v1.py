"""Qwen2.5-32B first-order request model: the Qwen2.5-14B model (qwen_first_order_v3) at 32B dimensions.

Same hidden size (5120), heads (40 query / 8 KV of 128) and vocabulary as 14B, so every matrix and vector
anchor transfers unchanged; 64 layers instead of 48 and FFN 27648 instead of 13824: down_proj runs as four
K=6912 pieces (the calibrated K of the 14B halves) with three FP16 partial-sum adds, and SiLU-mul scales with
the FFN width. Tile time and bytes as in v3 (P6..P8 linear in bits from the P4/P5 anchors).
"""
import collections, json, math, sys
from pathlib import Path
sys.argv = ['x', 'o']; sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_first_order_v1 as q
import qwen_first_order_v3 as v3
from qwen_first_order_v1 import H, HEADS, KV, HD, VOCAB, V, ATT_PS, HEAD_TILE
from qwen_first_order_v3 import tile, make_matrix

I, LAYERS = 27648, 64


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
        act = add(f'{pre}/silu_mul', eng, 'vector', 'VU', V('silu_mul')*m*I/13824, after=gu)
        halves = [mm(f'model.layers.{L_}.mlp.down_proj', 6912, [act], f'down_k{h}') for h in range(I//6912)]
        ps = add(f'{pre}/ksplit_add', eng, 'vector', 'VU', V('residual_add_n5120')*m*(I//6912-1), after=halves)
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
    """Replace the shared first-order workload with Qwen2.5-32B (same globals as v3.install)."""
    saved = v3.build; v3.build = build
    try: return v3.install(bits, **kw)
    finally: v3.build = saved


load_bits = v3.load_bits
