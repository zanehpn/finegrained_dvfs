"""Llama-3.1-8B first-order request model v2: P4..P8 tile anchors from RTL (p48_anchor_table_v1), affine in K.

v1 extrapolated P6..P8 linearly from P4/P5 (K-transfer RTL holdouts: up to 15% error on P6..P8 tiles);
v2 uses the merged P4..P8 nominal anchors (holdouts K=4096/7168: max 1.47%, mean 0.31%).

v1 description:

Same scheduler, ledger and V/f points as the Qwen2.5-14B model (qwen_first_order_v1/v3); only
the workload changes:
  * Llama dims: 32 layers, hidden 4096, FFN 14336, 32 query / 8 KV heads of 128, vocab 128256,
    no qkv bias; down_proj as two K=7168 halves + FP16 partial-sum add (as Qwen's K=6912 split).
  * matrix tiles: per-tile time affine in K through the calibrated K=5120 and K=6912 anchors of
    the same (phase, precision, context, next-precision) key (tile time is 1.34x for 1.35x K);
    P6..P8 linear in bits from P4/P5 as in v3. Weight bytes from the real channel compiler.
  * vector rows: the calibrated Qwen per-row times scaled by the instruction count of the same
    kernel program at the Llama width (RMSNorm 9/tile+15, SiLU 93/tile, residual 4/tile,
    argmax chunk 8/tile+23 over 4 x 32064).
K-interpolated anchors are a first-order assumption until the K=4096/7168 RTL holdouts confirm them.
"""
import collections, json, math, sys
from pathlib import Path
sys.argv = ['x', 'o']; sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_first_order_v1 as q
import qwen_first_order_v3 as v3
import p48_anchor_table_v1 as T
from qwen_first_order_v1 import V, ATT_PS, HEAD_TILE

H, I, LAYERS, HEADS, KV, HD, VOCAB = 4096, 14336, 32, 32, 8, 128, 128256
KDOWN = I//2
Q_RMS = 9*(5120//32)+15
SCALE = dict(rmsnorm=(9*(H//32)+15)/Q_RMS, silu_mul=(I//32)/(13824//32), residual=(H//32)/(5120//32),
             argmax=(8*(32064//32)+23)/(8*(38016//32)+23))


def anchor(k, phase, p, nxt, idx, n):
    """RTL P4..P8 anchors at K=5120/6912 (same tile-position contexts as qwen_first_order_v1), affine in K."""
    t = 1 if idx == 0 else (4 if idx == n-1 else 2)
    a = T.lookup(5120, phase, p, t, nxt if t != 4 else None); b = T.lookup(6912, phase, p, t, nxt if t != 4 else None)
    return a+(b-a)*(k-5120)/(6912-5120)


def make_matrix(bits_by_module):
    cache = {}
    def matrix(name, k, phase, m):
        key = (name, k, phase, m)
        if key in cache: return cache[key]
        bits = bits_by_module[name]; tiles = [bits[i] for i in range(0, len(bits), 32)]
        anchor_phase = 'P' if m > 1 else 'D'; mp = m if m == 1 else math.ceil(m/16)*16
        per = sum(anchor(k, anchor_phase, p, tiles[i+1] if i+1 < len(tiles) else None, i, len(tiles)) for i, p in enumerate(tiles))
        c = q.summarize(q.compile_channel_matrix(mp, len(bits), k, [4]*len(bits), 'activation_resident', a_capacity=524288, w_capacity=524288))[0]
        blocks = c['load_activation_events']; slot = lambda p: (k*32*p//8+63)//64*64+64
        extra = blocks*sum(slot(p)-slot(4) for p in tiles)
        hbm = c['hbm_total_bytes']+extra; sram = sum(vv for kk, vv in c.items() if kk.startswith('sram_') and kk.endswith('_bytes'))+extra
        prec = collections.Counter()
        for p in tiles: prec[p] += 32*k*mp
        mac_e = sum(v3.E_PE(p) for p in tiles)/len(tiles)/v3.E_REF
        cache[key] = (per*(mp//16 if mp > 1 else 1), hbm, dict(prec), c['array_mac_operations'], sram, mac_e, max(tiles))
        return cache[key]
    return matrix


def build(bits, mode='full', L=None, R=3, B=1, prompts=None, outputs=32, arrivals=None):
    matrix = make_matrix(bits); ops, deps = [], {}
    rms, silu, res, amax = V('rmsnorm')*SCALE['rmsnorm'], V('silu_mul')*SCALE['silu_mul'], V('residual_add_n5120')*SCALE['residual'], \
        4*V('argmax_chunk_n38016')*SCALE['argmax']+V('argmax_combine_n4')
    def add(oid, engine, family, unit, C, Bt=0, prec=None, macs=0, sram=0, after=(), mac_e=1., pmax=None):
        ops.append(dict(id=oid, engine=engine, family=family, unit=unit, C=C, C_true=C, B=Bt, B_true=Bt, prec=prec or {},
                        macs=macs, sram_bytes=sram, residual=0., mac_e=mac_e, pmax=pmax)); deps[oid] = list(after); return oid
    def layer(l, pre, eng, ph, m, ctxs, x):
        def mm(name, k, after, tag):
            C, Bt, prec, macs, sr, me, pm = matrix(name, k, ph, m)
            return add(f'{pre}/{tag}', eng, 'matrix', 'MATRIX', C, Bt, prec, macs, sram=sr, after=after, mac_e=me, pmax=pm)
        n1 = add(f'{pre}/input_norm', eng, 'vector', 'VU', rms*m, after=[x])
        qp, kp, vp = (mm(f'model.layers.{l}.self_attn.{p}_proj', H, [n1], f'{p}_proj') for p in 'qkv')
        rq = add(f'{pre}/rope_q', eng, 'vector', 'VU', V('rotary_n128')*m*HEADS, after=[qp])
        rk = add(f'{pre}/rope_k', eng, 'vector', 'VU', V('rotary_n128')*m*KV, after=[kp])
        kv = add(f'{pre}/kv_append', eng, 'state', None, 0., Bt=m*2*KV*HD*2, after=[rk, vp])
        rows = m//len(ctxs)
        cv = add(f'{pre}/fp32_cast', eng, 'vector', 'VU', V('FP16->FP32_n128')*sum(rows*HEADS+c*KV for c in ctxs), Bt=sum(c*2*KV*HD*2 for c in ctxs), after=[rq, kv])
        apn = 'P' if rows > 1 else 'D'
        qk = add(f'{pre}/attn_qk', eng, 'attention', 'ATT', sum(ATT_PS[(apn, 'QK')]*HEADS*rows*math.ceil(c/32)*HD for c in ctxs), after=[cv])
        sm = add(f'{pre}/softmax', eng, 'vector', 'VU', sum(V('scaled_masked_softmax')*rows*HEADS*c/270 for c in ctxs), after=[qk])
        av = add(f'{pre}/attn_av', eng, 'attention', 'ATT', sum(ATT_PS[(apn, 'AV')]*HEADS*rows*(HD//32)*c for c in ctxs), after=[sm])
        o = mm(f'model.layers.{l}.self_attn.o_proj', H, [av], 'o_proj')
        x1 = add(f'{pre}/residual1', eng, 'vector', 'VU', res*m, after=[o, x])
        n2 = add(f'{pre}/post_norm', eng, 'vector', 'VU', rms*m, after=[x1])
        gu = [mm(f'model.layers.{l}.mlp.{p}_proj', H, [n2], f'{p}_proj') for p in ('gate', 'up')]
        act = add(f'{pre}/silu_mul', eng, 'vector', 'VU', silu*m, after=gu)
        halves = [mm(f'model.layers.{l}.mlp.down_proj', KDOWN, [act], f'down_k{h}') for h in range(2)]
        ps = add(f'{pre}/ksplit_add', eng, 'vector', 'VU', res*m, after=halves)
        return add(f'{pre}/residual2', eng, 'vector', 'VU', res*m, after=[ps, x1])
    def head(tag, ph, m, x):
        fn = add(f'{tag}/final_norm', ph, 'vector', 'VU', rms*m, after=[x])
        return add(f'{tag}/lm_head', ph, 'fp16', 'ATT', HEAD_TILE*H/5120*(VOCAB//32)*m, Bt=VOCAB*H*2, after=[fn])  # anchor is K=5120
    requests = []
    if mode == 'full':
        for rid, P0 in enumerate(prompts):
            prev = None; requests.append(dict(request=rid, prompt_ids=[0]*P0))
            for t in range(outputs):
                ph = 'P' if t == 0 else 'D'; m = P0 if t == 0 else 1; tag = f'r{rid}/t{t}'
                entry = [prev] if prev else ([f'arr{rid}'] if arrivals else [])
                x = add(f'{tag}/embedding', ph, 'state', None, 0., Bt=m*H*2, after=entry)
                for l in range(LAYERS): x = layer(l, f'{tag}/L{l}', ph, ph, m, [P0+t], x)
                prev = add(f'{tag}/emit', ph, 'vector', 'VU', amax, after=[head(tag, ph, 1, x)])
    else:  # decode-only, KV of length L resident, R requests batched in groups of B
        for rid in range(R): requests.append(dict(request=rid, prompt_ids=[0]*L))
        prev = {}
        for g0 in range(0, R, B):
            group = list(range(g0, min(R, g0+B))); m = len(group)
            for t in range(outputs):
                tag = f'g{g0}/t{t}'; after = [prev[r] for r in group if r in prev]
                x = add(f'{tag}/embedding', 'D', 'state', None, 0., Bt=m*H*2, after=after)
                for l in range(LAYERS): x = layer(l, f'{tag}/L{l}', 'D', 'D', m, [L+t]*m, x)
                lm = head(tag, 'D', m, x)
                for r in group: prev[r] = add(f'r{r}/t{t}/emit', 'D', 'vector', 'VU', amax, after=[lm])
    return ops, deps, requests


def install(bits, **kw):
    """Replace the shared first-order workload with Llama-3.1-8B (same globals as v3.install)."""
    saved = v3.build; v3.build = build
    try: return v3.install(bits, **kw)
    finally: v3.build = saved


load_bits = v3.load_bits
