"""Llama-3.1-8B first-order model with BitMoD datatypes (3/4/6/8 bits) on the same array (generality study).

BitMoD's PE serializes a weight into ceil(b/2) terms (2, 2, 3, 4 for 3/4/6/8 bits). Our tile takes
14+40*terms cycles at two bits per term-pass, i.e. exactly the RTL P4/P6/P8 tile (94/134/174 cycles),
so a b-bit tile reuses the P4..P8 anchors of llama_first_order_v2 at the timing precision TP[b]
(3 and 4 bits -> P4, 6 -> P6, 8 -> P8); its MAC energy is that of the same timing precision.
Weight bytes follow the true bits plus BitMoD's per-group (128) scale and selector.
Costing for the planner (prec counters) is in timing precision: worst case = 8 bits (P8 timing),
average = weight-averaged timing precision.
"""
import collections, math, sys, types
from pathlib import Path
sys.argv = ['x', 'o']; sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_first_order_v1 as q
import qwen_first_order_v3 as v3
import llama_first_order_v2 as LL

TP = {3: 4, 4: 4, 6: 6, 8: 8}
OVERHEAD = {3: 10/128, 4: 10/128, 6: 8/128, 8: 8/128}


def make_matrix(bits_by_module):
    cache = {}
    def matrix(name, k, phase, m):
        key = (name, k, phase, m)
        if key in cache: return cache[key]
        bits = bits_by_module[name]; tiles = [bits[i] for i in range(0, len(bits), 32)]; tt = [TP[b] for b in tiles]
        anchor_phase = 'P' if m > 1 else 'D'; mp = m if m == 1 else math.ceil(m/16)*16
        per = sum(LL.anchor(k, anchor_phase, p, tt[i+1] if i+1 < len(tt) else None, i, len(tt)) for i, p in enumerate(tt))
        c = q.summarize(q.compile_channel_matrix(mp, len(bits), k, [4]*len(bits), 'activation_resident', a_capacity=524288, w_capacity=524288))[0]
        blocks = c['load_activation_events']; slot = lambda b: (math.ceil(k*32*(b+OVERHEAD[b])/8)+63)//64*64+64
        p4 = (k*32*4//8+63)//64*64+64  # the compiled byte count assumes plain 4-bit slots
        extra = blocks*sum(slot(b)-p4 for b in tiles)
        hbm = c['hbm_total_bytes']+extra; sram = sum(vv for kk, vv in c.items() if kk.startswith('sram_') and kk.endswith('_bytes'))+extra
        prec = collections.Counter()
        for p in tt: prec[p] += 32*k*mp
        mac_e = sum(v3.E_PE(p) for p in tt)/len(tt)/v3.E_REF
        cache[key] = (per*(mp//16 if mp > 1 else 1), hbm, dict(prec), c['array_mac_operations'], sram, mac_e, max(tt))
        return cache[key]
    return matrix


def install(bits, **kw):
    """bits: true BitMoD bits per output channel; prec counters and worst case are in timing precision."""
    saved_m, saved_b = LL.make_matrix, v3.build
    LL.make_matrix = make_matrix; v3.build = LL.build
    try: return v3.install(_timing_view(bits), **kw)
    finally: LL.make_matrix, v3.build = saved_m, saved_b


class _timing_view(dict):
    """Passes true bits to the matrix builder; v3.install's pmax_plan (worst case) sees the max timing precision."""
    def values(self): return [[TP[b] for b in v] for v in super().values()]


def mean_cost_module():
    """Average costing: every matrix tile at the plan's weight-averaged timing precision."""
    def inst(bits, **kw):
        n = install(bits, **kw)
        tot = sum(len(b) for b in bits.values()); pbar = sum(TP[x] for b in bits.values() for x in b)/tot
        for o in q.OPS:
            if o['prec']:
                c = sum(o['prec'].values()); mean = sum(p*x for p, x in o['prec'].items())/c
                o['C_p5'] = o['C']*(94+20*(pbar-4))/(94+20*(mean-4)); o['B_p5'] = o['B']
        return n
    return types.SimpleNamespace(install=inst, load_bits=load_bits)


load_bits = v3.load_bits
