"""Continuous batching with per-request output counts: serve_batch_v1 with `outputs` a list (Azure traces)."""
import sys
from pathlib import Path
sys.argv = ['x', 'o']; sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_first_order_v1 as q
import serve_batch_v1 as SB


def simulate_steps(pf, outs, dur):
    n = len(pf); nxt = [1]*n; steps = []; t = min(pf)
    while any(x < o for x, o in zip(nxt, outs)):
        mem = [(r, nxt[r]) for r in range(n) if nxt[r] < outs[r] and pf[r] <= t]
        if not mem: t = min(pf[r] for r in range(n) if nxt[r] < outs[r]); continue
        steps.append(mem); t += dur(len(mem))
        for r, _ in mem: nxt[r] += 1
    return steps


def measure(lat, arrivals, outs, steps):
    emit = []; i = 0
    for r, o in enumerate(outs):
        t = arrivals[r]; row = []
        for j in range(o): t += lat[i+j]; row.append(t)
        emit.append(row); i += o
    pf = [e[0] for e in emit]; samples = {}; prev_end = None
    for mem in steps:
        e = emit[mem[0][0]][mem[0][1]]; st = max([prev_end or 0.]+[emit[r][t-1] for r, t in mem])
        samples.setdefault(len(mem), []).append(e-st); prev_end = e
    pts = sorted((m, sum(v)/len(v)) for m, v in samples.items())
    def dur(m):
        if len(pts) == 1: return pts[0][1]
        lo = max([p for p in pts if p[0] <= m] or [pts[0]]); hi = min([p for p in pts if p[0] >= m] or [pts[-1]])
        return lo[1] if lo[0] == hi[0] else lo[1]+(hi[1]-lo[1])*(m-lo[0])/(hi[0]-lo[0])
    return pf, dur


def install(bits, prompts, outputs, arrivals, supply='hybrid_scvr_dldo', k=256, iters=8):
    q.ARRIVAL_END.clear(); q.ARRIVAL_END.update({f'arr{i}': float(a) for i, a in enumerate(arrivals)})
    q.OUT_LIST = list(outputs); hist = []
    pf = [a+2.e12 for a in arrivals]; steps = simulate_steps(pf, outputs, lambda m: 40e9+.5e9*m)
    for it in range(iters):
        ops, deps = SB._graph(bits, prompts, max(outputs), arrivals, steps); SB._load(ops, deps, prompts, max(outputs), bits)
        lat = q.run(k, {}, supply, False)[3]
        pf, dur = measure(lat, arrivals, outputs, steps); new = simulate_steps(pf, outputs, dur); hist.append(len(steps))
        if new == steps: break
        steps = new
    else:
        ops, deps = SB._graph(bits, prompts, max(outputs), arrivals, steps); SB._load(ops, deps, prompts, max(outputs), bits)
    return dict(ops=len(q.OPS), steps=len(steps), mean_batch=sum(map(len, steps))/len(steps), iterations=len(hist))


def per_req_fn(outs):
    outs = list(outs)
    def f(lat):
        out = []; i = 0
        for n in outs: out.append((lat[i], sum(lat[i+1:i+n])/(n-1))); i += n
        return out
    return f
