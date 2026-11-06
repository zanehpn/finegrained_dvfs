"""First-order HBM-controller (and HBM I/O) DVFS on top of qwen_first_order_v1 schedules.

The 16 HBM controllers are shared by both cores, so they form one domain. run_recording() is
qwen_first_order_v1.run with one addition: every HBM transfer interval booked on the shared
bandwidth server is recorded. hbm_dvfs() then cuts the makespan into WINDOW_PS windows and, per
window, picks the lowest operating point whose stretched busy time (MEM_STRETCH of the point times
the window's busy fraction) stays within MARGIN of the window: a free slowdown in the fluid sense,
so transfers keep their window and latency is unchanged to first order.
Energy, per window at point v:
  controller background (time-proportional, after the existing idle power-gating credit):
      power x (v/0.7) eta(0.7)/eta(v)   (leakage-like, linear in V; conservative: no f term)
  controller activity per byte:  x L.ratio(v)   (switching energy ~ V^2, regulator efficiency)
  I/O activity per byte ('ctrl+io', the eNPU 'full' configuration):  x L.ratio(v)
DRAM-die energy is not scaled (fixed array voltage). Not replayed: a first-order extension.
"""
import inspect, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_first_order_v1 as q

WINDOW_PS = 1e6
MARGIN = 0.9
HBM_IV = []
_src = inspect.getsource(q.run)
_src = _src.replace('def run(', 'def run_recording(', 1)
_src = _src.replace('hs = max(s, hbm_free) if m else s; hbm_free = hs+m if m else hbm_free',
                    'hs = max(s, hbm_free) if m else s; hbm_free = hs+m if m else hbm_free\n        if m: HBM_IV.append((hs, hs+m))')
assert 'HBM_IV.append' in _src
q.HBM_IV = HBM_IV  # defined in q's own namespace so it reads q's live globals (MEM_MODE, cap, OUT_TOKENS, ...)
exec(compile(_src, '<run_recording>', 'exec'), vars(q))


def run_recording(*a, **kw):
    HBM_IV.clear(); return q.run_recording(*a, **kw)


def hbm_dvfs(T_ps, comp, supply):
    """Return (delta_ctrl_j, delta_ctrl_io_j, point histogram by time) for one schedule."""
    eta = q.L.SUPPLY[supply]; n = int(T_ps//WINDOW_PS)+1; busy = [0.]*n
    for a, b in HBM_IV:
        w = int(a//WINDOW_PS)
        while a < b:
            e = min(b, (w+1)*WINDOW_PS); busy[w] += e-a; a = e; w += 1
    pts = q.points; st = q.J.MEM_STRETCH
    bg_pw = [(p['v']/.7)*eta(.7)/eta(p['v']) for p in pts]; act = [q.L.ratio(p['v'], eta) for p in pts]
    t_w = byte_w = 0.; hist = [0.]*len(pts); bsum = sum(busy) or 1.
    for b in busy:
        u = b/WINDOW_PS; pi = max([j for j in range(len(pts)) if st[j]*u <= MARGIN] or [0])
        t_w += bg_pw[pi]*WINDOW_PS; byte_w += act[pi]*b; hist[pi] += WINDOW_PS
    bg_remaining = comp['controller_background']+comp.get('hbm_controller_pg_credit', 0.)
    d_bg = bg_remaining*(1-t_w/(n*WINDOW_PS)); d_act = comp['controller_activity']*(1-byte_w/bsum)
    d_io = comp['io_activity']*(1-byte_w/bsum)
    return d_bg+d_act, d_bg+d_act+d_io, [h/(n*WINDOW_PS) for h in hist]
