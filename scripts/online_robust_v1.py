"""tab:robust on the online controller: one knob, applied to planning, schedule and pricing alike, on the main 3x2K workload.
usage: ROB_KNOB=<name> online_robust_v1.py; FREQ_VMIN is set by the caller for the catalog-floor rows.
Knobs: nominal, switch_x10 (regulator transition), eff_m10 (10 points less efficiency below 0.7 V), dldo (no SC stage), lowv20 (20% more delay and energy below 0.6 V)."""
import os, sys
from pathlib import Path
KNOB = os.environ.get('ROB_KNOB', 'nominal')
W = '/shared/s3/zqy/dac2027_p1'
os.environ.update(ON_DEST=f'{W}/results/abl_c/rob/{KNOB}', ON_CACHE=f'{W}/results/cache_c/rob_{KNOB}')
_r = sys.argv[1:] or ['6']; sys.argv = ['x', _r[0]]; sys.path.insert(0, str(Path(__file__).resolve().parent))
import online_eval_v1 as E
q = E.q; J = q.J; L = q.L; SUP = E.SUP; ETA = L.SUPPLY[SUP]
if KNOB == 'switch_x10': J.IVR_PS_PER_V *= 10
if KNOB == 'eff_m10': L.SUPPLY[SUP] = lambda v: ETA(v) if v >= .7 else ETA(v)-.10
if KNOB == 'dldo': L.SUPPLY[SUP] = L.SUPPLY['dldo']
if KNOB == 'lowv20':
    L.SUPPLY[SUP] = lambda v: ETA(v) if v >= .6 else ETA(v)/1.2
    for l in (J.MAC_STRETCH, J.VU_STRETCH, J.ATT_STRETCH):  # in place: q.STRETCH holds these lists, and MEM_STRETCH aliases MAC_STRETCH
        l[:] = [s*1.2 if p['v'] < .6 else s for s, p in zip(l, q.points)]
    assert J.MEM_STRETCH is J.MAC_STRETCH and q.STRETCH['MATRIX'] is J.MAC_STRETCH


def main():
    E.DEST.mkdir(parents=True, exist_ok=True)
    full = dict(mode='full', prompts=[1920]*3, outputs=128)
    print(E.job(('main', 'qwen', E.F, full, ((.05, .05), (.10, .10), (.25, .05)), 0.)), flush=True)


if __name__ == '__main__': main()
