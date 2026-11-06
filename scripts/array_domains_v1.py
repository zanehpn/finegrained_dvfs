"""Intra-array domain split (CTRL/DEC/OUT vs MAC) for fine-grained arms, from the RTL spatial-rail capture.

matched_spatial_rail_summary (one captured P4 tile, ASAP7 prelayout SPICE/Liberty power, matched clocks):
array core energy is MAC 91.2%, DEC 5.6%, CTRL 2.7%, OUT 0.2%, AON 0.2%; lowering CTRL/DEC/OUT to 0.6 V
with MAC at 0.7 V saved 2.33% of core energy at unchanged duration. At nominal clocks CTRL and DEC run at
1 ns and OUT at 2 ns against the 4 ns MAC clock, so they keep up with the MAC domain as long as their
stretch stays within 4x (CTRL, DEC) and 2x (OUT) of the MAC stretch.
A fine-grained (multi-domain) plan sets each of CTRL/DEC/OUT to the lowest operating point satisfying that
bound for the MAC point of every matrix op; a coarse (one rail per core) plan keeps them at the MAC point.
The matrix-core energy share of each domain is scaled by its own voltage ratio. Timing is unchanged by
construction. First-order: applied on top of qwen_first_order_v1.price.
"""
import json
from pathlib import Path
import qwen_first_order_v1 as q

O = Path('/shared/s3/zqy/DSE/npu_mac/outputs/flexposit_enpu_paper/asap7_calibration')
_r = json.loads((O/'matched_spatial_rail_summary/result.json').read_text())
_E = {d['domain']: d['uniform_j'] for d in _r['domain_energy_j'] if d['domain'] in ('AON', 'CTRL', 'DEC', 'MAC', 'OUT')}
SHARE = {d: e/sum(_E.values()) for d, e in _E.items()}
import os
HEADROOM = {'CTRL': 2.0, 'DEC': 2.0, 'OUT': 2.0} if os.environ.get('RAILS') == '4' else {'CTRL': 4.0, 'DEC': 4.0, 'OUT': 2.0}  # one shared rail follows its tightest member


def saving(assign, supply):
    """Energy (J) removed from the matrix-core term by per-domain CTRL/DEC/OUT points, for a fine plan."""
    eta = q.L.SUPPLY[supply]; r = [q.L.ratio(p['v'], eta) for p in q.points]; st = q.J.MAC_STRETCH
    out = 0.; per = q.S.mac_in/q.S.macs_total
    for i, o in enumerate(q.OPS):
        if o['unit'] != 'MATRIX': continue
        pi = assign.get(i, 0); e = per*o['macs']*o.get('mac_e', 1.)
        for d, h in HEADROOM.items():
            aux = max(j for j in range(len(q.points)) if st[j] <= h*st[pi]+1e-12)
            out += e*SHARE[d]*(r[pi]-r[aux])
    return out
