"""Regulator area estimate: peak power per rail from the first-order model at nominal (0.7 V), times the
IVR area per watt of eNPU's ASAP7 prototype (xue2026enpu Tab. hardware_area_power: DVFS-enabled IVR
6418.5 um^2 and fixed-voltage IVR 1800 um^2 for a core of 162.7 / 154.3 mW).

A rail is sized by the peak of its own domain; the fixed-voltage baseline by the peak of the whole core.
Rails: MAC (matrix ops), VU, ATT, and the retained/memory logic (MEM). CTRL/DEC/OUT draw from the
'other logic' and controller terms, sized here together with MEM (conservative: their peaks are added).
usage: regulator_area_v1.py [k]
"""
import json, sys
from pathlib import Path
sys.argv = ['x', 'o']; sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_first_order_v1 as F

DVFS_UM2_PER_W = 6418.5/0.162702
FIXED_UM2_PER_W = 1800/0.154295
ARRAY_UM2 = 0.273e6
SUP = 'hybrid_scvr_dldo'


def main(k=256):
    sc = F.run(k, {}, SUP, False); E, comp, _ = F.price(k, SUP, {}, sc, pg=False, mem_dvfs=False)
    T = sc[0]*1e-12
    peak = {}
    per = {'MATRIX': F.S.mac_in/F.S.macs_total, 'VU': F.J.vu_in/F.J.vu_C, 'ATT': F.J.att_in/F.J.att_C}
    other = F.J.other_work_in/F.S.macs_total
    for i, o in enumerate(F.OPS):
        u = o['unit']
        if not u: continue
        c = F.DUR[i][0]*1e-12
        if c <= 0: continue
        if u == 'MATRIX': e = per[u]*o['macs']*o.get('mac_e', 1.); e_other = other*o['macs']
        else: e = per[u]*o['C']; e_other = 0.
        key = (o['engine'], u); peak[key] = max(peak.get(key, 0.), e/c)
        ko = (o['engine'], 'LOGIC'); peak[ko] = max(peak.get(ko, 0.), e_other/c)
    # time-constant terms (W), split evenly over the two cores
    const = sum(comp[n] for n in ('dram_die_background', 'controller_background', 'frontend_controller', 'pd_link_fifo', 'array_leakage'))/T/2
    ret = F.J.RET_W_PER_ENGINE*k**(F.R.alpha_for(k))
    rows = {}
    for eng in ('P', 'D'):
        rails = {u: peak.get((eng, u), 0.) for u in ('MATRIX', 'VU', 'ATT', 'LOGIC')}
        rails['MEM'] = ret+const
        if __import__('os').environ.get('RAILS') == '4':  # VU and ATT never run together on a core, so their shared rail takes the larger peak
            rails['VEC'] = max(rails.pop('VU'), rails.pop('ATT'))
        core_peak = max(peak.get((eng, 'MATRIX'), 0.)+peak.get((eng, 'LOGIC'), 0.), peak.get((eng, 'VU'), 0.), peak.get((eng, 'ATT'), 0.))+rails['MEM']
        rows[eng] = dict(rails_W=rails, sum_rails_W=sum(rails.values()), core_peak_W=core_peak)
    dvfs = sum(r['sum_rails_W'] for r in rows.values())*DVFS_UM2_PER_W
    fixed = sum(r['core_peak_W'] for r in rows.values())*FIXED_UM2_PER_W
    arr = 2*k*ARRAY_UM2
    single = sum(r['core_peak_W'] for r in rows.values())*DVFS_UM2_PER_W  # one DVFS rail per core
    out = dict(k=k, cores=rows, dvfs_ivr_mm2=dvfs*1e-6, fixed_ivr_mm2=fixed*1e-6, single_rail_dvfs_ivr_mm2=single*1e-6,
               array_mm2=arr*1e-6, extra_pct_of_array=100*(dvfs-fixed)/arr, dvfs_pct_of_array=100*dvfs/arr,
               single_rail_extra_pct_of_array=100*(single-fixed)/arr, multi_over_single_pct_of_array=100*(dvfs-single)/arr,
               density=dict(dvfs_mm2_per_W=DVFS_UM2_PER_W*1e-6, fixed_mm2_per_W=FIXED_UM2_PER_W*1e-6),
               request_energy_J=E, mean_power_W=E/T)
    print(json.dumps(out, indent=1))
    d = F.O/'regulator_area_v1'; d.mkdir(exist_ok=True)
    (d/f'k{k}.json').write_text(json.dumps(out, indent=1)+'\n')


if __name__ == '__main__': main(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != 'o' else 256)
