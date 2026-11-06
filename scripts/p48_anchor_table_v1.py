"""Nominal-point (1111 / MEM 1200 ps) P4..P8 tile anchors at K=5120 and K=6912.

Merges the P4/P5 context anchors (qwen14b_phased_request_timing_calibration_v1/tile_anchor_model.json)
with the training splits of the P6..P8 phased-SRAM RTL calibrations (qwen14b_p6p8_matrix_calibration_v1
and _v2; holdout splits are not used). Rule as the P4/P5 anchors: max operation_duration_ps over
observations with the same (K, phase, precision, tile index, next precision). Lookup falls back to
the same (precision, next precision) at another middle tile index (2, 3, 1), then to the nearest
next precision at the same tile index.
"""
import collections, glob, json, statistics
from pathlib import Path
O = Path('/shared/s3/zqy/DSE/npu_mac/outputs/flexposit_enpu_paper/asap7_calibration')
TAB = dict(json.loads((O/'qwen14b_phased_request_timing_calibration_v1/tile_anchor_model.json').read_text())['anchors'])
_obs = collections.defaultdict(list)
for d in ('qwen14b_p6p8_matrix_calibration_v1', 'qwen14b_p6p8_matrix_calibration_v2'):
    for f in glob.glob(str(O/d/'*training*'/'*_1111_mem1200'/'result.json')):
        r = json.loads(Path(f).read_text())
        for o in r['observations']:
            _obs[f"{r['k']}|{r['phase']}|{o['precision']}|1111|1200|{o['tile']}|{o['next_precision']}"].append(o['operation_duration_ps'])
for k, v in _obs.items(): TAB[k] = max(v+([TAB[k]] if k in TAB else []))
_by_ctx = collections.defaultdict(list)
for k, v in TAB.items(): _by_ctx[k.rsplit('|', 1)[0]].append(v)


def lookup(k, phase, p, t, nxt):
    order = [t]+[x for x in (2, 3, 1) if x != t] if t != 4 else [4]
    for tt in order:
        key = f'{k}|{phase}|{p}|1111|1200|{tt}|{nxt}'
        if key in TAB: return TAB[key]
    if nxt is None: return None
    for tt in order:
        cands = [(abs(int(kk.rsplit('|', 1)[1])-nxt), v) for kk, v in TAB.items()
                 if kk.startswith(f'{k}|{phase}|{p}|1111|1200|{tt}|') and kk.rsplit('|', 1)[1] != 'None']
        if cands: return min(cands)[1]
    return None
