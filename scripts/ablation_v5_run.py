"""2x2 ablation v4: whole-core replication (98.6% of retained peripheral energy replicated),
agnostic planner uses P5 cycles AND P5 packed bytes, absolute SLO check."""
import contextlib, importlib.util, io, json, math, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('a5', HERE/'ablation_v5.py')
with contextlib.redirect_stdout(io.StringIO()):
    A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
# load-J split of retained peripheral logic (ideal-gated nominal ledger)
REPLICATED = 2.904484190180921+2.1105701303920394+0.336210656255518+1.3196703937091034
TOTAL = 2.904484190180921+2.5400677039053057+1.3222815149691356
FRAC = REPLICATED/TOTAL


def alpha_for(k): return math.log(FRAC*k+1-FRAC)/math.log(k)


def pipeline(args):
    k, b, aware, pg = args
    base = (k, alpha_for(k), 'hybrid_scvr_dldo', b)
    coarse = A.arm(base+(True, aware, pg))
    scratch = A.arm(base+(False, aware, pg))
    warm = A.arm(base+(False, aware, pg, {int(i): p for i, p in coarse['assignment'].items()}))
    ok = [r for r in (scratch, warm) if r['budget_met']]
    fine = min(ok, key=lambda r: r['final_j']); fine['fine_start'] = 'warm' if fine is warm else 'scratch'
    return [coarse, fine]


if __name__ == '__main__':
    jobs = [(k, b, aw, pg) for k in (64, 128) for b in (0., .05) for aw in (False, True) for pg in (True, False)]
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        rows = [r for pair in ex.map(pipeline, jobs) for r in pair]
    Path(sys.argv[1]).mkdir(exist_ok=True)
    (Path(sys.argv[1])/'result.json').write_text(json.dumps(dict(status='FIRST_ORDER_DVFS_ABLATION_V5_BW85_ANCHORED_TO_REPLAY',
        replicated_fraction=FRAC, rows=rows), indent=1)+'\n')
    print('replicated fraction', FRAC)
    for r in sorted(rows, key=lambda r: (r['k'], r['budget'], not r['pg'], not r['coarse'], r['aware'])):
        name = ('coarse' if r['coarse'] else 'fine')+'+'+('aware' if r['aware'] else 'agnostic')
        print(f"k={r['k']} b={r['budget']:.0%} pg={int(r['pg'])} {name:16s} E {r['final_j']:6.2f}J vs ref {r['saving_vs_reference_percent']:5.2f}% "
              f"vs PG-only {r['saving_vs_pg_only_percent'] if r['pg'] else float('nan'):5.2f}% lat +{r['worst_latency_increase_percent']:.2f}% met={r['budget_met']} {r.get('fine_start','')}", flush=True)
