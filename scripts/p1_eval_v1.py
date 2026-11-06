"""P1 evaluations (first-order model, same planner/repair/pricing as insight_eval_v1).
  lowv     : delay/energy of the fitted 0.45-0.55 V points scaled, or those points removed;
  variance : high-variance precision allocations under worst/average/per-tile costing;
  azure    : Azure LLM trace windows (Splitwise, ISCA'24) dilated to a target prefill load.
usage: P1_DEST=<dir> P1_WORKERS=<n> P1_SWEEPS='lowv variance azure' p1_eval_v1.py;   finished jobs are skipped, so a killed run resumes.
"""
import csv, datetime as dt, json, os, sys, types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
DEST = Path(os.environ['P1_DEST']); WORKERS = int(os.environ['P1_WORKERS']); SWEEPS = os.environ['P1_SWEEPS'].split()  # env: spawn children rerun this module
sys.argv = ['x', 'o']; sys.path.insert(0, str(Path(__file__).resolve().parent))
import insight_eval_v1 as IE
import qwen_first_order_v1 as q, qwen_first_order_v3 as v3
import serve_var_v1 as SV

FIT = (2, 3, 4)  # indices of 0.55, 0.50, 0.45 V: below the STA-qualified 0.6 V point
LOWV = {'nominal': {}, 'delay_p10': dict(d=1.1), 'delay_p20': dict(d=1.2), 'energy_p10': dict(e=.1), 'energy_p20': dict(e=.2),
        'both_p20': dict(d=1.2, e=.2), 'sta_only': dict(drop=FIT), 'floor_050': dict(drop=(4,))}  # floor_050: only STA-measured points (0.5 V and up), 0.55 V interpolated between them
Fq = '/shared/s3/zqy/DSE/npu_mac/outputs/flexposit_precision_quality'
VARIANCE = {'fisher_b5.5': IE.S6.F_QWEN, 'bimodal_window': f'{Fq}/qwen14b_bimodal_plan_v1/plan_bimodal_window_b5.5.json', 'bimodal_module': f'{Fq}/qwen14b_bimodal_plan_v1/plan_bimodal_module_b5.5.json',
            'kv8_rest4': f'{Fq}/qwen14b_fisher_p4p8_plan_v1/plan_kv8_rest4.json', 'b4.5': f'{Fq}/qwen14b_fisher_p4p8_plan_v1/plan_b4.5.json'}
TRACE = '/shared/s3/zqy/DSE/npu_mac/evidence/traces/AzureLLMInferenceTrace_%s.csv'
S_PER_TOKEN = 7.105/1920  # nominal prefill of the 1920-token replay
full = lambda p: dict(mode='full', prompts=[p]*3, outputs=128)


def window(kind, w, n=32, limit=2048):
    rows = list(csv.DictReader(open(TRACE % kind))); start = len(rows)*(2*w+1)//6; out = []
    for x in rows[start:]:
        c, g = int(x['ContextTokens']), int(x['GeneratedTokens'])
        if c+g <= limit and g >= 2: out.append((dt.datetime.fromisoformat(x['TIMESTAMP'][:26]), c, g))
        if len(out) == n: break
    return out


def azure_kw(kind, w, rho):
    rows = window(kind, w); t0 = rows[0][0]; raw = [(r[0]-t0).total_seconds() for r in rows]
    prompts = [r[1] for r in rows]; outputs = [r[2] for r in rows]
    scale = (sum(prompts)*S_PER_TOKEN/rho)/max(raw[-1], 1e-9)
    return dict(mode='serve', prompts=prompts, outputs=outputs, arrivals=[a*scale*1e12 for a in raw]), scale


def job(spec):
    tag, sweep, base, extra = spec
    if sweep == 'lowv':
        k = LOWV[extra]
        if k.get('drop'):
            for i in k['drop']: q.J.MAC_STRETCH[i] = 1e6; q.J.VU_STRETCH[i] = 1e6; q.J.MEM_STRETCH[i] = 1e6
        elif 'd' in k:
            for i in FIT: q.J.MAC_STRETCH[i] *= k['d']; q.J.VU_STRETCH[i] *= k['d']
        if 'e' in k:
            src = q.L.SUPPLY[IE.SUP]; f = 1+k['e']
            q.L.SUPPLY[IE.SUP] = lambda v, src=src, f=f: src(v)/f if v < .6 else src(v)
    if sweep == 'azure':
        IE.LL = types.SimpleNamespace(install=lambda bits, **kw: SV.install(bits, kw['prompts'], kw['outputs'], kw['arrivals']), load_bits=v3.load_bits)
        IE.S6.per_req_fn = SV.per_req_fn
    r = IE.job(base); r['tag'] = tag; r['sweep'] = sweep; r['extra'] = extra; r.pop('workload', None)
    return r


def specs():
    out = []
    if 'lowv' in SWEEPS:
        for kn in LOWV:
            for c in ('aware', 'worst', 'mean'):
                for nm, kw, bt in (('t5', full(1920), .05), ('t10', full(1920), .10), ('batch16', dict(mode='decode', L=1920, R=16, B=16, outputs=32), .05)):
                    out.append((f'lowv_{kn}_{nm}_{c}', 'lowv', (nm, 'qwen', IE.S6.F_QWEN, kw, bt, .05, c), kn))
    if 'variance' in SWEEPS:
        for pn, pf in VARIANCE.items():
            for bt in (.05, .10):
                for c in ('worst', 'mean', 'aware'):
                    out.append((f'var_{pn}_t{int(bt*100)}_{c}', 'variance', (f'{pn}_t{int(bt*100)}', 'qwen', pf, full(1920), bt, .05, c), pn))
    if 'azure' in SWEEPS:
        for kind in ('conv', 'code'):
            for w in range(3):
                for lname, rho in (('low', .25), ('mid', .5), ('high', .8)):
                    for c in ('worst', 'mean', 'aware'):
                        kw, scale = azure_kw(kind, w, rho)
                        out.append((f'az_{kind}_w{w}_{lname}_{c}', 'azure', (f'{kind}_w{w}_{lname}', 'llama', IE.S6.F_QWEN, kw, .05, .05, c), dict(scale=scale, rho=rho)))
    return out


def main():
    DEST.mkdir(parents=True, exist_ok=True); jobs = [j for j in specs() if not (DEST/f'{j[0]}.json').exists()]
    print(len(jobs), 'jobs', flush=True)
    with ProcessPoolExecutor(min(WORKERS, max(1, len(jobs))), max_tasks_per_child=1, initializer=IE.S6.limit_memory) as ex:
        futs = {ex.submit(job, j): j for j in jobs}
        for f in as_completed(futs):
            try: r = f.result()
            except Exception as e: print('FAILED', futs[f][0], repr(e)[:300], flush=True); continue
            (DEST/f"{r['tag']}.json").write_text(json.dumps(r, indent=1)+'\n')
            print(f"{r['tag']:40s} {r['saving_vs_pg_cg']:6.2f}% vsNoDVFS {r['saving_vs_nodvfs']:6.2f}% met={r['met']} viol={r['pre_repair_violations']} repair={r['repair_rounds']}", flush=True)


if __name__ == '__main__': main()
