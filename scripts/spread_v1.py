"""How far does the lead over average and worst-case costing grow with the spread of precision? Closed workloads, fair costings, Qwen2.5-14B."""
import os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
_r = sys.argv[1:] if len(sys.argv) > 1 and sys.argv[1].isdigit() else []; W = int(_r[0]) if _r else 12
R = '/shared/s3/zqy/dac2027_p1/results'; os.environ.update(ON_CACHE=f'{R}/cache_spread', ON_DEST=f'{R}/spread')
sys.argv = ['x', 'o']; sys.path.insert(0, str(Path(__file__).resolve().parent))
import online_eval_v1 as E
B = ((.02, .02), (.05, .05), (.10, .10), (.25, .05))


def main():
    E.DEST.mkdir(parents=True, exist_ok=True); V = E.P.VARIANCE; print(V, flush=True)
    js = [(f'{n}_p{p}', 'qwen', V[n], dict(mode='full', prompts=[p]*3, outputs=128), B, 0.) for n in V for p in (512, 1920, 3968)]
    with ProcessPoolExecutor(min(W, len(js)), max_tasks_per_child=1, initializer=E.limit_memory) as ex:
        fs = {ex.submit(E.job, j): j[0] for j in js}
        for x in as_completed(fs):
            try: print(x.result(), flush=True)
            except Exception as e: print('FAILED', fs[x], repr(e)[:300], flush=True)


if __name__ == '__main__': main()
