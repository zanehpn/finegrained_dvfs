"""fig_units from the 45 precision runs of results/motivation_c: domain load and energy versus weight precision."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_figures_v2 import FIG, BLUE, ORANGE, AQUA, YELLOW, INK, MUTED, style
M = Path('/shared/s3/zqy/dac2027_p1/results/motivation_c'); P = ['P4', 'P5', 'mixed', 'P6', 'P8']; X = [4, 5, 5.5, 6, 8]; LAB = ['P4', 'P5', 'Fisher', 'P6', 'P8']


def load(m, p): return {x: json.loads((M/f'{m}_p{p}_{x}.json').read_text()) for x in P}


def fig_units():
    r = load('q14', 1920); fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.42), gridspec_kw=dict(width_ratios=[1.15, 1]))
    busy = lambda x, u: r[x]['nominal']['busy_s']['prefill'][u]
    for u, c, mk, ls, lab in (('HBM', ORANGE, 's', '--', 'HBM'), ('MAC', BLUE, 'o', '-', 'MAC'), ('MEM', AQUA, '^', ':', 'MEM'), ('VEC', YELLOW, 'D', '-.', 'VEC')):
        y = [busy(x, u)/busy('P4', u) for x in P]; a.plot(X, y, color=c, marker=mk, ls=ls, ms=3, lw=1, label=lab)
        a.annotate(lab, (8, y[-1]), textcoords='offset points', xytext=(3, -2), fontsize=5.5, color=INK)
    a.set_xlim(3.8, 9.1); a.set_xticks([4, 5, 6, 8]); a.set_xticklabels(['P4', 'P5', 'P6', 'P8'], fontsize=6); a.set_ylim(.9, 2.05); a.set_ylabel('Prefill busy time / P4', fontsize=6); a.set_title('(a) Load by domain', fontsize=7, color=INK); style(a)
    cm = lambda x: busy(x, 'MAC')/busy(x, 'HBM'); a.text(4.0, 1.86, f'$C/M$: {cm("P4"):.2f} at P4\n{cm("P8"):.2f} at P8', fontsize=5.5, color=INK)
    bot = [0.]*5; t = [sum(r[x]['dom_pgcg'].values()) for x in P]
    for ds, c, h, lab in ((('MAC',), BLUE, '', 'MAC'), (('MEM',), AQUA, '\\\\\\\\', 'MEM'), (('HBM',), ORANGE, '////', 'HBM'), (('VEC', 'LOGIC', 'OTHER'), YELLOW, '', 'Other')):
        y = [100*sum(r[x]['dom_pgcg'][d] for d in ds)/s for x, s in zip(P, t)]; b.bar(range(5), y, .74, bottom=bot, color=c, hatch=h, edgecolor='white', lw=.4, label=lab); bot = [p+q for p, q in zip(bot, y)]
    b.set_xticks(range(5)); b.set_xticklabels(LAB, fontsize=6); b.set_ylim(0, 100); b.set_ylabel('Energy under PG+CG (%)', fontsize=6); b.set_title('(b) Energy by domain', fontsize=7, color=INK); style(b)
    b.legend(frameon=False, fontsize=5, ncol=4, loc='upper center', bbox_to_anchor=(.5, -.13), handlelength=1, columnspacing=.6, handletextpad=.25)
    fig.tight_layout(pad=.25, w_pad=.6); fig.savefig(FIG/'fig_units.pdf'); plt.close(fig)


if __name__ == '__main__': fig_units()
