"""Section 6 figures from the fair online ablation (results/abl_c: 4 rails, 34 points, fair bytes, first-order model).
Open marker: budget feedback tightened the run. Trace misses are reported in the text."""
import json, glob, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_figures_v1 import FIG, COL, NAME, STY, INK, style
A = Path('/shared/s3/zqy/dac2027_p1/results/abl_c'); ORDER = ('worst', 'mean', 'aware')


def r(d, n, c): return json.loads((A/d/f'{n}_{c}.json').read_text())


def line(ax, d, names, c):
    rows = [r(d, n, c) for n in names]
    ax.plot(range(len(rows)), [x['saving_vs_pg_cg'] for x in rows], color=COL[c], label=NAME[c], **STY[c])
    for i, x in enumerate(rows):
        if x['rounds']: ax.plot(i, x['saving_vs_pg_cg'], marker=STY[c]['marker'], mfc='white', mec=COL[c], ls='none', ms=4)


def fig_slack_batch():
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.55), gridspec_kw=dict(width_ratios=[1, 1]))
    B = [(2, 2), (5, 5), (10, 10), (25, 5)]
    for c in ORDER: line(a, 'core', [f'main_t{t}_d{d}' for t, d in B], c)
    a.set_xticks(range(4)); a.set_xticklabels([f'({t}, {d})' for t, d in B], fontsize=6); a.set_xlabel('Slowdown budgets (%)'); a.set_ylabel('Energy savings\nover PG+CG (%)'); a.set_title('(a) Budget sweep', fontsize=7, color=INK)
    for c in ORDER: line(b, 'core', [f'qwen_batch{n}_t5_d5' for n in (1, 4, 16, 64)], c)
    b.set_xticks(range(4)); b.set_xticklabels(['1', '4', '16', '64'], fontsize=6); b.set_xlabel('Decode batch size'); b.set_title('(b) Decode batches', fontsize=7, color=INK)
    for ax in (a, b): style(ax); ax.set_ylim(0, 25)
    a.legend(frameon=False, fontsize=4.8, loc='upper left', handlelength=2.2)
    fig.tight_layout(pad=.3, w_pad=.6); fig.savefig(FIG/'fig_slack_batch.pdf'); plt.close(fig)


def fig_slo():
    B = (2, 5, 10); fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.5), sharey=True)
    for ax, tr, title in zip(axes, ('serve', 'servebatch'), ('(a) One request per\ndecode iteration', '(b) Continuous batching')):
        for c in ORDER:
            v = [[r('core', f'{tr}_{l}_t{b}_d{b}', c)['saving_vs_pg_cg'] for l in ('low', 'mid', 'high')] for b in B]
            ax.fill_between(range(3), [min(x) for x in v], [max(x) for x in v], color=COL[c], alpha=.13, lw=0)
            line(ax, 'core', [f'{tr}_mid_t{b}_d{b}' for b in B], c)
        ax.set_xticks(range(3)); ax.set_xticklabels([str(b) for b in B], fontsize=6); ax.set_xlabel('TTFT and TPOT\nslowdown budget (%)'); ax.set_title(title, fontsize=7, color=INK); style(ax)
    axes[0].set_ylabel('Energy savings\nover PG+CG (%)'); axes[0].set_ylim(0, 18); axes[0].legend(frameon=False, fontsize=5.5, loc='upper left', handlelength=2.2)
    fig.tight_layout(pad=.3, w_pad=.5); fig.savefig(FIG/'fig_slo.pdf'); plt.close(fig)


def fig_breakdown():
    """Saving over NoDVFS by configuration. The single-rail column appears once its 4-rail, 34-point run exists."""
    cols = [('PG+CG', None), ('Worst-case\nprecision', 'worst'), ('Average\nprecision', 'mean'), ('Per-tile\nprecision\n(MIDAS)', 'aware')]
    G = {n: A/'coarse_granularity_v1'/f'{n}_layer.json' for n in ('main_t5_d5', 'main_t25_d5')}
    rail = all(f.exists() for f in G.values())
    if rail: cols.insert(1, ('Single-rail DVFS\nper layer', 'layer'))
    fig, ax = plt.subplots(figsize=(3.5, 1.7)); w = .38
    for k, (n, lab, col) in enumerate((('main_t5_d5', '5%', '#86b6ef'), ('main_t25_d5', '25%', '#1c5cab'))):
        x = r('core', n, 'aware'); pg = 100*(1-(1-x['saving_vs_nodvfs']/100)/(1-x['saving_vs_pg_cg']/100))
        vals = [pg if c is None else json.loads(G[n].read_text())['saving_vs_nodvfs'] if c == 'layer' else r('core', n, c)['saving_vs_nodvfs'] for _, c in cols]
        xs = [i+(k-.5)*w for i in range(len(cols))]
        ax.bar(xs, vals, w*.92, color=col, label=lab, edgecolor='white', lw=.6)
        for a, v in zip(xs, vals): ax.text(a, v+.6, f'{v:.1f}', ha='center', va='bottom', fontsize=5.5, color=INK)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[0] for c in cols], fontsize=5.8); ax.set_ylabel('Energy savings\nover NoDVFS (%)'); ax.set_ylim(0, 48); style(ax)
    ax.legend(frameon=False, fontsize=6, loc='upper left', ncol=2, title='TTFT slowdown budget', title_fontsize=6); ax.text(.74, -.28, 'Multi-domain DVFS cost model', transform=ax.transAxes, ha='center', fontsize=5.7); fig.subplots_adjust(bottom=.32); fig.savefig(FIG/'fig_breakdown.pdf', bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    for f in (sys.argv[1:] or ['breakdown', 'slack_batch', 'slo']): globals()['fig_'+f]()
