"""Figures for the DAC paper (single-column IEEE width, vector PDF).

fig_breakdown : cumulative saving against NoDVFS, step by step (gating, memory DVFS, compute DVFS by
                costing), for Qwen2.5-14B 3x2K at the 5%/5% and the P8-provisioned TTFT 25% objectives.
fig_slack     : saving against NoDVFS vs objective budget, fine domains, three tile costings.
fig_batch     : saving against PG+CG vs decode batch, Qwen and Llama, three tile costings.
Colors: fixed categorical order of the dataviz reference palette (per-tile = slot 1 blue, average =
slot 2 orange, worst case = slot 3 aqua); every series also has its own marker and line style, so
identity never depends on color (print and CVD). Inputs are first-order results until the calibrated
replays replace the headline points.
"""
import os, json, glob, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W = Path('/shared/s3/zqy/dac2027_p1')
O = Path(os.environ.get('FIG_DATA', W/'results/cal'))
FIG = Path(os.environ.get('FIG_DIR', W/'paper/figures')); FIG.mkdir(exist_ok=True)
COL = {'aware': '#2a78d6', 'mean': '#eb6834', 'worst': '#1baf7a'}
NAME = {'aware': 'Per-tile precision\n(MIDAS)', 'mean': 'Average precision', 'worst': 'Worst-case precision'}
STY = {'aware': dict(marker='o', ls='-'), 'mean': dict(marker='s', ls='--'), 'worst': dict(marker='^', ls=':')}
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3df'
plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'lines.linewidth': 1.2, 'lines.markersize': 4, 'pdf.fonttype': 42})


def style(ax):
    ax.grid(axis='y', color=GRID, lw=.5); ax.set_axisbelow(True)


def fig_breakdown():
    r = json.loads((O/'pivot_breakdown_v2/result.json').read_text())['scenarios']
    steps = [('PG+CG', 'PG+CG'), ('coarse_aware', 'Coarse,\nper block'), ('fine_agnostic', 'Fine,\nworst case'), ('fine_aware', 'Fine, per tile\n(MIDAS)')]
    fig, ax = plt.subplots(figsize=(3.5, 1.75)); w = .38
    for k, (scen, lab, col) in enumerate((('b5.5_ttft5_tpot5_self', 'TTFT 5%', '#86b6ef'), ('b5.5_ttft25_tpot5_p8', 'TTFT 25% (P8-provisioned)', '#1c5cab'))):
        rows = {x['step']: x for x in r[scen]}; vals = []
        for key, _ in steps:
            row = rows['PG+CG'] if key == 'PG+CG' else next(x for s, x in rows.items() if key in s and '+HBM' not in s)
            vals.append(row['vs_nodvfs'])
        xs = [i+(k-.5)*w for i in range(len(steps))]
        bars = ax.bar(xs, vals, w*.92, color=col, label=lab, edgecolor='white', lw=.6)
        for x, v in zip(xs, vals): ax.text(x, v+.6, f'{v:.1f}', ha='center', va='bottom', fontsize=5.5, color=INK)
    ax.set_xticks(range(len(steps))); ax.set_xticklabels([s[1] for s in steps], fontsize=6)
    ax.set_ylabel('Energy savings\nover NoDVFS (%)'); ax.set_ylim(0, 40); style(ax)
    ax.legend(frameon=False, fontsize=6, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2)
    fig.tight_layout(pad=.3); fig.savefig(FIG/'fig_breakdown.pdf'); plt.close(fig)


def load(dirname, prefix):
    out = {}
    for f in glob.glob(str(O/dirname/f'{prefix}*.json')):
        r = json.loads(Path(f).read_text()); out[(r['name'], r['coarse'], r['baseline'])] = r
    return out


def fig_slack():
    R = load('pivot_sensitivity_v7', 'main_')
    budgets = [(0, 0), (2, 2), (5, 5), (10, 10), (25, 5)]
    fig, ax = plt.subplots(figsize=(3.5, 1.6))
    for b in ('worst', 'mean', 'aware'):
        ys = [R[(f'main_t{t}_d{d}', False, b)]['saving_vs_nodvfs'] if (f'main_t{t}_d{d}', False, b) in R else None for t, d in budgets]
        xs = [i for i, y in enumerate(ys) if y is not None]; ys = [y for y in ys if y is not None]
        ax.plot(xs, ys, color=COL[b], label=NAME[b], **STY[b])
        if ys: ax.annotate(f'{ys[-1]:.1f}', (xs[-1], ys[-1]), textcoords='offset points', xytext=(4, -2), fontsize=5.5, color=INK)
    ax.set_xticks(range(len(budgets))); ax.set_xticklabels([f'{t}/{d}' for t, d in budgets])
    ax.set_xlabel('TTFT/TPOT slowdown budget (%)'); ax.set_ylabel('Energy savings\nover NoDVFS (%)'); style(ax)
    ax.legend(frameon=False, fontsize=6, loc='upper left'); fig.tight_layout(pad=.3); fig.savefig(FIG/'fig_slack.pdf'); plt.close(fig)


def fig_batch():
    R = load('pivot_sensitivity_v7', '')
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.5), sharey=True)
    for ax, model in zip(axes, ('qwen', 'llama')):
        for b in ('worst', 'mean', 'aware'):
            pts = [(B, R[(f'{model}_batch{B}', False, b)]['saving_vs_pg_cg']) for B in (1, 4, 16, 64) if (f'{model}_batch{B}', False, b) in R]
            if pts: ax.plot(range(len(pts)), [y for _, y in pts], color=COL[b], label=NAME[b], **STY[b])
        ax.set_xticks(range(4)); ax.set_xticklabels(['1', '4', '16', '64']); ax.set_xlabel('Decode batch size')
        ax.set_title('Qwen2.5-14B' if model == 'qwen' else 'Llama-3.1-8B', fontsize=7, color=INK); style(ax)
    axes[0].set_ylabel('Energy savings\nover PG+CG (%)'); axes[1].legend(frameon=False, fontsize=5.5, loc='upper right')
    fig.tight_layout(pad=.3); fig.savefig(FIG/'fig_batch.pdf'); plt.close(fig)


if __name__ == '__main__':
    for f in (sys.argv[1:] or ['breakdown', 'slack', 'batch']): globals()['fig_'+f]()
