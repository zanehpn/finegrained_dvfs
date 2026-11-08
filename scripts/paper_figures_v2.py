"""Technique figures for the DAC paper (single-column IEEE width, vector PDF).

fig_motivation : (a) precision-dependent imbalance: saving of a single rail switched per phase / per layer / per operator
                     vs MIDAS's multi-domain plan at two TTFT budgets (C1);
                 (b) balance shift: compute time over HBM time vs tile precision (C3).
fig_arch       : MIDAS core organization and V/f domains (T1).
fig_vu         : VU return policy: slowdown vs return-queue entries, write back all vs export only (T2).
fig_planner    : how the planner orders slack: energy vs latency of three prefill operators (T3).
Data: tab:balance (first-order cost model), coarse_granularity_v1 / insight assignments (first order),
bridge_writeback_rtl_validation_v1..v3 (RTL lockstep). Colors: dataviz reference categorical order;
every series also has its own marker/line style or hatch, so identity never depends on color alone.
"""
import os, json, collections
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

W = Path('/shared/s3/zqy/dac2027_p1')
O = Path(os.environ.get('FIG_DATA', W/'results/cal'))
FIG = Path(os.environ.get('FIG_DIR', W/'paper/figures')); FIG.mkdir(exist_ok=True)
O0 = Path('/shared/s3/zqy/DSE/npu_mac/outputs/flexposit_enpu_paper/asap7_calibration')
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID, LIGHT = '#1f1f1e', '#6b6a64', '#e4e3df', '#cde2fb'
plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'lines.linewidth': 1.2, 'lines.markersize': 4, 'pdf.fonttype': 42, 'hatch.linewidth': .5})


def style(ax): ax.grid(axis='y', color=GRID, lw=.5); ax.set_axisbelow(True)


def fig_motivation():
    fig, (b, a) = plt.subplots(1, 2, figsize=(3.5, 1.55), gridspec_kw=dict(width_ratios=[1.15, 1]))  # (a) granularity, (b) balance
    P = [4, 5, 6, 8]
    for lab, ys, c, st in (('Query (prefill)', [1.31, 1.15, 1.04, .90], BLUE, dict(marker='o', ls='-')),
                           ('FFN down (prefill)', [.89, .78, .70, .61], ORANGE, dict(marker='s', ls='--')),
                           ('Query (decode)', [.44, .40, .37, .33], AQUA, dict(marker='^', ls=':'))):
        a.plot(P, ys, color=c, label=lab, **st)
    a.axhline(1, color=INK, lw=.7); a.text(5.15, .985, 'C/M = 1', fontsize=5.5, color=INK, ha='left', va='top')
    a.set_xlabel('Weight precision'); a.set_xticks(P); a.set_xticklabels([f'P{p}' for p in P]); a.set_ylabel('Compute-to-HBM time ratio'); a.set_ylim(.25, 1.75)
    a.set_title('(b) Compute-memory balance shift', fontsize=7, color=INK); style(a); a.legend(frameon=False, fontsize=5, loc='upper right', handlelength=1.6)
    groups = ['5', '25']
    N = ('main_t5_d5', 'main_t25_d5'); AB = W/'results/abl_c'
    def sv(d, stem):  # 0 until the 4-rail, 34-point single-rail run exists
        fs = [AB/d/f'{n}_{stem}.json' for n in N]
        return [json.loads(x.read_text())['saving_vs_pg_cg'] for x in fs] if all(x.exists() for x in fs) else [0., 0.]
    series = [('Single-rail, per phase', sv('coarse_granularity_v1', 'phase'), LIGHT, '////'), ('Single-rail, per layer', sv('coarse_granularity_v1', 'layer'), '#86b6ef', '\\\\\\\\'),
              ('Multi-domain (MIDAS)', sv('core', 'aware'), '#1c5cab', '')]
    w = .27
    for k, (lab, ys, c, h) in enumerate(series):
        xs = [g+(k-1)*w for g in range(2)]
        b.bar(xs, ys, w*.9, color=c, hatch=h, edgecolor='white', lw=.4, label=lab)
        for x, y in zip(xs, ys): b.text(x, y+.3, f'{y:.1f}' if y else 'TBD', ha='center', va='bottom', fontsize=4.8, color=INK)
    b.set_xlabel('TTFT slowdown budget (%)'); b.set_xticks(range(2)); b.set_xticklabels(groups); b.set_ylabel('Energy savings\nover PG+CG (%)'); b.set_ylim(0, 25)
    b.set_title('(a) Component imbalance', fontsize=7, color=INK); style(b); b.legend(frameon=False, fontsize=4.8, loc='upper left')
    fig.tight_layout(pad=.25, w_pad=.6); fig.savefig(FIG/'fig_motivation.pdf'); plt.close(fig)


def fig_units():
    """Per-unit busy time of Qwen2.5-14B layer 23 versus weight precision; numbers above the bars are compute over HBM time."""
    R = json.loads((O/'motivation_workload_v1/result.json').read_text())['rows']
    names = ['P4', 'P5', 'P6', 'P8', 'Fisher 5.5b']; labs = ['P4', 'P5', 'P6', 'P8', 'Fisher-\nguided']
    units = (('MATRIX', 'MAC', BLUE, ''), ('HBM', 'HBM', ORANGE, '////'), ('VU', 'VU', AQUA, '\\\\\\\\'))
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.3)); w = .27
    for ax, ph, scale, unit, title in ((axes[0], 'prefill', 1e-3, 'ms', '(a) Prefill, 1920-token prompt'), (axes[1], 'decode', 1., '$\\mu$s', '(b) Decode, one token')):
        for k, (key, lab, c, h) in enumerate(units):
            ax.bar([i+(k-1)*w for i in range(len(names))], [R[n][ph][key]*scale for n in names], w*.92, color=c, hatch=h, edgecolor='white', lw=.3, label=lab)
        top = max(R[n][ph][u]*scale for n in names for u in ('MATRIX', 'HBM'))
        for i, n in enumerate(names):
            r = R[n][ph]['MATRIX']/R[n][ph]['HBM']; y = max(R[n][ph]['MATRIX'], R[n][ph]['HBM'])*scale+top*.03
            ax.text(i, y, f'{r:.2f}', ha='center', va='bottom', fontsize=5.5, color=BLUE if r >= 1 else INK, fontweight='bold' if r >= 1 else 'normal')
        ax.set_xticks(range(len(names))); ax.set_xticklabels(labs, fontsize=6); ax.set_ylim(0, top*1.42)
        ax.set_ylabel(f'Busy time per layer ({unit})', fontsize=6); ax.set_title(title, fontsize=7, color=INK); style(ax)
    axes[1].legend(frameon=False, fontsize=5.5, loc='upper left', ncol=3, handlelength=1.2, columnspacing=.8, handletextpad=.3)
    fig.tight_layout(pad=.25, w_pad=.6); fig.savefig(FIG/'fig_units.pdf'); plt.close(fig)


def fig_vu():
    rows = [r for r in json.loads((W/'results/t2_real/writeback.json').read_text())['rows'] if (r['fe_ps'], r['vu_ps']) == (1200, 4200)]
    base = {r['kernel']: r['duration_ps'] for r in rows if r['policy'] == 'export'}
    d = collections.defaultdict(list)
    for r in rows: d[(r['policy'], r['out_entries'])].append(100*(r['duration_ps']/base[r['kernel']]-1))
    area = {('all', 4): 8015, ('all', 8): 9959, ('all', 16): 14056, ('export', 4): 5799}
    fig, ax = plt.subplots(figsize=(3.5, 1.45))
    xs = [4, 8, 16]
    mx = [max(d[('all', n)]) for n in xs]; mn = [min(d[('all', n)]) for n in xs]
    ax.fill_between(range(3), mn, mx, color=ORANGE, alpha=.18, lw=0)
    ax.plot(range(3), mx, color=ORANGE, marker='s', ls='--', label='Write-back-all')
    ax.plot([0], [max(d[('export', 4)])], color=BLUE, marker='o', ls='none', ms=5, label='Export-only return (MIDAS)')
    for i, n in enumerate(xs): ax.annotate(f'{area[("all", n)]:,} $\\mu$m$^2$', (i, mx[i]), textcoords='offset points', xytext=(4, 3), fontsize=5, color=INK)
    ax.annotate('5,799 $\\mu$m$^2$, 0%', (0, 0), textcoords='offset points', xytext=(6, 6), fontsize=5, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels([str(n) for n in xs]); ax.set_xlabel('Return-queue entries'); ax.set_ylabel('Kernel slowdown (%)')
    ax.set_xlim(-.3, 2.6); ax.set_ylim(-5, 70); style(ax); ax.legend(frameon=False, fontsize=5.5, loc='upper right')
    fig.tight_layout(pad=.25); fig.savefig(FIG/'fig_vu.pdf'); plt.close(fig)


def fig_t2():
    """T2 interface and a worked SiLU example; lockstep RTL durations at the 1.2/4.2 ns frontend/VU periods."""
    W_, S_ = (json.loads((W/'results/t2_real'/f).read_text())['rows'] for f in ('writeback.json', 'shadow_off.json'))
    r = [x for x in W_ if x['kernel'] == 'silu_mul' and x['bridge'] == 'v2' and (x['fe_ps'], x['vu_ps']) == (1200, 4200)]
    exp = next(x for x in r if x['policy'] == 'export' and x['out_entries'] == 4); al4 = next(x for x in r if x['policy'] == 'all' and x['out_entries'] == 4)
    nos = next(x for x in S_ if x['kernel'] == 'silu_mul' and x['variant'] == 'noshadow' and (x['fe_ps'], x['vu_ps']) == (1200, 4200))
    N = exp['commands']
    fig = plt.figure(figsize=(3.5, 2.35)); ax = fig.add_axes([0, .42, 1, .58]); ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis('off')
    ax.text(.08, 4.02, '(a) Export-only return', fontsize=6.5, color=INK, va='center')
    box(ax, .1, .15, 2.75, 3.5, '', '#f7f6f3', lw=.8); ax.text(.22, 3.42, 'Frontend domain', fontsize=5.3, color=INK, fontweight='bold', va='center')
    box(ax, .35, 2.1, 2.25, .95, 'Dispatch\n132-bit descriptors', '#dbe9fb', fs=4.8); box(ax, .35, .45, 2.25, .95, 'Register file\nexported results', '#dbe9fb', fs=4.8)
    box(ax, 3.0, .15, 1.15, 3.5, '', '#fff7df', lw=.6); ax.text(3.575, 3.2, 'Async\nFIFOs', fontsize=4.6, color=INK, ha='center', va='center'); ax.text(3.575, 1.75, 'clock\ncrossing', fontsize=4.6, color=MUTED, ha='center', va='center')
    box(ax, 4.3, .15, 5.6, 3.5, '', '#fdf1ea', lw=.8); ax.text(4.42, 3.42, 'VU domain', fontsize=5.3, color=INK, fontweight='bold', va='center')
    box(ax, 4.55, 2.1, 1.6, .95, 'Issue', '#fde3d6', fs=4.8); box(ax, 6.55, 2.1, 1.3, .95, 'Vector\nALU', '#fde3d6', fs=4.8)
    box(ax, 8.15, 2.1, 1.55, .95, '4 shadow\nslots', '#f6b89b', fs=4.8); box(ax, 4.55, .45, 1.6, .95, 'Return queue\n4 entries', '#fde3d6', fs=4.8)
    ar = lambda a, b, c=INK, **k: ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='->', lw=.7, color=c, shrinkA=0, shrinkB=0, **k))
    ar((2.6, 2.57), (4.55, 2.57)); ar((6.15, 2.57), (6.55, 2.57)); ar((7.85, 2.57), (8.15, 2.57))
    ar((8.92, 2.1), (6.15, .92), ORANGE, connectionstyle='angle,angleA=90,angleB=0,rad=0'); ar((4.55, .92), (2.6, .92), ORANGE)
    ax.plot([8.92, 8.92, 5.35], [3.05, 3.22, 3.22], color=BLUE, lw=.7, solid_capstyle='butt'); ar((5.35, 3.22), (5.35, 3.05), BLUE)
    ax.text(7.6, 3.42, 'bypass: next VU cycle', fontsize=4.7, color=BLUE, ha='center', va='center')
    ax.text(7.6, .6, 'only compiler-marked\nexports cross back', fontsize=4.7, color=ORANGE, ha='center', va='center')
    b = fig.add_axes([.31, .125, .36, .225])
    rows = [('No local forwarding', nos['returned'], nos['duration_ps']), ('Write-back-all, 4 entries', al4['returned'], al4['duration_ps']), ('Export-only return, 4 entries', exp['returned'], exp['duration_ps'])]
    cols = ('#c9c8c2', ORANGE, BLUE)
    for i, ((_, ret, d), c) in enumerate(zip(rows, cols)): b.barh(2-i, d/1e6, .62, color=c)
    b.set_yticks([2, 1, 0]); b.set_yticklabels([f'{n}\n{ret:,} returns' for n, ret, _ in rows], fontsize=4.8, color=INK); b.tick_params(axis='y', length=0)
    b.set_xlabel('Kernel time ($\\mu$s)', fontsize=5.2, labelpad=1); b.tick_params(axis='x', labelsize=5); b.set_xlim(0, 98); b.grid(axis='x', color=GRID, lw=.5); b.set_axisbelow(True)
    for i, (_, _, d) in enumerate(rows):
        fig.text(.685, .125+.225*(2-i+.5)/3, f'{d/1e6:.2f} $\\mu$s', fontsize=4.8, color=INK, va='center')
    fig.text(.012, .385, f'(b) SiLU: {N:,} instructions, {exp["returned"]} exported results ({100*exp["returned"]/N:.1f}%)', fontsize=6, color=INK, va='center')
    fig.savefig(FIG/'fig_t2.pdf'); plt.close(fig)


def fig_planner():
    st = [1.0, 1.5, 1.65, 2.125, 2.975]; r = [1.0, .916, .853, .705, .571]
    fig, ax = plt.subplots(figsize=(3.5, 1.5))
    for lab, cm, c, sty in (('Prefill (C/M = 1.04)', 1.04, BLUE, dict(marker='o', ls='-')),
                            ('Prefill (C/M = 1.31)', 1.31, ORANGE, dict(marker='s', ls='--')),
                            ('Decode (C/M = 0.40)', .40, AQUA, dict(marker='^', ls=':'))):
        T = [max(cm*s_, 1.) for s_ in st]
        ax.plot(T, r, color=c, label=lab, **sty)
    ax.annotate('Free points:\noverlap approximation', xy=(1.0, .705), xytext=(1.3, .64), fontsize=5.3, color=INK,
                arrowprops=dict(arrowstyle='->', lw=.5, color=MUTED))
    ax.annotate('Lower operating points', xy=(1.30, .958), xytext=(1.55, .53), fontsize=5.3, color=INK,
                arrowprops=dict(arrowstyle='->', lw=.5, color=MUTED))
    ax.set_xlabel('Block latency / HBM service time $M$'); ax.set_ylabel('Normalized MAC energy'); ax.set_ylim(.5, 1.12); style(ax)
    ax.legend(frameon=False, fontsize=5.0, loc='upper right'); fig.tight_layout(pad=.25); fig.savefig(FIG/'fig_planner.pdf'); plt.close(fig)


def box(ax, x, y, w, h, text, fc, ec=MUTED, fs=5.5, lw=.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01,rounding_size=0.02', fc=fc, ec=ec, lw=lw))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold' if bold else 'normal')


def fig_arch():
    fig, ax = plt.subplots(figsize=(3.5, 1.85)); ax.set_xlim(0, 10); ax.set_ylim(-.7, 5.4); ax.axis('off')
    for cx, name in ((0.1, 'Prefill core (P)'), (5.1, 'Decode core (D)')):
        box(ax, cx, .55, 4.8, 4.35, '', '#f7f6f3', lw=.8)
        ax.text(cx+.12, 4.62, name, fontsize=6, color=INK, va='center', fontweight='bold')
        for j, yy in enumerate((3.4, 2.5)):
            ax.text(cx+.1, yy+.31, f'Array {"1" if j == 0 else "k"}', fontsize=4.8, color=MUTED, va='center')
            for i, (d, c) in enumerate((('CTRL', '#dbe9fb'), ('DEC', '#dbe9fb'), ('MAC', '#3987e5'), ('OUT', '#dbe9fb'))):
                box(ax, cx+1.0+i*.94, yy, .86, .62, d, c, fs=5)
        ax.text(cx+2.85, 2.2, '$\\vdots$', fontsize=6, color=MUTED, ha='center')
        box(ax, cx+.2, .75, 1.4, 1.15, 'VU\nshadow slots,\nreturn queue', '#fde3d6', fs=4.7)
        box(ax, cx+1.72, .75, 1.3, 1.15, 'Attention\nunit', '#fde3d6', fs=4.8)
        box(ax, cx+3.14, .75, 1.56, 1.15, 'MEM\nSRAM ctrl,\nDMA', '#fdf0cc', fs=4.8)
    ax.annotate('', xy=(5.1, 1.3), xytext=(4.9, 1.3), arrowprops=dict(arrowstyle='->', lw=.7, color=INK))
    ax.annotate('KV-cache link', xy=(5.0, 1.3), xytext=(5.0, .05), fontsize=4.8, color=MUTED, ha='center', arrowprops=dict(arrowstyle='-', linestyle=':', lw=.5, color=MUTED))
    for xpos, label, color in ((2.2, 'MAC', '#3987e5'), (4.0, 'LOGIC', '#dbe9fb'), (6.0, 'VEC', '#fde3d6'), (7.8, 'MEM', '#fdf0cc')):
        ax.scatter([xpos], [5.22], marker='s', s=20, c=color, edgecolors=MUTED, linewidths=.3)
        ax.text(xpos+.22, 5.22, label, fontsize=5.7, color=INK, va='center')
    ax.text(5.0, -.5, 'Both cores share 16 HBM3 channels', fontsize=5, color=MUTED, ha='center')
    fig.tight_layout(pad=.1); fig.savefig(FIG/'fig_arch.pdf'); plt.close(fig)


if __name__ == '__main__':
    import sys
    for f in (sys.argv[1:] or ['motivation', 'vu', 'planner', 'arch']): globals()['fig_'+f]()
