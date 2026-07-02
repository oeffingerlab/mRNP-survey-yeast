#!/usr/bin/env python3
"""make_figure7D.py — model panel D for Figure 7 (REVISED: stage-specific RBP label)"""
import matplotlib
matplotlib.use('Agg')
import sys, numpy as np, shutil
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, '/sessions/lucid-modest-goldberg/mnt/outputs')
from figure_style import apply_style, GROUP_COLORS
apply_style()
plt.rcParams['pdf.fonttype'] = 42

STAGES    = ['Co-txp /\nTHO-TREX', 'Splicing', '3′-end\nprocessing',
             'Surveillance\n/ TREX-2', 'Nuclear\nexport']
STAGE_COL = [GROUP_COLORS['Co-Txp/THO/TREX'], GROUP_COLORS['Splicing'],
             GROUP_COLORS["3'end processing"],  GROUP_COLORS['Surveillance'],
             GROUP_COLORS['Export/pore']]

ROWS = [
    ('CBC  (Cbp80p · Cbp20p)',             0.0, 5.0, 'CBC',              True ),
    ('Yra1p',                                    0.0, 5.0, 'Co-Txp/THO/TREX', True ),
    ('Pab1p',                                    2.0, 5.0, 'Cytoplasmic',      True ),
    None,
    ('THO-TREX  (Sub2p · Hpr1p)',           0.0, 1.7, 'Co-Txp/THO/TREX', False),
    None,
    ('Splicing  (Prp8p, U-snRNPs)',              1.0, 2.0, 'Splicing',         False),
    ('3′-end factors  (Rna14p, Pcf11p)',   2.0, 3.0, "3'end processing", False),
    None,
    ('Npl3p',                                    0.0, 3.5, 'Export/pore',      False),
    ('Nab2p',                                    0.8, 3.8, 'TREX-2/NPC',       False),
    ('Rrp6p · exosome',                    2.5, 4.0, 'Surveillance',      False),
    None,
    ('TREX-2  (Sac3p, Thp1p)',                  3.5, 4.8, 'TREX-2/NPC',       False),
    ('Mex67p · Mtr2p',                     4.0, 5.0, 'Export/pore',       False),
]

ROW_H, SPACER_H = 1.0, 0.55
temp, cy = [], 0.0
for r in reversed(ROWS):
    if r is None:
        temp.append(None); cy += SPACER_H
    else:
        temp.append(cy);   cy += ROW_H
y_pos   = list(reversed(temp))
y_total = cy

y_by_fi, fi = {}, 0
for r, y in zip(ROWS, y_pos):
    if r is not None:
        y_by_fi[fi] = y; fi += 1

def br(i_top, i_bot):
    return y_by_fi[i_top], y_by_fi[i_bot]

BRACKETS = [
    (*br(0, 2), 'Persistent\ncore',              '#444444'),
    (*br(3, 3), 'Module 1\nCo-Txp /\nTHO-TREX', GROUP_COLORS['Co-Txp/THO/TREX']),
    (*br(4, 5), 'Module 2\nSplicing /\n3′-end',  GROUP_COLORS['Splicing']),
    (*br(6, 8), 'Module 3\nhnRNP /\nExport',    GROUP_COLORS['TREX-2/NPC']),
    (*br(9,10), 'Module 4\nTREX-2 /\nExport',   GROUP_COLORS['Export/pore']),
]

fig = plt.figure(figsize=(14.0, 7.0))
gs  = GridSpec(1, 2, width_ratios=[2.2, 1],
               left=0.26, right=0.97, top=0.90, bottom=0.08, wspace=0.30)
ax_t = fig.add_subplot(gs[0])
ax_m = fig.add_subplot(gs[1])
fig.text(0.005, 0.98, 'D', fontsize=16, fontweight='bold', va='top')

# ── TIMELINE ──────────────────────────────────────────────────────────────────
for i, sc in enumerate(STAGE_COL):
    ax_t.axvspan(i, i+1, alpha=0.09, color=sc, lw=0)
    ax_t.axvline(i+1, color='#e0e0e0', lw=0.7, zorder=1)

for i, (s, sc) in enumerate(zip(STAGES, STAGE_COL)):
    ax_t.text(i+0.5, y_total+0.5, s,
              ha='center', va='bottom', fontsize=6.5, color=sc, fontweight='bold')

ax_t.annotate('', xy=(5.03, y_total+0.38), xytext=(0.0, y_total+0.38),
              arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=0.9))

for row, yv in zip(ROWS, y_pos):
    if row is None: continue
    name, x0, x1, group, core = row
    col = GROUP_COLORS.get(group, '#888888')
    bh  = 0.58 if core else 0.44
    ax_t.barh(yv, x1-x0, left=x0, height=bh, color=col, alpha=0.88, lw=0, zorder=3)
    if core:
        ax_t.barh(yv, x1-x0, left=x0, height=bh,
                  color='none', edgecolor='white', lw=1.0, zorder=4)
    ax_t.text(-0.12, yv, name, ha='right', va='center',
              fontsize=6.0, fontweight='bold' if core else 'normal',
              color='#111111', clip_on=False)

BX = 5.10
for y_top, y_bot, lbl, col in BRACKETS:
    mid = (y_top + y_bot) / 2
    ax_t.plot([BX, BX], [y_bot-0.25, y_top+0.25],
              color=col, lw=2.0, solid_capstyle='butt', zorder=5)
    ax_t.plot([BX-0.05, BX], [y_bot-0.25, y_bot-0.25], color=col, lw=1.5, zorder=5)
    ax_t.plot([BX-0.05, BX], [y_top+0.25, y_top+0.25], color=col, lw=1.5, zorder=5)
    ax_t.text(BX+0.15, mid, lbl, ha='left', va='center',
              fontsize=5.5, color=col, fontweight='bold',
              linespacing=1.3, clip_on=False)

ax_t.set_xlim(-0.1, 7.2)
ax_t.set_ylim(-0.65, y_total+1.2)
ax_t.axis('off')
ax_t.set_title('Factor association across mRNA biogenesis stages',
               fontsize=8.5, pad=8, fontweight='bold', color='#333333')

# ── MASS COMPOSITION DONUT ────────────────────────────────────────────────────
dvals = [400, 105, 284, 32, 379]
dcols = ['#AACCE4', GROUP_COLORS['CBC'], GROUP_COLORS['Cytoplasmic'],
         GROUP_COLORS['Co-Txp/THO/TREX'], '#C0C0C0']
wedges, _ = ax_m.pie(dvals, colors=dcols, startangle=90,
                     wedgeprops=dict(width=0.48, edgecolor='white', linewidth=1.8))
wedges[-1].set_hatch('////')
wedges[-1].set_alpha(0.70)

ax_m.text(0,  0.10, '~1.2 MDa',   ha='center', va='center',
          fontsize=9, fontweight='bold', color='#333333')
ax_m.text(0, -0.13, 'total mRNP', ha='center', va='center',
          fontsize=6.5, color='#666666')

# REVISED: label the hatched segment as stage-specific RBPs (= Modules 1–4)
leg_h = [
    mpatches.Patch(fc='#AACCE4',                        label='RNA backbone (~400 kDa)'),
    mpatches.Patch(fc=GROUP_COLORS['CBC'],              label='CBC: Cbp80p + Cbp20p (~105 kDa)'),
    mpatches.Patch(fc=GROUP_COLORS['Cytoplasmic'],      label='Pab1p (~284 kDa; ×4 copies)'),
    mpatches.Patch(fc=GROUP_COLORS['Co-Txp/THO/TREX'], label='Yra1p (~32 kDa)'),
    mpatches.Patch(fc='#C0C0C0', hatch='////', alpha=0.70,
                   label='Stage-specific RBPs\n(Modules 1–4, ~379 kDa)'),
]
ax_m.legend(handles=leg_h, loc='lower center', bbox_to_anchor=(0.5, -0.62),
            fontsize=5.5, frameon=False, ncol=1, labelspacing=0.55,
            handlelength=1.4, handleheight=1.0)
ax_m.set_title('mRNP mass composition\n(2:1 protein : RNA)',
               fontsize=8.5, pad=6, fontweight='bold', color='#333333')

# ── SAVE ──────────────────────────────────────────────────────────────────────
CURRENT = '/sessions/lucid-modest-goldberg/mnt/MS results/Assembled_Figures/current_2026-07-01'
OUT = '/sessions/lucid-modest-goldberg/mnt/outputs/Figure7D_model_draft'

fig.savefig(OUT+'.pdf', dpi=150, bbox_inches='tight')
fig.savefig(OUT+'.png', dpi=150, bbox_inches='tight')
shutil.copy2(OUT+'.pdf', f'{CURRENT}/Figure7D_model_draft.pdf')
print("Saved to current_2026-07-01/")
plt.close()
